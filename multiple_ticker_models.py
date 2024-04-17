import math
from functools import lru_cache
from typing import List, Dict

import pandas as pd
import copy
from APIs.close_price import get_all_adjusted_prices
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import random
from tqdm import tqdm

model_path = './models'

# 'LSTM' or 'GRU'
# If randomize testing is True, randomly choose test tickers from ticker list
# Otherwise, training and testing tickers must be specified.
randomize_testing = False
ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'UBER', 'TSLA', 'ORCL', 'CRM', 'NFLX']
training_split = 0.8
test_split = 1 - training_split
# If randomize testing is false, use existing training and testing list
training_tickers = ['MSFT', 'GOOGL', 'META', 'TLSA', 'ORCL', 'CRM', 'NFLX']
# testing_tickers = ['NVDA', 'UBER', 'AAPL']
testing_tickers = ['HUBS']

scalers = {}
dates = {}

input_dim = 1

# nodes per layer
output_dim = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# sns.set_style("darkgrid")


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device).requires_grad_()
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def save_model(self, path=f"{model_path}/lstm_multiple_ticker.pt"):
        torch.save(self.state_dict(), path)


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def save_model(self, path=f"{model_path}/gru_multiple_ticker.pt"):
        torch.save(self.state_dict(), path)


# @lru_cache(maxsize=None)
def get_data_frames(tickers, existing_training, existing_testing, randomize):
    if not randomize:
        tickers = existing_training + existing_testing

    # Mappings of ticker to data frame
    result = dict()
    for i in tqdm(range(len(tickers))):
        ticker = tickers[i]
        data = get_all_adjusted_prices(ticker, verbose=False)

        df = pd.DataFrame(data, columns=["Date", "Close"])
        # dates = df["Date"].values
        dates[ticker] = list(df["Date"])
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = df["Close"].astype(float)
        df = df.set_index("Date")

        result[ticker] = df

    return result


def preprocess(ticker_dataframes_map):
    ticker_prices_map = dict()
    for ticker in ticker_dataframes_map:
        df = ticker_dataframes_map[ticker]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        prices = df.copy()[["Close"]]

        prices["Close"] = scaler.fit_transform(prices["Close"].values.reshape(-1, 1))
        scalers[ticker] = scaler
        # dates[ticker] = df.copy()[["Date"]]

        ticker_prices_map[ticker] = prices

    return ticker_prices_map


def sequence_and_split(ticker_prices_map, sequence_length, randomize):
    x_train = dict()
    y_train = dict()
    x_test = dict()
    y_test = dict()
    complete_x = dict()
    complete_y = dict()
    for ticker in ticker_prices_map:
        stock = ticker_prices_map[ticker]

        raw = stock.to_numpy()[::-1]
        data = []

        for index in range(len(raw) - sequence_length):
            data.append(raw[index: index + sequence_length])

        data = np.array(data)

        x = data[:, :-1]
        y = data[:, -1]

        complete_x[ticker] = x
        complete_y[ticker] = y

        # x_train[i] = some sequence of prices
        # y_train[i] = the next price

    if randomize:
        test_stock_size = max(round(len(complete_x.keys()) * test_split), 1)
        test_stock_tickers = random.sample(list(complete_x.keys()), test_stock_size)

        for ticker in complete_x:
            if ticker in test_stock_tickers:
                x_test[ticker] = complete_x[ticker]
                y_test[ticker] = complete_y[ticker]
            else:
                x_train[ticker] = complete_x[ticker]
                y_train[ticker] = complete_y[ticker]
    else:
        for ticker in complete_x:
            if ticker in training_tickers:
                x_train[ticker] = complete_x[ticker]
                y_train[ticker] = complete_y[ticker]
            else:
                x_test[ticker] = complete_x[ticker]
                y_test[ticker] = complete_y[ticker]

    return [x_train, y_train, x_test, y_test]


def train(model, ticker_x_map, ticker_y_map, num_epochs, lr, verbose=True):
    for ticker in ticker_x_map:
        if verbose:
            print(f"Training using {ticker}...")
        x_train = torch.from_numpy(ticker_x_map[ticker]).type(torch.Tensor).to(device)
        y_train = torch.from_numpy(ticker_y_map[ticker]).type(torch.Tensor).to(device)

        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = np.zeros(num_epochs)
        for t in range(num_epochs):
            prediction = model(x_train)

            loss = criterion(prediction, y_train)
            if verbose:
                print(f"Epoch: {t}, MSE: {loss.item()}")
            history[t] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print()

    model.save_model()

    return model


def eval_and_plot_model(model, x_train_map, y_train_map, x_test_map, y_test_map, model_type, sequence_length):
    for ticker in x_test_map:
        x_test = torch.from_numpy(x_test_map[ticker]).type(torch.Tensor).to(device)
        # y_test = torch.from_numpy(y_test_map[ticker]).type(torch.Tensor).to(device)
        y_test = y_test_map[ticker]
        testing_predictions = model(x_test)
        testing_predictions = testing_predictions.cpu().detach().numpy()
        test_rmse = mean_squared_error(y_test[:, 0], testing_predictions[:, 0])

        print(f"MSE for {model_type} with {ticker}: {test_rmse}")

        plot_singular_complete(testing_predictions, y_test, ticker, model_type, sequence_length)


def average_mse_test(model, x_test_map, y_test_map):
    total_mse = 0
    for ticker in x_test_map:
        x_test = torch.from_numpy(x_test_map[ticker]).type(torch.Tensor).to(device)
        # y_test = torch.from_numpy(y_test_map[ticker]).type(torch.Tensor).to(device)
        y_test = y_test_map[ticker]
        testing_predictions = model(x_test)
        testing_predictions = testing_predictions.cpu().detach().numpy()
        total_mse += mean_squared_error(y_test[:, 0], testing_predictions[:, 0])

    return total_mse / len(x_test_map)


def plot_singular_complete(predicted, actual, ticker, model_info, sequence_length):
    scaler_predicted = copy.deepcopy(scalers[ticker])
    scaler_actual = copy.deepcopy(scalers[ticker])
    predicted = scaler_predicted.inverse_transform(predicted)
    actual = scaler_actual.inverse_transform(actual)

    dates_x = dates[ticker]
    dates_x.reverse()
    dates_x = dates_x[sequence_length:]
    dates_x = pd.to_datetime(dates_x)

    df = pd.DataFrame({'Actual price': actual.flatten(),
                       'Predicted price': predicted.flatten()},
                      index=dates_x)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    fig.set_figheight(8)
    fig.set_figwidth(16)
    sns.set_style("darkgrid")
    plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x=df.index, y='Actual price', label='Actual')
    sns.lineplot(data=df, x=df.index, y='Predicted price', label='Predicted')

    n = len(dates_x)
    step_size = max(n // 4, 1)  # Divide into roughly 4 sections
    display_dates = dates_x[::-step_size]
    plt.xticks(display_dates)

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Predictions - {model_info} All dates")
    plt.legend()

    recent_days_back = 365

    predicted_recent = predicted[-recent_days_back:]
    actual_recent = actual[-recent_days_back:]
    dates_x_recent = dates_x[-recent_days_back:]
    df_recent = pd.DataFrame({'Actual price': actual_recent.flatten(),
                              'Predicted price': predicted_recent.flatten()},
                             index=dates_x_recent)

    sns.set_style("darkgrid")
    plt.subplot(2, 1, 2)
    sns.lineplot(data=df_recent, x=df_recent.index, y='Actual price', label='Actual')
    sns.lineplot(data=df_recent, x=df_recent.index, y='Predicted price', label='Predicted')

    n = len(dates_x_recent)
    step_size = max(n // 4, 1)
    display_dates_recent = dates_x_recent[::-step_size]
    plt.xticks(display_dates_recent)

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Predictions - {model_info} Recent {recent_days_back} weekdays")
    plt.legend()

    plt.show()


def run_with_training(training, testing, use_model, sequence_length,
                      hidden_dim, num_layers, num_epochs, lr):
    ticker_dataframes_map = get_data_frames(tuple(ticker_list), tuple(training), tuple(testing), False)
    prices = preprocess(ticker_dataframes_map)
    x_train, y_train, x_test, y_test = sequence_and_split(prices, sequence_length, False)

    print(f"Training tickers: {str(list(x_train.keys()))}")
    print(f"Testing tickers: {str(list(x_test.keys()))}")

    x_train_copy = copy.deepcopy(x_train)
    y_train_copy = copy.deepcopy(y_train)

    if use_model == 'LSTM':
        lstm = train(LSTM(input_dim, hidden_dim, num_layers, output_dim)
                     .to(device), x_train, y_train, num_epochs, lr)

        # UNCOMMENT FOR LOADING SAVED MODEL
        # lstm = LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
        # lstm.load_state_dict(torch.load(f"{model_path}/lstm_multiple_ticker.pt"))
        # lstm.eval()

        eval_and_plot_model(lstm, x_train_copy, y_train_copy, x_test, y_test, "LSTM", sequence_length)
    elif use_model == 'GRU':
        # gru = train(GRU(input_dim, hidden_dim, num_layers, output_dim)
        #             .to(device), x_train, y_train, num_epochs, lr)
        
        # UNCOMMENT FOR LOADING SAVED MODEL
        gru = GRU(input_dim, hidden_dim, num_layers, output_dim).to(device)
        gru.load_state_dict(torch.load(f"{model_path}/gru_multiple_ticker.pt", map_location=torch.device('cpu')))
        gru.eval()

        eval_and_plot_model(gru, x_train_copy, y_train_copy, x_test, y_test, "GRU", sequence_length)


def run_and_eval(training, testing, use_model, sequence_length,
                 hidden_dim, num_layers, num_epochs, lr, verbose=False):
    ticker_dataframes_map = get_data_frames(tuple(ticker_list), tuple(training), tuple(testing), False)
    prices = preprocess(ticker_dataframes_map)
    x_train, y_train, x_test, y_test = sequence_and_split(prices, sequence_length, False)

    x_train_copy = copy.deepcopy(x_train)
    y_train_copy = copy.deepcopy(y_train)

    model = None
    if use_model == 'LSTM':
        model = train(LSTM(input_dim, hidden_dim, num_layers, output_dim).to(device), x_train, y_train, num_epochs, lr, verbose)

    elif use_model == 'GRU':
        model = train(GRU(input_dim, hidden_dim, num_layers, output_dim).to(device), x_train, y_train, num_epochs, lr, verbose)

    avg_mse = average_mse_test(model, x_train_copy, y_train_copy)
    return model, avg_mse


if __name__ == '__main__':
    run_with_training(training_tickers, testing_tickers, 'GRU', 20, 32, 2, 100, 0.01)
    # ticker_dataframes_map = get_data_frames(ticker_list, training_tickers, testing_tickers)
    # prices = preprocess(ticker_dataframes_map)
    # x_train, y_train, x_test, y_test = sequence_and_split(prices)
    #
    # print(f"Training tickers: {str(list(x_train.keys()))}")
    # print(f"Testing tickers: {str(list(x_test.keys()))}")
    #
    # x_train_copy = copy.deepcopy(x_train)
    # y_train_copy = copy.deepcopy(y_train)
    #
    # LOAD_SAVED_MODEL = False
    #
    # if LOAD_SAVED_MODEL:
    #     if use_model == 'LSTM':
    #         lstm = LSTM().to(device)
    #         lstm.load_state_dict(torch.load(f"{model_path}/lstm_multiple_ticker.pt"))
    #         lstm.eval()
    #
    #         eval_model(lstm, x_train_copy, y_train_copy, x_test, y_test, "LSTM")
    #
    #     elif use_model == 'GRU':
    #         gru = GRU().to(device)
    #         gru.load_state_dict(torch.load(f"{model_path}/gru_multiple_ticker.pt"))
    #         gru.eval()
    # else:
    #     if use_model == 'LSTM':
    #         lstm = train(LSTM().to(device), x_train, y_train)
    #
    #         eval_model(lstm, x_train_copy, y_train_copy, x_test, y_test, "LSTM")
    #     elif use_model == 'GRU':
    #         gru = train(GRU().to(device), x_train, y_train)
    #
    #         eval_model(gru, x_train_copy, y_train_copy, x_test, y_test, "GRU")
