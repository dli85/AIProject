import math
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

# 'LSTM' or 'GRU'
use_model = 'LSTM'
# If randomize testing is True, randomly choose test tickers from ticker list
# Otherwise, training and testing tickers must be specified.
randomize_testing = False
ticker_list = ['AAPL', 'MFST', 'GOOGL', 'NVDA', 'META', 'UBER', 'TLSA', 'ORCL', 'CRM', 'NFLX']
training_split = 0.8
test_split = 1 - training_split
# If randomize testing is false, use existing training and testing list
training_tickers = ['AAPL', 'MFST', 'GOOGL', 'META', 'TLSA', 'ORCL', 'CRM', 'NFLX']
testing_tickers = ['NVDA', 'UBER']

scalers = {}
dates = {}

sequence_length = 20

input_dim = 1

# nodes per layer
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# sns.set_style("darkgrid")


class LSTM(nn.Module):
    def __init__(self, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim):
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

    def save_model(self, path="lstm_multiple_ticker.pt"):
        torch.save(self.state_dict(), path)


class GRU(nn.Module):
    def __init__(self, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim):
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

    def save_model(self, path="gru_multiple_ticker.pt"):
        torch.save(self.state_dict(), path)


def get_data_frames(tickers: List[str], existing_training, existing_testing):
    if not randomize_testing:
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


def sequence_and_split(ticker_prices_map):
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

    if randomize_testing:
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


def train(model, ticker_x_map, ticker_y_map):
    for ticker in ticker_x_map:
        x_train = torch.from_numpy(ticker_x_map[ticker]).type(torch.Tensor).to(device)
        y_train = torch.from_numpy(ticker_y_map[ticker]).type(torch.Tensor).to(device)

        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        history = np.zeros(num_epochs)
        for t in range(num_epochs):
            prediction = model(x_train)

            loss = criterion(prediction, y_train)
            print(f"Epoch: {t}, MSE: {loss.item()}")
            history[t] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print()

    model.save_model()

    return model


def eval_model(model, x_train_map, y_train_map, x_test_map, y_test_map, model_type):
    for ticker in x_test_map:
        x_test = torch.from_numpy(x_test_map[ticker]).type(torch.Tensor).to(device)
        # y_test = torch.from_numpy(y_test_map[ticker]).type(torch.Tensor).to(device)
        y_test = y_test_map[ticker]
        testing_predictions = model(x_test)
        testing_predictions = testing_predictions.cpu().detach().numpy()
        # test_mse = mean_squared_error(y_test[:, 0], testing_predictions[:, 0])

        scaler = copy.deepcopy(scalers[ticker])
        scaler2 = copy.deepcopy(scalers[ticker])

        y_test_inverted = scaler.inverse_transform(np.array(y_test))
        testing_predictions_inverted = scaler2.inverse_transform(np.array(testing_predictions))

        print(y_test_inverted[:, 0])
        print(testing_predictions_inverted[:, 0])

        test_rmse = math.sqrt(mean_squared_error(y_test_inverted[:, 0], testing_predictions_inverted[:, 0]))

        print(f"{model_type} Test RMSE for {ticker}: {test_rmse}")

        plot_singular_complete(testing_predictions, y_test, ticker, model_type)


def plot_singular_complete(predicted, actual, ticker, model_info):
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
    fig.set_figheight(8)
    fig.set_figwidth(10)
    sns.set_style("darkgrid")
    sns.lineplot(data=df, x=df.index, y='Actual price', label='Actual')
    sns.lineplot(data=df, x=df.index, y='Predicted price', label='Predicted')

    n = len(dates_x)
    step_size = max(n // 4, 1)  # Divide into roughly 4 sections
    display_dates = dates_x[::step_size]
    plt.xticks(display_dates, rotation=45)

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Predictions - {model_info}")
    plt.legend()
    plt.show()


def plot_results(y_train, y_test, lstm_training_predictions, lstm_testing_predictions,
                 gru_train_predictions, gru_testing_predictions, lstm_history, gru_history):
    # lstm graphs
    df = pd.DataFrame({'Actual': np.concatenate((y_train.flatten(), y_test.flatten())),
                       'Predicted': np.concatenate((lstm_training_predictions.flatten(),
                                                    lstm_testing_predictions.flatten()))})

    split_index = int(training_split * (len(lstm_training_predictions) + len(lstm_testing_predictions)))
    predicted_1 = df['Predicted'].iloc[:split_index]
    predicted_2 = df['Predicted'].iloc[split_index:]

    fig = plt.figure()

    sns.set_style("darkgrid")
    plt.subplot(2, 2, 1)
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    sns.lineplot(data=df, x=df.index, y='Actual', label='Actual')
    sns.lineplot(x=predicted_1.index, y=predicted_1, label='Predicted Training', color='blue')
    sns.lineplot(x=predicted_2.index, y=predicted_2, label='Predicted Testing', color='red')

    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.title("LSTM Predictions")
    plt.legend()

    plt.subplot(2, 2, 2)
    ax = sns.lineplot(data=lstm_history, color='royalblue')
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    ax.set_title("LSTM Training Loss", size=14, fontweight='bold')

    # GRU graphs
    df = pd.DataFrame({'Actual': np.concatenate((y_train.flatten(), y_test.flatten())),
                       'Predicted': np.concatenate(
                           (gru_train_predictions.flatten(), gru_testing_predictions.flatten()))})

    split_index = int(training_split * (len(gru_train_predictions) + len(gru_testing_predictions)))
    predicted_1 = df['Predicted'].iloc[:split_index]
    predicted_2 = df['Predicted'].iloc[split_index:]

    sns.set_style("darkgrid")
    plt.subplot(2, 2, 3)
    sns.lineplot(data=df, x=df.index, y='Actual', label='Actual')
    sns.lineplot(x=predicted_1.index, y=predicted_1, label='Predicted Training', color='blue')
    sns.lineplot(x=predicted_2.index, y=predicted_2, label='Predicted Testing', color='red')

    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.title("GRU Predictions")
    plt.legend()

    plt.subplot(2, 2, 4)
    ax = sns.lineplot(data=gru_history, color='royalblue')
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    ax.set_title("GRU Training Loss", size=14, fontweight='bold')

    fig.set_figheight(10)
    fig.set_figwidth(16)

    plt.show()


if __name__ == '__main__':
    ticker_dataframes_map = get_data_frames(ticker_list, training_tickers, testing_tickers)
    prices = preprocess(ticker_dataframes_map)
    x_train, y_train, x_test, y_test = sequence_and_split(prices)

    print(f"Training tickers: {str(list(x_train.keys()))}")
    print(f"Testing tickers: {str(list(x_test.keys()))}")

    x_train_copy = copy.deepcopy(x_train)
    y_train_copy = copy.deepcopy(y_train)

    if use_model == 'LSTM':
        lstm = train(LSTM().to(device), x_train, y_train)

        eval_model(lstm, x_train_copy, y_train_copy, x_test, y_test, "LSTM")
    elif use_model == 'GRU':
        gru = train(GRU().to(device), x_train, y_train)

        eval_model(gru, x_train_copy, y_train_copy, x_test, y_test, "GRU")
