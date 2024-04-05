import math

import pandas as pd
from APIs.close_price import get_all_adjusted_prices
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns
import random

training_split = 0.95
test_split = 1 - training_split
sequence_length = 20

input_dim = 1

# nodes per layer
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

device = torch.device("cuda")

# sns.set_style("darkgrid")


class LSTM(nn.Module):
    def __init__(self, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        # out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def save_model(self, path="lstm.pt"):
        torch.save(self.state_dict(), path)


class GRU(nn.Module):
    def __init__(self, input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def save_model(self, path="gru.pt"):
        torch.save(self.state_dict(), path)


def get_data_frames(ticker_list):
    for ticker in ticker_list:
        data = get_all_adjusted_prices(ticker)
        dates = [x[1] for x in data]

        df = pd.DataFrame(data, columns=["Date", "Close"])
        # dates = df["Date"].values
        df["Date"] = pd.to_datetime(df["Date"])
        df["Close"] = df["Close"].astype(float)
        df = df.set_index("Date")

        return df, dates


def preprocess(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    prices = df.copy()[["Close"]]

    prices["Close"] = scaler.fit_transform(prices["Close"].values.reshape(-1, 1))

    return prices, scaler


def split(stock):
    raw = stock.to_numpy()[::-1]
    data = []

    for index in range(len(raw) - sequence_length):
        data.append(raw[index: index + sequence_length])

    data = np.array(data)
    test_set_size = int(np.round(test_split * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    # x_train[i] = some sequence of prices
    # y_train[i] = the next price

    plt.plot(y_train, color='blue', marker='*', linestyle='--')
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.title("Data Plot")
    plt.grid(True)
    plt.show()

    return [x_train, y_train, x_test, y_test]


def train(model, x_train, y_train):
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = np.zeros(num_epochs)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    for t in range(num_epochs):
        prediction = model(x_train)
        print(prediction)
        print(x_train.shape)
        input()

        loss = criterion(prediction, y_train)
        print(f"Epoch: {t}, MSE: {loss.item()}")
        history[t] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.save_model()

    return model, prediction, history


def eval_model(model, y_train, train_predictions, x_test, y_test, scaler, model_type):
    # x_test = torch.from_numpy(x_test).type(torch.Tensor)
    testing_predictions = model(x_test)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    sns.set_style("darkgrid")

    # invert predictions back to stock values
    train_predictions = scaler.inverse_transform(train_predictions.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    testing_predictions = scaler.inverse_transform(testing_predictions.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    train_mse = math.sqrt(mean_squared_error(y_train[:, 0], train_predictions[:, 0]))
    test_mse = math.sqrt(mean_squared_error(y_test[:, 0], testing_predictions[:, 0]))
    print(y_train, train_predictions)
    print(f"Training Mean squared error for {model_type}: {train_mse}")
    print(f"Testing Mean squared error for {model_type}: {test_mse}")

    return y_train, y_test, train_predictions, testing_predictions


def plot_results(y_train, y_test, lstm_training_predictions, lstm_testing_predictions,
                 gru_train_predictions, gru_testing_predictions, lstm_history, gru_history, dates):


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
                       'Predicted': np.concatenate((gru_train_predictions.flatten(), gru_testing_predictions.flatten()))})

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


def predict_next_n_days(model, n, x_test, y_test, scaler, sequence_length):
    def shift(arr, new_elem):
        """
        Shifts an array to the right by 1: removes the first element and adds an element at the end
        :param arr: The arr to shift
        :param new_elem: Element to be added. Should be in the shape [number]
        :return: THe shifted array
        """
        modified_arr = [arr[0][1:]]
        # Create a new array to hold the element to be added
        new_element = np.array([new_elem])
        # Combine the arrays (concatenate along the correct axis)
        modified_arr[0] = np.concatenate((modified_arr[0], new_element), axis=0)

        return modified_arr

    # create the window: most recent sequence of prices
    x_test = x_test.numpy()
    window = [x_test[-1]]
    window = shift(window, y_test[-1])

    predictions = []

    for i in range(n):  # Predict the next n days
        window_tensors = torch.from_numpy(np.array(window)).type(torch.Tensor)
        next_prediction = model(window_tensors)
        predicted_val = next_prediction[0][0].detach().numpy()
        window = shift(window, [predicted_val])

        predictions.append(predicted_val.item())

    print(predictions)

    return predictions


if __name__ == '__main__':
    df_date_close, dates = get_data_frame('NVDA')

    prices, scaler = preprocess(df_date_close)
    x_train, y_train, x_test, y_test = split(prices)

    y_train_initial = np.copy(y_train)
    y_test_initial = np.copy(y_test)

    x_train_for_lstm = torch.from_numpy(x_train).type(torch.Tensor)
    x_test_for_lstm = torch.from_numpy(x_test).type(torch.Tensor)

    x_train_for_gru = torch.from_numpy(x_train).type(torch.Tensor)
    x_test_for_gru = torch.from_numpy(x_test).type(torch.Tensor)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)

    lstm, lstm_training_predictions, lstm_history, = train(LSTM(), x_train_for_lstm, y_train)
    y_train_for_lstm, y_test_for_lstm, lstm_train_prediction, lstm_testing_predictions = \
        eval_model(lstm, y_train, lstm_training_predictions, x_test, y_test, scaler, "LSTM")

    gru, gru_training_prediction, gru_history, = train(GRU(), x_train_for_gru, y_train)
    y_train_for_gru, y_test_for_gru, gru_train_predictions, gru_testing_predictions = \
        eval_model(gru, y_train, gru_training_prediction, x_test, y_test, scaler, "GRU")

    plot_results(scaler.inverse_transform(np.copy(y_train_initial)), scaler.inverse_transform(np.copy(y_test_initial)),
                 lstm_train_prediction, lstm_testing_predictions, gru_train_predictions, gru_testing_predictions,
                 lstm_history, gru_history, dates)
    predict_next_n_days(gru, 30, x_test, np.copy(y_test_initial), scaler, sequence_length)
    print(gru)
