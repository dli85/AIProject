import pandas as pd
from APIs.close_price import get_all_adjusted_prices
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

training_split = 0.8
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


def get_data_frame(ticker):
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

    print(y_train)
    print(y_test)

    # plt.plot(y_train, color='blue', marker='*', linestyle='--')
    # plt.xlabel("Index")
    # plt.ylabel("Values")
    # plt.title("Data Plot")
    # plt.grid(True)
    # plt.show()

    return [x_train, y_train, x_test, y_test]


def train(model, x_train, y_train):
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = np.zeros(num_epochs)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    for t in range(num_epochs):
        prediction = model(x_train)

        loss = criterion(prediction, y_train)
        print(f"Epoch: {t}, MSE: {loss.item()}")
        history[t] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.save_model()

    return model, prediction, history


def eval_model(model, y_train, train_predictions, x_test, y_test, scaler):
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

    print(y_test)
    print(type(y_test))

    return y_train, y_test, train_predictions, testing_predictions


def plot_results(y_train, y_test, train_predictions, testing_predictions, history, dates):
    df = pd.DataFrame({'Actual': np.concatenate((y_train.flatten(), y_test.flatten())),  # Concatenate
                       'Predicted': np.concatenate((train_predictions.flatten(), testing_predictions.flatten()))})

    split_index = int(training_split * (len(train_predictions) + len(testing_predictions)))
    predicted_1 = df['Predicted'].iloc[:split_index]
    predicted_2 = df['Predicted'].iloc[split_index:]

    fig = plt.figure()
    sns.set_style("darkgrid")
    plt.subplot(1, 2, 1)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    sns.lineplot(data=df, x=df.index, y='Actual', label='Actual')
    sns.lineplot(x=predicted_1.index, y=predicted_1, label='Predicted Training', color='blue')
    sns.lineplot(x=predicted_2.index, y=predicted_2, label='Predicted Testing', color='red')

    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Predictions")
    plt.legend()

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=history, color='royalblue')
    ax.set_xlabel("Epoch", size=14)
    ax.set_ylabel("Loss", size=14)
    ax.set_title("Training Loss", size=14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)

    plt.show()


# def predict_next_30_days(model, y_test, scaler, sequence_length):
#     """
#     Predicts the stock prices for the next 30 days.
#
#     Args:
#         model (nn.Module): The trained LSTM or GRU model.
#         y_test (np.ndarray): The most recent 20% of close prices (used as starting point).
#         scaler (MinMaxScaler): The scaler used for preprocessing.
#         sequence_length (int): The sequence length used in model training
#
#     Returns:
#         list: A list of predicted stock prices for the next 30 days.
#     """
#
#     predictions = []
#     current_batch = y_test[-sequence_length:]  # Last sequence_length prices from y_test
#     current_batch = scaler.transform(current_batch.reshape(-1, 1))  # Scale
#
#     for i in range(30):  # Predict the next 30 days
#         current_batch = torch.from_numpy(current_batch).type(torch.Tensor).to(device)
#         output = model(current_batch)
#         prediction = scaler.inverse_transform(output.cpu().detach().numpy())
#         predictions.append(prediction[0, 0])
#
#         # Add the prediction to the input and remove the oldest value:
#         current_batch = np.append(current_batch, prediction).reshape(-1, 1)  # Reshape directly
#         current_batch = current_batch[1:]  # Remove oldest value
#         current_batch = scaler.transform(current_batch)
#
#     print(predictions)
#
#     return predictions


if __name__ == '__main__':
    df_date_close, dates = get_data_frame('AAPL')

    prices, scaler = preprocess(df_date_close)
    x_train, y_train, x_test, y_test = split(prices)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)

    model, training_prediction, history, = train(GRU(), x_train, y_train)
    y_train, y_test, train_predictions, testing_predictions = \
        eval_model(model, y_train, training_prediction, x_test, y_test, scaler)
    plot_results(y_train, y_test, train_predictions, testing_predictions, history, dates)
    # predict_next_30_days(model, y_test, scaler, sequence_length)
    # print(x_train)
