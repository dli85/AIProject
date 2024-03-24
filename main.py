import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

sns.set_style("darkgrid")


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
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

    def save_model(self, path="lstm.pt"):
        torch.save(self.state_dict(), path)


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
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


def preprocess(data_df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    price = data_df.copy()[['Close']]
    price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1, 1))

    return price, scaler


def split(stock, lookback=20):
    raw = stock.to_numpy()
    data = []

    for index in range(len(raw) - lookback):
        data.append(raw[index: index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - test_set_size

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


def load_data(path: str):
    data = pd.read_csv(path).sort_values('Date')
    data.head()
    # display(data)
    return data


def display(data_df):
    plt.plot(data_df[['Close']])
    plt.show()


def train_lstm(x_train, y_train):
    model = LSTM()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    history = np.zeros(num_epochs)
    for t in range(num_epochs):
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train_lstm)
        print(f"Epoch: {t}, MSE: {loss.item()}")
        history[t] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.save_model()
    return model, y_train_pred


def test_model(model, y_train, y_train_pred, x_test, y_test, scaler):
    # make predictions
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    # y_test = torch.from_numpy(y_test).type(torch.Tensor)
    y_test_pred = model(x_test)
    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    lstm = []

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))
    lstm.append(trainScore)
    lstm.append(testScore)
    print("Train score:", trainScore)
    print("Test score:", testScore)


if __name__ == '__main__':
    data_df = load_data('amazon.csv')
    price, scaler = preprocess(data_df)
    x_train, y_train, x_test, y_test = split(price)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)

    # y_train = torch.from_numpy(y_train).type(torch.Tensor)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model_lstm, y_train_pred = train_lstm(x_train, y_train)
    test_model(model_lstm, y_train, y_train_pred, x_test, y_test, scaler)


