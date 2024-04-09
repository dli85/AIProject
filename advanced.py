import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ratios import get_all_ratios
import numpy as np
import torch
import torch.nn as nn


# enums
CLOSING_PRICE = 'fiscal_closing_date'

models_path = './models'

training_split = 0.95
test_split = 1 - training_split
input_sequence_length = 4


input_dim = 8

# nodes per layer
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100


scaler_storage = {}
# pd.set_option('display.max_columns', 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    def save_model(self, path=f"{models_path}/lstm_advanced.pt"):
        torch.save(self.state_dict(), path)


def create_stock_dataframe(data, ticker):
    stock_data = data[ticker]
    data_list = []
    for date in stock_data:
        ratios = dict(stock_data[date])
        ratios['fiscal_closing_date'] = date
        data_list.append(ratios)

    if len(data_list) == 0:
        return


    df = pd.DataFrame(data_list)
    df.set_index('fiscal_closing_date', inplace=True)
    # print(df)
    # print(df.columns)

    scalers = {}
    for col in df.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = scaler

    # print(df)

    scaler_storage[ticker] = scalers

    # print(scaler_storage)


    # Test inverse transformations
    # print("After")
    # for col in df.columns:
    #     scaler = scaler_storage[ticker][col]
    #     df[col] = scaler.inverse_transform(df[[col]].values)
    #
    # print(df)
    # input()

    return df


def sequence_data(dataframes_map):
    sequenced_data = dict()
    skipped_tickers = []
    for ticker in dataframes_map:
        df = dataframes_map[ticker]
        testing_df = df.copy()
        if len(df) <= input_sequence_length + 1:
            skipped_tickers.append(ticker)
            continue
        df_array = testing_df.to_numpy()
        df_array = np.transpose(df_array)
        X = []
        y = []
        for index in range(len(df_array[0]) - input_sequence_length):
            temp = []
            for row in df_array:
                temp.append(row[index:index + input_sequence_length])
            X.append(temp)
            y.append(df_array[-1][index + input_sequence_length])
        #
        # print(df)
        # for a in df_array:
        #     print([*a])
        # # print(df_array)
        # print("\n\n DATA:", ticker)
        # try:
        #     for row in X[0]:
        #         print(row)
        # except:
        #     print(ticker)
        #     input()
        # print("p2")
        # for row in X[-1]:
        #     print(row)
        # print(y)
        # print(len(X))
        # print(len(y))
        sequenced_data[ticker] = {
            'X': X,
            'y': y
        }

    print("Skipped the following tickers, not enough data:")
    print(skipped_tickers)

    return sequenced_data


def train(model, training_data, num_epochs=100):
    for ticker in training_data:
        temp = []
        X_train = np.array(training_data[ticker]['X'])
        for i in range(len(X_train)):
            temp.append(np.array(X_train[i].transpose()))
        # print(np.array(training_data[ticker]['X'][0]).transpose())
        # print(np.array(training_data[ticker]['X']).shape)
        # print(len(training_data[ticker]['X'][0]))
        # input()
        # X_train = torch.from_numpy(np.array(training_data[ticker]['X'])).type(torch.Tensor).to(device)
        X_train = torch.from_numpy(np.array(temp)).type(torch.Tensor).to(device)
        y_train = torch.from_numpy(np.array(training_data[ticker]['y'])).type(torch.Tensor).to(device)

        # print(X_train.shape)
        # print(y_train.shape)
        # input()

        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for t in range(num_epochs):
            prediction = model(X_train)
            prediction = prediction.squeeze(1)

            loss = criterion(prediction, y_train)

            print(f"Epoch: {t}, MSE: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.save_model()
    return model


def eval_model_and_plot(model, testing_data):
    for ticker in testing_data:
        pass


def train_ratio_model(testing_tickers):
    data = get_all_ratios()
    tickers_dataframes_map = dict()
    input_size = 0
    for ticker in data:
        df = create_stock_dataframe(data, ticker)
        if df is None:
            continue
        tickers_dataframes_map[ticker] = df
        input_size = len(df.columns)

    testing_dfs = dict()
    training_dfs = dict()
    for ticker in tickers_dataframes_map:
        if ticker in testing_tickers:
            testing_dfs[ticker] = tickers_dataframes_map[ticker]
        else:
            training_dfs[ticker] = tickers_dataframes_map[ticker]

    testing_X_y = sequence_data(testing_dfs)
    training_X_y = sequence_data(training_dfs)

    train(LSTM(input_dim=8).to(device), training_X_y)


# Tips for guiding LSTM to focus on predicting the closing price:
# 1. Make closing price the last feature
# 2. Use MSE? Importantly: put higher weight to the error on close price.
# 3. single output node: Design the LSTM's output layer to have a single node representing the next
#    closing price prediction. This structures the model to explicitly focus on this single output.
# 4. attention mechanisms
# 5.
if __name__ == '__main__':
    train_ratio_model(['AAPL', 'META', 'NVDA'])
