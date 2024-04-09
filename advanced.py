import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ratios import get_all_ratios
import numpy as np
import torch
import torch.nn as nn

training_split = 0.95
test_split = 1 - training_split
sequence_length = 5

input_dim = 2

# nodes per layer
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100


scaler_storage = {}
# pd.set_option('display.max_columns', 10)


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


def train(model, dataframes):
    for df in dataframes:
        testing_df = df.copy()
        # testing_df.drop('fiscal_closing_date', inplace=True)
        df_array =testing_df.to_numpy()
        df_array = np.transpose(df_array)
        data = []
        for index in range(len(df_array[0]) - sequence_length):
            temp = []
            for row in df_array:
                pass
            data.append(df_array[''])

        print(df)
        for a in df_array:
            print(*a)
        # print(df_array)
        input()


def transform_stock_data(dataframes, max_len=None):
  """
  Transforms a list of DataFrames containing financial ratios and close prices
  into a 4D PyTorch tensor for a stock prediction model. Close price is
  included as an input feature.

  Args:
      dataframes (list): A list of Pandas DataFrames, where each DataFrame
          represents data for a single stock.
      max_len (int, optional): Maximum sequence length for padding. If None,
          the maximum length among all DataFrames is used.

  Returns:
      torch.Tensor: A 4D PyTorch tensor with shape
          (num_stocks, num_timesteps, num_features, sequence_length).
  """

  all_data = []  # Store features (including close price) for all stocks
  max_len = 0

  for df in dataframes:
    features = df.values  # All columns are now input features
    max_len = max(max_len, len(features))
    all_data.append(features)

  # Pad sequences to same length
  padded_data = [np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode='constant', constant_values=0) for seq in all_data]

  # Convert to NumPy array and PyTorch tensor
  data = np.stack(padded_data)
  tensor_data = torch.from_numpy(data)

  return tensor_data


# Tips for guiding LSTM to focus on predicting the closing price:
# 1. Make closing price the last feature
# 2. Use MSE? Importantly: put higher weight to the error on close price.
# 3. single output node: Design the LSTM's output layer to have a single node representing the next
#    closing price prediction. This structures the model to explicitly focus on this single output.
# 4. attention mechanisms
# 5.
if __name__ == '__main__':
    data = get_all_ratios()
    dataframes = []
    for ticker in data:
        df = create_stock_dataframe(data, ticker)
        if df is None:
            continue
        dataframes.append(df)

    train(LSTM(), dataframes)

    # tensors = transform_stock_data(dataframes)
