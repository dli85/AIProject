import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ratios import get_all_ratios

scaler_storage = {}


def create_stock_dataframe(data, ticker):
    stock_data = data[ticker]
    data_list = []
    for date in stock_data:
        ratios = dict(stock_data[date])
        ratios['fiscal_closing_date'] = date
        data_list.append(ratios)

    print(data_list[0])

    df = pd.DataFrame(data_list)
    df.set_index('fiscal_closing_date', inplace=True)
    print(df)
    print(df.columns)

    scalers = {}
    for col in df.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        scalers[col] = scaler

    scaler_storage[ticker] = scalers

    print(scaler_storage)

    print(df)

    print("After")
    for col in df.columns:
        scaler = scaler_storage[ticker][col]
        df[col] = scaler.inverse_transform(df[[col]].values)

    print(df)

    return df


# Tips for guiding LSTM to focus on predicting the closing price:
# 1. Make closing price the last feature
# 2. Use MSE? Importantly: put higher weight to the error on close price.
# 3. single output node: Design the LSTM's output layer to have a single node representing the next
#    closing price prediction. This structures the model to explicitly focus on this single output.
# 4. attention mechanisms
# 5.
if __name__ == '__main__':
    data = get_all_ratios()
    create_stock_dataframe(data, 'AAPL')
