import copy
import math
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import multiple_ticker_models as mtm
from APIs.close_price import get_all_adjusted_prices

import streamlit as st

st.title('Predicting Stocks using RNNs')

target_ticker = st.text_input('Enter the ticker you want to predict', 'AAPL')
target_ticker = target_ticker.upper()

print("target ticker")
print(target_ticker)


def plotly_plot_single(predicted, actual, ticker, model_info):
    scaler_predicted = copy.deepcopy(mtm.scalers[ticker])
    scaler_actual = copy.deepcopy(mtm.scalers[ticker])
    predicted = scaler_predicted.inverse_transform(predicted)
    actual = scaler_actual.inverse_transform(actual)

    dates_x = mtm.dates[ticker].copy()
    dates_x.reverse()
    dates_x = dates_x[mtm.sequence_length:]
    dates_x = pd.to_datetime(dates_x)

    df = pd.DataFrame({'Actual price': actual.flatten(),
                       'Predicted price': predicted.flatten()},
                      index=dates_x)

    fig = px.line(df, title=f'Predictions for {ticker} - {model_info}')
    return fig

    # fig.show()


def eval_model(model, x_train_map, y_train_map, x_test_map, y_test_map, model_type):
    ret = {}
    for ticker in x_test_map:
        x_test = torch.from_numpy(x_test_map[ticker]).type(torch.Tensor).to(mtm.device)
        # y_test = torch.from_numpy(y_test_map[ticker]).type(torch.Tensor).to(device)
        y_test = y_test_map[ticker]
        testing_predictions = model(x_test)
        testing_predictions = testing_predictions.cpu().detach().numpy()
        # test_mse = mean_squared_error(y_test[:, 0], testing_predictions[:, 0])

        scaler = copy.deepcopy(mtm.scalers[ticker])
        scaler2 = copy.deepcopy(mtm.scalers[ticker])

        y_test_inverted = scaler.inverse_transform(np.array(y_test))
        testing_predictions_inverted = scaler2.inverse_transform(np.array(testing_predictions))

        print(y_test_inverted[:, 0])
        print(testing_predictions_inverted[:, 0])

        test_rmse = math.sqrt(mean_squared_error(y_test_inverted[:, 0], testing_predictions_inverted[:, 0]))

        # print(f"{model_type} Test RMSE for {ticker}: {test_rmse}")

        # mtm.plot_singular_complete(testing_predictions, y_test, ticker, model_type)
        ret[ticker] = (testing_predictions, y_test, test_rmse)
        # TODO remove the func call later after testing
        # plotly_plot_single(testing_predictions, y_test, ticker, model_type)
    
    return ret

def render_streamlit_page():
    # These are just there to make the code work lol
    ticker_list = list()
    training_tickers = list()
    testing_tickers = [target_ticker]

    # Using saved model files
    lstm_model = mtm.LSTM()
    lstm_model.load_state_dict(torch.load('./models/lstm_multiple_ticker2.pt', map_location=torch.device('cpu')))
    lstm_model.eval()

    gru_model = mtm.GRU()
    gru_model.load_state_dict(torch.load('./models/gru_multiple_ticker.pt', map_location=torch.device('cpu')))
    gru_model.eval()

    ticker_dataframes_map = mtm.get_data_frames(ticker_list, training_tickers, testing_tickers)
    prices = mtm.preprocess(ticker_dataframes_map)
    x_train, y_train, x_test, y_test = mtm.sequence_and_split(prices)
    x_train_copy = copy.deepcopy(x_train)
    y_train_copy = copy.deepcopy(y_train)

    ret = eval_model(lstm_model, x_train_copy, y_train_copy, x_test, y_test, "LSTM")
    ret_gru = eval_model(gru_model, x_train_copy, y_train_copy, x_test, y_test, "GRU")

    ret = list(ret.items())[0][1]
    ret_gru = list(ret_gru.items())[0][1]

    f = plotly_plot_single(ret[0], ret[1], target_ticker, 'LSTM')
    st.plotly_chart(f, use_container_width=False)
    st.write(f"RMSE for LSTM for ticker {target_ticker}: {ret[2]} USD")

    f2 = plotly_plot_single(ret_gru[0], ret_gru[1], target_ticker, 'GRU')
    st.plotly_chart(f2, use_container_width=False)
    st.write(f"RMSE for GRU for ticker {target_ticker}: {ret_gru[2]} USD")


if __name__ == "__main__":
    render_streamlit_page()
