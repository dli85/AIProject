import argparse
from basic_models import *

def main(ticker):
    # Your application logic here
    print(f'Selected ticker: {ticker}')

    df_date_close, dates = get_data_frame(ticker)

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
        eval_model(lstm, y_train, lstm_training_predictions, x_test, y_test, scaler)

    gru, gru_training_prediction, gru_history, = train(GRU(), x_train_for_gru, y_train)
    y_train_for_gru, y_test_for_gru, gru_train_predictions, gru_testing_predictions = \
        eval_model(gru, y_train, gru_training_prediction, x_test, y_test, scaler)

    plot_results(scaler.inverse_transform(np.copy(y_train_initial)), scaler.inverse_transform(np.copy(y_test_initial)),
                 lstm_train_prediction, lstm_testing_predictions, gru_train_predictions, gru_testing_predictions,
                 lstm_history, gru_history, dates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict a USA stock ticker.')
    parser.add_argument('--ticker', type=str, required=True, help='The ticker to predict')
    args = parser.parse_args()
    main(args.ticker)
