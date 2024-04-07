import streamlit as st
import time
import numpy as np
from  basic_models import *

# st.title('Predicting Stocks using RNNs')
# st.text('This is a simple web app to predict stocks using RNNs')
# ticker = st.text_input('Ticker', 'AAPL')
# st.write('The current movie title is', ticker)

df_date_close, dates = get_data_frame('AAPL')
st.line_chart(data=df_date_close)

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

df = pd.DataFrame({'Actual': np.concatenate((y_train.flatten(), y_test.flatten())),
                    'Predicted': np.concatenate((lstm_training_predictions.detach().numpy().flatten(),
                                                lstm_testing_predictions.flatten()))})

split_index = int(training_split * (len(lstm_training_predictions) + len(lstm_testing_predictions)))
predicted_1 = df['Predicted'].iloc[:split_index]
predicted_2 = df['Predicted'].iloc[split_index:]

st.line_chart(data=df, x=None, y='Actual')
st.line_chart(data=predicted_1)
st.line_chart(data=predicted_2)



# sns.lineplot(data=df, x=df.index, y='Actual', label='Actual')
# sns.lineplot(x=predicted_1.index, y=predicted_1, label='Predicted Training', color='blue')
# sns.lineplot(x=predicted_2.index, y=predicted_2, label='Predicted Testing', color='red')
# plt.show()



# plot_results(scaler.inverse_transform(np.copy(y_train_initial)), scaler.inverse_transform(np.copy(y_test_initial)),
#                 lstm_train_prediction, lstm_testing_predictions, gru_train_predictions, gru_testing_predictions,
#                 lstm_history, gru_history, dates)

#-----------------#

# df_date_close, dates = get_data_frame('AAPL')

# st.write('Dataframe:')
# st.write(df_date_close)

# chart = st.line_chart(df_date_close)
