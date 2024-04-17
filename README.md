# Predicting Stocks using RNNs

- `.env` file required for AlphaVantage API key in the form `MY_API_KEY=...`
- `multiple_ticker_models.py` contains the code to train our multiple ticker models (LSTM and GRU)
- `gridsearch_multiple_tickers.py` runs grid search on the model from the above file and logs output as a csv `models/gridsearch.csv`
- `plotlydemo.py` runs our [Streamlit](https://streamlit.io/) web application
  - Run using `streamlit run plotlydemo.py`
- `/experimentation` contains code from our lengthy experimentation/research we conducted at the start of the project
- `/APIs` contain custom fetching code to retrieve stock and fundamentals data using the AlphaVantage API
- `/models` contain our models and grid search results
