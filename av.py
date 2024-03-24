import time
import requests
import requests
import pandas as pd
from io import StringIO
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stderr)
stdout_handler.setLevel(logging.DEBUG)  # Set the level for this handler
logger.addHandler(stdout_handler)

BASE_AV_URL = 'https://www.alphavantage.co/query'

BASE_PARAMS = {
    'function': 'TIME_SERIES_INTRADAY',
    'symbol': 'QQQ',
    'interval': '5min',
    'apikey': 'Q9O1WK5BTAERX17D',
    'outputsize': 'full',
    'datatype': 'csv',
    # 'month': '2024-02'  # Note: 'month' is not a standard API parameter for Alpha Vantage and might be ignored by the API.
}

def _get_all_months(start, end):
    start_year, start_month = map(int, start.split('-'))
    end_year, end_month = map(int, end.split('-'))
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                continue
            months.append(f'{year}-{str(month).zfill(2)}')
    return months

def _get_monthly_data(symbol, month):
    logger.info(f"Getting data for {symbol} for month {month}")
    params = BASE_PARAMS.copy()
    params['symbol'] = symbol
    params['month'] = month
    response = requests.get(BASE_AV_URL, params=params)
    csv_data = StringIO(response.text)
    data = pd.read_csv(csv_data)
    logger.info(f"Got data for {symbol} for month {month}")
    return data

def get_data(symbol, start, end):
    months = _get_all_months(start, end)
    data = pd.DataFrame()
    for month in months:
        data = pd.concat([data, _get_monthly_data(symbol, month)])
    return data

if __name__ == "__main__":
    df = get_data('QQQ', '2024-01', '2024-02')
    print(df.head())
    df.to_csv(f'qqq_df-{time.time()}.csv', index=False)


