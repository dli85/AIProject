import requests
import os

BASE_AV_URL = 'https://www.alphavantage.co/query'

def get_income_statement(ticker):

    params = {
        'function': 'INCOME_STATEMENT',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY')
    }

    response = requests.get(BASE_AV_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        return data['quarterlyReports']