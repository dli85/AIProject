from dotenv import load_dotenv
import requests
import os

BASE_AV_URL = 'https://www.alphavantage.co/query'

load_dotenv()

def get_close_price(date, ticker):
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY')
    }

    response = requests.get(BASE_AV_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        print(data['Time Series (Daily)'])
        #return data['Time Series (Daily)']
        # return data['Time Series (Daily)'][date]
    else:
        return None


def get_earnings(date, ticker):
    params = {
        'function': 'EARNINGS',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY')
    }

    response = requests.get(BASE_AV_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        print(data['quarterlyEarnings'])
        #return data['quarterlyEarnings']
    else:
        return None

if __name__ == "__main__":
    print(get_earnings('2023-12-31', 'AAPL'))
    #print(get_close_price('2023-12-31', 'AAPL'))
