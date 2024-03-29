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
        earnings_dates = get_earnings(ticker)
        reported_date = [earning_date['reportedDate'] for earning_date in earnings_dates if earning_date['fiscalDateEnding'] == date][0]
        return data['Time Series (Daily)'][reported_date]['4. close']
    else:
        return None


def get_earnings(ticker):
    params = {
        'function': 'EARNINGS',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY')
    }

    response = requests.get(BASE_AV_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        list_dates = []
        for i in range(len(data['quarterlyEarnings'])):
            return_dict = {'fiscalDateEnding': data['quarterlyEarnings'][i]['fiscalDateEnding'],
                           'reportedDate': data['quarterlyEarnings'][i]['reportedDate']}
            list_dates.append(return_dict)
        return list_dates
    else:
        return None

if __name__ == "__main__":
    #print(get_earnings('2023-12-31', 'AAPL'))
    print(get_close_price('2023-12-31', 'AAPL'))
