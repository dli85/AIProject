from dotenv import load_dotenv
import requests
import os
import json

from datetime import date, datetime, timedelta

BASE_AV_URL = 'https://www.alphavantage.co/query'
load_dotenv()


def within_x_days(date1_str, date2_str, x=15):
    date1 = date.fromisoformat(date1_str)
    date2 = date.fromisoformat(date2_str)

    delta = abs(date1 - date2).days

    return delta <= x


def get_adjusted_prices_complete_json(ticker, save_json=False):
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY'),
        'outputsize': 'full',
    }

    response = requests.get(BASE_AV_URL, params=params)

    if response.status_code != 200:
        raise Exception("TIME_SERIES_DAILY_ADJUSTED request failed")

    data = response.json()
    if save_json:
        with open(f"{ticker} daily adjusted prices.json", 'w') as file:
            json.dump(data, file, indent=4)

    return data['Time Series (Daily)']


def get_earning_dates(ticker):
    params = {
        'function': 'EARNINGS',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY')
    }

    response = requests.get(BASE_AV_URL, params=params)
    if response.status_code != 200:
        raise Exception("TIME_SERIES_DAILY_ADJUSTED request failed")

    data = response.json()

    # with open('earnings.json', 'w') as file:
    #     json.dump(data, file, indent=4)

    return data['quarterlyEarnings']


def get_surprise(fiscal_end_date, earnings_dates):
    for report in earnings_dates:
        if within_x_days(report['fiscalDateEnding'], fiscal_end_date):
            return report['surprise']


def get_adjusted_price_at_date(fiscal_end_date, earnings_dates, adjusted_prices):
    reported_date = match_fiscal_date_ending(fiscal_end_date, earnings_dates)

    if reported_date in adjusted_prices:
        return adjusted_prices[reported_date]['5. adjusted close']

    reported_date_copy = reported_date

    def increment_date(date_str):
        """Increments a date by one day."""
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        new_date = date_obj + timedelta(days=1)
        return new_date.strftime("%Y-%m-%d")

    def decrement_date(date_str):
        """Decrements a date by one day."""
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        new_date = date_obj - timedelta(days=1)
        return new_date.strftime("%Y-%m-%d")

    for i in range(5):
        reported_date = increment_date(reported_date)
        if reported_date in adjusted_prices:
            return adjusted_prices[reported_date]['5. adjusted close']

    reported_date = reported_date_copy

    for i in range(5):
        reported_date = decrement_date(reported_date)
        if reported_date in adjusted_prices:
            return adjusted_prices[reported_date]['5. adjusted close']
    raise Exception("No adjusted closing prices for reported date, and backups failed " + reported_date)


# Returns the corresponding reported date
def match_fiscal_date_ending(date, earnings_dates):
    for earning_date in earnings_dates:
        if within_x_days(date, earning_date['fiscalDateEnding']):
            return earning_date['reportedDate']

    raise RuntimeError("fiscal end date match not found")


def get_close_price(date, ticker):
    earnings_dates = get_earnings(ticker)
    # print(earnings_dates)
    reported_date = match_fiscal_date_ending(date, earnings_dates)

    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY'),
        'outputsize': 'full',
    }

    response = requests.get(BASE_AV_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        with open('test.json', 'w') as file:
            json.dump(data, file)
        # print(data)

        # reported_date = [earning_date['reportedDate'] for earning_date in earnings_dates if earning_date['fiscalDateEnding'] == date][0]
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
        # with open('earnings.json', 'w') as file:
        #     json.dump(data, file)
        list_dates = []
        for i in range(len(data['quarterlyEarnings'])):
            return_dict = {'fiscalDateEnding': data['quarterlyEarnings'][i]['fiscalDateEnding'],
                           'reportedDate': data['quarterlyEarnings'][i]['reportedDate'],
                           'surprise': data['quarterlyEarnings'][i]['surprise']}
            list_dates.append(return_dict)
        return list_dates
    else:
        return None


def get_all_adjusted_prices(ticker, verbose=True):
    prices = get_adjusted_prices_complete_json(ticker, False)

    result = []

    for date in prices:
        result.append((date, prices[date]["5. adjusted close"]))

    if verbose:
        print(result)

    return result


if __name__ == "__main__":
    # print(get_earnings('2023-12-31', 'AAPL'))
    # print(get_close_price('2023-12-31', 'AAPL'))
    # print(get_earning_dates('NVDA'))
    # print(get_adjusted_prices_complete_json('NVDA', True))
    get_all_adjusted_prices('TSLA')