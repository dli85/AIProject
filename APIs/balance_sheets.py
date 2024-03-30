from dotenv import load_dotenv
import requests
import os
load_dotenv()

BASE_AV_URL = 'https://www.alphavantage.co/query'

BASE_PARAMS = {
    'function': 'BALANCE_SHEET',
    'symbol': None,
    'apikey': os.getenv('MY_API_KEY'),
    'datatype': 'csv',
}


def get_balance_sheets(ticker):
    params = BASE_PARAMS.copy()
    params['symbol'] = ticker
    response = requests.get(BASE_AV_URL, params)
    data = response.json()

    quarterly_reports = data['quarterlyReports']

    def filter_dict(dictionary, valid_keys):
        return {key: dictionary[key] for key in dictionary if key in valid_keys}

    kept_keys = ['fiscalDateEnding', 'totalCurrentAssets', 'totalCurrentLiabilities', 'cashAndShortTermInvestments', 'currentDebt']

    result = []

    for d in quarterly_reports:
        result.append(filter_dict(d, kept_keys))

    # current assets, current liabilities, short term investments, current debt
    return result

