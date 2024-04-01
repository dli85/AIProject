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


def get_quarterly_balance_sheets(ticker):
    params = BASE_PARAMS.copy()
    params['symbol'] = ticker
    response = requests.get(BASE_AV_URL, params)
    data = response.json()

    return data['quarterlyReports']


def get_balance_sheets(ticker):
    quarterly_reports = get_quarterly_balance_sheets(ticker)

    def filter_dict(dictionary, valid_keys):
        return {key: dictionary[key] for key in dictionary if key in valid_keys}

    kept_keys = ['fiscalDateEnding', 'totalCurrentAssets', 'totalCurrentLiabilities', 'cashAndShortTermInvestments',
                 'currentDebt']

    result = []

    for d in quarterly_reports:
        result.append(filter_dict(d, kept_keys))

    # current assets, current liabilities, short term investments, current debt
    return result


# Current ratio = current assets/current liabilities
# Debt/equity = (Total liabilities) / Shareholder's Equity
def get_balance_sheet_ratios(ticker):
    quarterly_reports = get_quarterly_balance_sheets(ticker)

    result = []
    for report in quarterly_reports:
        d = {'fiscalDateEnding': report['fiscalDateEnding'],
             'currentRatio': int(report['totalCurrentAssets']) / int(report['totalCurrentLiabilities']),
             'debtAssetRatio': int(report['longTermDebt']) / int(report['totalAssets'])}
        result.append(d)

    print(quarterly_reports[0])

    return result


if __name__ == '__main__':
    get_balance_sheet_ratios('META')
