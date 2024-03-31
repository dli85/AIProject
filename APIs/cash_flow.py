import requests
import os
from dotenv import load_dotenv

load_dotenv()


def get_quarterly_reports(ticker):
    BASE_AV_URL = 'https://www.alphavantage.co/query'

    params = {
        'function': 'CASH_FLOW',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY'),
    }
    response = requests.get(BASE_AV_URL, params=params)
    data = response.json()

    return data['quarterlyReports']


def get_cash_flow(ticker):
    return_dict_list = []
    
    quarterly_reports = get_quarterly_reports(ticker)

    for i in range(len(quarterly_reports)):
        return_dict = {'fiscalDateEnding': quarterly_reports[i]['fiscalDateEnding'],
                       'netIncome': quarterly_reports[i]['netIncome'],
                       'capitalExpenditures': quarterly_reports[i]['capitalExpenditures'],
                       'operatingCashflow': quarterly_reports[i]['operatingCashflow']}
        return_dict_list.append(return_dict)
    return return_dict_list


# TODO NEED TO FINISH
def get_cash_flow_ratios(ticker, ):
    quarterly_reports = get_quarterly_reports(ticker)

    print(quarterly_reports[0])


if __name__ == "__main__":
    print(get_cash_flow_ratios('AAPL'))
