import requests
import os
from dotenv import load_dotenv

BASE_AV_URL = 'https://www.alphavantage.co/query'

load_dotenv()


def get_quarterly_reports(ticker):
    params = {
        'function': 'INCOME_STATEMENT',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY')
    }

    response = requests.get(BASE_AV_URL, params=params)
    data = response.json()

    return data['quarterlyReports']


def get_income_statement(ticker):

    return_dict_list = []

    quarterly_reports = get_quarterly_reports(ticker)

    for i in range(len(quarterly_reports)):
        return_dict = {}
        return_dict['fiscalDateEnding'] = quarterly_reports[i]['fiscalDateEnding']
        return_dict['grossProfit'] = quarterly_reports[i]['grossProfit']
        return_dict['totalRevenue'] = quarterly_reports[i]['totalRevenue']
        return_dict_list.append(return_dict)
    return return_dict_list


# gross profit margin = gross profit / total revenue
# operating profit margin = operating income / total revenue
# net profit margin = net income / total revenue
# ebita margin = ebita / total revenue
def get_income_statement_ratios(ticker):
    result = []

    quarterly_reports = get_quarterly_reports(ticker)

    print(quarterly_reports[0])

    for report in quarterly_reports:
        d = {
            'fiscalDateEnding': report['fiscalDateEnding'],
            'grossProfitMargin': int(report['grossProfit']) / int(report['totalRevenue']),
            'operatingProfitMargin': int(report['operatingIncome']) / int(report['totalRevenue']),
            'netProfitMargin': int(report['netIncome']) / int(report['totalRevenue']),
            'ebitdaMargin': int(report['ebitda']) / int(report['totalRevenue']),
        }
        result.append(d)

    print(result[0])

    return result


if __name__ == "__main__":
    # print(get_income_statement('AAPL')[0])
    print(get_income_statement_ratios('AAPL')[0])