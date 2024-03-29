import requests
import os
from dotenv import load_dotenv

BASE_AV_URL = 'https://www.alphavantage.co/query'

load_dotenv()

def get_income_statement(ticker):

    return_dict_list = []

    params = {
        'function': 'INCOME_STATEMENT',
        'symbol': ticker,
        'apikey': os.getenv('MY_API_KEY')
    }

    response = requests.get(BASE_AV_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        for i in range(len(data['quarterlyReports'])):
            return_dict = {}
            return_dict['fiscalDateEnding'] = data['quarterlyReports'][i]['fiscalDateEnding']
            return_dict['grossProfit'] = data['quarterlyReports'][i]['grossProfit']
            return_dict['totalRevenue'] = data['quarterlyReports'][i]['totalRevenue']
            return_dict_list.append(return_dict)
        return return_dict_list


if __name__ == "__main__":
    print(get_income_statement('AAPL'))