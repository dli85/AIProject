import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_cash_flow(symbol):
    BASE_AV_URL = 'https://www.alphavantage.co/query'
    return_dict_list = []
    
    params = {
        'function': 'CASH_FLOW',
        'symbol': symbol,
        'apikey': os.getenv('MY_API_KEY'),
    }
    response = requests.get(BASE_AV_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        for i in range(len(data['quarterlyReports'])):
            return_dict = {}
            return_dict['fiscalDateEnding'] = data['quarterlyReports'][i]['fiscalDateEnding']
            return_dict['netIncome'] = data['quarterlyReports'][i]['netIncome']
            return_dict['capitalExpenditures'] = data['quarterlyReports'][i]['capitalExpenditures']
            return_dict['operatingCashflow'] = data['quarterlyReports'][i]['operatingCashflow']
            return_dict_list.append(return_dict)
        return return_dict_list
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None


if __name__ == "__main__":
    data = get_cash_flow('AAPL')
    if data is not None:
        print(data)
