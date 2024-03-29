import requests
import os
from dotenv import load_dotenv

load_dotenv()

def get_cash_flow(symbol):
    BASE_AV_URL = 'https://www.alphavantage.co/query'
    params = {
        'function': 'CASH_FLOW',
        'symbol': symbol,
        'apikey': os.getenv('MY_API_KEY'),
    }
    response = requests.get(BASE_AV_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data['quarterlyReports']
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None

if __name__ == "__main__":
    data = get_cash_flow('AAPL')
    if data is not None:
        print(data)
