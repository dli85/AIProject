import json
import requests

ticker = 'KVYO'
key = 'YX89AQX3V3CNVRSK'

url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={key}'

r = requests.get(url)

if r.status_code == 200:
    data_flights = r.json()
    filename = f'income_statement_{ticker}.json'

    # Write the JSON data to a file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_flights, f, ensure_ascii=False, indent=4)
else:
    print(f"Error: {r.status_code}")
