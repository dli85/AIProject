import pandas as pd
import json

tickers = ['AAPL', 'IBM', 'TSLA', 'KVYO']

file_paths = [f'income_statement_{ticker}.json' for ticker in tickers]

all_data = []

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        for report in data['annualReports']:
            report['symbol'] = data['symbol']
        all_data.extend(data['annualReports'])

df = pd.DataFrame(all_data)

print(df['symbol'].value_counts())
