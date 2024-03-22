import pandas as pd
import json

file_paths = ['income_statement_AAPL.json', 'income_statement_IBM.json', 'income_statement_TSLA.json',
              'income_statement_KVYO.json']

all_data = []

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        for report in data['annualReports']:
            report['symbol'] = data['symbol']
        all_data.extend(data['annualReports'])

df = pd.DataFrame(all_data)

print(df['symbol'].value_counts())
