import time
# from alpha_vantage.timeseries import TimeSeries

# from pprint import pprint
# ts = TimeSeries(key='Q9O1WK5BTAERX17D', output_format='pandas')
# data, meta_data = ts.get_monthly(symbol='QQQ')
# pprint(data.head(2))
# # write data to a csv
# data.to_csv('QQQ.csv')

import requests
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=QQQ&interval=5min&apikey=Q9O1WK5BTAERX17D&outputsize=full&datatype=csv&month=2024-02'
response = requests.get(url)
print(response.text)
with open(f'QQQ-{time.time()}.csv', 'w') as f:
    f.write(response.text)
