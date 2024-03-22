from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
ts = TimeSeries(key='Q9O1WK5BTAERX17D', output_format='pandas')
data, meta_data = ts.get_monthly(symbol='QQQ')
pprint(data.head(2))
# write data to a csv
data.to_csv('QQQ.csv')
