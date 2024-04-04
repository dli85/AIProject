import json
import os


def calculate_working_capital(data):
    if data['totalCurrentAssets'] == 'None' or data['totalCurrentLiabilities'] == 'None':
        return 0
    working_capital = float(data['totalCurrentAssets']) / float(data['totalCurrentLiabilities'])
    return working_capital


def calculate_debt_to_equity(data):
    if data['currentDebt'] == 'None' or data['totalShareholderEquity'] == 'None':
        return 0
    debt_to_equity = int(data['currentDebt']) / int(data['totalShareholderEquity'])
    return debt_to_equity


def calculate_gross_profit_margin(data):
    if data['grossProfit'] == 'None' or data['costofGoodsAndServicesSold'] == 'None' \
            or data['totalRevenue'] == 'None' or data['totalRevenue'] == '0':
        return 0
    gross_profit_margin = (float(data['grossProfit']) - float(data['costofGoodsAndServicesSold'])) / float(data['totalRevenue'])
    return gross_profit_margin


def calculate_operating_profit_margin(data):
    if data['operatingIncome'] == 'None' or data['totalRevenue'] == 'None' or data['totalRevenue'] == '0':
        return 0
    operating_profit_margin = float(data['operatingIncome']) / float(data['totalRevenue'])
    return operating_profit_margin


def calculate_interest_coverage_ratio(data):
    if data['ebit'] == 'None' or data['interestExpense'] == 'None' or data['interestExpense'] == '0':
        return 0
    interest_coverage_ratio = float(data['ebit']) / float(data['interestExpense'])
    return interest_coverage_ratio


def calculate_operating_cash_flow_ratio(data):
    if data['operatingCashflow'] == 'None' or data['totalRevenue'] == 'None' or data['totalRevenue'] == '0':
        return 0
    operating_cash_flow_ratio = float(data['operatingCashflow']) / float(data['totalRevenue'])
    return operating_cash_flow_ratio


def calculate_cashflow_coverage_ratio(data):
    if data['operatingCashflow'] == 'None' or data['currentDebt'] == 'None' or int(data['currentDebt']) == 0:
        return 0
    cashflow_coverage_ratio = int(data['operatingCashflow']) / int(data['currentDebt'])
    return cashflow_coverage_ratio


def get_ratios(ticker):
    ticker = ticker.upper()
    file = f'APIs/CompleteTechJSONs/{ticker}.json'
    with open(file) as f:
        data = json.load(f)[ticker]

    results = {}

    for entry in data:
        date = entry['fiscalDateEnding']
        current = dict()
        current['working_capital'] = calculate_working_capital(entry)
        current['debt_to_equity'] = calculate_debt_to_equity(entry)
        current['gross_profit_margin'] = calculate_gross_profit_margin(entry)
        current['operating_profit_margin'] = calculate_operating_profit_margin(entry)
        current['interest_coverage_ratio'] = calculate_interest_coverage_ratio(entry)
        current['operating_cash_flow_ratio'] = calculate_operating_cash_flow_ratio(entry)
        current['cashflow_coverage_ratio'] = calculate_cashflow_coverage_ratio(entry)
        current['close_price'] = entry['closePrice']
        results[date] = current

    return results


def get_all_ratios(jsons_path='./APIs/CompleteTechJSONs/'):
    data = dict()
    for filename in os.listdir(jsons_path):
        ticker = filename.split('.')[0]
        data[ticker] = get_ratios(ticker)

    return data


if __name__ == '__main__':
    print(get_all_ratios()['AAPL'])
    print(get_all_ratios()['AAPL']['2023-12-31'])
    print(get_all_ratios()['AAPL']['2023-09-30'])

