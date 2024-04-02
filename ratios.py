import json


def calculate_working_capital(data):
    working_capital_list = []
    for i in range(len(data)):
        if data[i]['totalCurrentAssets'] == 'None' or data[i]['totalCurrentLiabilities'] == 'None':
            working_capital_list.append(0)
            continue
        working_capital = int(data[i]['totalCurrentAssets']) / int(data[i]['totalCurrentLiabilities'])
        working_capital_list.append(working_capital)
    return working_capital_list

def calculate_debt_to_equity(data):
    debt_to_equity_list = []
    for i in range(len(data)):
        if data[i]['currentDebt'] == 'None' or data[i]['totalShareholderEquity'] == 'None':
            debt_to_equity_list.append(0)
            continue
        debt_to_equity = int(data[i]['currentDebt']) / int(data[i]['totalShareholderEquity'])
        debt_to_equity_list.append(debt_to_equity)
    return debt_to_equity_list

def calculate_gross_profit_margin(data):
    gross_profit_margin_list = []
    for i in range(len(data)):
        if data[i]['grossProfit'] == 'None' or data[i]['costofGoodsAndServicesSold'] == 'None' or data[i]['totalRevenue'] == 'None':
            gross_profit_margin_list.append(0)
            continue
        gross_profit_margin = (int(data[i]['grossProfit']) - int(data[i]['costofGoodsAndServicesSold'])) / int(data[i]['totalRevenue'])
        gross_profit_margin_list.append(gross_profit_margin)
    return gross_profit_margin_list

def calculate_operating_profit_margin(data):
    operating_profit_margin_list = []
    for i in range(len(data)):
        if data[i]['operatingIncome'] == 'None' or data[i]['totalRevenue'] == 'None':
            operating_profit_margin_list.append(0)
            continue
        operating_profit_margin = int(data[i]['operatingIncome']) / int(data[i]['totalRevenue'])
        operating_profit_margin_list.append(operating_profit_margin)
    return operating_profit_margin_list

def calculate_interest_coverage_ratio(data):
    interest_coverage_ratio_list = []
    for i in range(len(data)):
        if data[i]['ebit'] == 'None' or data[i]['interestExpense'] == 'None' or int(data[i]['interestExpense']) == 0:
            interest_coverage_ratio_list.append(0)
            continue
        interest_coverage_ratio = int(data[i]['ebit']) / int(data[i]['interestExpense'])
        interest_coverage_ratio_list.append(interest_coverage_ratio)
    return interest_coverage_ratio_list

def calculate_operating_cash_flow_ratio(data):
    operating_cash_flow_ratio_list = []
    for i in range(len(data)):
        if data[i]['operatingCashflow'] == 'None' or data[i]['totalRevenue'] == 'None':
            operating_cash_flow_ratio_list.append(0)
            continue
        operating_cash_flow_ratio = int(data[i]['operatingCashflow']) / int(data[i]['totalRevenue'])
        operating_cash_flow_ratio_list.append(operating_cash_flow_ratio)
    return operating_cash_flow_ratio_list

def calculate_cashflow_coverage_ratio(data):
    cashflow_coverage_ratio_list = []
    for i in range(len(data)):
        if data[i]['operatingCashflow'] == 'None' or data[i]['currentDebt'] == 'None':
            cashflow_coverage_ratio_list.append(0)
            continue
        cashflow_coverage_ratio = int(data[i]['operatingCashflow']) / int(data[i]['currentDebt'])
        cashflow_coverage_ratio_list.append(cashflow_coverage_ratio)
    return cashflow_coverage_ratio_list

def get_ratios(ticker):
    ticker = ticker.upper()
    file = f'APIs/CompleteTechJSONs/{ticker}.json'
    with open(file) as f:
        data = json.load(f)[ticker]

    working_capital = calculate_working_capital(data)
    debt_to_equity = calculate_debt_to_equity(data)
    gross_profit_margin = calculate_gross_profit_margin(data)
    operating_profit_margin = calculate_operating_profit_margin(data)
    interest_coverage_ratio = calculate_interest_coverage_ratio(data)
    operating_cash_flow_ratio = calculate_operating_cash_flow_ratio(data)
    cashflow_coverage_ratio = calculate_cashflow_coverage_ratio(data)

    return (working_capital, debt_to_equity, gross_profit_margin, operating_profit_margin, interest_coverage_ratio,
            operating_cash_flow_ratio, cashflow_coverage_ratio)