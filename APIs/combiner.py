from balance_sheets import get_balance_sheets
from cash_flow import get_cash_flow
from income_statement import get_income_statement
from extract_sentiment import get_range_data


def combiner(ticker):
    balance_sheets = get_balance_sheets(ticker)
    cash_flow = get_cash_flow(ticker)
    income_statement = get_income_statement(ticker)

    rows = []

    print(len(balance_sheets), len(cash_flow), len(income_statement))
    for i in range(min(len(balance_sheets), len(cash_flow), len(income_statement))):
        if not balance_sheets[i]['fiscalDateEnding'] == cash_flow[i]['fiscalDateEnding'] == income_statement[i]['fiscalDateEnding']:
            raise Exception(f'Expected entries for balance_sheets, cash_flow, '
                            f' and income statements to have the same fiscal ending date: \n'
                            f'balance sheet: {balance_sheets[i]["fiscalDateEnding"]}, '
                            f'cash_flow: {cash_flow[i]["fiscalDateEnding"]}, '
                            f'income_statement: f{income_statement[i]["fiscalDateEnding"]}')

        result_dict = {}
        result_dict.update(balance_sheets[i])
        result_dict.update(cash_flow[i])
        result_dict.update(income_statement[i])

    print(balance_sheets[0])
    print(cash_flow[0])
    print(income_statement[0])


if __name__ == '__main__':
    combiner('META')
