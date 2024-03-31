import json

from balance_sheets import get_balance_sheets
from cash_flow import get_cash_flow
from income_statement import get_income_statement
from extract_sentiment import get_range_data
from close_price import *
from tqdm import tqdm
from constants import tech_ticker_list

JSON_PATH = './JSONs'


def combiner(ticker):
    balance_sheets = get_balance_sheets(ticker)
    cash_flow = get_cash_flow(ticker)
    income_statement = get_income_statement(ticker)

    rows = []

    adjusted_prices_json = get_adjusted_prices_complete_json(ticker)
    earnings_reported_date_mapping = get_earnings(ticker)

    for i in tqdm(range(min(len(balance_sheets), len(cash_flow), len(income_statement)))):
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

        fiscal_end_date = balance_sheets[i]['fiscalDateEnding']

        # decrease the year by 1 in a string date (yyyy-mm-dd)
        def decrement_year(d: str):
            nums = d.split('-')
            nums[0] = str(int(nums[0]) - 1)
            return f"{nums[0]}-{nums[1]}-{nums[2]}"

        if i == min(len(balance_sheets), len(cash_flow), len(income_statement)) - 1:
            year_from = decrement_year(balance_sheets[i - 3]['fiscalDateEnding'])
        else:
            year_from = balance_sheets[i + 1]['fiscalDateEnding']

        # print(year_from)
        # print(fiscal_end_date)
        sentiment = get_range_data(ticker, year_from, fiscal_end_date)
        result_dict['sentiment'] = sentiment

        surprise = get_surprise(fiscal_end_date, earnings_reported_date_mapping)
        result_dict['surprise'] = surprise

        close_price = get_adjusted_price_at_date(fiscal_end_date, earnings_reported_date_mapping, adjusted_prices_json)
        result_dict['closePrice'] = close_price

        rows.append(result_dict)
        # print(result_dict)

    save_rows(ticker, rows)


def save_rows(ticker, rows):
    data = {ticker: rows}
    with open(f"{JSON_PATH}/{ticker}.json", "w") as file:
        json.dump(data, file, indent=4)


if __name__ == '__main__':
    for i in range(len(tech_ticker_list)):
        ticker = tech_ticker_list[i]
        if os.path.exists(f"{JSON_PATH}/{ticker}.json"):
            continue

        print("working on", ticker)
        combiner(ticker)
        print(f"Completed {i + 1}/{len(tech_ticker_list)}")
    # combiner('NVDA')
