import requests
import logging
import sys
import os

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)  # Set the level for this handler
logger.addHandler(stdout_handler)

BASE_AV_URL = 'https://www.alphavantage.co/query'

BASE_PARAMS = {
    'function': 'NEWS_SENTIMENT',
    'tickers': None,
    'apikey': os.getenv('MY_API_KEY'),
    'sort': "EARLIEST",
    'limit': 1000
}


def _check_range_format(range_from, range_to):
    # time format is 
    # YYYYMMDDTHHMM
    if len(range_from) != 12 or len(range_to) != 12:
        return False
    if "T" not in range_from or "T" not in range_to:
        return False
    return True


def get_sentiment_data(ticker, range_from=None, range_to=None):
    # time format is 
    # YYYYMMDDTHHMM
    # if not _check_range_format(range_from, range_to):
    #     raise ValueError("Time format is incorrect")
    
    # logger.info(f"Getting data for {ticker}")
    params = BASE_PARAMS.copy()
    params['tickers'] = ticker

    if range_from:
        params['time_from'] = range_from
        if range_to:
            params['time_to'] = range_to

    response = requests.get(BASE_AV_URL, params=params)
    data = response.json()
    # if verbose:
    #     with open("sentiment_data.json", "w") as f:
    #         f.write(response.text)

    # pprint(data)
    try:
        temp = len(data['feed'])
        # logger.info(f"Got data for {ticker}. {len(data['feed'])} articles found.")
        return data
    except KeyError:
        return {'feed': []}


def parse_scores_and_relevance(data, ticker):
    """
    Returns a list of 2-tuples, each tuple contains the sentiment score and relevance score of an article.
    """
    articles = data['feed']
    ret = [] # list of tuples
    for article in articles:
        sentiments = article["ticker_sentiment"]
        # Take the first item from the filter iterator, uses next for more efficient code.
        target_sentiment_dict = next(filter(lambda x: x["ticker"] == ticker, sentiments))
        article_sentiment_score = target_sentiment_dict["ticker_sentiment_score"]
        article_relevance_score = target_sentiment_dict["relevance_score"]
        # convert to floats 
        article_sentiment_score = float(article_sentiment_score)
        article_relevance_score = float(article_relevance_score)
        # print(article_sentiment_score, article_relevance_score)
        ret.append((article_sentiment_score, article_relevance_score))
    
    return ret


def get_range_data(ticker, range_from=None, range_to=None):
    """
    range_from and range_to are in the format of "YYYY-MM-DD"
    """
    # convert to format of "YYYYMMDDTHHMM"
    range_from_query = range_from.replace("-", "") + "T0000" if range_from else None
    range_to_query = range_to.replace("-", "") + "T0000" if range_to else None
    rd = get_sentiment_data(ticker, range_from_query, range_to_query)
    scores_and_relevance = parse_scores_and_relevance(rd, ticker)

    # weighted average of scores with relevance
    total_score = 0
    total_relevance = 0
    for score, relevance in scores_and_relevance:
        total_score += score * relevance
        total_relevance += relevance
    if total_relevance == 0:
        return 0
    
    return total_score / total_relevance


if __name__ == "__main__":
    # get_sentiment_data(ticker, "20210901T0000", "20210930T0000")
    # rd = get_sentiment_data(ticker)
    # parse_scores_and_relevance(rd)
    # res = get_range_data(ticker, "2022-09-01")
    # pprint(res)
    # res = get_range_data('META', "2021-09-30", "2021-12-31")
    res = get_range_data('NVDA', "2022-12-31", "2023-03-31")
    print(res)
