import yfinance as yf


class ThreeFactors:

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.ret = None
        self.book_to_market = None
        self.market_cap = None
        self.__set_metrics()

    def __set_metrics(self):

        stock = self.__get_historical_data()
        sp500 = self.__get_historical_data('^GSPC')
        # TODO: Retrieve historical fundamentals
        # TODO: Set class attributes
        # TODO: What kind of average do we want to use?

    def __get_historical_data(self, ticker=None, start='2000-01-01', end='2020-12-31', interval='3mo'):
        if not ticker:
            ticker = self.ticker

        return yf.download(ticker, start=start, end=end, interval=interval)
