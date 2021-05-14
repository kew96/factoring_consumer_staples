from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
consumer_staples = sp500[sp500['GICS Sector'] == 'Consumer Staples']

with open(DATA_PATH.joinpath('raw', 'consumer_staples.txt'), 'w') as file:
    file.writelines(ticker.replace('.', '')+'\n' for ticker in consumer_staples.Symbol)
