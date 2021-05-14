"""
Combines all tickers from the consumer staples portfolios and tickers that are currently in the S&P500
"""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')

# Use set so that there are no duplicates
ticker_set = set()

# Add all tickers from portfolios
for txt_file in DATA_PATH.joinpath('interim').glob('*_holdings_values.txt'):
    df = pd.read_table(txt_file)
    ticker_set.update(df.columns)

# Add all tickers currently in consumer staples
sp_consumer_staples = pd.read_table(DATA_PATH.joinpath('raw', 'consumer_staples_cusips.txt'))
ticker_set.update(sp_consumer_staples.TICKER)

with open(DATA_PATH.joinpath('interim', 'all_tickers.txt'), 'w') as file:
    file.write('\n'.join(ticker_set))
