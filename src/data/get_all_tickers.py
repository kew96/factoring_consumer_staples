from pathlib import Path

import pandas as pd

DATA_PATH = Path().cwd().parent.parent.joinpath('data')

ticker_set = set()

for csv in DATA_PATH.joinpath('interim').glob('*_holdings_values.txt'):
    df = pd.read_table(csv)
    for v in list(df.columns):
        ticker_set.add(v)

sp_consumer_staples = pd.read_table(DATA_PATH.joinpath('raw', 'consumer_staples_cusips.txt'))
for ticker in sp_consumer_staples.TICKER:
    ticker_set.add(ticker)

with open(DATA_PATH.joinpath('interim/all_tickers.txt'), 'w') as file:
    file.write('\n'.join(ticker_set))
