from pathlib import Path

import pandas as pd

DATA_PATH = Path().cwd().parent.parent.joinpath('data')

tickers_set = set()

for csv in DATA_PATH.joinpath('interim').glob('*.csv'):
    df = pd.read_csv(csv)
    for v in list(df.columns):
        if not v.isdigit():
            tickers_set.add(v)

with open(DATA_PATH.joinpath('raw/consumer_staples.txt'), 'r') as file:
    consumer_tickers = file.read().split('\n')

tickers_set.update(consumer_tickers[:-1])

with open(DATA_PATH.joinpath('interim/all_tickers.txt'), 'w') as file:
    file.write(' '.join(tickers_set))