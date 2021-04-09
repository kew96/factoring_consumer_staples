from pathlib import Path, PurePath

import pandas as pd


DATA_PATH = Path().cwd().parent.parent.joinpath('data')

all_holdings = pd.read_csv(DATA_PATH.joinpath('raw/holdings.csv'))

for portfolio in all_holdings.crsp_portno.unique():
    holdings = all_holdings[all_holdings.crsp_portno == portfolio]
    holdings_pivot = holdings.pivot_table(index='eff_dt', columns='ticker', values='market_val')
    holdings_pivot.to_csv(DATA_PATH.joinpath(f'interim/{portfolio}_holdings_values.csv'), index_label=False)
