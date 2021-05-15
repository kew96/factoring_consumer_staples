from pathlib import Path

import pandas as pd

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')

# Loads all data

prices_returns = pd.read_table(DATA_PATH.joinpath('interim', 'prices_returns.txt'),
                               usecols=['datadate', 'tic', 'conm', 'chng', 'Price_tb', 'mkt_excess'])

book_ratios = pd.read_table(DATA_PATH.joinpath('interim', 'book_ratios.txt'),
                            usecols=['datadate', 'tic', 'btm'])

market_values = pd.read_table(DATA_PATH.joinpath('interim', 'market_values.txt'),
                              usecols=['datadate', 'tic', 'mkvaltq'])

# Merge all data on the new temp_date and tickers
pr_br = prices_returns.merge(book_ratios, how='inner', on=['datadate', 'tic'])

pr_br_mv = pr_br.merge(market_values, how='inner', on=['datadate', 'tic'])

# Ensures that each asset has 4 data points so that the rows of factors are linearly separable and there is a final
# period to evaluate performance

little_data = list()
for ticker in pr_br_mv.tic.unique():
    if len(pr_br_mv[pr_br_mv.tic==ticker]) < 5:
        little_data.append(ticker)

pr_br_mv = pr_br_mv[~pr_br_mv.tic.isin(little_data)]

pr_br_mv.to_csv(DATA_PATH.joinpath('processed', 'three_factor_model.txt'), index=False, sep='\t')

print('Merged data for three-factor model!')
