from pathlib import Path

import pandas as pd

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = Path.cwd().parent.parent.joinpath('data')

prices_returns = pd.read_table(DATA_PATH.joinpath('interim', 'prices_returns.txt'),
                               parse_dates=['datadate'],
                               usecols=['datadate', 'tic', 'conm', 'chng', 'Price_tb', 'mkt_excess'])

book_ratios = pd.read_table(DATA_PATH.joinpath('interim', 'book_ratios.txt'),
                            parse_dates=['datadate'],
                            usecols=['datadate', 'tic', 'ptb'])

market_values = pd.read_table(DATA_PATH.joinpath('interim', 'market_values.txt'),
                              parse_dates=['datadate'],
                              usecols=['datadate', 'tic', 'mkvaltq'])

pr_br = prices_returns.merge(book_ratios, how='inner', on=['datadate', 'tic'])

pr_br_mv = pr_br.merge(market_values, how='inner', on=['datadate', 'tic'])

little_data = list()
for ticker in pr_br_mv.tic.unique():
    if len(pr_br_mv[pr_br_mv.tic==ticker]) < 4:
        little_data.append(ticker)

pr_br_mv = pr_br_mv[~pr_br_mv.tic.isin(little_data)]

pr_br_mv.to_csv(DATA_PATH.joinpath('processed', 'three_factor_model.txt'), index=False, sep='\t')

print('Merged data for three-factor model!')
