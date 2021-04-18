from pathlib import Path

import pandas as pd

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')

prices_returns = pd.read_table(DATA_PATH.joinpath('interim', 'prices_returns.txt'),
                               parse_dates=['datadate'],
                               usecols=['datadate', 'tic', 'conm', 'chng', 'Price_tb', 'mkt_excess'])

prices_returns['temp_date'] = prices_returns.datadate.dt.strftime('%Y-%b')

book_ratios = pd.read_table(DATA_PATH.joinpath('interim', 'book_ratios.txt'),
                            parse_dates=['datadate'],
                            usecols=['datadate', 'tic', 'ptb'])

book_ratios['temp_date'] = book_ratios.datadate.dt.strftime('%Y-%b')
book_ratios = book_ratios.drop('datadate', axis=1)

market_values = pd.read_table(DATA_PATH.joinpath('interim', 'market_values.txt'),
                              parse_dates=['datadate'],
                              usecols=['datadate', 'tic', 'mkvaltq'])

market_values['temp_date'] = market_values.datadate.dt.strftime('%Y-%b')
market_values = market_values.drop('datadate', axis=1)

pr_br = prices_returns.merge(book_ratios, how='inner', on=['temp_date', 'tic'])

pr_br_mv = pr_br.merge(market_values, how='inner', on=['temp_date', 'tic'])

pr_br_mv = pr_br_mv.drop('temp_date', axis=1)

little_data = list()
for ticker in pr_br_mv.tic.unique():
    if len(pr_br_mv[pr_br_mv.tic==ticker]) < 4:
        little_data.append(ticker)

pr_br_mv = pr_br_mv[~pr_br_mv.tic.isin(little_data)]
pr_br_mv = pr_br_mv.dropna()

pr_br_mv.to_csv(DATA_PATH.joinpath('processed', 'three_factor_model.txt'), index=False, sep='\t')

print('Merged data for three-factor model!')
