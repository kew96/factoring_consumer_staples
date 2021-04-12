from pathlib import Path
from datetime import date
from collections import Counter

import pandas as pd
import numpy as np

from calendar import monthrange

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = Path.cwd().parent.parent.joinpath('data')

prices_returns = pd.read_table(DATA_PATH.joinpath('interim', 'prices_returns.txt'),
                               parse_dates=['datadate'],
                               usecols=['datadate', 'tic', 'conm', 'prccq', 'chng', 'chng_cs', 'Price_tb', 'excess'])

book_ratios = pd.read_table(DATA_PATH.joinpath('interim', 'book_ratios.txt'),
                            parse_dates=['datadate'],
                            usecols=['datadate', 'tic', 'book_value', 'bv_per_share', 'ptb'])

market_values = pd.read_table(DATA_PATH.joinpath('interim', 'market_values.txt'),
                              parse_dates=['datadate'],
                              usecols=['datadate', 'tic', 'cshoc', 'prccd', 'prcod', 'mkvaltq'])

pr_br = prices_returns.merge(book_ratios, how='inner', on=['datadate', 'tic'])

pr_br_mv = pr_br.merge(market_values, how='inner', on=['datadate', 'tic'])

pr_br_mv.to_csv(DATA_PATH.joinpath('processed', 'three_factor_model.txt'), index=False, sep='\t')

print('Merged data for three-factor model!')
