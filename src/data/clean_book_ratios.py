from pathlib import Path
from collections import Counter
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')

# Loading all data

book_ratios = pd.read_table(DATA_PATH.joinpath('raw', 'prices_assets_liabilities_quarterly.txt'),
                            parse_dates=['datadate'], usecols=['datadate', 'tic', 'conm', 'atq', 'ltq'])
book_ratios['book_value'] = (book_ratios.atq - book_ratios.ltq) * 1_000_000
book_ratios.datadate = book_ratios.datadate.dt.strftime('%Y-%m')

shares = pd.read_table(DATA_PATH.joinpath('raw', 'prices_shares_outstanding_monthly.txt'),
                       usecols=['date', 'TICKER', 'SHROUT', 'PRC'], parse_dates=['date'])
shares.columns = ['datadate', 'tic', 'prccq', 'cshoc']
shares.datadate = shares.datadate.dt.strftime('%Y-%m')

book_ratios = book_ratios.merge(shares, on=['datadate', 'tic'], how='left')
book_ratios = book_ratios.dropna(subset=['tic'])

book_ratios.cshoc = book_ratios.cshoc * 1_000

book_ratios = book_ratios.sort_values(['tic', 'datadate']).reset_index(drop=True)

for ticker in book_ratios.tic.unique():
    subset = book_ratios[book_ratios.tic == ticker]
    subset.prccm = subset.prccq.interpolate(method='slinear', limit=2, limit_area='outside')
    subset.cshoc = subset.cshoc.interpolate(method='pad', limit=2, limit_area='outside')
    # Linearly interpolate period end prices
    end_btm = subset.prccq[-4:]
    if end_btm.isna().sum() == 2:
        # If there are two missing values and they are the last two values
        if np.isnan(end_btm.iloc[2]) and np.isnan(end_btm.iloc[3]):
            step = end_btm.iloc[1] - end_btm.iloc[0]
            end_btm.iloc[2] = end_btm.iloc[1] + step
            end_btm.iloc[3] = end_btm.iloc[1] + 2 * step
    elif end_btm.isna().sum() == 1:
        # If there is one missing value and it is the last value
        if np.isnan(end_btm.iloc[-1]):
            step = end_btm.iloc[-2] - end_btm.iloc[-3]
            end_btm.iloc[-1] = end_btm.iloc[-2] + step
    book_ratios[book_ratios.tic == ticker] = subset

# Calculating monthly market value
book_ratios['mkvaltq'] = book_ratios.prccq * book_ratios.cshoc

# Book to market ratio = book value / market value
book_ratios['btm'] = book_ratios.book_value / book_ratios.mkvaltq

# Ensures that each asset has at least three data points

little_data = list()
for ticker in book_ratios.tic.unique():
    if len(book_ratios[book_ratios.tic==ticker]) < 2:
        little_data.append(ticker)

book_ratios = book_ratios[~book_ratios.tic.isin(little_data)]
book_ratios = book_ratios.dropna(subset=['btm'])

book_ratios.to_csv(DATA_PATH.joinpath('interim', 'book_ratios.txt'), index=False, sep='\t')

print('Cleaned book ratios!')
