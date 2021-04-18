from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')


market_values = pd.read_table(DATA_PATH.joinpath('raw', 'prices_shares_outstanding_daily.txt'),
                              dtype={'cusip': str, 'tpci': str}, parse_dates=['datadate'])

# Calculating daily market value
market_values['mkvaltq'] = market_values.prccd * market_values.cshoc

# Finding all tickers with missing values
def get_bad_data(values):
    tickers = list()
    for ticker in values.tic.unique():
        subset = values[values.tic == ticker]
        current = 'na' if np.isnan(subset.mkvaltq.iloc[0]) else 'val'
        for v in subset.mkvaltq[1:]:
            if not np.isnan(v):
                current = 'val'
            elif current == 'val' and np.isnan(v):
                tickers.append(ticker)
                continue
    return Counter(tickers)

bad_data = get_bad_data(market_values)

# One period missing market values imputation
def impute_one_period(data):
    if np.isnan(data.iloc[0]): # If the first period is missing
        diff = data.iloc[2] - data.iloc[1] # assume constant growth
        return data.iloc[1] - diff
    elif np.isnan(data.iloc[1]): # If the middle value is missing
        return (data.iloc[0] + data.iloc[2]) / 2 # take average of previous and next period
    else: # If the last period is missing
        diff = data.iloc[1] - data.iloc[0] # assume constant growth
        return data.iloc[1] + diff


for elem in bad_data.items():
    subset = market_values[market_values.tic == elem[0]]
    for ind, value in zip(subset.index, subset.mkvaltq):
        if np.isnan(value):
            if ind == subset.index[0]:
                three_set = subset.loc[ind:ind+3, 'mkvaltq']
            elif ind == subset.index[-1]:
                three_set = subset.loc[ind-2:ind, 'mkvaltq']
            else:
                three_set = subset.loc[ind-1:ind+1, 'mkvaltq']
            if np.sum(np.isnan(three_set)) == 1:
                new_value = impute_one_period(three_set)
                subset.loc[ind, 'mkvaltq'] = new_value
    market_values[market_values.tic == elem[0]] = subset

monthly_vals = market_values[market_values.datadate.dt.is_month_end]
quarterly_vals = monthly_vals[monthly_vals.datadate.dt.month.isin([3, 6, 9, 12])]

little_data = list()
for ticker in quarterly_vals.tic.unique():
    if len(quarterly_vals[quarterly_vals.tic==ticker]) < 2:
        little_data.append(ticker)

quarterly_vals = quarterly_vals[~quarterly_vals.tic.isin(little_data)]

quarterly_vals.to_csv(DATA_PATH.joinpath('interim', 'market_values.txt'), index=False, sep='\t')

print('Cleaned market values!')
