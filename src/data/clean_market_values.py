from pathlib import Path
from collections import Counter
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')


market_values = pd.read_table(DATA_PATH.joinpath('raw', 'prices_shares_outstanding_daily.txt'),
                              parse_dates=['date'], usecols=['TICKER', 'date', 'COMNAM', 'PRC', 'SHROUT'])

market_values.columns = ['datadate', 'tic', 'conm', 'prccd', 'cshoc']

market_values = market_values.dropna(subset=['tic'])

nan_shares = list()
for entry in market_values.cshoc:
    try:
        nan_shares.append(float(entry) * 1_000)
    except ValueError:
        nan_shares.append(np.nan)

market_values.cshoc = nan_shares

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

# One period missing market values imputation, assumes linearity
def impute_one_period(data):
    if np.isnan(data.iloc[0]): # If the first period is missing
        diff = data.iloc[2] - data.iloc[1] # assume constant growth
        return data.iloc[1] - diff
    elif np.isnan(data.iloc[1]): # If the middle value is missing
        return (data.iloc[0] + data.iloc[2]) / 2 # take average of previous and next period
    else: # If the last period is missing
        diff = data.iloc[1] - data.iloc[0] # assume constant growth
        return data.iloc[1] + diff

market_values = market_values.sort_values(['tic', 'datadate']).reset_index(drop=True)

for elem in bad_data.items():
    subset = market_values[market_values.tic == elem[0]]
    for ind, value in zip(subset.index, subset.mkvaltq):
        if np.isnan(value):
            if ind == subset.index[0]:
                three_set = subset.loc[ind:ind+3, 'mkvaltq']
            elif ind == subset.index[-1]:
                three_set = subset.loc[ind-2:ind, 'mkvaltq']
            else:
                three_set = subset.loc[ind-2:ind+1, 'mkvaltq']
            if np.sum(np.isnan(three_set)) == 1:
                try:
                    new_value = impute_one_period(three_set)
                    subset.loc[ind, 'mkvaltq'] = new_value
                except:
                    print(subset.info())
                    raise Exception
    market_values[market_values.tic == elem[0]] = subset

# Reduces data to only months that are quarter end and that have a date after the 25th

quarter_market_vals = market_values[market_values.datadate.dt.month.isin([3, 6, 9, 12])]
quarterly_vals = quarter_market_vals[quarter_market_vals.datadate.dt.day > 25]

# Iterates through all dates to find the latest date available since quarters may not end on an available day

good_dates = list()
for year in range(2000, 2021):
    for month in [3, 6, 9, 12]:
        last_day = pd.Timestamp(year, month, 1) + relativedelta(months=1, days=-1)
        while not last_day in quarterly_vals.datadate.values:
            last_day = last_day - relativedelta(days=1)
        good_dates.append(last_day)

quarterly_vals = quarterly_vals[quarterly_vals.datadate.isin(good_dates)]

# Ensures each asset has at least three data points

little_data = list()
for ticker in quarterly_vals.tic.unique():
    if len(quarterly_vals[quarterly_vals.tic==ticker]) < 2:
        little_data.append(ticker)

quarterly_vals = quarterly_vals[~quarterly_vals.tic.isin(little_data)]

quarterly_vals.to_csv(DATA_PATH.joinpath('interim', 'market_values.txt'), index=False, sep='\t')

print('Cleaned market values!')
