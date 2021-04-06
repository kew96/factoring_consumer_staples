from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path().cwd().parent.parent.joinpath('data')


market_values = pd.read_csv(DATA_PATH.joinpath('raw', 'market_values.csv'))

# Scaling market values from millions of dollars to dollars
market_values.mkvaltq = market_values.mkvaltq * 1_000_000

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

# Two consecutive missing market values imputation
def impute_two_periods(data):
    # print(data)
    if np.sum(np.isnan(data.iloc[:2])) == 2: # If first two periods missing
        diff = data.iloc[3] - data.iloc[2] # assume trend holds
        return data.iloc[2] - diff, data.iloc[2] - diff * 2
    elif np.sum(np.isnan(data.iloc[2:])) == 2: # If last two periods missing
        diff = data.iloc[1] - data.iloc[0] # assume trend holds
        return data.iloc[1] + diff, data.iloc[1] + diff * 2
    elif np.sum(np.isnan(data.iloc[1:3])): # If middle two periods missing
        diff = data.iloc[3] - data.iloc[0] # assume constant growth/decay
        step = diff / 3
        return data.iloc[0] + step, data.iloc[0]  + step * 2
    # Individual case handled later


for elem in bad_data.items():
    subset = market_values[market_values.tic == elem[0]]
    for ind, value1, value2 in zip(subset.index, subset.mkvaltq[:-1], subset.mkvaltq[1:]):
        if np.sum(np.isnan([value1, value2])) == 2:
            if ind == subset.index[0]:
                data = subset.loc[ind:ind+4, 'mkvaltq']
            elif ind == subset.index[-2]:
                data = subset.loc[ind-2:, 'mkvaltq']
            else:
                data = subset.loc[ind-1:ind+2, 'mkvaltq']
            if np.sum(np.isnan(data)) > 2:
                continue
            else:
                new_vals = impute_two_periods(data)
            subset.loc[ind:ind+1, 'mkvaltq'] = new_vals
    market_values[market_values.tic == elem[0]] = subset

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


new_bad_data = get_bad_data(market_values)
bad_inds = list()

for tick in new_bad_data.keys():
    bad_inds.extend(list(market_values[market_values.tic == tick].index))

market_values = market_values.drop(bad_inds)

with open(DATA_PATH.joinpath('interim', 'dropped_tickers.txt'), 'a') as file:
    for ticker in new_bad_data.keys():
        file.write(f'{ticker}\n')

market_values.to_csv(DATA_PATH.joinpath('interim', 'market_values.csv'), index=False)
