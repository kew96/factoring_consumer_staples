from pathlib import Path
from datetime import date
from collections import Counter

import pandas as pd
import numpy as np

from calendar import monthrange

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = Path.cwd().parent.parent.joinpath('data')

consumer_staples = pd.read_csv(DATA_PATH.joinpath('raw', 'sp500_consumer_staples.csv'))
t_bill = pd.read_csv(DATA_PATH.joinpath('raw', '3_month_t_bill.csv'))
prices = pd.read_table(DATA_PATH.joinpath('raw', 'prices_assets_liabilities_quarterly.txt'),
                       parse_dates=['datadate'], usecols=['datadate', 'tic', 'conm', 'prccq'],
                       index_col=['datadate', 'tic'])

t_bill.Date = pd.to_datetime(t_bill.Date, format='%b %y')
consumer_staples.Date = pd.to_datetime(consumer_staples.Date, format='%b %y')

def convert_to_last_day(dt):
    year = dt.year
    month = dt.month
    last_day = monthrange(year, month)[1]
    return date(year, month, last_day)

t_bill.Date = t_bill.Date.apply(convert_to_last_day)
consumer_staples.Date = consumer_staples.Date.apply(convert_to_last_day)

t_bill = t_bill.set_index('Date').sort_index()
consumer_staples = consumer_staples.set_index('Date').sort_index()

def get_bad_data(prices):
    bad_data = list()
    for ticker in prices.index.get_level_values(1).unique():
        subset = prices.loc[pd.IndexSlice[:, ticker], :]
        current = 'na' if np.isnan(subset.prccq.iloc[0]) else 'val'
        for v in subset.prccq[1:]:
            if not np.isnan(v):
                current = 'val'
            elif current == 'val' and np.isnan(v):
                bad_data.append((ticker, subset.conm.iloc[0]))
                continue
    return Counter(bad_data)

bad_data = get_bad_data(prices)

def impute_two_periods(data):
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
    subset = prices.loc[pd.IndexSlice[:, elem[0][0]], :]
    for ind, (value1, value2) in enumerate(zip(subset.prccq[:-1], subset.prccq[1:])):
        if np.sum(np.isnan([value1, value2])) == 2:
            if ind == 0:
                data = subset.iloc[ind:ind+4].prccq
            elif ind == len(subset)-3:
                data = subset.iloc[ind-4:].prccq
            else:
                data = subset.iloc[ind-1:ind+3].prccq
            if np.sum(np.isnan(data)) > 2:
                continue
            else:
                new_vals = impute_two_periods(data)
            subset.iloc[ind:ind+2].prccq = new_vals
    prices.loc[pd.IndexSlice[:, elem[0]], :] = subset

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
    subset = prices.loc[pd.IndexSlice[:, elem[0][0]], :]
    for ind, value in enumerate(subset.prccq):
        if np.isnan(value):
            if ind == 0:
                three_set = subset.iloc[ind:ind+3].prccq
            elif ind == len(subset)-1:
                three_set = subset.iloc[ind-2:ind].prccq
            else:
                three_set = subset.iloc[ind-1:ind+2].prccq
            if np.sum(np.isnan(three_set)) == 1:
                new_value = impute_one_period(three_set)
                subset.iloc[ind] = subset.iloc[ind].fillna(new_value)
    prices.loc[pd.IndexSlice[:, elem[0][0]], :] = subset

prices['chng'] = np.nan

for ticker in prices.index.get_level_values(1).unique():
    subset = prices.loc[pd.IndexSlice[:, ticker], :]
    subset.chng = subset.prccq.pct_change()
    prices.loc[pd.IndexSlice[:, ticker], 'chng'] = subset.chng

consumer_staples['chng'] = consumer_staples.Price.pct_change()

consumer_staples['excess'] = consumer_staples.chng - t_bill.Price/100

prices = prices.reset_index()

consumer_staples = consumer_staples.reset_index()
consumer_staples.Date = pd.to_datetime(consumer_staples.Date)

prices_w_cs = prices.merge(consumer_staples.reset_index(), how='left',
                           left_on='datadate', right_on='Date', suffixes=('', '_cs'))

t_bill = t_bill.reset_index()
t_bill.Date = pd.to_datetime(t_bill.Date)

prices_w_cs_w_tb = prices_w_cs.merge(t_bill, how='left', left_on='datadate', right_on='Date', suffixes=('', '_tb'))

prices_w_cs_w_tb.to_csv(DATA_PATH.joinpath('interim', 'prices_returns.txt'), index=False, sep='\t')

print('Cleaned prices and returns!')
