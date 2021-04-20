from pathlib import Path
from datetime import date
from collections import Counter
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

from calendar import monthrange

import warnings
warnings.filterwarnings("ignore")


DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')

consumer_staples = pd.read_csv(DATA_PATH.joinpath('raw', 'sp500_consumer_staples.csv'))
t_bill = pd.read_csv(DATA_PATH.joinpath('raw', '3_month_t_bill.csv'))
prices = pd.read_table(DATA_PATH.joinpath('raw', 'prices_shares_outstanding_daily.txt'),
                       parse_dates=['date'], usecols=['TICKER', 'date', 'COMNAM', 'PRC', 'SHROUT', 'RET'])

prices.columns = ['datadate', 'tic', 'conm', 'prccd', 'chng', 'cshoc']

nan_prices = list()
for entry in prices.cshoc:
    try:
        nan_prices.append(float(entry)*1_000)
    except ValueError:
        nan_prices.append(np.nan)

prices.cshoc = nan_prices

nan_chng = list()
for entry in prices.chng:
    try:
        nan_chng.append(float(entry))
    except ValueError:
        nan_chng.append(np.nan)

prices.chng = nan_chng

prices = prices.set_index(['tic', 'datadate'])

num_tickers = len(prices.index.get_level_values('tic').unique())

# Finds the latest date available for each quarter, out of all assets

good_dates = list()
for year in range(2000, 2021):
    for month in range(1, 13):
        last_day = pd.Timestamp(year, month, 1) + relativedelta(months=1, days=-1)
        while not last_day in prices.index.get_level_values('datadate').values:
            last_day = last_day - relativedelta(days=1)
        good_dates.append(last_day)
prices = prices.loc[pd.IndexSlice[:, good_dates], ['conm', 'prccd', 'chng']]

prices = prices.rename({'prccd': 'prccm'}, axis=1)

# Converts month and year combos to datetimes

t_bill.Date = pd.to_datetime(t_bill.Date, format='%b %y')
consumer_staples.Date = pd.to_datetime(consumer_staples.Date, format='%b %y')

# Converts datetimes to the last day of the month

def convert_to_last_day(dt):
    year = dt.year
    month = dt.month
    last_day = monthrange(year, month)[1]
    return date(year, month, last_day)

t_bill.Date = t_bill.Date.apply(convert_to_last_day)
consumer_staples.Date = consumer_staples.Date.apply(convert_to_last_day)

# Sets date as the index

t_bill = t_bill.set_index('Date').sort_index()
consumer_staples = consumer_staples.set_index('Date').sort_index()

# Retrieves missing number of missing data points (outside of beginning and end) and prints the count, ticker,
# and company name

def get_bad_data(prices):
    bad_data = list()
    for ticker in prices.index.get_level_values('tic').unique():
        subset = prices.loc[pd.IndexSlice[ticker, :], :]
        current = 'na' if np.isnan(subset.chng.iloc[0]) else 'val'
        for v in subset.chng.iloc[1:]:
            if not np.isnan(v):
                current = 'val'
            elif current == 'val' and np.isnan(v):
                bad_data.append((ticker, subset.conm.iloc[0]))
                continue
    return Counter(bad_data)

bad_data = get_bad_data(prices)

# Imputes two periods of missing data, assuming linearity

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
    subset = prices.loc[pd.IndexSlice[elem[0][0], :], :]
    for ind, (value1, value2) in enumerate(zip(subset.chng[:-1], subset.chng[1:])):
        if np.sum(np.isnan([value1, value2])) == 2:
            if ind == 0:
                data = subset.iloc[ind:ind+4].chng
            elif ind == len(subset)-2:
                data = subset.iloc[ind-4:].chng
            else:
                data = subset.iloc[ind-1:ind+3].chng
            if np.sum(np.isnan(data)) > 2:
                continue
            else:
                new_vals = impute_two_periods(data)
            subset.iloc[ind:ind+2].chng = new_vals
    prices.loc[pd.IndexSlice[elem[0], :], :] = subset

# Imputes one period of missing data, assuming linearity

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
    subset = prices.loc[pd.IndexSlice[elem[0][0], :], :]
    for ind, value in enumerate(subset.chng):
        if np.isnan(value):
            if ind == 0:
                three_set = subset.iloc[ind:ind+3].chng
            elif ind == len(subset)-1:
                three_set = subset.iloc[ind-2:ind].chng
            else:
                three_set = subset.iloc[ind-1:ind+2].chng
            if np.sum(np.isnan(three_set)) == 1:
                new_value = impute_one_period(three_set)
                subset.iloc[ind] = subset.iloc[ind].fillna(new_value)
    prices.loc[pd.IndexSlice[elem[0][0], :], :] = subset

# Percent change for consumer staples index

consumer_staples['chng'] = consumer_staples.Price.pct_change()

# Market (consumer staples index) excess return is the percent change - the return on the t-bill

consumer_staples['mkt_excess'] = consumer_staples.chng - t_bill.Price/100

# Creates a column for the year and month to avoid issues with days of the month

prices = prices.reset_index()
prices['temp_date'] = prices.datadate.dt.strftime('%Y-%b')

consumer_staples = consumer_staples.reset_index()
consumer_staples.Date = pd.to_datetime(consumer_staples.Date)
consumer_staples['temp_date'] = consumer_staples.Date.dt.strftime('%Y-%b')

prices_w_cs = prices.merge(consumer_staples, how='left',
                           left_on='temp_date', right_on='temp_date', suffixes=('', '_cs'))

t_bill = t_bill.reset_index()
t_bill.Date = pd.to_datetime(t_bill.Date)
t_bill['temp_date'] = t_bill.Date.dt.strftime('%Y-%b')

prices_w_cs_w_tb = prices_w_cs.merge(t_bill, how='left', left_on='temp_date', right_on='temp_date',
                                     suffixes=('', '_tb'))
prices_w_cs_w_tb.Price_tb = prices_w_cs_w_tb.Price_tb / 100
prices_w_cs_w_tb = prices_w_cs_w_tb.dropna(subset=['chng'])
prices_w_cs_w_tb = prices_w_cs_w_tb.drop(
    ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Date_tb', 'Open_tb', 'High_tb', 'Low_tb',
     'Change %_tb', 'temp_date'],
    axis=1
)

# Ensures that each asset has at least three data points

little_data = list()
for ticker in prices_w_cs_w_tb.tic.unique():
    if len(prices_w_cs_w_tb[prices_w_cs_w_tb.tic==ticker]) < 2:
        little_data.append(ticker)

prices_w_cs_w_tb = prices_w_cs_w_tb[~prices_w_cs_w_tb.tic.isin(little_data)]

prices_w_cs_w_tb = prices_w_cs_w_tb.dropna(subset=['chng', 'chng_cs', 'mkt_excess', 'Price_tb'])

prices_w_cs_w_tb.to_csv(DATA_PATH.joinpath('interim', 'prices_returns.txt'), index=False, sep='\t')

print('Cleaned prices and returns!')
