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
consumer_staples.Date = consumer_staples.Date.apply(lambda x: x+' 1')
consumer_staples.Date = pd.to_datetime(consumer_staples.Date, format='%b %y %d').dt.strftime('%Y-%m')

t_bill = pd.read_csv(DATA_PATH.joinpath('raw', '3_month_t_bill.csv'))
t_bill.Date = t_bill.Date.apply(lambda x: x+' 1')
t_bill.Date = pd.to_datetime(t_bill.Date, format='%b %y %d').dt.strftime('%Y-%m')

prices = pd.read_table(DATA_PATH.joinpath('raw', 'prices_shares_outstanding_monthly.txt'),
                       parse_dates=['date'], usecols=['TICKER', 'date', 'COMNAM', 'PRC', 'RET'])
prices.columns = ['datadate', 'tic', 'conm', 'prccm', 'chng']
prices.datadate = prices.datadate.dt.strftime('%Y-%m')

prices = prices.dropna(subset=['tic'])

prices = prices.sort_values(['tic', 'datadate']).reset_index(drop=True)

nan_chng = list()
for entry in prices.chng:
    try:
        nan_chng.append(float(entry))
    except ValueError:
        nan_chng.append(np.nan)

prices.chng = nan_chng

# Sets date as the index

t_bill = t_bill.sort_values(['Date']).reset_index(drop=True)
consumer_staples = consumer_staples.sort_values(['Date']).reset_index(drop=True)

# Retrieves missing number of missing data points (outside of beginning and end) and prints the count, ticker,
# and company name

# Percent change for consumer staples index

consumer_staples['chng'] = consumer_staples['Change %'].apply(lambda x: float(x[:-1])/100)

# Market (consumer staples index) excess return is the percent change - the return on the t-bill

consumer_staples['mkt_excess'] = consumer_staples.chng - t_bill.Price/100

prices_w_cs = prices.merge(consumer_staples, how='left', left_on='datadate', right_on='Date', suffixes=('', '_cs'))

prices_w_cs_w_tb = prices_w_cs.merge(t_bill, how='left', left_on='datadate', right_on='Date', suffixes=('', '_tb'))

prices_w_cs_w_tb.Price_tb = prices_w_cs_w_tb.Price_tb / 100
prices_w_cs_w_tb = prices_w_cs_w_tb.dropna(subset=['chng'])

prices_w_cs_w_tb = prices_w_cs_w_tb.drop(
    ['Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %', 'Date_tb', 'Open_tb', 'High_tb', 'Low_tb',
     'Change %_tb'],
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
