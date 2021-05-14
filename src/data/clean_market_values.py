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

market_values.columns = ['datadate', 'tic', 'conm', 'prccm', 'cshoc']

market_values.datadate = market_values.datadate.dt.strftime('%Y-%m')

market_values = market_values.dropna(subset=['tic'])

market_values.cshoc = market_values.cshoc * 1_000

market_values = market_values.sort_values(['tic', 'datadate']).reset_index(drop=True)

for ticker in market_values.tic.unique():
    subset = market_values[market_values.tic == ticker]
    subset.prccm = subset.prccm.interpolate(method='slinear', limit=2, limit_area='outside')
    subset.cshoc = subset.cshoc.interpolate(method='pad', limit=2, limit_area='outside')
    # Linearly interpolate period end prices
    end_prices = subset.prccm[-4:]
    if end_prices.isna().sum() == 2:
        # If there are two missing values and they are the last two values
        if np.isnan(end_prices.iloc[2]) and np.isnan(end_prices.iloc[3]):
            step = end_prices.iloc[1] - end_prices.iloc[0]
            end_prices.iloc[2] = end_prices.iloc[1] + step
            end_prices.iloc[3] = end_prices.iloc[1] + 2*step
    elif end_prices.isna().sum() == 1:
        # If there is one missing value and it is the last value
        if np.isnan(end_prices.iloc[-1]):
            step = end_prices.iloc[-2] - end_prices.iloc[-3]
            end_prices.iloc[-1] = end_prices.iloc[-2] + step
    market_values[market_values.tic == ticker] = subset

# Calculating monthly market value
market_values['mkvaltq'] = market_values.prccm * market_values.cshoc


# Reduces data to only months that are quarter end and that have a date after the 25th

quarterly_vals = market_values[market_values.datadate.str.contains('|'.join(['-03', '-06', '-09', '-12']))]

# Ensures each asset has at least three data points

little_data = list()
for ticker in quarterly_vals.tic.unique():
    if len(quarterly_vals[quarterly_vals.tic==ticker]) < 2:
        little_data.append(ticker)

quarterly_vals = quarterly_vals[~quarterly_vals.tic.isin(little_data)]

quarterly_vals.to_csv(DATA_PATH.joinpath('interim', 'market_values.txt'), index=False, sep='\t')

print('Cleaned market values!')
