from pathlib import Path
from collections import Counter
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path().cwd().parent.parent.joinpath('data')


book_ratios = pd.read_table(DATA_PATH.joinpath('raw', 'prices_assets_liabilities_quarterly.txt'),
                            parse_dates=['datadate'])
book_ratios['book_value'] = (book_ratios.atq - book_ratios.ltq) * 1_000_000

shares = pd.read_table(DATA_PATH.joinpath('raw', 'prices_shares_outstanding_daily.txt'),
                       usecols=['datadate', 'tic', 'cshoc'], parse_dates=['datadate'])
book_ratios = book_ratios.merge(shares, on=['datadate', 'tic'], how='left')

def fix_missing_shares(df):
    for ticker in tqdm(df.tic.unique(), desc='tickers'):
        subset = shares[shares.tic==ticker].dropna()
        for row in tqdm(df[df.tic==ticker].itertuples(), desc=ticker, leave=False):
            if np.isnan(row.cshoc):
                val = np.nan
                dt = row.datadate
                if row.datadate.year == 2000 and row.datadate.month == 3:
                    while np.isnan(val):
                        dt = dt + relativedelta(days=1)
                        if dt > date(2021, 4, 1):
                            val = subset[subset.datadate==subset.datadate.max()].cshoc.values[0]
                        else:
                            vals_array = subset[subset.datadate==dt].cshoc.values
                            val = vals_array[0] if len(vals_array) else np.nan
                else:
                    while np.isnan(val):
                        dt = dt - relativedelta(days=1)
                        if dt < date(2000, 1, 1):
                            val = subset[subset.datadate==subset.datadate.min()].cshoc.values[0]
                        else:
                            vals_array = subset[subset.datadate==dt].cshoc.values
                            val = vals_array[0] if len(vals_array) else np.nan
                df.loc[row.Index, 'cshoc'] = val
    return df

book_ratios = fix_missing_shares(book_ratios)

book_ratios['bv_per_share'] = book_ratios.book_value / book_ratios.cshoc

book_ratios['ptb'] = book_ratios.prccq / book_ratios.bv_per_share

def get_bad_data(book_ratios):
    bad_data = list()
    for ticker in book_ratios.tic.unique():
        subset = book_ratios[book_ratios.tic==ticker]
        if np.isnan(subset.ptb.iloc[-1]):
            subset = subset.iloc[:-1]
        current = 'na' if np.isnan(subset.ptb.iloc[0]) else 'val'
        for v in subset.ptb[1:]:
            if not np.isnan(v):
                current = 'val'
            elif current == 'val' and np.isnan(v):
                name = subset.conm.values[0]
                bad_data.append((ticker, name))
                continue
    return Counter(bad_data)

bad_data_1 = get_bad_data(book_ratios)

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


for elem in bad_data_1.items():
    subset = book_ratios[book_ratios.tic == elem[0]]
    for ind, value1, value2 in zip(subset.index, subset.ptb[:-1], subset.ptb[1:]):
        if np.sum(np.isnan([value1, value2])) == 2:
            if ind == subset.index[0]:
                data = subset.loc[ind:ind+4, 'ptb']
            elif ind == subset.index[-2]:
                data = subset.loc[ind-2:, 'ptb']
            else:
                data = subset.loc[ind-1:ind+2, 'ptb']
            if np.sum(np.isnan(data)) > 2:
                continue
            else:
                new_vals = impute_two_periods(data)
            subset.loc[ind:ind+1, 'ptb'] = new_vals
    book_ratios[book_ratios.tic == elem[0]] = subset

def impute_one_period(data):
    if np.isnan(data.iloc[0]): # If the first period is missing
        diff = data.iloc[2] - data.iloc[1] # assume constant growth
        return data.iloc[1] - diff
    elif np.isnan(data.iloc[1]): # If the middle value is missing
        return (data.iloc[0] + data.iloc[2]) / 2 # take average of previous and next period
    else: # If the last period is missing
        diff = data.iloc[1] - data.iloc[0] # assume constant growth
        return data.iloc[1] + diff


for elem in bad_data_1.items():
    subset = book_ratios[book_ratios.tic == elem[0]]
    for ind, value in zip(subset.index, subset.ptb):
        if np.isnan(value):
            if ind == subset.index[0]:
                three_set = subset.loc[ind:ind+3, 'ptb']
            elif ind == subset.index[-1]:
                three_set = subset.loc[ind-2:ind, 'ptb']
            else:
                three_set = subset.loc[ind-1:ind+1, 'ptb']
            if np.sum(np.isnan(three_set)) == 1:
                new_value = impute_one_period(three_set)
                subset.loc[ind, 'ptb'] = new_value
    book_ratios[book_ratios.tic == elem[0]] = subset

book_ratios.to_csv(DATA_PATH.joinpath('interim', 'book_ratios.txt'), index=False, sep='\t')

print('Cleaned book ratios!')
