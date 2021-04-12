from pathlib import Path
from collections import Counter
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.regression.linear_model import OLS
from datetime import datetime
import calendar


DATA_PATH = Path.cwd().parent.parent.joinpath('data')

consumer_staples = pd.read_csv(DATA_PATH.joinpath('raw', 'sp500_consumer_staples.csv'))
t_bill = pd.read_csv(DATA_PATH.joinpath('raw', '3_month_t_bill.csv'))
prices = pd.read_table(DATA_PATH.joinpath('raw', 'prices_assets_liabilities_quarterly.txt'),
                       parse_dates=['datadate'], usecols=['datadate', 'tic', 'conm', 'prccq'],
                       index_col=['datadate', 'tic'])

t_bill['Date'] = pd.to_datetime(t_bill.Date, format='%b %y')

print(type(t_bill.Date), t_bill.head())
