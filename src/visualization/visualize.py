from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data', 'processed')

data = pd.read_table(DATA_PATH.joinpath('performance_data.txt'), parse_dates={'date': ['year', 'month']},
                     index_col='date')

sns.lineplot