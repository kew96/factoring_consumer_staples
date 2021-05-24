from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data', 'processed', 'visualization')
FIGURE_PATH = Path(__file__).parent.parent.parent.joinpath('reports', 'figures')

returns_data = pd.read_table(DATA_PATH.joinpath('performance_data.txt'), parse_dates={'date': ['year', 'month']})

fig = plt.figure(figsize=(15, 8))
plt.plot(returns_data.date, returns_data.actual_ret, label='Realized Return')
plt.plot(returns_data.date, returns_data.expected_ret, label='Expected Return')
plt.title('Actual vs. Expected Excess Return of Fama French 3-Factor Model')
plt.legend()
plt.ylabel('Return')
plt.xlabel('Date')
plt.xticks(returns_data.date[1::2], labels=returns_data.date[1::2].dt.strftime('%y-%b'), rotation=-80)

plt.savefig(FIGURE_PATH.joinpath('actual_vs_expected_returns.png'),
            format='png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=0.3)