from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data', 'processed', 'visualization')

returns_data = pd.read_table(DATA_PATH.joinpath('performance_data.txt'), parse_dates={'date': ['year', 'month']})
mod_returns_data = pd.read_table(DATA_PATH.joinpath('modified_alpha_performance_data.txt'),
                                 parse_dates={'date': ['year', 'month']})

fig = plt.figure(figsize=(15,10))
plt.plot(returns_data.date, returns_data.actual_ret, label='Base')
plt.plot(mod_returns_data.date, mod_returns_data.expected_ret, label='Alternative')
plt.title('Base vs. Alternative Model Excess Returns')
plt.legend()
plt.ylabel('Return')
plt.xlabel('Date')
plt.xticks(returns_data.date[1::2], labels=returns_data.date[1::2].dt.strftime('%y-%b'), rotation=-80)

plt.savefig(DATA_PATH.joinpath('base_vs_alternative_returns.png'),
            format='png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=0.3)