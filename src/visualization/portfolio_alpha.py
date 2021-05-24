from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

DATA_PATH = Path.cwd().parent.parent.joinpath('data', 'processed')
FIGURE_PATH = Path(__file__).parent.parent.parent.joinpath('reports', 'figures')

weights_data = pd.read_table(DATA_PATH.joinpath('visualization', 'weights_data.txt'), parse_dates={'date': ['year', 'month']})
weights_data = weights_data.fillna(0)

alphas = list()
for dt in weights_data.date:
    file = dt.strftime('%Y.%m.txt')
    df = pd.read_table(DATA_PATH.joinpath('factor_data', 'alphas', file))
    subset = weights_data[weights_data.date==dt].drop('date', axis=1).T
    subset.columns = ['weight']
    subset = subset.merge(df, how='left', left_index=True, right_on='tic')
    subset = subset[subset.weight!=0]
    a = (subset.weight * subset.alpha).sum()
    alphas.append(a)

fig = plt.figure(figsize=(15,5))
plt.plot(weights_data.date, alphas)
plt.title('Portfolio Alpha')
plt.ylabel('Alpha')
plt.xlabel('Date')
plt.xticks(weights_data.date[1::2], labels=weights_data.date[1::2].dt.strftime('%y-%b'), rotation=-80)

plt.savefig(FIGURE_PATH.joinpath('visualization', 'portfolio_alpha.png'),
            format='png',
            transparent=False,
            bbox_inches='tight',
            pad_inches=0.3)
