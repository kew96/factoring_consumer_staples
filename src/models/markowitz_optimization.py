import numpy as np
import cvxpy as cp
import pandas as pd
from tqdm import trange


class Markowitz:

    def __init__(self, sigma):
        self.sigma = sigma

    def __optimal_one_period_weights(self, expected_return, max_variance):

        # Removing stocks not in universe
        sigma = self.sigma.loc[expected_return.index][expected_return.index]

        # Initiate variable for the weights to be optimized over
        weights = cp.Variable(len(expected_return))

        # Define the objective function for the expected return
        total_return = cp.matmul(expected_return.values.reshape((1, -1)), weights)

        # Define the variance of the portfolio given the weights
        variance = cp.quad_form(weights, sigma.values)

        # Define the total amount invested between all assets
        total_invested = cp.sum(weights)

        # Must go long the first 10 assets and short the last 10 assets
        long_short = [
            weights[:10] <= 1, # long
            weights[:10] >= 0,
            weights[10:] <= 0, # short
            weights[10:] >= -1
        ]

        # Aggregate constraints into one list
        constraints = long_short
        constraints.append(variance <= max_variance)
        constraints.append(total_invested == 0)

        portfolio_opt = cp.Problem(cp.Maximize(total_return), constraints=constraints)

        portfolio_opt.solve(verbose=False, solver='SCS')

        return {'excess_return': total_return.value[0], 'weights': weights.value}

    def _max_one_period_sharpe(self, expected_return, num_points=300, *, min_variance=0, max_variance=3):

        all_returns = np.zeros(num_points)
        all_weights = np.zeros((num_points, 20))
        if not min_variance:
            all_variances = np.linspace(min_variance, max_variance, num_points+1)[1:]
        else:
            all_variances = np.linspace(min_variance, max_variance, num_points)

        for ind, variance in enumerate(all_variances):
            # Get optimal weights and associated return given a portfolio variance
            result = self.__optimal_one_period_weights(expected_return, variance)
            # print(result)

            all_returns[ind] = result['excess_return']
            all_weights[ind] = result['weights']

        # Sharpe ratio defined as return divided by volatility
        sharpe_ratio = all_returns / np.sqrt(all_variances)

        # Find the index of the max sharpe ratio portfolio
        max_sharpe_ind = np.argmax(sharpe_ratio)

        # Return the weights associated with the max sharpe ratio portfolio
        return pd.DataFrame(all_weights[max_sharpe_ind], index=expected_return.index, columns=['weight'])


class ThreeFactorMarkowitz(Markowitz):

    def __init__(self, data, alphas, factor_loadings, sigma):
        self.__data = data
        self.alphas = alphas
        self.factor_loadings = factor_loadings
        self._expected_return = self._expected_asset_return()
        super().__init__(sigma)

    def _expected_asset_return(self):
        factor_returns_loadings = self.__data.merge(self.factor_loadings.reset_index(), left_on='tic', right_on='index',
                                                    how='left', suffixes=('', '_loading'))
        modified_alpha = self.alphas.reset_index().rename({0: 'alpha'}, axis=1)
        factor_returns_loadings_alphas = factor_returns_loadings.merge(modified_alpha, left_on='tic',
                                                                       right_on='index', how='left')
        mkt_excess_return = factor_returns_loadings_alphas.mkt_excess * \
                            factor_returns_loadings_alphas.mkt_excess_loading

        smb_return = factor_returns_loadings_alphas.smb * factor_returns_loadings_alphas.smb_loading

        hml_return = factor_returns_loadings_alphas.hml * factor_returns_loadings_alphas.hml_loading

        mu = factor_returns_loadings_alphas.alpha + mkt_excess_return + smb_return + hml_return

        return pd.DataFrame({'tic': self.__data.tic,
                             'datadate': pd.to_datetime(self.__data.datadate),
                             'expected_return': mu})

    @staticmethod
    def __next_period(year, quarter):
        if quarter == 4:
            return year+1, 3
        else:
            return year, (quarter+1)*3

    def __retrieve_universe(self, year, quarter):
        month = quarter * 3
        date_subset = self._expected_return[
            (self._expected_return.datadate.dt.year==year) & (self._expected_return.datadate.dt.month==month)
            ]
        sorted_date_subset = date_subset.sort_values('expected_return', ascending=False)
        next_year, next_month = self.__next_period(year, quarter)
        next_subset = self._expected_return[
            (self._expected_return.datadate.dt.year == next_year) & (
                        self._expected_return.datadate.dt.month == next_month)
            ]
        sorted_date_subset = sorted_date_subset[sorted_date_subset.tic.isin(list(next_subset.tic))]
        sub_universe = sorted_date_subset.iloc[list(range(10))+list(range(-10, 0))]
        return sub_universe.drop('datadate', axis=1).set_index('tic')

    @staticmethod
    def __last_period(year, quarter):
        if quarter == 1:
            return year-1, 4
        else:
            return year, quarter-1

    def max_sharpe_portfolios(self, start_year=2000, end_year=2020, num_points=300, *, min_variance=0, max_variance=3):
        weights = list()
        total_returns = list()
        years = list()
        months = list()
        for year in trange(start_year, end_year+1, desc='Year'):
            for quarter in trange(1, 5, desc='Quarter', leave=False):
                if year == start_year and quarter == 1:
                    continue
                elif year == end_year and quarter == 4:
                    continue
                prev_year, prev_quarter = self.__last_period(year, quarter)
                universe = self.__retrieve_universe(prev_year, prev_quarter)
                actual_returns = self.__data[(self.__data.datadate.dt.year==year) & (
                        self.__data.datadate.dt.month==quarter*3)].set_index('tic')
                wgts = self._max_one_period_sharpe(universe, num_points,
                                                   min_variance=min_variance, max_variance=max_variance)
                total_return = 0
                for ticker, w in zip(universe.index, wgts.weight):
                    ret = actual_returns.loc[ticker, 'chng']
                    total_return += ret * w
                years.append(year)
                months.append(quarter * 3)
                total_returns.append(total_return)
                weights.append(wgts)
        return pd.DataFrame({'year': years, 'month': months, 'ret': total_returns}).set_index(['year', 'month'])
