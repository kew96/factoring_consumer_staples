import numpy as np
import cvxpy as cp
import pandas as pd


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
        constraints.append(total_invested == 1)

        portfolio_opt = cp.Problem(cp.Maximize(total_return), constraints=constraints)

        portfolio_opt.solve(verbose=True)

        return {'excess_return': total_return.value, 'weights': weights.value}

    def _max_one_period_sharpe(self, expected_return, num_points=300, *, min_variance=0, max_variance=3):

        all_returns = np.zeros(num_points)
        all_weights = np.array(num_points)
        all_variances = np.linspace(min_variance, max_variance, num_points)

        for ind, variance in enumerate(all_variances):
            # Get optimal weights and associated return given a portfolio variance
            result = self.__optimal_one_period_weights(expected_return, variance)

            all_returns[ind] = result['period_return']
            all_weights[ind] = result['weights']

        # Sharpe ratio defined as return divided by volatility
        sharpe_ratio = all_returns / np.sqrt(all_variances)

        # Find the index of the max sharpe ratio portfolio
        max_sharpe_ind = np.argmax(sharpe_ratio)

        # Return the weights associated with the max sharpe ratio portfolio
        return all_weights[max_sharpe_ind]


class ThreeFactorMarkowitz(Markowitz):

    def __init__(self, data, alphas, loadings, sigma):
        self.__data = data
        self.alphas = alphas
        self.loadings = loadings
        self._expected_return = self._expected_asset_return()
        super().__init__(sigma)

    def _expected_asset_return(self):
        factor_returns_loadings = self.__data.merge(self.loadings.reset_index(), left_on='tic', right_on='index',
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

    def __retrieve_universe(self, year, quarter):
        month = quarter * 3
        date_subset = self._expected_return[
            (self._expected_return.datadate.dt.year==year) & (self._expected_return.datadate.dt.month==month)
            ]
        sorted_date_subset = date_subset.sort_values('expected_return', ascending=False)
        sub_universe = sorted_date_subset.iloc[list(range(10))+list(range(-10, 0))]
        return sub_universe.drop('datadate', axis=1).set_index('tic')

    def max_sharpe_portfolios(self, start_year=2000, end_year=2020):
        weights = list()
        for year in range(start_year, end_year+1):
            for quarter in range(1, 5):
                universe = self.__retrieve_universe(year, quarter)
                print(self._max_one_period_sharpe(universe))


if __name__ == '__main__':
    from src.features import MacroModel
    tfl = MacroModel.ThreeFactorLoadings()
    tfm = ThreeFactorMarkowitz(tfl.data, tfl.alpha, tfl.factor_loading, tfl.Sigma)
    print(tfm.max_sharpe_portfolios(2000, 2002))
