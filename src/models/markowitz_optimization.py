import sys
from pathlib import Path

import numpy as np
import cvxpy as cp
import pandas as pd

if 'ipykernel' in sys.modules:
    from tqdm.notebook import trange
else:
    from tqdm import trange


class Markowitz:
    """
    An implementation of Markowitz Portfolio optimization that ensures you are long the first half assets and short the
    last half assets.

    Methods
    -------
    max_one_period_sharpe(expected_return, num_points=300, min_variance=0, max_variance=3)
        Returns a pandas.DataFrame with tickers as the index and their respective weights as the values.
    """

    def __init__(self):
        """
        Initializes the standard Markowitz portfolio optimization.
        """

    @staticmethod
    def __optimal_one_period_weights(expected_return, sigma, max_variance):

        assert len(expected_return) % 2 == 0, 'There must be an even number of assets.'

        # Removing stocks not in universe
        sigma = sigma.loc[expected_return.index][expected_return.index]

        # Initiate variable for the weights to be optimized over
        weights = cp.Variable(len(expected_return))

        # Define the objective function for the expected return
        total_return = cp.matmul(expected_return.values.reshape((1, -1)), weights)

        # Define the variance of the portfolio given the weights
        variance = cp.quad_form(weights, sigma.values)

        # Define the total amount invested between all assets
        total_invested = cp.sum(weights)

        # Must go long the first half assets and short the last half assets
        half = len(expected_return) // 2
        long_short = [
            weights[:half] <= 0.5, # long
            weights[:half] >= 0,
            weights[half:] <= 0, # short
            weights[half:] >= -0.5
        ]

        # Aggregate constraints into one list
        constraints = long_short
        constraints.append(variance <= max_variance)
        constraints.append(total_invested <= 1e-6) # Instead of strict equality to zero, create a tight band
        constraints.append(total_invested >= -1e-6)

        portfolio_opt = cp.Problem(cp.Maximize(total_return), constraints=constraints)

        portfolio_opt.solve(verbose=False)

        return {'excess_return': total_return.value[0], 'weights': weights.value}

    def max_one_period_sharpe(self, expected_return, sigma, num_points=300, *, min_variance=0, max_variance=3):
        """
        Finds the weights of the portfolio that correspond to the maximum Sharpe ratio

        Parameters
        ----------
        expected_return: pandas.DataFrame
            The expected return for the next period of the assets with ticker as the index and the expected return
            values column named "expected_return". The first half of assets will have a long position and the last
            half will have a short position.
        sigma: pandas.DataFrame
            An n x n DataFrame that represents the covariance matrix. Uses tickers as both the index and column names.
        num_points: int
            The number of points to search over to approximate the maximum Sharpe ratio portfolio.
            Default: 300
        min_variance: int
            The minimum variance to start searching for the maximum Sharpe ratio portfolio.
            Default: 0
            * Keyword argument only
        max_variance: int
            The minimum variance to start searching for the maximum Sharpe ratio portfolio.
            Default: 3
            * Keyword argument only

        Returns
        -------
        weights: pandas.DataFrame
            The weights associated with the maximum Sharpe ratio portfolio with tickers as the index and the weights
            as the values in a column titled "weight".

        """

        all_returns = np.zeros(num_points)
        all_weights = np.zeros((num_points, len(expected_return)))
        if not min_variance:
            all_variances = np.linspace(min_variance, max_variance, num_points+1)[1:]
        else:
            all_variances = np.linspace(min_variance, max_variance, num_points)

        for ind, variance in enumerate(all_variances):
            # Get optimal weights and associated return given a portfolio variance

            result = self.__optimal_one_period_weights(expected_return, sigma, variance)

            all_returns[ind] = result['excess_return']
            all_weights[ind] = result['weights']

        # Sharpe ratio defined as return divided by volatility
        sharpe_ratio = all_returns / np.sqrt(all_variances)

        # Find the index of the max sharpe ratio portfolio
        max_sharpe_ind = np.argmax(sharpe_ratio)

        # Return the weights associated with the max sharpe ratio portfolio
        return pd.DataFrame(all_weights[max_sharpe_ind], index=expected_return.index, columns=['weight'])


class ThreeFactorMarkowitz(Markowitz):
    """
    An extension of the class Markowitz that is focused on optimizing three-factor models.

    Methods
    -------
    max_sharpe_portfolios(self, start_year=2000, end_year=2020, num_points=300, min_variance=0, max_variance=3)
        Calculates the realized return for each quarter from start_year to end_year based off expected asset returns
        from previous quarter.
    max_one_period_sharpe(expected_return, num_points=300, min_variance=0, max_variance=3)
        Returns a pandas.DataFrame with tickers as the index and their respective weights as the values.
    """

    def __init__(self, raw_data, *, data=None):
        """
        Initializes the ThreeFactorMarkowitz portfolio optimization.

        Parameters
        ----------
        raw_data: pandas.DataFrame
            The overall data for the class. Must include the columns "tic", "datadate", "chng" which corresponds to
            the percent change in that period, "mkt_excess", "smb", and "hml". The last three correspond to the
            expected factor return.

        data: str
            Path to the data folder
            * Optional
            * Keyword only
        """
        self.__raw_data = raw_data
        if not data:
            self.__FACTOR_DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data', 'processed', 'factor_data')
        else:
            self.__FACTOR_DATA_PATH = data

    @staticmethod
    def __next_period(year, quarter):
        if quarter == 4:
            return year+1, 3
        else:
            return year, (quarter+1)*3

    @staticmethod
    def __last_period(year, quarter):
        if quarter == 1:
            return year-1, 4
        else:
            return year, quarter-1

    def __retrieve_universe(self, year, quarter, total_size=20):
        month = quarter * 3
        date_subset = pd.read_table(self.__FACTOR_DATA_PATH.joinpath('expected_returns', f'{year}.{month:02}.txt'))
        sorted_date_subset = date_subset.sort_values('ret', ascending=False)
        next_year, next_month = self.__next_period(year, quarter)
        next_subset = pd.read_table(self.__FACTOR_DATA_PATH.joinpath(
            'expected_returns', f'{next_year}.{next_month:02}.txt'))
        sorted_date_subset = sorted_date_subset[sorted_date_subset.tic.isin(list(next_subset.tic))]
        sorted_date_subset = sorted_date_subset.drop_duplicates(subset=['tic'])
        sub_universe = sorted_date_subset.iloc[list(range(total_size//2))+list(range(-total_size//2, 0))]
        return sub_universe.set_index('tic')

    @staticmethod
    def __historical_portfolio_return(sub_universe, weights):
        weighted_returns = sub_universe.merge(weights, left_on='tic', right_index=True)
        weighted_returns['wgt_return'] = weighted_returns.chng * weighted_returns.weight
        date_returns = weighted_returns.groupby('datadate').agg({'wgt_return': sum})
        return date_returns

    @staticmethod
    def __calculate_sharpe(historical_returns):
        expected_excess = historical_returns.wgt_return.mean()
        vol = historical_returns.wgt_return.std()
        return expected_excess / vol


    def max_sharpe_portfolios(self, start_year=2005, end_year=2020, num_points=300, *, min_variance=0,
                              max_variance=3, universe_size=20, return_sharpe=False, return_weights=False):
        """
        Calculates the realized return for each quarter from start_year to end_year based off expected asset returns
        from previous quarter. Utilizes methods from the class Markowitz to find the optimal weights for the
        following period.

        Parameters
        ----------
        start_year: int
            The first year to start optimizing portfolios.
            Default: 2000
        end_year: int
            The last year to start optimizing portfolios.
            Default: 2020
        num_points: int
            The number of points to search over to approximate the maximum Sharpe ratio portfolio.
            Default: 300
        min_variance: int
            The minimum variance to start searching for the maximum Sharpe ratio portfolio.
            Default: 0
            * Keyword argument only
        max_variance: int
            The minimum variance to start searching for the maximum Sharpe ratio portfolio.
            Default: 3
            * Keyword argument only
        universe_size: int
            The total assets in the universe. Must be an even number.
            Default: 20
            * Keyword argument only
        return_sharpe: bool
            If True, will include the Sharpe ratio of the portfolio for the given time period.
            Default: False
            * Keyword argument only
        return_weights: bool
            If True, will include the weights of the portfolio for the given time period for all assets invested over
            the complete period. Will result in a second pandas.DataFrame being returned.
            Default: False
            * Keyword argument only

        Returns
        -------
        realized_returns: pandas.DataFrame
            Has the "year" and "month" that correspond to the respective quarter as the index with the realized
            and excess returns as values. Can include the Sharpe ratio through optional arguments.

        weights: pandas.DataFrame
            Has the "year" and "month" that correspond to the respective quarter as the index with tickers of assets
            invested in as the columns and values as the entries.

        """
        if return_weights:
            weights = pd.DataFrame()
        years = list()
        months = list()
        total_returns = list()
        expected_returns = list()
        sharpe = list()
        for year in trange(start_year, end_year+1, desc='Year', leave=False):
            for quarter in trange(1, 5, desc=f'{year}', leave=False):
                if year == 2020 and quarter == 4:
                    continue
                universe = self.__retrieve_universe(year, quarter, universe_size)
                sigma = pd.read_table(self.__FACTOR_DATA_PATH.joinpath(
                    'covariance_matrices', f'{year}.{quarter*3:02}.txt'), index_col=['tic'])

                # Retrieve the actual return for the current period
                actual_returns = self.__raw_data[(self.__raw_data.datadate.dt.year == year) & (
                        self.__raw_data.datadate.dt.month == quarter * 3)]
                actual_returns = actual_returns.drop_duplicates(subset=['tic']).set_index('tic')

                # Calculate the optimal weights given a universe
                wgts = self.max_one_period_sharpe(universe, sigma, num_points,
                                                  min_variance=min_variance, max_variance=max_variance)

                # Simply multiply the expected return for each asset by the assigned weight and sum over all assets
                expected_return = sum(universe.ret.values * wgts.weight.values)

                # Calculate realized return, weight * actual return
                total_return = 0
                for ticker, w in zip(universe.index, wgts.weight):
                    ret = actual_returns.loc[ticker, 'chng']
                    total_return += ret * w
                if return_sharpe:
                    # Subsets data to get historical data
                    historical_universe = self.__raw_data[(self.__raw_data.tic.isin(universe.index)) &
                                                          (self.__raw_data.datadate < f'{year}-{quarter*3}-1')
                                                          ][['tic', 'datadate', 'chng']]

                    # Finds the historical returns given the current portfolio weights
                    historical_returns = self.__historical_portfolio_return(historical_universe, wgts)

                    sharpe_ratio = self.__calculate_sharpe(historical_returns)
                    sharpe.append(sharpe_ratio)


                years.append(year)
                months.append(quarter * 3)
                total_returns.append(total_return)
                expected_returns.append(expected_return)
                if return_weights:
                    wgt_dict = dict(zip(wgts.index, wgts.weight.values))
                    weights = weights.append(wgt_dict, ignore_index=True)

        if return_weights:
            weights['year'] = years
            weights['month'] = months
            weights = weights.set_index(['year', 'month'])

        if return_sharpe and return_weights:
            return (pd.DataFrame({
                'year': years, 'month': months, 'actual_ret': total_returns, 'expected_ret': expected_returns,
                'sharpe': sharpe
            }).set_index(['year', 'month']), weights)
        elif return_weights:
            return (pd.DataFrame({
                'year': years, 'month': months, 'actual_ret': total_returns, 'expected_ret': expected_returns
            }).set_index(['year', 'month']), weights)
        elif return_sharpe:
            return pd.DataFrame({
                'year': years, 'month': months, 'actual_ret': total_returns, 'expected_ret': expected_returns,
                'sharpe': sharpe
            }).set_index(['year', 'month'])
        else:
            return pd.DataFrame({
                'year': years, 'month': months, 'actual_ret': total_returns, 'expected_ret': expected_returns
            }).set_index(['year', 'month'])
