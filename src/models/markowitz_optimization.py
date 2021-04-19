import numpy as np
import cvxpy as cp
import pandas as pd
from tqdm import trange


class Markowitz:
    """
    An implementation of Markowitz Portfolio optimization that ensures you are long the first half assets and short the
    last half assets.

    Attributes
    ----------
    sigma: pandas.DataFrame
        An n x n DataFrame that represents the covariance matrix.

    Methods
    -------
    max_one_period_sharpe(expected_return, num_points=300, min_variance=0, max_variance=3)
        Returns a pandas.DataFrame with tickers as the index and their respective weights as the values.
    """

    def __init__(self, sigma):
        """
        Initializes the standard Markowitz portfolio optimization.

        Parameters
        ----------
        sigma: pandas.DataFrame
            An n x n DataFrame that represents the covariance matrix. Uses tickers as both the index and column names.
        """
        self.sigma = sigma

    def __optimal_one_period_weights(self, expected_return, max_variance):

        assert len(expected_return) % 2 == 0, 'There must be an even number of assets.'

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

        # Must go long the first half assets and short the last half assets
        half = len(expected_return) // 2
        long_short = [
            weights[:half] <= 1, # long
            weights[:half] >= 0,
            weights[half:] <= 0, # short
            weights[half:] >= -1
        ]

        # Aggregate constraints into one list
        constraints = long_short
        constraints.append(variance <= max_variance)
        constraints.append(total_invested == 0)

        portfolio_opt = cp.Problem(cp.Maximize(total_return), constraints=constraints)

        portfolio_opt.solve(verbose=False, solver='SCS')

        return {'excess_return': total_return.value[0], 'weights': weights.value}

    def max_one_period_sharpe(self, expected_return, num_points=300, *, min_variance=0, max_variance=3):
        """
        Finds the weights of the portfolio that correspond to the maximum Sharpe ratio

        Parameters
        ----------
        expected_return: pandas.DataFrame
            The expected return for the next period of the assets with ticker as the index and the expected return
            values column named "expected_return". The first half of assets will have a long position and the last
            half will have a short position.
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
            result = self.__optimal_one_period_weights(expected_return, variance)

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

    Attributes
    ----------
    alphas: pandas.Series
        The alphas of the assets based on the three-factor model.
    factor_loadings: pandas.DataFrame
        The factor loadings of each asset. Uses the tickers as the index.
    sigma: pandas.DataFrame
        An n x n DataFrame that represents the covariance matrix.

    Methods
    -------
    max_sharpe_portfolios(self, start_year=2000, end_year=2020, num_points=300, min_variance=0, max_variance=3)
        Calculates the realized return for each quarter from start_year to end_year based off expected asset returns
        from previous quarter.
    max_one_period_sharpe(expected_return, num_points=300, min_variance=0, max_variance=3)
        Returns a pandas.DataFrame with tickers as the index and their respective weights as the values.
    """

    def __init__(self, data, alphas, factor_loadings, sigma):
        """
        Initializes the ThreeFactorMarkowitz portfolio optimization.

        Parameters
        ----------
        data: pandas.DataFrame
            The overall data for the class. Must include the columns "tic", "datadate", "chng" which corresponds to
            the percent change in that period, "mkt_excess", "smb", and "hml". The last three correspond to the
            expected factor return.
        alphas: pandas.Series
            The alphas of the assets based on the three-factor model. These are generated via a weighted least squares
            regression. Use tickers as the index.
        factor_loadings: pandas.DataFrame
            The factor loadings of each asset. Uses the tickers as the index. Column names assume the traditional
            Fama-French three-factor model and names the columns "mkt_excess", "smb", and "hml" respectively.
        sigma: pandas.DataFrame
            An n x n DataFrame that represents the covariance matrix. Uses tickers as both the index and column names.
        """
        self.__raw_data = data
        self.alphas = alphas
        self.factor_loadings = factor_loadings
        self._expected_return = self._expected_asset_return()
        super().__init__(sigma)

    def _expected_asset_return(self):

        # Combines the factor loadings, expected factor returns, and alphas
        factor_returns_loadings = self.__raw_data.merge(self.factor_loadings.reset_index(), left_on='tic', right_on='index',
                                                        how='left', suffixes=('', '_loading'))

        modified_alpha = self.alphas.reset_index().rename({0: 'alpha'}, axis=1)
        factor_returns_loadings_alphas = factor_returns_loadings.merge(modified_alpha, left_on='tic',
                                                                       right_on='index', how='left')

        # Calculates the expected return for each asset from each factor.
        mkt_excess_return = factor_returns_loadings_alphas.mkt_excess * \
                            factor_returns_loadings_alphas.mkt_excess_loading

        smb_return = factor_returns_loadings_alphas.smb * factor_returns_loadings_alphas.smb_loading

        hml_return = factor_returns_loadings_alphas.hml * factor_returns_loadings_alphas.hml_loading

        # Combines all expected returns
        mu = factor_returns_loadings_alphas.alpha + mkt_excess_return + smb_return + hml_return

        return pd.DataFrame({'tic': self.__raw_data.tic,
                             'datadate': pd.to_datetime(self.__raw_data.datadate),
                             'expected_return': mu})

    @staticmethod
    def __next_period(year, quarter):
        if quarter == 4:
            return year+1, 3
        else:
            return year, (quarter+1)*3

    def __retrieve_universe(self, year, quarter, total_size=20):
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
        sub_universe = sorted_date_subset.iloc[list(range(total_size//2))+list(range(-total_size//2, 0))]
        return sub_universe.drop('datadate', axis=1).set_index('tic')

    @staticmethod
    def __last_period(year, quarter):
        if quarter == 1:
            return year-1, 4
        else:
            return year, quarter-1

    def max_sharpe_portfolios(self, start_year=2000, end_year=2020, num_points=300, *, min_variance=0,
                              max_variance=3, universe_size=20):
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

        Returns
        -------
        realized_returns: pandas.DataFrame
            Has the "year" and "month" that correspond to the respective quarter as the index with the realized
            return as values. The column is named "ret".

        """
        weights = list()
        total_returns = list()
        years = list()
        months = list()
        for year in trange(start_year, end_year+1, desc='Year'):
            for quarter in trange(1, 5, desc='Quarter', leave=False):
                if year == start_year and quarter == 1:
                    # We have no prior data to get the expected return for the current period
                    continue
                # Retrieve desired universe based off expected returns as of the previous period
                prev_year, prev_quarter = self.__last_period(year, quarter)
                universe = self.__retrieve_universe(prev_year, prev_quarter, universe_size)

                # Retrieve the actual return for the current period
                actual_returns = self.__raw_data[(self.__raw_data.datadate.dt.year == year) & (
                        self.__raw_data.datadate.dt.month == quarter * 3)].set_index('tic')

                # Calculate the optimal weights given a universe
                wgts = self.max_one_period_sharpe(universe, num_points,
                                                  min_variance=min_variance, max_variance=max_variance)

                # Calculate realized return, weight * actual return
                total_return = 0
                for ticker, w in zip(universe.index, wgts.weight):
                    ret = actual_returns.loc[ticker, 'chng']
                    total_return += ret * w
                years.append(year)
                months.append(quarter * 3)
                total_returns.append(total_return)
                weights.append(wgts)
        return pd.DataFrame({'year': years, 'month': months, 'ret': total_returns}).set_index(['year', 'month'])
