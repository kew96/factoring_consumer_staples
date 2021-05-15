from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.api import WLS

from src.models.markowitz_optimization import ThreeFactorMarkowitz


class ThreeFactorModel(ThreeFactorMarkowitz):
    """
    The overarching model for the Fama-French Three-Factor model that performs factor calculations, portfolio
    optimization, and back testing. Inherits from ThreeFactorMarkowitz

    Attributes
    ----------
    alphas: pandas.Series
        The alphas of the assets based on the three-factor model.
    delta: pandas.DataFrame
        The idiosyncratic error variance for each asset as a diagonal matrix.
    factor_loadings: pandas.DataFrame
        The factor loadings of each asset. Uses the tickers as the index.
    p_values: pandas.DataFrame
        The p-values associated with each asset's factor loadings from the weighted least squares regression.
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
    __DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')

    def __init__(self, start_year=2005, start_quarter=1, *, data=None):
        """
        Initializes the ThreeFactorModel by calculating the factor loadings, alphas, deltas, p-values for factor
        loadings, covariance matrix and initializes inherited class.

        Parameters
        ----------
        start_year: int
            The first year to start generating data from.
            Default: 2005
        start_quarter: int
            The first quarter within the start year to generate data from.
            Default: 1
        data: str
            The file path to a tab-separated txt file containing all data needed to generate expected returns,
            actual returns, dates, and factors. Must include columns "tic" (ticker), "datadate" (date of entry),
            "chng" (percent change in asset's price from previous period), "mkt_excess" (asset's excess return over
            market), "Price_tb" (the risk-free rate), "btm" (book-to-market ratio), and "mkvaltq" (market
            capitalization of asset).
            * Keyword argument only
            Default: a file named "three_factor_model.txt" in the directory "data/processed/".
        """
        if not data:
            data = self.__DATA_PATH.joinpath('processed', 'three_factor_model.txt')
        self.__raw_data = pd.read_table(data, parse_dates=['datadate'])
        self._generate_smb()
        self._generate_hml()
        FACTOR_DATA_PATH = self.__DATA_PATH.joinpath('processed', 'factor_data')

        if not FACTOR_DATA_PATH.joinpath('covariance_matrices').exists() or \
            not FACTOR_DATA_PATH.joinpath('expected_returns').exists() or \
            not any(FACTOR_DATA_PATH.joinpath('covariance_matrices').iterdir()) or \
            not any(FACTOR_DATA_PATH.joinpath('expected_returns').iterdir()):
            ans = input('Would you like to generate the required data? (y/n): ')
            if ans.lower() == 'y':
                print('Generating data! (This may take a bit.)')
                self._generate_distinct_factors(start_year=start_year, start_quarter=start_quarter)
            else:
                raise FileNotFoundError(f'Please add required data to {FACTOR_DATA_PATH}')
        ThreeFactorMarkowitz.__init__(self, self.__raw_data)

    @property
    def data(self):
        # A getter for the underlying data
        return self.__raw_data

    def _generate_smb(self):
        smb = list()
        for date in self.__raw_data.datadate.unique():
            # Iterate through each available date
            temp = self.__raw_data[self.__raw_data.datadate == date][['chng', 'mkvaltq']]

            ten = temp.mkvaltq.quantile(0.1)
            ninety = temp.mkvaltq.quantile(0.9)

            # Identifies the assets in these groups and calculate the average returns
            small = temp[temp.mkvaltq < ten]
            big = temp[temp.mkvaltq > ninety]
            small_rets = small.chng.mean()
            big_rets = big.chng.mean()

            # Calculate SMB factor as small - big
            factor = small_rets - big_rets

            smb.append(factor)

        SMB = pd.DataFrame({'datadate': self.__raw_data.datadate.unique(), 'smb': smb})

        # Modifies the base DataFrame
        self.__raw_data = self.__raw_data.merge(SMB, how='left', on='datadate')

    def _generate_hml(self):
        hml = list()
        for date in self.__raw_data.datadate.unique():
            # Iterate through each available date
            temp = self.__raw_data[self.__raw_data.datadate == date][['chng', 'btm']]

            ten = temp.btm.quantile(0.1)
            ninety = temp.btm.quantile(0.9)

            # Identifies the assets in these groups and calculate the average returns
            growth = temp[temp.btm < ten]
            value = temp[temp.btm > ninety]
            val_rets = value.chng.mean()
            growth_rets = growth.chng.mean()

            # Calculates the HML factor as value - growth
            factor = val_rets - growth_rets

            hml.append(factor)

        HML = pd.DataFrame({'datadate': self.__raw_data.datadate.unique(), 'hml': hml})

        # Modifies the base DataFrame
        self.__raw_data = self.__raw_data.merge(HML, how='left', on='datadate')

    @staticmethod
    def __check_dir(path):
        if not path.exists():
            path.mkdir()

    def _generate_distinct_factors(self, start_year=2005, start_quarter=1):

        FACTOR_DATA_PATH = self.__DATA_PATH.joinpath('processed', 'factor_data')
        FACTOR_LOADINGS = FACTOR_DATA_PATH.joinpath('factor_loadings')
        ALPHAS = FACTOR_DATA_PATH.joinpath('alphas')
        P_VALUES = FACTOR_DATA_PATH.joinpath('p_values')
        DELTAS = FACTOR_DATA_PATH.joinpath('deltas')
        EXPECTED_RETURN = FACTOR_DATA_PATH.joinpath('expected_returns')
        COVARIANCE = FACTOR_DATA_PATH.joinpath('covariance_matrices')

        self.__check_dir(FACTOR_DATA_PATH)
        self.__check_dir(FACTOR_LOADINGS)
        self.__check_dir(ALPHAS)
        self.__check_dir(P_VALUES)
        self.__check_dir(DELTAS)
        self.__check_dir(EXPECTED_RETURN)
        self.__check_dir(COVARIANCE)


        for date in self.__raw_data.datadate[
            self.__raw_data.datadate >= f'{start_year}-{start_quarter*3:02}-01'].unique():

            tickers = list()
            alphas = list()
            beta_mkt_exc = list()
            beta_smb = list()
            beta_hml = list()
            p_mx = list()
            p_smb = list()
            p_hml = list()
            r2 = list()
            delta_diag = list()

            for ticker in self.__raw_data.tic.unique():
                if ticker in tickers:
                    continue

                # Iterate through each asset, individually
                ticker_date_df = self.__raw_data[(self.__raw_data.tic == ticker) & (self.__raw_data.datadate < date)]

                if len(ticker_date_df) < 4:
                    continue

                # Separate data into the dependent and independent variables and convert to NumPy arrays
                y = ticker_date_df.chng.values - ticker_date_df.Price_tb.values
                X = ticker_date_df[['mkt_excess', 'smb', 'hml']].values

                # Create list of weights to use for weighted least squares so that older data points carry less weight
                denom = sum(range(len(ticker_date_df)))
                weights = [val / denom for val in range(1, len(ticker_date_df) + 1)]

                # Actual weighted least squares model
                model = WLS(y, X, weights=weights).fit()

                # Calculate the predicted returns for each asset
                predictions = list()
                for row in X:
                    predicted = 0
                    for factor, beta in zip(row, model.params):
                        # Sum of the factor return * the factor loading
                        predicted += factor * beta
                    predictions.append(predicted)

                # Calculate the alpha for each asset
                alpha = 0
                for actual_return, wgt, predicted_return in zip(y, weights, predictions):
                    alpha += (actual_return - predicted_return) * wgt

                # The error in each prediction compared to the actual excess return
                errors = list()
                for predicted_return, actual_return in zip(predictions, y):
                    error = actual_return - alpha - predicted_return
                    errors.append(error)

                # Calculate the squared error
                squared_error = sum([w * error ** 2 for w, error in zip(weights, errors)])

                tickers.append(ticker)
                alphas.append(alpha)
                beta_mkt_exc.append(model.params[0])
                beta_smb.append(model.params[1])
                beta_hml.append(model.params[2])
                p_mx.append(model.pvalues[0])
                p_smb.append(model.pvalues[1])
                p_hml.append(model.pvalues[2])
                r2.append(model.rsquared)
                delta_diag.append(squared_error)

            alpha = pd.Series(alphas, index=tickers).dropna()
            delta = pd.DataFrame(np.diagflat(delta_diag), columns=tickers, index=tickers).dropna()
            loadings = pd.DataFrame({'mkt_excess': beta_mkt_exc, 'smb': beta_smb, 'hml': beta_hml},
                                    index=tickers).dropna()
            p_values = pd.DataFrame({'mkt_excess': p_mx, 'smb': p_smb, 'hml': p_hml}, index=tickers).dropna()

            expected_return = self.__expected_returns(date, loadings, alpha).dropna()
            covariance_matrix = self.__factor_covariance(date, loadings, delta)

            date_str = '.'.join(np.datetime_as_string(date).split('-')[:2]) + '.txt'

            alpha.to_csv(ALPHAS.joinpath(date_str), sep='\t', header=['alpha'], index_label='tic')
            delta.to_csv(DELTAS.joinpath(date_str), sep='\t', index_label='tic')
            loadings.to_csv(FACTOR_LOADINGS.joinpath(date_str), sep='\t', index_label='tic')
            p_values.to_csv(P_VALUES.joinpath(date_str), sep='\t', index_label='tic')
            expected_return.to_csv(EXPECTED_RETURN.joinpath(date_str), sep='\t',
                                                     header=['ret'], index_label='tic')
            covariance_matrix.to_csv(COVARIANCE.joinpath(date_str), sep='\t', index_label='tic')

    def __expected_returns(self, date, loadings, alphas):

        # Combines the factor loadings, expected factor returns, and alphas
        subset_factor_returns = self.__raw_data[self.__raw_data.datadate == date][['tic', 'mkt_excess', 'smb', 'hml']]
        factor_returns_loadings = loadings.reset_index().merge(subset_factor_returns, left_on='index',
                                                        right_on='tic', how='left', suffixes=('_loading', ''))

        modified_alpha = alphas.reset_index().rename({0: 'alpha'}, axis=1)
        factor_returns_loadings_alphas = factor_returns_loadings.merge(modified_alpha, left_on='tic',
                                                                       right_on='index', how='left')

        # Calculates the expected return for each asset from each factor.
        mkt_excess_return = factor_returns_loadings_alphas.mkt_excess * \
                            factor_returns_loadings_alphas.mkt_excess_loading

        smb_return = factor_returns_loadings_alphas.smb * factor_returns_loadings_alphas.smb_loading

        hml_return = factor_returns_loadings_alphas.hml * factor_returns_loadings_alphas.hml_loading

        # Combines all expected returns
        mu = factor_returns_loadings_alphas.alpha + mkt_excess_return + smb_return + hml_return

        expected_return = pd.Series(mu)
        expected_return.index = factor_returns_loadings_alphas.tic

        return expected_return

    def __factor_covariance(self, date, loadings, deltas):
        date_subset = self.__raw_data[self.__raw_data.datadate < date]
        f = date_subset[['datadate', 'mkt_excess', 'smb', 'hml']].drop_duplicates().sort_values('datadate').copy()
        denom = sum(range(len(f)))
        weights = np.array([val / denom for val in range(1, len(f) + 1)])
        F = np.zeros((3, 3), dtype=float)
        weighted_factor_sum = np.sum(f.values[:, 1:] * weights.reshape((-1, 1)), axis=0)
        for wgt, factors in zip(weights, f.values[:, 1:]):
            factor_diff = factors - weighted_factor_sum
            weighted_date = wgt * np.dot(factor_diff.reshape((-1, 1)), factor_diff.reshape((1, -1)))
            F = F + weighted_date
        V = np.dot(np.dot(loadings.values, F), loadings.values.T) + deltas.values
        return pd.DataFrame(V, columns=loadings.index, index=loadings.index)


if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    tfm = ThreeFactorModel()
    print(tfm.data.head())
