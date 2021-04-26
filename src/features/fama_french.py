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

    def __init__(self, *, data=None):
        """
        Initializes the ThreeFactorModel by calculating the factor loadings, alphas, deltas, p-values for factor
        loadings, covariance matrix and initializes inherited class.

        Parameters
        ----------
        data: str
            The file path to a tab-separated txt file containing all data needed to generate expected returns,
            actual returns, dates, and factors. Must include columns "tic" (ticker), "datadate" (date of entry),
            "chng" (percent change in asset's price from previous period), "mkt_excess" (asset's excess return over
            market), "Price_tb" (the risk-free rate), "ptb" (price-to-book ratio), and "mkvaltq" (market
            capitalization of asset).
            Default: a file named "three_factor_model.txt" in the directory "data/processed/".
        """
        if not data:
            data = self.__DATA_PATH.joinpath('processed', 'three_factor_model.txt')
        self.__raw_data = pd.read_table(data, parse_dates=['datadate'])
        self._generate_smb()
        self._generate_hml()
        FACTOR_DATA_PATH = self.__DATA_PATH.joinpath('processed', 'factor_data')
        if any(FACTOR_DATA_PATH.joinpath('covariance_matrices')) or any(FACTOR_DATA_PATH.joinpath('expected_returns')):
            ans = input('Would you like to generate the required data? (y/n):')
            if ans.lower() == 'y':
                self._generate_distinct_factors()
            else:
                raise FileNotFoundError('Please add required data to data/factor_data/')
        # super().__init__(self.__raw_data, self.alphas, self.factor_loadings, self.sigma)

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
        self.__raw_data = self.__raw_data.merge(SMB, how='left', on='datadate').drop_duplicates()

    def _generate_hml(self):
        hml = list()
        for date in self.__raw_data.datadate.unique():
            # Iterate through each available date
            temp = self.__raw_data[self.__raw_data.datadate == date][['chng', 'ptb']]

            ten = temp.ptb.quantile(0.1)
            ninety = temp.ptb.quantile(0.9)

            # Identifies the assets in these groups and calculate the average returns
            value = temp[temp.ptb < ten]
            growth = temp[temp.ptb > ninety]
            val_rets = value.chng.mean()
            growth_rets = growth.chng.mean()

            # Calculates the HML factor as value - growth
            factor = val_rets - growth_rets

            hml.append(factor)

        HML = pd.DataFrame({'datadate': self.__raw_data.datadate.unique(), 'hml': hml})

        # Modifies the base DataFrame
        self.__raw_data = self.__raw_data.merge(HML, how='left', on='datadate').drop_duplicates()

    @staticmethod
    def __check_dir(path):
        if not path.exists():
            path.mkdir()

    def _generate_distinct_factors(self, start_year=2005):

        MODEL_DATA = self.__DATA_PATH.joinpath('processed', 'factor_data')
        FACTOR_LOADINGS = MODEL_DATA.joinpath('factor_loadings')
        ALPHAS = MODEL_DATA.joinpath('alphas')
        P_VALUES = MODEL_DATA.joinpath('p_values')
        DELTAS = MODEL_DATA.joinpath('deltas')
        EXPECTED_RETURN = MODEL_DATA.joinpath('expected_returns')
        COVARIANCE = MODEL_DATA.joinpath('covariance_matrices')

        self.__check_dir(MODEL_DATA)
        self.__check_dir(FACTOR_LOADINGS)
        self.__check_dir(ALPHAS)
        self.__check_dir(P_VALUES)
        self.__check_dir(DELTAS)
        self.__check_dir(EXPECTED_RETURN)
        self.__check_dir(COVARIANCE)


        for date in self.__raw_data.datadate[self.__raw_data.datadate.dt.year >= start_year].unique():

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
                ticker_date_df = self.__raw_data[(self.__raw_data.tic == ticker) &
                                                 (self.__raw_data.datadate < date)]

                if len(ticker_date_df) < 2:
                    continue

                # Separate data into the dependent and independent variables and convert to NumPy arrays
                y = ticker_date_df.chng.values - ticker_date_df.Price_tb.values
                X = ticker_date_df[['mkt_excess', 'smb', 'hml']].values

                # Create list of weights to use for weighted least squares so that older data points carry less weight
                denom = sum(range(len(ticker_date_df)))
                try:
                    weights = [val / denom for val in range(1, len(ticker_date_df) + 1)]
                except:
                    print(ticker_date_df)
                    raise Exception

                # Actual weighted least squares model
                model = WLS(y, X, weights=weights).fit()

                # Calculate the predicted returns for each asset
                predictions = list()
                for row in X:
                    predicted = 0
                    for x_val, param_val in zip(row, model.params):
                        # Sum of the factor return * the factor loading
                        predicted += x_val * param_val
                    predictions.append(predicted)

                # Calculate the alpha for each asset
                alpha = 0
                for y_val, w, pred in zip(y, weights, predictions):
                    alpha += (y_val - pred) * w

                # The error in each prediction compared to the actual excess return
                errors = list()
                for pred, y_val in zip(predictions, y):
                    error = y_val - alpha - pred
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

            alpha = pd.Series(alphas, index=tickers).drop_duplicates()
            delta = pd.DataFrame(np.diagflat(delta_diag), columns=tickers,
                                 index=tickers).drop_duplicates()
            loadings = pd.DataFrame({'mkt_excess': beta_mkt_exc, 'smb': beta_smb, 'hml': beta_hml},
                                    index=tickers).drop_duplicates()
            p_values = pd.DataFrame({'mkt_excess': p_mx, 'smb': p_smb, 'hml': p_hml},
                                    index=tickers).drop_duplicates()
            expected_return = self.__expected_returns(date, loadings, alpha).drop_duplicates()
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
        subset_factor_returns = self.__raw_data[self.__raw_data.datadate==date][['tic', 'mkt_excess', 'smb', 'hml']]
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
        date_subset = self.__raw_data[self.__raw_data.datadate == date]
        f = date_subset[date_subset.tic == date_subset.tic.mode()[0]][
            ['datadate', 'mkt_excess', 'smb', 'hml']]
        denom = sum(range(len(f)))
        weights = [val / denom for val in range(1, len(f) + 1)]
        F = pd.DataFrame(0, columns=['mkt_excess', 'smb', 'hml'], index=['mkt_excess', 'smb', 'hml'])
        for w1, row1 in zip(weights, f.values):
            weighted_factors = np.zeros(3)
            for w2, row2 in zip(weights, f.values):
                weighted_factors = weighted_factors + np.multiply(w2, row2[1:])

            diff = row1[1:] - weighted_factors

            cov = np.dot(diff.reshape((3,1)), diff.reshape((1,3)))
            F += w1 * cov

        V = np.dot(np.dot(loadings.values, F), loadings.values.T) + deltas.values
        return pd.DataFrame(V, columns=loadings.index, index=loadings.index)
