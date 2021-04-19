from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.api import WLS

from src.models.markowitz_optimization import ThreeFactorMarkowitz


class ThreeFactorModel(ThreeFactorMarkowitz):
    """
    The overarching model for the Fama-French Three-Factor model that performs factor calculations, portfolio
    optimization, and backtesting. Inherits from ThreeFactorMarkowitz

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
        self._generate_distinct_factors()
        self.alphas, self.delta, self.factor_loadings, self.p_values = self._generate_distinct_factors()
        self.sigma = self._generate_factor_covariance()
        super().__init__(self.__raw_data, self.alphas, self.factor_loadings, self.sigma)

    @property
    def data(self):
        # A getter for the underlying data
        return self.__raw_data

    def _generate_smb(self):
        smb = list()
        for date in self.__raw_data.datadate.unique():
            # Iterate through each available date
            temp = self.__raw_data[self.__raw_data.datadate == date][['chng', 'mkvaltq']]

            # Find top and bottom decile based on the market capitalization
            ten = self.__raw_data.mkvaltq.quantile(0.1)
            ninety = self.__raw_data.mkvaltq.quantile(0.9)

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

            # Find top and bottom decile based on the price-to-book ratio
            ten = self.__raw_data.ptb.quantile(0.1)
            ninety = self.__raw_data.ptb.quantile(0.9)

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

    def _generate_distinct_factors(self):
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

            # Iterate through each asset, individually
            temp_df = self.__raw_data[self.__raw_data.tic == ticker]

            # Separate data into the dependent and independent variables and convert to NumPy arrays
            y = temp_df.chng.values - temp_df.Price_tb.values
            X = temp_df[['mkt_excess', 'smb', 'hml']].values

            # Create list of weights to use for weighted least squares so that older data points carry less weight
            denom = sum(range(len(temp_df)))
            weights = [val / denom for val in range(1, len(temp_df) + 1)]

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
                alpha += y_val - pred

            # The error in each prediction compared to the actual excess return
            errors = list()
            for pred, y_val in zip(predictions, y):
                error = y_val - alpha - pred
                errors.append(error)

            # Calculate the squared error
            squared_error = sum([w * error ** 2 for w, error in zip(weights, errors)])

            alphas.append(alpha)
            beta_mkt_exc.append(model.params[0])
            beta_smb.append(model.params[1])
            beta_hml.append(model.params[2])
            p_mx.append(model.pvalues[0])
            p_smb.append(model.pvalues[1])
            p_hml.append(model.pvalues[2])
            r2.append(model.rsquared)
            delta_diag.append(squared_error)

        return (
            pd.Series(alphas,
                      index=self.__raw_data.tic.unique()).drop_duplicates(),
            pd.DataFrame(np.diagflat(delta_diag), columns=self.__raw_data.tic.unique(),
                         index=self.__raw_data.tic.unique()).drop_duplicates(),
            pd.DataFrame({'mkt_excess': beta_mkt_exc, 'smb': beta_smb, 'hml': beta_hml},
                         index=self.__raw_data.tic.unique()).drop_duplicates(),
            pd.DataFrame({'mkt_excess': p_mx, 'smb': p_smb, 'hml': p_hml},
                         index=self.__raw_data.tic.unique()).drop_duplicates()
        )

    def _generate_factor_covariance(self):
        f = self.__raw_data[self.__raw_data.tic == self.__raw_data.tic.mode()[0]][
            ['datadate', 'mkt_excess', 'smb', 'hml']
        ]
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

        V = np.dot(np.dot(self.factor_loadings.values, F), self.factor_loadings.values.T)
        return pd.DataFrame(V, columns=self.factor_loadings.index, index=self.factor_loadings.index)
