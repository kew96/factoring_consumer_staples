from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.api import WLS

from src.models.markowitz_optimization import ThreeFactorMarkowitz


class ThreeFactorModel(ThreeFactorMarkowitz):
    __DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data')

    def __init__(self, *, data=None):
        if not data:
            data = self.__DATA_PATH.joinpath('processed', 'three_factor_model.txt')
        self.__data = pd.read_table(data, parse_dates=['datadate'])
        pd.options.display.max_columns = None
        print(self.__data.head())
        self._generate_smb()
        self._generate_hml()
        self._generate_distinct_factors()
        self.alphas, self.delta, self.factor_loadings, self.p_values = self._generate_distinct_factors()
        self.sigma = self._generate_factor_covariance()
        super().__init__(self.__data, self.alphas, self.factor_loadings, self.sigma)

    @property
    def data(self):
        return self.__data

    def _generate_smb(self):
        smb = list()
        for date in self.__data.datadate.unique():
            temp = self.__data[self.__data.datadate == date][['chng', 'mkvaltq']]

            ten = self.__data.mkvaltq.quantile(0.1)
            ninety = self.__data.mkvaltq.quantile(0.9)

            small = temp[temp.mkvaltq < ten]
            big = temp[temp.mkvaltq > ninety]
            small_rets = small.chng.mean()
            big_rets = big.chng.mean()

            factor = small_rets - big_rets

            smb.append(factor)

        SMB = pd.DataFrame({'datadate': self.__data.datadate.unique(), 'smb': smb})
        self.__data = self.__data.merge(SMB, how='left', on='datadate').drop_duplicates()

    def _generate_hml(self):
        hml = list()
        for date in self.__data.datadate.unique():
            temp = self.__data[self.__data.datadate == date][['chng', 'ptb']]

            ten = self.__data.ptb.quantile(0.1)
            ninety = self.__data.ptb.quantile(0.9)

            value = temp[temp.ptb < ten]
            growth = temp[temp.ptb > ninety]
            val_rets = value.chng.mean()
            growth_rets = growth.chng.mean()

            factor = val_rets - growth_rets
            if np.isnan(factor):
                pd.options.display.max_rows = None
                print(temp)
                raise Exception

            hml.append(factor)

        HML = pd.DataFrame({'datadate': self.__data.datadate.unique(), 'hml': hml})
        self.__data = self.__data.merge(HML, how='left', on='datadate').drop_duplicates()

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

        for ticker in self.__data.tic.unique():
            temp_df = self.__data[self.__data.tic == ticker]

            y = temp_df.chng.values - temp_df.Price_tb.values
            X = temp_df[['mkt_excess', 'smb', 'hml']].values

            denom = sum(range(len(temp_df)))
            weights = [val / denom for val in range(1, len(temp_df) + 1)]

            print(y)

            print(X)

            model = WLS(y, X, weights=weights).fit()

            predictions = list()
            for row in X:
                predicted = 0
                for x_val, param_val in zip(row, model.params):
                    predicted += x_val * param_val
                predictions.append(predicted)

            alpha = 0
            for y_val, w, pred in zip(y, weights, predictions):
                alpha += y_val - pred

            errors = list()
            for pred, y_val in zip(predictions, y):
                error = y_val - alpha - pred
                errors.append(error)

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
                      index=self.__data.tic.unique()).drop_duplicates(),
            pd.DataFrame(np.diagflat(delta_diag), columns=self.__data.tic.unique(),
                         index=self.__data.tic.unique()).drop_duplicates(),
            pd.DataFrame({'mkt_excess': beta_mkt_exc, 'smb': beta_smb, 'hml': beta_hml},
                         index=self.__data.tic.unique()).drop_duplicates(),
            pd.DataFrame({'mkt_excess': p_mx, 'smb': p_smb, 'hml': p_hml},
                         index=self.__data.tic.unique()).drop_duplicates()
        )

    def _generate_factor_covariance(self):
        f = self.__data[self.__data.tic==self.__data.tic.mode()[0]][['datadate', 'mkt_excess', 'smb', 'hml']]
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

if __name__ == '__main__':
    tfm = ThreeFactorModel()
    pd.options.display.max_rows = None
    print(tfm.max_sharpe_portfolios())
