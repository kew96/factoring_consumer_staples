from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.api import WLS

np.seterr('raise')


class ThreeFactorLoadings:
    __DATA_PATH = Path.cwd().parent.parent.joinpath('data')

    def __init__(self, *, data=None):
        if not data:
            data = self.__DATA_PATH.joinpath('processed', 'three_factor_model.txt')
        self.__data = pd.read_table(data)
        self._generate_smb()
        self._generate_hml()
        self._generate_distinct_factors()
        self.alpha, self.factor_loading, self.p_values = self._generate_distinct_factors()

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
        self.__data = self.__data.merge(SMB, how='left', on='datadate')

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

            hml.append(factor)

        HML = pd.DataFrame({'datadate': self.__data.datadate.unique(), 'hml': hml})
        self.__data = self.__data.merge(HML, how='left', on='datadate')

    def _generate_distinct_factors(self):
        alphas = list()
        beta_mkt_exc = list();
        beta_smb = list();
        beta_hml = list()
        p_mx = list();
        p_smb = list();
        p_hml = list()
        r2 = list()

        for ticker in self.__data.tic.unique():
            temp_df = self.__data[self.__data.tic == ticker]

            y = temp_df.chng.values - temp_df.Price_tb.values
            X = temp_df[['mkt_excess', 'smb', 'hml']].values

            denom = sum(range(len(temp_df)))
            weights = [val / denom for val in range(1, len(temp_df) + 1)]

            model = WLS(y, X, weights=weights).fit()

            predictions = list()
            for row in X:
                predicted = 0
                for x_val, param_val in zip(row, model.params):
                    predicted += x_val * param_val
                predictions.append(predicted)

            alpha = 0
            for y_val, w, pred in zip(y, weights, predictions):
                alpha += (y - pred) * w

            alphas.append(alpha)
            beta_mkt_exc.append(model.params[0])
            beta_smb.append(model.params[1])
            beta_hml.append(model.params[2])
            p_mx.append(model.pvalues[0])
            p_smb.append(model.pvalues[1])
            p_hml.append(model.pvalues[2])
            r2.append(model.rsquared)

        return (pd.Series(alphas, index=self.__data.tic.unique()),
                pd.DataFrame({'mkt_excess': beta_mkt_exc, 'smb': beta_smb, 'hml': beta_hml}),
                pd.DataFrame({'mkt_excess': p_mx, 'smb': p_smb, 'hml': p_hml}))


# DATA_PATH = Path.cwd().parent.parent.joinpath('data')
# df = pd.read_table(DATA_PATH.joinpath('processed', 'three_factor_model.txt'))

# dat = {'datadate': [], 'SMB value': []}
# SMB = pd.DataFrame(data=dat)
# j = 0
# for i in np.unique(df["datadate"]):
# temp = df[df.datadate == i][["chng", "mkvaltq"]]
#
# ten = np.nanpercentile(temp["mkvaltq"], 10)
# ninety = np.nanpercentile(temp["mkvaltq"], 90)
# small = temp[temp["mkvaltq"] < ten]
# big = temp[temp["mkvaltq"] > ninety]
# small_rets = np.nanmean(small["chng"])
# # big_rets = np.nanmean(big["chng"])
# factor = small_rets - big_rets
# SMB.loc[j] = [i, factor]
# j += 1

# dat = {'datadate': [], 'HML value': []}
# HML = pd.DataFrame(data=dat)
# j = 0
# for i in np.unique(df["datadate"]):
#     temp = df[df.datadate == i][["tic", "chng", "ptb"]]
#
#     ten = np.nanpercentile(temp["ptb"], 10)
#     ninety = np.nanpercentile(temp["ptb"], 90)
#     value = temp[temp["ptb"] < ten]
#     growth = temp[temp["ptb"] > ninety]
#     val_rets = np.nanmean(value["chng"])
#     growth_rets = np.nanmean(growth["chng"])
#     factor = val_rets - growth_rets
#     HML.loc[j] = [i, factor]
#     j += 1

# df = df.merge(SMB, how="left", on="datadate")
# df = df.merge(HML, how="left", on="datadate")

##need to apply model to all stocks to get B
# factors = pd.DataFrame(columns=['Tic', 'Alpha', 'Beta_mkt_Exc', 'Beta_SMB', 'Beta_HML', 'p_MX', 'p_smb', 'p_hml', 'r2'])
# delta = np.zeros(shape=(len(np.unique(df["tic"])), len(np.unique(df["tic"]))))
# q = -1


# def int_sum(x):
#     sum1 = 0
#     for i in range(1, x + 1):
#         sum1 += i
#     return sum1
#
# for tic in np.unique(df["tic"]):
#     try:
#         q += 1
#         temp_df = df[df["tic"] == tic]
#         temp_df = temp_df.reset_index(drop=True)
#         temp_df = temp_df.dropna()
#         Y = temp_df["chng"] - (temp_df["Price_tb"])
#         X = temp_df[["mkt_excess", "SMB value", "HML value"]]
#         w = np.ones(temp_df.shape[0])
#         denom = int_sum(len(w))
#         for i in range(len(w)):
#             w[i] += i
#         w = w / denom
#         model = sm.WLS(Y, X, weights=w).fit()
#         alpha = 0
#         for i in range(len(weights)):
#             no = 0
#             for j in range(len(model.params)):
#                 no += X.iloc[i][j] * model.params[j]
#             yes = (Y.iloc[i] - no) * weights[i]
#             alf += yes
#
# e_i = []
# for i in range(len(weights)):
#     kyo = Y.iloc[i] - alf
#     predicted = 0
#     for j in range(len(model.params)):
#         predicted += X.iloc[i][j] * model.params[j]
#     kyo -= predicted
#     e_i.append(kyo)
#
# eii = 0
# for i in range(len(weights)):
#     eii += weights[i] * (e_i[i] ** 2)
# delta[q][q] = eii
#
#     factors.loc[q] = [tics, alf, model.params[0], model.params[1], model.params[2], model.pvalues[0],
#                       model.pvalues[1],
#                       model.pvalues[2], model.rsquared]
#
# except Exception as e:
#     print(e)
#     raise Exception

# B = factors[["Beta_mkt_Exc", "Beta_SMB", "Beta_HML"]]
# f = df[df["tic"] == (df.tic).mode()[0]].iloc[3:][["datadate", "mkt_excess", "SMB value", "HML value"]]
# w = np.zeros(len(f))
# denom = int_sum(len(w))
# for j in range(len(w)):
#     w[j] += i
# w = w / denom
# F = np.zeros(shape=(3, 3))
# for t in range(len(f)):
#
#     temp = np.zeros(3)
#     for s in range(len(f)):
#         temp = temp + w[s] * f.iloc[s][["mkt_excess", "SMB value", "HML value"]]
#
#     term1 = f.iloc[t][["mkt_excess", "SMB value", "HML value"]] - temp
#
#     mat1 = np.dot(np.reshape(np.array(term1), (3, 1)), np.transpose(np.reshape(np.array(term1), (3, 1))))
#     mat1 = mat1 * w[t]
#     F = F + mat1
#
# V = np.dot(np.dot(np.array(B), F), np.transpose(np.array(B)))
#
# print(V)


if __name__ == '__main__':
    tfl = ThreeFactorLoadings()
