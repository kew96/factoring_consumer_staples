from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

np.seterr('raise')

class ThreeFactorLoadings:

    def __init__(self, *, data=None):
        ...


DATA_PATH = Path.cwd().parent.parent.joinpath('data')
df = pd.read_table(DATA_PATH.joinpath('processed', 'three_factor_model.txt'))

dat = {'datadate': [], 'SMB value': []}
SMB = pd.DataFrame(data=dat)
j = 0
for i in np.unique(df["datadate"]):
    temp = df[df.datadate == i][["chng", "mkvaltq"]]

    ten = np.nanpercentile(temp["mkvaltq"], 10)
    ninety = np.nanpercentile(temp["mkvaltq"], 90)
    small = temp[temp["mkvaltq"] < ten]
    big = temp[temp["mkvaltq"] > ninety]
    small_rets = np.nanmean(small["chng"])
    big_rets = np.nanmean(big["chng"])
    factor = small_rets - big_rets
    SMB.loc[j] = [i, factor]
    j += 1

dat = {'datadate': [], 'HML value': []}
HML = pd.DataFrame(data=dat)
j = 0
for i in np.unique(df["datadate"]):
    temp = df[df.datadate == i][["tic", "chng", "ptb"]]

    ten = np.nanpercentile(temp["ptb"], 10)
    ninety = np.nanpercentile(temp["ptb"], 90)
    value = temp[temp["ptb"] < ten]
    growth = temp[temp["ptb"] > ninety]
    val_rets = np.nanmean(value["chng"])
    growth_rets = np.nanmean(growth["chng"])
    factor = val_rets - growth_rets
    HML.loc[j] = [i, factor]
    j += 1

df = df.merge(SMB, how="left", on="datadate")
df = df.merge(HML, how="left", on="datadate")

##need to apply model to all stocks to get B
factors = pd.DataFrame(columns=['Tic', 'Alpha', 'Beta_mkt_Exc', 'Beta_SMB', 'Beta_HML', 'p_MX', 'p_smb', 'p_hml', 'r2'])
delta = np.zeros(shape=(len(np.unique(df["tic"])), len(np.unique(df["tic"]))))
q = -1


def int_sum(x):
    sum1 = 0
    for i in range(1, x + 1):
        sum1 += i
    return sum1

for tic in np.unique(df["tic"]):
    try:
        q += 1
        temp_df = df[df["tic"] == tic]
        temp_df = temp_df.reset_index(drop=True)
        temp_df = temp_df.dropna()
        Y = temp_df["chng"] - (temp_df["Price_tb"])
        X = temp_df[["mkt_excess", "SMB value", "HML value"]]
        w = np.ones(temp_df.shape[0])
        denom = int_sum(len(w))
        for i in range(len(w)):
            w[i] += i
        w = w / denom
        model = sm.WLS(Y, X, weights=w).fit()
        alf = 0
        for i in range(len(w)):
            no = 0
            for j in range(len(model.params)):
                no += X.iloc[i][j] * model.params[j]
            yes = (Y.iloc[i] - no) * w[i]
            alf += yes

        e_i = []
        for i in range(len(w)):
            kyo = Y.iloc[i] - alf
            no = 0
            for j in range(len(model.params)):
                no += X.iloc[i][j] * model.params[j]
            kyo -= no
            e_i.append(kyo)

        eii = 0
        for i in range(len(w)):
            eii += w[i] * (e_i[i] ** 2)
        delta[q][q] = eii

        factors.loc[q] = [tic, alf, model.params[0], model.params[1], model.params[2], model.pvalues[0],
                          model.pvalues[1],
                          model.pvalues[2], model.rsquared]

    except Exception as e:
        print(e)
        raise Exception

B = factors[["Beta_mkt_Exc", "Beta_SMB", "Beta_HML"]]
f = df[df["tic"] == (df.tic).mode()[0]].iloc[3:][["datadate", "mkt_excess", "SMB value", "HML value"]]
w = np.zeros(len(f))
denom = int_sum(len(w))
for j in range(len(w)):
    w[j] += i
w = w / denom
F = np.zeros(shape=(3, 3))
for t in range(len(f)):

    temp = np.zeros(3)
    for s in range(len(f)):
        temp = temp + w[s] * f.iloc[s][["mkt_excess", "SMB value", "HML value"]]

    term1 = f.iloc[t][["mkt_excess", "SMB value", "HML value"]] - temp

    mat1 = np.dot(np.reshape(np.array(term1), (3, 1)), np.transpose(np.reshape(np.array(term1), (3, 1))))
    mat1 = mat1 * w[t]
    F = F + mat1

V = np.dot(np.dot(np.array(B), F), np.transpose(np.array(B)))

print(V)
