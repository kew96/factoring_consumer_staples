from itertools import permutations

import pandas as pd

import markowitz_optimization
from src.features import MacroModel

class ThreeFactorMarkowitz(markowitz_optimization.Markowitz):

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
    tfl = MacroModel.ThreeFactorLoadings()
    tfm = ThreeFactorMarkowitz(tfl.data, tfl.alpha, tfl.factor_loading, tfl.Sigma)
    print(tfm.max_sharpe_portfolios(2000, 2002))
