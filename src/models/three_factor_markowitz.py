import pandas as pd

import markowitz_optimization
from src.features import MacroModel

class ThreeFactorMarkowitz(markowitz_optimization.Markowitz):

    def __init__(self, data, alphas, loadings, sigma):
        self.__data = data
        self.alphas = alphas
        self.loadings = loadings
        self._expected_return = self._expected_asset_return()
        super().__init__(self._expected_return, sigma)

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

        return pd.DataFrame({'tic': self.__data.tic, 'datadate': self.__data.datadate, 'expected_return': mu})

if __name__ == '__main__':
    tfl = MacroModel.ThreeFactorLoadings()
    tfm = ThreeFactorMarkowitz(tfl.data, tfl.alpha, tfl.factor_loading, tfl.Sigma)
    print(tfm.max_sharpe())
