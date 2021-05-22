from pathlib import Path

from src.features import fama_french

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data', 'processed')

tfm = fama_french.ThreeFactorModel()

return_data, weights_data = tfm.max_sharpe_portfolios(start_year=2005,
                                                      end_year=2020,
                                                      num_points=300,
                                                      min_variance=0,
                                                      max_variance=3,
                                                      universe_size=20,
                                                      return_sharpe=True,
                                                      return_weights=True)

return_data.to_csv(DATA_PATH.joinpath('modified_alpha_performance_data.txt'), sep='\t')
weights_data.to_csv(DATA_PATH.joinpath('modified_alpha_weights_data.txt'), sep='\t')
