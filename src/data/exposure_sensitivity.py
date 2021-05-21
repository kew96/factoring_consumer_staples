from pathlib import Path

from src.features import fama_french

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data', 'processed', 'sensitivity_analysis', 'exposure')

tfm = fama_french.ThreeFactorModel()

for exposure in {-1, -0.5, 0, 0.5, 1}:
    return_data, weights_data = tfm.max_sharpe_portfolios(start_year=2005,
                                                          end_year=2020,
                                                          num_points=300,
                                                          min_variance=0,
                                                          max_variance=3,
                                                          universe_size=20,
                                                          exposure=exposure,
                                                          return_sharpe=True,
                                                          return_weights=True)

    return_data.to_csv(DATA_PATH.joinpath(f'performance_data_{exposure}.txt'), sep='\t')
    weights_data.to_csv(DATA_PATH.joinpath(f'weights_data_{exposure}.txt'), sep='\t')
