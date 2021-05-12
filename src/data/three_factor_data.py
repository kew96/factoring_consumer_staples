from pathlib import Path

from src.features import fama_french

DATA_PATH = Path(__file__).parent.parent.parent.joinpath('data', 'processed')

tfm = fama_french.ThreeFactorModel()

model_data = tfm.max_sharpe_portfolios(start_year=2005,
                                       end_year=2020,
                                       num_points=1000,
                                       min_variance=0,
                                       max_variance=5,
                                       universe_size=20,
                                       return_sharpe=True)

model_data.to_csv(DATA_PATH.joinpath('performance_data.txt'), sep='\t')
