from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = Path().cwd().parent.parent.joinpath('data')


book_ratios = pd.read_csv(DATA_PATH.joinpath('raw', 'book_ratios.csv'))