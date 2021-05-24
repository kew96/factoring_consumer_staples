#!/bin/bash

python3 clean_holdings.py
python3 current_consumer_staples.py
python3 get_all_tickers.py
python3 clean_book_ratios.py
python3 clean_market_values.py
python3 clean_prices_returns.py
python3 concat_data_3_factor.py
python3 three_factor_data.py
python3 basket_sensitivity.py
python3 exposure_sensitivity.py