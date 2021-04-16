Factoring Consumer Staples
==============================

This project aims to apply the Fama-French Three-Factor Model to Consumer Staples ETFs while exploring alternative 
factors and rankings metrics.

Project Organization
------------

    ├── README.md          <- The top-level README for this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Utilizing models from src.
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── environment   <- The requirements file for reproducing the analysis environment, e.g.
    │         │             generated with `pip freeze > requirements.txt`
    │         ├── requirements.txt
    │         └── environement.yml
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download, generate, and manipulate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts and classes to aid in modeling outside of actual models
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts and classes used to train, predict, and optimize
    │   │   │                           over data
    │   │   └── markowitz_optimization.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    └──       └── visualize.py


--------
