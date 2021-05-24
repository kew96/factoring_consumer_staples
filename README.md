Factoring Consumer Staples
==============================

In this project, we attempt to determine if the Fama-French 3-Factor Model can be used with Markowitz Portfolio
Optimization on the Consumer Staples sector. The Fama-French model is typically applied to the broader market, and we
wanted to explore Consumer Staples because of the market crash due to the global pandemic. We implemented a Weighted
Least Squares regressor to calculate assets' factor loadings and Markowitz Optimization for selecting maximum Sharpe
Ratio portfolios. While this framework does achieve positive excess returns over the past fifteen years, the volatility
makes it difficult to add to any portfolio. Sensitivity analysis did provide some insight into how parameters affect the
model but nothing at any significance value. Achieving a positive excess return gives us hope for this framework but
necessitates future research into new factors and different techniques for calculating factor loadings.

- Kyle Walsh: [kew96@cornell.edu](mailto:kew96@cornell.edu?Subject=Factoring%20Consumer%20Staples%20Project)
- Siddharth Kantamneni: [skk82@cornell.edu](mailto:skk82@cornell.edu?Subject=Factoring%20Consumer%20Staples%20Project)

Project Organization
------------

    ├── README.md          <- The top-level README for this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Referenced papers
    │
    ├── reports
    │   └── figures        <- Generated graphics and figures to be used in report
    │
    ├── environment   <- Files for recreating environment in pip or conda
    │         ├── requirements.txt
    │         └── environement.yml
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download, generate, and manipulate data
    │   │   └── gen_data.sh <- Runs all python scripts in correct order (will take a LONG time)
    │   │
    │   ├── features       <- Scripts and classes to aid in modeling outside of actual models
    │   │   └── fama_french.py <- Class that calculates factor loadings and performs optimization
    │   │
    │   ├── models         <- Scripts and classes used for model optimization
    │   │   └── markowitz_optimization.py <- The base classes for Markowitz optimization
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    └──       └── gen_visualizations <- Creates all visualizations used in report

--------
