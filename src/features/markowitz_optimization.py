import pandas as pd
import numpy as np
import cvxpy as cp


class Markowitz:

    def __init__(self, df):
        self.expected_return = df.pop('expected_return')
        self.sigma = df

    def optimal_weights(self, max_variance):

        # Initiate variable for the weights to be optimized over
        weights = cp.Variable(len(self.expected_return))

        # Define the objective function for the expected return
        total_return = cp.matmul(self.expected_return.values, weights)

        # Define the variance of the portfolio given the weights
        variance = cp.quad_form(weights, self.sigma.values)

        # Define the total amount invested between all assets
        total_invested = cp.sum(weights)

        # Must go long the first 10 assets and short the last 10 assets
        long_short = [
            0 <= weights[:10] <= 1, # long
            -1 <= weights[10:] <= 0 # short
        ]

        # Aggregate constraints into one list
        constraints = long_short
        constraints.append(variance <= max_variance)
        constraints.append(total_invested == 1)

        portfolio_opt = cp.Problem(cp.Maximize(total_return, constraints=constraints))

        portfolio_opt.solve()

        return {'excess_return': total_return.value, 'weights': weights.value}

    def max_sharpe(self, num_points=300, *, min_variance=0, max_variance=3):

        all_returns = np.zeros(num_points)
        all_weights = np.array(num_points)
        all_variances = np.linspace(min_variance, max_variance, num_points)

        for ind, variance in enumerate(all_variances):
            # Get optimal weights and associated return given a portfolio variance
            result = self.optimal_weights(variance)

            all_returns[ind] = result['period_return']
            all_weights[ind] = result['weights']

        # Sharpe ratio defined as return divided by risk
        sharpe_ratio = all_returns / all_variances

        # Find the index of the max sharpe ratio portfolio
        max_sharpe_ind = np.argmax(sharpe_ratio)

        # Return the weights associated with the max sharpe ratio portfolio
        return all_weights[max_sharpe_ind]
