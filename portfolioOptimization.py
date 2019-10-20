import pandas as pd
import numpy as np
import scipy.optimize as sco


class Portfolio:
    def __init__(self, price_df, risk_free_rate=0.0):
        """
        Portfolio_Optimization
        -------------------------
        Input: 
        price_df: pandas DataFrame (date x asset)
        risk_free_rate: float 
        min_return: float      

        # Author: Jerry Lo
        # Ref: https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f   
        """
        self.num_assets = price_df.shape[1]
        self.price_df = price_df
        self.returns = price_df.pct_change()[1:]
        self.risk_free_rate = risk_free_rate
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

    def portfolio_annualised_performance(self, weights):
        weights = np.array(weights)

        daily_returns = np.dot(self.returns.fillna(0), weights)
        annual_abs_dev = np.mean(
            np.abs(daily_returns - np.mean(daily_returns))) * 252
        annual_returns = np.sum(self.mean_returns*weights) * 252
        annual_std = np.sqrt(np.dot(weights.T, np.dot(
            self.cov_matrix, weights))) * np.sqrt(252)
        min_return = np.min(np.dot(self.returns.fillna(0), weights))

        annual_returns = np.sum(self.mean_returns*weights) * 252
        return {'annaul_std': annual_std,
                'annual_returns': annual_returns,
                'annual_abs_dev': annual_abs_dev,
                'min_return': min_return,
                'annual_sharpe': (annual_returns-self.risk_free_rate)/annual_std
                }

    def _neg_sharpe_ratio(self, weights):
        return -self.portfolio_annualised_performance(weights)['annual_sharpe']

    def _var(self, weights):
        return self.portfolio_annualised_performance(weights)['annaul_std']**2

    def _abs_dev(self, weights):
        return self.portfolio_annualised_performance(weights)['annual_abs_dev']

    def _min_return(self, weights):
        return -self.portfolio_annualised_performance(weights)['min_return']

    def optimize(self, optimize_val, min_return=0.2):

        self.constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                            {'type': 'ineq', 'fun': lambda x: np.sum(self.mean_returns*x)*252 - min_return})
        self.bound = (0.0, 1.0)
        self.bounds = tuple(self.bound for asset in range(self.num_assets))

        optimize_func = {'max_sharpe': self._neg_sharpe_ratio,
                         'min_var': self._var,
                         'min_abs_dev': self._abs_dev,
                         'min_max': self._min_return}
        result = sco.minimize(optimize_func[optimize_val], self.num_assets*[1./self.num_assets, ],
                              method='SLSQP', bounds=self.bounds, constraints=self.constraints)

        allocation = result['x']
        self.result = self.portfolio_annualised_performance(
            np.array(allocation))
        self.result['optimize'] = optimize_val
        self.result['allocation'] = pd.DataFrame(
            allocation, index=self.returns.columns, columns=['allocation'])

        return self

    def optimize_summary(self):
        print("-"*80)
        print(self.result['optimize'] + " Allocation\n")
        print("Annualised Return:", round(self.result['annual_returns'], 2))
        print("Annualised Volatility:", round(self.result['annaul_std']**2, 2))
        print("Annualised Absolute Deviation:",
              round(self.result['annual_abs_dev'], 2))
        print("Minimax:", round(self.result['min_return'], 2))
        print("Annualised Sharpe Ratio:", round(
            self.result['annual_sharpe'], 2))
        print("\n")
        print(round(self.result['allocation'], 3).T)
