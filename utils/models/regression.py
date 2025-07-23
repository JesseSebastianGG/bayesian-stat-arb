# filepath: /Users/jesse/Documents/work/StatArbInvestigate/helpers/technical.py

from ..helpers.technical import compute_zscore

import statsmodels.api as sm
import pandas as pd
# import numpy as np
import pymc as pm
import arviz as az


def estimate_rolling_beta(X, Y, window):
    betas = []
    # alphas = []
    for i in range(window, len(X)):
        Y_window = Y.iloc[i-window:i]
        X_window = sm.add_constant(X.iloc[i-window:i])
        model = sm.OLS(Y_window, X_window).fit()
        # alphas.append(model.params.const)
        betas.append(model.params.iloc[1])
    
    return pd.Series(betas, index=X.index[window:])

def compute_rolling_spread_and_z(X, Y, beta_series, window):
    """Compute spread and z-score using rolling beta."""
    if beta_series is None:
        beta_series = estimate_rolling_beta(X, Y, window)
    else:
        beta_series = beta_series.squeeze()
    aligned_X = X.loc[beta_series.index]
    aligned_Y = Y.loc[beta_series.index]
    spread = aligned_Y - beta_series * aligned_X
    z = compute_zscore(spread, window)
    return spread, z


def estimate_rolling_residuals(returns_X, returns_Y, window):
    residuals = []
    for i in range(window, len(returns_X)):
        X_window = sm.add_constant(returns_X.iloc[i-window:i])
        Y_window = returns_Y.iloc[i-window:i]
        # fit model on past data
        model = sm.OLS(Y_window, X_window).fit()
        # Predict the next value
        X_ahead = pd.DataFrame(returns_X.iloc[i-window:i+1])
        X_ahead = sm.add_constant(X_ahead)  # Add constant for intercept
        predicted_Y = model.predict(X_ahead)  # 1-step ahead
        residual = returns_Y.iloc[i] - predicted_Y.iloc[0]
        residuals.append(residual)

    residuals_series = pd.Series(residuals, index=returns_X.index[window:])
    return residuals_series


def bayesian_regression(x, y, draws=2000, tune=1000, seed=42, production=False):
    """
    Perform Bayesian regression on rolling windows of X and Y.
    Prior choice improvements left for future.
    
    Parameters:
    - X: pd.Series or np.ndarray, independent variable
    - Y: pd.Series or np.ndarray, dependent variable
    - draws: from posterior distirbution (pymc var)
    - tune: num draws to wait for convergence (pymc var)
    - seed: random seed (pymc var)
    - production: if True then only returns beta values, else az summaries
    
    Returns:
    - if production=True (not default): beta_bayes: pd.Series, estimated coefficients
    for each window, else: trace, model (use az to viz)
    """
    with pm.Model() as model:
        # Priors (regression: alpha, beta (y=a+b*x+eps); y: sigma (y~N(mu, sigma)))
        alpha = pm.Normal(name='alpha', mu=0, sigma=1)  # quite informed, believe no outperformance
        beta = pm.Normal(name='beta', mu=0.5, sigma=1)   # ditto, believe positive cointegration
        sigma_y = pm.HalfNormal(name='sigma', sigma=1)

        # Expected value
        mu_y = alpha + beta * x

        # Likelihood - this Python variable not used, but since it takes
        # observed= pymc interprets this as the likelihood (needed for inference)
        y_likelihood = pm.Normal(name="y", mu=mu_y, sigma=sigma_y, observed=y)

        # Inference
        trace = pm.sample(draws=1000, tune=1000, chains=4, return_inferencedata=True,
                          progressbar=not production)

    # return trace, model
    return (trace, model) if not production else az.summary(
        trace, var_names=['beta'])['mean'].values[0]
