# Bayes Cointegration for Statistical Arbitrage

## Overview

This project explores applying Bayesian inference to statistical arbitrage, focusing on modelling cointegration relationships and mean-reversion dynamics between asset pairs. Using Bayesian methods enables a probabilistic treatment of key parameters like the cointegration coefficient, allowing more informed and robust trading signals.

The main notebook compares classical frequentist approaches to a Bayesian framework, showcasing improvements in signal stability and backtest performance.

## Features

- Frequentist vs Bayesian rolling beta estimation  
- Bayesian model implemented with PyMC for uncertainty quantification  
- Diagnostic charts: z-score, ADF p-values, half-life, volatility  
- Simple backtesting prototype to compare strategies  
- Modular helper functions for stats, transforms, and backtesting

## Motivation

Traditional cointegration methods provide point estimates, which can be noisy and lead to unstable signals. Bayesian inference treats parameters as distributions, incorporating prior beliefs and producing credible intervals. This can lead to smoother, more reliable signals and potentially better trading outcomes.

## Usage

1. Clone the repo  
2. Set up your environment (see requirements.txt)  
3. Run the `bayes_cointegration.ipynb` notebook to explore data, compare models, and run backtests  

## Future Work

- Extend Bayesian modelling to regime detection and mean reversion signals  
- Use posterior predictive distributions for trade decision-making  
- Empirically informed priors for better model calibration  
- Multi-pair portfolio management with Bayesian risk weighting

## Requirements

- Python 3.11+  
- PyMC  
- ArviZ  
- pandas, numpy, matplotlib, seaborn  
- statsmodels  
- tqdm  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
