import numpy as np
import pandas as pd
from ..helpers.technical import z_to_positions, fracdiff
from ..helpers.stats import rolling_adf, rolling_half_life
from ..models.regression import estimate_rolling_beta, compute_rolling_spread_and_z
pd.set_option('future.no_silent_downcasting', True)
import matplotlib.pyplot as plt
plt.style.use('dark_background')

def compute_good_windows(
    X,
    Y,
    beta_z_spread_window=60,
    adf_window=120,
    hl_window=120,
    z_window=60,
    adf_threshold=0.05,
    min_half_life=2,
    max_half_life=30,
    max_z_std=2.0,
    step=1
):
    """
    Eg X = log_prices['AAPL'], Y = log_prices['MSFT'].
    Compute good windows for pairs trading based on ADF test, half-life, and z-score
    return s a boolean Series indicating good windows (mask Positions df).
    """

    # 1. Rolling beta, spread and z-score
    beta_series = estimate_rolling_beta(X, Y, beta_z_spread_window)
    spread, z = compute_rolling_spread_and_z(X, Y, beta_series, beta_z_spread_window)

    # 2. Rolling diagnostics
    adf_df = rolling_adf(spread.dropna(), window=adf_window, step=step)
    hl_df = rolling_half_life(z.dropna(), window=hl_window, step=step)
    z_std = z.rolling(z_window).std().dropna()

    # 3. Combine diagnostics into a DataFrame
    common_index = adf_df.index.intersection(hl_df.index).intersection(z_std.index)
    combined_df = pd.concat([
                        adf_df.loc[common_index, 'p_value'],
                        hl_df.loc[common_index, 'half_life'],
                        z_std.loc[common_index]
                        ], axis=1)  # type: ignore

    combined_df.columns = ['adf_p_value', 'half_life', 'z_std'] 

    good_window = (
        (combined_df['adf_p_value'] < adf_threshold) &
        (combined_df['half_life'] > min_half_life) & (combined_df['half_life'] < max_half_life) &
        (combined_df['z_std'] < max_z_std)
    )

    return good_window  # pd.Series of 0s and 1s or True/False


def backtest(prices,
             window = 60,
            ticker_AAPL = 'AAPL',
            ticker_MSFT = 'MSFT',
            use_log_prx=False,
            transaction_cost = 0.001,  # 10 bps per trade,
            use_rtns = True,  # use returns instead of prices for backtest (default: True, use returns
            beta_z_spread_window=60,
            rolling_beta=None,
            adf_window=120,
            hl_window=120,
            z_window=60,
            adf_threshold=0.05,
            min_half_life=2,
            max_half_life=30,
            max_z_std=2.0,
            step=1,
            regime_filter=True  # if True, apply regime filter
    ):
    
    returns = prices.pct_change().dropna()
    if use_log_prx:
        log_prices = np.log(prices)
        X = log_prices[ticker_AAPL]
        Y = log_prices[ticker_MSFT]
    else:
        X = prices[ticker_AAPL]
        Y = prices[ticker_MSFT]

    if use_rtns:
        X = returns[ticker_AAPL]
        Y = returns[ticker_MSFT]

    if rolling_beta is None:
        rolling_beta = estimate_rolling_beta(X, Y, window)
    else:
        rolling_beta = rolling_beta.squeeze()
    rolling_beta = rolling_beta.reindex(returns.index, method='ffill').infer_objects(copy=False)
    
    spread = Y - rolling_beta * X
    spread = spread.dropna()

    z = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    z = z.dropna()
    positions = z_to_positions(z)

    if regime_filter:
        # apply regime filter
        good_window = compute_good_windows(X=X, Y=Y, beta_z_spread_window=beta_z_spread_window,
                                           adf_window=adf_window, hl_window=hl_window,
                                           z_window=z_window, adf_threshold=adf_threshold,
                                           min_half_life=min_half_life, max_half_life=max_half_life,
                                           max_z_std=max_z_std, step=step).reindex(
                                               positions.index, method='ffill').fillna(False).infer_objects(copy=False)
        positions = positions * good_window
        positions = positions.fillna(0).infer_objects(copy=False)  # fill NaNs with 0 positions

    aligned_returns_X = returns.loc[z.index, ticker_AAPL]
    aligned_returns_Y = returns.loc[z.index, ticker_MSFT]
    aligned_beta = rolling_beta.loc[z.index]

    strategy_returns = positions.shift(1) * (
        aligned_returns_Y - aligned_beta * aligned_returns_X
        )
    cum_strategy = (1 + strategy_returns).cumprod()  # .values[-1]

    # trans costs
    trade_signals = positions.diff().abs()

    # Cost applies on previous day’s returns because trade is executed at open next day
    costs = trade_signals.shift(1).fillna(0).infer_objects(copy=False) * transaction_cost

    strategy_returns_tc = strategy_returns - costs

    cum_strategy_tc = (1 + strategy_returns_tc).cumprod()

    return cum_strategy, cum_strategy_tc


def backtest_frac_diff(prices,
                       ticker_AAPL = 'AAPL',
                        ticker_MSFT = 'MSFT',
                        window = 60,
                        transaction_cost = 0.001,  # 10 bps transaction cost
                        d=0.35,
                        # mask arguments
                        beta_z_spread_window=60,
                        adf_window=120,
                        hl_window=120,
                        z_window=60,
                        adf_threshold=0.05,
                        min_half_life=2,
                        max_half_life=30,
                        max_z_std=2.0,
                        step=1,
                        regime_filter=True  # if True, apply regime filter
                    ):

    log_prices = np.log(prices)
    fdiff_X = fracdiff(log_prices[ticker_AAPL], d).dropna()
    fdiff_Y = fracdiff(log_prices[ticker_MSFT], d).dropna()


    #fdiff_X = fracdiff(prices[ticker_AAPL], d).dropna()
    #fdiff_Y = fracdiff(prices[ticker_MSFT], d).dropna()

    X = fdiff_X
    Y = fdiff_Y
    common_index = X.index.intersection(Y.index)  # type: ignore
    X = X.loc[common_index]
    Y = Y.loc[common_index]

    rolling_beta = estimate_rolling_beta(X, Y, window)
    rolling_beta = rolling_beta.reindex(X.index, method='ffill').infer_objects(copy=False)

    spread = Y - rolling_beta * X
    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    z = (spread - spread_mean) / spread_std

    positions = z_to_positions(z)
    if regime_filter:
        # filter by regime
        good_window = compute_good_windows(X=X, Y=Y,
                                           beta_z_spread_window=beta_z_spread_window,
                                           adf_window=adf_window, hl_window=hl_window,
                                           z_window=z_window, adf_threshold=adf_threshold,
                                           min_half_life=min_half_life, max_half_life=max_half_life,
                                           max_z_std=max_z_std, step=step).reindex(
                                               positions.index, method='ffill').fillna(False)\
                                                .infer_objects(copy=False)
        positions = positions * good_window
        positions = positions.fillna(0).infer_objects(copy=False)  # fill NaNs with 0 positions

    returns = prices[[ticker_AAPL, ticker_MSFT]].pct_change().dropna()
    returns = returns.loc[z.index]  # align to z-score series

    strategy_returns = - positions.shift(1) * (
        returns[ticker_MSFT] - rolling_beta * returns[ticker_AAPL])


    trade_signals = positions.diff().abs()
    costs = trade_signals.shift(1).fillna(0).infer_objects(copy=False) * 2 * transaction_cost
    strategy_returns_tc = strategy_returns - costs
    cum_strategy = (1 + strategy_returns).cumprod()
    cum_strategy_tc = (1 + strategy_returns_tc).cumprod()

    return cum_strategy, cum_strategy_tc
