from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np

def rolling_adf(series, window=120, step=5):
    """
    Apply ADF test on rolling windows of `series`.
    Returns a DataFrame of (timestamp, adf_stat, p_value).
    """
    results = []
    for i in range(0, len(series) - window + 1, step):
        window_data = series[i:i+window]
        result = adfuller(window_data)
        adf_stat, p_value = result[0], result[1]
        timestamp = series.index[i+window-1]
        results.append((timestamp, adf_stat, p_value))
    output = pd.DataFrame(results, columns=['timestamp', 'adf_stat', 'p_value'])
    output = output.set_index('timestamp')
    return output


def rolling_half_life(series, window=120, step=1):  # step=5
    results = []
    for i in range(0, len(series) - window + 1, step):
        y = series[i:i+window]
        y_lag = y.shift(1).dropna()
        y_ret = y.diff().dropna()
        y_lag, y_ret = y_lag.align(y_ret, join='inner')

        if len(y_lag) < 10 or y_lag.std() < 1e-5:  # avoid near-flat series
            results.append((y.index[-1], np.nan))
            continue

        beta = np.polyfit(y_lag, y_ret, 1)[0]

        if abs(beta) < 1e-6:  # prevent division explosion
            halflife = np.nan
        else:
            halflife = -np.log(2) / beta if beta < 0 else np.nan

        results.append((y.index[-1], halflife))

    return pd.DataFrame(results, columns=['timestamp', 'half_life']).set_index('timestamp')

