import numpy as np
import pandas as pd

def z_to_positions(z, short_entry_z=1.2, short_exit=0,
                  long_enter=-1.2, long_exit=0):
    positions = pd.Series(index=z.index, dtype='int', data=0, name='positions')
    current = 0
    for dt in positions.index:
        # entry conditions
        if current == 0:
            if z[dt] > short_entry_z:
                current = -1  # enter short
            elif z[dt] < long_enter:
                current = 1  # enter long
        # exit conditions
        elif current == 1 and z[dt] <= short_exit:
                current = 0
        elif current == -1 and z[dt] >= long_exit:
                current = 0
        positions[dt] = current
    return positions


def get_fracdiff_weights(d, max_lags=1000, tol=1e-5):
    w = [1.0]
    for k in range(1, max_lags):
        w_k = -w[-1] * (d - (k - 1)) / k
        if abs(w_k) < tol:
            break
        w.append(w_k)
    return np.array(w)


def fracdiff(series, d, tol=1e-5, max_lags=1000):
    w = get_fracdiff_weights(d, max_lags, tol)
    width = len(w)
    
    diff_series = pd.Series(index=series.index, dtype='float64')

    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1]
        if window.isnull().any():
            continue
        diff_series.iloc[i] = np.dot(w[::-1], window)
        
    return diff_series


def compute_zscore(series, window):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    zscore = (series - rolling_mean) / rolling_std
    return zscore
