# tools for extracting spread features

import numpy as np
import statsmodels.api as sm


def get_current_df_zscore(series, rolling_window=None):
    if rolling_window is not None:
        # this windows definitely could be optimized or sth
        spread_mean = series.rolling(
            center=False, window=rolling_window).mean()
        spread_std = series.rolling(center=False, window=rolling_window).std()
        z_score_series = (series-spread_mean)/spread_std
    else:
        z_score_series = (series-series.mean())/series.std()

    return z_score_series.iloc[-1], z_score_series


def half_life(spread):
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1], 0))
    if halflife <= 1:
        halflife = 1
    return halflife
