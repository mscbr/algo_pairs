# tools for calculating backtest metrics

import numpy as np


def calculate_max_dd(cumret):
    # ======================================================
    # calculation of maximum drawdown and maximum drawdown duration based on
    # cumulative COMPOUNDED returns. cumret must be a compounded cumulative return.
    # i is the index of the day with maxDD.
    # ======================================================
    highwatermark = np.zeros(cumret.shape)
    drawdown = np.zeros(cumret.shape)
    drawdown_duration = np.zeros(cumret.shape)
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t] = np.maximum(highwatermark[t-1],
                                      cumret[t])
        drawdown[t] = (1+cumret[t])/(1+highwatermark[t])-1
        if drawdown[t] == 0:
            drawdown_duration[t] = 0
        else:
            drawdown_duration[t] = drawdown_duration[t-1]+1
    max_dd, i = np.min(drawdown), np.argmin(drawdown)
    # drawdown < 0 always
    max_ddd = np.max(drawdown_duration)
    return max_dd, max_ddd, i


def calculate_annualized_mean(tp_year, series):
    return series.mean() * tp_year


def calculate_annualized_std(tp_year, series):
    return series.std() * np.sqrt(tp_year)
