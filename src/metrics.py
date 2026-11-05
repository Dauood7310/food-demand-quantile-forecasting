# Error measures. All of them take the actual orders and the forecast orders
# on the ORIGINAL scale (number of orders, not log).

import numpy as np


def rmse(y, pred):
    return np.sqrt(np.mean((y - pred) ** 2))


def mae(y, pred):
    return np.mean(np.abs(y - pred))


def rmsle(y, pred):
    # same as RMSE but on log(1 + x), used by Panda and Mohanty
    return np.sqrt(np.mean((np.log1p(pred) - np.log1p(y)) ** 2))


def wape(y, pred):
    # total absolute error divided by total demand, in %
    return np.sum(np.abs(y - pred)) / np.sum(y) * 100


def pinball(y, pred, tau):
    # quantile loss: under-forecasts cost tau, over-forecasts cost (1 - tau)
    diff = y - pred
    return np.mean(np.maximum(tau * diff, (tau - 1) * diff))


def bias(y, pred):
    # positive = forecasts are too high on average
    return np.mean(pred - y)


def over_forecast_rate(y, pred):
    # share of weeks (in %) where the forecast is above the actual orders
    return np.mean(pred > y) * 100


def all_metrics(y, pred):
    """All point-forecast measures in one dictionary (one row of the results table)."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        'RMSE': rmse(y, pred),
        'MAE': mae(y, pred),
        'RMSLE': rmsle(y, pred),
        'WAPE': wape(y, pred),
        'Bias': bias(y, pred),
        'Over %': over_forecast_rate(y, pred),
    }


def diebold_mariano(loss_a, loss_b):
    """Diebold-Mariano test: is model A's loss different from model B's?

    loss_a, loss_b: the loss of each model in each test week (same weeks, same order).
    Returns the test statistic and the two-sided p-value.
    Negative statistic = model A has lower loss. p < 0.05 = the difference is significant.
    (simple version: assumes the weekly loss differences are not autocorrelated)
    """
    from scipy import stats
    d = np.asarray(loss_a) - np.asarray(loss_b)
    dm = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm)))
    return dm, p_value
