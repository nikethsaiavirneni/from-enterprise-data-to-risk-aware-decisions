"""Forecast metrics suited to intermittent counts (no MAPE / sMAPE: ~60% of kept days are zero)."""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y, p):
    """Weighted absolute percentage error: sum|e| / sum|y| (defined when sum|y| > 0)."""
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def naive_scale(sales_by_series):
    """Per-series scale for RMSSE: mean squared one-step naive error on the training window.
    sales_by_series: iterable of 1-D arrays (training sales of one series, in time order)."""
    return np.array([np.mean(np.diff(s.astype(np.float64)) ** 2) if len(s) > 1 else np.nan
                     for s in sales_by_series])


def rmsse(y, p, sid, scale_by_sid):
    """Mean over series of sqrt(MSE_i / scale_i); series with scale 0 or NaN are skipped.
    scale_by_sid: array indexed by series id."""
    err = (y.astype(np.float64) - p) ** 2
    order = np.argsort(sid, kind="stable")
    sid_s, err_s = sid[order], err[order]
    uniq, start = np.unique(sid_s, return_index=True)
    mse = np.add.reduceat(err_s, start) / np.diff(np.append(start, len(sid_s)))
    sc = scale_by_sid[uniq]
    ok = np.isfinite(sc) & (sc > 0)
    return float(np.mean(np.sqrt(mse[ok] / sc[ok])))


def all_metrics(y, p, sid, scale_by_sid):
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "WAPE": wape(y, p),
            "RMSSE": rmsse(y, p, sid, scale_by_sid)}
