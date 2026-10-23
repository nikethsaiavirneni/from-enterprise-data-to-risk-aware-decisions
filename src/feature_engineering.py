"""Phase 4: leakage-safe features for a direct forecast h days ahead (h = 1 and h = 28).

A row is (series, day t); the target is sales on day t. Every sales-derived feature uses
only days <= t - h. Price and calendar features describe day t itself and are treated as
known-in-advance covariates (planned price, calendar, events, SNAP) - an assumption that
is stated in the paper. Run:  python src/feature_engineering.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

HORIZONS = (1, 28)
LAGS = {1: (1, 7, 14, 28), 28: (28, 35, 42, 56)}
WARMUP = 60  # drop days <= WARMUP so every lag/rolling window is fully observed


def shift(a, k):
    """out[:, t] = a[:, t-k]; NaN where t < k."""
    out = np.full(a.shape, np.nan, dtype=np.float64)
    out[:, k:] = a[:, :-k]
    return out


def rolling(z, w, kind):
    """Trailing window of length w ending at column t over z (full window required)."""
    zz = np.nan_to_num(z, nan=0.0)
    ok = (~np.isnan(z)).astype(np.float64)
    cs, co = np.cumsum(zz, axis=1), np.cumsum(ok, axis=1)
    pad = np.zeros((z.shape[0], 1))
    cs, co = np.hstack([pad, cs]), np.hstack([pad, co])
    n = z.shape[1]
    hi = np.arange(1, n + 1)
    lo = np.maximum(hi - w, 0)
    s, c = cs[:, hi] - cs[:, lo], co[:, hi] - co[:, lo]
    full = c == w
    mean = np.where(full, s / w, np.nan)
    if kind == "mean":
        return mean
    cs2 = np.hstack([pad, np.cumsum(zz ** 2, axis=1)])
    s2 = cs2[:, hi] - cs2[:, lo]
    var = np.maximum(s2 / w - mean ** 2, 0.0)
    return np.sqrt(var)


def sales_features(y, h):
    """Sales-derived features for horizon h. y: (S, D) daily sales. Uses days <= t-h only."""
    z = shift(y.astype(np.float64), h)  # z[:, t] = y[:, t-h]
    f = {f"lag_{k}": shift(y.astype(np.float64), k) for k in LAGS[h]}
    f["roll_mean_7"] = rolling(z, 7, "mean")
    f["roll_mean_28"] = rolling(z, 28, "mean")
    f["roll_std_28"] = rolling(z, 28, "std")
    f["roll_zero_28"] = rolling((z == 0).astype(np.float64) + 0 * z, 28, "mean")
    return f


def price_features(price, h):
    """price: (S, D), NaN when not listed. Day-t price and its context."""
    ok = ~np.isnan(price)
    f = {"price": price.astype(np.float64)}
    f["price_ratio_7"] = price / shift(price, 7)
    cs = np.cumsum(np.nan_to_num(price, nan=0.0), axis=1)
    cn = np.cumsum(ok, axis=1)
    hist_mean = shift(cs / np.maximum(cn, 1), h)  # mean listed price up to t-h
    f["price_rel_hist"] = price / np.where(hist_mean > 0, hist_mean, np.nan)
    f["listed_days"] = cn.astype(np.float64)       # days listed so far (known at t)
    return f


def calendar_features(cal, state_idx, n_series):
    """Per-day calendar columns broadcast to (S, D). cal has D rows."""
    D = len(cal)
    evt = cal.event_type_1.astype("category").cat.codes.to_numpy() + 1  # 0 = no event
    dt = pd.to_datetime(cal.date)
    day = {
        "wday": cal.wday.to_numpy(), "month": cal.month.to_numpy(),
        "day_of_month": dt.dt.day.to_numpy(),
        "week_of_year": dt.dt.isocalendar().week.to_numpy().astype(int),
        "event_type": evt, "is_event": cal.event_name_1.notna().to_numpy().astype(int),
    }
    f = {k: np.broadcast_to(v[None, :], (n_series, D)).astype(np.float64) for k, v in day.items()}
    snap = np.stack([cal.snap_CA.to_numpy(), cal.snap_TX.to_numpy(), cal.snap_WI.to_numpy()])
    f["snap"] = snap[state_idx].astype(np.float64)
    return f


DESCRIPTIONS = {
    "lag": "sales k days before the target day (k >= h)",
    "roll_mean_7": "mean sales over the 7 days ending h days before the target",
    "roll_mean_28": "mean sales over the 28 days ending h days before the target",
    "roll_std_28": "std of sales over the same 28 days",
    "roll_zero_28": "share of zero-sales days in the same 28 days (intermittency)",
    "price": "listed price on the target day (known-in-advance covariate)",
    "price_ratio_7": "price / price 7 days earlier (promotion / price change)",
    "price_rel_hist": "price / mean listed price up to t-h (relative discount)",
    "listed_days": "number of days the item has had a listed price up to the target day",
    "wday": "weekday", "month": "month", "day_of_month": "day of month",
    "week_of_year": "ISO week", "event_type": "event type code (0 = none)",
    "is_event": "event flag", "snap": "SNAP benefit day flag for the store's state",
    "dept": "department code", "cat": "category code", "store": "store code",
    "state": "state code",
}


def main():
    C.FEATURES.mkdir(parents=True, exist_ok=True)
    long = pd.read_parquet(C.PROCESSED / "long.parquet")
    meta = pd.read_parquet(C.PROCESSED / "series_meta.parquet").reset_index(drop=True)
    cal = pd.read_parquet(C.PROCESSED / "calendar.parquet")
    D, S = len(cal), len(meta)
    local = pd.Series(np.arange(S), index=meta.sid.to_numpy())

    y = np.zeros((S, D), dtype=np.int16)
    price = np.full((S, D), np.nan)
    li = local.reindex(long.sid.to_numpy()).to_numpy()
    y[li, long.d.to_numpy() - 1] = long.sales.to_numpy()
    price[li, long.d.to_numpy() - 1] = long.price.to_numpy()

    codes = {c: meta[c].astype("category").cat.codes.to_numpy() for c in
             ("dept_id", "cat_id", "store_id", "state_id")}
    static = {"dept": codes["dept_id"], "cat": codes["cat_id"], "store": codes["store_id"],
              "state": codes["state_id"]}
    cal_f = calendar_features(cal, codes["state_id"], S)

    keep = long.d.to_numpy() > WARMUP
    base = long.loc[keep, ["sid", "d", "sales", "split"]].reset_index(drop=True)
    r = li[keep]
    c = long.d.to_numpy()[keep] - 1

    report = {}
    for h in HORIZONS:
        feats = {**sales_features(y, h), **price_features(price, h), **cal_f}
        df = base.copy()
        for name, arr in feats.items():
            df[name] = arr[r, c].astype(np.float32)
        for name, v in static.items():
            df[name] = v[r].astype(np.int8)
        cols = [k for k in df.columns if k not in ("sid", "d", "sales", "split")]
        nan_cols = {k: int(df[k].isna().sum()) for k in cols if df[k].isna().any()}
        df.to_parquet(C.FEATURES / f"features_h{h}.parquet", index=False)
        report[h] = {"rows": len(df), "features": cols, "nan_counts": nan_cols,
                     "memory_gb": round(df.memory_usage(deep=True).sum() / 1e9, 3),
                     "rows_per_split": {k: int((df.split == v).sum())
                                        for k, v in (("train", 0), ("val", 1), ("test", 2))}}
        print(f"h={h}: {len(df)} rows, {len(cols)} features, {report[h]['memory_gb']} GB, "
              f"NaN columns: {nan_cols}")
        del feats, df

    desc = {"horizons": list(HORIZONS), "lags": {str(k): list(v) for k, v in LAGS.items()},
            "warmup_days": WARMUP, "descriptions": DESCRIPTIONS,
            "excluded": {"item_id": "high-cardinality identifier; risks memorising series",
                         "year": "non-stationary; test year not seen in training",
                         "MAPE/sMAPE": "not applicable: 59.6% of kept days are zero"},
            "per_horizon": report}
    (C.RESULTS / "feature_report.json").write_text(json.dumps(desc, indent=2))
    cfg_path = C.ROOT / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["features"] = {"lags": desc["lags"], "warmup_days": WARMUP,
                       "n_features": {str(h): len(report[h]["features"]) for h in HORIZONS}}
    cfg_path.write_text(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
