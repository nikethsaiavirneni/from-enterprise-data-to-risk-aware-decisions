"""Data stage: forecast metrics, M5 sampling and chronological split, leakage-safe features.

A (series, day) is kept only if the item has a listed price in that Walmart week; a listed price
is known in advance, so the rule uses no future information. The split is chronological
(train < validation < test), never shuffled. For a direct forecast h days ahead, every
sales-derived feature uses only days <= t - h; price and calendar features describe day t itself
and are treated as known-in-advance covariates (a stated assumption of the paper).

Run:  python src/data.py        -> downloads M5 if data/raw is empty, then writes data/processed,
                                   data/features, results/data_report.json and results/run_log.txt
"""
import hashlib
import importlib.metadata as md
import io
import json
import platform
import subprocess
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, FEATURES = (ROOT / "data" / p for p in ("raw", "processed", "features"))
RESULTS = ROOT / "results"
SEED = 42
N_ITEMS, VAL_DAYS, TEST_DAYS = 1000, 91, 91   # items sampled (all 10 stores kept), split lengths
MEM_CAP_GB, EST_FEATURE_COLS = 5.0, 30        # memory budget used to justify the sample
M5_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"   # public community copy
M5_FILES = ("sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv")
HORIZONS = (1, 28)
LAGS = {1: (1, 7, 14, 28), 28: (28, 35, 42, 56)}
WARMUP = 60                                    # drop days <= WARMUP so every window is observed


# ---- run log and public-data download ---------------------------------------------------
def log(stage, msg):
    """Append one timestamped line to results/run_log.txt (and echo it)."""
    RESULTS.mkdir(exist_ok=True)
    line = f"{datetime.now().isoformat(timespec='seconds')} [{stage}] {msg}"
    print(line, flush=True)
    with open(RESULTS / "run_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_environment():
    pkgs = ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "torch", "scipy", "matplotlib")
    ver = {p: md.version(p) for p in pkgs if md.distribution(p)}
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      text=True).strip().splitlines()[0]
    except Exception:
        gpu = "none"
    log("env", f"python {platform.python_version()} on {platform.platform()}; gpu: {gpu}; seed {SEED}")
    log("env", "packages: " + ", ".join(f"{k} {v}" for k, v in ver.items()))


def download():
    """Fetch the public M5 files into data/raw (skipped when all three are present); log SHA-256."""
    RAW.mkdir(parents=True, exist_ok=True)
    if not all((RAW / n).exists() for n in M5_FILES):
        log("download", f"fetching {M5_URL}")
        with urllib.request.urlopen(M5_URL) as r:
            blob = r.read()
        log("download", f"archive sha256 {hashlib.sha256(blob).hexdigest()} ({len(blob) / 1e6:.1f} MB)")
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for n in M5_FILES:
                (RAW / n).write_bytes(z.read(n))
    for n in M5_FILES:
        log("download", f"{n} sha256 {hashlib.sha256((RAW / n).read_bytes()).hexdigest()}")


# ---- forecast metrics (no MAPE / sMAPE: ~60% of kept days are zero) ---------------------
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def wape(y, p):
    """Weighted absolute percentage error: sum|e| / sum|y| (defined when sum|y| > 0)."""
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def naive_scale(sales_by_series):
    """Per-series RMSSE scale: mean squared one-step naive error on the training window."""
    return np.array([np.mean(np.diff(s.astype(np.float64)) ** 2) if len(s) > 1 else np.nan
                     for s in sales_by_series])


def rmsse(y, p, sid, scale_by_sid):
    """Mean over series of sqrt(MSE_i / scale_i); series with scale 0 or NaN are skipped."""
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


def scale_by_sid(train):
    """RMSSE scale indexed by series id, from a training DataFrame with sid, d, sales."""
    tr = train.sort_values(["sid", "d"])
    parts = np.split(tr.sales.to_numpy(), np.flatnonzero(np.diff(tr.sid.to_numpy())) + 1)
    sids = tr.sid.drop_duplicates().to_numpy()
    out = np.full(sids.max() + 1, np.nan)
    out[sids] = naive_scale(parts)
    return out


# ---- sampling and chronological split ---------------------------------------------------
def sample_items(items, n, seed):
    """Stratified (by department) random sample of item ids. items: DataFrame item_id, dept_id."""
    rng = np.random.default_rng(seed)
    picked = []
    for _, g in items.groupby("dept_id"):
        k = max(1, round(n * len(g) / len(items)))
        picked.extend(rng.choice(g.item_id.to_numpy(), size=min(k, len(g)), replace=False))
    return sorted(picked)


def split_bounds(n_days, val_days, test_days):
    """1-based inclusive day indices for train / validation / test."""
    test_start = n_days - test_days + 1
    val_start = test_start - val_days
    return {"train": (1, val_start - 1), "val": (val_start, test_start - 1),
            "test": (test_start, n_days)}


def assign_split(d, bounds):
    """0 = train, 1 = validation, 2 = test, for 1-based day indices d."""
    d = np.asarray(d)
    out = np.zeros(d.shape, dtype=np.int8)
    out[d >= bounds["val"][0]] = 1
    out[d >= bounds["test"][0]] = 2
    return out


def on_shelf_mask(price_wide, week_pos):
    """True where the series has a listed price in the day's week. price_wide: (S, W)."""
    return ~np.isnan(price_wide[:, week_pos])


def preprocess():
    """Sample series, apply the on-shelf rule, split chronologically; returns a report dict."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    cal = pd.read_csv(RAW / "calendar.csv")
    price = pd.read_csv(RAW / "sell_prices.csv")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    n_days = len(day_cols)
    meta = sales[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    y = sales[day_cols].to_numpy().astype(np.int16)
    S = len(meta)

    # price matrix (series x week), NaN = no listed price
    weeks = np.sort(price.wm_yr_wk.unique())
    sid_of = pd.Series(np.arange(S), index=pd.MultiIndex.from_frame(meta[["item_id", "store_id"]]))
    sid = sid_of.reindex(pd.MultiIndex.from_frame(price[["item_id", "store_id"]])).to_numpy()
    price_wide = np.full((S, len(weeks)), np.nan, dtype=np.float32)
    price_wide[sid, np.searchsorted(weeks, price.wm_yr_wk.to_numpy())] = price.sell_price.to_numpy()
    week_pos = np.searchsorted(weeks, cal.wm_yr_wk.to_numpy()[:n_days])

    mask_full = on_shelf_mask(price_wide, week_pos)       # evidence for the rule, on the FULL data
    dropped = ~mask_full
    bounds = split_bounds(n_days, VAL_DAYS, TEST_DAYS)
    dates = cal.date.to_numpy()[:n_days]
    items = meta[["item_id", "dept_id"]].drop_duplicates()
    chosen = sample_items(items, N_ITEMS, SEED)
    rows = np.flatnonzero(meta.item_id.isin(chosen).to_numpy())

    si, di = np.nonzero(mask_full[rows])
    long = pd.DataFrame({"sid": rows[si].astype(np.int32), "d": (di + 1).astype(np.int16),
                         "sales": y[rows][si, di], "price": price_wide[rows][si, week_pos[di]]})
    long["split"] = assign_split(long.d.to_numpy(), bounds)
    assert long.price.notna().all() and (long.sales >= 0).all()

    bpr = long.memory_usage(deep=True).sum() / len(long)
    est = lambda n: (bpr + 4 * EST_FEATURE_COLS) * n / 1e9
    meta.iloc[rows].assign(sid=rows).to_parquet(PROCESSED / "series_meta.parquet", index=False)
    cal.iloc[:n_days].assign(d=np.arange(1, n_days + 1, dtype=np.int16)).to_parquet(
        PROCESSED / "calendar.parquet", index=False)
    long.to_parquet(PROCESSED / "long.parquet", index=False)

    return {
        "source": "M5 Forecasting Accuracy (community mirror github.com/Nixtla/m5-forecasts)",
        "seed": SEED, "n_items": len(chosen), "n_series": int(len(rows)), "n_series_total": S,
        "val_days": VAL_DAYS, "test_days": TEST_DAYS,
        "split": {k: {"d_start": a, "d_end": b, "date_start": str(dates[a - 1]),
                      "date_end": str(dates[b - 1])} for k, (a, b) in bounds.items()},
        "rule": "keep series-day only if a price is listed in that week",
        "series_days_total": int(y.size), "dropped_share": float(dropped.mean()),
        "dropped_with_sales": int((y[dropped] > 0).sum()),
        "zero_share_all": float((y == 0).mean()), "zero_share_kept": float((y[mask_full] == 0).mean()),
        "rows_after_rule": len(long), "est_gb_sample": est(len(long)),
        "est_gb_full": est(int(mask_full.sum())), "mem_cap_gb": MEM_CAP_GB,
    }


# ---- leakage-safe features --------------------------------------------------------------
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
    mean = np.where(c == w, s / w, np.nan)
    if kind == "mean":
        return mean
    cs2 = np.hstack([pad, np.cumsum(zz ** 2, axis=1)])
    var = np.maximum((cs2[:, hi] - cs2[:, lo]) / w - mean ** 2, 0.0)
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
    f = {"price": price.astype(np.float64), "price_ratio_7": price / shift(price, 7)}
    cs = np.cumsum(np.nan_to_num(price, nan=0.0), axis=1)
    cn = np.cumsum(ok, axis=1)
    hist_mean = shift(cs / np.maximum(cn, 1), h)  # mean listed price up to t-h
    f["price_rel_hist"] = price / np.where(hist_mean > 0, hist_mean, np.nan)
    f["listed_days"] = cn.astype(np.float64)       # days listed so far (known at t)
    return f


def calendar_features(cal, state_idx, n_series):
    """Per-day calendar columns broadcast to (S, D). cal has D rows."""
    D = len(cal)
    dt = pd.to_datetime(cal.date)
    day = {"wday": cal.wday.to_numpy(), "month": cal.month.to_numpy(),
           "day_of_month": dt.dt.day.to_numpy(),
           "week_of_year": dt.dt.isocalendar().week.to_numpy().astype(int),
           "event_type": cal.event_type_1.astype("category").cat.codes.to_numpy() + 1,  # 0 = none
           "is_event": cal.event_name_1.notna().to_numpy().astype(int)}
    f = {k: np.broadcast_to(v[None, :], (n_series, D)).astype(np.float64) for k, v in day.items()}
    snap = np.stack([cal.snap_CA.to_numpy(), cal.snap_TX.to_numpy(), cal.snap_WI.to_numpy()])
    f["snap"] = snap[state_idx].astype(np.float64)
    return f


def build_features():
    """Write data/features/features_h{h}.parquet; returns the feature report dict."""
    FEATURES.mkdir(parents=True, exist_ok=True)
    long = pd.read_parquet(PROCESSED / "long.parquet")
    meta = pd.read_parquet(PROCESSED / "series_meta.parquet").reset_index(drop=True)
    cal = pd.read_parquet(PROCESSED / "calendar.parquet")
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
    r, c = li[keep], long.d.to_numpy()[keep] - 1

    report = {}
    for h in HORIZONS:
        feats = {**sales_features(y, h), **price_features(price, h), **cal_f}
        df = base.copy()
        for name, arr in feats.items():
            df[name] = arr[r, c].astype(np.float32)
        for name, v in static.items():
            df[name] = v[r].astype(np.int8)
        cols = [k for k in df.columns if k not in ("sid", "d", "sales", "split")]
        df.to_parquet(FEATURES / f"features_h{h}.parquet", index=False)
        report[str(h)] = {"rows": len(df), "features": cols,
                          "rows_per_split": {k: int((df.split == v).sum())
                                             for k, v in (("train", 0), ("val", 1), ("test", 2))}}
        print(f"h={h}: {len(df)} rows, {len(cols)} features")
        del feats, df
    return {"lags": {str(k): list(v) for k, v in LAGS.items()}, "warmup_days": WARMUP,
            "per_horizon": report}


def main():
    RESULTS.mkdir(exist_ok=True)
    log_environment()
    download()
    rep = {"data": preprocess()}
    d = rep["data"]
    log("preprocess", f"{d['n_series']} of {d['n_series_total']} series, {d['rows_after_rule']} rows after the "
                      f"on-shelf rule ({100 * d['dropped_share']:.1f}% of series-days dropped, "
                      f"{d['dropped_with_sales']} of them with sales)")
    rep["features"] = build_features()
    for h, v in rep["features"]["per_horizon"].items():
        log("features", f"h={h}: {v['rows']} rows, {len(v['features'])} features, split rows {v['rows_per_split']}")
    (RESULTS / "data_report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
