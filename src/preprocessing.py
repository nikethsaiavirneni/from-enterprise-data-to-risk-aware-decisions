"""Phase 3: sample series, drop not-on-shelf days, split chronologically, save.

Rules (all measured and written to results/preprocessing_report.txt):
  * A (series, day) is kept only if the item has a listed price in that Walmart week.
    A listed price is known in advance, so this rule uses no future information.
  * Split is chronological: train < validation < test. Never shuffled.
Run:  python src/preprocessing.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

N_ITEMS = 1000          # items sampled; every item is kept in all 10 stores
VAL_DAYS = 91
TEST_DAYS = 91
MEM_CAP_GB = 5.0
EST_FEATURE_COLS = 30   # float32 columns Phase 4 may add; used only for the memory budget


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


def main():
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = []

    def log(msg=""):
        out.append(msg)
        print(msg)

    sales = pd.read_csv(C.SALES_FILE)
    cal = pd.read_csv(C.CALENDAR_FILE)
    price = pd.read_csv(C.PRICES_FILE)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    n_days = len(day_cols)
    meta = sales[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    y = sales[day_cols].to_numpy().astype(np.int16)
    S = len(meta)

    # price matrix (series x week), NaN = no listed price
    weeks = np.sort(price.wm_yr_wk.unique())
    sid_of = pd.Series(np.arange(S), index=pd.MultiIndex.from_frame(meta[["item_id", "store_id"]]))
    keys = pd.MultiIndex.from_frame(price[["item_id", "store_id"]])
    sid = sid_of.reindex(keys).to_numpy()
    price_wide = np.full((S, len(weeks)), np.nan, dtype=np.float32)
    price_wide[sid, np.searchsorted(weeks, price.wm_yr_wk.to_numpy())] = price.sell_price.to_numpy()
    week_pos = np.searchsorted(weeks, cal.wm_yr_wk.to_numpy()[:n_days])

    # ---- evidence for the on-shelf rule, on the FULL data -----------------------------
    mask_full = on_shelf_mask(price_wide, week_pos)
    dropped = ~mask_full
    log("== ON-SHELF RULE: keep a day only if a price is listed that week (full data) ==")
    log(f"series-days total                      : {y.size}")
    log(f"series-days dropped (no listed price)  : {int(dropped.sum())} ({dropped.mean():.1%})")
    log(f"  of which sales > 0                   : {int((y[dropped] > 0).sum())}")
    log(f"  share of dropped days with zero sales: {(y[dropped] == 0).mean():.4%}")
    log(f"zero share, all days / kept days       : {(y == 0).mean():.3f} / {(y[mask_full] == 0).mean():.3f}")

    bounds = split_bounds(n_days, VAL_DAYS, TEST_DAYS)
    dates = cal.date.to_numpy()[:n_days]
    log("\n== CHRONOLOGICAL SPLIT (1-based day index, inclusive) ==")
    for k, (a, b) in bounds.items():
        log(f"{k:<6}: d_{a} .. d_{b}  ({dates[a - 1]} .. {dates[b - 1]}, {b - a + 1} days)")

    # ---- sample -----------------------------------------------------------------------
    items = meta[["item_id", "dept_id"]].drop_duplicates()
    chosen = sample_items(items, N_ITEMS, C.SEED)
    rows = np.flatnonzero(meta.item_id.isin(chosen).to_numpy())
    log(f"\n== SAMPLE (seed {C.SEED}, stratified by department) ==")
    log(f"items sampled                          : {len(chosen)} of {items.item_id.nunique()}")
    log(f"series sampled                         : {len(rows)} of {S}")
    log(f"stores / states covered                : {meta.store_id.iloc[rows].nunique()} / "
        f"{meta.state_id.iloc[rows].nunique()}")
    log("departments covered                    : "
        f"{meta.dept_id.iloc[rows].nunique()} of {meta.dept_id.nunique()}")

    m = mask_full[rows]
    si, di = np.nonzero(m)
    long = pd.DataFrame({
        "sid": rows[si].astype(np.int32),
        "d": (di + 1).astype(np.int16),
        "sales": y[rows][si, di],
        "price": price_wide[rows][si, week_pos[di]],
    })
    long["split"] = assign_split(long.d.to_numpy(), bounds)
    assert long.price.notna().all() and (long.sales >= 0).all()

    bpr = long.memory_usage(deep=True).sum() / len(long)
    full_rows = int(mask_full.sum())
    log(f"rows after rule                        : {len(long)}")
    log(f"bytes per row (base table)             : {bpr:.1f}")
    log(f"sample base table                      : {len(long) * bpr / 1e9:.3f} GB")
    est = (bpr + 4 * EST_FEATURE_COLS) * len(long) / 1e9
    est_full = (bpr + 4 * EST_FEATURE_COLS) * full_rows / 1e9
    log(f"est. with {EST_FEATURE_COLS} float32 features, sample : {est:.2f} GB (cap {MEM_CAP_GB} GB)")
    log(f"est. same, ALL series ({full_rows} rows)  : {est_full:.2f} GB -> "
        f"{'fits' if est_full < MEM_CAP_GB else 'exceeds cap; sample required'}")
    for k, v in zip(("train", "val", "test"), range(3)):
        log(f"rows in {k:<5}                          : {int((long.split == v).sum())}")

    meta.iloc[rows].assign(sid=rows).to_parquet(C.PROCESSED / "series_meta.parquet", index=False)
    cal.iloc[:n_days].assign(d=np.arange(1, n_days + 1, dtype=np.int16)).to_parquet(
        C.PROCESSED / "calendar.parquet", index=False)
    long.to_parquet(C.PROCESSED / "long.parquet", index=False)

    cfg_path = C.ROOT / "experiment_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    cfg["data"] = {
        "source": "M5 Forecasting Accuracy (community mirror github.com/Nixtla/m5-forecasts)",
        "seed": C.SEED, "n_items": len(chosen), "n_series": int(len(rows)),
        "val_days": VAL_DAYS, "test_days": TEST_DAYS,
        "split": {k: {"d_start": a, "d_end": b, "date_start": str(dates[a - 1]),
                      "date_end": str(dates[b - 1])} for k, (a, b) in bounds.items()},
        "rule": "keep series-day only if a price is listed in that week",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    (C.RESULTS / "preprocessing_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    log(f"\nWrote {C.PROCESSED} (long, series_meta, calendar), experiment_config.json, "
        "results/preprocessing_report.txt")


if __name__ == "__main__":
    main()
