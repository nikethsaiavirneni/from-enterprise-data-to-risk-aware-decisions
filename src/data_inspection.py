"""Phase 2: inspect the raw M5 files and write dataset_report.txt / .csv plus plots.

Every value in the report is measured from the files on disk; nothing is assumed.
Run:  python src/data_inspection.py
"""
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

rows = []  # (section, item, value) -> dataset_report.csv
lines = []


def add(section, item, value):
    rows.append((section, item, value))
    lines.append(f"{item:<44}: {value}")


def head(title):
    lines.append("")
    lines.append(f"== {title} ==")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    C.RESULTS.mkdir(exist_ok=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)

    head("FILES")
    for p in (C.SALES_FILE, C.CALENDAR_FILE, C.PRICES_FILE):
        add("files", p.name, f"{p.stat().st_size / 1e6:.2f} MB, sha256 {sha256(p)[:16]}...")

    sales = pd.read_csv(C.SALES_FILE)
    cal = pd.read_csv(C.CALENDAR_FILE)
    price = pd.read_csv(C.PRICES_FILE)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    id_cols = [c for c in sales.columns if not c.startswith("d_")]

    head("SALES (sales_train_evaluation.csv)")
    add("sales", "rows x columns", f"{sales.shape[0]} x {sales.shape[1]}")
    add("sales", "identifier columns", id_cols)
    add("sales", "daily columns", f"{len(day_cols)} ({day_cols[0]} .. {day_cols[-1]})")
    add("sales", "unique items", sales.item_id.nunique())
    add("sales", "unique stores", sales.store_id.nunique())
    add("sales", "unique states", sorted(sales.state_id.unique().tolist()))
    add("sales", "unique departments", sales.dept_id.nunique())
    add("sales", "categories", sorted(sales.cat_id.unique().tolist()))
    add("sales", "unique (item, store) series",
        len(sales[["item_id", "store_id"]].drop_duplicates()))
    add("sales", "missing values (all columns)", int(sales.isna().sum().sum()))
    add("sales", "daily column dtype", str(sales[day_cols[0]].dtype))
    mem_int64 = sales[day_cols].memory_usage(deep=True).sum() / 1e6
    mem_int16 = sales[day_cols].astype("int16").memory_usage(deep=True).sum() / 1e6
    add("sales", "wide table memory int64 / int16 (MB)", f"{mem_int64:.1f} / {mem_int16:.1f}")

    y = sales[day_cols].to_numpy()
    add("target", "target variable", "daily unit sales per item per store (non-negative integer count)")
    add("target", "min / max / mean", f"{y.min()} / {y.max()} / {y.mean():.4f}")
    add("target", "negative values", int((y < 0).sum()))
    zero_share = (y == 0).mean()
    add("target", "share of zero-sales entries", f"{zero_share:.4f} ({zero_share:.1%})")
    series_zero = (y == 0).mean(axis=1)
    add("target", "series with >50% zero days", f"{(series_zero > 0.5).mean():.1%}")
    add("target", "series with >90% zero days", f"{(series_zero > 0.9).mean():.1%}")
    add("target", "series with all-zero history", int((y.sum(axis=1) == 0).sum()))
    first_sale = (y > 0).argmax(axis=1)
    add("target", "series with first sale after d_1", f"{(first_sale > 0).mean():.1%} "
        "(late launches / not yet stocked)")
    add("target", "median first-sale day index", int(np.median(first_sale)) + 1)

    head("CALENDAR (calendar.csv)")
    add("calendar", "rows x columns", f"{cal.shape[0]} x {cal.shape[1]}")
    add("calendar", "columns", cal.columns.tolist())
    add("calendar", "date range", f"{cal.date.min()} .. {cal.date.max()}")
    add("calendar", "granularity", "daily")
    add("calendar", "days with sales data / without",
        f"{len(day_cols)} / {len(cal) - len(day_cols)}")
    add("calendar", "has 'd' day-key column", "d" in cal.columns)
    add("calendar", "day mapping used",
        "row order: d_k = k-th calendar row (calendar has no 'd' column)"
        if "d" not in cal.columns else "via 'd' column")
    last = (cal.loc[cal.d == day_cols[-1], "date"].iloc[0] if "d" in cal.columns
            else cal.date.iloc[len(day_cols) - 1])
    add("calendar", "date of d_1 / last sales day", f"{cal.date.iloc[0]} / {last}")
    add("calendar", "days with event_name_1", int(cal.event_name_1.notna().sum()))
    add("calendar", "event types", sorted(cal.event_type_1.dropna().unique().tolist()))
    for s in ("CA", "TX", "WI"):
        add("calendar", f"SNAP days {s}", int(cal[f"snap_{s}"].sum()))
    add("calendar", "missing values (non-event columns)",
        int(cal.drop(columns=["event_name_1", "event_type_1", "event_name_2",
                              "event_type_2"]).isna().sum().sum()))

    head("PRICES (sell_prices.csv)")
    add("prices", "rows x columns", f"{price.shape[0]} x {price.shape[1]}")
    add("prices", "columns", price.columns.tolist())
    add("prices", "missing values", int(price.isna().sum().sum()))
    add("prices", "price min / median / max",
        f"{price.sell_price.min():.2f} / {price.sell_price.median():.2f} / {price.sell_price.max():.2f}")
    add("prices", "unique (store, item) with a price",
        len(price[["store_id", "item_id"]].drop_duplicates()))
    add("prices", "weeks covered", f"{price.wm_yr_wk.min()} .. {price.wm_yr_wk.max()}")
    add("prices", "memory (MB)", f"{price.memory_usage(deep=True).sum() / 1e6:.1f}")

    head("AVAILABLE COVARIATES")
    add("covariates", "static", "item_id, dept_id, cat_id, store_id, state_id")
    add("covariates", "calendar", "weekday, wday, month, year, events, SNAP flags")
    add("covariates", "price", "sell_price by store, item, week (not available before a product is sold)")

    head("SALES BY GROUP (total units, share of total)")
    tot = y.sum()
    for col in ("cat_id", "state_id", "store_id"):
        g = sales.groupby(col)[day_cols].sum().sum(axis=1)
        for k, v in g.items():
            add("groups", f"{col}={k}", f"{int(v)} ({v / tot:.1%})")

    # ---- plots (exploratory; not the paper's final figures) -----------------
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    dates = pd.to_datetime(cal.date.iloc[: len(day_cols)])
    daily = pd.Series(y.sum(axis=0), index=dates)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily.index, daily.values, lw=0.6, color="#1f4e79", label="daily total")
    ax.plot(daily.rolling(28).mean(), lw=1.6, color="#c0392b", label="28-day mean")
    ax.set(title="Total daily unit sales, all stores", ylabel="units")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_total_daily_sales.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    cat = sales.groupby("cat_id")[day_cols].sum().sum(axis=1)
    axes[0].bar(cat.index, cat.values / 1e6, color="#1f4e79")
    axes[0].set(title="Units sold by category", ylabel="million units")
    store = sales.groupby("store_id")[day_cols].sum().sum(axis=1).sort_values()
    axes[1].barh(store.index, store.values / 1e6, color="#2e86ab")
    axes[1].set(title="Units sold by store", xlabel="million units")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_sales_by_group.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(series_zero, bins=40, color="#1f4e79")
    ax.set(title="Share of zero-sales days per series", xlabel="share of days with zero sales",
           ylabel="number of series")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_zero_share.png", dpi=200)
    plt.close(fig)

    (C.RESULTS / "dataset_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(rows, columns=["section", "item", "value"]).to_csv(
        C.RESULTS / "dataset_report.csv", index=False)
    print("\n".join(lines))
    print(f"\nWrote {C.RESULTS / 'dataset_report.txt'}, dataset_report.csv and 3 plots in {C.FIGURES}")


if __name__ == "__main__":
    main()
