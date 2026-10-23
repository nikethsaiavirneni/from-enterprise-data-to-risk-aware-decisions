#!/usr/bin/env python
"""Inspect locally supplied M5 CSVs without assuming their shape.

Run from the repository root after Phase 1:
    python dataset_inspection.py

This script never downloads data and never manufactures demo records. It scans
CSV files found in data/raw, writes inventory/schema reports, and creates
exploratory plots only when the relevant real CSVs are available.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results"
FIG_DIR = OUT_DIR / "figures"

EXPECTED_FILES = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "sample_submission.csv",
]
ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def safe_cell(value: Any) -> str:
    """Make a value safe and compact for a plain-text report."""
    text = str(value)
    return text.replace("\n", " ").strip()


def inspect_csv(path: Path) -> tuple[dict[str, Any], dict[str, int], pd.Series | None]:
    """Scan a CSV in chunks and return file summary, missing counts, and daily sales sums."""
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    is_wide_sales = any(re.fullmatch(r"d_\d+", str(c)) for c in columns)
    day_columns = [c for c in columns if re.fullmatch(r"d_\d+", str(c))]
    chunk_size = 500 if is_wide_sales else 100_000

    rows = 0
    missing = pd.Series(0, index=columns, dtype="int64")
    unique_values: dict[str, set[str]] = {
        name: set() for name in ID_COLUMNS if name in columns
    }
    daily_sales_total = pd.Series(0.0, index=day_columns, dtype="float64") if day_columns else None
    zero_count = 0
    observed_sales_count = 0
    date_min = None
    date_max = None
    first_chunk_dtypes: dict[str, str] = {}

    for chunk_number, chunk in enumerate(
        pd.read_csv(path, chunksize=chunk_size, low_memory=False), start=1
    ):
        rows += len(chunk)
        missing = missing.add(chunk.isna().sum().astype("int64"), fill_value=0).astype("int64")

        if chunk_number == 1:
            first_chunk_dtypes = {name: str(dtype) for name, dtype in chunk.dtypes.items()}

        for name, values in unique_values.items():
            values.update(chunk[name].dropna().astype(str).unique().tolist())

        if "date" in chunk.columns:
            dates = pd.to_datetime(chunk["date"], errors="coerce")
            current_min = dates.min()
            current_max = dates.max()
            if pd.notna(current_min):
                date_min = current_min if date_min is None else min(date_min, current_min)
            if pd.notna(current_max):
                date_max = current_max if date_max is None else max(date_max, current_max)

        if day_columns:
            sales = chunk[day_columns]
            zero_count += int((sales == 0).sum().sum())
            observed_sales_count += int(sales.notna().sum().sum())
            assert daily_sales_total is not None
            daily_sales_total = daily_sales_total.add(sales.sum(axis=0, numeric_only=True), fill_value=0)

        if chunk_number % 10 == 0:
            print(f"Scanned {path.name}: {rows:,} rows...", flush=True)

    unique_counts = {name: len(values) for name, values in unique_values.items()}
    missing_cols = {name: int(count) for name, count in missing.items() if int(count) > 0}
    summary: dict[str, Any] = {
        "file": path.name,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "columns": len(columns),
        "missing_cells": int(missing.sum()),
        "columns_with_missing_values": len(missing_cols),
        "missing_column_names": ";".join(missing_cols.keys()),
        "column_names": json.dumps(columns, ensure_ascii=False),
        "first_20_columns": ";".join(columns[:20]),
        "sample_dtypes": json.dumps(first_chunk_dtypes, ensure_ascii=False),
        "unique_id_counts": json.dumps(unique_counts, ensure_ascii=False),
        "date_min": date_min.date().isoformat() if date_min is not None else "",
        "date_max": date_max.date().isoformat() if date_max is not None else "",
        "sales_day_columns": len(day_columns),
        "zero_sales_percent": (
            round(100.0 * zero_count / observed_sales_count, 6)
            if observed_sales_count
            else None
        ),
    }
    missing_rows = {column: int(count) for column, count in missing.items()}
    return summary, missing_rows, daily_sales_total


def make_calendar_plot(calendar_path: Path) -> None:
    """Plot observed calendar coverage by year from the real calendar file."""
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    if "date" not in calendar.columns:
        return
    yearly_days = calendar.dropna(subset=["date"]).groupby(calendar["date"].dt.year).size()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    yearly_days.plot(kind="bar", ax=ax)
    ax.set_title("M5 calendar coverage by year")
    ax.set_xlabel("Calendar year")
    ax.set_ylabel("Number of calendar records")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(FIG_DIR / "dataset_calendar_coverage.png", dpi=300)
    plt.close(fig)


def make_sales_plot(calendar_path: Path, daily_total: pd.Series | None) -> None:
    """Plot aggregated daily sales only when real sales and calendar files exist."""
    if daily_total is None or not calendar_path.is_file():
        return
    calendar = pd.read_csv(calendar_path, usecols=["date", "d"], parse_dates=["date"])
    mapping = calendar.set_index("d")["date"]
    x_dates = [mapping.get(day) for day in daily_total.index]
    valid = [(date, float(value)) for date, value in zip(x_dates, daily_total.values) if pd.notna(date)]
    if not valid:
        return
    dates, values = zip(*valid)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.plot(dates, values, linewidth=0.8)
    ax.set_title("M5 aggregate daily unit sales (observed sales columns)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total units")
    ax.grid(True, alpha=0.25)
    fig.savefig(FIG_DIR / "dataset_aggregate_daily_sales.png", dpi=300)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    missing_details: dict[str, dict[str, int]] = {}
    all_daily_sales: dict[str, pd.Series] = {}
    found_files: list[str] = []

    for filename in EXPECTED_FILES:
        path = RAW_DIR / filename
        if not path.is_file():
            summaries.append(
                {
                    "file": filename,
                    "exists": False,
                    "size_bytes": None,
                    "rows": None,
                    "columns": None,
                    "missing_cells": None,
                    "columns_with_missing_values": None,
                    "missing_column_names": "",
                    "column_names": "",
                    "first_20_columns": "",
                    "sample_dtypes": "",
                    "unique_id_counts": "",
                    "date_min": "",
                    "date_max": "",
                    "sales_day_columns": None,
                    "zero_sales_percent": None,
                }
            )
            continue

        found_files.append(filename)
        print(f"Inspecting {path} ...", flush=True)
        summary, missing, daily_sales = inspect_csv(path)
        summaries.append(summary)
        missing_details[filename] = missing
        if daily_sales is not None:
            all_daily_sales[filename] = daily_sales

    report_df = pd.DataFrame(summaries)
    report_csv = OUT_DIR / "dataset_report.csv"
    report_txt = OUT_DIR / "dataset_report.txt"
    report_df.to_csv(report_csv, index=False, encoding="utf-8-sig")

    missing_csv = OUT_DIR / "dataset_missing_by_column.csv"
    missing_records = [
        {"file": filename, "column": column, "missing_values": count}
        for filename, counts in missing_details.items()
        for column, count in counts.items()
    ]
    pd.DataFrame(missing_records, columns=["file", "column", "missing_values"]).to_csv(
        missing_csv, index=False, encoding="utf-8-sig"
    )

    calendar_path = RAW_DIR / "calendar.csv"
    if calendar_path.is_file():
        make_calendar_plot(calendar_path)
    if all_daily_sales:
        selected = "sales_train_evaluation.csv" if "sales_train_evaluation.csv" in all_daily_sales else next(iter(all_daily_sales))
        make_sales_plot(calendar_path, all_daily_sales[selected])

    lines = [
        "M5 DATASET INSPECTION REPORT",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Repository root: {ROOT}",
        f"Raw data folder: {RAW_DIR}",
        "",
        "This report summarizes files actually found and scanned. Missing files are not inferred or substituted.",
        f"Files found: {len(found_files)} of {len(EXPECTED_FILES)} expected.",
        "Found: " + (", ".join(found_files) if found_files else "none"),
        "Missing: " + (", ".join(x for x in EXPECTED_FILES if x not in found_files) if len(found_files) < len(EXPECTED_FILES) else "none"),
        "",
    ]
    for row in summaries:
        lines.extend(
            [
                f"FILE: {row['file']}",
                f"  Present: {row['exists']}",
                f"  Size (bytes): {row['size_bytes']}",
                f"  Rows: {row['rows']}",
                f"  Columns: {row['columns']}",
                f"  Columns with missing values: {row['columns_with_missing_values']}",
                f"  Date range (if applicable): {row['date_min']} to {row['date_max']}",
                f"  Unique ID counts: {row['unique_id_counts']}",
                f"  Sales day columns: {row['sales_day_columns']}",
                f"  Zero-sales percentage across observed day cells: {row['zero_sales_percent']}",
                f"  First 20 columns: {row['first_20_columns']}",
                "",
            ]
        )
    lines.extend(
        [
            f"Summary CSV: {report_csv}",
            f"Missing-value CSV: {missing_csv}",
            f"Figures folder: {FIG_DIR}",
        ]
    )
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    if not found_files:
        print("\nSTATUS: No expected M5 CSV files were found. This is an inventory result, not a dataset analysis.")
        return 2
    print(f"\nSaved reports:\n- {report_txt}\n- {report_csv}\n- {missing_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
