"""Shared paths and constants. Nothing here depends on the data's contents."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"

SALES_FILE = RAW / "sales_train_evaluation.csv"
CALENDAR_FILE = RAW / "calendar.csv"
PRICES_FILE = RAW / "sell_prices.csv"

SEED = 42
MAX_GPU_MEMORY_GB = 4.0
