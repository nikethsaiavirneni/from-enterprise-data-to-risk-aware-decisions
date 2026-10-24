# A knowledge-based AI framework for data-driven digital transformation

Experiment code, tests and outputs for a *Knowledge-Based Systems* short communication:
**risk-aware decision support for retail innovation**. The repository downloads the public data, runs the
experiment and writes the figures, tables, logs and saved models; it does not write the paper.

A forecast is an input to a decision, not the decision. Using the public M5 retail data, the project
forecasts demand (XGBoost, LSTM, two baselines) at 1-day and 28-day lead times, explains the forecasts
(SHAP), calibrates forecast-error risk on validation data only, and tests whether a risk-aware order
policy lowers held-out inventory cost compared with ordering the point forecast. Cost ratios are
assumed scenarios, not retailer data.

## Layout

| Path | Contents |
|---|---|
| `src/data.py` | public-data download, metrics, M5 sampling and chronological split, leakage-safe features, run log |
| `src/models.py` | XGBoost (GPU), LSTM (GPU, memory-capped), SHAP |
| `src/decision.py` | risk strata, rule base, quantile order policies, decision-cost test, robustness checks |
| `src/report.py` | Figures 1-2, Tables 1-2 and `results_pack.md` (every number, setting and caveat in one file) |
| `tests/test_pipeline.py` | 20 unit tests: chronological split, leakage, metrics, risk labels, decision rules |
| `results/` | `Figure_1/2` (accuracy and coverage panels; cost curves), `Table_1/2.csv`, `results_pack.md`, `run_log.txt`, metrics, SHAP, strata, robustness |

Large files (`data/`, `models/`, `results/predictions/`) are rebuilt by the scripts and are not committed.

## Setup (Windows PowerShell, Python 3.12, NVIDIA GPU optional)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Data

`python src/data.py` downloads `sales_train_evaluation.csv`, `calendar.csv` and `sell_prices.csv` from a
public community copy of the M5 Forecasting Accuracy data (github.com/Nixtla/m5-forecasts) into
`data/raw/` and logs their SHA-256 hashes. The files are not stored in this repository. If you prefer the
original competition files, place them in `data/raw/` first and the download is skipped.

## Run order

```powershell
python src/data.py            # download, sample series, on-shelf rule, split, features
python src/models.py xgb      # XGBoost and baselines
python src/models.py lstm     # LSTM
python src/models.py shap     # SHAP on a documented test sample
python src/decision.py        # strata, policies, cost test, robustness
python src/report.py          # figures, tables, results pack
python -m pytest              # tests
```

Seed 42 is used throughout; `results/run_log.txt` records versions, GPU, file hashes and per-stage results. Calibration quantities are fitted on validation data only and applied
unchanged to the test split. Only the stages up to `models.py` need the GPU.
