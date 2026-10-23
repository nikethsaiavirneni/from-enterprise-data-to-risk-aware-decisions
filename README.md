# From enterprise data to risk-aware decisions

Code, tests and manuscript for a *Knowledge-Based Systems* short communication:
**a knowledge-based AI framework for digital transformation of retail operations**.

A forecast is an input to a decision, not the decision. Using the public M5 retail data, the project
forecasts demand (XGBoost, LSTM, two baselines) at 1-day and 28-day lead times, explains the
forecasts (SHAP), calibrates forecast-error risk on validation data only, and tests whether a
risk-aware order policy lowers held-out inventory cost compared with ordering the point forecast.
Cost ratios are assumed scenarios, not retailer data. Results are in the manuscript
(`paper/manuscript.docx`); every number there is generated from files in `results/`.

## Layout

| Path | Contents |
|---|---|
| `src/` | one script per stage (see run order) |
| `tests/` | 20 unit tests: chronological split, leakage, metrics, risk labels, decision rules |
| `paper/` | manuscript, highlights, title and abstract, two figures, two tables |
| `results/` | reports, metrics, SHAP and decision summaries (large files are rebuilt, not stored) |
| `experiment_config.json`, `experiment_log.csv` | seed, split dates, sample and run log |

## Setup (Windows PowerShell, Python 3.12, NVIDIA GPU optional)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python environment_check.py
```

## Data

Download the M5 Forecasting Accuracy files `sales_train_evaluation.csv`, `calendar.csv` and
`sell_prices.csv` into `data/raw/`. They are not stored in this repository. The results reported in
the paper were produced from a public community copy (github.com/Nixtla/m5-forecasts) whose
dimensions match the competition description.

## Run order

```powershell
python src/data_inspection.py     # dataset report and exploratory plots
python src/preprocessing.py       # sample series, on-shelf rule, chronological split
python src/feature_engineering.py # leakage-safe features for 1-day and 28-day leads
python src/train_xgboost.py       # XGBoost (GPU) and baselines
python src/train_lstm.py          # small LSTM (GPU memory capped at 4 GB)
python src/explainability.py      # SHAP on a documented test sample
python src/decision_engine.py     # risk strata, recommendations, decision-cost test
python src/paper_assets.py        # Figures 1-2 and Tables 1-2
python src/build_manuscript.py    # manuscript.docx, highlights, title and abstract
python -m pytest                  # tests
```

Seed 42 is used throughout. Calibration quantities are fitted on validation data only and applied
unchanged to the test split.
