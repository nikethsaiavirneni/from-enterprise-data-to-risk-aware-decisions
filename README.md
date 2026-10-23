# From Enterprise Data to Risk-Aware Decisions: A Knowledge-Grounded AI Framework for Digital Transformation and Innovation

**Official research title:** From Enterprise Data to Risk-Aware Decisions: A Knowledge-Grounded AI Framework for Digital Transformation and Innovation

**Empirical case study:** Walmart M5 demand forecasting, explainability, forecast-risk analysis, and decision-level evaluation.

Mini research project — DBA 802 Data Analytics and Strategic Decision Intelligence.

## Research idea

Test whether forecasts combined with transparent domain knowledge and validation-derived risk information can produce better inventory decisions than forecasts alone. The planned empirical case study uses the real Walmart M5 sales dataset, subject to local data verification.

**Core argument:** a prediction is not a decision.

## Reference paper

Sai Krishna Thota, “Quantum-inspired knowledge-guided decision support for risk-aware UAV mission planning in contested environments,” *Knowledge-Based Systems*, DOI: [10.1016/j.knosys.2026.116815](https://doi.org/10.1016/j.knosys.2026.116815). It is a structural and methodological reference only. We will not copy its text, algorithm, figures, tables, or results. Its UAV-routing domain differs from our retail forecasting case study.

## Manuscript design

The target is a focused short communication with **exactly two main-paper figures and two main-paper tables**. The framework figure will explain the proposed method. The second figure will compare held-out decision costs across a predeclared sensitivity range. Table 1 will report measured forecasting results; Table 2 will report decision-policy comparison and ablation. Empirical visuals and numerical cells must be generated from actual experiment outputs.

## Current status

The research design is being refined. The M5 data files, code execution, model training, and decision-level outcomes must be verified before empirical findings can be reported. Do not treat synthetic demonstrations or unrun experiments as evidence.

## Documents

| File | Purpose |
|---|---|
| [PLAN.md](PLAN.md) | Project plan, phases, decisions, and status |
| [PROMPT.md](PROMPT.md) | Original project instructions, retained as historical instructions |
| [paper/REFERENCE_AND_RESEARCH_DESIGN.md](paper/REFERENCE_AND_RESEARCH_DESIGN.md) | Reference-paper notes, experimental safeguards, and manuscript design |

## Ground rules

1. Never fabricate numbers, results, or citations.
2. Every reported number must trace to an output actually produced by the code.
3. Use chronological time-series splits and prevent data leakage.
4. Target approximately 4 GB maximum GPU memory use on an RTX 3060 with 6 GB VRAM.
5. Use Windows and PowerShell-compatible setup instructions.
6. State all inventory cost assumptions explicitly and test sensitivity to them.
7. Do not claim the risk score is calibrated uncertainty unless calibration is demonstrated.

## Phase 1 — Environment check

Use Windows PowerShell from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python environment_check.py
```

The command reports Python/package versions, PyTorch/CUDA availability, GPU details when available, and writes a machine-specific report to `results/environment_report.json`. CUDA is not mandatory; CPU fallback is supported. A successful check validates only the software environment, not the research experiment.

## Phase 2 — Inspect the actual data

The M5 files are not included in this repository. Obtain them from the official [Kaggle M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) after accepting its rules, or another source whose terms permit your use. Place the available original CSVs in `data/raw/` locally; do not commit them or Kaggle credentials.

After Phase 1 is checked, run:

```powershell
python dataset_inspection.py
```

The script scans the files it actually finds in chunks and writes `results/dataset_report.txt`, `results/dataset_report.csv`, `results/dataset_missing_by_column.csv`, and diagnostic plots under `results/figures/`. Missing files are explicitly reported. These are data-inspection artifacts, not model results. If no M5 CSV is found, the script reports that state and stops without manufacturing sample data.
