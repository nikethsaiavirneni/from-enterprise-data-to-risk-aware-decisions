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

## Before the data phase

The M5 data files must be verified in the actual working folder. Follow [PLAN.md](PLAN.md) before implementing assumptions about the dataset or reporting results.
