# Reference paper and manuscript design

## Official working title

From Enterprise Data to Risk-Aware Decisions: A Knowledge-Grounded AI Framework for Digital Transformation and Innovation

## Accepted reference

Sai Krishna Thota, “Quantum-inspired knowledge-guided decision support for risk-aware UAV mission planning in contested environments,” *Knowledge-Based Systems*, Short Communication, article 116815. DOI: https://doi.org/10.1016/j.knosys.2026.116815.

The accessible ScienceDirect record describes a compact technical contribution: explicit mission-domain rules integrated into route optimization, comparison with baseline methods, a factorial ablation separating the knowledge-rule layer and Q-bit search, and reproducible simulation experiments. We use those as examples of research structure and evidentiary discipline only. We will not copy its wording, algorithm, figures, tables, numerical results, or claims. Its UAV-routing domain is different from our planned retail demand-forecasting case study.

The original Word source file named `KNOSYS_116815_Thota_SourceFile.docx` was not present among the files attached to this conversation. Until it is provided, use the accessible ScienceDirect record/preview as the reference and do not claim to have reviewed the original Word file.

## Research proposition

A demand forecast alone does not specify an action. This study will test whether a transparent, knowledge-grounded risk layer can improve a defined inventory decision objective relative to a forecast-only policy on chronological held-out Walmart M5 observations.

## Research questions

- RQ1: How do a seasonal-naïve baseline, XGBoost, and a compact LSTM compare on a strictly chronological M5 evaluation split at the selected horizons?
- RQ2: Can explanations and validation-derived forecast-error risk information be converted into transparent inventory recommendations without data leakage?
- RQ3: Under explicitly stated asymmetric understocking and overstocking cost assumptions, does the risk-aware policy change held-out decision cost relative to the forecast-only policy, and how sensitive is that result to the assumptions?

## Experimental safeguards

1. Inspect the actual M5 files before writing assumptions into code. Record filenames, schema, date coverage, missingness, and subset selection.
2. Use chronological train/validation/test periods. Tune models and risk thresholds on training/validation only; evaluate the frozen pipeline on test once.
3. Compare against a seasonal-naïve baseline as well as learned models. Use metrics appropriate to zero-heavy sales (MAE, RMSE, WAPE and/or RMSSE); avoid MAPE when zeros make it invalid.
4. For 28-day-ahead forecasting, prevent the use of lag values unavailable at the forecast origin. Document the multi-step prediction strategy.
5. Derive risk thresholds from validation errors, then freeze them before test evaluation. Describe this as an experimental risk proxy, not calibrated uncertainty unless calibration is demonstrated.
6. Compare forecast-only and risk-aware policies on the same test observations and understocking/overstocking assumptions. Predeclare a sensitivity grid; do not select costs after seeing which method wins.
7. Record seeds, software versions, feature definitions, split dates, model parameters, runtimes, and GPU memory. Target no more than approximately 4 GB GPU VRAM; use CPU fallback when needed.
8. No empirical claim, number, table, or plot may be written until its underlying output exists and has been checked.

## Exactly two main-paper figures

- **Figure 1 — Proposed knowledge-grounded forecast-to-decision framework.** A clean vector-style conceptual diagram. Label it as the proposed framework, not a measured result.
- **Figure 2 — Held-out decision-cost sensitivity.** A publication-quality plot comparing forecast-only and risk-aware policy cost across a predeclared understocking/overstocking cost grid. Generate it only from actual test predictions and policy outputs.

## Exactly two main-paper tables

- **Table 1 — Forecasting results.** Seasonal-naïve, XGBoost, and LSTM if successfully trained; metrics, horizon, runtime, and split protocol. Never fill cells with estimated values.
- **Table 2 — Decision-level results and ablation.** Forecast-only versus risk-aware policy, cost assumptions, measured cost, change versus baseline, and sensitivity/uncertainty where justified.

Additional debugging charts and diagnostics may be retained as supplementary artifacts, but the submitted short communication will contain only these two main figures and two main tables unless the journal or editor requests otherwise.

## Manuscript narrative

Use a compact short-communication structure: problem and gap; why prediction is not a decision; proposed framework and testable contribution; data and leakage-safe protocol; models and risk/decision method; results with the two tables and two figures; discussion of whether the policy actually improves decision cost; limitations, reproducibility, and conclusion. Separate literature-derived claims from our measured results.

## What is not yet established

The actual M5 files and their availability in the user's current local folder have not been verified in this working session. No M5 model results or decision-level improvements have been established here. The research gap must be checked against verified literature before claiming novelty. The title alone does not guarantee journal fit or acceptance.
