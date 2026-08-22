# PLAN.md — What We Are Building and How

**Project:** AI-Integrated Data Analytics for Strategic Decision Intelligence
**Written:** 22 August 2026
**Status:** Planning done. No code written yet. No experiments run yet. No results exist yet.

> The original instructions are stored word-for-word in `PROMPT.md`.
> This file is the plain-language plan: what we decided, and what happens next.

---

## 1. What this project is, in one paragraph

We take real Walmart sales data. We train models to predict how much of each product
each store will sell. Then — and this is the actual point of the research — we do not
stop at the prediction. We explain *why* the model predicted that, we measure *how
uncertain* the prediction is, we turn that uncertainty into a *risk level*, and we turn
the risk level into a *recommendation a manager could act on*. Finally we test whether
those risk-aware recommendations actually lose less money than plain predictions.

The core argument of the paper: **a prediction is not a decision.**

---

## 2. Where we are right now — honest status

| Thing | Status |
|---|---|
| Plan agreed | Done |
| Project folders | Not created |
| Code | None written |
| Walmart data | **Not on this computer** |
| Models trained | None |
| Results | None |
| Paper | Not started |

Nothing has been run. There are no numbers yet. Any number that appears later in this
project must come from a file that our code actually produced.

---

## 3. Decisions locked in

These were agreed on 22 August 2026. If we change any of them later, we write down why.

### Decision 1 — Getting the data: **script**
We write a script that downloads the M5 files from Kaggle automatically.

This needs one thing from you first: a Kaggle API token. Steps are in section 5 below.

### Decision 2 — Forecast horizon: **both 1-day and 28-day**
We run the whole experiment twice — once predicting 1 day ahead, once predicting 28 days
ahead — and compare them.

**Why this matters more than it sounds.** If we predict 28 days ahead, we are not allowed
to use "yesterday's sales" as an input. We would not know it yet. Using it would be
cheating, and cheating in forecasting has a name: **leakage**. It makes results look
brilliant and makes them worthless.

So the two runs use *different* feature sets:

- **1-day run:** may use lag 1, lag 7, rolling averages over recent days
- **28-day run:** every lag must be at least 28 days old. No exceptions.

This doubles the training work, but it gives the paper a real comparison — how much
harder does forecasting get when you need a realistic amount of warning? That directly
feeds the sensitivity analysis the spec asks for in section 22.

### Decision 3 — How much data: **mix of all 3 states, capped at 5 GB**
We use products from all three states (California, Texas, Wisconsin) and all ten stores,
but only a sample of products — not all of them.

**Hard limit: the processed dataset must stay under 5 GB in memory.**

We will not guess how many products that allows. In Phase 2 we measure the real memory
cost per row, then size the sample to fit under 5 GB. The exact number of products and
the random seed used to pick them get written into the paper.

Why a mix instead of one state: the decision layer is about comparing stores and regions.
Using one state throws that away.

### Decision 4 — Money test: **yes, include it**
We add a decision-level evaluation based on stock costs.

**The idea in plain terms.** If a store orders too little, it runs out and loses a sale.
If it orders too much, it wastes money on stock that sits there. These two mistakes do
not cost the same amount.

So we compare two ordering policies against what really sold:

1. **Plain policy:** order exactly what the model predicts
2. **Risk-aware policy:** order what our decision layer recommends, which adjusts for how
   uncertain the forecast is

Then we add up the cost of both and see which is cheaper.

**Honesty requirement:** the cost of understocking versus overstocking is an *assumption
we make*, not a real Walmart figure. It will be labelled that way in the paper, and we
will test a range of values to show the conclusion does not depend on one lucky choice.

This matters because without it the paper only says "we built a system." A reviewer will
ask "did it help?" This gives us a real, measured answer.

### Decision 5 — Save the plan: **yes**
Done. This file and `PROMPT.md`.

---

## 4. The rules we work under

These are not negotiable and they shape every phase.

**Rule 1 — Nothing is invented.**
No made-up numbers. No made-up citations. We never say code ran unless it ran. Every
figure, table, and number in the paper traces back to an output file.

**Rule 2 — Everything gets actually run.**
Scripts get executed on this machine, real output gets read, errors get fixed with the
smallest change that works.

**Rule 3 — Stay inside the hardware.**
GPU is an RTX 3060 Laptop with 6 GB. We stay under **4 GB**, leaving 2 GB for Windows.
Data loading and preparation happen on the CPU. The GPU only sees training batches.
Memory gets logged to `results/gpu_usage.csv`. If CUDA is missing, the code falls back to
CPU instead of crashing.

**Rule 4 — Never assume the data's shape.**
We inspect the real files before writing anything that depends on them.

**Rule 5 — Respect time.**
Splits are chronological. Never shuffled. Every feature must be knowable at the moment
of prediction. Split dates get documented.

---

## 5. What you need to do before Phase 2

The Walmart data is not on this computer. I searched the whole D drive, E drive,
Downloads, Desktop, Documents and OneDrive. It is not there.

Since we chose the download-script option, you need a Kaggle API token:

1. Go to kaggle.com and sign in
2. Click your profile picture, then **Settings**
3. Scroll to the **API** section, click **Create New Token**
4. A file called `kaggle.json` downloads
5. Move it to: `C:\Users\avirn\.kaggle\kaggle.json`
6. On the Kaggle page for the M5 Forecasting Accuracy competition, click **Join
   Competition** and accept the rules — downloads fail without this

Then tell me, and the download script runs.

Phase 1 does not need the data, so it can start immediately.

---

## 6. Hardware and environment — what is actually on this machine

Checked on 22 August 2026:

| Item | Found |
|---|---|
| Python | 3.12.10 |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB |
| GPU driver | 610.88 |
| System RAM | 15.7 GB |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.4.1 |
| SHAP | 0.52.0 |
| matplotlib | 3.11.1 |
| **PyTorch** | **not installed** — needed for the LSTM, ~2.5 GB download |
| **pytest** | **not installed** — needed for the tests |

Note: pandas 3 and numpy 2.5 are both very recent major versions. Some older code
patterns break on them. We pin versions deliberately and deal with breakage if it shows up.

---

## 7. The models

**XGBoost** — the traditional machine-learning baseline. Fast, strong on tabular data,
and works cleanly with SHAP explanations. This is our main workhorse.

**LSTM** — the deep-learning model. Reads sales as a sequence over time. Kept small on
purpose: 1–2 layers, small hidden size, batch size 16–32, early stopping.

**Transformer** — *only if it earns its place.* We decide using a rule written down **now**,
before we see any results, so it cannot become an excuse afterwards:

> Build the Transformer only if **both** are true:
> 1. The LSTM beats XGBoost on validation, or the LSTM clearly underfits in a way more
>    capacity would plausibly fix
> 2. Estimated peak GPU memory stays under 4 GB
>
> If either fails, we skip it and write down honestly that we skipped it and why.

---

## 8. Splitting the data by time

Three blocks, in chronological order, never shuffled:

```
[========== TRAIN ==========][== VALIDATION ==][== TEST ==]
        oldest data              middle            newest
```

- **Train** — the model learns here
- **Validation** — we tune settings here, and measure forecast errors for the risk layer here
- **Test** — touched once, at the end, for final numbers only

Exact dates come from the real files in Phase 2 and get written into the paper.

**A leakage trap worth naming.** The risk layer is built from forecast errors. Those
errors must be measured on **validation**, never on test. If we used test errors, the risk
system would have secretly seen the answers, and every headline metric would still look
clean while the decision results were quietly worthless. We calculate error thresholds on
validation, freeze them, then apply them unchanged to test.

---

## 9. How we score the models

Walmart sales are **counts with a lot of zeros** — many products sell nothing on many days.
That single fact decides our metrics.

| Metric | Use it? | Reason |
|---|---|---|
| MAE | Yes | Average size of error, in units. Easy to explain to a manager. |
| RMSE | Yes | Punishes large misses harder. Big misses are what hurt in stocking. |
| WAPE | Yes | A percentage-style score that survives zeros, because it divides by total sales rather than each row. |
| RMSSE | Yes | Scale-free, so slow and fast sellers can be compared fairly. This is M5's own metric. |
| **MAPE** | **No** | Divides by the actual value. Actual is often zero. It breaks. |
| **sMAPE** | **No** | Degenerates near zero in the same way. |

We will report the **real measured percentage of zero sales days** from Phase 2 as the
evidence for excluding MAPE. We are not guessing that number now.

---

## 10. Explaining the predictions (SHAP)

SHAP tells us which inputs pushed a prediction up or down. We run it on XGBoost, where it
is exact and cheap.

We do **not** run it on millions of rows. We take a documented sample from the test set,
record the sampling method and the random seed, and report:

- which features matter most overall
- a summary plot
- a few individual examples, explained one at a time

**One caution stated plainly in the paper:** SHAP shows what the *model* used. It does not
prove what *causes* sales. These are different claims and we will not blur them.

---

## 11. Turning risk into decisions

The chain:

```
Forecast  →  Uncertainty  →  Risk level  →  Priority  →  Recommendation
```

Risk levels (Low / Moderate / High) come from the spread of forecast errors on the
validation set. Every threshold is written down and reproducible.

Example logic, in words:

- High demand + low risk → stock confidently, high priority
- High demand + high risk → flag for human review before committing
- Low demand + high confidence → low priority

**These are experimental rules we designed to demonstrate the framework. They are not
Walmart's real business rules and the paper will never suggest otherwise.**

Output: `results/decision_outputs/decision_recommendations.csv` with the product, store,
forecast, actual, error, risk, priority, and recommendation.

---

## 12. Phases and where we are

| Phase | What happens | Status |
|---|---|---|
| 1 | Folders, `requirements.txt`, `environment_check.py`, install PyTorch, run the check | **Next** |
| 2 | Download data, inspect it for real, produce `dataset_report.txt` / `.csv` and plots | Blocked on Kaggle token |
| 3 | Preprocessing | Not started |
| 4 | Feature engineering — two versions, 1-day and 28-day | Not started |
| — | pytest tests, actually run | Not started |
| 5 | XGBoost, small tuning budget, record params and timings | Not started |
| 6 | LSTM, loss curve images, GPU memory log | Not started |
| 7 | Transformer — only if the rule in section 7 is satisfied | Not started |
| 8 | SHAP on a justified sample | Not started |
| 9 | Risk methodology from validation errors | Not started |
| 10 | Decision layer + money test | Not started |
| 11 | Final comparison table | Not started |
| 12 | All paper figures and tables from real outputs | Not started |
| 13 | Literature matrix | Not started — see section 14 |
| 14 | Write the manuscript | Not started |
| 15 | Final honesty audit | Not started |

After each phase: a short summary of what ran, the real numbers, and where files were saved.

**Extra checkpoint added:** we pause after Phase 4 as well. Phases 3 and 4 lock in the
leakage decisions, and it is much cheaper to catch a mistake there than after training.

---

## 13. What gets produced

```
data/raw/          the downloaded Walmart CSVs
data/processed/    cleaned data
data/features/     model-ready features
models/            trained models
results/metrics/           scores
results/predictions/       forecasts
results/figures/           charts
results/explanations/      SHAP output
results/decision_outputs/  decision_recommendations.csv
results/gpu_usage.csv      GPU memory log
paper/figures/     the 10 paper figures
paper/tables/      the 7 paper tables
experiment_config.json     seeds, versions, split dates, parameters
experiment_log.csv         every run recorded
```

---

## 14. Open items we still have to settle

**The citation problem.** The spec asks for 20–30 real peer-reviewed papers in
`literature_matrix.csv`. It also forbids inventing citations. Those two pull against each
other, because listing papers from memory is exactly how wrong titles, wrong years and
wrong DOIs get introduced — which is fabrication, even when unintentional.

Two honest routes, to be chosen at Phase 13:

- Search for papers online and enter only ones actually retrieved, recording the link for each
- Ship the table as empty columns for you to fill with verified sources

**The reference paper has not been read.** The ScienceDirect article in `PROMPT.md`
section 2 has not been opened. Nothing about its contents, methods or results will be
written until it is actually read.

**Possible overlap with your other project.** There is a folder next door called
`ai-governance-decision-intelligence` containing its own `paper` directory. If it is for
the same course, the two papers should be checked against each other before submission.

---

## 15. What "done" means

The project is finished when every number in the paper can be traced to a file our code
produced, the audit checklist in `PROMPT.md` section 36 has been run honestly, and the
things that failed or were skipped are written down as plainly as the things that worked.

A paper that reports a limitation accurately is worth more than one that hides it.
