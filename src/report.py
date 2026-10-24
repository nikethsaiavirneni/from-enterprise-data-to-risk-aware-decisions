"""Report stage: Figures 1-2, Tables 1-2 and a plain-text results pack, built from saved results only.

Nothing here writes the paper. It turns the experiment outputs into publication-quality figures and
tables and one Markdown file (results/results_pack.md) that lists every number, setting and caveat a
writer needs. Every value is read from results/ (nothing is typed in).
Run:  python src/report.py
Outputs (results/): Figure_1/2 (.png 600 dpi, .pdf), Table_1/2.csv, results_pack.md
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from data import HORIZONS, LAGS, RESULTS, WARMUP, log
from decision import BOOT, N_BINS, RATIOS, RULES
from models import BASE, BASELINES, CAT_COLS, EARLY, GRID, MAX_ROUNDS, NUM_COLS, N_SAMPLE

MN = "−"
INK, MUTED, GRID_C = "#0b0b0b", "#52514e", "#dcdcd8"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3
RATIO_FOCUS, H_FOCUS = 8, 28
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def read_json(name):
    return json.loads((RESULTS / name).read_text())


def load():
    """All saved results as one dict; horizon keys are ints."""
    fm = read_json("forecast_metrics.json")
    dc, st = pd.read_csv(RESULTS / "decision_cost.csv"), pd.read_csv(RESULTS / "strata.csv")
    rep = read_json("data_report.json")
    return {"X": {h: fm["xgb"][str(h)] for h in (1, 28)}, "L": {h: fm["lstm"][str(h)] for h in (1, 28)},
            "cost": {(m, h): dc[(dc.model == m) & (dc.horizon == h)] for m in ("xgb", "lstm") for h in (1, 28)},
            "strata": {(m, h): st[(st.model == m) & (st.horizon == h)].set_index("risk")
                       for m in ("xgb", "lstm") for h in (1, 28)},
            "shap": read_json("shap.json"), "rob": read_json("robustness.json"),
            "data": rep["data"], "feat": rep["features"]}


def table1(R):
    rows = []
    for h in (1, 28):
        x = R["X"][h]
        for name, m, secs in (("Seasonal naive", x["seasonal_naive_test"], None),
                              ("28-day rolling mean", x["roll_mean_28_test"], None),
                              ("XGBoost", x["xgboost_test"], x["tuning"][x["chosen_index"]]["train_seconds"]),
                              ("LSTM", R["L"][h]["lstm_test"], R["L"][h]["train_seconds"])):
            rows.append({"Horizon (days)": h, "Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                         "WAPE (%)": 100 * m["WAPE"], "RMSSE": m["RMSSE"], "Training time (s)": secs})
    return pd.DataFrame(rows)


def table2(R):
    rows = []
    for h in (1, 28):
        c = R["cost"][("xgb", h)]
        for ratio in sorted(c.ratio.unique()):
            g = c[c.ratio == ratio].set_index("policy")
            rows.append({"Horizon (days)": h, "Cu:Co": f"{ratio}:1",
                         "Point forecast cost": g.loc["point", "cost_per_unit"],
                         "Global correction cost": g.loc["global", "cost_per_unit"],
                         "Global change (%)": -100 * g.loc["global", "reduction_vs_point"],
                         "Risk-aware cost": g.loc["risk_aware", "cost_per_unit"],
                         "Risk-aware change (%)": -100 * g.loc["risk_aware", "reduction_vs_point"],
                         "CI low (%)": -100 * g.loc["risk_aware", "ci_high"],
                         "CI high (%)": -100 * g.loc["risk_aware", "ci_low"]})
    return pd.DataFrame(rows)


def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def figure1(R):
    h = H_FOCUS
    x28, n28 = R["X"][h], R["feat"]["per_horizon"][str(h)]
    st = R["strata"][("xgb", h)]["wape"]
    focus = R["cost"][("xgb", h)].query(f"ratio == {RATIO_FOCUS} and policy == 'risk_aware'").iloc[0]
    pct = lambda v: f"{100 * v:.0f}"
    boxes = [
        ("1  Data", [f"{R['data']['n_series']:,} series", f"{n28['rows'] / 1e6:.1f} M series-days",
                     "chronological split", "91-day val., test"]),
        ("2  Forecast", ["test RMSE:", f"XGBoost {x28['xgboost_test']['RMSE']:.2f}",
                         f"LSTM {R['L'][h]['lstm_test']['RMSE']:.2f}",
                         f"naive {x28['seasonal_naive_test']['RMSE']:.2f}"]),
        ("3  Explain", ["SHAP on XGBoost",
                        f"{pct(R['shap']['h28']['group_share']['recent sales (lags, rolling)'])}% from recent",
                        "sales features", "not causation"]),
        ("4  Risk strata", ["test WAPE:", f"Low {pct(st['Low'])}%", f"Moderate {pct(st['Moderate'])}%",
                            f"High {pct(st['High'])}%", "validation cut-offs"]),
        ("5  Decision", ["forecast + residual", "quantile of its bin",
                         f"cost {MN}{pct(focus.reduction_vs_point)}% at {RATIO_FOCUS}:1",
                         f"CI [{MN}{100 * focus.ci_high:.1f}, {MN}{100 * focus.ci_low:.1f}]%"]),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))   # drawn at print size: fonts are true point sizes
    ax.set_xlim(-0.5, 101)
    ax.set_ylim(0, 40)
    ax.axis("off")
    w, gap, y0, hgt = 17.8, 2.8, 12, 25
    for i, (title, lines) in enumerate(boxes):
        x = i * (w + gap) + 0.4
        ax.add_patch(FancyBboxPatch((x, y0), w, hgt, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#f4f6fa", ec=BLUE, lw=1.0))
        ax.text(x + w / 2, y0 + hgt - 3.4, title, ha="center", va="center", fontsize=7.8,
                fontweight="bold", color=INK)
        for j, ln in enumerate(lines):
            muted = j == len(lines) - 1 and (len(lines) > 4 or i in (0, 1, 2))
            ax.text(x + w / 2, y0 + hgt - 8.4 - j * 3.4, ln, ha="center", va="center",
                    fontsize=6.2, color=MUTED if muted else INK)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.5, y0 + hgt / 2), (x + w + gap - 0.4, y0 + hgt / 2),
                                         arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1.0))
    ax.annotate("", xy=(1, 7.6), xytext=(99, 7.6), arrowprops={"arrowstyle": "<->", "color": ORANGE, "lw": 1.0})
    ax.text(50, 4.6, "Calibration is fitted on validation data only and applied unchanged to the test split",
            ha="center", fontsize=6.9, color=INK)
    ax.text(50, 1.2, "Values are read from saved experiment outputs (28-day horizon)",
            ha="center", fontsize=6.2, color=MUTED)
    save_fig(fig, "Figure_1")


def figure2(R):
    c = R["cost"][("xgb", H_FOCUS)]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))   # drawn at print size
    spec = [("point", "Point forecast", BLUE, "o", "-"), ("global", "Global correction", ORANGE, "s", "--"),
            ("risk_aware", "Risk-aware (forecast-bin quantile)", AQUA, "D", "-")]
    ratios = sorted(c.ratio.unique())
    for key, label, col, mk, ls in spec:
        g = c[c.policy == key].sort_values("ratio")
        ax.plot(g.ratio, g.cost_per_unit, color=col, marker=mk, ls=ls, lw=1.8, ms=5.5,
                mec="white", mew=0.8, label=label)
    last = ratios[-1]
    for key, *_ in spec:
        v = c[(c.policy == key) & (c.ratio == last)].iloc[0]
        txt = f"{v.cost_per_unit:.2f}" + ("" if key == "point" else f"  ({MN}{100 * v.reduction_vs_point:.0f}%)")
        ax.annotate(txt, (last, v.cost_per_unit), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.6, color=INK)
    f = c[(c.ratio == RATIO_FOCUS) & (c.policy == "risk_aware")].iloc[0]
    p = c[(c.ratio == RATIO_FOCUS) & (c.policy == "point")].iloc[0]
    ax.annotate("", xy=(RATIO_FOCUS * 1.07, f.cost_per_unit), xytext=(RATIO_FOCUS * 1.07, p.cost_per_unit),
                arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.9, "shrinkA": 3, "shrinkB": 3})
    ax.annotate(f"At {RATIO_FOCUS}:1, risk-aware vs point forecast:\n{MN}{100 * f.reduction_vs_point:.0f}% "
                f"(95% CI {MN}{100 * f.ci_high:.1f} to {MN}{100 * f.ci_low:.1f}%)",
                xy=(RATIO_FOCUS * 1.07, p.cost_per_unit), xytext=(1.12, 4.25), fontsize=7.4, color=INK,
                va="center", ha="left", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{x}:1" for x in ratios])
    ax.minorticks_off()
    ax.set_xlim(0.9, last * 1.9)
    ax.set_xlabel("Understock : overstock unit-cost ratio (assumed scenarios)")
    ax.set_ylabel("Cost per unit of realised demand")
    ax.grid(axis="y", color=GRID_C, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    save_fig(fig, "Figure_2")


def make_assets(R):
    """Write Figures 1-2 and Tables 1-2 (CSV) from the saved results."""
    t1, t2 = table1(R), table2(R)
    t1.to_csv(RESULTS / "Table_1.csv", index=False)
    t2.to_csv(RESULTS / "Table_2.csv", index=False)
    figure1(R)
    figure2(R)
    return t1, t2


# ---- results pack (Markdown) ------------------------------------------------------------
def md_table(df, fmt=None):
    """Markdown table from a DataFrame; fmt maps column -> format string."""
    fmt = fmt or {}
    cell = lambda c, v: "" if pd.isna(v) else (fmt[c].format(v) if c in fmt else str(v))
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(c, v) for c, v in zip(df.columns, row)) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def results_pack(R, t1, t2):
    X, L, d, FEAT, SH, rob = R["X"], R["L"], R["data"], R["feat"], R["shap"], R["rob"]
    sp, per = d["split"], FEAT["per_horizon"]
    rules_fired = pd.read_csv(RESULTS / "rule_counts.csv")[["horizon", "demand_level", "risk", "priority", "cases"]]
    cost = lambda m, h, r, pol, col="cost_per_unit": R["cost"][(m, h)].query(
        f"ratio == {r} and policy == '{pol}'")[col].iloc[0]
    out = ["# Results pack", "",
           "Generated by `src/report.py`. Every number comes from files in `results/`; nothing is typed in. "
           "Cost ratios and the recommendation rules are experimental constructs, not retailer data or policy.", ""]

    out += ["## 1. Data and split", "",
            f"- Source: {d['source']}; files `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` "
            "(SHA-256 of each file is in `run_log.txt`).",
            f"- Full data: {d['n_series_total']:,} item-store series, {d['series_days_total']:,} series-days "
            f"(29 January 2011 to 22 May 2016). Sample: {d['n_items']:,} items drawn at random, stratified by "
            f"department (seed {d['seed']}), all 10 stores, giving {d['n_series']:,} series and "
            f"{d['rows_after_rule']:,} series-days after the rule below.",
            f"- On-shelf rule: {d['rule']}. In the full data it removes {100 * d['dropped_share']:.1f}% of "
            f"series-days, of which {d['dropped_with_sales']} had sales. Zero-sales share: "
            f"{100 * d['zero_share_all']:.1f}% of all entries, {100 * d['zero_share_kept']:.1f}% of kept entries "
            "(hence no MAPE/sMAPE).",
            f"- Memory cap {d['mem_cap_gb']} GB: estimated feature table {d['est_gb_sample']:.2f} GB for the sample "
            f"versus {d['est_gb_full']:.2f} GB for all series.",
            "- Chronological split (never shuffled):", ""]
    out += [f"  - {k}: days {v['d_start']}-{v['d_end']} ({v['date_start']} to {v['date_end']}), "
            f"{FEAT['per_horizon']['28']['rows_per_split'][k]:,} rows at h=28" for k, v in sp.items()]
    out += ["", "- Validation window contains the year-end holiday season; the test window does not.", ""]

    out += ["## 2. Method specification", "",
            "Forecast layer. Direct forecast for lead time h in {1, 28} days: ŷ(i,t) = f_h(y(i,s), s <= t-h ; z(i,t)), "
            "where z holds known-in-advance covariates of day t (listed price, calendar, events, SNAP, store and "
            "department codes). Every sales-derived feature uses only days <= t-h (unit-tested by corrupting later "
            "sales).", "",
            f"- Features per horizon: {len(per['28']['features'])} ({', '.join(per['28']['features'])}).",
            f"- Lags: {{h=1: {list(LAGS[1])}, h=28: {list(LAGS[28])}}}; rolling mean/std/zero-share over the last 28 "
            f"observable days; warm-up of {WARMUP} days dropped.",
            f"- Baselines: seasonal naive ({', '.join(f'h={h}: {c}' for h, c in BASELINES.items())}) and the 28-day "
            "rolling mean.", "",
            "Explanation layer. SHAP (TreeExplainer) on the XGBoost log-rate scale, on "
            f"{N_SAMPLE:,} random test rows per horizon. It describes what the model used, not causation.", "",
            f"Risk layer. On validation, forecasts are grouped into {N_BINS} deciles; each bin stores its residuals "
            "e = y - ŷ. A bin's relative uncertainty is u_k = sum|e| / sum(y) (its validation WAPE). Cases are "
            "Low / Moderate / High risk at the 33rd and 67th percentiles of u over validation cases. Bins, "
            "residuals and cut-offs are frozen and applied unchanged to test. (A first version used absolute "
            "residual spread and labelled near-zero forecasts Low; it was replaced after inspecting strata.)", "",
            "Decision layer. Loss for order Q and demand y: l(Q,y) = Cu*max(y-Q,0) + Co*max(Q-y,0) with Co = 1 and "
            f"Cu:Co in {list(RATIOS)}; tau = Cu/(Cu+Co) is the newsvendor critical ratio. Policies on identical "
            "test cases: point Q = ŷ; global Q = max(0, ŷ + q_tau(all validation residuals)); risk-aware "
            "Q = max(0, ŷ + q_tau(validation residuals of the case's forecast bin)). Reported cost = sum l / sum y. "
            f"Intervals: 95% percentile bootstrap, {BOOT} resamples of series (sampling over series only).", "",
            "Rule base (experimental). Demand level = High if ŷ >= validation 67th percentile of forecasts, "
            "else Low:", "",
            md_table(pd.DataFrame([{"Demand level": a, "Risk": b, "Priority": v[0], "Action": v[1]}
                                   for (a, b), v in RULES.items()])), "",
            "Algorithm. (1) Fit f_h on train; predict validation and test. (2) On validation, bin forecasts and store "
            "residuals per bin. (3) Compute u_k and risk cut-offs; freeze. (4) tau = Cu/(Cu+Co). (5) For each test "
            "case find its bin, assign its risk label, set Q = max(0, ŷ + tau-quantile of the bin's residuals), "
            "emit priority and action from the rule base. (6) Score l once on test.", ""]

    out += ["## 3. Models", "",
            f"- XGBoost (GPU, Poisson objective, eta {BASE["eta"]}, early stopping {EARLY} rounds, max {MAX_ROUNDS}): {len(GRID)} configurations "
            "compared on validation RMSE; the best is scored once on test.",
            *[f"  - h={h}: chosen {X[h]['chosen_config']} after {X[h]['rounds']} rounds; validation RMSE by "
              f"configuration {[round(t['val']['RMSE'], 4) for t in X[h]['tuning']]}; test inference "
              f"{X[h]['inference_seconds_test']} s" for h in HORIZONS],
            f"- LSTM (Poisson loss): one layer, {L[28]['config']['hidden']} units over the last "
            f"{L[28]['config']['seq']} days of log1p(sales); embeddings (dim {L[28]['config']['emb']}) for "
            f"{', '.join(CAT_COLS)}; numeric covariates {', '.join(NUM_COLS)}; {L[28]['params']:,} parameters; batch "
            f"{L[28]['config']['batch']}; Adam lr {L[28]['config']['lr']}; {L[28]['config']['samples_per_epoch']:,} "
            f"samples per epoch; early stopping patience {L[28]['config']['patience']}; not tuned beyond this one "
            "configuration.",
            *[f"  - h={h}: {L[h]['epochs_run']} epochs, best validation Poisson NLL {L[h]['best_val_poisson_nll']:.4f}, "
              f"training {L[h]['train_seconds']:.0f} s, GPU peak {L[h]['gpu_peak_allocated_mb']:.0f} MB (cap 4096 MB), "
              f"device {L[h]['device']}" for h in HORIZONS], ""]

    out += ["## 4. Table 1 - forecast accuracy on the test split", "",
            md_table(t1, {"MAE": "{:.3f}", "RMSE": "{:.3f}", "WAPE (%)": "{:.1f}", "RMSSE": "{:.3f}",
                          "Training time (s)": "{:.0f}"}), "",
            "RMSSE is the mean over series of RMSE scaled by the series' training one-step naive error. Training time "
            "is wall-clock on the stated GPU; baselines need no training.", ""]
    rel = lambda a, b: 100 * (1 - a / b)
    for h in HORIZONS:
        x, l = X[h], L[h]
        out.append(f"- h={h}: XGBoost RMSE is {rel(x['xgboost_test']['RMSE'], x['seasonal_naive_test']['RMSE']):.1f}% "
                   f"below seasonal naive and {rel(x['xgboost_test']['RMSE'], x['roll_mean_28_test']['RMSE']):.1f}% "
                   f"below the 28-day rolling mean; LSTM vs XGBoost RMSE difference "
                   f"{100 * (l['lstm_test']['RMSE'] / x['xgboost_test']['RMSE'] - 1):+.2f}%.")
    ptest = rob["paired_model_test"]
    out += [f"- Paired Wilcoxon test of per-case squared error, XGBoost vs LSTM, h=28 (n={ptest['n']:,}): "
            f"p {'< 1e-300 (numerical underflow)' if ptest['p_value'] == 0 else format(ptest['p_value'], '.3g')}; mean squared error {ptest['mean_sq_err_xgb']:.3f} vs "
            f"{ptest['mean_sq_err_lstm']:.3f}; LSTM lower in {100 * ptest['lstm_lower_sq_err_share']:.1f}% of cases.", ""]

    out += ["## 5. SHAP attribution (XGBoost)", ""]
    for h in HORIZONS:
        s = SH[f"h{h}"]
        out += [f"- h={h}: top features " + ", ".join(f"{t['feature']} ({100 * t['share']:.1f}%)" for t in s["top5"])
                + "; group shares " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in s["group_share"].items())]
    out.append("")

    out += ["## 6. Risk strata (test split, XGBoost)", ""]
    for h in HORIZONS:
        st = R["strata"][("xgb", h)].reset_index()[["risk", "cases", "mean_demand", "mae", "wape", "understock_rate"]]
        out += [f"h={h}:", "", md_table(st, {"cases": "{:.0f}", "mean_demand": "{:.3f}", "mae": "{:.3f}",
                                             "wape": "{:.3f}", "understock_rate": "{:.3f}"}), ""]
    out += ["Strata largely track sales volume (mean demand differs strongly by stratum), so they express relative "
            "reliability and are not an independent uncertainty model.", "",
            "Rule base: number of test cases handled by each rule (XGBoost forecasts).", "",
            md_table(rules_fired), ""]
    never = sorted(set(RULES) - set(zip(rules_fired["demand_level"], rules_fired["risk"], strict=True)))
    out += [f"Rules that never fired in either horizon: {never or 'none'}. Rules that fire are the ones the "
            "paper can evidence; do not describe unused rules as validated.", ""]

    out += ["## 7. Table 2 - decision cost per unit of realised demand (XGBoost, test split)", "",
            md_table(t2, {c: "{:.3f}" for c in t2.columns if "cost" in c or "(%)" in c}), "",
            "Change columns are relative to the point policy (negative = cheaper). CI is the 95% bootstrap interval of "
            "the risk-aware change.", ""]
    for m in ("xgb", "lstm"):
        for h in HORIZONS:
            c = R["cost"][(m, h)]
            ra = c[(c.policy == "risk_aware") & (c.ratio == RATIO_FOCUS)].iloc[0]
            out.append(f"- {m} h={h} at {RATIO_FOCUS}:1: point {cost(m, h, RATIO_FOCUS, 'point'):.3f}, global "
                       f"{cost(m, h, RATIO_FOCUS, 'global'):.3f}, risk-aware {ra.cost_per_unit:.3f}; reduction vs point "
                       f"{100 * ra.reduction_vs_point:.1f}% (95% CI {100 * ra.ci_low:.1f}-{100 * ra.ci_high:.1f}); vs global "
                       f"{100 * ra.reduction_vs_global:.1f}% (CI {100 * ra.g_ci_low:.1f}-{100 * ra.g_ci_high:.1f}); coverage "
                       f"of risk-aware orders {100 * ra.coverage:.1f}% against target {100 * ra.tau:.1f}%.")
    out.append("")

    out += ["## 8. Robustness (h=28, focus ratio 8:1)", "",
            "- Policy ablation by forecaster: " + "; ".join(
                f"{m}: point {v['point']:.3f}, global {v['global']:.3f}, risk-aware {v['risk_aware']:.3f} "
                f"(-{100 * v['reduction_risk_vs_point']:.1f}% vs point, -{100 * v['reduction_risk_vs_global']:.1f}% vs global)"
                for m, v in rob["ablation"].items()),
            "- Forecast-bin sensitivity (reduction vs point): " + ", ".join(
                f"K={k}: {100 * v:.1f}%" for k, v in rob["bin_sensitivity"].items()),
            "- Stability across three equal sub-windows of the test period (calibration fixed on full validation): " +
            "; ".join(f"{k} (days {v['days'][0]}-{v['days'][1]}, n={v['n_cases']:,}): {100 * v['reduction_vs_point']:.1f}% "
                      f"[{100 * v['ci_low']:.1f}, {100 * v['ci_high']:.1f}]" for k, v in rob["window_stability"].items()),
            ""]

    out += ["## 9. Figures", "",
            "- `Figure_1`: five-box framework diagram (data, forecast, explain, risk strata, decision) annotated with "
            "values from this run (h=28).",
            "- `Figure_2`: cost per unit of demand against the assumed cost ratio for the three policies, with the "
            f"{RATIO_FOCUS}:1 reduction and its 95% CI marked (XGBoost, h=28).", ""]

    out += ["## 10. Claims the evidence does and does not support", "",
            "Supported: forecast accuracy alone did not separate the approaches (a 28-day rolling mean is close); the "
            "validation-calibrated decision layer lowered assumed cost versus point and global policies; the gain held "
            "across both forecasters, bin counts and test sub-windows; calibration used validation data only.",
            "Not supported / not measured: real retailer savings (cost ratios are assumed); multi-period inventory "
            "dynamics, lead times, capacity or service levels; results across several years, samples or seeds; any "
            "digital-transformation, innovation or entrepreneurship outcome (M5 holds no such data); causation from "
            "SHAP; the files come from a community copy of M5, not the original Kaggle download (their SHA-256 hashes are in "
            "`run_log.txt`; equality with the Kaggle files was not checked).", "",
            "## 11. Reproducibility", "",
            "Seed 42 throughout; `run_log.txt` holds environment versions, GPU, file hashes and per-stage results; "
            "`console_log.txt` holds the full console output including per-epoch LSTM training; 20 unit tests cover "
            "splits, leakage, metrics, risk labels and decision rules.", "",
            "## 12. Implementation notes a paper should disclose", "",
            "- The validation split is used three times: early stopping, XGBoost configuration choice and "
            "calibration (residual quantiles, bins, risk cut-offs). The test split is used once for scoring. "
            "Validation residuals can therefore be slightly optimistic; the coverage check in section 7 compares "
            "achieved coverage with the target.",
            "- Days without a listed price are not kept as rows, but they enter lag, rolling and LSTM-sequence "
            "inputs as zero sales (the data show no sales on those days).",
            "- RMSSE scale is the training one-step naive error computed on training rows after the "
            f"{WARMUP}-day warm-up.",
            "- Price and calendar covariates of the target day are assumed known at order time (planned prices).",
            "- The Wilcoxon test is on per-case squared errors with a normal approximation; cases are not "
            "independent (repeated series and days), so the p-value is descriptive.",
            "- `recommendations_sample.csv` is a seeded sample of the rule-base output; full counts per "
            "priority and risk stratum are in `run_log.txt`.", ""]
    return "\n".join(out)


def main():
    R = load()
    t1, t2 = make_assets(R)
    (RESULTS / "results_pack.md").write_text(results_pack(R, t1, t2), encoding="utf-8")
    log("report", "wrote Figure_1/2 (.png, .pdf), Table_1/2.csv and results_pack.md in results/")


if __name__ == "__main__":
    main()
