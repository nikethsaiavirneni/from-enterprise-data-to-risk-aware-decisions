"""Phase 12: build the paper's 2 figures and 2 tables from saved result files only.

Every number printed on a figure or in a table is read from results/ (nothing typed in).
Run:  python src/paper_assets.py
Outputs: paper/figures/Figure_1.{png,pdf}, Figure_2.{png,pdf}; paper/tables/Table_1.csv, Table_2.csv
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

FIG, TAB = C.ROOT / "paper" / "figures", C.ROOT / "paper" / "tables"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdcd8"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots 1-3
RATIO_FOCUS, H_FOCUS = 8, 28
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def load():
    r = {"xgb": {}, "lstm": {}}
    for m in r:
        for h in (1, 28):
            p = C.METRICS / f"{m}_h{h}.json"
            if p.exists():
                r[m][h] = json.loads(p.read_text())
    cost = {}
    for m in ("xgb", "lstm"):
        for h in (1, 28):
            p = C.RESULTS / "decision_outputs" / f"cost_{m}_h{h}.csv"
            if p.exists():
                cost[(m, h)] = pd.read_csv(p)
    strata = {h: pd.read_csv(C.RESULTS / "decision_outputs" / f"strata_xgb_h{h}.csv", index_col=0)
              for h in (1, 28)}
    shap = json.loads((C.RESULTS / "explanations" / "shap_summary.json").read_text())
    cfg = json.loads((C.ROOT / "experiment_config.json").read_text())
    feat = json.loads((C.RESULTS / "feature_report.json").read_text())
    return r, cost, strata, shap, cfg, feat


def table1(r):
    rows = []
    for h in (1, 28):
        base = r["xgb"][h]
        entries = [("Seasonal naive", base["seasonal_naive_test"], None),
                   ("28-day rolling mean", base["roll_mean_28_test"], None),
                   ("XGBoost", base["xgboost_test"], base["tuning"][base["chosen_index"]]["train_seconds"])]
        if h in r["lstm"]:
            entries.append(("LSTM", r["lstm"][h]["lstm_test"], r["lstm"][h]["train_seconds"]))
        for name, m, secs in entries:
            rows.append({"Horizon (days)": h, "Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"],
                         "WAPE (%)": 100 * m["WAPE"], "RMSSE": m["RMSSE"],
                         "Training time (s)": secs})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "Table_1.csv", index=False)
    return t


def table2(cost):
    rows = []
    for h in (1, 28):
        c = cost[("xgb", h)]
        for ratio in sorted(c.ratio.unique()):
            g = c[c.ratio == ratio].set_index("policy")
            rows.append({"Horizon (days)": h, "Cu:Co": f"{ratio}:1",
                         "Point forecast cost": g.loc["point", "cost_per_unit"],
                         "Global correction cost": g.loc["global", "cost_per_unit"],
                         "Global change (%)": -100 * g.loc["global", "reduction_vs_point"],
                         "Risk-aware cost": g.loc["risk_aware", "cost_per_unit"],
                         "Risk-aware change (%)": -100 * g.loc["risk_aware", "reduction_vs_point"],
                         "95% CI of change (%)": f"[{-100 * g.loc['risk_aware', 'ci_high']:.1f}, "
                                                 f"{-100 * g.loc['risk_aware', 'ci_low']:.1f}]"})
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "Table_2.csv", index=False)
    return t


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def figure1(r, cost, strata, shap, cfg, feat):
    h = H_FOCUS
    x28, n28 = r["xgb"][h], feat["per_horizon"][str(h)]
    st = strata[h]["wape"]
    c = cost[("xgb", h)]
    focus = c[(c.ratio == RATIO_FOCUS) & (c.policy == "risk_aware")].iloc[0]
    lstm = f"LSTM {r['lstm'][h]['lstm_test']['RMSE']:.2f}" if h in r["lstm"] else "LSTM (see Table 1)"
    pct = lambda v: f"{100 * v:.0f}"
    mn = "−"
    boxes = [
        ("1  Data", [f"{cfg['data']['n_series']:,} series", f"{n28['rows'] / 1e6:.1f} M series-days",
                     "chronological split", "91-day val., test"]),
        ("2  Forecast", ["test RMSE:", f"XGBoost {x28['xgboost_test']['RMSE']:.2f}", lstm,
                         f"naive {x28['seasonal_naive_test']['RMSE']:.2f}"]),
        ("3  Explain", ["SHAP on XGBoost",
                        f"{pct(shap['h28']['group_share']['recent sales (lags, rolling)'])}% from recent",
                        "sales features", "not causation"]),
        ("4  Risk strata", ["test WAPE:", f"Low {pct(st['Low'])}%",
                            f"Moderate {pct(st['Moderate'])}%", f"High {pct(st['High'])}%",
                            "validation cut-offs"]),
        ("5  Decision", ["forecast + residual", "quantile of its bin",
                         f"cost {mn}{pct(focus.reduction_vs_point)}% at {RATIO_FOCUS}:1",
                         f"CI [{mn}{100 * focus.ci_high:.1f}, {mn}{100 * focus.ci_low:.1f}]%"]),
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
            ax.text(x + w / 2, y0 + hgt - 8.4 - j * 3.4, ln, ha="center", va="center",
                    fontsize=6.2, color=MUTED if j == len(lines) - 1 and len(lines) > 4 or
                    (j == 3 and len(lines) == 4 and i in (0, 1, 2)) else INK)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.5, y0 + hgt / 2), (x + w + gap - 0.4, y0 + hgt / 2),
                                         arrowstyle="-|>", mutation_scale=9, color=MUTED, lw=1.0))
    ax.annotate("", xy=(1, 7.6), xytext=(99, 7.6),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.0))
    ax.text(50, 4.6, "Calibration is fitted on validation data only and applied unchanged to the test split",
            ha="center", fontsize=6.9, color=INK)
    ax.text(50, 1.2, "Values are read from saved experiment outputs (28-day horizon)",
            ha="center", fontsize=6.2, color=MUTED)
    save(fig, "Figure_1")


def figure2(cost):
    c = cost[("xgb", H_FOCUS)]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))   # drawn at print size
    spec = [("point", "Point forecast", BLUE, "o", "-"), ("global", "Global correction", ORANGE, "s", "--"),
            ("risk_aware", "Risk-aware (forecast-bin quantile)", AQUA, "D", "-")]
    ratios = sorted(c.ratio.unique())
    for key, label, col, mk, ls in spec:
        g = c[c.policy == key].sort_values("ratio")
        ax.plot(g.ratio, g.cost_per_unit, color=col, marker=mk, ls=ls, lw=1.8, ms=5.5,
                mec="white", mew=0.8, label=label)
    last = ratios[-1]
    for key, label, col, mk, ls in spec:
        v = c[(c.policy == key) & (c.ratio == last)].iloc[0]
        txt = f"{v.cost_per_unit:.2f}" + ("" if key == "point" else
                                          f"  (−{100 * v.reduction_vs_point:.0f}%)")
        ax.annotate(txt, (last, v.cost_per_unit), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.6, color=INK)
    f = c[(c.ratio == RATIO_FOCUS) & (c.policy == "risk_aware")].iloc[0]
    p = c[(c.ratio == RATIO_FOCUS) & (c.policy == "point")].iloc[0]
    ax.annotate("", xy=(RATIO_FOCUS * 1.07, f.cost_per_unit), xytext=(RATIO_FOCUS * 1.07, p.cost_per_unit),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=3, shrinkB=3))
    ax.annotate(f"At {RATIO_FOCUS}:1, risk-aware vs point forecast:\n−{100 * f.reduction_vs_point:.0f}% "
                f"(95% CI −{100 * f.ci_high:.1f} to −{100 * f.ci_low:.1f}%)",
                xy=(RATIO_FOCUS * 1.07, p.cost_per_unit), xytext=(1.12, 4.25), fontsize=7.4, color=INK,
                va="center", ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{x}:1" for x in ratios])
    ax.minorticks_off()
    ax.set_xlim(0.9, last * 1.9)
    ax.set_xlabel("Understock : overstock unit-cost ratio (assumed scenarios)")
    ax.set_ylabel("Cost per unit of realised demand")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    save(fig, "Figure_2")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    r, cost, strata, shap, cfg, feat = load()
    t1, t2 = table1(r), table2(cost)
    figure1(r, cost, strata, shap, cfg, feat)
    figure2(cost)
    print(t1.round(3).to_string(index=False))
    print(t2.round(3).to_string(index=False))
    print("wrote", FIG, TAB)


if __name__ == "__main__":
    main()
