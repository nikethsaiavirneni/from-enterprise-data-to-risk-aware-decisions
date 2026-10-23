"""Supplementary evidence for the manuscript, all from saved predictions:
  (A) policy ablation: point / global / risk-aware cost for every model at the focus ratio;
  (B) sensitivity of the risk-aware gain to the number of forecast bins;
  (C) stability of the decision-cost gain across three equal test sub-windows;
  (D) paired test (Diebold-Mariano-style sign/Wilcoxon) of XGBoost vs LSTM squared error.
Writes results/decision_outputs/extra_analysis.json. Run:  python src/extra_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from decision_engine import POLICIES, adjustment, cost_parts, evaluate, fit_calibration

RATIO = 8
MODELS = ("xgb", "lstm")


def load(model, h):
    df = pd.read_parquet(C.RESULTS / "predictions" / f"{model}_h{h}.parquet")
    va, te = df[df.split == 1], df[df.split == 2].reset_index(drop=True)
    return (va.pred.to_numpy(float), va.sales.to_numpy(float),
            te.sid.to_numpy(), te.pred.to_numpy(float), te.sales.to_numpy(float), te.d.to_numpy())


def policy_cost(sid, y, f, cal, ratio, kind):
    tau = ratio / (1 + ratio)
    q = np.maximum(f + adjustment(cal, f, tau, kind), 0.0)
    u, o = cost_parts(y, q, ratio, 1.0)
    return (u + o).sum() / y.sum()


def main():
    out = {"focus_ratio": RATIO, "ablation": {}, "bin_sensitivity": {}, "window_stability": {},
           "paired_model_test": {}}

    # (A) policy ablation for every model at the focus ratio, horizon 28
    for model in MODELS:
        fva, yva, sid, f, y, _ = load(model, 28)
        cal = fit_calibration(fva, yva)
        row = {k: policy_cost(sid, y, f, cal, RATIO, k) for k in POLICIES}
        row["reduction_risk_vs_point"] = 1 - row["risk_aware"] / row["point"]
        row["reduction_risk_vs_global"] = 1 - row["risk_aware"] / row["global"]
        out["ablation"][model] = row

    # (B) bin-count sensitivity (risk-aware vs point at focus ratio), XGBoost h=28
    fva, yva, sid, f, y, dday = load("xgb", 28)
    for nb in (5, 10, 15, 20):
        cal = fit_calibration(fva, yva, n_bins=nb)
        red = 1 - policy_cost(sid, y, f, cal, RATIO, "risk_aware") / policy_cost(sid, y, f, cal, RATIO, "point")
        out["bin_sensitivity"][nb] = red

    # (C) stability across three equal test sub-windows (calibration fixed on full validation)
    cal = fit_calibration(fva, yva)
    edges = np.quantile(dday, [1 / 3, 2 / 3])
    win = np.digitize(dday, edges)
    res = evaluate(sid, y, f, cal, ratios=(RATIO,), boot=200)
    for w in (0, 1, 2):
        m = win == w
        r = evaluate(sid[m], y[m], f[m], cal, ratios=(RATIO,), boot=200)
        ra = r[r.policy == "risk_aware"].iloc[0]
        out["window_stability"][f"window_{w + 1}"] = {
            "days": [int(dday[m].min()), int(dday[m].max())], "n_cases": int(m.sum()),
            "reduction_vs_point": ra.reduction_vs_point, "ci_low": ra.ci_low, "ci_high": ra.ci_high}

    # (D) paired XGBoost vs LSTM on identical test cases, horizon 28
    xa = pd.read_parquet(C.RESULTS / "predictions" / "xgb_h28.parquet")
    la = pd.read_parquet(C.RESULTS / "predictions" / "lstm_h28.parquet")
    xt = xa[xa.split == 2].set_index(["sid", "d"]).sort_index()
    lt = la[la.split == 2].set_index(["sid", "d"]).sort_index()
    common = xt.index.intersection(lt.index)
    ex = (xt.loc[common, "sales"] - xt.loc[common, "pred"]).to_numpy() ** 2
    el = (lt.loc[common, "sales"] - lt.loc[common, "pred"]).to_numpy() ** 2
    diff = ex - el
    w = stats.wilcoxon(diff, zero_method="zsplit", alternative="two-sided", mode="approx")
    out["paired_model_test"] = {"n": int(len(diff)), "mean_sq_err_xgb": float(ex.mean()),
                                "mean_sq_err_lstm": float(el.mean()),
                                "lstm_lower_sq_err_share": float((el < ex).mean()),
                                "wilcoxon_stat": float(w.statistic), "p_value": float(w.pvalue)}

    (C.RESULTS / "decision_outputs" / "extra_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
