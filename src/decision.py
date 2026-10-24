"""Decision stage: validation-calibrated risk strata, rule-based recommendations, decision-cost test.

Everything that sets a number (residual quantiles, forecast bins, risk cut-offs) is fitted on the
VALIDATION split and then applied unchanged to TEST. Cost ratios and recommendation rules are
experimental constructs, not Walmart policy.

Policies compared on identical test cases (order quantity Q for day-t demand D, forecast f):
  point        Q = f
  global       Q = f + q_tau(validation residuals)              (one correction for all items)
  risk_aware   Q = f + q_tau(validation residuals | forecast bin)
with tau = Cu / (Cu + Co), the newsvendor critical ratio, and residual e = D - f.
Cost = Cu * max(D - Q, 0) + Co * max(Q - D, 0), reported per unit of realised demand.

Run:  python src/decision.py   -> results/decision_cost.csv, strata.csv, robustness.json, rule_counts.csv,
      recommendations_sample.csv
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

from data import RESULTS, SEED, log

PRED = RESULTS / "predictions"
RATIOS = (1, 2, 4, 8, 16)         # Cu:Co scenarios (Co = 1)
N_BINS, BOOT = 10, 500
POLICIES = ("point", "global", "risk_aware")
FOCUS_RATIO = 8
N_REC_SAMPLE = 2000               # test cases per horizon kept in results/recommendations_sample.csv


def cost_parts(y, q, cu, co):
    """Per-case understock and overstock cost."""
    return cu * np.maximum(y - q, 0.0), co * np.maximum(q - y, 0.0)


def fit_calibration(f_val, y_val, n_bins=N_BINS):
    """Fit on validation only: forecast-bin edges and residuals per bin."""
    e = y_val - f_val
    edges = np.unique(np.quantile(f_val, np.linspace(0, 1, n_bins + 1)[1:-1]))
    b = np.searchsorted(edges, f_val, side="right")
    return {"edges": edges, "resid_all": e, "resid_bin": [e[b == k] for k in range(len(edges) + 1)]}


def adjustment(cal, f, tau, kind):
    """Additive correction to the forecast for each case."""
    if kind == "point":
        return np.zeros_like(f)
    if kind == "global":
        return np.full_like(f, np.quantile(cal["resid_all"], tau))
    q = np.array([np.quantile(r, tau) if len(r) else 0.0 for r in cal["resid_bin"]])
    return q[np.searchsorted(cal["edges"], f, side="right")]


def risk_strata(cal, f, f_val):
    """Relative uncertainty of a forecast = validation WAPE of its forecast bin (sum|e| / sum(D)).
    Low / Moderate / High at the 33rd / 67th percentile of that quantity over validation cases.
    (A first version used the absolute residual spread; it labelled near-zero forecasts 'Low'
    although their relative error is largest, so it was replaced by this relative measure.)"""
    b_val = np.searchsorted(cal["edges"], f_val, side="right")
    sp = np.array([np.abs(r).sum() / max((f_val[b_val == k] + r).sum(), 1e-9) if len(r) else np.nan
                   for k, r in enumerate(cal["resid_bin"])])
    cuts = np.quantile(sp[b_val], [1 / 3, 2 / 3])
    s = sp[np.searchsorted(cal["edges"], f, side="right")]
    return np.where(s <= cuts[0], "Low", np.where(s <= cuts[1], "Moderate", "High")), cuts, sp


# Experimental knowledge base: (demand level, risk) -> (priority, action)
RULES = {
    ("High", "Low"): ("P1", "Stock to forecast; confident replenishment"),
    ("High", "Moderate"): ("P1", "Stock to risk-adjusted quantity"),
    ("High", "High"): ("P2", "Stock to risk-adjusted quantity; flag for human review"),
    ("Low", "Low"): ("P4", "Minimal stock; routine replenishment"),
    ("Low", "Moderate"): ("P3", "Stock to risk-adjusted quantity"),
    ("Low", "High"): ("P3", "Small buffer; monitor demand signal"),
}


def recommend(f, f_val, risk):
    lvl = np.where(f >= np.quantile(f_val, 2 / 3), "High", "Low")
    pr = np.array([RULES[(a, b)][0] for a, b in zip(lvl, risk)])
    act = np.array([RULES[(a, b)][1] for a, b in zip(lvl, risk)])
    return lvl, pr, act


def evaluate(sid, y, f, cal, ratios=RATIOS, boot=BOOT, seed=SEED):
    """Cost per unit of demand for each policy and ratio, with a series-level bootstrap CI for the
    relative change versus the point policy. Returns a DataFrame."""
    sids, inv = np.unique(sid, return_inverse=True)
    n = len(sids)
    dem = np.bincount(inv, weights=y, minlength=n)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(boot, n))
    rows = []
    for r in ratios:
        tau = r / (1 + r)
        per, cover = {}, {}
        for k in POLICIES:
            q = np.maximum(f + adjustment(cal, f, tau, k), 0.0)
            u, o = cost_parts(y, q, r, 1.0)
            per[k] = (np.bincount(inv, weights=u, minlength=n), np.bincount(inv, weights=o, minlength=n))
            cover[k] = float(np.mean(y <= q))   # share of cases where Q covers demand; target tau
        tot = {k: per[k][0] + per[k][1] for k in POLICIES}
        for k in POLICIES:
            row = {"ratio": r, "tau": round(tau, 4), "policy": k,
                   "cost_per_unit": tot[k].sum() / dem.sum(),
                   "understock_per_unit": per[k][0].sum() / dem.sum(),
                   "overstock_per_unit": per[k][1].sum() / dem.sum(), "n_cases": len(y),
                   "coverage": cover[k]}
            if k != "point":
                rel = 1 - tot[k].sum() / tot["point"].sum()
                b = 1 - tot[k][draws].sum(1) / tot["point"][draws].sum(1)
                row.update(reduction_vs_point=rel, ci_low=np.quantile(b, .025),
                           ci_high=np.quantile(b, .975))
                if k == "risk_aware":   # extra comparison against the stronger global-correction policy
                    relg = 1 - tot[k].sum() / tot["global"].sum()
                    bg = 1 - tot[k][draws].sum(1) / tot["global"][draws].sum(1)
                    row.update(reduction_vs_global=relg, g_ci_low=np.quantile(bg, .025),
                               g_ci_high=np.quantile(bg, .975))
            rows.append(row)
    return pd.DataFrame(rows)


def strata_table(risk, y, f):
    d = pd.DataFrame({"risk": risk, "ae": np.abs(y - f), "under": y > f, "y": y})
    g = d.groupby("risk").agg(cases=("ae", "size"), mae=("ae", "mean"), mean_demand=("y", "mean"),
                              understock_rate=("under", "mean"))
    g["wape"] = d.groupby("risk").ae.sum() / d.groupby("risk").y.sum()
    return g.reindex(["Low", "Moderate", "High"])


def load(model, h):
    """Validation and test arrays of one saved forecast."""
    df = pd.read_parquet(PRED / f"{model}_h{h}.parquet")
    va, te = df[df.split == 1], df[df.split == 2].reset_index(drop=True)
    return (va.pred.to_numpy(float), va.sales.to_numpy(float),
            te.sid.to_numpy(), te.pred.to_numpy(float), te.sales.to_numpy(float), te.d.to_numpy())


def policy_cost(sid, y, f, cal, ratio, kind):
    tau = ratio / (1 + ratio)
    u, o = cost_parts(y, np.maximum(f + adjustment(cal, f, tau, kind), 0.0), ratio, 1.0)
    return (u + o).sum() / y.sum()


def robustness():
    """(A) policy ablation per model, (B) forecast-bin sensitivity, (C) stability across three test
    sub-windows, (D) paired Wilcoxon test of XGBoost vs LSTM squared error. All at horizon 28."""
    out = {"focus_ratio": FOCUS_RATIO, "ablation": {}, "bin_sensitivity": {}, "window_stability": {}}
    for model in ("xgb", "lstm"):
        fva, yva, sid, f, y, _ = load(model, 28)
        cal = fit_calibration(fva, yva)
        row = {k: policy_cost(sid, y, f, cal, FOCUS_RATIO, k) for k in POLICIES}
        row["reduction_risk_vs_point"] = 1 - row["risk_aware"] / row["point"]
        row["reduction_risk_vs_global"] = 1 - row["risk_aware"] / row["global"]
        out["ablation"][model] = row

    fva, yva, sid, f, y, dday = load("xgb", 28)
    for nb in (5, 10, 15, 20):
        cal = fit_calibration(fva, yva, n_bins=nb)
        out["bin_sensitivity"][nb] = 1 - (policy_cost(sid, y, f, cal, FOCUS_RATIO, "risk_aware")
                                          / policy_cost(sid, y, f, cal, FOCUS_RATIO, "point"))

    cal = fit_calibration(fva, yva)           # calibration fixed on the full validation split
    win = np.digitize(dday, np.quantile(dday, [1 / 3, 2 / 3]))
    for w in (0, 1, 2):
        m = win == w
        ra = evaluate(sid[m], y[m], f[m], cal, ratios=(FOCUS_RATIO,), boot=200).query(
            "policy == 'risk_aware'").iloc[0]
        out["window_stability"][f"window_{w + 1}"] = {
            "days": [int(dday[m].min()), int(dday[m].max())], "n_cases": int(m.sum()),
            "reduction_vs_point": ra.reduction_vs_point, "ci_low": ra.ci_low, "ci_high": ra.ci_high}

    xt = pd.read_parquet(PRED / "xgb_h28.parquet").query("split == 2").set_index(["sid", "d"]).sort_index()
    lt = pd.read_parquet(PRED / "lstm_h28.parquet").query("split == 2").set_index(["sid", "d"]).sort_index()
    common = xt.index.intersection(lt.index)
    ex = (xt.loc[common, "sales"] - xt.loc[common, "pred"]).to_numpy() ** 2
    el = (lt.loc[common, "sales"] - lt.loc[common, "pred"]).to_numpy() ** 2
    w = stats.wilcoxon(ex - el, zero_method="zsplit", alternative="two-sided", mode="approx")
    out["paired_model_test"] = {"n": int(len(ex)), "mean_sq_err_xgb": float(ex.mean()),
                                "mean_sq_err_lstm": float(el.mean()),
                                "lstm_lower_sq_err_share": float((el < ex).mean()),
                                "wilcoxon_stat": float(w.statistic), "p_value": float(w.pvalue)}
    return out


def main():
    costs, strata, recs, rule_counts = [], [], [], []
    for model in ("xgb", "lstm"):
        for h in (1, 28):
            fva, yva, sid, f, y, d = load(model, h)
            cal = fit_calibration(fva, yva)
            res = evaluate(sid, y, f, cal)
            risk, _, _ = risk_strata(cal, f, fva)
            st = strata_table(risk, y, f)
            costs.append(res.assign(model=model, horizon=h))
            strata.append(st.reset_index().assign(model=model, horizon=h))
            if model == "xgb":     # the rule base applied to every test case; a seeded sample is saved
                lvl, pr, act = recommend(f, fva, risk)
                rec = pd.DataFrame({"sid": sid, "d": d, "forecast": f, "actual": y, "risk": risk,
                                    "demand_level": lvl, "priority": pr, "recommendation": act})
                recs.append(rec.sample(N_REC_SAMPLE, random_state=SEED).assign(horizon=h))
                cnt = rec.groupby(["demand_level", "risk", "priority"]).size().rename("cases").reset_index()
                rule_counts.append(cnt.assign(horizon=h))
                log("decision", f"h={h}: rule base fired for {len(rec):,} test cases; rules never fired: "
                                f"{sorted(set(RULES) - set(zip(cnt.demand_level, cnt.risk, strict=True)))}")
            print(f"\n{model} h={h}\n", res[res.policy != "point"].round(4).to_string(index=False),
                  "\n", st.round(3).to_string())
    pd.concat(costs).to_csv(RESULTS / "decision_cost.csv", index=False)
    pd.concat(strata).to_csv(RESULTS / "strata.csv", index=False)
    pd.concat(recs).to_csv(RESULTS / "recommendations_sample.csv", index=False)
    pd.concat(rule_counts).to_csv(RESULTS / "rule_counts.csv", index=False)
    rob = robustness()
    (RESULTS / "robustness.json").write_text(json.dumps(rob, indent=2))
    print(json.dumps(rob, indent=2))


if __name__ == "__main__":
    main()
