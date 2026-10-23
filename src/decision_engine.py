"""Phases 9-10: validation-calibrated risk strata, rule-based recommendations, and a decision-cost test.

Everything that sets a number (residual quantiles, forecast bins, risk cut-offs) is fitted on the
VALIDATION split and then applied unchanged to TEST. Cost ratios and recommendation rules are
experimental constructs, not Walmart policy.

Policies compared on identical test cases (order quantity Q for day-t demand D):
  point        Q = f
  global       Q = f + q_tau(validation residuals)              (one correction for all items)
  risk_aware   Q = f + q_tau(validation residuals | forecast bin)
with tau = Cu / (Cu + Co), the newsvendor critical ratio, and residual e = D - f.
Cost = Cu * max(D - Q, 0) + Co * max(Q - D, 0), reported per unit of realised demand.
Run:  python src/decision_engine.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

RATIOS = (1, 2, 4, 8, 16)         # Cu:Co scenarios (Co = 1)
N_BINS, BOOT, SEED = 10, 500, C.SEED
POLICIES = ("point", "global", "risk_aware")


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


# Experimental recommendation rules: (demand level, risk) -> (priority, action)
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
            rows.append(row)
    return pd.DataFrame(rows)


def strata_table(risk, y, f):
    d = pd.DataFrame({"risk": risk, "ae": np.abs(y - f), "under": y > f, "y": y})
    g = d.groupby("risk").agg(cases=("ae", "size"), mae=("ae", "mean"), mean_demand=("y", "mean"),
                              understock_rate=("under", "mean"))
    g["wape"] = d.groupby("risk").ae.sum() / d.groupby("risk").y.sum()
    return g.reindex(["Low", "Moderate", "High"])


def main():
    out_dir = C.RESULTS / "decision_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for model in ("xgb", "lstm"):
        for h in (1, 28):
            p = C.RESULTS / "predictions" / f"{model}_h{h}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            va, te = df[df.split == 1], df[df.split == 2].reset_index(drop=True)
            f_va, y_va = va.pred.to_numpy().astype(float), va.sales.to_numpy().astype(float)
            f, y = te.pred.to_numpy().astype(float), te.sales.to_numpy().astype(float)
            cal = fit_calibration(f_va, y_va)
            res = evaluate(te.sid.to_numpy(), y, f, cal)
            res.insert(0, "horizon", h)
            res.insert(0, "model", model)
            risk, cuts, _ = risk_strata(cal, f, f_va)
            st = strata_table(risk, y, f)
            res.to_csv(out_dir / f"cost_{model}_h{h}.csv", index=False)
            st.to_csv(out_dir / f"strata_{model}_h{h}.csv")
            summary[f"{model}_h{h}"] = {"risk_cuts": cuts.tolist(), "strata": st.reset_index().to_dict("records")}
            if model == "xgb":
                lvl, pr, act = recommend(f, f_va, risk)
                rec = pd.DataFrame({"sid": te.sid, "d": te.d, "forecast": f, "actual": y,
                                    "error": y - f, "risk": risk, "demand_level": lvl,
                                    "priority": pr, "recommendation": act})
                rec.to_csv(out_dir / f"decision_recommendations_h{h}.csv", index=False)
            print(f"\n{model} h={h}\n", res[res.policy != "point"].round(4).to_string(index=False),
                  "\n", st.round(3).to_string())
    (out_dir / "decision_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
