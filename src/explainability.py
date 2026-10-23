"""Phase 8: SHAP attributions for the XGBoost models on a documented test-set sample.

Sample: 20,000 test rows drawn uniformly at random (seed from config). SHAP values are on the
model's log-rate scale (Poisson objective). They describe what the MODEL used, not what causes
sales. Run:  python src/explainability.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

N_SAMPLE = 20_000
KEYS = ("sid", "d", "sales", "split")
GROUPS = {"recent sales (lags, rolling)": ("lag_", "roll_"),
          "price": ("price", "listed_days"),
          "calendar and events": ("wday", "month", "day_of_month", "week_of_year", "event",
                                  "is_event", "snap"),
          "store and product hierarchy": ("dept", "cat", "store", "state")}


def group_of(name):
    for g, prefixes in GROUPS.items():
        if name.startswith(prefixes):
            return g
    return "other"


def main():
    out_dir = C.RESULTS / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for h in (1, 28):
        df = pd.read_parquet(C.FEATURES / f"features_h{h}.parquet")
        te = df[df.split == 2]
        feats = [c for c in df.columns if c not in KEYS]
        sample = te.sample(N_SAMPLE, random_state=C.SEED)
        booster = xgb.Booster()
        booster.load_model(str(C.ROOT / "models" / f"xgb_h{h}.json"))
        booster.set_param({"device": "cpu"})
        sv = shap.TreeExplainer(booster).shap_values(sample[feats])
        mean_abs = np.abs(sv).mean(axis=0)
        x = sample[feats].to_numpy(dtype=float)
        corr = [np.corrcoef(np.nan_to_num(x[:, j], nan=np.nanmedian(x[:, j])), sv[:, j])[0, 1]
                if np.std(x[:, j]) > 0 else 0.0 for j in range(len(feats))]
        t = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs,
                          "share": mean_abs / mean_abs.sum(), "value_shap_corr": corr,
                          "group": [group_of(f) for f in feats]}).sort_values(
            "mean_abs_shap", ascending=False)
        t.to_csv(out_dir / f"shap_global_h{h}.csv", index=False)
        np.save(out_dir / f"shap_values_h{h}.npy", sv.astype(np.float32))
        sample[feats].to_parquet(out_dir / f"shap_sample_h{h}.parquet")
        gshare = t.groupby("group").share.sum().sort_values(ascending=False)
        summary[f"h{h}"] = {"n_sample": N_SAMPLE, "seed": C.SEED,
                            "top5": t.head(5)[["feature", "share"]].to_dict("records"),
                            "group_share": gshare.round(4).to_dict()}
        print(f"\nh={h} top features:\n", t.head(8).round(4).to_string(index=False))
        print("group share:", gshare.round(3).to_dict())
    (out_dir / "shap_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
