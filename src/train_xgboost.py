"""Phase 5: XGBoost (GPU) per horizon, small validation-based tuning, plus naive baselines.

Selection uses validation only; the test split is scored once for the chosen config.
Run:  python src/train_xgboost.py [h ...]      (default: 1 28)
"""
import csv
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from metrics import all_metrics, naive_scale

KEYS = ("sid", "d", "sales", "split")
BASELINES = {1: "lag_7", 28: "lag_28"}  # same-weekday naive for each horizon
GRID = [  # deliberately small tuning budget (4 configs)
    {"max_depth": 6, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 8, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "min_child_weight": 200, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 10, "min_child_weight": 200, "subsample": 0.7, "colsample_bytree": 0.7},
]
BASE = {"objective": "count:poisson", "eta": 0.1, "tree_method": "hist", "device": "cuda",
        "eval_metric": "rmse", "seed": C.SEED, "max_bin": 128, "reg_lambda": 1.0}
MAX_ROUNDS, EARLY = 500, 30


def gpu_used_mb():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                       "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


class GpuPoller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop_flag = gpu_used_mb(), threading.Event()

    def run(self):
        while not self.stop_flag.wait(1.0):
            self.peak = max(self.peak, gpu_used_mb())


def scale_by_sid(train):
    tr = train.sort_values(["sid", "d"])
    parts = np.split(tr.sales.to_numpy(), np.flatnonzero(np.diff(tr.sid.to_numpy())) + 1)
    sids = tr.sid.drop_duplicates().to_numpy()
    out = np.full(sids.max() + 1, np.nan)
    out[sids] = naive_scale(parts)
    return out


def run(h):
    df = pd.read_parquet(C.FEATURES / f"features_h{h}.parquet")
    feats = [c for c in df.columns if c not in KEYS]
    tr, va, te = (df[df.split == k] for k in (0, 1, 2))
    scale = scale_by_sid(tr)
    print(f"\n=== h={h}: train {len(tr)} val {len(va)} test {len(te)}, {len(feats)} features ===")

    dtr = xgb.QuantileDMatrix(tr[feats], tr.sales, max_bin=BASE["max_bin"])
    dva = xgb.QuantileDMatrix(va[feats], va.sales, ref=dtr, max_bin=BASE["max_bin"])
    dte = xgb.DMatrix(te[feats])

    poll = GpuPoller()
    gpu_base = gpu_used_mb()
    poll.start()
    results, best = [], None
    for i, cfg in enumerate(GRID):
        t0 = time.time()
        b = xgb.train({**BASE, **cfg}, dtr, MAX_ROUNDS, evals=[(dva, "val")],
                      early_stopping_rounds=EARLY, verbose_eval=False)
        secs = time.time() - t0
        pv = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
        m = all_metrics(va.sales.to_numpy(), pv, va.sid.to_numpy(), scale)
        results.append({"config": cfg, "best_iteration": b.best_iteration,
                        "train_seconds": round(secs, 1), "val": m})
        print(f"cfg{i} {cfg} -> rounds {b.best_iteration + 1}, val RMSE {m['RMSE']:.4f}, "
              f"MAE {m['MAE']:.4f}, {secs:.0f}s")
        if best is None or m["RMSE"] < best[0]:
            best = (m["RMSE"], i, b)
    poll.stop_flag.set()
    _, bi, booster = best

    t0 = time.time()
    n = booster.best_iteration + 1
    pt = booster.predict(dte, iteration_range=(0, n))
    infer = time.time() - t0
    pv = booster.predict(dva, iteration_range=(0, n))

    out = {"horizon": h, "chosen_config": GRID[bi], "chosen_index": bi, "rounds": n,
           "tuning": results, "inference_seconds_test": round(infer, 2),
           "gpu_mb_baseline": gpu_base, "gpu_mb_peak": poll.peak,
           "gpu_mb_peak_above_baseline": poll.peak - gpu_base, "n_features": len(feats),
           "features": feats}
    base_col = BASELINES[h]
    for name, part, pred in (("val", va, pv), ("test", te, pt)):
        y, sid = part.sales.to_numpy(), part.sid.to_numpy()
        out[f"xgboost_{name}"] = all_metrics(y, pred, sid, scale)
        out[f"seasonal_naive_{name}"] = all_metrics(y, part[base_col].to_numpy(), sid, scale)
        out[f"roll_mean_28_{name}"] = all_metrics(y, part["roll_mean_28"].to_numpy(), sid, scale)
    out["baselines"] = {"seasonal_naive": base_col, "roll_mean_28": "roll_mean_28"}

    C.METRICS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "predictions").mkdir(exist_ok=True)
    (C.ROOT / "models").mkdir(exist_ok=True)
    booster.save_model(str(C.ROOT / "models" / f"xgb_h{h}.json"))
    keep = ["sid", "d", "split", "sales", base_col, "roll_mean_28"]
    pred_df = pd.concat([va[keep].assign(pred=pv), te[keep].assign(pred=pt)])
    pred_df.rename(columns={base_col: "naive_pred"}).to_parquet(
        C.RESULTS / "predictions" / f"xgb_h{h}.parquet", index=False)
    (C.METRICS / f"xgb_h{h}.json").write_text(json.dumps(out, indent=2))

    gpu_log = C.RESULTS / "gpu_usage.csv"
    new = not gpu_log.exists()
    with open(gpu_log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "stage", "baseline_mb", "peak_mb", "cap_mb"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), f"xgboost_h{h}", gpu_base,
                    poll.peak, int(C.MAX_GPU_MEMORY_GB * 1024)])
    log = C.ROOT / "experiment_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "model", "horizon", "seed", "config", "rounds",
                        "val_RMSE", "test_RMSE", "test_MAE", "test_WAPE"])
        t = out["xgboost_test"]
        w.writerow([datetime.now().isoformat(timespec="seconds"), "xgboost", h, C.SEED,
                    json.dumps(GRID[bi]), n, round(out["xgboost_val"]["RMSE"], 5),
                    round(t["RMSE"], 5), round(t["MAE"], 5), round(t["WAPE"], 5)])
    for k in ("val", "test"):
        print(k, {m: round(v, 4) for m, v in out[f"xgboost_{k}"].items()}, "| naive",
              {m: round(v, 4) for m, v in out[f"seasonal_naive_{k}"].items()})
    print(f"GPU MB baseline {gpu_base}, peak {poll.peak} (cap {int(C.MAX_GPU_MEMORY_GB * 1024)})")


if __name__ == "__main__":
    for h in ([int(a) for a in sys.argv[1:]] or [1, 28]):
        run(h)
