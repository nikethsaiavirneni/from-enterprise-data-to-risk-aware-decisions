"""Model stage: XGBoost (GPU) and a small LSTM per horizon, plus SHAP attributions.

Selection uses validation only; the test split is scored once for the chosen model. SHAP values are
on the XGBoost log-rate scale (Poisson objective) and describe what the MODEL used, not what causes
sales.

Run:  python src/models.py xgb [h ...]     XGBoost + naive baselines (default horizons 1 28)
      python src/models.py lstm [h ...]    LSTM (GPU memory capped at 4 GB)
      python src/models.py shap            SHAP on 20,000 random test rows per horizon
Outputs: results/forecast_metrics.json, results/shap.json, results/predictions/*.parquet (ignored).
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from data import FEATURES, HORIZONS, PROCESSED, RESULTS, ROOT, SEED, all_metrics, log, scale_by_sid

KEYS = ("sid", "d", "sales", "split")
PRED, MODELS = RESULTS / "predictions", ROOT / "models"
METRICS_FILE = RESULTS / "forecast_metrics.json"
MAX_GPU_MEMORY_GB = 4.0

# XGBoost: deliberately small tuning budget (4 configs), Poisson objective
BASELINES = {1: "lag_7", 28: "lag_28"}  # same-weekday naive for each horizon
GRID = [
    {"max_depth": 6, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 8, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 6, "min_child_weight": 200, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 10, "min_child_weight": 200, "subsample": 0.7, "colsample_bytree": 0.7},
]
BASE = {"objective": "count:poisson", "eta": 0.1, "tree_method": "hist", "device": "cuda",
        "eval_metric": "rmse", "seed": SEED, "max_bin": 128, "reg_lambda": 1.0}
MAX_ROUNDS, EARLY = 500, 30

# LSTM: last 28 days of log1p(sales) ending at t-h, plus known-in-advance covariates of day t
SEQ, HIDDEN, EMB, BATCH = 28, 64, 8, 256
SAMPLES_PER_EPOCH, MAX_EPOCHS, PATIENCE, LR = 2_000_000, 12, 3, 2e-3
NUM_COLS = ["price", "price_ratio_7", "price_rel_hist", "listed_days", "is_event", "snap"]
CAT_COLS = {"wday": 8, "month": 13, "event_type": 8, "store": 10, "dept": 7}

# SHAP
N_SAMPLE = 20_000
GROUPS = {"recent sales (lags, rolling)": ("lag_", "roll_"),
          "price": ("price", "listed_days"),
          "calendar and events": ("wday", "month", "day_of_month", "week_of_year", "event",
                                  "is_event", "snap"),
          "store and product hierarchy": ("dept", "cat", "store", "state")}


def save_metrics(model, h, out):
    """Merge one run into results/forecast_metrics.json as {model: {horizon: record}}."""
    RESULTS.mkdir(exist_ok=True)
    data = json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else {}
    data.setdefault(model, {})[str(h)] = out
    METRICS_FILE.write_text(json.dumps(data, indent=2))


def xgboost_run(h):
    import xgboost as xgb
    df = pd.read_parquet(FEATURES / f"features_h{h}.parquet")
    feats = [c for c in df.columns if c not in KEYS]
    tr, va, te = (df[df.split == k] for k in (0, 1, 2))
    scale = scale_by_sid(tr)
    print(f"\n=== XGBoost h={h}: train {len(tr)} val {len(va)} test {len(te)}, {len(feats)} features ===")

    dtr = xgb.QuantileDMatrix(tr[feats], tr.sales, max_bin=BASE["max_bin"])
    dva = xgb.QuantileDMatrix(va[feats], va.sales, ref=dtr, max_bin=BASE["max_bin"])
    dte = xgb.DMatrix(te[feats])

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
        print(f"cfg{i} {cfg} -> rounds {b.best_iteration + 1}, val RMSE {m['RMSE']:.4f}, {secs:.0f}s")
        if best is None or m["RMSE"] < best[0]:
            best = (m["RMSE"], i, b)
    _, bi, booster = best

    n = booster.best_iteration + 1
    t0 = time.time()
    pt = booster.predict(dte, iteration_range=(0, n))
    infer = time.time() - t0
    pv = booster.predict(dva, iteration_range=(0, n))

    base_col = BASELINES[h]
    out = {"horizon": h, "chosen_config": GRID[bi], "chosen_index": bi, "rounds": n,
           "tuning": results, "inference_seconds_test": round(infer, 2), "n_features": len(feats),
           "features": feats, "baselines": {"seasonal_naive": base_col, "roll_mean_28": "roll_mean_28"}}
    for name, part, pred in (("val", va, pv), ("test", te, pt)):
        y, sid = part.sales.to_numpy(), part.sid.to_numpy()
        out[f"xgboost_{name}"] = all_metrics(y, pred, sid, scale)
        out[f"seasonal_naive_{name}"] = all_metrics(y, part[base_col].to_numpy(), sid, scale)
        out[f"roll_mean_28_{name}"] = all_metrics(y, part["roll_mean_28"].to_numpy(), sid, scale)

    PRED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    booster.save_model(str(MODELS / f"xgb_h{h}.json"))
    keep = ["sid", "d", "split", "sales", base_col, "roll_mean_28"]
    pd.concat([va[keep].assign(pred=pv), te[keep].assign(pred=pt)]).rename(
        columns={base_col: "naive_pred"}).to_parquet(PRED / f"xgb_h{h}.parquet", index=False)
    save_metrics("xgb", h, out)
    for k in ("val", "test"):
        log("xgb", f"h={h} {k}: " + ", ".join(f"{m} {v:.4f}" for m, v in out[f"xgboost_{k}"].items())
            + " | naive RMSE " + f"{out[f'seasonal_naive_{k}']['RMSE']:.4f}")
    log("xgb", f"h={h} chosen config {GRID[bi]}, {n} rounds, tuning seconds "
               f"{[r['train_seconds'] for r in results]}")


def lstm_run(h, device):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
            self.emb = nn.ModuleDict({k: nn.Embedding(n, EMB) for k, n in CAT_COLS.items()})
            self.head = nn.Sequential(nn.Linear(HIDDEN + EMB * len(CAT_COLS) + len(NUM_COLS), 64),
                                      nn.ReLU(), nn.Linear(64, 1))

        def forward(self, seq, num, cat):
            _, (hid, _) = self.lstm(seq.unsqueeze(-1))
            z = torch.cat([hid[-1], num] + [self.emb[k](cat[:, i]) for i, k in enumerate(CAT_COLS)], 1)
            return self.head(z).squeeze(-1)  # log-rate

    def prep_known(p):
        num = p[NUM_COLS].to_numpy(np.float32).copy()
        num[:, 0] = np.log1p(num[:, 0])                 # price
        num[:, 3] = np.log1p(num[:, 3])                 # listed_days
        num[:, 1:3] = np.nan_to_num(num[:, 1:3], nan=1.0)
        return num, p[list(CAT_COLS)].to_numpy(np.int64)

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(FEATURES / f"features_h{h}.parquet",
                         columns=list(KEYS) + NUM_COLS + list(CAT_COLS))
    long = pd.read_parquet(PROCESSED / "long.parquet", columns=["sid", "d", "sales"])
    sid_u = np.sort(df.sid.unique())
    row_of = np.full(sid_u.max() + 1, -1, dtype=np.int64)
    row_of[sid_u] = np.arange(len(sid_u))
    ylog = np.zeros((len(sid_u), int(long.d.max())), dtype=np.float32)
    ylog[row_of[long.sid.to_numpy()], long.d.to_numpy() - 1] = np.log1p(long.sales.to_numpy())
    ylog_g = torch.from_numpy(ylog).to(device)
    offs = torch.arange(-(h + SEQ - 1), -h + 1, device=device)   # days t-h-27 .. t-h

    parts, tens = {}, {}
    for k, v in (("train", 0), ("val", 1), ("test", 2)):
        p = df[df.split == v].reset_index(drop=True)
        num, cat = prep_known(p)
        parts[k] = p
        tens[k] = (torch.from_numpy(row_of[p.sid.to_numpy()]).to(device),
                   torch.from_numpy(p.d.to_numpy().astype(np.int64)).to(device),
                   torch.from_numpy(num).to(device), torch.from_numpy(cat).to(device),
                   torch.from_numpy(p.sales.to_numpy().astype(np.float32)).to(device))
    scale = scale_by_sid(df[df.split == 0])
    del df, long

    def batch(k, idx):
        r, d, num, cat, y = (t[idx] for t in tens[k])
        return ylog_g[r[:, None], d[:, None] - 1 + offs[None, :]], num, cat, y

    def predict(k, bs=8192):
        net.eval()
        out = []
        with torch.no_grad():
            n = len(parts[k])
            for i in range(0, n, bs):
                seq, num, cat, _ = batch(k, torch.arange(i, min(i + bs, n), device=device))
                out.append(torch.exp(net(seq, num, cat).clamp(-10, 10)).cpu())
        return torch.cat(out).numpy()

    def val_loss():
        p, y = predict("val"), parts["val"].sales.to_numpy()
        return float(np.mean(p - y * np.log(p + 1e-8)))

    net = Net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.PoissonNLLLoss(log_input=True)
    n_train = len(parts["train"])
    hist, best, bad, best_state = [], np.inf, 0, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        perm = torch.from_numpy(rng.choice(n_train, SAMPLES_PER_EPOCH, replace=False)).to(device)
        tot = 0.0
        for i in range(0, SAMPLES_PER_EPOCH, BATCH):
            seq, num, cat, y = batch("train", perm[i:i + BATCH])
            loss = loss_fn(net(seq, num, cat).clamp(-10, 10), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
        vl = val_loss()
        sched.step(vl)
        hist.append({"epoch": ep, "train_loss": tot / SAMPLES_PER_EPOCH, "val_loss": vl,
                     "seconds": round(time.time() - t_start, 1)})
        print(f"h={h} epoch {ep}: train {hist[-1]['train_loss']:.4f} val {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    train_secs = time.time() - t_start
    net.load_state_dict(best_state)
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0
    assert peak_mb < MAX_GPU_MEMORY_GB * 1024, "GPU memory cap exceeded"

    t0 = time.time()
    pt = predict("test")
    infer = time.time() - t0
    pv = predict("val")
    out = {"horizon": h, "device": str(device), "epochs_run": len(hist), "history": hist,
           "best_val_poisson_nll": best, "train_seconds": round(train_secs, 1),
           "inference_seconds_test": round(infer, 2), "gpu_peak_allocated_mb": round(peak_mb, 1),
           "params": sum(p.numel() for p in net.parameters()),
           "config": {"seq": SEQ, "hidden": HIDDEN, "emb": EMB, "batch": BATCH, "lr": LR,
                      "samples_per_epoch": SAMPLES_PER_EPOCH, "patience": PATIENCE}}
    for k, p in (("val", pv), ("test", pt)):
        out[f"lstm_{k}"] = all_metrics(parts[k].sales.to_numpy(), p, parts[k].sid.to_numpy(), scale)
    save_metrics("lstm", h, out)
    MODELS.mkdir(exist_ok=True)
    torch.save(net.state_dict(), MODELS / f"lstm_h{h}.pt")
    PRED.mkdir(parents=True, exist_ok=True)
    pd.concat([parts["val"][list(KEYS)].assign(pred=pv),
               parts["test"][list(KEYS)].assign(pred=pt)]).to_parquet(
        PRED / f"lstm_h{h}.parquet", index=False)
    for k in ("val", "test"):
        log("lstm", f"h={h} {k}: " + ", ".join(f"{m} {v:.4f}" for m, v in out[f"lstm_{k}"].items()))
    log("lstm", f"h={h} {len(hist)} epochs, GPU peak {peak_mb:.0f} MB (cap {int(MAX_GPU_MEMORY_GB * 1024)}), "
                f"train {train_secs:.0f}s, params {out['params']}")


def group_of(name):
    return next((g for g, prefixes in GROUPS.items() if name.startswith(prefixes)), "other")


def shap_run():
    import shap
    import xgboost as xgb
    summary = {}
    for h in HORIZONS:
        df = pd.read_parquet(FEATURES / f"features_h{h}.parquet")
        feats = [c for c in df.columns if c not in KEYS]
        sample = df[df.split == 2].sample(N_SAMPLE, random_state=SEED)
        booster = xgb.Booster()
        booster.load_model(str(MODELS / f"xgb_h{h}.json"))
        booster.set_param({"device": "cpu"})
        mean_abs = np.abs(shap.TreeExplainer(booster).shap_values(sample[feats])).mean(axis=0)
        t = pd.DataFrame({"feature": feats, "share": mean_abs / mean_abs.sum(),
                          "group": [group_of(f) for f in feats]}).sort_values("share", ascending=False)
        gshare = t.groupby("group").share.sum().sort_values(ascending=False)
        summary[f"h{h}"] = {"n_sample": N_SAMPLE, "seed": SEED,
                            "top5": t.head(5).drop(columns="group").to_dict("records"),
                            "group_share": gshare.round(4).to_dict()}
        print(f"\nh={h} top features:\n", t.head(8).round(4).to_string(index=False))
        print("group share:", gshare.round(3).to_dict())
    (RESULTS / "shap.json").write_text(json.dumps(summary, indent=2))
    for h, v in summary.items():
        log("shap", f"{h}: n={v['n_sample']}, top feature {v['top5'][0]['feature']}, group share {v['group_share']}")


def main():
    stage, hs = sys.argv[1] if len(sys.argv) > 1 else "", [int(a) for a in sys.argv[2:]] or list(HORIZONS)
    if stage == "xgb":
        for h in hs:
            xgboost_run(h)
    elif stage == "lstm":
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
        for h in hs:
            lstm_run(h, dev)
    elif stage == "shap":
        shap_run()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
