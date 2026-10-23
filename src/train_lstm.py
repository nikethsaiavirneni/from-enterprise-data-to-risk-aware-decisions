"""Phase 6: small LSTM forecaster (GPU, memory-capped) for each horizon.

Input for target day t: the last 28 days of log1p(sales) ending at day t-h (sequence), plus
known-in-advance covariates of day t (price context, calendar, store/department).
Output: Poisson rate (same objective as XGBoost). Windows are gathered on the GPU per batch
from one dense (series x day) matrix, so no sequence tensor is ever materialised.
Run:  python src/train_lstm.py [h ...]     (default: 1 28)
"""
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from metrics import all_metrics
from train_xgboost import scale_by_sid

SEQ = 28
HIDDEN, EMB = 64, 8
BATCH = 256              # larger than 16-32: 13M training rows; GPU memory is logged against the cap
SAMPLES_PER_EPOCH = 2_000_000
MAX_EPOCHS, PATIENCE, LR = 12, 3, 2e-3
NUM_COLS = ["price", "price_ratio_7", "price_rel_hist", "listed_days", "is_event", "snap"]
CAT_COLS = {"wday": 8, "month": 13, "event_type": 8, "store": 10, "dept": 7}


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
        self.emb = nn.ModuleDict({k: nn.Embedding(n, EMB) for k, n in CAT_COLS.items()})
        self.head = nn.Sequential(nn.Linear(HIDDEN + EMB * len(CAT_COLS) + len(NUM_COLS), 64),
                                  nn.ReLU(), nn.Linear(64, 1))

    def forward(self, seq, num, cat):
        _, (h, _) = self.lstm(seq.unsqueeze(-1))
        z = torch.cat([h[-1], num] + [self.emb[k](cat[:, i]) for i, k in enumerate(CAT_COLS)], 1)
        return self.head(z).squeeze(-1)  # log-rate


def prep_known(df):
    num = df[NUM_COLS].to_numpy(np.float32).copy()
    num[:, 0] = np.log1p(num[:, 0])                 # price
    num[:, 3] = np.log1p(num[:, 3])                 # listed_days
    num[:, 1:3] = np.nan_to_num(num[:, 1:3], nan=1.0)
    cat = df[list(CAT_COLS)].to_numpy(np.int64)
    return num, cat


def run(h, device):
    torch.manual_seed(C.SEED)
    rng = np.random.default_rng(C.SEED)
    df = pd.read_parquet(C.FEATURES / f"features_h{h}.parquet",
                         columns=["sid", "d", "sales", "split"] + NUM_COLS + list(CAT_COLS))
    long = pd.read_parquet(C.PROCESSED / "long.parquet", columns=["sid", "d", "sales"])
    sid_u = np.sort(df.sid.unique())
    row_of = np.full(sid_u.max() + 1, -1, dtype=np.int64)
    row_of[sid_u] = np.arange(len(sid_u))
    D = int(long.d.max())
    ylog = np.zeros((len(sid_u), D), dtype=np.float32)
    ylog[row_of[long.sid.to_numpy()], long.d.to_numpy() - 1] = np.log1p(long.sales.to_numpy())
    ylog_g = torch.from_numpy(ylog).to(device)
    offs = torch.arange(-(h + SEQ - 1), -h + 1, device=device)   # days t-h-27 .. t-h (0-based: t-1+offs)

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
        seq = ylog_g[r[:, None], d[:, None] - 1 + offs[None, :]]
        return seq, num, cat, y

    def predict(k, bs=8192):
        net.eval()
        out = []
        with torch.no_grad():
            n = len(parts[k])
            for i in range(0, n, bs):
                idx = torch.arange(i, min(i + bs, n), device=device)
                seq, num, cat, _ = batch(k, idx)
                out.append(torch.exp(net(seq, num, cat).clamp(-10, 10)).cpu())
        return torch.cat(out).numpy()

    def val_loss():
        p = predict("val")
        y = parts["val"].sales.to_numpy()
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
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    assert peak_mb < C.MAX_GPU_MEMORY_GB * 1024, "GPU memory cap exceeded"

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
    (C.METRICS / f"lstm_h{h}.json").write_text(json.dumps(out, indent=2))
    pd.concat([parts["val"][["sid", "d", "split", "sales"]].assign(pred=pv),
               parts["test"][["sid", "d", "split", "sales"]].assign(pred=pt)]).to_parquet(
        C.RESULTS / "predictions" / f"lstm_h{h}.parquet", index=False)
    torch.save(net.state_dict(), C.ROOT / "models" / f"lstm_h{h}.pt")

    with open(C.RESULTS / "gpu_usage.csv", "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), f"lstm_h{h}", 0,
                                round(peak_mb), int(C.MAX_GPU_MEMORY_GB * 1024)])
    t = out["lstm_test"]
    with open(C.ROOT / "experiment_log.csv", "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), "lstm", h, C.SEED,
                                json.dumps(out["config"]), len(hist),
                                round(out["lstm_val"]["RMSE"], 5), round(t["RMSE"], 5),
                                round(t["MAE"], 5), round(t["WAPE"], 5)])
    for k in ("val", "test"):
        print(k, {m: round(v, 4) for m, v in out[f"lstm_{k}"].items()})
    print(f"GPU peak allocated {peak_mb:.0f} MB (cap {int(C.MAX_GPU_MEMORY_GB * 1024)}), "
          f"train {train_secs:.0f}s, params {out['params']}")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    for h in ([int(a) for a in sys.argv[1:]] or [1, 28]):
        run(h, dev)
