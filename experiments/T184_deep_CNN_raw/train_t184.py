"""T184: TickTransformer on raw 100-tick LOB window (v2 cache: 31 features).

M7 protocol: full data 0-119, no val, 15 epochs fixed.
Architecture: 2-layer Transformer with learnable positional encoding.
Window normalization: per-window per-feature standardize (critical from T95 analysis).

CRITICAL_CONSTRAINTS compliance:
  - No sym embedding, no date features
  - No cross-call state (stateless forward)
  - Sym-agnostic (global normalization statistics only)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T95_DIR = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob")
CACHE_DIR = os.path.join(T95_DIR, "cache")
FEE = 0.0001
NUM_CLASS = 3


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


class TickTransformer(nn.Module):
    """1D Transformer over 100-tick LOB window. Input (B, W, F).

    Window normalization is applied OUTSIDE this module (in training loop and Predictor).
    """
    def __init__(self, n_features: int = 31, d_model: int = 64,
                 n_layers: int = 2, nhead: int = 4, ff_dim: int = 256,
                 dropout: float = 0.1, window: int = 100):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_layers = n_layers
        self.nhead = nhead
        self.window = window

        self.proj = nn.Linear(n_features, d_model)
        self.pe = nn.Embedding(window, d_model)  # learnable positional encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,  # pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F) - already window-normalized
        B, W, F = x.shape
        pos = torch.arange(W, device=x.device)
        tokens = self.proj(x) + self.pe(pos).unsqueeze(0)  # (B, W, d_model)
        tokens = self.encoder(tokens)  # (B, W, d_model)
        # Use last token (most recent tick) for prediction
        last = tokens[:, -1, :]  # (B, d_model)
        last = self.dropout(last)
        return self.head(last).squeeze(-1)  # (B,)


def window_normalize(x_raw: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
    """Per-window per-feature standardize. x_raw: (B, W, F)."""
    m = x_raw.mean(dim=1, keepdim=True)
    s = x_raw.std(dim=1, keepdim=True)
    x = (x_raw - m) / s.clamp_min(1e-8)
    return x.clamp(-clip, clip)


class WindowDataset(Dataset):
    def __init__(self, lob_arr: np.ndarray, sess_id: np.ndarray, t: np.ndarray,
                 y: np.ndarray, sw: np.ndarray, window: int = 100):
        self.lob_arr = lob_arr  # (n_sess, T_max, F)
        self.sess_id = sess_id
        self.t = t
        self.y = y
        self.sw = sw
        self.W = window
        valid = t >= (window - 1)
        self.idx = np.where(valid)[0].astype(np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        s = int(self.sess_id[j])
        ti = int(self.t[j])
        win = self.lob_arr[s, ti - self.W + 1: ti + 1]  # (W, F)
        return win.astype(np.float32), self.y[j], self.sw[j]


def class_balanced_weight(y_cls: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y_cls, minlength=num_class).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights_per_class = 1.0 / counts
    weights_per_class /= weights_per_class.mean()
    return weights_per_class[y_cls].astype(np.float32)


def predict_chunked(model, dataset, batch=2048, device="cuda", num_workers=2):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    preds = np.zeros(len(dataset), dtype=np.float32)
    pos = 0
    with torch.no_grad():
        for win, y, sw in loader:
            win = win.to(device, non_blocking=True)
            xb = window_normalize(win)
            yb = model(xb).detach().cpu().numpy()
            preds[pos: pos + len(yb)] = yb
            pos += len(yb)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="v2", choices=["top5", "v2"])
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--aug-scale", type=float, default=0.02)
    ap.add_argument("--aug-noise", type=float, default=0.01)
    ap.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(f"=== T184 TickTransformer M7 window={args.window} seed={args.seed} ===", flush=True)
    print(f"  device={device}", flush=True)

    progress("loading_cache", seed=args.seed)
    t0 = time.time()
    train_full = np.load(os.path.join(CACHE_DIR, f"lob_train_{args.scheme}.npz"))
    test_full = np.load(os.path.join(CACHE_DIR, f"lob_test_{args.scheme}.npz"))
    print(f"  loaded cache in {time.time()-t0:.1f}s", flush=True)

    lob_tr = train_full["lob_arr"]        # (n_sess, T_max, F)
    sid_tr = train_full["sess_id_for_sample"]
    t_tr = train_full["t_for_sample"]
    y_cls_tr = train_full["y60"].astype(np.int64)
    mp_t_tr = train_full["mp_t"]
    mp_th_tr = train_full["mp_t60"]
    y_tr = regr_target(mp_t_tr, mp_th_tr)

    # M7: full data (no val), all 0-119
    # Compute target scale on full train
    target_scale = 1.0 / float(max(y_tr.std(), 1e-8))
    y_tr_s = (y_tr * target_scale).astype(np.float32)
    sw_tr = class_balanced_weight(y_cls_tr, NUM_CLASS)

    lob_te = test_full["lob_arr"]
    sid_te = test_full["sess_id_for_sample"]
    t_te = test_full["t_for_sample"]
    y_cls_te = test_full["y60"].astype(np.int64)
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]
    y_te = regr_target(mp_t_te, mp_th_te)

    F_dim = lob_tr.shape[-1]
    print(f"  n_train={len(y_tr):,}  n_test={len(y_te):,}  F={F_dim}  target_scale={target_scale:.4f}",
          flush=True)

    ds_train = WindowDataset(lob_tr, sid_tr, t_tr, y_tr_s, sw_tr, args.window)
    ds_test = WindowDataset(lob_te, sid_te, t_te,
                             np.zeros(len(t_te), dtype=np.float32),
                             np.ones(len(t_te), dtype=np.float32), args.window)
    print(f"  ds_train={len(ds_train):,}  ds_test={len(ds_test):,}", flush=True)

    model = TickTransformer(
        n_features=F_dim, d_model=args.d_model, n_layers=args.n_layers,
        nhead=args.nhead, ff_dim=args.ff_dim, dropout=args.dropout,
        window=args.window,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model n_params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CLIP = 10.0

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wm
            wandb = wm
            run_name = f"T184-Transformer-seed{args.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T184_raw_transformer_M7",
                    "scheme": args.scheme,
                    "F_dim": F_dim, "window": args.window, "seed": args.seed,
                    "d_model": args.d_model, "n_layers": args.n_layers,
                    "nhead": args.nhead, "ff_dim": args.ff_dim,
                    "dropout": args.dropout, "lr": args.lr, "epochs": args.epochs,
                    "n_params": n_params,
                },
                tags=["T184", "TickTransformer", "raw-LOB", "M7", "regression"],
            )
        except Exception as e:
            print(f"  WandB init failed: {e!r}", flush=True)
            use_wandb = False

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False,
                              persistent_workers=(args.num_workers > 0))

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None
    use_amp = scaler is not None
    print(f"  mixed precision: {use_amp}", flush=True)

    progress("training", n_train=len(ds_train), seed=args.seed)
    t_train_start = time.time()
    epoch_logs = []

    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        rl, n_seen = 0.0, 0
        for win, y, sw in train_loader:
            win = win.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True)

            # Augmentation: multiplicative scale on raw before window-norm
            if args.aug_scale > 0:
                scl = torch.empty(win.size(0), 1, win.size(2), device=device).uniform_(
                    1.0 - args.aug_scale, 1.0 + args.aug_scale)
                win = win * scl

            xb = window_normalize(win, CLIP)

            if args.aug_noise > 0:
                xb = xb + torch.randn_like(xb) * args.aug_noise

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(xb)
                    mse_per = (pred - y) ** 2
                    loss = (mse_per * sw).sum() / sw.sum().clamp_min(1.0)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(xb)
                mse_per = (pred - y) ** 2
                loss = (mse_per * sw).sum() / sw.sum().clamp_min(1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            bs = xb.size(0)
            rl += float(loss.item()) * bs
            n_seen += bs

        scheduler.step()
        train_loss = rl / max(n_seen, 1)
        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {"epoch": epoch, "train_loss": train_loss, "lr": cur_lr, "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d}  train_loss={train_loss:.6e}  lr={cur_lr:.2e}  ({ep_time:.1f}s)",
              flush=True)
        if use_wandb:
            wandb.log(log)

    train_time = time.time() - t_train_start
    print(f"\n  train done {train_time:.1f}s", flush=True)

    # Test predictions (for holdout eval)
    t0_pred = time.time()
    te_pred_s = predict_chunked(model, ds_test, batch=2048, device=device, num_workers=2)
    te_pred = te_pred_s / target_scale  # unscale
    pred_time = time.time() - t0_pred
    print(f"  test pred done in {pred_time:.1f}s for {len(te_pred):,} rows", flush=True)

    test_valid_mask = t_te >= (args.window - 1)
    y_te_valid = y_te[test_valid_mask]
    te_mse = float(((te_pred - y_te_valid) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_te_valid)[0, 1]) if te_pred.std() > 0 else 0.0
    print(f"  TEST: mse={te_mse:.7e}  corr={te_corr:.4f}", flush=True)

    # Quick EV-gate PnL
    fee_thr = 2.0 * FEE
    te_pred_full = np.full(len(t_te), np.nan, dtype=np.float32)
    te_pred_full[test_valid_mask] = te_pred
    valid_pred = ~np.isnan(te_pred_full)
    pred_action = np.full(len(te_pred_full), 1, dtype=np.int8)
    pred_action[valid_pred & (te_pred_full > fee_thr)] = 2
    pred_action[valid_pred & (te_pred_full < -fee_thr)] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  TEST EV-gate: cum_pnl={cum_pnl:+.4f}  n_active={n_active:,}", flush=True)

    # Save model
    model_path = os.path.join(HERE, f"model_t184_seed{args.seed}.pt")
    checkpoint = {
        "state_dict": model.state_dict(),
        "n_features": F_dim,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "nhead": args.nhead,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "window": args.window,
        "target_scale": float(target_scale),
        "scheme": args.scheme,
        "seed": args.seed,
        "args": vars(args),
    }
    torch.save(checkpoint, model_path)
    print(f"  saved {model_path}  ({os.path.getsize(model_path)/1e6:.2f} MB)", flush=True)

    # Save npz for numpy/torch-CPU inference
    npz_path = os.path.join(HERE, f"nn_raw_h60_seed{args.seed}.npz")
    sd = model.state_dict()
    npz_data = {
        "n_features": np.array([F_dim]),
        "d_model": np.array([args.d_model]),
        "n_layers": np.array([args.n_layers]),
        "nhead": np.array([args.nhead]),
        "ff_dim": np.array([args.ff_dim]),
        "dropout": np.array([args.dropout]),
        "window": np.array([args.window]),
        "target_scale": np.array([float(target_scale)]),
        "clip": np.array([CLIP]),
    }
    for k, v in sd.items():
        npz_data[k.replace(".", "/")] = v.detach().cpu().numpy()
    np.savez(npz_path, **npz_data)
    print(f"  saved npz {npz_path}  ({os.path.getsize(npz_path)/1e6:.2f} MB)", flush=True)

    # Save predictions for ensemble eval
    import pandas as pd
    sess_map_inv = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map_inv.get(int(s), "am") for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_te.astype(np.float32),
        "pred_dmid_norm": te_pred_full.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_t184_seed{args.seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    summary = {
        "task": f"T184 TickTransformer M7 seed={args.seed}",
        "scheme": args.scheme,
        "F_dim": int(F_dim),
        "n_params": int(n_params),
        "target_scale": float(target_scale),
        "test_mse": te_mse,
        "test_corr": te_corr,
        "test_ev_gate_k1_pnl": cum_pnl,
        "test_ev_gate_k1_n_active": n_active,
        "train_time_sec": float(train_time),
        "epoch_logs": epoch_logs,
        "args": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_t184_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb.log({
            "train_time_sec": float(train_time),
            "test_mse": te_mse,
            "test_corr": te_corr,
            "test_ev_gate_k1_pnl": cum_pnl,
            "n_params": int(n_params),
        })
        wandb.finish()

    progress("done", seed=args.seed, test_corr=te_corr, pnl=cum_pnl)
    print(f"\nRESULT: task=T184 seed={args.seed} metrics={{test_corr={te_corr:.4f},ev_pnl={cum_pnl:.4f}}} notes=[M7 TickTransformer 31-feat window-norm]")


if __name__ == "__main__":
    main()
