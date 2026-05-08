"""T95: Sequence model on raw LOB windows with regression on Δmid_norm + EV gate.

Architectures: minicnn | gru | deeplob | tcn

CRITICAL_CONSTRAINTS:
  * Model.forward() takes only window (B, W, F). No sym, no date, no time.
  * Per-feature global standardize on train stats. NaN→0 (already ffilled). Clip to ±10.
  * Target = (mp_t60 - mp_t) / (mp_t + 1), rescaled by 1/std(y).
  * Window = ticks [t-W+1, t] (strict, no future leakage). Target uses mp at t+60.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")

NUM_CLASS = 3
FEE = 0.0001


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


# -------- Models --------

class MiniCNN(nn.Module):
    """Simple 1D conv over time. Input (B, W, F) -> (B, F, W) -> conv -> pool -> FC."""
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, W, F)
        x = x.transpose(1, 2)  # (B, F, W)
        x = self.bn_in(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (B, hidden)
        x = self.dropout(x)
        return self.fc(x).squeeze(-1)


class MiniGRU(nn.Module):
    """1-2 layer GRU. Input (B, W, F)."""
    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.1, bidir: bool = False):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.gru = nn.GRU(in_dim, hidden, num_layers=n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0, bidirectional=bidir)
        out_dim = hidden * (2 if bidir else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        x = self.in_norm(x)
        out, h = self.gru(x)  # out: (B, W, hidden*dir)
        # take last timestep
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


class DeepLOB(nn.Module):
    """DeepLOB-style CNN (Zhang 2019 simplified) for 20-col LOB (top 5 levels, P+V interleaved).
    Input (B, W, 20) reshape to (B, 1, W, 20).
    """
    def __init__(self, in_dim: int = 20, hidden: int = 32, gru_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        # Convolutional block 1: combine price/volume per level → (B, hidden, W, 10)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
        )
        # Block 2: combine across price levels → (B, hidden, W, 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
        )
        # Block 3: combine all levels → (B, hidden, W, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=(1, 5)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
            nn.Conv2d(hidden, hidden, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(hidden),
        )
        # GRU on the time dim
        self.gru = nn.GRU(input_size=hidden, hidden_size=gru_hidden, num_layers=1,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden, 1)

    def forward(self, x):
        # x: (B, W, 20)
        x = x.unsqueeze(1)  # (B, 1, W, 20)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # (B, hidden, W', 1)
        x = x.squeeze(-1)  # (B, hidden, W')
        x = x.transpose(1, 2)  # (B, W', hidden)
        out, h = self.gru(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.pad = pad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=0, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=0, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        # causal pad: pad only on left
        h = F.pad(x, (self.pad, 0))
        h = F.gelu(self.bn1(self.conv1(h)))
        h = F.pad(h, (self.pad, 0))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        return h + self.proj(x)


class TCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 48, n_layers: int = 4, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.bn_in = nn.BatchNorm1d(in_dim)
        layers = []
        c_in = in_dim
        for i in range(n_layers):
            layers.append(TCNBlock(c_in, hidden, kernel, dilation=2**i, dropout=dropout))
            c_in = hidden
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (B, W, F)
        x = x.transpose(1, 2)  # (B, F, W)
        x = self.bn_in(x)
        x = self.tcn(x)
        # take last time step (causal)
        last = x[:, :, -1]
        return self.fc(last).squeeze(-1)


def build_model(arch: str, in_dim: int, **kwargs):
    if arch == "minicnn":
        return MiniCNN(in_dim, hidden=kwargs.get("hidden", 64), dropout=kwargs.get("dropout", 0.1))
    elif arch == "gru":
        return MiniGRU(in_dim, hidden=kwargs.get("hidden", 64), n_layers=kwargs.get("n_layers", 1),
                       dropout=kwargs.get("dropout", 0.1), bidir=kwargs.get("bidir", False))
    elif arch == "deeplob":
        assert in_dim == 20, "DeepLOB expects top5 (20 cols) input"
        return DeepLOB(in_dim, hidden=kwargs.get("hidden", 32), gru_hidden=kwargs.get("gru_hidden", 64),
                       dropout=kwargs.get("dropout", 0.1))
    elif arch == "tcn":
        return TCN(in_dim, hidden=kwargs.get("hidden", 48), n_layers=kwargs.get("n_layers", 4),
                   kernel=kwargs.get("kernel", 3), dropout=kwargs.get("dropout", 0.1))
    else:
        raise ValueError(arch)


# -------- Dataset --------

class WindowDataset(Dataset):
    """Slices windows from preloaded per-session LOB arrays.

    The standardization is done **per-batch on raw values** (consistent with training).
    Returns: (window_raw, y, sw)
    """
    def __init__(self, lob_arr: np.ndarray, sess_id: np.ndarray, t: np.ndarray,
                 y: np.ndarray, sw: np.ndarray, window: int):
        self.lob_arr = lob_arr  # (n_sess, T_max, F)
        self.sess_id = sess_id  # (n_samples,)
        self.t = t  # (n_samples,)
        self.y = y  # scaled regression target (float32)
        self.sw = sw  # sample weights
        self.W = window
        T_max = lob_arr.shape[1]
        # Filter to only samples where window fits: t-W+1 >= 0 i.e. t >= W-1
        # schemeP cache has t in [99, 1940], so for W<=100 always fine; for W>100 need to filter.
        # We'll filter at index time, but build a valid index mask:
        valid = t >= (window - 1)
        self.idx = np.where(valid)[0].astype(np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        s = self.sess_id[j]
        ti = int(self.t[j])
        win = self.lob_arr[s, ti - self.W + 1: ti + 1]  # (W, F), raw
        return win, self.y[j], self.sw[j]


def standardize_batch(x_raw: torch.Tensor, fm_t: torch.Tensor, fs_t: torch.Tensor,
                      clip: float = 10.0, mode: str = "global"):
    """
    mode = "global": (x - feat_mean) / feat_std using train stats
    mode = "window": per-window per-feature standardize (subtract window mean / std)
                     This removes absolute level info, focuses on intra-window dynamics.
    mode = "hybrid": concat global-standardized + window-standardized along feature dim
                     -> output (B, W, 2F)
    """
    if mode == "global":
        x = (x_raw - fm_t) / fs_t
        return torch.clamp(x, -clip, clip)
    elif mode == "window":
        # x_raw: (B, W, F); compute per (B, F) statistics over W axis
        m = x_raw.mean(dim=1, keepdim=True)  # (B, 1, F)
        s = x_raw.std(dim=1, keepdim=True)   # (B, 1, F)
        x = (x_raw - m) / s.clamp_min(1e-8)
        return torch.clamp(x, -clip, clip)
    elif mode == "hybrid":
        x_glob = torch.clamp((x_raw - fm_t) / fs_t, -clip, clip)
        m = x_raw.mean(dim=1, keepdim=True)
        s = x_raw.std(dim=1, keepdim=True)
        x_win = torch.clamp((x_raw - m) / s.clamp_min(1e-8), -clip, clip)
        return torch.cat([x_glob, x_win], dim=-1)
    else:
        raise ValueError(mode)


def predict_chunked(model, dataset, fm_t, fs_t, batch=4096, device="cuda", clip=10.0, num_workers=2,
                    norm_mode: str = "global"):
    """Run inference; return (n,) numpy array of preds, in scaled space (caller must descale)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=num_workers,
                         pin_memory=True)
    preds = np.zeros(len(dataset), dtype=np.float32)
    pos = 0
    with torch.no_grad():
        for win, y, sw in loader:
            win = win.to(device, non_blocking=True).float()
            xb = standardize_batch(win, fm_t, fs_t, clip, mode=norm_mode)
            yb = model(xb).detach().cpu().numpy()
            preds[pos: pos+len(yb)] = yb
            pos += len(yb)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="minicnn", choices=["minicnn", "gru", "deeplob", "tcn"])
    ap.add_argument("--scheme", default="top5",
                    help="Cache file suffix: top5 / top10 / v2")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gru-hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--bidir", action="store_true")
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--target-scale", type=float, default=0.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--tag", default="A", help="run tag for filenames")
    ap.add_argument("--use-sample-weight", action="store_true", default=True)
    ap.add_argument("--no-sample-weight", dest="use_sample_weight", action="store_false")
    ap.add_argument("--max-train-samples", type=int, default=0,
                    help="If >0, subsample for fast smoke test")
    ap.add_argument("--norm-mode", default="global", choices=["global", "window", "hybrid"],
                    help="Feature standardization mode")
    ap.add_argument("--aug-scale", type=float, default=0.0,
                    help="If >0, multiply each batch by per-(B,F) random uniform [1-x, 1+x]")
    ap.add_argument("--aug-noise", type=float, default=0.0,
                    help="Add per-element Gaussian noise std=this to STANDARDIZED inputs (after norm)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(f"=== T95 {args.arch} window={args.window} seed={args.seed} tag={args.tag} ===", flush=True)
    print(f"  device={device}", flush=True)

    progress("loading_caches", arch=args.arch, seed=args.seed, window=args.window)
    t0 = time.time()
    train_full = np.load(os.path.join(CACHE_DIR, f"lob_train_{args.scheme}.npz"))
    test_full = np.load(os.path.join(CACHE_DIR, f"lob_test_{args.scheme}.npz"))
    print(f"  loaded caches in {time.time()-t0:.1f}s", flush=True)

    lob_tr_full = train_full["lob_arr"]  # (n_sess, T_max, F)
    sid_tr = train_full["sess_id_for_sample"]
    t_tr = train_full["t_for_sample"]
    date_tr = train_full["date"]
    y_cls_tr_all = train_full["y60"].astype(np.int64)
    mp_t_tr_all = train_full["mp_t"]
    mp_th_tr_all = train_full["mp_t60"]

    # V4 split via date
    m_va = date_tr >= 76
    m_tr = ~m_va

    sid_train = sid_tr[m_tr]
    t_train = t_tr[m_tr]
    y_cls_train = y_cls_tr_all[m_tr]
    mp_t_train = mp_t_tr_all[m_tr]
    mp_th_train = mp_th_tr_all[m_tr]
    y_regr_train = regr_target(mp_t_train, mp_th_train)

    sid_val = sid_tr[m_va]
    t_val = t_tr[m_va]
    y_cls_val = y_cls_tr_all[m_va]
    mp_t_val = mp_t_tr_all[m_va]
    mp_th_val = mp_th_tr_all[m_va]
    y_regr_val = regr_target(mp_t_val, mp_th_val)

    # test
    lob_te_full = test_full["lob_arr"]
    sid_te = test_full["sess_id_for_sample"]
    t_te = test_full["t_for_sample"]
    y_cls_te = test_full["y60"].astype(np.int64)
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]
    sym_te = test_full["sym"]
    y_regr_te = regr_target(mp_t_te, mp_th_te)

    print(f"  V4 split: n_train={len(y_regr_train):,} n_val={len(y_regr_val):,} n_test={len(y_regr_te):,}", flush=True)

    # Compute global feature stats from train windows.
    # To be efficient, just use the entire (n_sess_train, T, F) arrays' values
    # restricted to train sessions (date < 76).
    # But date is per-sample, not per-session. Get sess_ids that belong to train.
    sess_keys_tr = train_full["sess_keys"]  # (n_sess, 3)
    # date per session
    sess_dates = sess_keys_tr[:, 1]
    train_sess_mask = sess_dates < 76
    train_sess_ids = np.where(train_sess_mask)[0]
    val_sess_mask = sess_dates >= 76
    print(f"  n_train_sessions={train_sess_mask.sum()} n_val_sessions={val_sess_mask.sum()}", flush=True)

    # Compute mean/std using only train sessions' LOB data
    # Reshape (n_train_sess, T, F) → (n_train_sess*T, F) then mean/std
    lob_train_subset = lob_tr_full[train_sess_ids]  # (n_train_sess, T, F)
    lob_flat = lob_train_subset.reshape(-1, lob_train_subset.shape[-1])
    feat_mean = lob_flat.mean(axis=0).astype(np.float32)
    feat_std = lob_flat.std(axis=0).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-6)
    print(f"  feat_mean range [{feat_mean.min():.3e}, {feat_mean.max():.3e}]"
          f"  feat_std range [{feat_std.min():.3e}, {feat_std.max():.3e}]", flush=True)
    F_dim = lob_flat.shape[-1]

    # Target scale
    if args.target_scale > 0:
        target_scale = float(args.target_scale)
    else:
        target_scale = 1.0 / float(max(y_regr_train.std(), 1e-8))
    print(f"  target_scale={target_scale:.4f}  (y_regr_train.std()={y_regr_train.std():.6e})", flush=True)
    y_train_s = (y_regr_train * target_scale).astype(np.float32)
    y_val_s = (y_regr_val * target_scale).astype(np.float32)

    # Sample weights (class-balanced like T75/T81)
    if args.use_sample_weight:
        sw_train = class_balanced_weight(y_cls_train, num_class=NUM_CLASS).astype(np.float32)
    else:
        sw_train = np.ones(len(y_train_s), dtype=np.float32)
    sw_val = np.ones(len(y_val_s), dtype=np.float32)
    sw_test = np.ones(len(y_regr_te), dtype=np.float32)

    # Optional subsample for smoke test
    if args.max_train_samples > 0 and len(sid_train) > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        sub = rng.choice(len(sid_train), args.max_train_samples, replace=False)
        sid_train = sid_train[sub]
        t_train = t_train[sub]
        y_train_s = y_train_s[sub]
        y_cls_train = y_cls_train[sub]
        sw_train = sw_train[sub]
        print(f"  SUBSAMPLED to {len(sid_train):,} train rows", flush=True)

    # Build datasets
    ds_train = WindowDataset(lob_tr_full, sid_train, t_train, y_train_s, sw_train, args.window)
    ds_val   = WindowDataset(lob_tr_full, sid_val,   t_val,   y_val_s,   sw_val,   args.window)
    ds_test  = WindowDataset(lob_te_full, sid_te,    t_te,    np.zeros(len(t_te), dtype=np.float32),
                              sw_test, args.window)
    print(f"  ds: train={len(ds_train):,} val={len(ds_val):,} test={len(ds_test):,}"
          f" (window={args.window})", flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T95-{args.arch}-w{args.window}-{args.tag}-seed{args.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T95_cnn_rnn_deeplob",
                    "arch": args.arch,
                    "window": args.window,
                    "scheme": args.scheme,
                    "F_dim": F_dim,
                    "horizon": 60,
                    "seed": args.seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T95", args.arch, "regression", "EV-gate", f"w{args.window}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # Model — input dim depends on norm_mode (hybrid doubles)
    in_dim_model = F_dim * (2 if args.norm_mode == "hybrid" else 1)
    model = build_model(args.arch, in_dim_model,
                         hidden=args.hidden, gru_hidden=args.gru_hidden,
                         n_layers=args.n_layers, bidir=args.bidir,
                         kernel=args.kernel, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: arch={args.arch} F={F_dim} W={args.window} n_params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(feat_std).to(device)
    CLIP = 10.0

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=False,
                               persistent_workers=(args.num_workers > 0))

    best_val_mse = float("inf")
    best_epoch = -1
    best_state = None
    bad = 0
    epoch_logs = []

    progress("training", arch=args.arch, n_train=len(ds_train), n_val=len(ds_val))
    t_train_start = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        rl, n_seen = 0.0, 0
        for batch in train_loader:
            win, y, sw = batch
            win = win.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True)
            # Aug: multiplicative scale on raw before norm
            if args.aug_scale > 0:
                # per-feature scaling (broadcast over W axis)
                scl = torch.empty(win.size(0), 1, win.size(2), device=win.device).uniform_(
                    1.0 - args.aug_scale, 1.0 + args.aug_scale)
                win = win * scl
            xb = standardize_batch(win, fm_t, fs_t, CLIP, mode=args.norm_mode)
            if args.aug_noise > 0:
                xb = xb + torch.randn_like(xb) * args.aug_noise
            pred = model(xb)
            mse_per = (pred - y) ** 2
            loss = (mse_per * sw).sum() / sw.sum().clamp_min(1.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = xb.size(0)
            rl += float(loss.item()) * bs
            n_seen += bs

        scheduler.step()
        train_loss = rl / max(n_seen, 1)

        # Validation
        va_pred_s = predict_chunked(model, ds_val, fm_t, fs_t, batch=2048, device=device,
                                     clip=CLIP, num_workers=args.num_workers, norm_mode=args.norm_mode)
        va_pred = va_pred_s / target_scale
        # ds_val may have filtered out early t (t < W-1); must apply same mask to y_regr_val
        valid_mask = t_val >= (args.window - 1)
        y_val_eff = y_regr_val[valid_mask]
        val_mse = float(((va_pred - y_val_eff) ** 2).mean())
        val_corr = float(np.corrcoef(va_pred, y_val_eff)[0, 1]) if va_pred.std() > 0 else 0.0

        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {"epoch": epoch, "train_loss": train_loss,
               "val_mse": val_mse, "val_corr": val_corr,
               "lr": cur_lr, "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d}  train_loss={train_loss:.6e}  "
              f"val_mse={val_mse:.6e}  val_corr={val_corr:.4f}  lr={cur_lr:.2e}  ({ep_time:.1f}s)",
              flush=True)
        if use_wandb:
            wandb.log({**log, "epoch": epoch})

        if val_mse < best_val_mse - 1e-10:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at ep {epoch}", flush=True)
                break

    train_time = time.time() - t_train_start
    print(f"\n  train done {train_time:.1f}s; best_epoch={best_epoch} best_val_mse={best_val_mse:.6e}", flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)

    # Save model
    model_path = os.path.join(HERE, f"model_T95_{args.arch}_w{args.window}_{args.tag}_seed{args.seed}.pt")
    torch.save({
        "state_dict": best_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "in_dim": F_dim,
        "window": args.window,
        "arch": args.arch,
        "hidden": args.hidden,
        "gru_hidden": args.gru_hidden,
        "n_layers": args.n_layers,
        "bidir": args.bidir,
        "kernel": args.kernel,
        "dropout": args.dropout,
        "scheme": args.scheme,
        "target_scale": float(target_scale),
        "clip": float(CLIP),
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "args": vars(args),
    }, model_path)
    print(f"  saved {model_path}", flush=True)

    # Final test predictions
    t0_pred = time.time()
    te_pred_s = predict_chunked(model, ds_test, fm_t, fs_t, batch=2048, device=device,
                                 clip=CLIP, num_workers=args.num_workers, norm_mode=args.norm_mode)
    te_pred = te_pred_s / target_scale
    pred_time = time.time() - t0_pred
    print(f"  test pred done in {pred_time:.1f}s for {len(te_pred):,} rows", flush=True)

    # Apply same valid mask filter (t >= W-1)
    test_valid_mask = t_te >= (args.window - 1)
    n_filtered_out = int((~test_valid_mask).sum())
    print(f"  test: kept {test_valid_mask.sum()} rows; dropped {n_filtered_out} (t < W-1)", flush=True)

    # Build full pred array (NaN for filtered rows; we'll handle later)
    te_pred_full = np.full(len(t_te), np.nan, dtype=np.float32)
    te_pred_full[test_valid_mask] = te_pred

    # For PnL eval, the dropped rows just default to flat (action=1).
    # Compute test metrics on valid rows
    y_te_valid = y_regr_te[test_valid_mask]
    te_mse = float(((te_pred - y_te_valid) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_te_valid)[0, 1]) if te_pred.std() > 0 else 0.0
    print(f"  TEST: mse={te_mse:.7e} corr={te_corr:.4f}", flush=True)

    # Quick EV-gate k=1 sanity
    fee_thr = 2.0 * FEE
    pred_action = np.full(len(te_pred_full), 1, dtype=np.int8)
    valid_pred = ~np.isnan(te_pred_full)
    pred_action[valid_pred & (te_pred_full > fee_thr)] = 2
    pred_action[valid_pred & (te_pred_full < -fee_thr)] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  TEST EV-gate k=1 (thr={fee_thr:.5f}): cum_pnl={cum_pnl:+.4f} n_active={n_active:,}",
          flush=True)

    # Save preds parquet (matching T75 schema)
    sess_map_inv = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map_inv[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred_full.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T95_{args.arch}_w{args.window}_{args.tag}_seed{args.seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    summary = {
        "task": f"T95 {args.arch} w{args.window} seed={args.seed} tag={args.tag}",
        "arch": args.arch,
        "window": args.window,
        "scheme": args.scheme,
        "seed": args.seed,
        "F_dim": int(F_dim),
        "n_params": int(n_params),
        "split_info": {"strategy": "V4", "train_dates": "0-75", "val_dates": "76-79", "test_dates": "96-119"},
        "dropped_for_window_test": n_filtered_out,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": te_mse,
        "test_corr": te_corr,
        "test_ev_gate_k1_cum_pnl": cum_pnl,
        "test_ev_gate_k1_n_active": n_active,
        "train_time_sec": float(train_time),
        "pred_time_sec_test": float(pred_time),
        "epoch_logs": epoch_logs,
        "args": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T95_{args.arch}_w{args.window}_{args.tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  summary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "train_time_sec": float(train_time),
            "test_mse": te_mse,
            "test_corr": te_corr,
            "test_ev_gate_k1_cum_pnl": cum_pnl,
            "test_ev_gate_k1_n_active": n_active,
            "n_params": int(n_params),
        })
        wandb.finish()

    progress("done", arch=args.arch, seed=args.seed, best_epoch=best_epoch,
             test_corr=te_corr, test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
