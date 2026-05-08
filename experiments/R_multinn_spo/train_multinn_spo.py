"""R_multinn_spo: 4 NN architectures × 3 seeds, L2 warm-start + SPO+ fine-tune.

Single unified script. Does:
  Phase 1: 5 epochs L2 regression warm-start (lr=3e-4)
  Phase 2: 15 epochs SPO+ DFL fine-tune (lr=3e-5, λ_spo=30)

Architectures:
  t87_baseline  - MLP [359→256→128→64→1] LayerNorm GELU dropout 0.10  (T87 same)
  wide_mlp      - MLP [359→512→256→128→1] LayerNorm GELU dropout 0.15
  deep_resnet   - TabResNet [359→128 →ResBlock×3(skip)→ 1]
  glu_mlp       - MLP with GeGLU activation [359→256→128→64→1]

CRITICAL_CONSTRAINTS: sym/date NOT in features; sym-agnostic; stateless.

Saves model_<arch>_seed<S>.pt and pred_<arch>_seed<S>.parquet.
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

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get("LWB_CACHE_DIR", "/root/lwb_remote_pkg/cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


# === class_balanced_weight (copied from build_aug.py) ===
def class_balanced_weight(y_cls, num_class=3):
    """Inverse-class-frequency weight. Total weight per class = N/num_class."""
    n = len(y_cls)
    out = np.zeros(n, dtype=np.float32)
    for k in range(num_class):
        m = (y_cls == k)
        cnt = int(m.sum())
        if cnt > 0:
            out[m] = float(n) / float(num_class) / float(cnt)
    return out


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def fee_eff(mp_t, mp_th):
    return (FEE * ((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


# === Architectures ===

class MLPRegr(nn.Module):
    """Plain MLP with LayerNorm + GELU + Dropout."""
    def __init__(self, in_dim, hidden, dropout=0.10, use_layernorm=True):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.10):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.ln2(h)
        return self.act(h + x)


class TabResNet(nn.Module):
    """Input -> proj 128 -> 3 ResBlocks(128) -> head 1."""
    def __init__(self, in_dim, dim=128, n_blocks=3, dropout=0.10):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.proj_ln = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([ResBlock(dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.act(self.proj_ln(self.proj(x)))
        h = self.drop(h)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h).squeeze(-1)


class GeGLULayer(nn.Module):
    """GeGLU: y = (W_a x) * GELU(W_b x).  Out dim = h."""
    def __init__(self, in_dim, out_dim, dropout=0.10):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2 * out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        h = a * F.gelu(b)
        return self.drop(self.ln(h))


class GeGLUMLP(nn.Module):
    """MLP with GeGLU activation, [in→256→128→64→1]."""
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.10):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(GeGLULayer(d, h, dropout))
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


def build_model(arch, in_dim):
    if arch == "t87_baseline":
        return MLPRegr(in_dim, hidden=(256, 128, 64), dropout=0.10)
    elif arch == "wide_mlp":
        return MLPRegr(in_dim, hidden=(512, 256, 128), dropout=0.15)
    elif arch == "deep_resnet":
        return TabResNet(in_dim, dim=128, n_blocks=3, dropout=0.10)
    elif arch == "glu_mlp":
        return GeGLUMLP(in_dim, hidden=(256, 128, 64), dropout=0.10)
    else:
        raise ValueError(arch)


# === SPO+ loss ===
def spo_plus_loss(pred_scaled, y_scaled, fee_scaled):
    z_star = torch.where(
        y_scaled > fee_scaled, torch.ones_like(y_scaled),
        torch.where(y_scaled < -fee_scaled, -torch.ones_like(y_scaled),
                    torch.zeros_like(y_scaled)),
    )
    abs_z_star = torch.abs(z_star)
    spread = 2.0 * pred_scaled - y_scaled
    relu_max = F.relu(torch.abs(spread) - fee_scaled)
    return relu_max - z_star * spread + fee_scaled * abs_z_star


def predict_chunked(model, X, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            out[s:s+batch] = yb
    return out


def evaluate_pnl(pred, mp_t, mp_th, thr=2*FEE):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr] = 2
    a[pred < -thr] = 0
    side = a.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((a != 1).sum())


# === Main ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["t87_baseline", "wide_mlp", "deep_resnet", "glu_mlp"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--warmstart-lr", type=float, default=3e-4)
    ap.add_argument("--warmstart-epochs", type=int, default=5)
    ap.add_argument("--spo-lr", type=float, default=3e-5)
    ap.add_argument("--spo-epochs", type=int, default=15)
    ap.add_argument("--lambda-spo", type=float, default=30.0)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.horizon
    seed = args.seed
    arch = args.arch
    print(f"=== R_multinn_spo arch={arch} seed={seed} ===", flush=True)
    print(f"  device={device}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    progress("loading_caches", arch=arch, seed=seed)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim, (len(all_feat_names), total_dim)

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), \
        f"FORBIDDEN feat leak: {forbidden & set(feat_names)}"
    print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)} fail features)", flush=True)

    # V4 walk-forward split
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr = train_full["X"][m_t][:, keep_idx]
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    mp_t_tr = train_full["mp_t"][m_t]
    mp_th_tr = train_full["mp_t60"][m_t]
    fee_eff_tr = fee_eff(mp_t_tr, mp_th_tr)

    X_va = train_full["X"][m_va][:, keep_idx]
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]

    X_te = test_full["X"][:, keep_idx]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    print(f"  V4 split: n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,}",
          flush=True)

    # Standardization
    feat_mean = np.nanmean(X_tr, axis=0).astype(np.float32)
    feat_std = np.nanstd(X_tr, axis=0).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-6)
    CLIP = 10.0

    def standardize(X):
        Xs = (X - feat_mean) / feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -CLIP, CLIP)
        return Xs.astype(np.float32)

    X_va_std = torch.from_numpy(standardize(X_va))
    X_te_std = torch.from_numpy(standardize(X_te))

    # Pre-impute train NaN with mean
    nan_mask_tr = np.isnan(X_tr)
    if nan_mask_tr.any():
        for d_idx in np.where(nan_mask_tr.any(axis=0))[0]:
            col_nan = np.isnan(X_tr[:, d_idx])
            X_tr[col_nan, d_idx] = feat_mean[d_idx]
        print(f"  imputed {int(nan_mask_tr.sum()):,} NaN cells in train", flush=True)

    # target_scale
    target_scale = 1.0 / float(max(y_regr_tr.std(), 1e-8))
    print(f"  target_scale={target_scale:.4f}", flush=True)

    # Aug-concat (T81/T87 default)
    seed_rng = np.random.default_rng(seed * 7919 + 137)
    n_orig = len(X_tr)
    X_tr_full = np.empty((2 * n_orig, X_tr.shape[1]), dtype=np.float32)
    X_tr_full[:n_orig] = X_tr
    for i in range(0, n_orig, 100_000):
        j = min(i + 100_000, n_orig)
        scales = seed_rng.uniform(args.aug_lo, args.aug_hi,
                                  size=(j - i, X_tr.shape[1])).astype(np.float32)
        X_tr_full[n_orig + i:n_orig + j] = X_tr[i:j] * scales
    del X_tr  # free

    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
    fee_eff_tr_full = np.concatenate([fee_eff_tr, fee_eff_tr], axis=0)

    y_scaled_tr = (y_regr_tr_full * target_scale).astype(np.float32)
    fee_scaled_tr = (fee_eff_tr_full * target_scale).astype(np.float32)
    sw_tr_full = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS).astype(np.float32)
    print(f"  aug=concat X_tr_full={X_tr_full.shape}", flush=True)

    # Tensors
    X_tr_raw = torch.from_numpy(X_tr_full)
    y_scaled_tr_t = torch.from_numpy(y_scaled_tr)
    fee_scaled_tr_t = torch.from_numpy(fee_scaled_tr)
    sw_tr_t = torch.from_numpy(sw_tr_full)

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(feat_std).to(device)

    # Model
    model = build_model(arch, feat_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  arch={arch} params={n_params:,}", flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"R_multinn-{arch}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "R_multinn_spo",
                    "arch": arch,
                    "seed": seed,
                    "warmstart_epochs": args.warmstart_epochs,
                    "warmstart_lr": args.warmstart_lr,
                    "spo_epochs": args.spo_epochs,
                    "spo_lr": args.spo_lr,
                    "lambda_spo": args.lambda_spo,
                    "horizon": H,
                    "n_params": n_params,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["R_multinn_spo", arch, f"seed{seed}", "SPO+"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    n_train = len(X_tr_raw)
    epoch_logs = []
    best_state = None
    best_val_pnl = -float("inf")
    best_te_pnl = -float("inf")
    best_epoch = -1

    # === Phase 1: L2 warm-start ===
    print(f"\n--- Phase 1: L2 warm-start ({args.warmstart_epochs} epochs) lr={args.warmstart_lr} ---",
          flush=True)
    progress("warmstart_l2", arch=arch, seed=seed)

    optim = torch.optim.AdamW(model.parameters(), lr=args.warmstart_lr,
                              weight_decay=args.weight_decay)
    t_phase1 = time.time()
    for ep in range(args.warmstart_epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train)
        run_l2 = 0.0
        run_n = 0
        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)
            yb = y_scaled_tr_t[idx].to(device, non_blocking=True)
            wb = sw_tr_t[idx].to(device, non_blocking=True)
            xb = (xb_raw - fm_t) / fs_t
            xb = torch.clamp(xb, -CLIP, CLIP)
            pred = model(xb)
            l2_per = (pred - yb) ** 2
            loss = (l2_per * wb).sum() / wb.sum().clamp_min(1.0)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            bs = xb.size(0)
            run_l2 += float(loss.item()) * bs
            run_n += bs
        train_l2 = run_l2 / run_n

        # Val eval
        va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
        val_pnl, val_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)
        te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
        te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
        te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)

        ep_time = time.time() - t_ep
        log = {"phase": "warmstart", "epoch": ep, "train_l2": train_l2,
               "val_corr": val_corr, "val_pnl": val_pnl, "te_corr": te_corr,
               "te_pnl": te_pnl, "time": ep_time}
        epoch_logs.append(log)
        print(f"  [WS] ep {ep:3d}  l2={train_l2:.4e}  val_corr={val_corr:.4f}  "
              f"val_pnl={val_pnl:+.3f}  te_pnl={te_pnl:+.3f}  ({ep_time:.1f}s)",
              flush=True)
        if use_wandb:
            wandb.log(log)

    print(f"  phase1 done in {time.time()-t_phase1:.1f}s", flush=True)

    # === Phase 2: SPO+ fine-tune ===
    print(f"\n--- Phase 2: SPO+ fine-tune ({args.spo_epochs} epochs) lr={args.spo_lr} "
          f"λ={args.lambda_spo} ---", flush=True)
    progress("spo_finetune", arch=arch, seed=seed)

    optim = torch.optim.AdamW(model.parameters(), lr=args.spo_lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.spo_epochs)

    # Initial eval (after warmstart)
    va_pred_init = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    val_pnl_init, _ = evaluate_pnl(va_pred_init, mp_t_va, mp_th_va)
    te_pred_init = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    te_pnl_init, _ = evaluate_pnl(te_pred_init, mp_t_te, mp_th_te)
    best_val_pnl = val_pnl_init
    best_te_pnl = te_pnl_init
    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    best_epoch = -1
    print(f"  WS-end init: val_pnl={val_pnl_init:+.3f}  te_pnl={te_pnl_init:+.3f}",
          flush=True)

    bad_epochs = 0
    t_phase2 = time.time()
    for ep in range(args.spo_epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train)
        run_l2 = 0.0
        run_spo = 0.0
        run_n = 0
        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)
            yb = y_scaled_tr_t[idx].to(device, non_blocking=True)
            fb = fee_scaled_tr_t[idx].to(device, non_blocking=True)
            wb = sw_tr_t[idx].to(device, non_blocking=True)
            xb = (xb_raw - fm_t) / fs_t
            xb = torch.clamp(xb, -CLIP, CLIP)
            pred = model(xb)
            l2_per = (pred - yb) ** 2
            spo_per = spo_plus_loss(pred, yb, fb)
            l2_loss = (l2_per * wb).sum() / wb.sum().clamp_min(1.0)
            spo_loss = (spo_per * wb).sum() / wb.sum().clamp_min(1.0)
            loss = l2_loss + args.lambda_spo * spo_loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            bs = xb.size(0)
            run_l2 += float(l2_loss.item()) * bs
            run_spo += float(spo_loss.item()) * bs
            run_n += bs
        sched.step()
        train_l2 = run_l2 / run_n
        train_spo = run_spo / run_n

        va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
        val_pnl, val_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)
        te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
        te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
        te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)

        ep_time = time.time() - t_ep
        log = {"phase": "spo", "epoch": ep, "train_l2": train_l2, "train_spo": train_spo,
               "val_corr": val_corr, "val_pnl": val_pnl, "te_corr": te_corr,
               "te_pnl": te_pnl, "lr": optim.param_groups[0]["lr"], "time": ep_time}
        epoch_logs.append(log)
        print(f"  [SPO] ep {ep:3d}  l2={train_l2:.3e} spo={train_spo:.3e}  "
              f"val_corr={val_corr:.4f}  val_pnl={val_pnl:+.3f}  te_pnl={te_pnl:+.3f}  "
              f"({ep_time:.1f}s)", flush=True)
        if use_wandb:
            wandb.log(log)

        if val_pnl > best_val_pnl:
            best_val_pnl = val_pnl
            best_te_pnl = te_pnl
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            best_epoch = ep
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop at ep {ep}", flush=True)
                break

    train_time = time.time() - t_phase2
    print(f"\n  phase2 done in {train_time:.1f}s; best_epoch={best_epoch} "
          f"best_val_pnl={best_val_pnl:+.3f} best_te_pnl={best_te_pnl:+.3f}",
          flush=True)
    model.load_state_dict(best_state)

    # Final preds + save
    va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    va_pnl, _ = evaluate_pnl(va_pred, mp_t_va, mp_th_va)
    te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)
    print(f"\n  FINAL test: corr={te_corr:.4f} EV-gate(k=1) pnl={te_pnl:+.4f} n_act={te_n_act:,}",
          flush=True)

    # Save model
    model_path = os.path.join(HERE, f"model_{arch}_seed{seed}.pt")
    torch.save({
        "state_dict": best_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx,
        "feat_names": feat_names,
        "arch": arch,
        "in_dim": feat_dim,
        "target_scale": float(target_scale),
        "clip": float(CLIP),
        "best_epoch": best_epoch,
        "lambda_spo": args.lambda_spo,
    }, model_path)
    print(f"  saved {model_path}", flush=True)

    # Save pred parquet
    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_{arch}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    summary = {
        "task": f"R_multinn_spo arch={arch} seed={seed}",
        "arch": arch,
        "seed": seed,
        "horizon": H,
        "n_features": feat_dim,
        "n_params": int(n_params),
        "warmstart_epochs": args.warmstart_epochs,
        "spo_epochs_run": len([l for l in epoch_logs if l["phase"] == "spo"]),
        "best_spo_epoch": best_epoch,
        "lambda_spo": args.lambda_spo,
        "spo_lr": args.spo_lr,
        "ws_init": {"val_pnl": val_pnl_init, "te_pnl": te_pnl_init},
        "test": {"corr": te_corr, "ev_gate_k1_cum_pnl": te_pnl,
                 "n_active": te_n_act},
        "val": {"corr": va_corr, "ev_gate_k1_cum_pnl": va_pnl},
        "epoch_logs": epoch_logs,
    }
    out_path = os.path.join(HERE, f"summary_{arch}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_spo_epoch": best_epoch,
            "best_val_pnl": best_val_pnl,
            "final_te_corr": te_corr,
            "final_te_pnl": te_pnl,
            "delta_vs_ws_init": te_pnl - te_pnl_init,
        })
        wandb.finish()

    progress("done", arch=arch, seed=seed, best_epoch=best_epoch,
             te_pnl=te_pnl, delta_ws=te_pnl - te_pnl_init)


if __name__ == "__main__":
    main()
