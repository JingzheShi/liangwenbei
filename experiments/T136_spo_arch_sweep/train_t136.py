"""T136: SPO+ NN architecture sweep.

Variants:
  A: MLP 512-256-128 (wider) — from scratch (input dim mismatch w/ T81)
  B: MLP 256-256-256-128 with residual — from scratch
  C: MLP 256-128-64 dropout=0.20 — warm-start from T81
  D: MLP 256-128-64 vanilla (no LayerNorm) — from scratch

For from-scratch variants: combined L2 + lambda_spo*SPO+ for full schedule.
For warm-start C: short SPO+ fine-tune like T87.

3 seeds {1, 7, 42}, share data load across seeds in one process.
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

# Remote cache layout (vast.ai r2): /root/lwb_remote_pkg/cache/
# Local fallback: /root/projects/liangwenbei_workdir/experiments/T68_stage5_features/cache/
DEFAULT_CACHE = "/root/lwb_remote_pkg/cache"
DEFAULT_T81 = "/root/lwb_remote_pkg/T81_weights"
NUM_CLASS = 3
FEE = 0.0001


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


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


def fee_eff(mp_t, mp_th):
    return (FEE * ((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


# ---------------- Architectures ----------------

class MLPVariant(nn.Module):
    """Generic MLP. hidden=tuple of widths."""
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.10, use_layernorm=True):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.10, use_layernorm=True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.ln1(self.fc1(x)))
        h = self.drop(h)
        h = self.ln2(self.fc2(h))
        return F.gelu(x + h)


class MLPResidual(nn.Module):
    """Variant B: 256→[ResBlock 256]→[ResBlock 256]→128→1."""
    def __init__(self, in_dim, hidden=(256, 256, 256, 128), dropout=0.10, use_layernorm=True):
        super().__init__()
        # First layer: in_dim -> 256
        self.in_proj = nn.Linear(in_dim, hidden[0])
        self.in_ln = nn.LayerNorm(hidden[0]) if use_layernorm else nn.Identity()
        # Residual blocks for matching dims
        blocks = []
        for i in range(1, len(hidden) - 1):
            if hidden[i] == hidden[i-1]:
                blocks.append(ResBlock(hidden[i], dropout=dropout, use_layernorm=use_layernorm))
            else:
                blocks.append(nn.Sequential(
                    nn.Linear(hidden[i-1], hidden[i]),
                    nn.LayerNorm(hidden[i]) if use_layernorm else nn.Identity(),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
        # Final downprojection to last dim
        last_h = hidden[-1]
        prev_h = hidden[-2]
        self.blocks = nn.Sequential(*blocks)
        self.down = nn.Sequential(
            nn.Linear(prev_h, last_h),
            nn.LayerNorm(last_h) if use_layernorm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(last_h, 1)
        self.in_drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.in_ln(self.in_proj(x)))
        h = self.in_drop(h)
        h = self.blocks(h)
        h = self.down(h)
        return self.head(h).squeeze(-1)


def build_model(variant, in_dim):
    if variant == "A":
        return MLPVariant(in_dim, hidden=(512, 256, 128), dropout=0.10, use_layernorm=True)
    if variant == "B":
        return MLPResidual(in_dim, hidden=(256, 256, 256, 128), dropout=0.10, use_layernorm=True)
    if variant == "C":
        return MLPVariant(in_dim, hidden=(256, 128, 64), dropout=0.20, use_layernorm=True)
    if variant == "D":
        return MLPVariant(in_dim, hidden=(256, 128, 64), dropout=0.10, use_layernorm=False)
    raise ValueError(f"unknown variant {variant}")


def variant_warm_start(variant):
    """Whether to load T81 weights as init."""
    return variant == "C"


# ---------------- SPO+ loss ----------------

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


# ---------------- Eval ----------------

def evaluate_pnl(pred_unscaled, mp_t, mp_th, thr_up=2*FEE, thr_dn=2*FEE):
    pred_action = np.full(len(pred_unscaled), 1, dtype=np.int8)
    pred_action[pred_unscaled > thr_up] = 2
    pred_action[pred_unscaled < -thr_dn] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((pred_action != 1).sum())


def predict_chunked(model, X, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            out[s:s+batch] = yb
    return out


# ---------------- Driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--t81-dir", default=DEFAULT_T81)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr-warm", type=float, default=3e-5,
                    help="LR for warm-start fine-tune (variant C)")
    ap.add_argument("--lr-scratch", type=float, default=3e-4,
                    help="LR for from-scratch (variants A,B,D)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lambda-spo", type=float, default=1.0)
    ap.add_argument("--target-scale", type=float, default=2000.0,
                    help="target scaling factor (T81 used 2000 in seed42 ckpt)")
    ap.add_argument("--feat-clip", type=float, default=10.0)
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--no-aug", action="store_true",
                    help="disable concat scale aug")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== T136 variant={args.variant} seeds={seeds} ===", flush=True)

    progress("loading_caches", variant=args.variant, seeds=seeds)

    t0 = time.time()
    train = np.load(os.path.join(args.cache_dir, "schemeP_train.npz"))
    test = np.load(os.path.join(args.cache_dir, "schemeP_test.npz"))
    train_data = {k: train[k] for k in train.files}
    test_data = {k: test[k] for k in test.files}
    print(f"loaded caches in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(args.cache_dir, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]

    # Load T81 ckpt for one seed to get feat_mean/std/keep_idx (these are
    # consistent across seeds — they're computed from the same cache).
    sample_seed = 42 if 42 in seeds else seeds[0]
    sample_ckpt_path = os.path.join(args.t81_dir, f"model_T81_seed{sample_seed}.pt")
    print(f"  loading meta from {sample_ckpt_path}", flush=True)
    sample_ckpt = torch.load(sample_ckpt_path, map_location="cpu", weights_only=False)
    feat_mean = np.asarray(sample_ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(sample_ckpt["feat_std"], dtype=np.float32)
    keep_idx = np.asarray(sample_ckpt["keep_idx"], dtype=np.int64)
    target_scale = float(sample_ckpt["target_scale"])
    in_dim = int(sample_ckpt["in_dim"])
    clip_val = float(sample_ckpt.get("clip", 10.0))
    print(f"  feat_mean.shape={feat_mean.shape} keep_idx.shape={keep_idx.shape} "
          f"in_dim={in_dim} target_scale={target_scale}", flush=True)

    feat_names_kept = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names_kept)), \
        f"FORBIDDEN feat leak: {forbidden & set(feat_names_kept)}"

    # V4 walk-forward split
    date_tr = train_data["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr = train_data["X"][m_t][:, keep_idx]
    y_cls_tr = train_data["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_data["mp_t"][m_t], train_data["mp_t60"][m_t])
    fee_eff_tr = fee_eff(train_data["mp_t"][m_t], train_data["mp_t60"][m_t])

    X_va = train_data["X"][m_va][:, keep_idx]
    y_regr_va = regr_target(train_data["mp_t"][m_va], train_data["mp_t60"][m_va])
    mp_t_va = train_data["mp_t"][m_va]
    mp_th_va = train_data["mp_t60"][m_va]

    X_te = test_data["X"][:, keep_idx]
    y_regr_te = regr_target(test_data["mp_t"], test_data["mp_t60"])
    mp_t_te = test_data["mp_t"]
    mp_th_te = test_data["mp_t60"]
    sym_te = test_data["sym"]

    # NaN imputation in train
    nan_mask_tr = np.isnan(X_tr)
    if nan_mask_tr.any():
        for d_idx in np.where(nan_mask_tr.any(axis=0))[0]:
            col_nan = np.isnan(X_tr[:, d_idx])
            X_tr[col_nan, d_idx] = feat_mean[d_idx]
        print(f"  imputed {int(nan_mask_tr.sum()):,} NaN in train", flush=True)

    def standardize(X):
        Xs = (X - feat_mean) / np.maximum(feat_std, 1e-6)
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        return np.clip(Xs, -clip_val, clip_val).astype(np.float32)

    X_va_std = torch.from_numpy(standardize(X_va))
    X_te_std = torch.from_numpy(standardize(X_te))

    # Free train cache headers
    del train_data, test_data, train, test

    print(f"  V4 split: n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,}",
          flush=True)
    print(f"  X_te.shape={X_te.shape} (full LOSO test set)", flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
        except ImportError:
            use_wandb = False

    is_warm = variant_warm_start(args.variant)
    use_aug = not args.no_aug

    # Pre-build aug copy once (shared across seeds via different RNG per seed
    # — no, must be seed-specific. We'll regenerate per seed but cheaply).
    # Pre-compute fee_eff_tr (shared)
    print(f"  variant={args.variant} warm_start={is_warm} use_aug={use_aug}", flush=True)

    summaries = []

    for seed in seeds:
        print(f"\n----- seed={seed} -----", flush=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Per-seed aug
        if use_aug:
            seed_rng = np.random.default_rng(seed * 7919 + 137)
            n_orig = len(X_tr)
            X_tr_full = np.empty((2 * n_orig, X_tr.shape[1]), dtype=np.float32)
            X_tr_full[:n_orig] = X_tr
            for i in range(0, n_orig, 100_000):
                j = min(i + 100_000, n_orig)
                scales = seed_rng.uniform(0.80, 1.20,
                                          size=(j - i, X_tr.shape[1])).astype(np.float32)
                X_tr_full[n_orig + i:n_orig + j] = X_tr[i:j] * scales
            y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
            y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
            fee_eff_tr_full = np.concatenate([fee_eff_tr, fee_eff_tr], axis=0)
        else:
            X_tr_full = X_tr
            y_regr_tr_full = y_regr_tr
            y_cls_tr_full = y_cls_tr
            fee_eff_tr_full = fee_eff_tr

        y_scaled_tr = (y_regr_tr_full * target_scale).astype(np.float32)
        fee_scaled_tr = (fee_eff_tr_full * target_scale).astype(np.float32)
        sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS).astype(np.float32)

        X_tr_t = torch.from_numpy(X_tr_full.astype(np.float32))
        y_scaled_t = torch.from_numpy(y_scaled_tr)
        fee_scaled_t = torch.from_numpy(fee_scaled_tr)
        sw_t = torch.from_numpy(sw_tr)
        fm_t = torch.from_numpy(feat_mean).to(device)
        fs_t = torch.from_numpy(np.maximum(feat_std, 1e-6)).to(device)

        # Build model
        model = build_model(args.variant, in_dim).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  model[{args.variant}]: params={n_params:,}", flush=True)

        if is_warm:
            t81_pt = os.path.join(args.t81_dir, f"model_T81_seed{seed}.pt")
            ck = torch.load(t81_pt, map_location="cpu", weights_only=False)
            # T81 saved with hidden=(256,128,64), dropout=0.10, layernorm=True.
            # Variant C has same hidden/LN but dropout=0.20 — weight shapes still match.
            try:
                model.load_state_dict(ck["state_dict"])
                print(f"  warm-started from {t81_pt}", flush=True)
            except Exception as e:
                print(f"  WARN warm-start failed: {e}; using random init", flush=True)

        lr = args.lr_warm if is_warm else args.lr_scratch
        epochs = args.epochs if not is_warm else min(args.epochs, 8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        if use_wandb:
            try:
                run_name = f"T136-{args.variant}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                wandb.init(
                    project="liangwenbei",
                    entity="cjxh21-Tsinghua University",
                    name=run_name,
                    config={
                        "task": "T136_spo_arch_sweep",
                        "variant": args.variant,
                        "seed": seed,
                        "warm_start": is_warm,
                        "lr": lr,
                        "epochs": epochs,
                        "lambda_spo": args.lambda_spo,
                        "n_params": n_params,
                    },
                    tags=["T136", "spo-arch-sweep", f"variant{args.variant}", f"seed{seed}"],
                    reinit=True,
                )
            except Exception as e:
                print(f"  wandb init failed: {e}", flush=True)
                use_wandb = False

        # Initial eval (T81 if warm, random if scratch)
        va_pred_init = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
        te_pred_init = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
        val_pnl_init, _ = evaluate_pnl(va_pred_init, mp_t_va, mp_th_va)
        te_pnl_init, _ = evaluate_pnl(te_pred_init, mp_t_te, mp_th_te)
        print(f"  init eval: val_pnl={val_pnl_init:+.3f} te_pnl={te_pnl_init:+.3f}", flush=True)

        best_val_pnl = -float("inf")
        best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        best_epoch = -1
        bad = 0
        epoch_logs = []
        n_train = len(X_tr_t)

        progress(f"training_{args.variant}_s{seed}", variant=args.variant, seed=seed)
        t_train = time.time()
        for ep in range(epochs):
            t_ep = time.time()
            model.train()
            perm = torch.randperm(n_train)
            run_l2 = run_spo = run_n = 0.0

            for s_idx in range(0, n_train, args.batch_size):
                idx = perm[s_idx:s_idx + args.batch_size]
                xb_raw = X_tr_t[idx].to(device, non_blocking=True)
                yb = y_scaled_t[idx].to(device, non_blocking=True)
                fb = fee_scaled_t[idx].to(device, non_blocking=True)
                wb = sw_t[idx].to(device, non_blocking=True)

                xb = (xb_raw - fm_t) / fs_t
                xb = torch.clamp(xb, -clip_val, clip_val)

                pred = model(xb)
                l2_per = (pred - yb) ** 2
                spo_per = spo_plus_loss(pred, yb, fb)
                l2_loss = (l2_per * wb).sum() / wb.sum().clamp_min(1.0)
                spo_loss = (spo_per * wb).sum() / wb.sum().clamp_min(1.0)
                loss = l2_loss + args.lambda_spo * spo_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                bs = xb.size(0)
                run_l2 += float(l2_loss.item()) * bs
                run_spo += float(spo_loss.item()) * bs
                run_n += bs

            scheduler.step()
            train_l2 = run_l2 / run_n
            train_spo = run_spo / run_n

            va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
            val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
            val_pnl, val_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)

            te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
            te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
            te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)

            ep_t = time.time() - t_ep
            log = {"epoch": ep, "train_l2": train_l2, "train_spo": train_spo,
                   "val_corr": val_corr, "val_pnl": val_pnl,
                   "te_corr": te_corr, "te_pnl": te_pnl,
                   "lr": optimizer.param_groups[0]["lr"], "time": ep_t}
            epoch_logs.append(log)
            print(f"  ep {ep:2d}  l2={train_l2:.3e} spo={train_spo:.3e} "
                  f"val_corr={val_corr:.4f} val_pnl={val_pnl:+.3f} "
                  f"te_corr={te_corr:.4f} te_pnl={te_pnl:+.3f} ({ep_t:.1f}s)",
                  flush=True)
            if use_wandb:
                wandb.log(log)

            if val_pnl > best_val_pnl:
                best_val_pnl = val_pnl
                best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                best_epoch = ep
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"  early stop at ep={ep}", flush=True)
                    break

        train_time = time.time() - t_train
        model.load_state_dict(best_state)

        # Final eval
        va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
        te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
        va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
        te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
        va_pnl, va_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)
        te_pnl_default, te_n_default = evaluate_pnl(te_pred, mp_t_te, mp_th_te,
                                                    thr_up=2*FEE, thr_dn=2*FEE)
        # iter_015 v1 thresh
        te_pnl_iter15, te_n_iter15 = evaluate_pnl(te_pred, mp_t_te, mp_th_te,
                                                  thr_up=4.21e-4, thr_dn=1.86e-4)

        print(f"  FINAL [{args.variant}/seed{seed}]: "
              f"va_corr={va_corr:.4f} va_pnl={va_pnl:+.3f} "
              f"te_corr={te_corr:.4f} te_pnl(2e-4)={te_pnl_default:+.3f} "
              f"te_pnl(iter15)={te_pnl_iter15:+.3f} train_time={train_time:.1f}s",
              flush=True)

        # Save pred + small ckpt
        sess_map = {0: "am", 1: "pm"}
        sess_idx = np.load(os.path.join(args.cache_dir, "schemeP_test.npz"))["sess_idx"]
        date_te = np.load(os.path.join(args.cache_dir, "schemeP_test.npz"))["date"]
        t_te = np.load(os.path.join(args.cache_dir, "schemeP_test.npz"))["t"]
        y_cls_te = np.load(os.path.join(args.cache_dir, "schemeP_test.npz"))["y60"].astype(np.int64)
        te_df = pd.DataFrame({
            "sym": sym_te.astype(np.int8),
            "date": date_te.astype(np.int16),
            "session": np.array([sess_map[int(x)] for x in sess_idx], dtype=object),
            "t": t_te.astype(np.int16),
            "true_label": y_cls_te.astype(np.int8),
            "true_dmid_norm": y_regr_te.astype(np.float32),
            "pred_dmid_norm": te_pred.astype(np.float32),
            "midprice_t": mp_t_te.astype(np.float32),
            "midprice_th": mp_th_te.astype(np.float32),
        })
        pred_path = os.path.join(HERE, f"pred_T136_{args.variant}_seed{seed}.parquet")
        te_df.to_parquet(pred_path, index=False)
        print(f"  saved {pred_path}", flush=True)

        # Save ckpt (only weights — small)
        ckpt_out = os.path.join(HERE, f"model_T136_{args.variant}_seed{seed}.pt")
        torch.save({
            "state_dict": best_state,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "keep_idx": keep_idx,
            "feat_names": feat_names_kept,
            "in_dim": in_dim,
            "target_scale": target_scale,
            "clip": clip_val,
            "variant": args.variant,
            "best_epoch": best_epoch,
            "warm_start": is_warm,
        }, ckpt_out)

        summary = {
            "variant": args.variant,
            "seed": seed,
            "warm_start": is_warm,
            "n_params": int(n_params),
            "epochs_run": len(epoch_logs),
            "best_epoch": best_epoch,
            "init": {"val_pnl": val_pnl_init, "te_pnl": te_pnl_init},
            "val": {"corr": va_corr, "ev_gate_k1_cum_pnl": va_pnl, "n_active": va_n_act},
            "test_default": {"thr_up": 2*FEE, "thr_dn": 2*FEE,
                             "te_corr": te_corr, "cum_pnl": te_pnl_default,
                             "n_active": te_n_default},
            "test_iter15": {"thr_up": 4.21e-4, "thr_dn": 1.86e-4,
                            "te_corr": te_corr, "cum_pnl": te_pnl_iter15,
                            "n_active": te_n_iter15},
            "train_time_sec": float(train_time),
        }
        sp = os.path.join(HERE, f"summary_T136_{args.variant}_seed{seed}.json")
        with open(sp, "w") as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)
        print(f"  summary -> {sp}", flush=True)

        if use_wandb:
            try:
                wandb.log({
                    "final_te_pnl_default": te_pnl_default,
                    "final_te_pnl_iter15": te_pnl_iter15,
                    "final_va_pnl": va_pnl,
                    "best_epoch": best_epoch,
                })
                wandb.finish()
            except Exception:
                pass

        # cleanup
        del X_tr_t, y_scaled_t, fee_scaled_t, sw_t, model, optimizer, scheduler
        if use_aug:
            del X_tr_full, y_regr_tr_full, y_cls_tr_full, fee_eff_tr_full
        torch.cuda.empty_cache()

    # Aggregate
    out = {
        "variant": args.variant,
        "seeds": seeds,
        "per_seed": summaries,
    }
    op = os.path.join(HERE, f"summary_T136_{args.variant}_all.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nALL DONE for variant {args.variant}; agg -> {op}", flush=True)


if __name__ == "__main__":
    main()
