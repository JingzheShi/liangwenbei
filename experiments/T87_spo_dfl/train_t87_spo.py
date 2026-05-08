"""T87: Decision-Focused Learning via SPO+ — warm-start fine-tune from T81.

Key idea:
  PnL per sample = z·c - fee_eff·|z|, where
    c = (mp_th - mp_t) / (mp_t + 1)    # Δmid_norm, our T75/T81 target
    fee_eff = FEE·((mp_th+1)+(mp_t+1))/(mp_t+1)  ≈ 2·FEE
    z ∈ {-1, 0, +1} is the trader's action (short/flat/long)
  Optimal action: z*(c) = sign(c) if |c| > fee_eff else 0.

  SPO+ loss (Elmachtoub & Grigas 2017, max-problem variant):
    ℓ_SPO+(ĉ, c) = max_z {(2ĉ - c)·z - fee_eff·|z|}
                   - z*(c)·(2ĉ - c) + fee_eff·|z*(c)|
                 = ReLU(|2ĉ - c| - fee_eff)
                   - z*(c)·(2ĉ - c) + fee_eff·|z*(c)|

  This is convex in ĉ and gives **constant subgradient magnitude** when
  prediction is wrong — unlike soft-action tanh PnL (T60/T81-B) whose
  gradient collapses when ĉ is near 0 or saturated.

We work in the network's scaled output space:
  c_scaled = c · target_scale
  fee_eff_scaled = fee_eff · target_scale
  Network output: pred_scaled
  All loss terms computed in scaled space; gradients balanced w/ L2 anchor.

Combined loss (per sample, weighted by class_balanced_weight on 3-class label):
  L = w·(pred_scaled - y_scaled)^2  +  λ_spo · w · SPO+_scaled

Warm-start from T81 model_T81_seed{S}.pt to avoid the cold-start collapse
that killed T60. Fine-tune at small LR (1e-5 to 3e-5) for ~10 epochs.

Output: model_T87_seed{S}.pt + pred_T87_seed{S}.parquet
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")

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
    """fee_eff = FEE·((mp_th+1)+(mp_t+1))/(mp_t+1).  Per-sample, ≈ 2·FEE."""
    return (FEE * ((mp_th.astype(np.float64) + 1.0) + (mp_t.astype(np.float64) + 1.0))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


class MLPRegr(nn.Module):
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


def predict_chunked(model, X, batch=16384, device="cuda"):
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            out[s:s+batch] = yb
    return out


def spo_plus_loss(pred_scaled, y_scaled, fee_scaled):
    """SPO+ loss in scaled space.

    pred_scaled, y_scaled, fee_scaled: all (B,) tensors.
    y_scaled = c · target_scale, fee_scaled = fee_eff · target_scale.

    z*(c) depends on sign of c relative to fee_eff — equivalent to checking
    sign of y_scaled relative to fee_scaled.
    """
    z_star = torch.where(
        y_scaled > fee_scaled, torch.ones_like(y_scaled),
        torch.where(y_scaled < -fee_scaled, -torch.ones_like(y_scaled),
                    torch.zeros_like(y_scaled)),
    )
    abs_z_star = torch.abs(z_star)
    spread = 2.0 * pred_scaled - y_scaled
    relu_max = F.relu(torch.abs(spread) - fee_scaled)
    ell = relu_max - z_star * spread + fee_scaled * abs_z_star
    return ell  # (B,) — convex per sample, ≥ 0


def evaluate_pnl(pred_unscaled, mp_t, mp_th, thr_up=2*FEE, thr_dn=2*FEE):
    """EV-gate cum_pnl evaluation."""
    pred_action = np.full(len(pred_unscaled), 1, dtype=np.int8)
    pred_action[pred_unscaled > thr_up] = 2
    pred_action[pred_unscaled < -thr_dn] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    return float(pnl.sum()), int((pred_action != 1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="Fine-tune LR (small to preserve T81 init)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lambda-spo", type=float, default=1.0,
                    help="Mixing coefficient for SPO+ term in scaled space."
                         " 0 = pure L2 (= T81 baseline)")
    ap.add_argument("--use-aug", action="store_true", default=True)
    ap.add_argument("--no-aug", dest="use_aug", action="store_false")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.horizon
    seed = args.seed
    tag = args.tag if args.tag else f"lam{args.lambda_spo:g}"
    print(f"=== T87 SPO+ DFL fine-tune seed={seed} h={H} lambda_spo={args.lambda_spo} tag={tag} ===",
          flush=True)
    print(f"  device={device}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    progress("loading_t81_init", seed=seed, lambda_spo=args.lambda_spo)
    t81_pt = os.path.join(T81_DIR, f"model_T81_seed{seed}.pt")
    print(f"  warm-start from {t81_pt}", flush=True)
    ckpt = torch.load(t81_pt, map_location="cpu", weights_only=False)
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
    keep_idx = np.asarray(ckpt["keep_idx"], dtype=np.int64)
    target_scale = float(ckpt["target_scale"])
    hidden = tuple(int(x) for x in ckpt["hidden"])
    dropout = float(ckpt["dropout"])
    use_layernorm = bool(ckpt.get("use_layernorm", True))
    feat_dim = int(ckpt["in_dim"])
    clip_val = float(ckpt.get("clip", 10.0))
    print(f"  T81 ckpt: hidden={hidden} dropout={dropout} use_ln={use_layernorm} "
          f"target_scale={target_scale:.4f} feat_dim={feat_dim} clip={clip_val}",
          flush=True)

    progress("loading_caches", seed=seed)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]

    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), f"FORBIDDEN feat leak: {forbidden & set(feat_names)}"

    # V4 walk-forward (matches T81 exactly): train=date 0-75, val=date 76-79.
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
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]
    fee_eff_va = fee_eff(mp_t_va, mp_th_va)

    X_te = test_full["X"][:, keep_idx]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]

    print(f"  V4 split: n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,}", flush=True)

    # NaN imputation
    nan_mask_tr = np.isnan(X_tr)
    if nan_mask_tr.any():
        for d_idx in np.where(nan_mask_tr.any(axis=0))[0]:
            col_nan = np.isnan(X_tr[:, d_idx])
            X_tr[col_nan, d_idx] = feat_mean[d_idx]
        print(f"  imputed {int(nan_mask_tr.sum()):,} NaN cells in train", flush=True)

    def standardize(X):
        Xs = (X - feat_mean) / np.maximum(feat_std, 1e-6)
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -clip_val, clip_val)
        return Xs.astype(np.float32)

    X_va_std = torch.from_numpy(standardize(X_va))
    X_te_std = torch.from_numpy(standardize(X_te))

    # Free the train_full dict (we already extracted what we need)
    del train_full

    # Pre-build aug copy (concat scheme, matches T81 default)
    if args.use_aug:
        seed_rng = np.random.default_rng(seed * 7919 + 137)
        n_orig = len(X_tr)
        # Allocate the doubled-size array directly, then fill, to avoid 3x peak.
        X_tr_full = np.empty((2 * n_orig, X_tr.shape[1]), dtype=np.float32)
        X_tr_full[:n_orig] = X_tr
        # Stream-multiply scales onto the second half
        for i in range(0, n_orig, 100_000):
            j = min(i + 100_000, n_orig)
            scales = seed_rng.uniform(args.aug_lo, args.aug_hi,
                                      size=(j - i, X_tr.shape[1])).astype(np.float32)
            X_tr_full[n_orig + i:n_orig + j] = X_tr[i:j] * scales
        del X_tr  # free the original train slice
        y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
        fee_eff_tr_full = np.concatenate([fee_eff_tr, fee_eff_tr], axis=0)
        print(f"  aug=concat: X_tr_full={X_tr_full.shape}", flush=True)
    else:
        X_tr_full = X_tr
        y_regr_tr_full = y_regr_tr
        y_cls_tr_full = y_cls_tr
        fee_eff_tr_full = fee_eff_tr

    # Targets: scale by target_scale
    y_scaled_tr = (y_regr_tr_full * target_scale).astype(np.float32)
    fee_scaled_tr = (fee_eff_tr_full * target_scale).astype(np.float32)

    sw_tr_full = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS).astype(np.float32)
    print(f"  sw_tr_full mean={sw_tr_full.mean():.4f} min={sw_tr_full.min():.4f} "
          f"max={sw_tr_full.max():.4f}", flush=True)

    # Tensors
    X_tr_raw = torch.from_numpy(X_tr_full.astype(np.float32))
    y_scaled_tr_t = torch.from_numpy(y_scaled_tr)
    fee_scaled_tr_t = torch.from_numpy(fee_scaled_tr)
    sw_tr_t = torch.from_numpy(sw_tr_full)

    # Build model + load T81 weights
    model = MLPRegr(feat_dim, hidden=hidden, dropout=dropout,
                    use_layernorm=use_layernorm).to(device)
    model.load_state_dict(ckpt["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: hidden={hidden} dropout={dropout} params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(np.maximum(feat_std, 1e-6)).to(device)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T87-spo-seed{seed}-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T87_spo_dfl",
                    "warm_start": "T81",
                    "lambda_spo": args.lambda_spo,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "seed": seed,
                    "horizon": H,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T87", "SPO+", "DFL", f"seed{seed}", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # --- Initial eval (== T81 baseline) ---
    va_pred_init = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred_init = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    va_corr_init = float(np.corrcoef(va_pred_init, y_regr_va)[0, 1])
    te_corr_init = float(np.corrcoef(te_pred_init, y_regr_te)[0, 1])
    val_pnl_init, val_n_act_init = evaluate_pnl(va_pred_init, mp_t_va, mp_th_va)
    cum_init, n_act_init = evaluate_pnl(te_pred_init, mp_t_te, mp_th_te)
    print(f"\n  T81 init eval:  va_corr={va_corr_init:.4f}  te_corr={te_corr_init:.4f}  "
          f"val_pnl={val_pnl_init:+.4f}  test_pnl={cum_init:+.4f}",
          flush=True)
    best_val_pnl = val_pnl_init
    best_te_pnl = cum_init
    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    best_epoch = -1  # -1 = T81 init

    # --- Training loop ---
    n_train = len(X_tr_raw)
    epoch_logs = []
    bad_epochs = 0
    progress("training", seed=seed, n_train=n_train)
    t_train_start = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(n_train)
        running_l2 = 0.0
        running_spo = 0.0
        running_n = 0

        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)
            yb = y_scaled_tr_t[idx].to(device, non_blocking=True)
            fb = fee_scaled_tr_t[idx].to(device, non_blocking=True)
            wb = sw_tr_t[idx].to(device, non_blocking=True)

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
            running_l2 += float(l2_loss.item()) * bs
            running_spo += float(spo_loss.item()) * bs
            running_n += bs

        scheduler.step()
        train_l2 = running_l2 / running_n
        train_spo = running_spo / running_n

        # Validation: corr + EV-gate cum_pnl on val (descaled)
        va_pred_s = predict_chunked(model, X_va_std, batch=16384, device=device)
        va_pred = va_pred_s / target_scale
        val_mse = float(((va_pred - y_regr_va) ** 2).mean())
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
        val_pnl, val_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)

        # Test: same (we track but not for selection — use val_pnl for early-stop)
        te_pred_s = predict_chunked(model, X_te_std, batch=16384, device=device)
        te_pred = te_pred_s / target_scale
        te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
        te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)

        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {"epoch": epoch, "train_l2": train_l2, "train_spo": train_spo,
               "val_mse": val_mse, "val_corr": val_corr,
               "val_pnl": val_pnl, "val_n_act": val_n_act,
               "te_corr": te_corr, "te_pnl": te_pnl, "te_n_act": te_n_act,
               "lr": cur_lr, "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d}  l2={train_l2:.6e}  spo={train_spo:.6e}  "
              f"val_mse={val_mse:.4e}  val_corr={val_corr:.4f}  "
              f"val_pnl={val_pnl:+.3f}  te_pnl={te_pnl:+.3f}  "
              f"lr={cur_lr:.2e}  ({ep_time:.1f}s)",
              flush=True)
        if use_wandb:
            wandb.log(log)

        # Selection: pick model with best val_pnl
        if val_pnl > log_best_val_pnl(epoch_logs, epoch):
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_te_pnl = te_pnl
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    train_time = time.time() - t_train_start
    print(f"\n  train done in {train_time:.1f}s; best_epoch={best_epoch} "
          f"best_val_pnl={[e['val_pnl'] for e in epoch_logs][best_epoch]:+.3f}",
          flush=True)
    model.load_state_dict(best_state)

    # Save model
    model_path = os.path.join(HERE, f"model_T87_seed{seed}_{tag}.pt")
    torch.save({
        "state_dict": best_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx,
        "feat_names": feat_names,
        "hidden": hidden,
        "dropout": dropout,
        "use_layernorm": use_layernorm,
        "in_dim": feat_dim,
        "target_scale": float(target_scale),
        "clip": float(clip_val),
        "best_epoch": best_epoch,
        "lambda_spo": args.lambda_spo,
        "warm_start_from": "T81",
    }, model_path)
    print(f"  saved {model_path}", flush=True)

    # Final pred
    va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    va_pnl, va_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)
    te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)
    print(f"\n  FINAL val:  corr={va_corr:.4f} EV-gate(k=1) cum_pnl={va_pnl:+.4f} n_act={va_n_act:,}",
          flush=True)
    print(f"  FINAL test: corr={te_corr:.4f} EV-gate(k=1) cum_pnl={te_pnl:+.4f} n_act={te_n_act:,}",
          flush=True)
    print(f"  vs T81 init: te_pnl change={te_pnl - cum_init:+.3f}", flush=True)

    # Save pred parquet (matches T81 schema for ev_gate_eval reuse)
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
    pred_path = os.path.join(HERE, f"pred_T87_seed{seed}_{tag}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    summary = {
        "task": f"T87 SPO+ DFL fine-tune seed={seed}",
        "seed": seed,
        "horizon": H,
        "lambda_spo": args.lambda_spo,
        "tag": tag,
        "warm_start_t81": True,
        "n_features": feat_dim,
        "hidden": list(hidden),
        "dropout": dropout,
        "lr": args.lr,
        "epochs_run": len(epoch_logs),
        "best_epoch": best_epoch,
        "n_params": int(n_params),
        "train_time_sec": float(train_time),
        "init": {"te_corr": te_corr_init, "te_pnl": cum_init, "te_n_act": n_act_init},
        "val": {"corr": va_corr, "ev_gate_k1_cum_pnl": va_pnl, "ev_gate_k1_n_active": va_n_act},
        "test": {"corr": te_corr, "ev_gate_k1_cum_pnl": te_pnl, "ev_gate_k1_n_active": te_n_act},
        "delta_te_pnl_vs_t81": te_pnl - cum_init,
        "epoch_logs": epoch_logs,
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T87_seed{seed}_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_epoch": best_epoch,
            "best_val_pnl": [e['val_pnl'] for e in epoch_logs][best_epoch] if best_epoch >= 0 else va_pnl,
            "final_te_corr": te_corr,
            "final_te_pnl": te_pnl,
            "delta_vs_t81_init": te_pnl - cum_init,
        })
        wandb.finish()

    progress("done", seed=seed, lambda_spo=args.lambda_spo,
             best_epoch=best_epoch, te_pnl=te_pnl, delta=te_pnl - cum_init)


def log_best_val_pnl(logs, current_epoch):
    """Return the best val_pnl seen *before* current_epoch.  If none, returns -inf."""
    if current_epoch == 0:
        return -float("inf")
    return max(l["val_pnl"] for l in logs[:current_epoch])


if __name__ == "__main__":
    main()
