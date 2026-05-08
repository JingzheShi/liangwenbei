"""T135: Snapshot Ensemble for T87 SPO+ NN.

Idea: cyclic LR (4 cycles x 4 epochs = 16 total) with cosine annealing warm
restarts. Save model snapshot at end of each cycle (epoch 3, 7, 11, 15) when
LR is at min. Following Huang et al 2017, "Snapshot Ensembles: Train 1, get
M for free" — averaging M snapshots typically yields +0.5~+1 PnL.

Training loop is identical to T87 SPO+ except:
  - Replace CosineAnnealingLR with CosineAnnealingWarmRestarts(T_0=4, T_mult=1)
  - Train 16 epochs (no early stopping)
  - At end of each cycle (LR low), save snapshot weights + test pred parquet.

Output per seed:
  model_T135_seed{S}_cyc{C}.pt        for C in [1,2,3,4]
  pred_T135_seed{S}_cyc{C}.parquet    for C in [1,2,3,4]
  summary_T135_seed{S}.json
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
    z_star = torch.where(
        y_scaled > fee_scaled, torch.ones_like(y_scaled),
        torch.where(y_scaled < -fee_scaled, -torch.ones_like(y_scaled),
                    torch.zeros_like(y_scaled)),
    )
    abs_z_star = torch.abs(z_star)
    spread = 2.0 * pred_scaled - y_scaled
    relu_max = F.relu(torch.abs(spread) - fee_scaled)
    ell = relu_max - z_star * spread + fee_scaled * abs_z_star
    return ell


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


def save_snapshot(model, cycle_idx, seed, target_scale, hidden, dropout,
                  use_layernorm, feat_dim, feat_mean, feat_std, keep_idx,
                  feat_names, clip_val, lambda_spo, X_va_std, X_te_std,
                  y_regr_va, y_regr_te, mp_t_va, mp_th_va, mp_t_te, mp_th_te,
                  test_full, y_cls_te, device):
    """Save model state + test pred parquet for one snapshot."""
    state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    model_path = os.path.join(HERE, f"model_T135_seed{seed}_cyc{cycle_idx}.pt")
    torch.save({
        "state_dict": state,
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
        "cycle_idx": cycle_idx,
        "lambda_spo": lambda_spo,
        "warm_start_from": "T81",
    }, model_path)

    va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
    va_pnl, va_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)
    te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)

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
    pred_path = os.path.join(HERE, f"pred_T135_seed{seed}_cyc{cycle_idx}.parquet")
    te_df.to_parquet(pred_path, index=False)
    return {
        "cycle_idx": cycle_idx,
        "model_path": model_path,
        "pred_path": pred_path,
        "va_corr": va_corr, "te_corr": te_corr,
        "va_pnl": va_pnl, "va_n_act": va_n_act,
        "te_pnl": te_pnl, "te_n_act": te_n_act,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-min", type=float, default=1e-7,
                    help="Cosine annealing min LR (eta_min)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--cycle-len", type=int, default=4,
                    help="Epochs per cosine cycle (T_0)")
    ap.add_argument("--n-cycles", type=int, default=4)
    ap.add_argument("--lambda-spo", type=float, default=30.0)
    ap.add_argument("--use-aug", action="store_true", default=True)
    ap.add_argument("--no-aug", dest="use_aug", action="store_false")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.horizon
    seed = args.seed
    n_epochs = args.cycle_len * args.n_cycles
    print(f"=== T135 Snapshot Ensemble seed={seed} h={H} cycle_len={args.cycle_len} "
          f"n_cycles={args.n_cycles} total_epochs={n_epochs} lambda_spo={args.lambda_spo} ===",
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

    # V4 walk-forward: train=date 0-75, val=date 76-79.
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

    X_te = test_full["X"][:, keep_idx]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

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

    del train_full

    # Aug
    if args.use_aug:
        seed_rng = np.random.default_rng(seed * 7919 + 137)
        n_orig = len(X_tr)
        X_tr_full = np.empty((2 * n_orig, X_tr.shape[1]), dtype=np.float32)
        X_tr_full[:n_orig] = X_tr
        for i in range(0, n_orig, 100_000):
            j = min(i + 100_000, n_orig)
            scales = seed_rng.uniform(args.aug_lo, args.aug_hi,
                                      size=(j - i, X_tr.shape[1])).astype(np.float32)
            X_tr_full[n_orig + i:n_orig + j] = X_tr[i:j] * scales
        del X_tr
        y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
        fee_eff_tr_full = np.concatenate([fee_eff_tr, fee_eff_tr], axis=0)
        print(f"  aug=concat: X_tr_full={X_tr_full.shape}", flush=True)
    else:
        X_tr_full = X_tr
        y_regr_tr_full = y_regr_tr
        y_cls_tr_full = y_cls_tr
        fee_eff_tr_full = fee_eff_tr

    y_scaled_tr = (y_regr_tr_full * target_scale).astype(np.float32)
    fee_scaled_tr = (fee_eff_tr_full * target_scale).astype(np.float32)
    sw_tr_full = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS).astype(np.float32)

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
    # Cyclic LR with cosine warm restarts: reset every cycle_len epochs.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.cycle_len, T_mult=1, eta_min=args.lr_min,
    )

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(np.maximum(feat_std, 1e-6)).to(device)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T135-snap-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T135_snapshot_ensemble",
                    "warm_start": "T81",
                    "lambda_spo": args.lambda_spo,
                    "lr": args.lr,
                    "lr_min": args.lr_min,
                    "cycle_len": args.cycle_len,
                    "n_cycles": args.n_cycles,
                    "n_epochs": n_epochs,
                    "seed": seed,
                    "horizon": H,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T135", "snapshot", f"seed{seed}", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # Initial eval (T81 baseline)
    va_pred_init = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred_init = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
    va_corr_init = float(np.corrcoef(va_pred_init, y_regr_va)[0, 1])
    te_corr_init = float(np.corrcoef(te_pred_init, y_regr_te)[0, 1])
    val_pnl_init, _ = evaluate_pnl(va_pred_init, mp_t_va, mp_th_va)
    cum_init, n_act_init = evaluate_pnl(te_pred_init, mp_t_te, mp_th_te)
    print(f"\n  T81 init eval:  va_corr={va_corr_init:.4f}  te_corr={te_corr_init:.4f}  "
          f"val_pnl={val_pnl_init:+.4f}  test_pnl={cum_init:+.4f}",
          flush=True)

    snapshot_logs = []
    epoch_logs = []
    n_train = len(X_tr_raw)
    progress("training", seed=seed, n_train=n_train, n_epochs=n_epochs)
    t_train_start = time.time()

    for epoch in range(n_epochs):
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

        # Step scheduler AFTER each epoch — drives the cosine cycle.
        scheduler.step()
        train_l2 = running_l2 / running_n
        train_spo = running_spo / running_n

        # Per-epoch eval (lightweight)
        va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
        val_pnl, val_n_act = evaluate_pnl(va_pred, mp_t_va, mp_th_va)
        te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale
        te_pnl, te_n_act = evaluate_pnl(te_pred, mp_t_te, mp_th_te)
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0

        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {"epoch": epoch, "train_l2": train_l2, "train_spo": train_spo,
               "val_corr": val_corr, "val_pnl": val_pnl, "val_n_act": val_n_act,
               "te_pnl": te_pnl, "te_n_act": te_n_act,
               "lr": cur_lr, "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d}  l2={train_l2:.6e}  spo={train_spo:.6e}  "
              f"val_corr={val_corr:.4f}  val_pnl={val_pnl:+.3f}  te_pnl={te_pnl:+.3f}  "
              f"lr={cur_lr:.2e}  ({ep_time:.1f}s)",
              flush=True)
        if use_wandb:
            wandb.log(log)

        # Snapshot at end of each cycle (LR low). After scheduler.step(),
        # T_cur=cycle_len → lr=eta_min on the LAST step of the cycle.
        # Last epoch in cycle c (0-indexed) = c*cycle_len + (cycle_len-1)
        cycle_completed = (epoch + 1) % args.cycle_len == 0
        if cycle_completed:
            cycle_idx = (epoch + 1) // args.cycle_len  # 1, 2, 3, 4
            print(f"  >> saving snapshot cycle {cycle_idx} at epoch {epoch}", flush=True)
            snap = save_snapshot(
                model, cycle_idx, seed, target_scale, hidden, dropout,
                use_layernorm, feat_dim, feat_mean, feat_std, keep_idx,
                feat_names, clip_val, args.lambda_spo, X_va_std, X_te_std,
                y_regr_va, y_regr_te, mp_t_va, mp_th_va, mp_t_te, mp_th_te,
                test_full, y_cls_te, device,
            )
            print(f"     snap te_pnl={snap['te_pnl']:+.3f} va_pnl={snap['va_pnl']:+.3f} "
                  f"te_corr={snap['te_corr']:.4f}", flush=True)
            snapshot_logs.append(snap)

    train_time = time.time() - t_train_start
    print(f"\n  train done in {train_time:.1f}s; saved {len(snapshot_logs)} snapshots",
          flush=True)

    # Compute snapshot ensemble (avg) for this seed
    pred_sum = np.zeros(len(X_te), dtype=np.float64)
    for snap in snapshot_logs:
        df_s = pd.read_parquet(snap["pred_path"])
        pred_sum += df_s["pred_dmid_norm"].to_numpy(np.float64)
    pred_avg = pred_sum / len(snapshot_logs)
    snap_avg_pnl, snap_avg_n_act = evaluate_pnl(pred_avg, mp_t_te, mp_th_te)
    snap_avg_corr = float(np.corrcoef(pred_avg, y_regr_te)[0, 1])
    print(f"\n  SEED {seed} snapshot avg: te_pnl={snap_avg_pnl:+.4f} corr={snap_avg_corr:.4f} "
          f"n_act={snap_avg_n_act:,}", flush=True)

    summary = {
        "task": f"T135 Snapshot Ensemble seed={seed}",
        "seed": seed,
        "horizon": H,
        "lambda_spo": args.lambda_spo,
        "cycle_len": args.cycle_len,
        "n_cycles": args.n_cycles,
        "n_epochs": n_epochs,
        "lr": args.lr,
        "lr_min": args.lr_min,
        "init": {"te_corr": te_corr_init, "te_pnl": cum_init, "te_n_act": n_act_init},
        "snapshots": snapshot_logs,
        "snapshot_avg": {"te_pnl": snap_avg_pnl, "te_corr": snap_avg_corr,
                         "n_act": snap_avg_n_act},
        "epoch_logs": epoch_logs,
        "train_time_sec": float(train_time),
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T135_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "snapshot_avg_te_pnl": snap_avg_pnl,
            "snapshot_avg_te_corr": snap_avg_corr,
            "delta_vs_t81_init": snap_avg_pnl - cum_init,
        })
        wandb.finish()

    progress("done", seed=seed, snapshot_avg_pnl=snap_avg_pnl,
             n_snapshots=len(snapshot_logs))


if __name__ == "__main__":
    main()
