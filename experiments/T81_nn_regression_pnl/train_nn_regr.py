"""T81: NN regression on Δmid_norm (sister of T75 LightGBM regression).

Architecture: MLP [359 → 512 → 256 → 128 → 1], GELU, Dropout, LayerNorm.
Loss: weighted L2 (sample-weight = class-balanced 3-class label, identical to T75).
Augment: per-(sample, feat) random scale [0.8, 1.2] applied **on raw features**
  before standardization (matches T75 aug_a semantics).
Optimizer: AdamW, lr=1e-3, weight_decay=1e-4.
Schedule: ~30 epochs with cosine LR + early stop on val MSE (patience=5).

Same V4 walk-forward split as T75: train=date 0-75, val=date 76-79.
Same 5-seed configs as T70/T75 (different seed only — width is shared).
Same drop set: T59 fail (10) + Stage5 fail (1) = 11 features → 359 dim.

CRITICAL_CONSTRAINTS compliance:
  * Model.forward() takes only X (359-d standardized vector). No sym, no date.
  * Standardization uses GLOBAL train stats (no per-sym).
  * Predictor uses identical pipeline at inference time, no state.
  * aug_a is a training-time-only randomization.

Saves model_T81_seed{S}.pt and pred_T81_seed{S}.parquet.
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
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# Same seed pool as T75. NN doesn't have feat/bag fraction analogues but we
# vary dropout/init slightly via seed.
SEED_LIST = (42, 1, 7, 13, 100)

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
    """y_regr = (mp_th - mp_t) / (mp_t + 1)  — exactly the PnL numerator/denominator."""
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


class MLPRegr(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 256, 128), dropout: float = 0.2,
                 use_layernorm: bool = True):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def predict_chunked(model: nn.Module, X: torch.Tensor, batch: int = 16384, device="cuda") -> np.ndarray:
    """Return predictions on X (already standardized, on CPU). Returns numpy array."""
    model.eval()
    out = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            out[s:s+batch] = yb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--hidden", default="256,128,64")
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--target-scale", type=float, default=0.0,
                    help="If 0, auto-set to 1/std(y_regr). Else explicit factor.")
    ap.add_argument("--use-sample-weight", action="store_true", default=True,
                    help="Use class-balanced sample weights (matches T75)")
    ap.add_argument("--no-sample-weight", dest="use_sample_weight", action="store_false")
    ap.add_argument("--aug-mode", default="concat", choices=["concat", "online", "none"],
                    help="concat: pre-build aug copy & concat (T75-style); "
                         "online: per-batch aug; none: no aug")
    ap.add_argument("--aug-prob", type=float, default=1.0,
                    help="Probability of applying aug_a to a batch (1.0 = always)")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--use-layernorm", action="store_true", default=True)
    ap.add_argument("--no-layernorm", dest="use_layernorm", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--variant", default="A_regr_l2",
                    choices=["A_regr_l2", "B_regr_pnl_aux", "C_pnl_only"],
                    help="Loss variant")
    ap.add_argument("--pnl-aux-weight", type=float, default=0.3,
                    help="Weight λ for aux PnL loss in variant B")
    ap.add_argument("--soft-thr", type=float, default=2e-4,
                    help="Soft threshold for action_soft = tanh(pred/thr) (variants B/C)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H = args.horizon
    seed = args.seed
    print(f"=== T81 NN regression seed={seed} h={H} variant={args.variant} ===", flush=True)
    print(f"  device={device}", flush=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    progress("loading_caches", seed=seed, variant=args.variant)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

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
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_idx], dtype=np.int64,
    )
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)} fail features)", flush=True)

    forbidden = {"date", "sym", "time"}
    feat_names = [all_feat_names[i] for i in keep_idx]
    assert not (forbidden & set(feat_names))

    # V4 walk-forward (matches T75 exactly)
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr = train_full["X"][m_t][:, keep_idx]
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    mp_t_tr = train_full["mp_t"][m_t]
    mp_th_tr = train_full["mp_t60"][m_t]

    X_va = train_full["X"][m_va][:, keep_idx]
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]

    print(f"  V4 split: n_train={len(X_tr):,} n_val={len(X_va):,}", flush=True)
    print(f"  y_regr_tr: mean={y_regr_tr.mean():.6f} std={y_regr_tr.std():.6f}", flush=True)
    print(f"  y_regr_va: mean={y_regr_va.mean():.6f} std={y_regr_va.std():.6f}", flush=True)

    # --- Standardization stats (GLOBAL, no per-sym) ---
    # Use nanmean/nanstd (some LOB features have NaN when fewer ask/bid levels exist).
    # NaN cells are imputed with the feature mean (= 0 after standardize).
    # Standardized values are clipped to ±10 to handle extreme outliers (e.g. kyle_inv).
    feat_mean = np.nanmean(X_tr, axis=0).astype(np.float32)
    feat_std = np.nanstd(X_tr, axis=0).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-6)
    CLIP = 10.0

    def standardize(X):
        Xs = (X - feat_mean) / feat_std
        # NaN → 0 (= imputed with mean)
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -CLIP, CLIP)
        return Xs.astype(np.float32)

    # Sanity: check val/test for any unseen NaN columns
    print(f"  std stats: feat_mean range [{feat_mean.min():.3e}, {feat_mean.max():.3e}]"
          f"  feat_std range [{feat_std.min():.3e}, {feat_std.max():.3e}]"
          f"  n_with_NaN_in_train={int(np.isnan(X_tr).any(axis=0).sum())}", flush=True)

    # Test slice
    X_te = test_full["X"][:, keep_idx]
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]

    # Standardized tensors for val/test (no aug → standardize once)
    X_va_std = torch.from_numpy(standardize(X_va))
    X_te_std = torch.from_numpy(standardize(X_te))
    y_va_t = torch.from_numpy(y_regr_va).float()

    # For training, pre-impute NaN with mean so aug_a multiplies clean values.
    X_tr_imputed = X_tr.copy()
    nan_mask = np.isnan(X_tr_imputed)
    if nan_mask.any():
        for d_idx in np.where(np.isnan(X_tr_imputed).any(axis=0))[0]:
            col_nan = np.isnan(X_tr_imputed[:, d_idx])
            X_tr_imputed[col_nan, d_idx] = feat_mean[d_idx]
    print(f"  imputed {int(nan_mask.sum()):,} NaN cells with feat_mean", flush=True)

    # --- Target scaling ---
    if args.target_scale > 0:
        target_scale = float(args.target_scale)
    else:
        target_scale = 1.0 / float(max(y_regr_tr.std(), 1e-8))
    print(f"  target_scale={target_scale:.4f}  (y_regr_tr.std()={y_regr_tr.std():.6e})", flush=True)
    y_regr_tr_s = (y_regr_tr * target_scale).astype(np.float32)
    y_regr_va_s = (y_regr_va * target_scale).astype(np.float32)

    # --- aug_a strategy ---
    if args.aug_mode == "concat":
        # Pre-build aug copy: same len as X_tr, scaled per-(sample, feat) [aug_lo, aug_hi].
        seed_rng = np.random.default_rng(seed * 7919 + 1)
        scales = seed_rng.uniform(args.aug_lo, args.aug_hi, size=X_tr_imputed.shape).astype(np.float32)
        X_aug = X_tr_imputed * scales
        X_tr_full = np.concatenate([X_tr_imputed, X_aug], axis=0)
        y_tr_full = np.concatenate([y_regr_tr_s, y_regr_tr_s], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
        mp_t_tr_full = np.concatenate([mp_t_tr, mp_t_tr], axis=0)
        mp_th_tr_full = np.concatenate([mp_th_tr, mp_th_tr], axis=0)
        print(f"  aug=concat: X_tr_full={X_tr_full.shape} (orig + aug)", flush=True)
    else:
        X_tr_full = X_tr_imputed
        y_tr_full = y_regr_tr_s
        y_cls_tr_full = y_cls_tr
        mp_t_tr_full = mp_t_tr
        mp_th_tr_full = mp_th_tr
        print(f"  aug={args.aug_mode}: X_tr_full={X_tr_full.shape}", flush=True)


    # --- Train tensors (raw imputed, will standardize per batch; aug already concatenated if mode=concat) ---
    X_tr_raw = torch.from_numpy(X_tr_full.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr_full).float()
    if args.use_sample_weight:
        sw_tr_full = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    else:
        sw_tr_full = np.ones(len(X_tr_full), dtype=np.float32)
    sw_tr_t = torch.from_numpy(sw_tr_full).float()
    # mp for variants B/C (aux PnL loss); already scaled with target_scale in y_tr_t
    mp_t_tr_t = torch.from_numpy(mp_t_tr_full.astype(np.float32)).float()
    mp_th_tr_t = torch.from_numpy(mp_th_tr_full.astype(np.float32)).float()
    print(f"  sw_tr_full: mean={sw_tr_full.mean():.4f} min={sw_tr_full.min():.4f} max={sw_tr_full.max():.4f}",
          flush=True)

    # --- WandB ---
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T81-nn-{args.variant}-seed{seed}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "MLP-LayerNorm-GELU",
                    "task": "T81_nn_regression_pnl",
                    "variant": args.variant,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T81", "NN", "regression", "EV-gate", f"h{H}", args.variant],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # --- Model & Optim ---
    hidden = tuple(int(x) for x in args.hidden.split(","))
    model = MLPRegr(feat_dim, hidden=hidden, dropout=args.dropout,
                    use_layernorm=args.use_layernorm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: in_dim={feat_dim} hidden={hidden} dropout={args.dropout} params={n_params:,}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Pre-compute mean/std as tensors for batched standardization
    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(feat_std).to(device)

    best_val_mse = float("inf")
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    n_train = len(X_tr_raw)
    epoch_logs = []

    progress("training", seed=seed, n_train=n_train, n_val=len(X_va))

    t_train_start = time.time()
    for epoch in range(args.epochs):
        t_ep = time.time()
        model.train()
        # Shuffle indices
        perm = torch.randperm(n_train)
        running_loss = 0.0
        running_mse = 0.0
        running_n = 0

        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)  # (B, D)
            yb = y_tr_t[idx].to(device, non_blocking=True)  # already target_scale-multiplied
            wb = sw_tr_t[idx].to(device, non_blocking=True)

            # aug_a online (only if --aug-mode online)
            if args.aug_mode == "online":
                scales = torch.empty_like(xb_raw).uniform_(args.aug_lo, args.aug_hi)
                xb_raw = xb_raw * scales

            # Standardize using global stats; clip to ±CLIP
            xb = (xb_raw - fm_t) / fs_t
            xb = torch.clamp(xb, -CLIP, CLIP)

            pred = model(xb)  # in scaled space (~ unit std)
            mse_per = (pred - yb) ** 2
            l2_loss = (mse_per * wb).sum() / wb.sum().clamp_min(1.0)

            if args.variant == "A_regr_l2":
                loss = l2_loss
            elif args.variant == "B_regr_pnl_aux":
                # PnL is computed on UNSCALED prediction (pred / target_scale).
                pred_unscaled = pred / target_scale
                mp_t_b = mp_t_tr_t[idx].to(device)
                mp_th_b = mp_th_tr_t[idx].to(device)
                action_soft = torch.tanh(pred_unscaled / args.soft_thr)
                diff = mp_th_b - mp_t_b
                fee_pnl = FEE * action_soft.abs() * ((mp_th_b + 1.0) + (mp_t_b + 1.0))
                denom = mp_t_b + 1.0
                pnl = (action_soft * diff - fee_pnl) / denom
                pnl_loss = -pnl.mean()
                loss = l2_loss + args.pnl_aux_weight * pnl_loss
            elif args.variant == "C_pnl_only":
                pred_unscaled = pred / target_scale
                mp_t_b = mp_t_tr_t[idx].to(device)
                mp_th_b = mp_th_tr_t[idx].to(device)
                action_soft = torch.tanh(pred_unscaled / args.soft_thr)
                diff = mp_th_b - mp_t_b
                fee_pnl = FEE * action_soft.abs() * ((mp_th_b + 1.0) + (mp_t_b + 1.0))
                denom = mp_t_b + 1.0
                pnl = (action_soft * diff - fee_pnl) / denom
                loss = -pnl.mean()
            else:
                raise ValueError(args.variant)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            running_mse += float(l2_loss.item()) * bs
            running_n += bs

        scheduler.step()
        train_loss_avg = running_loss / running_n
        train_mse_avg = running_mse / running_n

        # --- Validation (predict in scaled space, descale to compare to y_regr_va) ---
        va_pred_s = predict_chunked(model, X_va_std, batch=16384, device=device)
        va_pred = va_pred_s / target_scale  # back to y_regr scale
        val_mse = float(((va_pred - y_regr_va) ** 2).mean())
        val_mae = float(np.abs(va_pred - y_regr_va).mean())
        val_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0

        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {"epoch": epoch, "train_loss": train_loss_avg, "train_mse": train_mse_avg,
               "val_mse": val_mse, "val_mae": val_mae, "val_corr": val_corr,
               "lr": cur_lr, "time": ep_time}
        epoch_logs.append(log)
        print(f"  ep {epoch:3d}  train_loss={train_loss_avg:.6e}  train_mse={train_mse_avg:.6e}  "
              f"val_mse={val_mse:.6e}  val_corr={val_corr:.4f}  lr={cur_lr:.2e}  ({ep_time:.1f}s)",
              flush=True)

        if use_wandb:
            wandb.log({**log, "epoch": epoch})

        if val_mse < best_val_mse - 1e-10:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"  early stop at epoch {epoch} (no improvement for {args.patience})",
                      flush=True)
                break

    train_time = time.time() - t_train_start
    print(f"\n  train done in {train_time:.1f}s; best_epoch={best_epoch} best_val_mse={best_val_mse:.6e}",
          flush=True)
    assert best_state is not None, "training never improved"
    model.load_state_dict(best_state)

    # Save model & metadata
    model_path = os.path.join(HERE, f"model_T81_seed{seed}.pt")
    torch.save({
        "state_dict": best_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx,
        "feat_names": feat_names,
        "hidden": hidden,
        "dropout": args.dropout,
        "use_layernorm": args.use_layernorm,
        "in_dim": feat_dim,
        "target_scale": float(target_scale),
        "clip": float(CLIP),
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "variant": args.variant,
    }, model_path)
    print(f"  saved {model_path}", flush=True)

    # Final predictions on val and test (descale)
    va_pred = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scale
    te_pred = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scale

    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0

    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0
    print(f"  VAL: mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST: mse={te_mse:.7f} mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    # Save predictions parquet (matching T75 schema for ev_gate_eval reuse)
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
    pred_path = os.path.join(HERE, f"pred_T81_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    # Quick raw EV-gate sanity at k=1
    fee_thr = 2.0 * FEE
    pred_action = np.full(len(te_pred), 1, dtype=np.int8)
    pred_action[te_pred > fee_thr] = 2
    pred_action[te_pred < -fee_thr] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  TEST EV-gate k=1 (thr={fee_thr:.5f}): cum_pnl={cum_pnl:+.4f} n_active={n_active:,}",
          flush=True)

    summary = {
        "task": f"T81 NN regression seed={seed}",
        "seed": seed,
        "horizon": H,
        "variant": args.variant,
        "split_info": {"strategy": "V4", "train_dates": "0-75", "val_dates": "76-79"},
        "n_features": feat_dim,
        "drop_names": list(DROP_NAMES),
        "hidden": list(hidden),
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_run": epoch_logs[-1]["epoch"] + 1 if epoch_logs else 0,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "n_params": int(n_params),
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
        "epoch_logs": epoch_logs,
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T81_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "train_time_sec": float(train_time),
            "test_mse": te_mse, "test_mae": te_mae, "test_corr": te_corr,
            "test_ev_gate_k1_cum_pnl": cum_pnl,
            "test_ev_gate_k1_n_active": n_active,
        })
        wandb.finish()
    progress("done", seed=seed, variant=args.variant,
             best_epoch=best_epoch, val_mse=best_val_mse, test_corr=te_corr,
             test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
