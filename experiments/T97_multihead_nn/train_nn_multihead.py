"""T97: Multi-head NN regression on Δmid_norm at multiple horizons.

Architecture: shared trunk [359 -> 256 -> 128 -> 64] LayerNorm GELU Dropout
  + multiple horizon heads, each [64 -> 1].
  Default horizons = {10, 20, 40, 60}; loss weights w_h tuned so h=60 dominates.

Loss: weighted sum of per-head L2 losses on (per-horizon target_scale)-scaled targets.
Each head optimises its own per-horizon Δmid_norm regression target. Sample weight
is class-balanced based on h=60 3-class label (matches T81 / T75).

Goal: shared trunk learns more generic micro-structure features → reduced overfit on
h=60 specific noise while still optimising the h=60 head specifically.

Inference: only h=60 head output is used (cheap; trunk + 1 head ≈ T81 single-head cost).

Saves model_T97_seed{S}.pt and pred_T97_seed{S}.parquet.
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


class MultiHeadMLP(nn.Module):
    """Shared trunk + per-horizon heads. Single forward returns (B, n_heads)."""
    def __init__(self, in_dim: int, trunk=(256, 128, 64), n_heads: int = 4,
                 dropout: float = 0.10, use_layernorm: bool = True,
                 head_hidden: int = 0):
        super().__init__()
        layers = []
        d = in_dim
        for h in trunk:
            layers.append(nn.Linear(d, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d = h
        self.trunk = nn.Sequential(*layers)
        self.trunk_dim = d
        # Heads
        head_modules = []
        for _ in range(n_heads):
            if head_hidden > 0:
                head_modules.append(
                    nn.Sequential(
                        nn.Linear(d, head_hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(head_hidden, 1),
                    )
                )
            else:
                head_modules.append(nn.Linear(d, 1))
        self.heads = nn.ModuleList(head_modules)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        out = torch.stack([h(z).squeeze(-1) for h in self.heads], dim=1)  # (B, n_heads)
        return out


def predict_chunked(model: nn.Module, X: torch.Tensor, batch: int = 16384,
                    device="cuda") -> np.ndarray:
    """Returns (N, n_heads) numpy."""
    model.eval()
    out = None
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s:s+batch].to(device, non_blocking=True)
            yb = model(xb).detach().cpu().numpy()
            if out is None:
                out = np.zeros((len(X), yb.shape[1]), dtype=np.float32)
            out[s:s+batch] = yb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizons", default="10,20,40,60",
                    help="Comma-list of horizons to predict (must be subset of {5,10,20,40,60})")
    ap.add_argument("--head-weights", default="auto",
                    help="Comma-list matching --horizons or 'auto'. "
                         "'auto' = h=60 weight 1.0, others 0.3.")
    ap.add_argument("--main-horizon", type=int, default=60,
                    help="Horizon used for early stop / sample weight / inference.")
    ap.add_argument("--trunk", default="256,128,64")
    ap.add_argument("--head-hidden", type=int, default=0,
                    help="0 = simple Linear(64->1) head; >0 = [64->H->1] (soft-share)")
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--use-sample-weight", action="store_true", default=True)
    ap.add_argument("--no-sample-weight", dest="use_sample_weight", action="store_false")
    ap.add_argument("--aug-mode", default="concat", choices=["concat", "online", "none"])
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--use-layernorm", action="store_true", default=True)
    ap.add_argument("--no-layernorm", dest="use_layernorm", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--cuda", default="0")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    horizons = [int(x) for x in args.horizons.split(",")]
    n_heads = len(horizons)
    main_h = int(args.main_horizon)
    assert main_h in horizons, f"main_horizon {main_h} not in horizons {horizons}"
    main_idx = horizons.index(main_h)

    if args.head_weights == "auto":
        head_weights = np.array([1.0 if h == main_h else 0.3 for h in horizons],
                                dtype=np.float32)
    else:
        head_weights = np.array([float(x) for x in args.head_weights.split(",")],
                                dtype=np.float32)
        assert len(head_weights) == n_heads
    print(f"=== T97 multi-head NN seed={args.seed} horizons={horizons} weights={head_weights.tolist()} ===", flush=True)
    print(f"  main_horizon={main_h} main_idx={main_idx} tag={args.tag}", flush=True)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    progress("loading_caches", seed=seed, horizons=horizons)
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

    # V4 walk-forward
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr = train_full["X"][m_t][:, keep_idx]
    X_va = train_full["X"][m_va][:, keep_idx]

    # Build per-horizon regression targets (train + val)
    y_regr_tr_per = np.zeros((len(X_tr), n_heads), dtype=np.float32)
    y_regr_va_per = np.zeros((len(X_va), n_heads), dtype=np.float32)
    y_regr_te_per = np.zeros((len(test_full["mp_t"]), n_heads), dtype=np.float32)
    for hi, h in enumerate(horizons):
        mp_t_full = train_full["mp_t"]
        mp_th_full = train_full[f"mp_t{h}"]
        y_regr_tr_per[:, hi] = regr_target(mp_t_full[m_t], mp_th_full[m_t])
        y_regr_va_per[:, hi] = regr_target(mp_t_full[m_va], mp_th_full[m_va])
        y_regr_te_per[:, hi] = regr_target(test_full["mp_t"], test_full[f"mp_t{h}"])

    # Main-horizon labels (for sample weight + EV gate)
    y_cls_tr_main = train_full[f"y{main_h}"][m_t].astype(np.int64)
    y_cls_va_main = train_full[f"y{main_h}"][m_va].astype(np.int64)
    y_cls_te_main = test_full[f"y{main_h}"].astype(np.int64)
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{main_h}"]

    print(f"  V4 split: n_train={len(X_tr):,} n_val={len(X_va):,}", flush=True)
    for hi, h in enumerate(horizons):
        print(f"    h={h:3d}: y_regr_tr std={y_regr_tr_per[:,hi].std():.6e} "
              f"y_regr_va std={y_regr_va_per[:,hi].std():.6e}", flush=True)

    # Standardization (GLOBAL, no per-sym)
    feat_mean = np.nanmean(X_tr, axis=0).astype(np.float32)
    feat_std = np.nanstd(X_tr, axis=0).astype(np.float32)
    feat_std = np.maximum(feat_std, 1e-6)
    CLIP = 10.0

    def standardize(X):
        Xs = (X - feat_mean) / feat_std
        Xs = np.where(np.isnan(Xs), 0.0, Xs)
        Xs = np.clip(Xs, -CLIP, CLIP)
        return Xs.astype(np.float32)

    # Per-horizon target_scale = 1/std(y_regr_tr_h)
    target_scales = np.array([1.0 / max(y_regr_tr_per[:, hi].std(), 1e-8)
                              for hi in range(n_heads)], dtype=np.float32)
    print(f"  target_scales={target_scales.tolist()}", flush=True)

    y_regr_tr_per_s = (y_regr_tr_per * target_scales[None, :]).astype(np.float32)
    y_regr_va_per_s = (y_regr_va_per * target_scales[None, :]).astype(np.float32)

    # Test slice
    X_te = test_full["X"][:, keep_idx]
    sym_te = test_full["sym"]

    X_va_std = torch.from_numpy(standardize(X_va))
    X_te_std = torch.from_numpy(standardize(X_te))

    # Pre-impute NaN with mean for training (so aug multiplies clean values)
    X_tr_imputed = X_tr.copy()
    nan_mask = np.isnan(X_tr_imputed)
    if nan_mask.any():
        for d_idx in np.where(np.isnan(X_tr_imputed).any(axis=0))[0]:
            col_nan = np.isnan(X_tr_imputed[:, d_idx])
            X_tr_imputed[col_nan, d_idx] = feat_mean[d_idx]
    print(f"  imputed {int(nan_mask.sum()):,} NaN cells with feat_mean", flush=True)

    # aug_a strategy
    if args.aug_mode == "concat":
        seed_rng = np.random.default_rng(seed * 7919 + 1)
        scales = seed_rng.uniform(args.aug_lo, args.aug_hi,
                                  size=X_tr_imputed.shape).astype(np.float32)
        X_aug = X_tr_imputed * scales
        X_tr_full = np.concatenate([X_tr_imputed, X_aug], axis=0)
        y_tr_full_per_s = np.concatenate([y_regr_tr_per_s, y_regr_tr_per_s], axis=0)
        y_cls_tr_full_main = np.concatenate([y_cls_tr_main, y_cls_tr_main], axis=0)
        print(f"  aug=concat: X_tr_full={X_tr_full.shape}", flush=True)
    else:
        X_tr_full = X_tr_imputed
        y_tr_full_per_s = y_regr_tr_per_s
        y_cls_tr_full_main = y_cls_tr_main
        print(f"  aug={args.aug_mode}: X_tr_full={X_tr_full.shape}", flush=True)

    X_tr_raw = torch.from_numpy(X_tr_full.astype(np.float32))
    y_tr_per_t = torch.from_numpy(y_tr_full_per_s).float()  # (N, n_heads), scaled
    if args.use_sample_weight:
        sw = class_balanced_weight(y_cls_tr_full_main, num_class=NUM_CLASS)
    else:
        sw = np.ones(len(X_tr_full), dtype=np.float32)
    sw_tr_t = torch.from_numpy(sw).float()
    print(f"  sw: mean={sw.mean():.4f} min={sw.min():.4f} max={sw.max():.4f}",
          flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T97-mh-h{'_'.join(str(h) for h in horizons)}-seed{seed}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "MultiHead-MLP",
                    "task": "T97_multihead_nn",
                    "horizons": horizons,
                    "head_weights": head_weights.tolist(),
                    "main_horizon": main_h,
                    "n_features": feat_dim,
                    "seed": seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T97", "NN", "multi-head", "regression"] + [f"h{h}" for h in horizons],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # Model & Optim
    trunk = tuple(int(x) for x in args.trunk.split(","))
    model = MultiHeadMLP(feat_dim, trunk=trunk, n_heads=n_heads,
                         dropout=args.dropout,
                         use_layernorm=args.use_layernorm,
                         head_hidden=args.head_hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: in_dim={feat_dim} trunk={trunk} n_heads={n_heads} "
          f"head_hidden={args.head_hidden} dropout={args.dropout} params={n_params:,}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    fm_t = torch.from_numpy(feat_mean).to(device)
    fs_t = torch.from_numpy(feat_std).to(device)
    hw_t = torch.from_numpy(head_weights).to(device)  # (n_heads,)

    best_main_val_mse = float("inf")
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
        perm = torch.randperm(n_train)
        running_loss = 0.0
        running_mse_main = 0.0
        running_n = 0

        for s in range(0, n_train, args.batch_size):
            idx = perm[s:s+args.batch_size]
            xb_raw = X_tr_raw[idx].to(device, non_blocking=True)
            yb = y_tr_per_t[idx].to(device, non_blocking=True)  # (B, n_heads), scaled
            wb = sw_tr_t[idx].to(device, non_blocking=True)

            if args.aug_mode == "online":
                scales = torch.empty_like(xb_raw).uniform_(args.aug_lo, args.aug_hi)
                xb_raw = xb_raw * scales

            xb = (xb_raw - fm_t) / fs_t
            xb = torch.clamp(xb, -CLIP, CLIP)

            pred = model(xb)  # (B, n_heads), scaled space
            mse_per_head = ((pred - yb) ** 2 * wb.unsqueeze(1)).sum(0) / wb.sum().clamp_min(1.0)
            # Weighted sum of per-head MSE
            loss = (mse_per_head * hw_t).sum() / hw_t.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            running_mse_main += float(mse_per_head[main_idx].item()) * bs
            running_n += bs

        scheduler.step()
        train_loss_avg = running_loss / running_n
        train_mse_main = running_mse_main / running_n

        # Val: predict all heads, compute per-head MSE in unscaled space
        va_pred_s_all = predict_chunked(model, X_va_std, batch=16384, device=device)
        va_pred_all = va_pred_s_all / target_scales[None, :]
        per_head_val_mse = []
        per_head_val_corr = []
        for hi in range(n_heads):
            v_mse = float(((va_pred_all[:, hi] - y_regr_va_per[:, hi]) ** 2).mean())
            if va_pred_all[:, hi].std() > 0:
                v_corr = float(np.corrcoef(va_pred_all[:, hi], y_regr_va_per[:, hi])[0, 1])
            else:
                v_corr = 0.0
            per_head_val_mse.append(v_mse)
            per_head_val_corr.append(v_corr)
        main_val_mse = per_head_val_mse[main_idx]
        main_val_corr = per_head_val_corr[main_idx]

        ep_time = time.time() - t_ep
        cur_lr = optimizer.param_groups[0]["lr"]
        log = {
            "epoch": epoch, "train_loss": train_loss_avg,
            "train_mse_main": train_mse_main,
            "val_mse_main": main_val_mse,
            "val_corr_main": main_val_corr,
            "per_head_val_mse": per_head_val_mse,
            "per_head_val_corr": per_head_val_corr,
            "lr": cur_lr, "time": ep_time,
        }
        epoch_logs.append(log)
        head_str = " ".join(f"h{horizons[hi]}={per_head_val_corr[hi]:.4f}"
                            for hi in range(n_heads))
        print(f"  ep {epoch:3d}  loss={train_loss_avg:.6e}  main_mse={main_val_mse:.6e} "
              f"main_corr={main_val_corr:.4f}  [{head_str}]  lr={cur_lr:.2e}  ({ep_time:.1f}s)",
              flush=True)
        if use_wandb:
            wb_log = {**{k: v for k, v in log.items() if not isinstance(v, list)}, "epoch": epoch}
            for hi, h in enumerate(horizons):
                wb_log[f"val_mse_h{h}"] = per_head_val_mse[hi]
                wb_log[f"val_corr_h{h}"] = per_head_val_corr[hi]
            wandb.log(wb_log)

        if main_val_mse < best_main_val_mse - 1e-10:
            best_main_val_mse = main_val_mse
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
    print(f"\n  train done in {train_time:.1f}s; best_epoch={best_epoch} "
          f"best_main_val_mse={best_main_val_mse:.6e}", flush=True)
    assert best_state is not None, "training never improved"
    model.load_state_dict(best_state)

    # Save model
    model_path = os.path.join(HERE, f"model_T97_seed{seed}_{args.tag}.pt")
    torch.save({
        "state_dict": best_state,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "keep_idx": keep_idx,
        "feat_names": feat_names,
        "trunk": trunk,
        "head_hidden": int(args.head_hidden),
        "horizons": horizons,
        "head_weights": head_weights.tolist(),
        "main_horizon": main_h,
        "main_idx": main_idx,
        "dropout": args.dropout,
        "use_layernorm": args.use_layernorm,
        "in_dim": feat_dim,
        "target_scales": target_scales.tolist(),
        "clip": float(CLIP),
        "best_epoch": best_epoch,
        "best_main_val_mse": best_main_val_mse,
    }, model_path)
    print(f"  saved {model_path}", flush=True)

    # Final predictions
    va_pred_all = predict_chunked(model, X_va_std, batch=16384, device=device) / target_scales[None, :]
    te_pred_all = predict_chunked(model, X_te_std, batch=16384, device=device) / target_scales[None, :]

    # Per-head test metrics (focus on main but log all)
    test_metrics = {}
    for hi, h in enumerate(horizons):
        y_te_h = y_regr_te_per[:, hi]
        p_te_h = te_pred_all[:, hi]
        te_mse = float(((p_te_h - y_te_h) ** 2).mean())
        te_corr = float(np.corrcoef(p_te_h, y_te_h)[0, 1]) if p_te_h.std() > 0 else 0.0
        test_metrics[f"h{h}"] = {"mse": te_mse, "corr": te_corr}
        print(f"  TEST h={h}: mse={te_mse:.7f} corr={te_corr:.4f}", flush=True)

    # EV gate quick sanity at k=1 on main horizon
    te_pred_main = te_pred_all[:, main_idx]
    fee_thr = 2.0 * FEE
    pred_action = np.full(len(te_pred_main), 1, dtype=np.int8)
    pred_action[te_pred_main > fee_thr] = 2
    pred_action[te_pred_main < -fee_thr] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  TEST EV-gate k=1 (h={main_h}, thr={fee_thr:.5f}): cum_pnl={cum_pnl:+.4f} n_active={n_active:,}",
          flush=True)

    # Save predictions parquet (main horizon as primary, plus all heads)
    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te_main.astype(np.int8),
        "true_dmid_norm": y_regr_te_per[:, main_idx].astype(np.float32),
        "pred_dmid_norm": te_pred_main.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    # Add all heads under columns "pred_h{h}"
    for hi, h in enumerate(horizons):
        te_df[f"pred_h{h}"] = te_pred_all[:, hi].astype(np.float32)
    pred_path = os.path.join(HERE, f"pred_T97_seed{seed}_{args.tag}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    summary = {
        "task": f"T97 multi-head NN seed={seed} horizons={horizons}",
        "seed": seed,
        "horizons": horizons,
        "head_weights": head_weights.tolist(),
        "main_horizon": main_h,
        "trunk": list(trunk),
        "head_hidden": int(args.head_hidden),
        "split_info": {"strategy": "V4", "train_dates": "0-75", "val_dates": "76-79"},
        "n_features": feat_dim,
        "drop_names": list(DROP_NAMES),
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs_run": (epoch_logs[-1]["epoch"] + 1) if epoch_logs else 0,
        "best_epoch": best_epoch,
        "best_main_val_mse": best_main_val_mse,
        "n_params": int(n_params),
        "train_time_sec": float(train_time),
        "test_per_head": test_metrics,
        "test_ev_gate_k1": {"cum_pnl": cum_pnl, "n_active": n_active,
                            "horizon": main_h, "fee_thr": fee_thr},
        "epoch_logs": epoch_logs,
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T97_seed{seed}_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wb_final = {
            "best_epoch": best_epoch,
            "best_main_val_mse": best_main_val_mse,
            "train_time_sec": float(train_time),
            "test_ev_gate_k1_cum_pnl": cum_pnl,
            "test_ev_gate_k1_n_active": n_active,
        }
        for hi, h in enumerate(horizons):
            wb_final[f"test_mse_h{h}"] = test_metrics[f"h{h}"]["mse"]
            wb_final[f"test_corr_h{h}"] = test_metrics[f"h{h}"]["corr"]
        wandb.log(wb_final)
        wandb.finish()
    progress("done", seed=seed, horizons=horizons,
             best_epoch=best_epoch, val_mse=best_main_val_mse,
             test_corr_main=test_metrics[f"h{main_h}"]["corr"],
             test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
