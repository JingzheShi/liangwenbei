"""T32: Train DeepLOB single-head h_60 per LOSO fold with aug_a, emit OOF.

Per fold (held in {0..4}):
  train: (sym != held) ∩ (date 0..79)
  val:   (sym != held) ∩ (date 80..95)   -- early stop on val_h60_cum_pnl
  test:  (sym == held) ∩ (date 96..119)  -- held-out OOF

Strong regularization:
  - GroupNorm (model.py)
  - Dropout 0.4 conv / 0.5 fc
  - AdamW weight_decay=5e-3
  - Label smoothing ε=0.1
  - Cosine LR with linear warmup (100 steps)
  - Aug_a: per-feature random scale [0.8, 1.2] applied to z-scored input
  - Early stop on val h_60 cum_pnl (patience=1)
  - SWA: average top-3 best states by val h_60 cum_pnl

Outputs (in this directory):
  loso_pred_nn_h60_held{K}.parquet  -- OOF probs for h=60
  model_h60_held{K}.pt              -- SWA-averaged model state
  fold{K}_history.json              -- per-epoch metrics
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from src.data.dataset import LOBDataset  # noqa: E402
from src.data.split import get_split  # noqa: E402
from src.eval.pnl import HORIZON_NAMES, compute_pnl, sanitize_for_json  # noqa: E402
from model import DeepLOB_H60  # noqa: E402


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"T32-{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def progress_dump(held: int, status: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": status,
        "held": held,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(sanitize_for_json(payload), f, indent=2)


def normalize_dataset(ds: LOBDataset, mean: np.ndarray, std: np.ndarray, log1p_idx: np.ndarray, logger=None) -> int:
    """In-place: log1p amount_delta, then z-score, then replace NaN/Inf with 0."""
    bad_total = 0
    for s in range(len(ds._sessions_X)):
        X = ds._sessions_X[s]
        if X is None:
            continue
        X[:, log1p_idx] = np.log1p(X[:, log1p_idx])
        np.subtract(X, mean.astype(X.dtype), out=X)
        np.divide(X, std.astype(X.dtype), out=X)
        bad = ~np.isfinite(X)
        if bad.any():
            bad_total += int(bad.sum())
            X[bad] = 0.0
    if logger is not None:
        logger.info(f"  normalize: replaced {bad_total} non-finite values across {len(ds._sessions_X)} sessions")
    return bad_total


def compute_class_weights_h60(ds: LOBDataset, h60_idx: int, n_classes: int = 3, logger=None) -> torch.Tensor:
    """h_60 class weights = n_total / (n_class * count_c)."""
    Y_all = np.concatenate([Y for Y in ds._sessions_Y if Y is not None], axis=0)
    col = Y_all[:, h60_idx]
    counts = np.bincount(col, minlength=n_classes).astype(np.float64)
    n_total = counts.sum()
    cs = np.maximum(counts, 1.0)
    w = n_total / (n_classes * cs)
    if logger is not None:
        dist = (counts / max(n_total, 1)).tolist()
        logger.info(f"  class_weights h60: dist={[f'{d:.3f}' for d in dist]} weights={[f'{x:.3f}' for x in w.tolist()]}")
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def collect_probs_and_labels(model: nn.Module, loader: DataLoader, device, h60_idx: int):
    """Return (probs[N,3], preds[N], labels_h60[N], mp_t[N], mp_th[N])."""
    model.eval()
    all_probs = []
    all_preds = []
    all_label = []
    all_mp_t = []
    all_mp_th = []
    for batch in loader:
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        y = batch["y"].numpy().astype(np.int64)  # (B, n_h)
        mp_t = batch["midprice_t"].numpy()
        mp_tn = batch["midprice_tn"].numpy()  # (B, n_h)
        x = x.unsqueeze(1)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(p)
        all_preds.append(p.argmax(axis=1))
        all_label.append(y[:, h60_idx])
        all_mp_t.append(mp_t)
        all_mp_th.append(mp_tn[:, h60_idx])
    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_label, axis=0),
        np.concatenate(all_mp_t, axis=0),
        np.concatenate(all_mp_th, axis=0),
    )


@torch.no_grad()
def collect_meta(loader: DataLoader):
    """Walk dataset to collect (sym, date, session, t) per row."""
    syms, dates, sessions, ts = [], [], [], []
    ds = loader.dataset
    n = len(ds)
    for i in range(n):
        sidx, t = ds._idx_to_sess_t(i)
        sym, date, sess = ds.sym_dates[sidx]
        syms.append(int(sym))
        dates.append(int(date))
        sessions.append(str(sess))
        ts.append(int(t))
    return (
        np.asarray(syms, dtype=np.int8),
        np.asarray(dates, dtype=np.int16),
        np.asarray(sessions, dtype=object),
        np.asarray(ts, dtype=np.int16),
    )


def state_dict_average(states: list) -> dict:
    if not states:
        raise ValueError("no states to average")
    keys = list(states[0].keys())
    avg = {}
    for k in keys:
        stacked = torch.stack([s[k].float() for s in states], dim=0)
        avg[k] = stacked.mean(dim=0)
    return avg


def filter_split(keys: list, held: int, exclude_held: bool) -> list:
    return [k for k in keys if (k[0] != held if exclude_held else k[0] == held)]


def make_warmup_cosine_lr_lambda(warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine from 1.0 to 0.0 over remaining steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return fn


def cum_pnl_h60(probs: np.ndarray, labels: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray) -> dict:
    """Compute h_60 cum_pnl with default argmax (for early stop) and report dist."""
    from src.eval.pnl import _per_horizon_metrics
    pred = probs.argmax(axis=1).astype(np.int64)
    m = _per_horizon_metrics(pred, labels.astype(np.int64),
                              mp_t.astype(np.float32), mp_th.astype(np.float32),
                              fee_rate=0.0001)
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
        "pred_distribution": list(m["pred_distribution"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--swa_topk", type=int, default=1,
                    help="Number of best epochs to average. 1 = use single best state (no avg).")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--aug_a_lo", type=float, default=0.8)
    ap.add_argument("--aug_a_hi", type=float, default=1.2)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data"))
    ap.add_argument("--stats", type=str, default=os.path.join(ROOT, "experiments", "T1_nn_baseline", "stats.npz"))
    ap.add_argument("--out_dir", type=str, default=HERE)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="liangwenbei")
    ap.add_argument("--wandb_entity", type=str, default="jingzheshi",
                    help="API key only authorizes jingzheshi entity (T1/T19 noted same)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log_path = os.path.join(args.out_dir, f"fold{args.held}_train.log")
    logger = setup_logger(log_path)
    logger.info(f"args: {vars(args)}")

    progress_dump(args.held, "loading_data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    if device.type == "cuda":
        logger.info(f"gpu: {torch.cuda.get_device_name(0)}")

    # ---- Stats ----
    logger.info(f"Loading stats from {args.stats}")
    stats = np.load(args.stats, allow_pickle=True)
    feature_cols = list(stats["feature_cols"])
    log1p_cols = list(stats["log1p_cols"])
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    log1p_idx = np.array([feature_cols.index(c) for c in log1p_cols], dtype=np.int64)
    logger.info(f"  features: {len(feature_cols)}, log1p_cols: {log1p_cols}")

    # ---- Splits ----
    splits = get_split(mode="local_debug")
    train_keys = filter_split(splits["train"], args.held, exclude_held=True)
    val_keys = filter_split(splits["val"], args.held, exclude_held=True)
    test_keys = filter_split(splits["test"], args.held, exclude_held=False)
    logger.info(
        f"LOSO held={args.held}: train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}"
    )

    # ---- Datasets ----
    t0 = time.time()
    train_ds = LOBDataset(
        sym_dates=train_keys, feature_cols=feature_cols, window=100,
        data_dir=args.data_dir, cache_in_memory=True,
        return_midprices=False, return_meta=False,
    )
    logger.info(f"  train ds: {len(train_ds)} samples in {time.time()-t0:.1f}s")

    t0 = time.time()
    val_ds = LOBDataset(
        sym_dates=val_keys, feature_cols=feature_cols, window=100,
        data_dir=args.data_dir, cache_in_memory=True,
        return_midprices=True, return_meta=False,
    )
    logger.info(f"  val ds: {len(val_ds)} samples in {time.time()-t0:.1f}s")

    t0 = time.time()
    test_ds = LOBDataset(
        sym_dates=test_keys, feature_cols=feature_cols, window=100,
        data_dir=args.data_dir, cache_in_memory=True,
        return_midprices=True, return_meta=False,
    )
    logger.info(f"  test ds: {len(test_ds)} samples in {time.time()-t0:.1f}s")

    # ---- h_60 idx ----
    n_horizons = len(train_ds.label_cols)
    h60_idx = HORIZON_NAMES.index("label_60")
    logger.info(f"  h60_idx={h60_idx} (in {HORIZON_NAMES})")

    # ---- Class weights (BEFORE feature normalization) ----
    class_weights = compute_class_weights_h60(train_ds, h60_idx=h60_idx, logger=logger)

    # ---- Normalize features in-place ----
    logger.info("Normalizing train features...")
    normalize_dataset(train_ds, mean, std, log1p_idx, logger=logger)
    logger.info("Normalizing val features...")
    normalize_dataset(val_ds, mean, std, log1p_idx, logger=logger)
    logger.info("Normalizing test features...")
    normalize_dataset(test_ds, mean, std, log1p_idx, logger=logger)

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ---- Model ----
    model = DeepLOB_H60(
        seq_len=100, num_features=len(feature_cols),
        conv_dropout=0.4, fc_dropout=0.5,
    ).to(device)
    with torch.no_grad():
        dummy = torch.zeros(2, 1, 100, len(feature_cols), device=device)
        model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model: DeepLOB_H60, n_params={n_params}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = max(1, args.epochs * (len(train_ds) // args.batch_size))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_warmup_cosine_lr_lambda(args.warmup_steps, n_steps),
    )
    logger.info(f"scheduler: warmup({args.warmup_steps}) + cosine over {n_steps} steps; epochs={args.epochs}")

    # ---- WandB ----
    use_wandb = not args.no_wandb
    wb = None
    if use_wandb:
        try:
            import wandb
            wb = wandb
            wb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"T32-NN-h60-fold{args.held}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": "DeepLOB_H60",
                    "arch": "DeepLOB-aug_a single-head h_60",
                    "n_features": len(feature_cols),
                    "window": 100, "head": "h_60_only",
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "aug_a_range": [args.aug_a_lo, args.aug_a_hi],
                    "warmup_steps": args.warmup_steps,
                    "swa_topk": args.swa_topk,
                    "scheduler": "warmup+cosine",
                    "held_out_sym": args.held,
                    "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
                    "seed": args.seed, "tag": "T32_NN_h60",
                },
            )
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            wb = None

    # ---- Training loop ----
    history = []
    swa_pool: list[tuple[float, dict]] = []
    best_val_pnl = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    progress_dump(args.held, "training", n_train=len(train_ds), epochs=args.epochs)

    nF = len(feature_cols)
    aug_lo = float(args.aug_a_lo)
    aug_hi = float(args.aug_a_hi)
    aug_range = aug_hi - aug_lo

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        running_loss = 0.0
        running_n = 0
        for it, batch in enumerate(train_loader):
            x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)  # (B, T, F)
            y = batch["y"].to(device, dtype=torch.long, non_blocking=True)[:, h60_idx]
            # aug_a: per-sample per-feature random scale [lo, hi] in z-score space
            scale = torch.rand((x.size(0), 1, nF), device=device, dtype=x.dtype) * aug_range + aug_lo
            x = x * scale
            x = x.unsqueeze(1)  # (B, 1, T, F)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            bsz = x.size(0)
            running_loss += float(loss.item()) * bsz
            running_n += bsz
            if (it + 1) % 200 == 0:
                avg = running_loss / max(running_n, 1)
                logger.info(f"  epoch {epoch} step {it+1}/{len(train_loader)}: loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        train_loss = running_loss / max(running_n, 1)
        logger.info(f"epoch {epoch} train: loss={train_loss:.4f} time={time.time()-t_ep:.1f}s")

        # ---- Validation ----
        t_val = time.time()
        v_probs, v_preds, v_label, v_mp_t, v_mp_th = collect_probs_and_labels(
            model, val_loader, device, h60_idx,
        )
        val_metrics = cum_pnl_h60(v_probs, v_label, v_mp_t, v_mp_th)
        val_h60_pnl = val_metrics["cum_pnl"]
        # Also compute for thresholded T=0.45,d=0.10 as proxy
        from sweep_utils import thresholded_metrics
        val_thr = thresholded_metrics(v_probs, v_label, v_mp_t, v_mp_th, T=0.45, delta=0.10)
        val_loss_proxy = float(-np.mean(np.log(np.maximum(v_probs[np.arange(len(v_label)), v_label], 1e-12))))
        logger.info(
            f"epoch {epoch} val: loss_proxy={val_loss_proxy:.4f} h60_argmax_pnl={val_h60_pnl:.4f} "
            f"h60_thr(T=0.45,d=0.10)_pnl={val_thr['cum_pnl']:.4f} "
            f"acc={val_metrics['accuracy']:.3f} pred_dist={val_metrics['pred_distribution']} "
            f"time={time.time()-t_val:.1f}s"
        )

        if wb is not None:
            try:
                rec = {
                    "epoch": epoch, "fold": args.held,
                    "train/loss": train_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "val/loss_proxy": val_loss_proxy,
                    "val/h60_cum_pnl_argmax": val_h60_pnl,
                    "val/h60_cum_pnl_T045d010": val_thr["cum_pnl"],
                    "val/h60_acc": val_metrics["accuracy"],
                    "val/h60_n_active_T045d010": val_thr["n_active"],
                }
                wb.log(rec, step=epoch)
            except Exception as e:
                logger.warning(f"wandb log failed: {e}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_h60_cum_pnl_argmax": val_h60_pnl,
            "val_h60_cum_pnl_T045d010": val_thr["cum_pnl"],
            "val_h60_acc": val_metrics["accuracy"],
            "val_loss_proxy": val_loss_proxy,
        })

        # ---- SWA pool: keep top-K by val_h60_pnl_T045d010 (proxy for actual deployment) ----
        cur_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        score_for_swa = val_thr["cum_pnl"]
        swa_pool.append((score_for_swa, cur_state_cpu))
        swa_pool.sort(key=lambda kv: kv[0], reverse=True)
        if len(swa_pool) > args.swa_topk:
            swa_pool = swa_pool[: args.swa_topk]

        # Early stop on val_h60 thresholded pnl
        improved = score_for_swa > best_val_pnl + 1e-8
        if improved:
            best_val_pnl = score_for_swa
            best_epoch = epoch
            epochs_no_improve = 0
            logger.info(f"  -> NEW BEST val_h60_thr_pnl={best_val_pnl:.4f} at epoch {epoch}")
        else:
            epochs_no_improve += 1
            logger.info(f"  no improvement ({epochs_no_improve}/{args.patience})")
            if epochs_no_improve >= args.patience:
                logger.info(f"early stop at epoch {epoch} (patience={args.patience})")
                break

    # ---- SWA: average top-K best states ----
    swa_states = [s for _, s in swa_pool]
    swa_pnls = [v for v, _ in swa_pool]
    avg_state = state_dict_average(swa_states)
    logger.info(f"SWA averaging {len(swa_states)} states, val_h60_thr_pnls={swa_pnls}")
    model.load_state_dict(avg_state)
    model.eval()

    # ---- OOF on test (held-out sym) ----
    progress_dump(args.held, "predicting_test", best_epoch=best_epoch, swa_pnls=swa_pnls)
    t_oof = time.time()
    t_probs, t_preds, t_label, t_mp_t, t_mp_th = collect_probs_and_labels(
        model, test_loader, device, h60_idx,
    )
    v2_probs, v2_preds, v2_label, v2_mp_t, v2_mp_th = collect_probs_and_labels(
        model, val_loader, device, h60_idx,
    )
    val_swa_argmax = cum_pnl_h60(v2_probs, v2_label, v2_mp_t, v2_mp_th)
    test_argmax = cum_pnl_h60(t_probs, t_label, t_mp_t, t_mp_th)
    from sweep_utils import thresholded_metrics
    val_swa_thr = thresholded_metrics(v2_probs, v2_label, v2_mp_t, v2_mp_th, T=0.45, delta=0.10)
    test_thr = thresholded_metrics(t_probs, t_label, t_mp_t, t_mp_th, T=0.45, delta=0.10)
    logger.info(
        f"SWA val: argmax={val_swa_argmax['cum_pnl']:.4f} thr(0.45,0.10)={val_swa_thr['cum_pnl']:.4f} "
        f"| SWA test: argmax={test_argmax['cum_pnl']:.4f} thr(0.45,0.10)={test_thr['cum_pnl']:.4f} "
        f"time={time.time()-t_oof:.1f}s"
    )
    logger.info(
        f"  test argmax: acc={test_argmax['accuracy']:.3f} "
        f"pred_dist={test_argmax['pred_distribution']} n_active={test_thr['n_active']}"
    )

    # ---- Build OOF parquet (h_60) ----
    syms, dates, sessions, ts = collect_meta(test_loader)
    out_df = pd.DataFrame({
        "sym": syms, "date": dates, "session": sessions, "t": ts,
        "true_label": t_label.astype(np.int8),
        "pred_label": t_probs.argmax(axis=1).astype(np.int8),
        "prob_0": t_probs[:, 0].astype(np.float32),
        "prob_1": t_probs[:, 1].astype(np.float32),
        "prob_2": t_probs[:, 2].astype(np.float32),
        "midprice_t": t_mp_t.astype(np.float32),
        "midprice_th": t_mp_th.astype(np.float32),
    })
    out_pq = os.path.join(args.out_dir, f"loso_pred_nn_h60_held{args.held}.parquet")
    out_df.to_parquet(out_pq, index=False)
    logger.info(f"Wrote OOF -> {out_pq} ({len(out_df)} rows)")

    # ---- Save SWA model + history ----
    ckpt = {
        "model_state": avg_state,
        "swa_topk_pnls": [float(v) for v in swa_pnls],
        "best_epoch": best_epoch,
        "best_val_h60_cum_pnl": float(best_val_pnl),
        "test_h60_argmax_cum_pnl": float(test_argmax["cum_pnl"]),
        "test_h60_thr_T045d010_cum_pnl": float(test_thr["cum_pnl"]),
        "tag": "T32_NN_h60",
        "meta": {
            "label_columns": list(train_ds.label_cols),
            "feature_columns": list(feature_cols),
            "log1p_cols": list(log1p_cols),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "model_type": "deeplob_h60",
            "T": 100,
            "held_out_sym": args.held,
            "stats_source": args.stats,
        },
    }
    pt_path = os.path.join(args.out_dir, f"model_h60_held{args.held}.pt")
    torch.save(ckpt, pt_path)
    logger.info(f"Saved SWA model -> {pt_path}")

    hist_path = os.path.join(args.out_dir, f"fold{args.held}_history.json")
    with open(hist_path, "w") as f:
        json.dump(sanitize_for_json({
            "args": vars(args), "history": history,
            "best_epoch": best_epoch,
            "best_val_h60_thr_pnl": best_val_pnl,
            "swa_topk_pnls": swa_pnls,
            "val_swa_argmax_pnl": val_swa_argmax["cum_pnl"],
            "val_swa_thr_pnl": val_swa_thr["cum_pnl"],
            "test_argmax_pnl": test_argmax["cum_pnl"],
            "test_thr_pnl": test_thr["cum_pnl"],
            "test_argmax_dist": test_argmax["pred_distribution"],
        }), f, indent=2)
    logger.info(f"Saved history -> {hist_path}")

    if wb is not None:
        try:
            wb.summary["best_epoch"] = best_epoch
            wb.summary["best_val_h60_thr_pnl"] = best_val_pnl
            wb.summary["swa_test_argmax_pnl"] = test_argmax["cum_pnl"]
            wb.summary["swa_test_thr_pnl"] = test_thr["cum_pnl"]
            wb.summary["swa_n_states"] = len(swa_states)
            wb.finish()
        except Exception:
            pass

    progress_dump(
        args.held, "fold_done",
        best_epoch=best_epoch,
        best_val_h60_thr_pnl=float(best_val_pnl),
        swa_test_argmax_pnl=float(test_argmax["cum_pnl"]),
        swa_test_thr_pnl=float(test_thr["cum_pnl"]),
    )
    logger.info("Fold training done.")


if __name__ == "__main__":
    main()
