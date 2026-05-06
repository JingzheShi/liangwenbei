"""T19: Train DeepLOB-Reg (5-head) per LOSO fold and emit OOF predictions for h_10.

Per fold (held in {0..4}):
  train: (sym != held) ∩ (date 0..79)
  val:   (sym != held) ∩ (date 80..95)   -- early stop on val_h10_cum_pnl
  test:  (sym == held) ∩ (date 96..119)  -- held-out OOF predictions

Strong regularization vs. T1:
  - GroupNorm (in model.py)
  - Dropout 0.3 (conv) / 0.5 (fc)
  - AdamW weight_decay=1e-3 (10x stronger than T1)
  - Label smoothing ε=0.1
  - Cosine LR over 5 epochs
  - Early stop on val h_10 cum_pnl (patience=1)
  - SWA: average top-K (=2) best states by val h_10 cum_pnl

Outputs (in this directory):
  loso_pred_nn_h10_held{K}.parquet  -- OOF probs for h=10 (matching T11 schema)
  model_h10_held{K}.pt              -- SWA-averaged model state
  fold{K}_history.json              -- per-epoch metrics

Usage:
  python train_loso.py --held 0
  python train_loso.py --held 0 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from model import DeepLOBReg  # noqa: E402


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"T19-{log_path}")
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


def compute_class_weights(ds: LOBDataset, n_classes: int = 3, logger=None) -> list[torch.Tensor]:
    """Per-horizon class weights = n_total / (n_class * count_c)."""
    Y_all = np.concatenate([Y for Y in ds._sessions_Y if Y is not None], axis=0)
    weights_list = []
    for h in range(Y_all.shape[1]):
        col = Y_all[:, h]
        counts = np.bincount(col, minlength=n_classes).astype(np.float64)
        n_total = counts.sum()
        cs = np.maximum(counts, 1.0)
        w = n_total / (n_classes * cs)
        weights_list.append(torch.tensor(w, dtype=torch.float32))
        if logger is not None:
            dist = (counts / max(n_total, 1)).tolist()
            logger.info(f"  class_weights {HORIZON_NAMES[h]}: dist={[f'{d:.3f}' for d in dist]} weights={[f'{x:.3f}' for x in w.tolist()]}")
    return weights_list


@torch.no_grad()
def collect_probs_and_labels(model: nn.Module, loader: DataLoader, device, n_horizons: int):
    """Run forward and return (probs[N,n_horizons,3], preds[N,n_horizons], labels[N,n_horizons], mp_t[N], mp_tn[N,n_horizons])."""
    model.eval()
    all_probs = []
    all_preds = []
    all_label = []
    all_mp_t = []
    all_mp_tn = []
    for batch in loader:
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
        y = batch["y"].numpy().astype(np.int64)  # (B, n_h)
        mp_t = batch["midprice_t"].numpy()
        mp_tn = batch["midprice_tn"].numpy()
        x = x.unsqueeze(1)
        logits = model(x)  # tuple
        probs_b = []
        preds_b = []
        for h in range(n_horizons):
            p = torch.softmax(logits[h], dim=1).cpu().numpy()  # (B, 3)
            probs_b.append(p)
            preds_b.append(p.argmax(axis=1))
        all_probs.append(np.stack(probs_b, axis=1))  # (B, n_h, 3)
        all_preds.append(np.stack(preds_b, axis=1))
        all_label.append(y)
        all_mp_t.append(mp_t)
        all_mp_tn.append(mp_tn)
    return (
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_label, axis=0),
        np.concatenate(all_mp_t, axis=0),
        np.concatenate(all_mp_tn, axis=0),
    )


@torch.no_grad()
def collect_meta(loader: DataLoader, n_horizons: int):
    """Walk the dataset (no model) to collect (sym, date, session, t) per row."""
    syms, dates, sessions, ts = [], [], [], []
    ds = loader.dataset
    # We deliberately recompute meta per index, since we need exact alignment with predictions.
    # The loader doesn't carry meta when return_meta=False; do it via dataset directly.
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
    """Element-wise mean of a list of state dicts (assumed identical keys / shapes)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=1, help="early stop patience (in epochs)")
    ap.add_argument("--swa_topk", type=int, default=2, help="number of best epochs to average for SWA")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data"))
    ap.add_argument("--stats", type=str, default=os.path.join(ROOT, "experiments", "T1_nn_baseline", "stats.npz"))
    ap.add_argument("--out_dir", type=str, default=HERE)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="liangwenbei")
    ap.add_argument("--wandb_entity", type=str, default="jingzheshi",
                    help="API key only authorizes jingzheshi entity (Tsinghua entity rejects this key, T1 noted same)")
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
        sym_dates=train_keys,
        feature_cols=feature_cols,
        window=100,
        data_dir=args.data_dir,
        cache_in_memory=True,
        return_midprices=False,
        return_meta=False,
    )
    logger.info(f"  train ds: {len(train_ds)} samples in {time.time()-t0:.1f}s")

    t0 = time.time()
    val_ds = LOBDataset(
        sym_dates=val_keys,
        feature_cols=feature_cols,
        window=100,
        data_dir=args.data_dir,
        cache_in_memory=True,
        return_midprices=True,
        return_meta=False,
    )
    logger.info(f"  val ds: {len(val_ds)} samples in {time.time()-t0:.1f}s")

    t0 = time.time()
    test_ds = LOBDataset(
        sym_dates=test_keys,
        feature_cols=feature_cols,
        window=100,
        data_dir=args.data_dir,
        cache_in_memory=True,
        return_midprices=True,
        return_meta=False,
    )
    logger.info(f"  test ds: {len(test_ds)} samples in {time.time()-t0:.1f}s")

    # ---- Class weights (BEFORE feature normalization) ----
    n_horizons = len(train_ds.label_cols)
    class_weights = compute_class_weights(train_ds, n_classes=3, logger=logger)

    # ---- Normalize features in-place ----
    logger.info("Normalizing train features...")
    normalize_dataset(train_ds, mean, std, log1p_idx, logger=logger)
    logger.info("Normalizing val features...")
    normalize_dataset(val_ds, mean, std, log1p_idx, logger=logger)
    logger.info("Normalizing test features...")
    normalize_dataset(test_ds, mean, std, log1p_idx, logger=logger)

    # ---- DataLoaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ---- Model ----
    num_classes_per_head = [3] * n_horizons
    model = DeepLOBReg(
        num_classes_per_head=num_classes_per_head,
        seq_len=100,
        num_features=len(feature_cols),
        conv_dropout=0.3,
        fc_dropout=0.5,
    ).to(device)
    # Materialize LazyLinear
    with torch.no_grad():
        dummy = torch.zeros(2, 1, 100, len(feature_cols), device=device)
        model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model: DeepLOBReg, n_params={n_params}")

    criteria = [
        nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=args.label_smoothing) for w in class_weights
    ]
    val_criteria = [nn.CrossEntropyLoss(weight=w.to(device)) for w in class_weights]  # no smoothing for val loss display

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = max(1, args.epochs * (len(train_ds) // args.batch_size))
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)
    logger.info(f"scheduler: cosine over {n_steps} steps; epochs={args.epochs}")

    h10_idx = HORIZON_NAMES.index("label_10")

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
                name=f"T19-NN-reg-fold{args.held}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": "DeepLOBReg",
                    "arch": "DeepLOB+GroupNorm+Dropout(0.3conv,0.5fc)+SWA",
                    "n_features": len(feature_cols),
                    "window": 100,
                    "n_horizons": n_horizons,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "swa_topk": args.swa_topk,
                    "scheduler": "cosine",
                    "held_out_sym": args.held,
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                    "n_test": len(test_ds),
                    "seed": args.seed,
                    "tag": "T19_NN_regularized",
                },
            )
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            wb = None

    # ---- Training loop ----
    history = []
    # SWA buffer: list of (val_h10_pnl, state_dict_cpu)
    swa_pool: list[tuple[float, dict]] = []
    best_val_pnl = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    progress_dump(args.held, "training", n_train=len(train_ds), epochs=args.epochs)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        running_loss = [0.0] * n_horizons
        running_n = 0
        for it, batch in enumerate(train_loader):
            x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
            y = batch["y"].to(device, dtype=torch.long, non_blocking=True)
            x = x.unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            losses_h = [criteria[h](logits[h], y[:, h]) for h in range(n_horizons)]
            loss = sum(losses_h)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            bsz = x.size(0)
            for h in range(n_horizons):
                running_loss[h] += float(losses_h[h].item()) * bsz
            running_n += bsz
            if (it + 1) % 200 == 0:
                avg = sum(running_loss) / max(running_n, 1)
                logger.info(f"  epoch {epoch} step {it+1}/{len(train_loader)}: loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        train_loss_per_head = [l / max(running_n, 1) for l in running_loss]
        train_loss_total = float(np.sum(train_loss_per_head))
        logger.info(
            f"epoch {epoch} train: total_loss={train_loss_total:.4f} per_head={[f'{l:.4f}' for l in train_loss_per_head]} time={time.time()-t_ep:.1f}s"
        )

        # ---- Validation ----
        t_val = time.time()
        v_probs, v_preds, v_label, v_mp_t, v_mp_tn = collect_probs_and_labels(
            model, val_loader, device, n_horizons
        )
        val_pnl_out = compute_pnl(v_preds, v_label, v_mp_t.astype(np.float64), v_mp_tn.astype(np.float64))
        val_h10_pnl = val_pnl_out["label_10"]["cum_pnl"]
        # Quick mean loss (uncomputed during prob collection — recompute cheap-ish from logits)
        # Skip for speed; compute a rough loss from probs vs labels (no class weights).
        val_loss_proxy = float(
            -np.mean([np.log(np.maximum(v_probs[np.arange(len(v_label)), h, v_label[:, h]], 1e-12)).mean() for h in range(n_horizons)])
        )
        logger.info(
            f"epoch {epoch} val: loss_proxy={val_loss_proxy:.4f} h10_cum_pnl={val_h10_pnl:.4f} best_h={val_pnl_out['best_horizon']}({val_pnl_out['best_score']:.4f}) time={time.time()-t_val:.1f}s"
        )
        for hn in HORIZON_NAMES:
            m = val_pnl_out[hn]
            logger.info(
                f"  {hn}: acc={m['accuracy']:.3f} cum_pnl={m['cum_pnl']:.4f} pred_dist={m['pred_distribution']}"
            )

        # WandB log
        if wb is not None:
            try:
                rec = {
                    "epoch": epoch,
                    "fold": args.held,
                    "train/total_loss": train_loss_total,
                    "lr": scheduler.get_last_lr()[0],
                    "val/loss_proxy": val_loss_proxy,
                    "val/h10_cum_pnl": val_h10_pnl,
                    "val/best_cum_pnl": val_pnl_out["best_score"],
                }
                for h, hn in enumerate(HORIZON_NAMES):
                    rec[f"train/loss_{hn}"] = train_loss_per_head[h]
                    m = val_pnl_out[hn]
                    rec[f"val/cum_pnl_{hn}"] = m["cum_pnl"]
                    rec[f"val/single_pnl_{hn}"] = m["single_pnl"]
                    rec[f"val/accuracy_{hn}"] = m["accuracy"]
                wb.log(rec, step=epoch)
            except Exception as e:
                logger.warning(f"wandb log failed: {e}")

        history.append({
            "epoch": epoch,
            "train_loss_total": train_loss_total,
            "train_loss_per_head": train_loss_per_head,
            "val_h10_cum_pnl": val_h10_pnl,
            "val_loss_proxy": val_loss_proxy,
            "val_pnl_per_h": {hn: val_pnl_out[hn]["cum_pnl"] for hn in HORIZON_NAMES},
            "val_acc_per_h": {hn: val_pnl_out[hn]["accuracy"] for hn in HORIZON_NAMES},
        })

        # ---- SWA pool: keep top-K by val_h10_pnl ----
        cur_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        swa_pool.append((val_h10_pnl, cur_state_cpu))
        swa_pool.sort(key=lambda kv: kv[0], reverse=True)
        if len(swa_pool) > args.swa_topk:
            swa_pool = swa_pool[: args.swa_topk]

        # Early stop on val_h10_cum_pnl
        improved = val_h10_pnl > best_val_pnl + 1e-8
        if improved:
            best_val_pnl = val_h10_pnl
            best_epoch = epoch
            epochs_no_improve = 0
            logger.info(f"  -> NEW BEST val_h10_cum_pnl={best_val_pnl:.4f} at epoch {epoch}")
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
    logger.info(f"SWA averaging {len(swa_states)} states, val_h10_pnls={swa_pnls}")

    # Load avg state into model for final OOF prediction
    model.load_state_dict(avg_state)
    model.eval()

    # ---- OOF on test (held-out sym) ----
    progress_dump(args.held, "predicting_test", best_epoch=best_epoch, swa_pnls=swa_pnls)
    t_oof = time.time()
    t_probs, t_preds, t_label, t_mp_t, t_mp_tn = collect_probs_and_labels(
        model, test_loader, device, n_horizons
    )
    # Sanity: re-evaluate val with SWA model
    v2_probs, v2_preds, v2_label, v2_mp_t, v2_mp_tn = collect_probs_and_labels(
        model, val_loader, device, n_horizons
    )
    val_swa_out = compute_pnl(v2_preds, v2_label, v2_mp_t.astype(np.float64), v2_mp_tn.astype(np.float64))
    test_pnl_out = compute_pnl(t_preds, t_label, t_mp_t.astype(np.float64), t_mp_tn.astype(np.float64))
    logger.info(
        f"SWA val h10_cum_pnl={val_swa_out['label_10']['cum_pnl']:.4f}, "
        f"SWA test h10_cum_pnl={test_pnl_out['label_10']['cum_pnl']:.4f}, time={time.time()-t_oof:.1f}s"
    )
    for hn in HORIZON_NAMES:
        m = test_pnl_out[hn]
        logger.info(
            f"  test {hn}: acc={m['accuracy']:.3f} cum_pnl={m['cum_pnl']:.4f} pred_dist={m['pred_distribution']}"
        )

    # ---- Build OOF parquet (h_10) matching T11 schema ----
    syms, dates, sessions, ts = collect_meta(test_loader, n_horizons)
    p10 = t_probs[:, h10_idx, :].astype(np.float32)  # (N, 3)
    pred10 = t_preds[:, h10_idx].astype(np.int8)
    label10 = t_label[:, h10_idx].astype(np.int8)
    # midprice_th = midprice at t+10 (h_10 horizon). t_mp_tn shape (N, n_horizons)
    h10_horizon_idx = h10_idx  # since horizons are [5,10,20,40,60] -> idx 1
    mp_th_h10 = t_mp_tn[:, h10_horizon_idx].astype(np.float32)

    out_df = pd.DataFrame({
        "sym": syms,
        "date": dates,
        "session": sessions,
        "t": ts,
        "true_label": label10,
        "pred_label": pred10,
        "prob_0": p10[:, 0],
        "prob_1": p10[:, 1],
        "prob_2": p10[:, 2],
        "midprice_t": t_mp_t.astype(np.float32),
        "midprice_th": mp_th_h10,
    })
    out_pq = os.path.join(args.out_dir, f"loso_pred_nn_h10_held{args.held}.parquet")
    out_df.to_parquet(out_pq, index=False)
    logger.info(f"Wrote OOF -> {out_pq} ({len(out_df)} rows)")

    # ---- Save SWA model + history ----
    ckpt = {
        "model_state": avg_state,
        "swa_topk_pnls": [float(v) for v in swa_pnls],
        "best_epoch": best_epoch,
        "best_val_h10_cum_pnl": float(best_val_pnl),
        "test_h10_cum_pnl": float(test_pnl_out["label_10"]["cum_pnl"]),
        "tag": "T19_NN_regularized",
        "meta": {
            "label_columns": list(train_ds.label_cols),
            "num_classes_per_head": num_classes_per_head,
            "feature_columns": list(feature_cols),
            "log1p_cols": list(log1p_cols),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "model_type": "deeplob_reg",
            "T": 100,
            "held_out_sym": args.held,
            "stats_source": args.stats,
        },
    }
    pt_path = os.path.join(args.out_dir, f"model_h10_held{args.held}.pt")
    torch.save(ckpt, pt_path)
    logger.info(f"Saved SWA model -> {pt_path}")

    hist_path = os.path.join(args.out_dir, f"fold{args.held}_history.json")
    with open(hist_path, "w") as f:
        json.dump(sanitize_for_json({
            "args": vars(args),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_h10_cum_pnl": best_val_pnl,
            "swa_topk_pnls": swa_pnls,
            "val_swa_pnl_per_h": {hn: val_swa_out[hn]["cum_pnl"] for hn in HORIZON_NAMES},
            "test_pnl_per_h": {hn: test_pnl_out[hn]["cum_pnl"] for hn in HORIZON_NAMES},
            "test_acc_per_h": {hn: test_pnl_out[hn]["accuracy"] for hn in HORIZON_NAMES},
        }), f, indent=2)
    logger.info(f"Saved history -> {hist_path}")

    if wb is not None:
        try:
            wb.summary["best_epoch"] = best_epoch
            wb.summary["best_val_h10_cum_pnl"] = best_val_pnl
            wb.summary["swa_test_h10_cum_pnl"] = test_pnl_out["label_10"]["cum_pnl"]
            wb.summary["swa_val_h10_cum_pnl"] = val_swa_out["label_10"]["cum_pnl"]
            wb.summary["swa_n_states"] = len(swa_states)
            wb.finish()
        except Exception:
            pass

    progress_dump(
        args.held,
        "fold_done",
        best_epoch=best_epoch,
        best_val_h10_cum_pnl=float(best_val_pnl),
        swa_test_h10_cum_pnl=float(test_pnl_out["label_10"]["cum_pnl"]),
    )
    logger.info("Fold training done.")


if __name__ == "__main__":
    main()
