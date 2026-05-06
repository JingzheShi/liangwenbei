"""Train DeepLOB 5-head multi-task baseline on local train split.

Outputs:
  - best_model.pt (best by val cum_pnl on label_60)
  - train.log (text log)
  - WandB run (project=liangwenbei)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Add examples/mmpc_demo to import path for DeepLOB
EXAMPLE_DIR = os.path.join(ROOT, "examples", "mmpc_demo")
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from src.data.dataset import LOBDataset, get_default_feature_cols  # noqa: E402
from src.data.split import get_split  # noqa: E402
from src.eval.pnl import HORIZON_NAMES, HORIZONS, compute_pnl, sanitize_for_json  # noqa: E402
from model import DeepLOB  # noqa: E402  (from examples/mmpc_demo/model.py)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("T1")
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


def normalize_dataset(ds: LOBDataset, mean: np.ndarray, std: np.ndarray, log1p_idx: np.ndarray, logger=None) -> None:
    """In-place: log1p amount_delta, then z-score, then replace NaN/Inf with 0.

    Operates on the underlying _sessions_X numpy arrays to avoid recomputation.
    """
    n_sessions = len(ds._sessions_X)
    nan_count = 0
    for s in range(n_sessions):
        X = ds._sessions_X[s]
        if X is None:
            continue
        # log1p selected columns
        X[:, log1p_idx] = np.log1p(X[:, log1p_idx])
        # z-score
        np.subtract(X, mean.astype(X.dtype), out=X)
        np.divide(X, std.astype(X.dtype), out=X)
        # replace NaN/Inf with 0 (encodes missing as feature mean post-norm)
        bad = ~np.isfinite(X)
        if bad.any():
            nan_count += int(bad.sum())
            X[bad] = 0.0
    if logger is not None:
        logger.info(f"[normalize] replaced {nan_count} non-finite values across {n_sessions} sessions")


def compute_class_weights(ds: LOBDataset, n_classes: int = 3, logger=None) -> list[torch.Tensor]:
    """Per-horizon class weights using formula n_total / (3 * n_class_i)."""
    label_arrs = []
    for s in range(len(ds._sessions_Y)):
        Y = ds._sessions_Y[s]
        if Y is None:
            continue
        label_arrs.append(Y)
    Y_all = np.concatenate(label_arrs, axis=0)  # (T_total, n_horizons)
    n_horizons = Y_all.shape[1]

    weights_list = []
    if logger is not None:
        logger.info(f"[class_weights] total label rows: {Y_all.shape[0]}")
    for h in range(n_horizons):
        col = Y_all[:, h]
        counts = np.bincount(col, minlength=n_classes).astype(np.float64)
        n_total = counts.sum()
        # avoid div by zero
        counts_safe = np.maximum(counts, 1.0)
        w = n_total / (n_classes * counts_safe)
        weights_list.append(torch.tensor(w, dtype=torch.float32))
        if logger is not None:
            dist = (counts / n_total).tolist()
            logger.info(f"[class_weights] {HORIZON_NAMES[h]}: dist={[f'{d:.3f}' for d in dist]}, weights={[f'{x:.3f}' for x in w.tolist()]}")
    return weights_list


@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader, device, criteria: list[nn.Module], n_horizons: int) -> dict:
    """Run forward over val set, compute mean loss + PnL across all val samples."""
    model.eval()
    losses_per_head = [0.0] * n_horizons
    n_total = 0
    all_pred = []
    all_label = []
    all_mp_t = []
    all_mp_tn = []
    for batch in loader:
        x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)  # (B, T, F)
        y = batch["y"].to(device, dtype=torch.long, non_blocking=True)     # (B, n_h)
        mp_t = batch["midprice_t"].numpy()  # (B,)
        mp_tn = batch["midprice_tn"].numpy()  # (B, n_h)
        x = x.unsqueeze(1)
        logits = model(x)  # tuple of (B, 3) per head
        bsz = x.size(0)
        n_total += bsz
        preds_b = []
        for h in range(n_horizons):
            l = criteria[h](logits[h], y[:, h])
            losses_per_head[h] += float(l.item()) * bsz
            preds_b.append(logits[h].argmax(1).cpu().numpy())
        all_pred.append(np.stack(preds_b, axis=1))  # (B, n_h)
        all_label.append(y.cpu().numpy())
        all_mp_t.append(mp_t)
        all_mp_tn.append(mp_tn)

    pred_cat = np.concatenate(all_pred, axis=0)
    label_cat = np.concatenate(all_label, axis=0)
    mp_t_cat = np.concatenate(all_mp_t, axis=0)
    mp_tn_cat = np.concatenate(all_mp_tn, axis=0)
    mean_losses = [l / max(n_total, 1) for l in losses_per_head]

    pnl_out = compute_pnl(pred_cat, label_cat, mp_t_cat.astype(np.float64), mp_tn_cat.astype(np.float64))
    pnl_out["loss_per_head"] = mean_losses
    pnl_out["loss_mean"] = float(np.mean(mean_losses))
    pnl_out["n_samples"] = int(n_total)
    return pnl_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_epochs_extended", type=int, default=10, help="If val keeps improving, allow up to this many epochs")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data"))
    ap.add_argument("--stats", type=str, default=os.path.join(HERE, "stats.npz"))
    ap.add_argument("--out_dir", type=str, default=HERE)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_entity", type=str, default="jingzheshi", help="wandb entity (cjxh21-Tsinghua University not accessible by current API key)")
    ap.add_argument("--wandb_project", type=str, default="liangwenbei")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    log_path = os.path.join(args.out_dir, "train.log")
    logger = setup_logger(log_path)
    logger.info(f"args: {vars(args)}")

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
    logger.info(f"features: {len(feature_cols)}, log1p_cols: {log1p_cols}")
    log1p_idx = np.array([feature_cols.index(c) for c in log1p_cols], dtype=np.int64)

    # ---- Datasets ----
    splits = get_split(mode="local_debug")
    logger.info(f"splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    t0 = time.time()
    logger.info("Loading train dataset (in-memory)")
    train_ds = LOBDataset(
        sym_dates=splits["train"],
        feature_cols=feature_cols,
        window=100,
        data_dir=args.data_dir,
        cache_in_memory=True,
        return_midprices=False,
        return_meta=False,
    )
    logger.info(f"  train: {len(train_ds)} samples, {time.time()-t0:.1f}s")

    t0 = time.time()
    logger.info("Loading val dataset (in-memory, with midprices)")
    val_ds = LOBDataset(
        sym_dates=splits["val"],
        feature_cols=feature_cols,
        window=100,
        data_dir=args.data_dir,
        cache_in_memory=True,
        return_midprices=True,
        return_meta=False,
    )
    logger.info(f"  val: {len(val_ds)} samples, {time.time()-t0:.1f}s")

    # ---- Class weights (compute BEFORE normalization on labels — labels untouched anyway) ----
    n_horizons = len(train_ds.label_cols)
    class_weights = compute_class_weights(train_ds, n_classes=3, logger=logger)

    # ---- Normalize features in-place ----
    logger.info("Normalizing train features...")
    normalize_dataset(train_ds, mean, std, log1p_idx, logger=logger)
    logger.info("Normalizing val features...")
    normalize_dataset(val_ds, mean, std, log1p_idx, logger=logger)

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

    # ---- Model ----
    num_classes_per_head = [3] * n_horizons
    model = DeepLOB(num_classes_per_head=num_classes_per_head, seq_len=100, num_features=len(feature_cols)).to(device)

    # Force LazyLinear to materialize by running one dummy forward
    with torch.no_grad():
        dummy = torch.zeros(2, 1, 100, len(feature_cols), device=device)
        model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model: DeepLOB, n_params={n_params}")

    # ---- Loss + Optimizer + Sched ----
    criteria = [nn.CrossEntropyLoss(weight=w.to(device)) for w in class_weights]
    val_criteria = [nn.CrossEntropyLoss(weight=w.to(device)) for w in class_weights]
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_steps = max(1, args.max_epochs_extended * (len(train_ds) // args.batch_size))
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps)
    logger.info(f"scheduler: cosine over {n_steps} steps")

    # ---- WandB ----
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"T1-nn-baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": "DeepLOB",
                    "n_features": len(feature_cols),
                    "window": 100,
                    "n_horizons": n_horizons,
                    "epochs": args.epochs,
                    "max_epochs_extended": args.max_epochs_extended,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "optimizer": "AdamW",
                    "scheduler": "cosine",
                    "class_weights_strategy": "balanced (n_total/(3*n_class))",
                    "log1p_cols": log1p_cols,
                    "n_train_samples": len(train_ds),
                    "n_val_samples": len(val_ds),
                    "seed": args.seed,
                    "tag": "T1_nn_baseline",
                },
            )
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            use_wandb = False

    # ---- Training loop ----
    best_val_pnl = -float("inf")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    history = []

    max_epoch = args.max_epochs_extended  # we run up to this many but apply early stopping

    global_step = 0
    for epoch in range(1, max_epoch + 1):
        model.train()
        t_ep = time.time()
        running_loss = [0.0] * n_horizons
        running_n = 0
        for it, batch in enumerate(train_loader):
            x = batch["x"].to(device, dtype=torch.float32, non_blocking=True)
            y = batch["y"].to(device, dtype=torch.long, non_blocking=True)
            x = x.unsqueeze(1)  # (B, 1, T, F)
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
        logger.info(f"epoch {epoch} train: total_loss={train_loss_total:.4f} per_head={[f'{l:.4f}' for l in train_loss_per_head]} time={time.time()-t_ep:.1f}s")

        # ---- Validation ----
        t_val = time.time()
        val_out = run_validation(model, val_loader, device, val_criteria, n_horizons)
        val_pnl_per_h = {hn: val_out[hn]["cum_pnl"] for hn in HORIZON_NAMES}
        val_label60_pnl = val_out["label_60"]["cum_pnl"]
        logger.info(f"epoch {epoch} val: loss={val_out['loss_mean']:.4f} cum_pnl_label60={val_label60_pnl:.4f} best_score={val_out['best_score']:.4f} ({val_out['best_horizon']}) time={time.time()-t_val:.1f}s")
        for hn in HORIZON_NAMES:
            m = val_out[hn]
            logger.info(f"  {hn}: acc={m['accuracy']:.3f} prec_macro={m['precision_macro']!r} cum_pnl={m['cum_pnl']:.4f} pred_dist={m['pred_distribution']}")

        # WandB log
        if use_wandb:
            try:
                import wandb
                rec = {
                    "epoch": epoch,
                    "train/total_loss": train_loss_total,
                    "lr": scheduler.get_last_lr()[0],
                    "val/loss_mean": val_out["loss_mean"],
                    "val/best_cum_pnl": val_out["best_score"],
                }
                for h, hn in enumerate(HORIZON_NAMES):
                    rec[f"train/loss_{hn}"] = train_loss_per_head[h]
                    rec[f"val/loss_{hn}"] = val_out["loss_per_head"][h]
                    m = val_out[hn]
                    rec[f"val/cum_pnl_{hn}"] = m["cum_pnl"]
                    rec[f"val/single_pnl_{hn}"] = m["single_pnl"]
                    rec[f"val/accuracy_{hn}"] = m["accuracy"]
                    p_macro = m["precision_macro"]
                    if p_macro is not None and not (isinstance(p_macro, float) and np.isnan(p_macro)):
                        rec[f"val/precision_macro_{hn}"] = p_macro
                    f0_5 = m["f0_5_macro"]
                    if f0_5 is not None and not (isinstance(f0_5, float) and np.isnan(f0_5)):
                        rec[f"val/f0_5_macro_{hn}"] = f0_5
                    rec[f"val/pred_dist_0_{hn}"] = m["pred_distribution"][0]
                    rec[f"val/pred_dist_1_{hn}"] = m["pred_distribution"][1]
                    rec[f"val/pred_dist_2_{hn}"] = m["pred_distribution"][2]
                wandb.log(rec, step=epoch)
            except Exception as e:
                logger.warning(f"wandb log failed: {e}")

        history.append({
            "epoch": epoch,
            "train_loss_per_head": train_loss_per_head,
            "train_loss_total": train_loss_total,
            "val": sanitize_for_json(val_out),
        })

        # ---- Best model (by val cum_pnl on label_60) ----
        improved = val_label60_pnl > best_val_pnl + 1e-8
        if improved:
            best_val_pnl = val_label60_pnl
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"  -> NEW BEST val cum_pnl_label60={best_val_pnl:.4f} at epoch {epoch}")
        else:
            epochs_no_improve += 1
            logger.info(f"  no improvement ({epochs_no_improve}/{args.patience})")

        # ---- Early stop ----
        if epoch >= args.epochs:
            if epochs_no_improve >= args.patience:
                logger.info(f"early stop at epoch {epoch} (no improvement for {epochs_no_improve} epochs)")
                break

    # ---- Save best ----
    ckpt = {
        "model_state": best_state if best_state is not None else model.state_dict(),
        "epoch": best_epoch,
        "val_loss": float(np.nan),
        "tag": "T1_nn_baseline",
        "meta": {
            "label_columns": list(train_ds.label_cols),
            "num_classes_per_head": num_classes_per_head,
            "label_maps": [{0: 0, 1: 1, 2: 2}] * n_horizons,
            "feature_columns": list(feature_cols),
            "log1p_cols": list(log1p_cols),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "model_type": "deeplob",
            "T": 100,
            "trained_on": "local_debug train (date 0..79)",
            "best_val_cum_pnl_label60": best_val_pnl,
            "best_epoch": best_epoch,
        },
    }
    out_pt = os.path.join(args.out_dir, "best_model.pt")
    torch.save(ckpt, out_pt)
    logger.info(f"Saved best model -> {out_pt}")

    # ---- Save history ----
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(sanitize_for_json({
            "args": vars(args),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_cum_pnl_label60": best_val_pnl,
        }), f, indent=2)

    if use_wandb:
        try:
            import wandb
            wandb.summary["best_val_cum_pnl_label60"] = best_val_pnl
            wandb.summary["best_epoch"] = best_epoch
            wandb.finish()
        except Exception:
            pass

    logger.info("Training done.")


if __name__ == "__main__":
    main()
