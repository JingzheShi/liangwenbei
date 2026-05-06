"""T7 final D1 model: train on full train (all syms), val for early stopping.

This produces the model that ships in iter_001f. We use the same hyperparams
as the LOSO folds (lr=0.05, num_leaves=127, ...). Threshold (T=0.45, delta=0.05)
will be applied at inference time inside Predictor.predict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
NUM_CLASS = 3


def load_split(split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"schemeD1_{split}.npz")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-model", default=os.path.join(HERE, "model_final_D1.txt"))
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    print("=== T7 final D1 training (all syms; val for early stopping) ===", flush=True)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    print(f"  train X={train_full['X'].shape}, val X={val_full['X'].shape}", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeD1_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            init_kwargs = dict(
                project="liangwenbei",
                name=f"T7-final-D1-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": "lightgbm",
                    "scheme": "D1",
                    "n_features": len(feat_names),
                    "horizon": "label_60",
                    "purpose": "final model for iter_001f submission",
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T7-final", "scheme-D1", "label_60", "submission"],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception:
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    X_tr = train_full["X"]
    y_tr = train_full["y60"].astype(np.int64)
    X_va = val_full["X"]
    y_va = val_full["y60"].astype(np.int64)
    mp_t_va = val_full["mp_t"]
    mp_t60_va = val_full["mp_t60"]

    sw_tr = class_balanced_weight(y_tr)
    sw_va = class_balanced_weight(y_va)

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "multiclass", "num_class": NUM_CLASS,
        "metric": "multi_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "verbose": -1,
    }

    t_train_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
            lgb.log_evaluation(period=25),
        ],
    )
    train_time = time.time() - t_train_start
    print(f"  trained in {train_time:.1f}s, best_iter={booster.best_iteration}", flush=True)

    booster.save_model(args.out_model, num_iteration=booster.best_iteration)
    print(f"  saved -> {args.out_model} ({os.path.getsize(args.out_model)/1e6:.2f} MB)", flush=True)

    # Quick val evaluation for sanity (raw argmax + thresholded with iter_001f config)
    val_probs_chunks = []
    for s in range(0, len(X_va), 200_000):
        val_probs_chunks.append(booster.predict(X_va[s:s + 200_000]).astype(np.float32))
    val_probs = np.concatenate(val_probs_chunks, axis=0)
    val_pred_raw = val_probs.argmax(1).astype(np.int8)
    m_raw = _per_horizon_metrics(val_pred_raw, y_va, mp_t_va, mp_t60_va, fee_rate=0.0001)

    T_THRESH = 0.45
    DELTA_THRESH = 0.05
    p0, p1, p2 = val_probs[:, 0], val_probs[:, 1], val_probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T_THRESH) & (side_max > p1 + DELTA_THRESH)
    side_pred = np.where(p2 > p0, 2, 0)
    val_pred_th = np.where(take, side_pred, 1).astype(np.int8)
    m_th = _per_horizon_metrics(val_pred_th, y_va, mp_t_va, mp_t60_va, fee_rate=0.0001)

    print(f"\n  val raw argmax    : cum_pnl={m_raw['cum_pnl']:+.4f} acc={m_raw['accuracy']:.4f} "
          f"n_active={m_raw['n_predictions_active']:,}", flush=True)
    print(f"  val thresh ({T_THRESH},{DELTA_THRESH}): cum_pnl={m_th['cum_pnl']:+.4f} "
          f"acc={m_th['accuracy']:.4f} n_active={m_th['n_predictions_active']:,}", flush=True)

    summary = {
        "task": "T7 final D1 model",
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "n_features": len(feat_names),
        "val_raw_cum_pnl": float(m_raw["cum_pnl"]),
        "val_thresh_cum_pnl": float(m_th["cum_pnl"]),
        "thresh_T": T_THRESH,
        "thresh_delta": DELTA_THRESH,
    }
    with open(os.path.join(HERE, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb.summary.update({
            "final/best_iter": int(booster.best_iteration),
            "final/val_raw_cum_pnl": float(m_raw["cum_pnl"]),
            "final/val_thresh_cum_pnl": float(m_th["cum_pnl"]),
        })
        wandb.finish()

    print(f"\nDone in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
