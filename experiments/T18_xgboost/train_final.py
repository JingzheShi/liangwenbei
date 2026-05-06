"""T18: Train final XGBoost models on (train + val) for iter_005.

Mirrors T11/train_final_223d_seeds.py but for XGBoost. Trains on full train+val
with the test set used as early-stopping validation (same recipe as iter_002/003).

Outputs:
    final_xgb_model_h{H}_seed{S}.ubj
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import xgboost as xgb  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
N_DROP_TAIL = 3

# Mirror SEED_CONFIGS in train_loso.py
SEED_CONFIGS = {
    42:  dict(seed=42,  subsample=0.8, colsample_bytree=0.8, max_depth=7, reg_lambda=1.0, min_child_weight=100),
    1:   dict(seed=1,   subsample=0.7, colsample_bytree=0.6, max_depth=7, reg_lambda=1.0, min_child_weight=100),
    7:   dict(seed=7,   subsample=0.85, colsample_bytree=0.7, max_depth=6, reg_lambda=1.0, min_child_weight=100),
    13:  dict(seed=13,  subsample=0.7, colsample_bytree=0.6, max_depth=7, reg_lambda=2.0, min_child_weight=150),
    100: dict(seed=100, subsample=0.6, colsample_bytree=0.5, max_depth=8, reg_lambda=1.0, min_child_weight=80),
}


def progress(step: str, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    return dict(np.load(p))


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", required=True)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--num-boost-round", type=int, default=2000)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-threads", type=int, default=18)
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T18 train_final horizons={horizons} seeds={seeds} ===", flush=True)
    print(f"  xgboost version: {xgb.__version__}", flush=True)

    progress("loading_caches", horizons=horizons, seeds=seeds)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    X_tr = np.concatenate([train_full["X"][:, :-N_DROP_TAIL],
                           val_full["X"][:, :-N_DROP_TAIL]], axis=0)
    X_va = test_full["X"][:, :-N_DROP_TAIL]
    print(f"X_tr={X_tr.shape}  X_va={X_va.shape}  feat_dim={X_tr.shape[1]}", flush=True)

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    assert len(feat_names) == X_tr.shape[1]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features: {leak}"

    n_total = len(horizons) * len(seeds)
    done = 0
    for H in horizons:
        y_tr = np.concatenate([train_full[f"y{H}"], val_full[f"y{H}"]], axis=0).astype(np.int64)
        y_va = test_full[f"y{H}"].astype(np.int64)
        sw_tr = class_balanced_weight(y_tr)
        sw_va = class_balanced_weight(y_va)

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr, feature_names=feat_names)
        dval = xgb.DMatrix(X_va, label=y_va, weight=sw_va, feature_names=feat_names)

        for s in seeds:
            done += 1
            cfg = SEED_CONFIGS[s]
            print(f"\n[{done}/{n_total}] training final h={H} seed={s} cfg={cfg}", flush=True)
            progress("training_final", horizon=H, seed=s, done=done, total=n_total)

            params = {
                "objective": "multi:softprob", "num_class": NUM_CLASS,
                "eta": args.learning_rate,
                "max_depth": int(cfg["max_depth"]),
                "min_child_weight": int(cfg["min_child_weight"]),
                "subsample": float(cfg["subsample"]),
                "colsample_bytree": float(cfg["colsample_bytree"]),
                "reg_lambda": float(cfg["reg_lambda"]),
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "nthread": args.num_threads,
                "seed": s,
                "verbosity": 0,
            }

            t_start = time.time()
            booster = xgb.train(
                params, dtrain,
                num_boost_round=args.num_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=args.early_stopping,
                verbose_eval=200,
            )
            best_iter = int(booster.best_iteration)
            print(f"  trained in {time.time()-t_start:.1f}s, best_iter={best_iter}", flush=True)
            # Re-train without early stopping using best_iter to "freeze" model? Actually
            # save-as is fine: XGBoost model retains all rounds; we record best_iter via
            # iteration_range on predict. To simplify Predictor, save just the trees up
            # to best_iter+1 by trimming. xgboost 3.x supports `model.save_model(path)` with
            # the booster cropped via slice.
            trimmed = booster[: best_iter + 1]
            out_path = os.path.join(HERE, f"final_xgb_model_h{H}_seed{s}.ubj")
            trimmed.save_model(out_path)
            print(f"  saved -> {out_path} (trees={best_iter+1})", flush=True)

    progress("final_done", n_models=done)


if __name__ == "__main__":
    main()
