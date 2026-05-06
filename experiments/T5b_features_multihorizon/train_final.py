"""Train final Scheme C1 LightGBM models for chosen horizons.

Uses the FULL training data (all 5 syms) — date 0..95 for train, date 96..119 for
early-stopping val (mirrors iter_001 protocol). The cache already includes
(date 96..119) as the 'test' split; we treat it as val for early stopping.

Outputs:
  final_model_h{H}.txt    LightGBM booster, 226-dim, ready for iter_002.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
NUM_CLASS = 3


def load_split(split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", required=True, help="comma-separated horizons to train final models for")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    print(f"=== Train final models for horizons={horizons} ===", flush=True)

    t0 = time.time()
    train_full = load_split("train")  # date 0..79
    val_full = load_split("val")      # date 80..95  -> merge into train
    test_full = load_split("test")    # date 96..119 -> use as val/early-stopping
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Combine train + val into one larger train; use test as val
    X_tr = np.concatenate([train_full["X"], val_full["X"]], axis=0)
    X_va = test_full["X"]

    feat_names_path = os.path.join(CACHE_DIR, "schemeC_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    print(f"feat_dim = {len(feat_names)}", flush=True)
    assert "date" not in feat_names and "sym" not in feat_names and "time" not in feat_names

    for H in horizons:
        print(f"\n--- training final h={H} ---", flush=True)
        y_tr = np.concatenate([train_full[f"y{H}"], val_full[f"y{H}"]], axis=0).astype(np.int64)
        y_va = test_full[f"y{H}"].astype(np.int64)
        sw_tr = class_balanced_weight(y_tr)
        sw_va = class_balanced_weight(y_va)

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                             feature_name=feat_names, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                           feature_name=feat_names, reference=dtrain, free_raw_data=False)
        params = {
            "objective": "multiclass",
            "num_class": NUM_CLASS,
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
        t_start = time.time()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=args.num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        print(f"  trained in {time.time()-t_start:.1f}s, best_iter={booster.best_iteration}",
              flush=True)
        out_path = os.path.join(HERE, f"final_model_h{H}.txt")
        booster.save_model(out_path, num_iteration=booster.best_iteration)
        print(f"  saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
