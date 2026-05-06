"""Train final 223-d (no time-encoding) Scheme C1 LightGBM models.

Drops the last 3 cols of the cache (time_minutes_since_session_start,
time_session_progress, time_is_pm) so that the inference pipeline doesn't
depend on `time` being in config.feature (LocalEvaluator coerces all feature
cols to float32, breaking on datetime.time).

Outputs final_223_model_h{H}.txt — input dim 223, ready for iter_002.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
NUM_CLASS = 3
N_DROP_TAIL = 3  # last 3 cols are time encoding


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", required=True)
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
    print(f"=== train_final_223d horizons={horizons} ===", flush=True)

    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Drop last 3 cols (time-encoding)
    X_tr = np.concatenate([train_full["X"][:, :-N_DROP_TAIL],
                           val_full["X"][:, :-N_DROP_TAIL]], axis=0)
    X_va = test_full["X"][:, :-N_DROP_TAIL]
    print(f"X_tr={X_tr.shape}  X_va={X_va.shape}  feat_dim={X_tr.shape[1]}", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeC_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names_full = [line.strip() for line in f]
    feat_names = feat_names_full[:-N_DROP_TAIL]  # drop time encoding
    assert len(feat_names) == X_tr.shape[1]
    assert "time" not in feat_names and "date" not in feat_names and "sym" not in feat_names

    # Save feat_names_223
    out_names = os.path.join(CACHE_DIR, "schemeC_223d_feat_names.txt")
    with open(out_names, "w") as f:
        for n in feat_names: f.write(n + "\n")
    print(f"feat_names_223 -> {out_names}", flush=True)

    for H in horizons:
        print(f"\n--- training final 223d h={H} ---", flush=True)
        y_tr = np.concatenate([train_full[f"y{H}"], val_full[f"y{H}"]], axis=0).astype(np.int64)
        y_va = test_full[f"y{H}"].astype(np.int64)
        sw_tr = class_balanced_weight(y_tr)
        sw_va = class_balanced_weight(y_va)
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                             feature_name=feat_names, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                           feature_name=feat_names, reference=dtrain, free_raw_data=False)
        params = {
            "objective": "multiclass", "num_class": NUM_CLASS,
            "metric": "multi_logloss",
            "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": args.bagging_freq,
            "lambda_l2": args.lambda_l2,
            "num_threads": args.num_threads,
            "seed": args.seed, "verbose": -1,
        }
        t_start = time.time()
        booster = lgb.train(params, dtrain,
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
        out_path = os.path.join(HERE, f"final_223_model_h{H}.txt")
        booster.save_model(out_path, num_iteration=booster.best_iteration)
        print(f"  saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
