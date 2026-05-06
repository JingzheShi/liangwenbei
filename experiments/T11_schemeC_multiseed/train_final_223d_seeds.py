"""T11: Train final 223-d Scheme C LightGBM models per seed for iter_004.

Trains on (train + val) full data (all 5 syms, dates 0..95) with the held-out
test set (dates 96..119) used as the early-stopping validation set — same
recipe as T5b's train_final_223d.py used for iter_002. Produces one model file
per (seed, horizon) for the iter_004 ensemble Predictor.

Outputs:
    final_223_model_h{H}_seed{S}.txt
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

import lightgbm as lgb  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
N_DROP_TAIL = 3

# Same diversity-forcing configs as loso_train.py
SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63, lambda_l2=1.0),
    13:  dict(seed=13,  feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=2.0),
    100: dict(seed=100, feature_fraction=0.5, bagging_fraction=0.6, num_leaves=255, lambda_l2=1.0),
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
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T11 train_final_223d_seeds horizons={horizons} seeds={seeds} ===", flush=True)

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

        for s in seeds:
            done += 1
            cfg = SEED_CONFIGS[s]
            print(f"\n[{done}/{n_total}] training final h={H} seed={s} cfg={cfg}", flush=True)
            progress("training_final", horizon=H, seed=s, done=done, total=n_total)

            dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                                 feature_name=feat_names, free_raw_data=False)
            dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                               feature_name=feat_names, reference=dtrain, free_raw_data=False)
            params = {
                "objective": "multiclass", "num_class": NUM_CLASS,
                "metric": "multi_logloss",
                "learning_rate": args.learning_rate,
                "num_leaves": int(cfg["num_leaves"]),
                "min_data_in_leaf": args.min_data_in_leaf,
                "feature_fraction": float(cfg["feature_fraction"]),
                "bagging_fraction": float(cfg["bagging_fraction"]),
                "bagging_freq": args.bagging_freq,
                "lambda_l2": float(cfg["lambda_l2"]),
                "num_threads": args.num_threads,
                "seed": s,
                "feature_fraction_seed": s + 1,
                "bagging_seed": s + 2,
                "data_random_seed": s + 3,
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
                    lgb.log_evaluation(period=200),
                ],
            )
            print(f"  trained in {time.time()-t_start:.1f}s, best_iter={booster.best_iteration}",
                  flush=True)
            out_path = os.path.join(HERE, f"final_223_model_h{H}_seed{s}.txt")
            booster.save_model(out_path, num_iteration=booster.best_iteration)
            print(f"  saved -> {out_path}", flush=True)

    progress("final_done", n_models=done)


if __name__ == "__main__":
    main()
