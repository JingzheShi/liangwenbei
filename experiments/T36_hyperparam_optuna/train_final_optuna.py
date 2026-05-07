"""T36: train FINAL h_60 models (no held-out) using Optuna best hyperparam × N seeds.

For each seed:
    train: ALL train+val (sym in {0,1,2,3,4}, dates 0..95)  + aug_a augment
    val:   no held-out — use a deterministic random 12% split for early stopping
           (matches T27 train_final_aug_a.py pattern)

Outputs final_model_h60_optuna_seed{S}.txt under HERE/.

The N_BOOST_ROUND is read from the LOSO-fold best_iter (averaged) so we don't
overshoot. early_stopping is reduced to 30 since val is small.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
N_DROP_TAIL = 3
HORIZON = 60


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running", "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_concat(X, y, rng, aug_range):
    lo, hi = 1.0 - aug_range, 1.0 + aug_range
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return (np.concatenate([X, X * scales], axis=0),
            np.concatenate([y, y], axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params-json", required=True,
                    help="JSON with keys: params (LGBM hp dict), aug_a_range, suggested_num_boost_round")
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--num-boost-round", type=int, default=None,
                    help="override num_boost_round; otherwise from params-json")
    ap.add_argument("--early-stopping", type=int, default=30)
    ap.add_argument("--val-split", type=float, default=0.12)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--num-threads", type=int, default=18)
    args = ap.parse_args()

    with open(args.params_json) as f:
        cfg = json.load(f)
    base_params = dict(cfg["params"])
    aug_range = float(cfg["aug_a_range"])
    nbr = args.num_boost_round or int(cfg.get("suggested_num_boost_round", 250))

    print(f"=== final_train: aug_a_range={aug_range:.4f} nbr={nbr}", flush=True)
    print(f"  base_params={base_params}", flush=True)

    progress("loading_data")
    train_d = np.load(os.path.join(T5B_CACHE, "schemeC_train.npz"))
    val_d = np.load(os.path.join(T5B_CACHE, "schemeC_val.npz"))
    X_tr_orig = np.concatenate(
        [train_d["X"][:, :-N_DROP_TAIL], val_d["X"][:, :-N_DROP_TAIL]], axis=0
    ).astype(np.float32)
    y_tr_orig = np.concatenate([train_d["y60"], val_d["y60"]], axis=0).astype(np.int64)
    print(f"  full data: X={X_tr_orig.shape} y={y_tr_orig.shape}", flush=True)
    del train_d, val_d

    seeds = [int(s) for s in args.seeds.split(",")]
    summary = {"seeds": seeds, "results": []}

    for s in seeds:
        progress(f"final_seed_{s}")
        params = dict(base_params)
        params.update({
            "objective": "multiclass", "num_class": NUM_CLASS,
            "metric": "multi_logloss", "verbose": -1,
            "num_threads": args.num_threads,
            "seed": s,
            "feature_fraction_seed": s + 1,
            "bagging_seed": s + 2,
            "data_random_seed": s + 3,
        })
        if args.use_gpu:
            params["device"] = "gpu"; params["gpu_use_dp"] = False
        # Strip non-LGBM keys if any
        for k in list(params.keys()):
            if k in ("aug_a_range",):
                del params[k]

        rng = np.random.default_rng(s * 7919 + 1)
        # Hold out a random val_split for early stopping (deterministic per seed)
        n = len(X_tr_orig)
        idx = rng.permutation(n)
        n_val = int(round(args.val_split * n))
        va_idx, tr_idx = idx[:n_val], idx[n_val:]
        X_tr_pre = X_tr_orig[tr_idx]
        y_tr_pre = y_tr_orig[tr_idx]
        X_va = X_tr_orig[va_idx]
        y_va = y_tr_orig[va_idx]

        rng_aug = np.random.default_rng(s * 7919 + 17)
        X_tr, y_tr = aug_a_concat(X_tr_pre, y_tr_pre, rng_aug, aug_range)
        sw_tr = class_balanced_weight(y_tr, NUM_CLASS)
        sw_va = class_balanced_weight(y_va, NUM_CLASS)

        print(
            f"  seed={s} n_tr={len(X_tr):,} n_va={len(X_va):,}", flush=True,
        )
        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                           reference=dtrain, free_raw_data=False)

        t0 = time.time()
        booster = lgb.train(
            params, dtrain, num_boost_round=nbr,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        train_time = time.time() - t0

        out_path = os.path.join(HERE, f"final_model_h60_optuna_seed{s}.txt")
        booster.save_model(out_path, num_iteration=booster.best_iteration)
        size = os.path.getsize(out_path)
        print(
            f"  seed={s} best_iter={booster.best_iteration} "
            f"train_time={train_time:.1f}s saved {out_path} ({size:,} bytes)",
            flush=True,
        )
        summary["results"].append({
            "seed": s, "best_iter": int(booster.best_iteration or 0),
            "train_time_sec": train_time, "model_path": out_path,
            "n_train_used": int(len(X_tr)), "n_val": int(len(X_va)),
        })
        with open(os.path.join(HERE, "final_train_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    progress("final_train_done")
    print("=== final train all seeds done ===", flush=True)


if __name__ == "__main__":
    main()
