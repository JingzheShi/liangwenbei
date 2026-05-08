"""T115 Phase 3: Train LGB Huber alpha=1e-3 on T115 variant caches.

Reuses T75/T99 settings:
  V4 split (train 0-75, val 76-79), 60-tick horizon, aug_a x[0.8,1.2], 3-class balanced weights.

Variants: baseline, variant_A (PM additive), variant_B (PM replace), variant_C (smart additive).
Outputs:
  model_T115_<variant>_seed{S}.txt
  pred_T115_<variant>_seed{S}.parquet
  summary_T115_<variant>_seed{S}.json
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "T26_domain_randomization"))

import lightgbm as lgb  # noqa: E402
from build_aug import aug_a_scale, class_balanced_weight  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
NUM_CLASS = 3
FEE = 0.0001

SEED_CONFIGS = {
    42:  dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,  feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
}


def load_split(variant, split):
    p = os.path.join(CACHE_DIR, f"{variant}_{split}.npz")
    print(f"  loading {p}", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["baseline", "variant_A", "variant_B", "variant_C"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-wandb", action="store_true", default=True)
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    variant = args.variant
    tag = f"{variant}_huber{args.alpha:g}"
    print(f"=== T115 LGB seed={seed} variant={variant} alpha={args.alpha} ===", flush=True)

    t0 = time.time()
    train_full = load_split(variant, "train")
    test_full = load_split(variant, "test")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names = [l.strip() for l in open(
        os.path.join(CACHE_DIR, f"{variant}_feat_names.txt"))]
    feat_dim = train_full["X"].shape[1]
    assert len(feat_names) == feat_dim

    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"sym/date in feature names: {leak}"

    # V4 split
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])

    X_te = test_full["X"].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    # aug_a
    cfg = SEED_CONFIGS[seed]
    seed_rng = np.random.default_rng(seed * 7919 + 1)
    n_aug = int(round(args.aug_ratio * len(X_tr)))
    if n_aug > 0:
        aug_idx = (np.arange(len(X_tr)) if n_aug == len(X_tr)
                   else seed_rng.integers(0, len(X_tr), size=n_aug))
        X_aug = aug_a_scale(X_tr[aug_idx], seed_rng, lo=args.aug_lo, hi=args.aug_hi)
        y_cls_aug = y_cls_tr[aug_idx]
        y_regr_aug = y_regr_tr[aug_idx]
        X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
        y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)
    else:
        X_tr_full = X_tr; y_cls_tr_full = y_cls_tr; y_regr_tr_full = y_regr_tr

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  feat_dim={feat_dim}, n_train={len(X_tr_full):,}, n_val={len(X_va):,}, n_test={len(X_te):,}", flush=True)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "huber",
        "alpha": args.alpha,
        "metric": "l1",
        "learning_rate": args.learning_rate,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": 5,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": args.num_threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
    }
    if args.use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    t_start = time.time()
    booster = lgb.train(
        params=params, train_set=dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.time() - t_start
    best_iter = int(booster.best_iteration)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    model_path = os.path.join(HERE, f"model_T115_{tag}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])

    te_pred = predict_chunked(booster, X_te)
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  VAL: mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST: mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T115_{tag}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)

    # Quick EV-gate cum_pnl @ k=1 for ref
    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    print(f"  TEST EV-gate k=1: cum_pnl={cum_pnl:+.4f} n_active={n_active:,}", flush=True)

    summary = {
        "task": f"T115 LGB Huber {variant} seed={seed}",
        "variant": variant, "seed": seed, "alpha": args.alpha,
        "n_features": feat_dim, "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mae": va_mae, "corr": va_corr},
        "test": {"mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
    }
    with open(os.path.join(HERE, f"summary_T115_{tag}_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary saved", flush=True)


if __name__ == "__main__":
    main()
