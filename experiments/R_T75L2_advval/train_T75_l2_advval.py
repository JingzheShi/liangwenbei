"""T75 LGB regression_l2 with adversarial-validation sample weighting.

Identical to T75 baseline (V4 split, schemeP 359 feat) EXCEPT sample weight is
sw_class_balanced * sw_advval (multiplicative). When --no-advval flag is set,
falls back to baseline (sw_class_balanced only).

Usage:
  python3 train_T75_l2_advval.py --seed 42                # trick (default)
  python3 train_T75_l2_advval.py --seed 42 --no-advval    # baseline reproducer
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

HERE = "/root/lwb_work_t75_advval"
CACHE_DIR = "/root/lwb_remote_pkg/cache"
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# Same as T75
SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def class_balanced_weight(y_cls, num_class=NUM_CLASS):
    counts = np.bincount(y_cls, minlength=num_class).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    inv = inv / inv.sum() * num_class  # normalize
    return inv[y_cls].astype(np.float32)


def aug_a_scale(X, rng, lo=0.80, hi=1.20):
    """Per-(sample, feat) uniform scale in [lo, hi]."""
    s = rng.uniform(lo, hi, size=X.shape).astype(np.float32)
    return X * s


def progress(step, **extra):
    p = os.path.join(HERE, f"progress_seed{extra.get('seed','x')}_{extra.get('tag','x')}.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def build_v4_split(train_full, slicer, sw_advval=None):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]

    sw_advval_tr = None
    if sw_advval is not None:
        sw_advval_tr = sw_advval[m_t]
        sw_advval_va = sw_advval[m_va]  # not used for early stopping below
    else:
        sw_advval_va = None

    info = {
        "strategy": "V4",
        "train_dates": "0-75",
        "val_dates": "76-79",
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }
    return (X_tr, y_cls_tr, y_regr_tr, sw_advval_tr,
            X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--no-advval", action="store_true",
                    help="If set, skip adv-val reweighting (= baseline T75 LGB L2)")
    ap.add_argument("--sw-advval-file", default="sw_advval.npy",
                    help="filename within HERE")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    use_advval = not args.no_advval
    tag = "trick" if use_advval else "base"
    H = args.horizon
    seed = args.seed
    print(f"=== T75 L2 advval[{use_advval}] seed={seed} h={H} ===", flush=True)

    progress("loading", seed=seed, tag=tag)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_idx], dtype=np.int64,
    )
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak

    sw_advval = None
    if use_advval:
        sw_path = os.path.join(HERE, args.sw_advval_file)
        sw_advval = np.load(sw_path).astype(np.float32)
        assert len(sw_advval) == len(train_full["X"]), \
            f"sw_advval len {len(sw_advval)} != train {len(train_full['X'])}"
        print(f"  loaded sw_advval from {sw_path}: mean={sw_advval.mean():.4f} "
              f"std={sw_advval.std():.4f} min={sw_advval.min():.4f} "
              f"max={sw_advval.max():.4f}", flush=True)

    (X_tr, y_cls_tr, y_regr_tr, sw_advval_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va,
     info) = build_v4_split(train_full, keep_idx, sw_advval)
    print(f"  {info}", flush=True)

    X_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, tag=tag)
    seed_rng = np.random.default_rng(seed * 7919 + 1)

    # Augmentation aug_a_scale (1x ratio)
    n_aug = len(X_tr)
    aug_idx = np.arange(n_aug)
    X_aug = aug_a_scale(X_tr, seed_rng, lo=0.80, hi=1.20)
    y_cls_aug = y_cls_tr
    y_regr_aug = y_regr_tr
    if sw_advval_tr is not None:
        sw_advval_full = np.concatenate([sw_advval_tr, sw_advval_tr], axis=0)
    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)

    sw_cb = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    if use_advval:
        sw_tr = (sw_cb * sw_advval_full).astype(np.float32)
        # Renormalize so mean=mean(sw_cb) (preserves original total weight scale)
        sw_tr *= sw_cb.mean() / sw_tr.mean()
    else:
        sw_tr = sw_cb
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}  n_val={len(X_va):,}", flush=True)
    print(f"  sw_tr stats: mean={sw_tr.mean():.4f} std={sw_tr.std():.4f}", flush=True)

    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression_l2",
        "learning_rate": args.learning_rate,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": args.bagging_freq,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": args.num_threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
        "metric": "l2",
        "device": "gpu",
        "gpu_use_dp": False,
    }

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t_start
    best_iter = int(booster.best_iteration)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    suffix = "advval" if use_advval else "base"
    model_path = os.path.join(HERE, f"model_T75L2_{suffix}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])

    te_pred = predict_chunked(booster, X_te)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])

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
    pred_path = os.path.join(HERE, f"pred_T75L2_{suffix}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)

    summary = {
        "task": f"T75 L2 advval[{use_advval}] seed={seed}",
        "seed": seed, "horizon": H,
        "objective": "regression_l2",
        "advval_enabled": bool(use_advval),
        "split_info": info,
        "n_features": feat_dim,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr},
    }
    out_path = os.path.join(HERE, f"summary_T75L2_{suffix}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)
    print(f"  VAL mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST mse={te_mse:.7f} corr={te_corr:.4f}", flush=True)
    progress("done", seed=seed, tag=tag,
             best_iter=best_iter, val_mse=va_mse, test_corr=te_corr)


if __name__ == "__main__":
    main()
