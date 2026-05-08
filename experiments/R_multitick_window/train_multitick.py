"""Train LGB L2 GPU baseline vs multi-tick trick variant on T75 setup.

Loads schemeP base (370-d) + drops 11 fail features (-> 359-d). For trick
variant, additionally concatenates 80-d multi-tick features (-> 439-d).

V4 walk-forward split: train=date 0-75, val=date 76-79, test=date 96-119.
LightGBM regression_l2, GPU, num_boost=600, early_stop=40 on val MSE.

Saves: model_{variant}_seed{S}.txt, pred_{variant}_seed{S}.parquet,
       summary_{variant}_seed{S}.json
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
sys.path.insert(0, HERE)

import lightgbm as lgb  # noqa: E402

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

SEED_CONFIGS = {
    42: dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0),
    1:  dict(seed=1,  feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=1.0),
    7:  dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
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


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(base_cache, mt_cache, split):
    pb = os.path.join(base_cache, f"schemeP_{split}.npz")
    pm = os.path.join(mt_cache, f"multitick_{split}.npz")
    print(f"  loading {pb} ...", flush=True)
    db = np.load(pb)
    dm = np.load(pm)
    # verify alignment
    for k in ("sym", "date", "sess_idx", "t"):
        if not np.array_equal(db[k], dm[k]):
            raise AssertionError(f"alignment mismatch on {k} for split {split}")
    return {k: db[k] for k in db.files}, dm["X_multitick"]


def class_balanced_weight(y_cls, num_class=3):
    w = np.ones(len(y_cls), dtype=np.float32)
    counts = np.bincount(y_cls, minlength=num_class).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = (len(y_cls) / counts).astype(np.float32)
    inv = inv / inv.mean()
    for c in range(num_class):
        w[y_cls == c] = inv[c]
    return w


def aug_a_scale(X, rng, lo=0.80, hi=1.20):
    s = rng.uniform(lo, hi, size=X.shape).astype(np.float32)
    return X * s


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def build_v4_split(train_full, mt_train, slicer, use_multitick):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_base_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    X_base_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    if use_multitick:
        X_tr = np.concatenate([X_base_tr, mt_train[m_t]], axis=1)
        X_va = np.concatenate([X_base_va, mt_train[m_va]], axis=1)
    else:
        X_tr = X_base_tr
        X_va = X_base_va
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]
    info = {"strategy": "V4", "n_train": int(len(X_tr)), "n_val": int(len(X_va))}
    return (X_tr, y_cls_tr, y_regr_tr, X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=["baseline", "trick"], required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--objective", default="regression_l2")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--base-cache", default="/root/lwb_remote_pkg/cache")
    ap.add_argument("--mt-cache", default="/root/lwb_work_multitick")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    use_multitick = args.variant == "trick"
    print(f"=== T75 L2 {args.variant} seed={seed} h={H} ===", flush=True)

    progress("loading_caches", seed=seed, variant=args.variant)
    t0 = time.time()
    train_full, mt_train = load_split(args.base_cache, args.mt_cache, "train")
    test_full,  mt_test  = load_split(args.base_cache, args.mt_cache, "test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Drop fail features
    feat_names_path = os.path.join(args.base_cache, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    if len(all_feat_names) != total_dim:
        # the schemeP_feat_names.txt might list more; trim
        all_feat_names = all_feat_names[:total_dim]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_idx], dtype=np.int64,
    )
    base_feat_names = [all_feat_names[i] for i in keep_idx]

    if use_multitick:
        mt_names_path = os.path.join(args.mt_cache, "multitick_feat_names.txt")
        with open(mt_names_path) as f:
            mt_names = [line.strip() for line in f]
        feat_names = base_feat_names + mt_names
    else:
        feat_names = base_feat_names
    feat_dim = len(feat_names)
    print(f"  feat_dim={feat_dim} (base {len(base_feat_names)}, multitick {feat_dim - len(base_feat_names)})", flush=True)

    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"forbidden leak: {leak}"

    (X_tr, y_cls_tr, y_regr_tr, X_va, y_cls_va, y_regr_va,
     mp_t_va, mp_th_va, info) = build_v4_split(train_full, mt_train, keep_idx, use_multitick)
    print(f"  {info}", flush=True)

    # Test
    X_base_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    if use_multitick:
        X_te = np.concatenate([X_base_te, mt_test], axis=1)
    else:
        X_te = X_base_te
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]

    cfg = SEED_CONFIGS[seed]
    seed_rng = np.random.default_rng(seed * 7919 + 1)

    # aug_a
    n_aug = len(X_tr)
    aug_idx = np.arange(len(X_tr))
    X_aug = aug_a_scale(X_tr[aug_idx], seed_rng, lo=0.80, hi=1.20)
    y_cls_aug = y_cls_tr[aug_idx]
    y_regr_aug = y_regr_tr[aug_idx]
    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}  n_val={len(X_va):,}", flush=True)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": args.objective,
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
    }
    if args.use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    progress("training", seed=seed, variant=args.variant)
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

    model_path = os.path.join(HERE, f"model_{args.variant}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Eval
    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    te_pred = predict_chunked(booster, X_te)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  VAL: mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST: mse={te_mse:.7f} corr={te_corr:.4f}", flush=True)

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
    pred_path = os.path.join(HERE, f"pred_{args.variant}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    # Feature importance (gain)
    imp_gain = booster.feature_importance(importance_type="gain")
    imp_split = booster.feature_importance(importance_type="split")
    fi_pairs = sorted(zip(feat_names, imp_gain.tolist(), imp_split.tolist()),
                      key=lambda x: -x[1])
    fi_path = os.path.join(HERE, f"feature_importance_{args.variant}_seed{seed}.json")
    with open(fi_path, "w") as f:
        json.dump([{"name": n, "gain": float(g), "split": int(s)}
                   for n, g, s in fi_pairs[:50]], f, indent=2)

    summary = {
        "task": f"T75 L2 {args.variant} seed={seed}",
        "seed": seed,
        "variant": args.variant,
        "n_features": feat_dim,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr},
    }
    out_path = os.path.join(HERE, f"summary_{args.variant}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    progress("done", seed=seed, variant=args.variant, val_mse=va_mse, test_corr=te_corr)


if __name__ == "__main__":
    main()
