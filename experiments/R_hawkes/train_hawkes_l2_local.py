"""Train LGB regression_l2 with/without Hawkes features locally.

Variants:
  baseline: schemeP 359-d only
  hawkes:   schemeP 359-d + Hawkes 6-d = 365-d

3 seeds (1, 7, 42), GPU training.
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
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
HAWKES_DIR = os.path.join(ROOT, "experiments", "R_Hawkes_OFI")

FEE = 0.0001
SYMS = (0, 1, 2, 3, 4)

T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = set(T59_FAIL + STAGE5_FAIL)

SEED_CONFIGS = {
    42: dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0),
    1:  dict(seed=1,  feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=1.0),
    7:  dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63, lambda_l2=2.0),
}


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s:s+batch]).astype(np.float32))
    return np.concatenate(chunks)


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class)
    weights = 1.0 / (counts + 1.0)
    weights /= weights.sum()
    weights *= num_class
    return weights[y.astype(np.int64)]


def aug_a_scale(X, rng, lo=0.8, hi=1.2):
    scale = rng.uniform(lo, hi, size=(1, X.shape[1])).astype(np.float32)
    return X * scale


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=("baseline", "hawkes"), required=True)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    seed = args.seed
    variant = args.variant
    use_hawkes = (variant == "hawkes")
    tag = f"{variant}_seed{seed}"

    print(f"=== Hawkes L2 {variant} seed={seed} ===", flush=True)
    progress("loading", variant=variant, seed=seed)

    t0 = time.time()
    train_npz = np.load(os.path.join(CACHE, "schemeP_train.npz"))
    test_npz = np.load(os.path.join(CACHE, "schemeP_test.npz"))
    print(f"  loaded schemeP in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f if l.strip()]
    total_dim = train_npz["X"].shape[1]
    assert len(all_names) == total_dim

    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_names = [all_names[i] for i in keep_idx]

    hawkes_train = hawkes_test = None
    hawkes_names = []
    if use_hawkes:
        hawkes_train = np.load(os.path.join(HAWKES_DIR, "hawkes_train.npy"))
        hawkes_test = np.load(os.path.join(HAWKES_DIR, "hawkes_test.npy"))
        with open(os.path.join(HAWKES_DIR, "hawkes_train_names.txt")) as f:
            hawkes_names = [l.strip() for l in f if l.strip()]
        print(f"  +Hawkes: {hawkes_train.shape[1]} dims", flush=True)

    feat_names_full = feat_names + hawkes_names

    date_tr = train_npz["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    Xb_tr = train_npz["X"][m_t][:, keep_idx].astype(np.float32)
    Xb_va = train_npz["X"][m_va][:, keep_idx].astype(np.float32)

    if use_hawkes:
        X_tr = np.concatenate([Xb_tr, hawkes_train[m_t]], axis=1)
        X_va = np.concatenate([Xb_va, hawkes_train[m_va]], axis=1)
    else:
        X_tr = Xb_tr
        X_va = Xb_va
    del Xb_tr, Xb_va

    y_cls_tr = train_npz["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_npz["mp_t"][m_t], train_npz["mp_t60"][m_t])
    y_cls_va = train_npz["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_npz["mp_t"][m_va], train_npz["mp_t60"][m_va])

    print(f"  X_tr{X_tr.shape}  X_va{X_va.shape}  feat_dim={X_tr.shape[1]}", flush=True)

    Xb_te = test_npz["X"][:, keep_idx].astype(np.float32)
    if use_hawkes:
        X_te = np.concatenate([Xb_te, hawkes_test], axis=1)
    else:
        X_te = Xb_te
    del Xb_te

    y_regr_te = regr_target(test_npz["mp_t"], test_npz["mp_t60"])
    sym_te = test_npz["sym"]
    mp_t_te = test_npz["mp_t"]
    mp_th_te = test_npz["mp_t60"]
    sess_te = test_npz["sess_idx"]
    date_te = test_npz["date"]
    t_te = test_npz["t"]

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", variant=variant, seed=seed)

    rng = np.random.default_rng(seed * 7919 + 1)
    X_aug = aug_a_scale(X_tr, rng, lo=0.8, hi=1.2)
    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
    del X_tr, X_aug

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=3)
    sw_va = class_balanced_weight(y_cls_va, num_class=3)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names_full, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names_full, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": args.learning_rate,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": 100,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": 5,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": 16,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "device": "gpu",
        "gpu_use_dp": False,
        "verbose": -1,
    }

    t1 = time.time()
    booster = lgb.train(
        params=params, train_set=dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=True),
                   lgb.log_evaluation(period=50)],
    )
    train_time = time.time() - t1
    best_iter = int(booster.best_iteration)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    model_path = os.path.join(HERE, f"model_{tag}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)

    te_pred = predict_chunked(booster, X_te)
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  TEST corr={te_corr:.4f}", flush=True)

    sess_map = {0: "am", 1: "pm"}
    df = pd.DataFrame({
        "sym": sym_te.astype(np.int8),
        "date": date_te.astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in sess_te], dtype=object),
        "t": t_te.astype(np.int16),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_{tag}.parquet")
    df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(df):,} rows)", flush=True)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    per_sym = [float(pnl[sym_te == s].sum()) for s in SYMS]
    print(f"  fixed-thr cum_pnl={cum_pnl:+.4f} n_active={n_active:,}", flush=True)

    summary = {
        "task": f"Hawkes L2 {variant} seed={seed}",
        "variant": variant, "seed": seed,
        "objective": "regression_l2",
        "n_features": X_te.shape[1],
        "best_iter": best_iter, "train_time_sec": float(train_time),
        "val_mse": va_mse, "val_corr": va_corr,
        "test_corr": te_corr,
        "fixed_thr_pnl": cum_pnl,
        "fixed_thr_n_active": n_active,
        "per_sym_fixed": per_sym,
    }
    out_path = os.path.join(HERE, f"summary_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    progress("done", variant=variant, seed=seed,
             best_iter=best_iter, fixed_pnl=cum_pnl)
    print(f"DONE {tag} cum_pnl={cum_pnl:+.4f} val_corr={va_corr:.4f} test_corr={te_corr:.4f}")


if __name__ == "__main__":
    main()
