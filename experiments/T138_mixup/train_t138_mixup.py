"""T138: T75 L2 LightGBM with input Mixup augmentation.

Same base as T75 (regression_l2, aug_a scale) plus:
  - Mixup within same date: λ~Beta(α,α), mixed_X/y/cls from pairs i,j where date[i]==date[j]
  - N_mixed = 0.5 * len(X_tr)

Variants: α ∈ {0.2, 0.4, 1.0}  ×  seeds ∈ {42, 1}  = 6 runs total
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

import lightgbm as lgb  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import aug_a_scale, class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

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


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
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


def build_v4_split(train_full, slicer):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    y_date_tr = train_full["date"][m_t]

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]

    info = {
        "strategy": "V4",
        "train_dates": "0-75",
        "val_dates": "76-79 (walk-forward last 5%)",
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }
    return (X_tr, y_cls_tr, y_regr_tr, y_date_tr,
            X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info)


def build_mixup(X_tr, y_regr_tr, y_cls_tr, y_date_tr, alpha, seed, int_alpha, mixup_ratio=0.5):
    """Build mixup augmentation batches within same date."""
    rng = np.random.default_rng(seed * 13337 + int_alpha)
    unique_dates = np.unique(y_date_tr)
    all_i, all_j = [], []
    for d in unique_dates:
        idx_d = np.where(y_date_tr == d)[0]
        if len(idx_d) < 2:
            continue
        n_pairs = max(1, int(len(idx_d) * mixup_ratio))
        chosen = rng.choice(idx_d, size=n_pairs * 2, replace=True)
        all_i.append(chosen[:n_pairs])
        all_j.append(chosen[n_pairs:])
    all_i = np.concatenate(all_i)
    all_j = np.concatenate(all_j)
    lam = rng.beta(alpha, alpha, size=len(all_i)).astype(np.float32)
    mixed_X = lam[:, None] * X_tr[all_i] + (1 - lam[:, None]) * X_tr[all_j]
    mixed_y = lam * y_regr_tr[all_i] + (1 - lam) * y_regr_tr[all_j]
    mixed_cls_float = (lam * y_cls_tr[all_i].astype(np.float32)
                       + (1 - lam) * y_cls_tr[all_j].astype(np.float32))
    mixed_cls = np.round(mixed_cls_float).astype(np.int64).clip(0, 2)
    print(f"  mixup: {len(all_i):,} pairs from {len(unique_dates)} dates  "
          f"α={alpha}  lam_mean={lam.mean():.3f}", flush=True)
    return mixed_X, mixed_y, mixed_cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Beta distribution α for mixup (0 means no mixup)")
    ap.add_argument("--mixup-ratio", type=float, default=0.5,
                    help="Ratio of mixed samples to training samples")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    alpha = args.alpha
    alpha_str = f"{alpha:.1f}".replace(".", "p")
    int_alpha = int(alpha * 100)
    label = f"T138_a{alpha_str}_seed{seed}"
    print(f"=== T138 mixup augmentation: α={alpha} seed={seed} ===", flush=True)

    progress("loading_caches", seed=seed, alpha=alpha)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    slicer = keep_idx
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)} fail features)", flush=True)

    feat_names = [all_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"leaky features: {leak}"

    (X_tr, y_cls_tr, y_regr_tr, y_date_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info) = build_v4_split(train_full, slicer)
    print(f"  {info}", flush=True)

    X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
    y_cls_te = test_full["y60"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full["mp_t60"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T138-mixup-a{alpha_str}-seed{seed}-{datetime.now().strftime('%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "task": "T138_mixup",
                    "alpha": alpha,
                    "seed": seed,
                    "n_mixed_ratio": 0.5,
                    "aug_ratio": args.aug_ratio,
                    "aug_lo": args.aug_lo,
                    "aug_hi": args.aug_hi,
                    "n_features": feat_dim,
                },
                tags=["T138", "mixup", f"alpha{alpha_str}", f"seed{seed}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("building_augmentation", seed=seed, alpha=alpha)

    seed_rng = np.random.default_rng(seed * 7919 + 1)

    # aug_a augmentation (same as T75)
    n_aug = int(round(args.aug_ratio * len(X_tr)))
    if n_aug > 0:
        aug_idx = np.arange(len(X_tr)) if n_aug == len(X_tr) else seed_rng.integers(0, len(X_tr), size=n_aug)
        X_aug = aug_a_scale(X_tr[aug_idx], seed_rng, lo=args.aug_lo, hi=args.aug_hi)
        y_cls_aug = y_cls_tr[aug_idx]
        y_regr_aug = y_regr_tr[aug_idx]
        X_all = [X_tr, X_aug]
        y_cls_all = [y_cls_tr, y_cls_aug]
        y_regr_all = [y_regr_tr, y_regr_aug]
    else:
        X_all = [X_tr]
        y_cls_all = [y_cls_tr]
        y_regr_all = [y_regr_tr]

    # Mixup augmentation (new for T138)
    if alpha > 0:
        mixed_X, mixed_y, mixed_cls = build_mixup(
            X_tr, y_regr_tr, y_cls_tr, y_date_tr, alpha, seed, int_alpha, args.mixup_ratio)
        X_all.append(mixed_X)
        y_cls_all.append(mixed_cls)
        y_regr_all.append(mixed_y)

    X_tr_full = np.concatenate(X_all, axis=0)
    y_cls_tr_full = np.concatenate(y_cls_all, axis=0)
    y_regr_tr_full = np.concatenate(y_regr_all, axis=0)

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}  n_val={len(X_va):,}", flush=True)

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
    }
    if args.use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    progress("training", seed=seed, alpha=alpha)
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

    model_path = os.path.join(HERE, f"model_{label}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL: mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)

    te_pred = predict_chunked(booster, X_te)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
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
    pred_path = os.path.join(HERE, f"pred_{label}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    summary = {
        "task": f"T138 mixup α={alpha} seed={seed}",
        "label": label,
        "alpha": alpha,
        "seed": seed,
        "n_features": feat_dim,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "n_train_used": int(len(X_tr_full)),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr},
    }
    out_path = os.path.join(HERE, f"summary_{label}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb and wandb:
        wandb.log({
            "best_iter": best_iter,
            "train_time_sec": float(train_time),
            "val_mse": va_mse, "val_corr": va_corr,
            "test_mse": te_mse, "test_corr": te_corr,
        })
        wandb.finish()
    progress("done", seed=seed, alpha=alpha, best_iter=best_iter, val_mse=va_mse)

    print(f"\nDONE: {label}  best_iter={best_iter}  val_mse={va_mse:.7f}  test_corr={te_corr:.4f}")


if __name__ == "__main__":
    main()
