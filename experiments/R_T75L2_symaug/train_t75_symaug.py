"""R_T75L2_symaug: T75 LightGBM regression_l2 with sym-coupled group augmentation.

Variants (selected by --variant):
  - baseline    : per-(sample, feat) aug_a U[0.80, 1.20] (matches T75 LGB L2)
  - group_only  : 4-group coupled scale (price, size, flow, derived); per-element scaling OFF
  - combined    : per-element aug_a × group-coupled scale (multiplied)

Group-coupled aug (group_only / combined):
  s_price   = U[0.85, 1.15]
  s_size    = U[0.70, 1.40]   (larger spread - sizes vary more across syms)
  s_flow    = U[0.80, 1.20]
  s_derived = U[0.85, 1.15]
  X[:, group_idx] *= s_group  (one scalar per sample per group)

Cache: /root/lwb_remote_pkg/cache/schemeP_{train,val,test}.npz
Out  : /root/lwb_work_t75_symaug/

CRITICAL: sym/date NOT features, aug train-only, regression_l2.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import lightgbm as lgb

CACHE_DIR = "/root/lwb_remote_pkg/cache"
OUT_DIR = "/root/lwb_work_t75_symaug"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

SEED_CONFIGS = {
    42: dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:  dict(seed=1,  feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:  dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
}

T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP = T59_FAIL + STAGE5_FAIL

GROUP_SCALE = {
    "price":   (0.85, 1.15),
    "size":    (0.70, 1.40),
    "flow":    (0.80, 1.20),
    "derived": (0.85, 1.15),
}
GROUP_NAMES = ("price", "size", "flow", "derived")


def classify(name):
    if re.match(r'^(lb|la|mb|ma|cb|ca)_(intst|ind|acc)$', name):
        return 'flow'
    if re.match(r'^ewma_a[0-9.]+_(lb|la|mb|ma|cb|ca)_intst$', name):
        return 'flow'
    if name.startswith('mlofi_') or name.startswith('gofi_') or name.startswith('ewma_ofi_'):
        return 'flow'
    if re.match(r'^bsize\d+$', name) or re.match(r'^asize\d+$', name):
        return 'size'
    if name in ('bsize_mean', 'asize_mean', 'totalbsize', 'totalasize',
                'volume_delta', 'amount_delta', 'imbalance'):
        return 'size'
    if name in ('open', 'high', 'low', 'close', 'avgbid', 'avgask', 'cumspread',
                'bid_mean', 'ask_mean'):
        return 'price'
    if re.match(r'^bid\d+$', name) or re.match(r'^ask\d+$', name):
        return 'price'
    if re.match(r'^midprice\d+$', name):
        return 'price'
    if re.match(r'^spread\d+$', name):
        return 'price'
    if re.match(r'^bid_diff\d+$', name) or re.match(r'^ask_diff\d+$', name):
        return 'price'
    if re.match(r'^wmp_lvl\d+$', name):
        return 'price'
    return 'derived'


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def progress(step, **extra):
    p = os.path.join(OUT_DIR, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def aug_per_element(X, rng, lo=0.80, hi=1.20):
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return X * scales


def aug_group_coupled(X, rng, group_idx_map):
    """Per-sample, one scalar scale per group, applied to all features in that group.

    group_idx_map: {group_name: 1-D np.ndarray of column indices}
    X: shape [N, F]; mutated only via copy.
    """
    out = X.copy()
    N = X.shape[0]
    for g in GROUP_NAMES:
        idx = group_idx_map[g]
        if len(idx) == 0:
            continue
        lo, hi = GROUP_SCALE[g]
        scales = rng.uniform(lo, hi, size=(N, 1)).astype(X.dtype)
        # broadcast scale across the group columns
        out[:, idx] = (out[:, idx] * scales).astype(X.dtype)
    return out


def build_v4_split(train_full, slicer):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    mp_t_tr = train_full["mp_t"][m_t]
    mp_th_tr = train_full["mp_t60"][m_t]
    y_regr_tr = regr_target(mp_t_tr, mp_th_tr)

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]
    y_regr_va = regr_target(mp_t_va, mp_th_va)

    return (X_tr, y_cls_tr, y_regr_tr,
            X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=["baseline", "group_only", "combined"], required=True)
    ap.add_argument("--horizon", type=int, default=60)
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

    H = args.horizon
    seed = args.seed
    variant = args.variant
    print(f"=== R_T75L2_symaug variant={variant} seed={seed} h={H} ===", flush=True)

    progress("loading_caches", variant=variant, seed=seed)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f if line.strip()]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim, (len(all_feat_names), total_dim)

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_idx], dtype=np.int64,
    )
    slicer = keep_idx
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, leak
    print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)} fail features)", flush=True)

    # Build group index map (post-slice, so indices align with X_tr columns)
    group_idx_map = {g: [] for g in GROUP_NAMES}
    for i, n in enumerate(feat_names):
        group_idx_map[classify(n)].append(i)
    group_idx_map = {g: np.array(v, dtype=np.int64) for g, v in group_idx_map.items()}
    for g, idx in group_idx_map.items():
        print(f"  group '{g}': {len(idx)} features", flush=True)

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va) = build_v4_split(train_full, slicer)
    print(f"  n_train={len(X_tr):,} n_val={len(X_va):,}", flush=True)

    X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T75L2symaug-{variant}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu", "task": "R_T75L2_symaug",
                    "variant": variant, "n_features": feat_dim,
                    "horizon": H, "seed": seed,
                    "group_sizes": {g: int(len(group_idx_map[g])) for g in GROUP_NAMES},
                    "group_scales": GROUP_SCALE,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["R_T75L2_symaug", variant, f"seed{seed}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", variant=variant, seed=seed)

    seed_rng = np.random.default_rng(seed * 7919 + 1)

    # Build augmented batch (always doubles the size: orig + augmented copy)
    if variant == "baseline":
        X_aug = aug_per_element(X_tr, seed_rng, 0.80, 1.20)
    elif variant == "group_only":
        X_aug = aug_group_coupled(X_tr, seed_rng, group_idx_map)
    elif variant == "combined":
        X_aug = aug_per_element(X_tr, seed_rng, 0.80, 1.20)
        X_aug = aug_group_coupled(X_aug, seed_rng, group_idx_map)
    else:
        raise ValueError(variant)

    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
    del X_aug

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

    model_path = os.path.join(OUT_DIR, f"model_T75L2_{variant}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Eval val + test
    def predict_chunked(X, batch=200_000):
        chunks = []
        for s in range(0, len(X), batch):
            chunks.append(booster.predict(X[s:s+batch]).astype(np.float32))
        return np.concatenate(chunks, axis=0)

    va_pred = predict_chunked(X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    te_pred = predict_chunked(X_te)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  VAL  mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST mse={te_mse:.7f} corr={te_corr:.4f}", flush=True)

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
    pred_path = os.path.join(OUT_DIR, f"pred_T75L2_{variant}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    # Quick raw EV-gate sanity at k=1
    fee_thr = 2.0 * FEE
    side = np.zeros_like(te_pred, dtype=np.float64)
    side[te_pred > fee_thr] = 1.0
    side[te_pred < -fee_thr] = -1.0
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    cum_pnl = float(((side * diff - fee_pnl) / denom).sum())
    print(f"  TEST EV-gate k=1: cum_pnl={cum_pnl:+.4f}", flush=True)

    summary = {
        "task": f"R_T75L2_symaug variant={variant} seed={seed}",
        "variant": variant,
        "seed": seed,
        "horizon": H,
        "n_features": feat_dim,
        "group_sizes": {g: int(len(group_idx_map[g])) for g in GROUP_NAMES},
        "group_scales": GROUP_SCALE,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr, "ev_gate_k1_cum_pnl": cum_pnl},
        "params": vars(args),
    }
    out_path = os.path.join(OUT_DIR, f"summary_T75L2_{variant}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb.log({
            "best_iter": best_iter, "train_time_sec": float(train_time),
            "val_mse": va_mse, "val_corr": va_corr,
            "test_mse": te_mse, "test_corr": te_corr,
            "test_ev_gate_k1_cum_pnl": cum_pnl,
        })
        wandb.finish()
    progress("done", variant=variant, seed=seed,
             best_iter=best_iter, val_mse=va_mse, test_corr=te_corr,
             test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
