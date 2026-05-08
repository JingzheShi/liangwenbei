"""T137: Alt robust losses on T75 L2 base config.

Same pipeline as T75 (schemeP 359-d, same 5 seeds, same V4 split),
but swap objective to: mae / quantile_05 / huber_09.

Compare LOSO-equiv (DE 4D asym thresh) to T75 L2 = +36.23.
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

VARIANTS = {
    "mae":       {"objective": "regression_l1", "alpha": None},
    "quantile":  {"objective": "quantile",       "alpha": 0.5},
    "huber":     {"objective": "huber",           "alpha": 0.9},
}


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
        chunks.append(booster.predict(X[s:s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=list(VARIANTS.keys()), default="mae")
    ap.add_argument("--num-boost-round", type=int, default=1500)
    ap.add_argument("--early-stopping", type=int, default=60)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    variant = args.variant
    vconf = VARIANTS[variant]
    seed = args.seed
    obj = vconf["objective"]
    alpha = vconf["alpha"]

    print(f"=== T137 variant={variant} obj={obj} alpha={alpha} seed={seed} ===", flush=True)
    progress("loading_caches", variant=variant, seed=seed)

    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    slicer = keep_idx
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))
    print(f"  feat_dim={feat_dim}", flush=True)

    # V4 split
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va

    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])

    X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
    y_cls_te = test_full["y60"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full["mp_t60"])

    n_tr = len(X_tr)
    n_va = len(X_va)
    print(f"  n_train={n_tr:,}  n_val={n_va:,}  n_test={len(X_te):,}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wmod
            wandb = wmod
            run_name = f"T137-{variant}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={"task": "T137", "variant": variant, "objective": obj,
                        "alpha": alpha, "seed": seed, **vars(args)},
                tags=["T137", "regression", f"variant-{variant}", f"seed{seed}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r})", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", variant=variant, seed=seed)

    rng = np.random.default_rng(seed * 7919 + 1)
    X_aug = aug_a_scale(X_tr, rng, lo=0.80, hi=1.20)
    y_cls_aug = y_cls_tr.copy()
    y_regr_aug = y_regr_tr.copy()
    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)
    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_augmented={len(X_tr_full):,}", flush=True)

    metric_map = {
        "regression_l2": "l2",
        "regression_l1": "l1",
        "quantile": "quantile",
        "huber": "huber",
    }

    params = {
        "objective": obj,
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
        "metric": metric_map.get(obj, "l1"),
    }
    if alpha is not None:
        params["alpha"] = alpha
    if args.use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.time() - t_start
    best_iter = int(booster.best_iteration)
    print(f"  trained {train_time:.1f}s  best_iter={best_iter}", flush=True)

    model_path = os.path.join(HERE, f"model_T137_{variant}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Val metrics
    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)

    # Test predictions
    te_pred = predict_chunked(booster, X_te)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  TEST mse={te_mse:.7f} mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    # Quick EV-gate sanity
    fee_thr = 2.0 * FEE
    pred_action = np.full(len(te_pred), 1, dtype=np.int8)
    pred_action[te_pred > fee_thr] = 2
    pred_action[te_pred < -fee_thr] = 0
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = test_full["mp_t60"].astype(np.float64) - test_full["mp_t"].astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((test_full["mp_t60"].astype(np.float64) + 1.0)
                                       + (test_full["mp_t"].astype(np.float64) + 1.0))
    denom = test_full["mp_t"].astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  EV-gate k=1: cum_pnl={cum_pnl:+.4f}  n_active={n_active:,}", flush=True)

    # Save predictions
    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": test_full["mp_t"].astype(np.float32),
        "midprice_th": test_full["mp_t60"].astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T137_{variant}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    summary = {
        "task": f"T137-{variant} seed={seed}",
        "variant": variant, "seed": seed, "objective": obj, "alpha": alpha,
        "best_iter": best_iter, "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
    }
    out_path = os.path.join(HERE, f"summary_T137_{variant}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {out_path}", flush=True)

    if use_wandb and wandb:
        wandb.log({
            "best_iter": best_iter, "train_time_sec": float(train_time),
            "val_mse": va_mse, "val_mae": va_mae, "val_corr": va_corr,
            "test_mse": te_mse, "test_mae": te_mae, "test_corr": te_corr,
            "test_ev_gate_k1_cum_pnl": cum_pnl, "test_ev_gate_k1_n_active": n_active,
        })
        wandb.finish()

    progress("done", variant=variant, seed=seed, best_iter=best_iter,
             val_corr=va_corr, ev_gate_k1=cum_pnl)


if __name__ == "__main__":
    main()
