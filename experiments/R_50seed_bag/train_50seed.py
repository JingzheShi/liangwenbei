"""R_50seed_bag: 50-seed LightGBM L2 bag (only seed varies, fixed hyperparams).

Replicates T75 LGB L2 baseline (regression on Δmid normalized) but trains 50
seeds with a SINGLE FIXED hyperparam config (the canonical seed-42 cfg from
T75: feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0).

Diversity comes purely from random seed (LightGBM seeds + bagging/feature
sampling RNGs are derived from `seed` field).

Train: schemeP date 0-75 (V4 walk-forward).
Val:   schemeP date 76-79 (used for early stopping AND isotonic calibration set).
Test:  schemeP test split (V4 OOT).

Saves:
  model_h60_seed{S}.txt
  pred_T75_50seed_seed{S}.parquet  (test preds)
  pred_T75_50seed_val_seed{S}.parquet (val preds, for isotonic calibration)
  summary_seed{S}.json
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

# Single fixed config for the bag (= T75 seed-42 canonical config).
FIXED_CFG = dict(
    feature_fraction=0.8,
    bagging_fraction=0.8,
    num_leaves=127,
    lambda_l2=1.0,
)

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
    payload = {
        "status": "running", "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
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

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]
    sym_va = train_full["sym"][m_va] if "sym" in train_full else None

    info = {
        "strategy": "V4",
        "train_dates": "0-75",
        "val_dates": "76-79",
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }
    return (X_tr, y_cls_tr, y_regr_tr,
            X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, sym_va, info)


def train_one_seed(seed, train_full, test_full, slicer, feat_names, args, save_dir):
    H = args.horizon
    print(f"\n=== seed={seed} ===", flush=True)
    cfg = FIXED_CFG

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, sym_va, info) = build_v4_split(
        train_full, slicer,
    )

    X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]

    seed_rng = np.random.default_rng(seed * 7919 + 1)

    n_aug = int(round(args.aug_ratio * len(X_tr)))
    if n_aug > 0:
        if n_aug == len(X_tr):
            aug_idx = np.arange(len(X_tr))
        else:
            aug_idx = seed_rng.integers(0, len(X_tr), size=n_aug)
        X_aug = aug_a_scale(X_tr[aug_idx], seed_rng,
                            lo=args.aug_lo, hi=args.aug_hi)
        y_cls_aug = y_cls_tr[aug_idx]
        y_regr_aug = y_regr_tr[aug_idx]
        X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
        y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)
    else:
        X_tr_full = X_tr
        y_cls_tr_full = y_cls_tr
        y_regr_tr_full = y_regr_tr

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)

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

    t0 = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    train_time = time.time() - t0
    best_iter = int(booster.best_iteration)
    print(f"  seed={seed} trained {train_time:.1f}s best_iter={best_iter}", flush=True)

    model_path = os.path.join(save_dir, f"model_h60_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Val preds (raw) for isotonic calibration
    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])

    # Test preds
    te_pred = predict_chunked(booster, X_te)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])

    print(f"  seed={seed} VAL corr={va_corr:.4f} mse={va_mse:.7f}  TEST corr={te_corr:.4f}",
          flush=True)

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
    te_df.to_parquet(os.path.join(save_dir, f"pred_T75_50seed_seed{seed}.parquet"),
                     index=False)

    # Save val preds (with sym/mp/labels) for isotonic + DE
    val_df = pd.DataFrame({
        "sym": (sym_va.astype(np.int8) if sym_va is not None else
                np.full(len(X_va), -1, dtype=np.int8)),
        "true_label": y_cls_va.astype(np.int8),
        "true_dmid_norm": y_regr_va.astype(np.float32),
        "pred_dmid_norm": va_pred.astype(np.float32),
        "midprice_t": mp_t_va.astype(np.float32),
        "midprice_th": mp_th_va.astype(np.float32),
    })
    val_df.to_parquet(os.path.join(save_dir, f"pred_T75_50seed_val_seed{seed}.parquet"),
                      index=False)

    summary = {
        "seed": seed,
        "horizon": H,
        "objective": "regression_l2",
        "split_info": info,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr},
        "fixed_cfg": cfg,
    }
    with open(os.path.join(save_dir, f"summary_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-from", type=int, default=1)
    ap.add_argument("--seeds-to", type=int, default=50)  # inclusive
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--save-dir", default=HERE)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    progress("loading_caches")
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim, (len(all_feat_names), total_dim)

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
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
    assert not leak, f"leak: {leak}"
    print(f"feat_dim={feat_dim}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=f"R_50seed_bag-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "model": "lightgbm-gpu",
                    "task": "R_50seed_bag",
                    "objective": "regression_l2",
                    "fixed_cfg": FIXED_CFG,
                    "n_features": feat_dim,
                    "horizon": args.horizon,
                    "n_seeds": args.seeds_to - args.seeds_from + 1,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["R_50seed_bag", "M2", "isotonic", "regression"],
            )
        except Exception as e:
            print(f"WandB init failed: {e!r}", flush=True)
            use_wandb = False

    summaries = {}
    seeds = list(range(args.seeds_from, args.seeds_to + 1))
    for i, sd in enumerate(seeds):
        progress(f"training_seed_{sd}", idx=i + 1, total=len(seeds))
        try:
            s = train_one_seed(sd, train_full, test_full, slicer, feat_names,
                               args, save_dir)
            summaries[sd] = s
            if use_wandb:
                wandb.log({
                    "seed": sd,
                    "best_iter": s["best_iter"],
                    "train_time_sec": s["train_time_sec"],
                    "val_corr": s["val"]["corr"],
                    "val_mse": s["val"]["mse"],
                    "test_corr": s["test"]["corr"],
                    "test_mse": s["test"]["mse"],
                })
        except Exception as e:
            print(f"  seed={sd} FAILED: {e!r}", flush=True)
            summaries[sd] = {"seed": sd, "error": repr(e)}

    out = os.path.join(save_dir, "all_summaries.json")
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved {out}", flush=True)

    if use_wandb:
        wandb.finish()
    progress("training_done", n_seeds=len(seeds))


if __name__ == "__main__":
    main()
