"""T132: Train all 9 quantile models (3 seeds x 3 alphas) in one process.

Loads schemeP cache once, trains all configurations.
"""
from __future__ import annotations

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

SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
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

ALPHAS = (0.30, 0.50, 0.70)
SEEDS = (1, 7, 42)


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

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])

    info = {"strategy": "V4", "n_train": int(len(X_tr)), "n_val": int(len(X_va))}
    return (X_tr, y_cls_tr, y_regr_tr, X_va, y_cls_va, y_regr_va, info)


def main():
    H = 60
    print("=== T132 Train All: 3 seeds x 3 alphas (q30/q50/q70) ===", flush=True)
    progress("loading_caches")
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_names = [all_feat_names[i] for i in keep_idx]
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim}", flush=True)

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, info) = build_v4_split(train_full, keep_idx)
    print(f"  {info}", flush=True)

    X_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    sess_map = {0: "am", 1: "pm"}
    test_meta = {
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    }

    summaries = []
    cfg_count = 0
    total_cfg = len(SEEDS) * len(ALPHAS)

    for seed in SEEDS:
        seed_rng = np.random.default_rng(seed * 7919 + 1)
        # Build aug once per seed (shared across 3 alphas)
        aug_idx = np.arange(len(X_tr))  # full reuse aug_ratio=1.0
        X_aug = aug_a_scale(X_tr[aug_idx], seed_rng, lo=0.80, hi=1.20)
        y_cls_aug = y_cls_tr[aug_idx]
        y_regr_aug = y_regr_tr[aug_idx]
        X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
        y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)
        sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
        sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
        cfg = SEED_CONFIGS[seed]

        for alpha in ALPHAS:
            cfg_count += 1
            a_tag = f"{int(round(alpha*100)):02d}"
            tag = f"a{a_tag}_seed{seed}"
            print(f"\n--- [{cfg_count}/{total_cfg}] seed={seed} alpha={alpha} ---", flush=True)
            progress("training", seed=seed, alpha=alpha, cfg=cfg_count, total=total_cfg)

            dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                                 feature_name=feat_names, free_raw_data=False)
            dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                               feature_name=feat_names, reference=dtrain, free_raw_data=False)

            params = {
                "objective": "quantile",
                "alpha": float(alpha),
                "learning_rate": 0.05,
                "num_leaves": int(cfg["num_leaves"]),
                "min_data_in_leaf": 100,
                "feature_fraction": float(cfg["feature_fraction"]),
                "bagging_fraction": float(cfg["bagging_fraction"]),
                "bagging_freq": 5,
                "lambda_l2": float(cfg["lambda_l2"]),
                "num_threads": 18,
                "seed": seed,
                "feature_fraction_seed": seed + 1,
                "bagging_seed": seed + 2,
                "data_random_seed": seed + 3,
                "verbose": -1,
                "metric": "quantile",
                "device": "gpu",
                "gpu_use_dp": False,
            }

            t_start = time.time()
            booster = lgb.train(
                params, dtrain,
                num_boost_round=600,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=40, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
            train_time = time.time() - t_start
            best_iter = int(booster.best_iteration)

            model_path = os.path.join(HERE, f"model_T132_{tag}.txt")
            booster.save_model(model_path, num_iteration=best_iter)

            va_pred = predict_chunked(booster, X_va)
            va_pinball = float(np.where(y_regr_va >= va_pred,
                                         alpha * (y_regr_va - va_pred),
                                         (alpha - 1.0) * (y_regr_va - va_pred)).mean())
            va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])

            te_pred = predict_chunked(booster, X_te)
            te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
            print(f"  trained in {train_time:.1f}s best_iter={best_iter} | "
                  f"VAL pinball={va_pinball:.7f} corr={va_corr:.4f} | "
                  f"TEST corr={te_corr:.4f} mean={te_pred.mean():+.6f} std={te_pred.std():.6f}",
                  flush=True)

            te_df = pd.DataFrame({**test_meta,
                                  "pred_dmid_norm": te_pred.astype(np.float32)})
            pred_path = os.path.join(HERE, f"pred_T132_{tag}.parquet")
            te_df.to_parquet(pred_path, index=False)

            summary = {
                "tag": tag, "alpha": float(alpha), "seed": seed,
                "best_iter": best_iter,
                "train_time_sec": float(train_time),
                "val": {"pinball": va_pinball, "corr": va_corr},
                "test": {"corr": te_corr,
                         "pred_mean": float(te_pred.mean()),
                         "pred_std": float(te_pred.std())},
            }
            with open(os.path.join(HERE, f"summary_T132_{tag}.json"), "w") as f:
                json.dump(summary, f, indent=2)
            summaries.append(summary)

    with open(os.path.join(HERE, "all_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    progress("done_train_all", n_models=len(summaries))
    print(f"\n=== ALL DONE: trained {len(summaries)} models ===", flush=True)


if __name__ == "__main__":
    main()
