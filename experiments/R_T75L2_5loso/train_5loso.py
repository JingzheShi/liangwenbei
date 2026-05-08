"""T75 LGB regression_l2: 5-LOSO trainer.

Trains 5 syms x 3 seeds = 15 LightGBM models. For each held-out sym k,
mask out all rows in training where train_full['sym'] == k, then train as
regular T75 LGB L2 (V4 split, schemeP feat). Predict on full test set
(all 5 syms preserved). Saves preds + per-train summary.

Inference is sym-agnostic (model never sees sym feature; sym only used
to gate which training rows are masked out).

Reuses cached schemeP_train/test data from /root/lwb_remote_pkg/cache/.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = "/root/lwb_work_t75_5loso"
CACHE_DIR = "/root/lwb_remote_pkg/cache"
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
SEEDS = (42, 7, 13)
HORIZON = 60

# Same as T75
SEED_CONFIGS = {
    42: dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    7:  dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13: dict(seed=13, feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
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
    inv = inv / inv.sum() * num_class
    return inv[y_cls].astype(np.float32)


def aug_a_scale(X, rng, lo=0.80, hi=1.20):
    s = rng.uniform(lo, hi, size=X.shape).astype(np.float32)
    return X * s


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
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def build_v4_split_loso(train_full, slicer, held_out_sym):
    """V4 split: dates 0-75 train, 76-79 val. ALSO mask out held_out_sym rows from train+val."""
    date_tr = train_full["date"]
    sym_tr = train_full["sym"]
    if held_out_sym is None:
        sym_mask = np.ones_like(sym_tr, dtype=bool)
    else:
        sym_mask = (sym_tr != held_out_sym)
    m_va = (date_tr >= 76) & sym_mask
    m_t = (date_tr < 76) & sym_mask
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])

    info = {
        "strategy": "V4_LOSO",
        "held_out_sym": -1 if held_out_sym is None else int(held_out_sym),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }
    return X_tr, y_cls_tr, y_regr_tr, X_va, y_cls_va, y_regr_va, info


def train_one(seed, held_out_sym, train_full, test_full, keep_idx, feat_names,
              num_boost_round=600, early_stopping=40, learning_rate=0.05,
              min_data_in_leaf=100, bagging_freq=5, num_threads=18):
    """Train one (seed, held_out_sym) combo. held_out_sym=None means full 5-sym."""
    suffix = "full" if held_out_sym is None else f"loso{held_out_sym}"
    print(f"\n=== train seed={seed} held_out={held_out_sym} ===", flush=True)
    progress("training", seed=seed, held_out=str(held_out_sym))

    (X_tr, y_cls_tr, y_regr_tr, X_va, y_cls_va, y_regr_va, info
     ) = build_v4_split_loso(train_full, keep_idx, held_out_sym)
    print(f"  {info}", flush=True)

    seed_rng = np.random.default_rng(seed * 7919 + 1)
    X_aug = aug_a_scale(X_tr, seed_rng, lo=0.80, hi=1.20)
    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}  n_val={len(X_va):,}", flush=True)

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)

    import lightgbm as lgb
    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression_l2",
        "learning_rate": learning_rate,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": bagging_freq,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": num_threads,
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
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.time() - t_start
    best_iter = int(booster.best_iteration)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    model_path = os.path.join(HERE, f"model_T75L2_{suffix}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if len(np.unique(va_pred)) > 1 else 0.0

    # Predict on FULL test set (all 5 syms — model is sym-agnostic)
    X_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{HORIZON}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{HORIZON}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{HORIZON}"]

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
        "seed": seed,
        "held_out_sym": -1 if held_out_sym is None else int(held_out_sym),
        "split_info": info,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr},
    }
    out_path = os.path.join(HERE, f"summary_T75L2_{suffix}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  VAL mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST mse={te_mse:.7f} corr={te_corr:.4f}  --> {pred_path}", flush=True)
    return summary


def main():
    os.makedirs(HERE, exist_ok=True)
    print("=== T75 L2 5-LOSO trainer ===", flush=True)

    progress("loading_data")
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  train rows={len(train_full['X']):,}  test rows={len(test_full['X']):,}",
          flush=True)
    print(f"  train sym counts: {np.bincount(train_full['sym'].astype(int)).tolist()}",
          flush=True)

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
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"feature leak: {leak}"
    print(f"  feat_dim={len(keep_idx)} (dropped {len(drop_idx)})", flush=True)

    summaries = {}
    # 15 LOSO trains
    for held_out_sym in SYMS:
        for seed in SEEDS:
            key = f"loso{held_out_sym}_seed{seed}"
            print(f"\n##### {key} #####", flush=True)
            try:
                summaries[key] = train_one(
                    seed, held_out_sym, train_full, test_full, keep_idx, feat_names,
                )
            except Exception as e:
                print(f"  TRAIN FAILED for {key}: {e}", flush=True)
                summaries[key] = {"error": str(e)}

    out_path = os.path.join(HERE, "all_summaries.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)
    progress("training_done", n_models=len(summaries))


if __name__ == "__main__":
    main()
