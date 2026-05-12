"""T191 LGB trainer (T188 protocol; 375-d features incl. 5 trend cols)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR_SP = os.environ.get("T191_CACHE_SP", os.path.join(HERE, "cache_sp"))
CACHE_DIR_TR = os.environ.get("T191_CACHE_TR", os.path.join(HERE, "cache_trend"))

NUM_CLASS = 3
FEE = 0.0001

HP_CONFIGS = [
    dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
]

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES

TREND_FEAT_NAMES = [
    "mean_logret_W10", "mean_logret_W20", "mean_logret_W50",
    "mean_logret_W100", "mid_pct_change_W100",
]

CHUNK = 200_000
AUG_LO, AUG_HI = 0.80, 1.20


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running", "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def aug_a_scale_chunked(rng, X_src, X_dst, lo=AUG_LO, hi=AUG_HI):
    n = len(X_src)
    for i in range(0, n, CHUNK):
        end = min(i + CHUNK, n)
        sz = end - i
        scales = rng.uniform(lo, hi, size=(sz, X_src.shape[1])).astype(np.float32)
        X_dst[i:end] = X_src[i:end] * scales


def load_X(split: str, keep_idx: np.ndarray, name_to_idx: dict, sp_dim: int):
    """Load (N, len(keep_idx)) — composite of schemeP cols and trend cols."""
    sp_path = os.path.join(CACHE_DIR_SP, f"schemeP_{split}.npz")
    tr_path = os.path.join(CACHE_DIR_TR, f"trend_feat_{split}.npz")
    d = np.load(sp_path)
    X_sp = d["X"].astype(np.float32)
    n = X_sp.shape[0]
    mp_t = d["mp_t"].astype(np.float64)
    mp_th = d[f"mp_t60"].astype(np.float64)
    y_cls = d["y60"].astype(np.int64)
    d.close()
    dt = np.load(tr_path)
    T = dt["T"].astype(np.float32)
    dt.close()
    X = np.concatenate([X_sp, T], axis=1)
    return X[:, keep_idx], mp_t, mp_th, y_cls, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--num-boost-round", type=int, default=330)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=16)
    args = ap.parse_args()

    S = args.seed; H = args.horizon
    hp = HP_CONFIGS[(S - 1) % len(HP_CONFIGS)]
    print(f"T191 LGB seed={S} h={H} hp_group={(S-1)%5} {hp}", flush=True)

    out_path = os.path.join(HERE, f"model_h{H}_seed{S}_T191.txt")
    if os.path.isfile(out_path):
        print(f"  SKIP exists {out_path}", flush=True); return

    feat_names_path = os.path.join(CACHE_DIR_SP, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        sp_names = [l.strip() for l in f]
    assert len(sp_names) == 370
    all_feat_names = sp_names + TREND_FEAT_NAMES
    total_dim = len(all_feat_names)
    assert total_dim == 375
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    assert feat_dim == 364

    progress("counting", seed=S)
    t0 = time.time()
    split_sizes = {}
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(CACHE_DIR_SP, f"schemeP_{split}.npz"))
        split_sizes[split] = len(d["date"]); d.close()
    n_total = sum(split_sizes.values())
    n_train_used = n_total * 2
    print(f"  N total = {n_total:,}  n_train_used = {n_train_used:,}", flush=True)

    X_tr_full = np.empty((n_train_used, feat_dim), dtype=np.float32)
    y_regr_tr = np.empty(n_train_used, dtype=np.float32)
    y_cls_tr = np.empty(n_train_used, dtype=np.int64)
    gc.collect()

    progress("loading", seed=S)
    offset = 0
    for split in ["train", "val", "test"]:
        print(f"  loading {split}...", flush=True)
        X, mp_t, mp_th, y_cls, n = load_X(split, keep_idx, name_to_idx, 370)
        X_tr_full[offset:offset + n] = X
        y_regr_tr[offset:offset + n] = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
        y_cls_tr[offset:offset + n] = y_cls
        offset += n
        del X, mp_t, mp_th, y_cls
        gc.collect()
    print(f"  splits loaded {time.time()-t0:.1f}s", flush=True)

    progress("augmenting", seed=S)
    rng = np.random.default_rng(S * 7919 + 42)
    aug_a_scale_chunked(rng, X_tr_full[:n_total], X_tr_full[n_total:])
    y_regr_tr[n_total:] = y_regr_tr[:n_total]
    y_cls_tr[n_total:] = y_cls_tr[:n_total]
    gc.collect()

    sw = class_balanced_weight(y_cls_tr, num_class=NUM_CLASS)
    print(f"  y_regr mean={y_regr_tr.mean():.4e} std={y_regr_tr.std():.4e}", flush=True)

    params = {
        "objective": "regression_l2", "metric": "rmse",
        "learning_rate": args.learning_rate,
        "num_leaves": int(hp["num_leaves"]),
        "feature_fraction": float(hp["feature_fraction"]),
        "bagging_fraction": float(hp["bagging_fraction"]),
        "bagging_freq": args.bagging_freq,
        "lambda_l2": float(hp["lambda_l2"]),
        "min_data_in_leaf": args.min_data_in_leaf,
        "num_threads": args.num_threads,
        "seed": S, "verbose": -1,
        "device": "gpu", "gpu_use_dp": False,
    }
    dtrain = lgb.Dataset(
        X_tr_full, label=y_regr_tr, weight=sw,
        feature_name=feat_names, free_raw_data=True,
    )
    del X_tr_full, y_regr_tr, y_cls_tr, sw
    gc.collect()

    progress("training", seed=S, hp_group=(S - 1) % 5)
    t_tr = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(period=50)],
    )
    train_time = time.time() - t_tr
    del dtrain; gc.collect()
    booster.save_model(out_path)
    print(f"  LGB saved {out_path} ({train_time:.1f}s)", flush=True)

    summary = {"seed": S, "hp_group": (S-1)%5, "hp": hp, "train_time_sec": float(train_time)}
    with open(os.path.join(HERE, f"summary_lgb_seed{S}_T191.json"), "w") as f:
        json.dump(summary, f, indent=2)
    progress("done", seed=S)
    print(f"DONE LGB seed={S} time={train_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
