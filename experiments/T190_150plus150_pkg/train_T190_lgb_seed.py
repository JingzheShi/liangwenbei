"""T190: Extended 150 LGB ensemble.
Seeds 1-50:  original 5 HP configs cycling (same as T188v2).
Seeds 51-150: 10 new HP configs (5 + (S-51)//10), lr varies by (S-51)%3.
All: num_boost_round=330, M7 full retrain (dates 0-119).
"""
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

# Data directory: try /root/T190/cache first (remote), then local fallback
_DATA_CANDIDATES = [
    "/root/T190/cache",
    os.path.join(HERE, "cache"),
    os.path.join(os.path.abspath(os.path.join(HERE, "..", "..")),
                 "experiments", "T68_stage5_features", "cache"),
]
CACHE_DIR = next((d for d in _DATA_CANDIDATES if os.path.isdir(d)), _DATA_CANDIDATES[-1])

NUM_CLASS = 3
FEE = 0.0001

# Original 5 HP configs (seeds 1-50, cycling)
HP_CONFIGS = [
    dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
    # 10 new HP configs for seeds 51-150 (hp_idx = 5 + (S-51)//10)
    dict(feature_fraction=0.7, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0,
         max_depth=8,  min_data_in_leaf=100),       # group 0: seeds 51-60
    dict(feature_fraction=0.6, bagging_fraction=0.75, num_leaves=255, lambda_l2=0.5,
         max_depth=12, min_data_in_leaf=50),          # group 1: seeds 61-70
    dict(feature_fraction=0.8, bagging_fraction=0.9,  num_leaves=63,  lambda_l2=2.0,
         max_depth=16, min_data_in_leaf=100),         # group 2: seeds 71-80
    dict(feature_fraction=0.5, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.5,
         min_data_in_leaf=50,  min_gain_to_split=0.0),  # group 3: seeds 81-90
    dict(feature_fraction=0.6, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0,
         min_data_in_leaf=200, min_gain_to_split=0.0),  # group 4: seeds 91-100
    dict(feature_fraction=0.7, bagging_fraction=0.7,  num_leaves=255, lambda_l2=0.5,
         min_data_in_leaf=10,  min_gain_to_split=0.001), # group 5: seeds 101-110
    dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=127, lambda_l2=2.0,
         min_data_in_leaf=1000, min_gain_to_split=0.0),  # group 6: seeds 111-120
    dict(feature_fraction=0.8, bagging_fraction=0.85, num_leaves=63,  lambda_l2=3.0,
         max_depth=12, min_data_in_leaf=100),         # group 7: seeds 121-130
    dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0,
         min_gain_to_split=0.001, min_data_in_leaf=100), # group 8: seeds 131-140
    dict(feature_fraction=0.7, bagging_fraction=0.8,  num_leaves=255, lambda_l2=0.5,
         min_gain_to_split=0.01, min_data_in_leaf=100),  # group 9: seeds 141-150
]

# Extended LR for seeds 51-150
LR_LIST_EXT = [0.03, 0.05, 0.1]

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES

CHUNK = 200_000
AUG_LO, AUG_HI = 0.80, 1.20


def get_hp_and_lr(S: int):
    """Return (hp_dict, lr) for seed S."""
    if S <= 50:
        hp = HP_CONFIGS[(S - 1) % 5]
        lr = 0.05
    else:
        hp_idx = 5 + (S - 51) // 10
        hp = HP_CONFIGS[hp_idx]
        lr = LR_LIST_EXT[(S - 51) % len(LR_LIST_EXT)]
    return hp, lr


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
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
    del scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="Seed index 1-150")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--num-boost-round", type=int, default=330)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    S = args.seed
    H = args.horizon
    hp, lr = get_hp_and_lr(S)

    print(f"\n{'='*60}", flush=True)
    print(f"T190 LGB seed={S} h={H} lr={lr}", flush=True)
    print(f"  hp: {hp}", flush=True)

    out_path = os.path.join(HERE, f"model_h{H}_seed{S}.txt")
    if os.path.isfile(out_path):
        print(f"  SKIP: {out_path} already exists", flush=True)
        return

    try:
        import wandb
        if not args.no_wandb:
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=f"T190_LGB_seed{S}",
                config=dict(seed=S, horizon=H, lr=lr, **hp),
            )
    except Exception:
        pass

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = len(all_feat_names)
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))
    print(f"  feat_dim={feat_dim}", flush=True)

    progress("counting_rows", seed=S)
    t0 = time.time()

    split_sizes = {}
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        split_sizes[split] = len(d["date"])
        d.close()

    n_total = sum(split_sizes.values())
    n_train_used = n_total * 2
    print(f"  N total = {n_total:,}  n_train_used = {n_train_used:,}", flush=True)

    X_tr_full = np.empty((n_train_used, feat_dim), dtype=np.float32)
    y_regr_tr = np.empty(n_train_used, dtype=np.float32)
    y_cls_tr = np.empty(n_train_used, dtype=np.int64)
    gc.collect()

    progress("loading_splits", seed=S)
    offset = 0
    for split in ["train", "val", "test"]:
        print(f"  loading {split}...", flush=True)
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        n = split_sizes[split]
        X_tr_full[offset:offset + n] = d["X"][:, keep_idx].astype(np.float32)
        mp_t = d["mp_t"].astype(np.float64)
        mp_th = d[f"mp_t{H}"].astype(np.float64)
        y_regr_tr[offset:offset + n] = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
        y_cls_tr[offset:offset + n] = d[f"y{H}"]
        offset += n
        d.close()
        gc.collect()

    print(f"  splits loaded in {time.time()-t0:.1f}s", flush=True)

    progress("augmenting", seed=S)
    rng = np.random.default_rng(S * 7919 + 42)
    aug_a_scale_chunked(rng, X_tr_full[:n_total], X_tr_full[n_total:])
    y_regr_tr[n_total:] = y_regr_tr[:n_total]
    y_cls_tr[n_total:] = y_cls_tr[:n_total]
    gc.collect()

    sw = class_balanced_weight(y_cls_tr, num_class=NUM_CLASS)
    print(f"  y_regr stats: mean={y_regr_tr.mean():.4e} std={y_regr_tr.std():.4e}", flush=True)

    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": lr,
        "num_leaves": int(hp["num_leaves"]),
        "feature_fraction": float(hp["feature_fraction"]),
        "bagging_fraction": float(hp["bagging_fraction"]),
        "bagging_freq": args.bagging_freq,
        "lambda_l2": float(hp["lambda_l2"]),
        "min_data_in_leaf": int(hp.get("min_data_in_leaf", 100)),
        "num_threads": args.num_threads,
        "seed": S,
        "verbose": -1,
        "device": "gpu",
        "gpu_use_dp": False,
    }
    if "max_depth" in hp:
        params["max_depth"] = int(hp["max_depth"])
    if "min_gain_to_split" in hp:
        params["min_gain_to_split"] = float(hp["min_gain_to_split"])

    dtrain = lgb.Dataset(
        X_tr_full, label=y_regr_tr, weight=sw,
        feature_name=feat_names, free_raw_data=True,
    )
    del X_tr_full, y_regr_tr, y_cls_tr, sw
    gc.collect()

    progress("training", seed=S)
    t_tr = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dtrain],
        callbacks=[lgb.log_evaluation(period=50)],
    )
    train_time = time.time() - t_tr
    del dtrain
    gc.collect()

    booster.save_model(out_path)
    print(f"  LGB saved: {out_path}  ({train_time:.1f}s)", flush=True)

    summary = {
        "seed": S,
        "horizon": H,
        "hp": hp,
        "lr": lr,
        "num_boost_round": args.num_boost_round,
        "train_time_sec": float(train_time),
        "model_path": out_path,
    }
    sum_path = os.path.join(HERE, f"summary_lgb_seed{S}.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {sum_path}", flush=True)

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({"train_time_sec": train_time})
            wandb.finish()
    except Exception:
        pass

    progress("done", seed=S, train_time=float(train_time))
    print(f"\nDONE LGB seed={S}  time={train_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
