"""T166: CatBoost L2 (RMSE) 5-seed FULL RETRAIN on all data (0-119) for submission.

M7 trick: train on full data (train+val+test = dates 0-119) with augmentation.
These models are for the submission package (not for holdout eval).
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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from catboost import CatBoostRegressor, Pool

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
H = 60
CHUNK = 200_000

SEED_CONFIGS = {
    1:   dict(seed=1,   depth=6, l2_leaf_reg=3.0, learning_rate=0.05),
    7:   dict(seed=7,   depth=8, l2_leaf_reg=5.0, learning_rate=0.05),
    13:  dict(seed=13,  depth=6, l2_leaf_reg=2.0, learning_rate=0.05),
    42:  dict(seed=42,  depth=7, l2_leaf_reg=3.0, learning_rate=0.05),
    100: dict(seed=100, depth=8, l2_leaf_reg=5.0, learning_rate=0.05),
}

DROP_NAMES = {
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100", "liq_asym_top5_W5",
}


def aug_a_scale_chunked(rng, X_src, X_dst, lo=0.80, hi=1.20):
    n = len(X_src)
    for i in range(0, n, CHUNK):
        end = min(i + CHUNK, n)
        sz = end - i
        scales = rng.uniform(lo, hi, size=(sz, X_src.shape[1])).astype(np.float32)
        X_dst[i:end] = X_src[i:end] * scales
    del scales


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = ap.parse_args()

    seed = args.seed
    cfg = SEED_CONFIGS[seed]
    print(f"=== T166 CB FULL RETRAIN seed={seed} iters={args.iterations} cfg={cfg} ===", flush=True)

    # Load feature mapping
    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = len(all_feat_names)
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))
    print(f"  feat_dim={feat_dim}", flush=True)

    # Count total rows
    progress("counting_rows", seed=seed)
    t0 = time.time()
    split_sizes = {}
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        split_sizes[split] = len(d["date"])
        d.close()
    n_total = sum(split_sizes.values())
    print(f"  N total = {n_total:,}  (train={split_sizes['train']:,} "
          f"val={split_sizes['val']:,} test={split_sizes['test']:,})", flush=True)

    # Pre-allocate: original + augmented
    n_tr = n_total * 2
    print(f"  Pre-allocating {n_tr:,} x {feat_dim} float32 = "
          f"{n_tr*feat_dim*4/1e9:.2f} GB", flush=True)
    X_tr = np.empty((n_tr, feat_dim), dtype=np.float32)
    y_regr = np.empty(n_tr, dtype=np.float32)
    y_cls = np.empty(n_tr, dtype=np.int64)
    gc.collect()

    # Load all splits into first half
    progress("loading_splits", seed=seed)
    offset = 0
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        n = split_sizes[split]
        X_tr[offset:offset+n] = d["X"][:, keep_idx].astype(np.float32)
        mp_t = d["mp_t"].astype(np.float64)
        mp_th = d[f"mp_t{H}"].astype(np.float64)
        y_regr[offset:offset+n] = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
        y_cls[offset:offset+n] = d[f"y{H}"]
        offset += n
        d.close()
        gc.collect()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Augmentation into second half
    progress("augmentation", seed=seed)
    rng = np.random.default_rng(seed * 7919 + 1)
    print(f"  augmenting {n_total:,} rows ...", flush=True)
    aug_a_scale_chunked(rng, X_tr[:n_total], X_tr[n_total:], lo=0.80, hi=1.20)
    y_regr[n_total:] = y_regr[:n_total]
    y_cls[n_total:] = y_cls[:n_total]
    gc.collect()
    print(f"  augmented in {time.time()-t0:.1f}s", flush=True)

    sw = class_balanced_weight(y_cls, num_class=NUM_CLASS)

    # Build pool
    progress("building_pool", seed=seed)
    train_pool = Pool(X_tr, label=y_regr, weight=sw, feature_names=feat_names)
    del X_tr, y_regr, sw, y_cls
    gc.collect()

    cb_params = dict(
        iterations=args.iterations,
        learning_rate=cfg["learning_rate"],
        depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        verbose=100,
        early_stopping_rounds=None,
        allow_writing_files=False,
    )
    if args.use_gpu:
        cb_params["task_type"] = "GPU"
        cb_params["devices"] = "0"
        print("  Using GPU", flush=True)
    else:
        cb_params["thread_count"] = 18

    progress("training", seed=seed, n_total=n_total)
    t_start = time.time()
    model = CatBoostRegressor(**cb_params)
    model.fit(train_pool)
    train_time = time.time() - t_start
    print(f"  trained in {train_time:.1f}s, tree_count={model.tree_count_}", flush=True)

    model_path = os.path.join(HERE, f"model_CB_full_seed{seed}.cbm")
    model.save_model(model_path)
    print(f"  saved {model_path}", flush=True)

    summary = {
        "task": f"T166 CB FULL RETRAIN seed={seed}",
        "seed": seed, "n_total": n_total, "n_train_used": n_total * 2,
        "iterations": args.iterations, "tree_count": int(model.tree_count_),
        "train_time_sec": float(train_time),
        "cb_params": {k: v for k, v in cb_params.items() if k != "verbose"},
    }
    with open(os.path.join(HERE, f"summary_CB_full_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved summary", flush=True)

    progress("done", seed=seed, train_time_sec=float(train_time))
    print(f"\n=== FULL RETRAIN seed={seed} DONE in {train_time:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
