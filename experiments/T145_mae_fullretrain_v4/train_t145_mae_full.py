"""T145: Full-data retrain MAE LGB (M7+MAE compound trick).

Combines:
- train_full_memeff.py: full-data retrain (train+val+test, no early stop, M7 trick)
- T137 MAE: objective='regression_l1' instead of 'regression_l2'

Key fixes from T140c:
- NO data_random_seed (causes Dataset.construct() crash)
- free_raw_data=True for memory efficiency
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

import lightgbm as lgb

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3

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

CHUNK = 200_000


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
    print(f"  [PROGRESS] {step}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--num-boost-round", type=int, default=330)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=8)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    print(f"=== T145 MAE full-retrain seed={seed} h={H} num_boost={args.num_boost_round} ===", flush=True)
    progress("start", seed=seed)

    # Load feature names
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = len(all_feat_names)
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    print(f"  feat_dim={feat_dim}", flush=True)
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), f"Forbidden features found: {forbidden & set(feat_names)}"

    progress("counting_rows", seed=seed)
    t0 = time.time()

    # Count rows per split without loading full X
    split_sizes = {}
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        split_sizes[split] = len(d["date"])
        d.close()

    n_total = sum(split_sizes.values())
    print(f"  N total = {n_total:,} (train={split_sizes['train']:,} val={split_sizes['val']:,} test={split_sizes['test']:,})", flush=True)

    # Pre-allocate full training array (2x for augmentation)
    n_aug = n_total
    n_train_used = n_total + n_aug
    print(f"  Pre-allocating X_tr_full: {n_train_used:,} x {feat_dim} = {n_train_used*feat_dim*4/1e9:.2f} GB", flush=True)
    X_tr_full = np.empty((n_train_used, feat_dim), dtype=np.float32)
    y_regr_tr = np.empty(n_train_used, dtype=np.float32)
    y_cls_tr = np.empty(n_train_used, dtype=np.int64)
    print(f"  Allocated in {time.time()-t0:.1f}s", flush=True)
    gc.collect()

    # Load splits directly into pre-allocated arrays
    progress("loading_splits", seed=seed)
    offset = 0
    for split in ["train", "val", "test"]:
        print(f"  loading {split}...", flush=True)
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        n = split_sizes[split]
        X_tr_full[offset:offset+n] = d["X"][:, keep_idx].astype(np.float32)
        mp_t = d["mp_t"].astype(np.float64)
        mp_th = d[f"mp_t{H}"].astype(np.float64)
        y_regr_tr[offset:offset+n] = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
        y_cls_tr[offset:offset+n] = d[f"y{H}"]
        offset += n
        d.close()
        gc.collect()

    print(f"  splits loaded in {time.time()-t0:.1f}s", flush=True)

    # Apply augmentation in chunks into second half
    progress("augmentation", seed=seed)
    seed_rng = np.random.default_rng(seed * 7919 + 1)
    print(f"  augmenting in chunks of {CHUNK:,}...", flush=True)
    aug_a_scale_chunked(seed_rng, X_tr_full[:n_total], X_tr_full[n_total:],
                        lo=args.aug_lo, hi=args.aug_hi)
    y_regr_tr[n_total:] = y_regr_tr[:n_total]
    y_cls_tr[n_total:] = y_cls_tr[:n_total]
    gc.collect()
    print(f"  augmented in {time.time()-t0:.1f}s, n_train_used={n_train_used:,}", flush=True)

    sw_tr = class_balanced_weight(y_cls_tr, num_class=NUM_CLASS)

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)

    params = {
        "objective": "regression_l1",  # MAE (key change vs L2 full-retrain)
        "metric": "mae",
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
        # NO data_random_seed — causes Dataset.construct() crash (T140c bug fix)
        "verbose": -1,
    }
    if not args.no_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    # Build dataset with free_raw_data=True
    progress("dataset_construction", seed=seed)
    print(f"  building lgb.Dataset (free_raw_data=True)...", flush=True)
    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=True)

    print(f"  constructing bins...", flush=True)
    dtrain.construct()
    del X_tr_full, y_regr_tr, sw_tr
    gc.collect()
    print(f"  dataset ready at {time.time()-t0:.1f}s (raw data freed)", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wmod
            wandb = wmod
            run_name = f"T145-mae-fullretrain-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={"task": "T145", "seed": seed, "objective": "regression_l1",
                        "num_boost_round": args.num_boost_round, **cfg},
                tags=["T145", "mae", "full-retrain", f"seed{seed}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r})", flush=True)
            use_wandb = False

    progress("training", seed=seed, n_total=n_train_used)
    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        callbacks=[lgb.log_evaluation(period=50)],
    )
    train_time = time.time() - t_start
    print(f"  trained in {train_time:.1f}s", flush=True)

    model_path = os.path.join(HERE, f"model_h{H}_seed{seed}.txt")
    booster.save_model(model_path)
    print(f"  saved {model_path}", flush=True)

    summary = {
        "task": f"T145 MAE full-retrain seed={seed}",
        "seed": seed,
        "objective": "regression_l1",
        "horizon": H,
        "n_features": feat_dim,
        "num_boost_round": args.num_boost_round,
        "train_time_sec": float(train_time),
        "n_train_full": int(n_train_used),
        "cfg": cfg,
    }
    out_path = os.path.join(HERE, f"summary_h{H}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb and wandb:
        wandb.log({"train_time_sec": float(train_time), "n_train_full": int(n_train_used)})
        wandb.finish()

    progress("done", seed=seed, num_boost_round=args.num_boost_round, train_time_sec=float(train_time))
    print(f"DONE seed={seed}", flush=True)


if __name__ == "__main__":
    main()
