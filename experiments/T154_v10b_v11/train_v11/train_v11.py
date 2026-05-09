"""T154 v11: Train L2 LGB with 259 pruned LOB features + 5 TOD = 264 total.

Pipeline:
  - schemeP cache (370 features) → drop 11 FAIL_NAMES → 359 LOB features
  - slice to 259 LOB features using pruned_idx_in_359 (kept indices from T147/T152)
  - append 5 TOD features from t/sess_idx → 264 total
  - augment LOB features only (TOD unchanged)
  - train L2 LGB with same SEED_CONFIGS as v9b

Saves model to experiments/T154_v10b_v11/train_v11/model_h60_seed{N}.txt
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
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

PRUNED_JSON = os.path.join(ROOT, "experiments", "T152_iter019_v10", "pkg", "pruned_features.json")

NUM_CLASS = 3

SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.80, num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.70, num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.60, num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.50, num_leaves=127, lambda_l2=3.0),
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

TOD_FEAT_NAMES = ["tod_sin", "tod_cos", "is_first_20", "is_last_20", "is_lunch_prox"]

CHUNK = 200_000


def build_tod_features(t: np.ndarray, sess_idx: np.ndarray) -> np.ndarray:
    t = t.astype(np.float32)
    sess_idx = sess_idx.astype(np.float32)
    mins_into_session = t * 3.0 / 60.0
    mins_full = mins_into_session + 100.0 * sess_idx

    tod_sin = np.sin(2.0 * np.pi * mins_full / 200.0).astype(np.float32)
    tod_cos = np.cos(2.0 * np.pi * mins_full / 200.0).astype(np.float32)
    is_first_20 = (mins_into_session <= 20.0).astype(np.float32)
    is_last_20 = (mins_into_session >= 80.0).astype(np.float32)
    is_lunch_prox = ((mins_full >= 90.0) & (mins_full <= 110.0)).astype(np.float32)
    return np.stack([tod_sin, tod_cos, is_first_20, is_last_20, is_lunch_prox], axis=1)


def aug_a_scale_chunked(rng, X_src, X_dst, n_lob_feats, lo=0.80, hi=1.20):
    n = len(X_src)
    for i in range(0, n, CHUNK):
        end = min(i + CHUNK, n)
        sz = end - i
        scales = rng.uniform(lo, hi, size=(sz, n_lob_feats)).astype(np.float32)
        X_dst[i:end, :n_lob_feats] = X_src[i:end, :n_lob_feats] * scales
        X_dst[i:end, n_lob_feats:] = X_src[i:end, n_lob_feats:]  # TOD unchanged


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
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--num-boost-round", type=int, default=330)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    print(f"=== T154 v11 train (259 LOB + 5 TOD = 264) seed={seed} h={H} ===", flush=True)

    # Load feature names (370 total in schemeP cache)
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = len(all_feat_names)
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}

    # Step 1: drop FAIL_NAMES → 359 LOB features
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_359_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    assert len(keep_359_idx) == 359, f"Expected 359, got {len(keep_359_idx)}"
    lob_359_names = [all_feat_names[i] for i in keep_359_idx]

    # Step 2: apply pruning to get 259 LOB features
    with open(PRUNED_JSON) as f:
        pruned_info = json.load(f)
    pruned_idx_in_359 = np.array(pruned_info["pruned_idx_in_359"], dtype=np.int64)  # 259 kept indices
    assert len(pruned_idx_in_359) == 259, f"Expected 259, got {len(pruned_idx_in_359)}"
    lob_259_names = [lob_359_names[i] for i in pruned_idx_in_359]
    print(f"  LOB: 359→259 (dropped {359-len(pruned_idx_in_359)} by adversarial pruning)", flush=True)

    # Map: for each row, X_259 = X_370[:,keep_359_idx][:,pruned_idx_in_359]
    # Combined index: positions in the 370-dim cache for the 259 kept LOB features
    keep_259_in_370 = keep_359_idx[pruned_idx_in_359]

    feat_dim = 259 + 5  # 264
    feat_names = lob_259_names + TOD_FEAT_NAMES
    print(f"  Total feat_dim={feat_dim} (259 LOB + 5 TOD)", flush=True)

    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(lob_259_names))

    progress("counting_rows", seed=seed)
    t0 = time.time()

    split_sizes = {}
    for split in ["train", "val", "test"]:
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        split_sizes[split] = len(d["date"])
        d.close()

    n_total = sum(split_sizes.values())
    n_aug = n_total
    n_train_used = n_total + n_aug
    print(f"  N total = {n_total:,}, n_train_used = {n_train_used:,}", flush=True)
    print(f"  Pre-allocating: {n_train_used:,} x {feat_dim} = {n_train_used*feat_dim*4/1e9:.2f} GB", flush=True)

    X_tr_full = np.empty((n_train_used, feat_dim), dtype=np.float32)
    y_regr_tr = np.empty(n_train_used, dtype=np.float32)
    y_cls_tr = np.empty(n_train_used, dtype=np.int64)
    gc.collect()

    progress("loading_splits", seed=seed)
    offset = 0
    for split in ["train", "val", "test"]:
        print(f"  loading {split}...", flush=True)
        d = np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz"))
        n = split_sizes[split]

        # 259 LOB features
        X_tr_full[offset:offset+n, :259] = d["X"][:, keep_259_in_370].astype(np.float32)

        # 5 TOD features
        tod = build_tod_features(d["t"].astype(np.int32), d["sess_idx"].astype(np.int32))
        X_tr_full[offset:offset+n, 259:] = tod

        mp_t = d["mp_t"].astype(np.float64)
        mp_th = d[f"mp_t{H}"].astype(np.float64)
        y_regr_tr[offset:offset+n] = ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)
        y_cls_tr[offset:offset+n] = d[f"y{H}"]
        offset += n
        d.close()
        gc.collect()

    print(f"  splits loaded in {time.time()-t0:.1f}s", flush=True)

    progress("augmentation", seed=seed)
    seed_rng = np.random.default_rng(seed * 7919 + 1)
    aug_a_scale_chunked(seed_rng, X_tr_full[:n_total], X_tr_full[n_total:],
                        n_lob_feats=259, lo=args.aug_lo, hi=args.aug_hi)
    y_regr_tr[n_total:] = y_regr_tr[:n_total]
    y_cls_tr[n_total:] = y_cls_tr[:n_total]
    gc.collect()
    print(f"  augmented in {time.time()-t0:.1f}s", flush=True)

    sw_tr = class_balanced_weight(y_cls_tr, num_class=NUM_CLASS)

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)

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
        "verbose": -1,
        "metric": "l2",
    }
    if not args.no_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    progress("dataset_construction", seed=seed)
    print(f"  building lgb.Dataset...", flush=True)
    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=True)
    dtrain.construct()
    del X_tr_full, y_regr_tr, sw_tr
    gc.collect()
    print(f"  dataset ready at {time.time()-t0:.1f}s", flush=True)

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

    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                       name=f"T154_v11_seed{seed}_h{H}",
                       config={**cfg, "feat_dim": feat_dim, "lob_features": 259,
                               "tod_features": 5, "num_boost_round": args.num_boost_round})
            wandb.log({"train_time_sec": train_time, "feat_dim": feat_dim, "seed": seed})
            wandb.finish()
        except Exception as e:
            print(f"  wandb error (ignored): {e}", flush=True)

    summary = {
        "task": "T154 v11 (259 LOB + 5 TOD = 264 features)",
        "seed": seed, "horizon": H,
        "n_features": feat_dim,
        "lob_features": 259,
        "tod_features": 5,
        "num_boost_round": args.num_boost_round,
        "train_time_sec": float(train_time),
        "model_path": model_path,
    }
    summ_path = os.path.join(HERE, f"summary_h{H}_seed{seed}.json")
    with open(summ_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary → {summ_path}", flush=True)

    progress("done", seed=seed, train_time_sec=train_time)
    print(f"DONE seed={seed}", flush=True)


if __name__ == "__main__":
    main()
