"""T51 stage 2: train 5 final h=60 models on FULL data (train+val combined) for iter_007 packaging.

Reuses 327-d schemeM cache (after dropping 9 FAIL extras + 3 time-encoding tail).
For each seed in {1, 7, 13, 42, 100}:
  - X = train + val concat
  - aug_a applied on the concatenated set
  - num_boost_round fixed (no early stopping, since no held-out)
  - GPU LightGBM, multiclass logloss

Outputs HERE/full_model_h60_seed{S}.txt and a summary.
"""
from __future__ import annotations

import argparse, json, os, sys, time
from datetime import datetime, timezone
import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import build_train_for_variant, class_balanced_weight  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
NUM_CLASS = 3

SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}


def load_split(split: str):
    p = os.path.join(CACHE_DIR, f"schemeM_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def build_keep_idx():
    """Same drop list as LOSO training: 9 FAIL extras + 3 time-encoding tail."""
    base_dim = 226
    extras_path = os.path.join(CACHE_DIR, "schemeM_extra_feat_names.txt")
    with open(extras_path) as f:
        extras = [line.strip() for line in f]
    report_path = os.path.join(HERE, "sym_invariance_report.json")
    with open(report_path) as f:
        report = json.load(f)
    fail_names = report["summary"]["fail_names"]
    fail_global = [base_dim + extras.index(n) for n in fail_names]
    drop = set(fail_global) | {223, 224, 225}  # tail 3 = time encoding
    total = base_dim + len(extras)
    keep = np.array([i for i in range(total) if i not in drop], dtype=np.int64)
    return keep, fail_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,7,13,42,100")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--num-boost-round", type=int, default=300,
                    help="Fixed iterations (no early stopping). Default = mean of LOSO best_iter.")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--variant", default="aug_a")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--out-prefix", default="full_model_h60_seed")
    args = ap.parse_args()
    H = args.horizon
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"=== T51b full-data h={H} seeds={seeds} ===", flush=True)
    train_full = load_split("train")
    val_full = load_split("val")

    keep, fail_names = build_keep_idx()
    print(f"  keep_dim={len(keep)} dropped 9 FAIL + 3 tail")
    feat_names_path = os.path.join(CACHE_DIR, "schemeM_feat_names.txt")
    with open(feat_names_path) as f:
        all_names = [line.strip() for line in f]
    feat_names = [all_names[i] for i in keep]

    X_tr_base = train_full["X"][:, keep].astype(np.float32)
    y_tr_base = train_full[f"y{H}"].astype(np.int64)
    X_va = val_full["X"][:, keep].astype(np.float32)
    y_va = val_full[f"y{H}"].astype(np.int64)
    X_full = np.concatenate([X_tr_base, X_va], axis=0)
    y_full = np.concatenate([y_tr_base, y_va], axis=0)
    print(f"  full data shape: X={X_full.shape}, y={y_full.shape}")

    summary = {"task": "T51b full-data h60 5-seed", "n_full": int(len(y_full)),
               "feat_dim": int(X_full.shape[1]), "fail_dropped": fail_names,
               "n_iter": args.num_boost_round, "models": {}}

    for seed in seeds:
        cfg = SEED_CONFIGS[seed]
        print(f"\n--- seed={seed} cfg={cfg} ---")
        rng = np.random.default_rng(seed * 7919 + 1)
        X_tr, y_tr, sw_tr = build_train_for_variant(
            args.variant, X_full, y_full, None, None, rng,
            aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
        )
        print(f"  n_aug={len(X_tr):,}")
        dtrain = lgb.Dataset(
            X_tr, label=y_tr, weight=sw_tr,
            feature_name=feat_names, free_raw_data=False,
        )
        params = {
            "objective": "multiclass", "num_class": NUM_CLASS,
            "metric": "multi_logloss",
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
        }
        if args.use_gpu:
            params["device"] = "gpu"; params["gpu_use_dp"] = False
        t0 = time.time()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=args.num_boost_round,
            valid_sets=[dtrain], valid_names=["train"],
            callbacks=[lgb.log_evaluation(period=200)],
        )
        elapsed = time.time() - t0
        print(f"  trained in {elapsed:.1f}s")
        out_path = os.path.join(HERE, f"{args.out_prefix}{seed}.txt")
        booster.save_model(out_path, num_iteration=booster.current_iteration())
        summary["models"][f"seed{seed}"] = {"path": out_path, "n_iter": booster.current_iteration(),
                                            "train_time": elapsed}

    out_path = os.path.join(HERE, "full_data_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}")


if __name__ == "__main__":
    main()
