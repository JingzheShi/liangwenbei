"""T128: T75 LGB L2 retrained on FULL date 0-119 (train+val+test), no early stop.

Meta-paradigm M7: standard Kaggle final-submit trick.
- Same hyperparams + per-seed configs as T75_regression_dmid.
- num_boost_round fixed = round(avg(T75 best_iter) * 1.1).
- aug_a augment + class-balanced weights identical to T75.
- 5 seeds (1, 7, 13, 42, 100).

Output:
  model_T128_seed{S}.txt  — boosters
  summary_T128_seed{S}.json
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


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--objective", default="regression_l2")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--num-boost-round", type=int, default=96)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    print(f"=== T128 full-retrain seed={seed} h={H} num_boost={args.num_boost_round} ===", flush=True)

    progress("loading_caches", seed=seed)
    t0 = time.time()
    train_ = load_split("train")
    val_ = load_split("val")
    test_ = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_["X"].shape[1]
    assert len(all_feat_names) == total_dim, (len(all_feat_names), total_dim)

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_idx], dtype=np.int64,
    )
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    print(f"  feat_dim={feat_dim}", flush=True)
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))

    # Concatenate train + val + test → date 0-119
    X_all = np.concatenate([train_["X"][:, keep_idx],
                            val_["X"][:, keep_idx],
                            test_["X"][:, keep_idx]], axis=0).astype(np.float32, copy=False)
    y_cls_all = np.concatenate([train_[f"y{H}"], val_[f"y{H}"], test_[f"y{H}"]]).astype(np.int64)
    mp_t_all = np.concatenate([train_["mp_t"], val_["mp_t"], test_["mp_t"]])
    mp_th_all = np.concatenate([train_[f"mp_t{H}"], val_[f"mp_t{H}"], test_[f"mp_t{H}"]])
    date_all = np.concatenate([train_["date"], val_["date"], test_["date"]])
    y_regr_all = regr_target(mp_t_all, mp_th_all)
    print(f"  N total = {len(X_all):,}, date range {date_all.min()}-{date_all.max()}", flush=True)
    assert date_all.min() == 0 and date_all.max() == 119
    print(f"  y_regr stats: mean={y_regr_all.mean():.6f} std={y_regr_all.std():.6f}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T128-fullretrain-seed{seed}-{args.objective}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "task": "T128_full_retrain",
                    "objective": args.objective,
                    "n_features": feat_dim,
                    "n_total": int(len(X_all)),
                    "horizon": H,
                    "seed": seed,
                    "date_range": [0, 119],
                    "no_early_stop": True,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T128", "full_retrain", "M7", f"h{H}", "no_es"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, n_total=int(len(X_all)))

    seed_rng = np.random.default_rng(seed * 7919 + 1)

    n_aug = int(round(args.aug_ratio * len(X_all)))
    if n_aug > 0:
        if n_aug == len(X_all):
            aug_idx = np.arange(len(X_all))
        else:
            aug_idx = seed_rng.integers(0, len(X_all), size=n_aug)
        X_aug = aug_a_scale(X_all[aug_idx], seed_rng,
                            lo=args.aug_lo, hi=args.aug_hi)
        y_cls_aug = y_cls_all[aug_idx]
        y_regr_aug = y_regr_all[aug_idx]
        X_tr_full = np.concatenate([X_all, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_all, y_cls_aug], axis=0)
        y_regr_tr_full = np.concatenate([y_regr_all, y_regr_aug], axis=0)
    else:
        X_tr_full = X_all
        y_cls_tr_full = y_cls_all
        y_regr_tr_full = y_regr_all

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}", flush=True)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)

    params = {
        "objective": args.objective,
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
        "metric": "l2" if args.objective == "regression_l2" else "l1",
    }
    if args.use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        callbacks=[lgb.log_evaluation(period=20)],
    )
    train_time = time.time() - t_start
    print(f"  trained in {train_time:.1f}s", flush=True)

    model_path = os.path.join(HERE, f"model_T128_seed{seed}.txt")
    booster.save_model(model_path)
    print(f"  saved {model_path}", flush=True)

    summary = {
        "task": f"T128 full retrain seed={seed}",
        "seed": seed,
        "horizon": H,
        "objective": args.objective,
        "n_features": feat_dim,
        "drop_names": list(DROP_NAMES),
        "num_boost_round": args.num_boost_round,
        "train_time_sec": float(train_time),
        "n_train_full": int(len(X_tr_full)),
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T128_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "train_time_sec": float(train_time),
            "num_boost_round": args.num_boost_round,
        })
        wandb.finish()
    progress("done", seed=seed,
             num_boost_round=args.num_boost_round,
             train_time_sec=float(train_time))


if __name__ == "__main__":
    main()
