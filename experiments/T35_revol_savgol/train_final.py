"""T35: Train final Scheme K models (full train+val) for the iter_006 bundle.

For each seed in --seeds:
    train: (train + val cache, all 5 syms, dates 0..95)  + variant aug
    val (early-stopping): test split (dates 96..119)     -- NO aug

Outputs: HERE/final_model_h{H}_{variant}_seed{S}.txt

Mirrors T27/train_final_aug_a.py but on Scheme K (255-d).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import (  # noqa: E402
    build_train_for_variant,
    class_balanced_weight,
    compute_feat_stats,
    VARIANTS,
)

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
T35_CACHE = os.path.join(HERE, "cache")
NUM_CLASS = 3
N_DROP_TAIL = 3

SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}


def progress(step: str, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_combined(split):
    p_c = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    p_k = os.path.join(T35_CACHE, f"schemeK_extra_{split}.npz")
    print(f"  loading {p_c}", flush=True)
    dc = np.load(p_c)
    print(f"  loading {p_k}", flush=True)
    dk = np.load(p_k)
    for k in ("sym", "date", "sess_idx", "t"):
        assert np.array_equal(dc[k], dk[k]), f"alignment broken on {k}"
    X_base = dc["X"][:, :-N_DROP_TAIL].astype(np.float32, copy=False)
    X_extra = dk["X_extra"].astype(np.float32, copy=False)
    out = {k: dc[k] for k in dc.files if k != "X"}
    out["X"] = np.concatenate([X_base, X_extra], axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="60")
    ap.add_argument("--variant", default="aug_a", choices=list(VARIANTS))
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(
        f"=== T35 train_final {args.variant} horizons={horizons} seeds={seeds} use_gpu={args.use_gpu} ===",
        flush=True,
    )

    progress("loading_caches", horizons=horizons, seeds=seeds, variant=args.variant)
    t0 = time.time()
    train_full = load_combined("train")
    val_full = load_combined("val")
    test_full = load_combined("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    X_tr_orig = np.concatenate([train_full["X"], val_full["X"]], axis=0).astype(np.float32)
    X_va = test_full["X"].astype(np.float32)
    print(f"X_tr_orig={X_tr_orig.shape}  X_va={X_va.shape}  feat_dim={X_tr_orig.shape[1]}",
          flush=True)

    base_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(base_names_path) as f:
        base_names = [line.strip() for line in f]
    extra_names_path = os.path.join(T35_CACHE, "schemeK_extra_feat_names.txt")
    with open(extra_names_path) as f:
        extra_names = [line.strip() for line in f]
    feat_names = base_names + extra_names
    assert len(feat_names) == X_tr_orig.shape[1]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features: {leak}"

    feat_mean, feat_std = compute_feat_stats(X_tr_orig)

    n_total = len(horizons) * len(seeds)
    done = 0
    summary = []
    for H in horizons:
        y_tr_orig = np.concatenate(
            [train_full[f"y{H}"], val_full[f"y{H}"]], axis=0
        ).astype(np.int64)
        y_va = test_full[f"y{H}"].astype(np.int64)
        sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)

        for s in seeds:
            done += 1
            cfg = SEED_CONFIGS[s]
            print(
                f"\n[{done}/{n_total}] training final h={H} variant={args.variant} seed={s} cfg={cfg}",
                flush=True,
            )
            progress("training_final", horizon=H, seed=s, variant=args.variant,
                     done=done, total=n_total)

            seed_rng = np.random.default_rng(s * 7919 + 1)
            X_tr, y_tr, sw_tr = build_train_for_variant(
                args.variant, X_tr_orig, y_tr_orig,
                feat_std, feat_mean,
                seed_rng,
                aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
            )
            print(
                f"  n_tr_orig={len(X_tr_orig):,} n_tr_used={len(X_tr):,} n_va(test)={len(X_va):,}",
                flush=True,
            )

            dtrain = lgb.Dataset(
                X_tr, label=y_tr, weight=sw_tr,
                feature_name=feat_names, free_raw_data=False,
            )
            dval = lgb.Dataset(
                X_va, label=y_va, weight=sw_va,
                feature_name=feat_names, reference=dtrain, free_raw_data=False,
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
                "seed": s,
                "feature_fraction_seed": s + 1,
                "bagging_seed": s + 2,
                "data_random_seed": s + 3,
                "verbose": -1,
            }
            if args.use_gpu:
                params["device"] = "gpu"
                params["gpu_use_dp"] = False

            t_start = time.time()
            booster = lgb.train(
                params, dtrain,
                num_boost_round=args.num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                    lgb.log_evaluation(period=200),
                ],
            )
            train_time = time.time() - t_start
            print(f"  trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
                  flush=True)

            out_path = os.path.join(
                args.out_dir,
                f"final_model_h{H}_{args.variant}_seed{s}.txt",
            )
            booster.save_model(out_path, num_iteration=booster.best_iteration)
            print(f"  saved -> {out_path}", flush=True)

            summary.append({
                "horizon": H, "variant": args.variant, "seed": s,
                "best_iter": int(booster.best_iteration),
                "train_time_sec": float(train_time),
                "n_tr_orig": int(len(X_tr_orig)),
                "n_tr_used": int(len(X_tr)),
                "n_va": int(len(X_va)),
                "out_path": out_path,
            })

    summary_path = os.path.join(
        args.out_dir,
        f"train_final_{args.variant}_summary.json",
    )
    with open(summary_path, "w") as f:
        json.dump({
            "task": f"T35 train final {args.variant} (Scheme K 255-d)",
            "horizons": horizons, "seeds": seeds,
            "params": vars(args),
            "results": summary,
            "elapsed_sec": time.time() - t0,
        }, f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    progress("final_done", n_models=done, variant=args.variant)


if __name__ == "__main__":
    main()
