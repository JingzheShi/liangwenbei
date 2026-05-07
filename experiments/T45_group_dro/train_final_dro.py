"""T45 Group DRO — full-train (train+val merged) for SUBMISSION.

Mirrors T27 train_final_aug_a.py structure:
  train: train + val concatenated (dates 0..95, all 5 syms)
  val:   test                     (dates 96..119, all 5 syms) — early stopping
  +     aug_a (per-(sample,feature) uniform[0.8,1.2]) doubled concat

Adds Group DRO callback: every K rounds, compute per-sym CE on training,
softmax(loss/tau) -> per-sym sample weights (mean=1.0).

Output: final_dro_model_h60_seed{S}.txt (5 models for ensemble submission).
Drops trailing 3 time_* cols (-> 223-d) so models are drop-in compatible
with iter_006 Predictor template.
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

# Reuse the DRO callback / aug helper from train_loso.py
sys.path.insert(0, HERE)
from train_loso import GroupDROCallback, aug_uniform_concat, SEED_CONFIGS, progress  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
H = 60
N_DROP_TAIL = 3  # drop time_* to be 223-d


def load_split(split):
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    return dict(np.load(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--tag", default="finaldro")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--first-update-at", type=int, default=20, dest="first_update_at")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"unknown seed {s}, valid={list(SEED_CONFIGS)}")

    print(f"=== T45 train_final_dro h={H} tag={args.tag} tau={args.tau} K={args.K} "
          f"seeds={seeds} aug=[{args.aug_lo},{args.aug_hi}] ===", flush=True)

    progress("loading_caches", tag=args.tag, tau=args.tau, seeds=seeds)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time()-t0:.1f}s; train={train_full['X'].shape} "
        f"val={val_full['X'].shape} test={test_full['X'].shape}", flush=True,
    )

    # Concat train+val for full-train; test is early-stop val
    X_tr_orig = np.concatenate(
        [train_full["X"][:, :-N_DROP_TAIL], val_full["X"][:, :-N_DROP_TAIL]], axis=0
    ).astype(np.float32)
    y_tr_orig = np.concatenate(
        [train_full[f"y{H}"], val_full[f"y{H}"]], axis=0
    ).astype(np.int64)
    sym_tr_orig = np.concatenate(
        [train_full["sym"], val_full["sym"]], axis=0
    ).astype(np.int64)

    X_va = test_full["X"][:, :-N_DROP_TAIL].astype(np.float32)
    y_va = test_full[f"y{H}"].astype(np.int64)
    sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    if os.path.exists(feat_names_path):
        with open(feat_names_path) as f:
            feat_names = [ln.strip() for ln in f if ln.strip()]
    else:
        # fall back to schemeC_feat_names.txt minus last 3
        with open(os.path.join(T5B_CACHE, "schemeC_feat_names.txt")) as f:
            feat_names = [ln.strip() for ln in f if ln.strip()][:-N_DROP_TAIL]
    assert len(feat_names) == X_tr_orig.shape[1], (len(feat_names), X_tr_orig.shape[1])
    forbidden = {"date", "sym"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features: {leak}"
    print(f"  feat_dim={X_tr_orig.shape[1]} (last 5: {feat_names[-5:]})", flush=True)
    print(f"  X_tr_orig={X_tr_orig.shape}  X_va={X_va.shape}", flush=True)

    summary = []
    for i, s in enumerate(seeds):
        cfg = SEED_CONFIGS[s]
        print(f"\n[{i+1}/{len(seeds)}] full-train DRO seed={s} cfg={cfg}", flush=True)
        progress("training_final", seed=s, done=i+1, total=len(seeds))

        seed_rng = np.random.default_rng(s * 7919 + 1)
        X_tr, y_tr, sym_tr_dbl = aug_uniform_concat(
            X_tr_orig, y_tr_orig, sym_tr_orig, args.aug_lo, args.aug_hi, seed_rng,
        )
        sw_tr = np.ones(len(X_tr), dtype=np.float32)
        print(
            f"  n_tr_orig={len(X_tr_orig):,} n_tr_used={len(X_tr):,} n_va={len(X_va):,}",
            flush=True,
        )
        sym_counts = {int(t): int((sym_tr_dbl == t).sum()) for t in range(5)}
        print(f"  sym counts (post-aug): {sym_counts}", flush=True)

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
            params.update({"device": "gpu", "gpu_use_dp": False})

        log_lines = []
        dro_cb = GroupDROCallback(
            X_tr=X_tr, y_tr=y_tr, sym_tr=sym_tr_dbl, dtrain=dtrain,
            K=args.K, tau=args.tau, first_update_at=args.first_update_at,
            num_class=NUM_CLASS, log_lines=log_lines,
        )

        t_start = time.time()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=args.num_boost_round,
            valid_sets=[dval], valid_names=["val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                dro_cb,
                lgb.log_evaluation(period=100),
            ],
        )
        train_time = time.time() - t_start
        print(
            f"  trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
            flush=True,
        )

        out_path = os.path.join(HERE, f"{args.tag}_model_h{H}_seed{s}.txt")
        booster.save_model(out_path, num_iteration=booster.best_iteration)
        print(f"  saved -> {out_path}", flush=True)

        # Save DRO history
        hist_path = os.path.join(HERE, f"{args.tag}_drohist_seed{s}.json")
        with open(hist_path, "w") as f:
            json.dump({
                "tau": args.tau, "K": args.K,
                "first_update_at": args.first_update_at,
                "history": dro_cb.history,
                "log_lines": log_lines,
            }, f, indent=2)

        summary.append({
            "seed": s, "best_iter": int(booster.best_iteration),
            "train_time_sec": float(train_time),
            "n_tr_orig": int(len(X_tr_orig)),
            "n_tr_used": int(len(X_tr)),
            "n_dro_updates": len(dro_cb.history),
            "out_path": out_path,
        })

    summary_path = os.path.join(HERE, f"{args.tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "task": f"T45 train_final_dro h={H} tau={args.tau}",
            "seeds": seeds, "params": vars(args),
            "results": summary,
            "elapsed_sec": time.time() - t0,
        }, f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)
    progress("final_done", n_models=len(seeds))


if __name__ == "__main__":
    main()
