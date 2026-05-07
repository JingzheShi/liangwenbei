"""T42: train two One-vs-Rest binary LightGBM heads per LOSO fold (h=60).

Hypothesis: the multiclass softmax (iter_005b) under-confidently predicts the
extreme classes (label 0 / 2) because the centre class (label 1) "squeezes"
their probability mass — recall on the extremes is poor. A dedicated binary
head should produce sharper P(label==2) and P(label==0).

Per fold (held=0..4) we train:

    * B_up : label==2 vs not (label in {0,1})  -> P(label==2)
    * B_dn : label==0 vs not (label in {1,2})  -> P(label==0)

Single seed (=42), 223-dim Scheme C features (matches iter_005b — drop the
last 3 time-derived columns) and aug_a [0.80, 1.20] (matches iter_005b /
T26).  GPU LightGBM, h=60 only.

Outputs (per fold k, head h ∈ {B_up, B_dn}):
    {head}_seed42_held{k}.parquet     -- OOF prob on the held-out sym
    {head}_seed42_held{k}.txt         -- saved booster

Compliance with CRITICAL_CONSTRAINTS.md §1:
  - sym / date never in feature vector (we drop the last 3 features which are
    time-derived; sym/date were never in the X matrix to begin with)
  - Augmentation only at training time, never on val/test
  - Models are sym-agnostic
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
from build_aug import compute_feat_stats, aug_a_scale  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # drop the trailing 3 time-derived columns (matches iter_005b)


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split: str) -> dict:
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def class_balanced_weight_binary(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (2.0 * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def build_train_aug_a_binary(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    rng: np.random.Generator,
    aug_ratio: float = 1.0,
):
    """aug_a only — per-(sample, feature) scale [0.8, 1.2], concat with orig.

    Mirrors build_aug.build_train_for_variant("aug_a", ...) but with
    binary class-balanced weights (k=2).
    """
    n_aug = int(round(aug_ratio * len(X_orig)))
    if n_aug <= 0:
        w = class_balanced_weight_binary(y_orig)
        return X_orig, y_orig.astype(np.int64), w
    if n_aug == len(X_orig):
        idx = np.arange(len(X_orig))
    else:
        idx = rng.integers(0, len(X_orig), size=n_aug)
    X_src = X_orig[idx]
    y_src = y_orig[idx]
    X_aug = aug_a_scale(X_src, rng, lo=0.80, hi=1.20)
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_src], axis=0).astype(np.int64)
    w_merged = class_balanced_weight_binary(y_merged)
    return X_merged, y_merged, w_merged


def train_one_head(
    head: str,
    held: int,
    H: int,
    train_full: dict,
    val_full: dict,
    test_full: dict,
    feat_names: list,
    args,
):
    """Train one binary head on one fold."""
    assert head in {"B_up", "B_dn"}
    pos_class = 2 if head == "B_up" else 0

    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr_o = train_full["X"][m_tr][:, :-N_DROP_TAIL].astype(np.float32)
    y_tr_o = (train_full[f"y{H}"][m_tr] == pos_class).astype(np.int8)
    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL].astype(np.float32)
    y_va = (val_full[f"y{H}"][m_va] == pos_class).astype(np.int8)
    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL].astype(np.float32)
    y_te_mc = test_full[f"y{H}"][m_te].astype(np.int64)  # keep multiclass label for OOF parquet
    y_te_bin = (y_te_mc == pos_class).astype(np.int8)

    # Feature stats (used by aug if needed, kept for parity with multiclass)
    _ = compute_feat_stats(X_tr_o)

    fold_rng = np.random.default_rng(args.seed * 7919 + held * 17 + (1 if head == "B_up" else 2))
    X_tr, y_tr, sw_tr = build_train_aug_a_binary(
        X_tr_o, y_tr_o, fold_rng, aug_ratio=args.aug_ratio,
    )
    sw_va = class_balanced_weight_binary(y_va)
    pos_rate_tr = float((y_tr == 1).mean())
    pos_rate_va = float((y_va == 1).mean())
    pos_rate_te = float((y_te_bin == 1).mean())
    print(
        f"    [{head} held={held}] n_train_orig={len(X_tr_o):,} "
        f"n_train_used={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,} "
        f"feat_dim={X_tr.shape[1]} pos_rate(tr/va/te)="
        f"{pos_rate_tr:.4f}/{pos_rate_va:.4f}/{pos_rate_te:.4f}",
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
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l2": args.lambda_l2,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "feature_fraction_seed": args.seed + 1,
        "bagging_seed": args.seed + 2,
        "data_random_seed": args.seed + 3,
        "verbose": -1,
    }
    if args.use_gpu:
        params.update({"device": "gpu", "gpu_use_dp": False})

    t0 = time.time()
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
    train_time = time.time() - t0

    te_prob = predict_proba(booster, X_te)  # P(positive)
    print(
        f"    [{head} held={held}] trained in {train_time:.1f}s "
        f"best_iter={booster.best_iteration} test_p_mean={te_prob.mean():.4f} "
        f"test_p_quantiles={[float(np.quantile(te_prob, q)) for q in (0.5, 0.9, 0.99)]}",
        flush=True,
    )

    sess_map = {0: "am", 1: "pm"}
    df_oof = pd.DataFrame({
        "sym": test_full["sym"][m_te].astype(np.int8),
        "date": test_full["date"][m_te].astype(np.int16),
        "session": np.array([sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object),
        "t": test_full["t"][m_te].astype(np.int16),
        "true_label": y_te_mc.astype(np.int8),  # original multiclass label
        "true_pos": y_te_bin.astype(np.int8),
        "prob_pos": te_prob.astype(np.float32),
    })
    out_parquet = os.path.join(HERE, f"{head}_seed{args.seed}_held{held}.parquet")
    df_oof.to_parquet(out_parquet, index=False)

    out_model = os.path.join(HERE, f"{head}_seed{args.seed}_held{held}.txt")
    booster.save_model(out_model, num_iteration=booster.best_iteration)

    return {
        "head": head, "held": held, "horizon": H,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "pos_rate_train": pos_rate_tr,
        "pos_rate_val": pos_rate_va,
        "pos_rate_test": pos_rate_te,
        "test_prob_mean": float(te_prob.mean()),
        "test_prob_q50": float(np.quantile(te_prob, 0.5)),
        "test_prob_q90": float(np.quantile(te_prob, 0.9)),
        "test_prob_q99": float(np.quantile(te_prob, 0.99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--heads", default="B_up,B_dn",
                    help="comma-separated heads to train")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    # iter_005b seed42 hyperparams (matches T26 baseline)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = ap.parse_args()

    H = args.horizon
    target_syms = [int(x) for x in args.syms.split(",")]
    heads = [h.strip() for h in args.heads.split(",")]
    for h in heads:
        if h not in {"B_up", "B_dn"}:
            sys.exit(f"unknown head {h}")

    print(
        f"=== T42 OvA binary LightGBM h={H} seed={args.seed} "
        f"heads={heads} syms={target_syms} ===",
        flush=True,
    )

    progress("loading_caches", horizon=H, seed=args.seed)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    assert len(feat_names) == train_full["X"].shape[1] - N_DROP_TAIL
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features in feature list: {leak}"

    all_runs = []
    n_total = len(heads) * len(target_syms)
    done = 0
    for head in heads:
        print(f"\n{'='*78}\n=== HEAD {head} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            print(
                f"\n  [{done}/{n_total}] head={head} held={held}",
                flush=True,
            )
            progress("training", head=head, held=held,
                     done=done, total=n_total)
            r = train_one_head(head, held, H, train_full, val_full, test_full,
                               feat_names, args)
            all_runs.append(r)

    summary_path = os.path.join(HERE, f"train_binary_summary_seed{args.seed}.json")
    with open(summary_path, "w") as f:
        json.dump({
            "task": "T42 OvA binary LightGBM training",
            "horizon": H, "seed": args.seed,
            "heads": heads, "syms": target_syms,
            "params": vars(args),
            "runs": all_runs,
            "elapsed_sec": time.time() - t0,
        }, f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)
    progress("training_done", n_runs=len(all_runs))


if __name__ == "__main__":
    main()
