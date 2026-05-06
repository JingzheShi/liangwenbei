"""T27: LOSO LightGBM training with aug_a augment for 5 diverse seeds (h_60).

Combines T26 aug_a + T25 SEED_CONFIGS to enable a 5-seed OOF ensemble for h_60.

For each (seed, held_sym):
    train: (sym != held, dates 0..79)   + aug_a augment
    val:   (sym != held, dates 80..95)  -- early stopping (NO augment)
    test:  (sym == held, dates 96..119) -- held-out sym (NO augment)

Per-seed hyperparams from T25 (diverse for ensembling). aug_a is the same
per-(sample, feature) random scale [0.8, 1.2] used in T26.

Outputs (per (seed, held)):
    loso_pred_h60_aug_a_seed{S}_held{K}.parquet
    loso_model_h60_aug_a_seed{S}_held{K}.txt
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
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import (  # noqa: E402
    build_train_for_variant,
    class_balanced_weight,
    compute_feat_stats,
)

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3

# Match T25 SEED_CONFIGS exactly.
SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}


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


def evaluate(name, preds, y, mp_t, mp_th, fee_rate=0.0001):
    m = _per_horizon_metrics(preds, y, mp_t, mp_th, fee_rate=fee_rate)
    m["pred_distribution"] = dict(m["pred_distribution"])
    m["label_distribution"] = dict(m["label_distribution"])
    print(f"  --- {name} ---", flush=True)
    for k in ("accuracy", "cum_pnl", "single_pnl", "n_predictions_active", "f0_5_macro"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"    {k:25s}: {v:+.6g}", flush=True)
        else:
            print(f"    {k:25s}: {v}", flush=True)
    return m


def train_one(H, seed, held, train_full, val_full, test_full,
              feat_names, args, cfg):
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr_o = train_full["X"][m_tr][:, :-N_DROP_TAIL].astype(np.float32)
    y_tr_o = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL].astype(np.float32)
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL].astype(np.float32)
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[f"mp_t{H}"][m_te]

    feat_mean, feat_std = compute_feat_stats(X_tr_o)

    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
    X_tr, y_tr, sw_tr = build_train_for_variant(
        "aug_a", X_tr_o, y_tr_o, feat_std, feat_mean,
        fold_rng, aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)
    print(
        f"    n_train_orig={len(X_tr_o):,} n_train_used={len(X_tr):,} "
        f"n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]}",
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
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
    }
    if args.use_gpu:
        params.update({"device": "gpu", "gpu_use_dp": False})

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
    print(
        f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
        flush=True,
    )

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(
        f"HELD-OUT TEST h={H} seed={seed} held={held}",
        te_pred, y_te, mp_t_te, mp_th_te,
    )

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"][m_te].astype(np.int8),
        "date": test_full["date"][m_te].astype(np.int16),
        "session": np.array(
            [sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object
        ),
        "t": test_full["t"][m_te].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        "pred_label": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(
        HERE, f"loso_pred_h{H}_aug_a_seed{seed}_held{held}.parquet"
    )
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(
        HERE, f"loso_model_h{H}_aug_a_seed{seed}_held{held}.txt"
    )
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "horizon": H, "seed": seed, "held_out_sym": held,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "held_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_single_pnl": float(te_metrics["single_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seeds", default="1,7,13,100",
                    help="comma list (seed 42 already trained in T26)")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = ap.parse_args()

    H = args.horizon
    target_syms = [int(x) for x in args.syms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"unknown seed {s}, valid={list(SEED_CONFIGS)}")

    print(
        f"=== T27 LOSO aug_a 5-seed h={H} seeds={seeds} syms={target_syms} ===",
        flush=True,
    )

    progress("loading_caches", horizon=H, seeds=seeds)
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
    assert not leak, f"FORBIDDEN features: {leak}"

    all_results = []
    n_total = len(seeds) * len(target_syms)
    done = 0
    for s in seeds:
        cfg = SEED_CONFIGS[s]
        print(f"\n{'='*78}\n=== SEED {s} cfg={cfg} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            tag = f"h{H}_seed{s}_held{held}"
            print(f"\n  [{done}/{n_total}] fold {tag}", flush=True)
            progress("training", horizon=H, seed=s, held=held,
                     done=done, total=n_total)
            r = train_one(H, s, held, train_full, val_full, test_full,
                          feat_names, args, cfg)
            all_results.append(r)

    # Aggregate per seed
    df = pd.DataFrame(all_results)
    print(f"\n{'='*78}\n=== AGGREGATE h={H} ===\n{'='*78}", flush=True)
    agg = []
    for s in seeds:
        sub = df[df.seed == s]
        cums = sub["held_cum_pnl"].values
        rec = {
            "seed": s,
            "n_folds": int(len(cums)),
            "cum_pnl_per_fold": cums.tolist(),
            "cum_pnl_sum": float(cums.sum()),
            "cum_pnl_mean": float(cums.mean()) if len(cums) else 0.0,
            "cum_pnl_std": float(cums.std(ddof=0)) if len(cums) else 0.0,
            "n_pos_folds": int((cums > 0).sum()),
        }
        agg.append(rec)
        print(
            f"  seed{s}: sum={rec['cum_pnl_sum']:+.4f} "
            f"mean={rec['cum_pnl_mean']:+.4f} std={rec['cum_pnl_std']:.4f} "
            f"pos={rec['n_pos_folds']}/{rec['n_folds']} "
            f"per_fold={[f'{x:+.3f}' for x in cums]}",
            flush=True,
        )

    summary_path = os.path.join(HERE, "loso_5seed_aug_a_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json({
            "task": "T27 LOSO aug_a 5-seed h=60 (raw argmax)",
            "horizon": H, "seeds": seeds,
            "params": vars(args),
            "results": all_results,
            "aggregate": agg,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    progress("loso_done", n_runs=len(all_results), seeds=seeds)


if __name__ == "__main__":
    main()
