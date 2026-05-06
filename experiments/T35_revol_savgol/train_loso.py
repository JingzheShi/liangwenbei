"""T35: Scheme K LOSO LightGBM training (h_60 by default).

Scheme K = (Scheme C 226d - 3 time tail = 223 base) + ReVol (12) + SG (20)
        = 255 effective features for training.

Loads:
    schemeC_{split}.npz from T5b cache       -- X (N, 226), labels, midprices
    schemeK_extra_{split}.npz from T35 cache -- X_extra (N, 32) aligned

For each held_sym in 0..4:
    train: (sym != held, dates 0..79)   [+ optional aug_a]
    val:   (sym != held, dates 80..95)
    test:  (sym == held, dates 96..119)

Outputs:
    loso_pred_h{H}_{tag}_held{K}.parquet
    loso_model_h{H}_{tag}_held{K}.txt
    loso_summary_{tag}.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import (  # noqa: E402
    build_train_for_variant,
    class_balanced_weight,
    compute_feat_stats,
)

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
T35_CACHE = os.path.join(HERE, "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # drop time_minutes, time_session_progress, time_is_pm

# Same SEED_CONFIGS as T27 — keeps comparison apples-to-apples.
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


def load_combined(split: str) -> dict:
    """Concatenate Scheme C base (drop time tail) + Scheme K extras."""
    p_c = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    p_k = os.path.join(T35_CACHE, f"schemeK_extra_{split}.npz")
    print(f"  loading {p_c}", flush=True)
    dc = np.load(p_c)
    print(f"  loading {p_k}", flush=True)
    dk = np.load(p_k)
    # Sanity: alignment was verified at build time, but double-check.
    for k in ("sym", "date", "sess_idx", "t"):
        assert np.array_equal(dc[k], dk[k]), f"alignment broken on {k}"
    X_base = dc["X"][:, :-N_DROP_TAIL].astype(np.float32, copy=False)  # 223
    X_extra = dk["X_extra"].astype(np.float32, copy=False)             # 32
    X = np.concatenate([X_base, X_extra], axis=1)                       # 255
    out = {k: dc[k] for k in dc.files if k != "X"}
    out["X"] = X
    print(f"  combined X shape: {X.shape}", flush=True)
    return out


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000) -> np.ndarray:
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


def train_one(H, seed, held, variant, train_full, val_full, test_full,
              feat_names, args, cfg):
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr_o = train_full["X"][m_tr].astype(np.float32, copy=False)
    y_tr_o = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va].astype(np.float32, copy=False)
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te].astype(np.float32, copy=False)
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[f"mp_t{H}"][m_te]

    feat_mean, feat_std = compute_feat_stats(X_tr_o)

    if variant == "baseline":
        X_tr, y_tr, sw_tr = X_tr_o, y_tr_o, class_balanced_weight(y_tr_o, NUM_CLASS)
    else:
        fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
        X_tr, y_tr, sw_tr = build_train_for_variant(
            variant, X_tr_o, y_tr_o, feat_std, feat_mean,
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
        f"HELD-OUT TEST h={H} variant={variant} seed={seed} held={held}",
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
        HERE, f"loso_pred_h{H}_{variant}_seed{seed}_held{held}.parquet"
    )
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(
        HERE, f"loso_model_h{H}_{variant}_seed{seed}_held{held}.txt"
    )
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "horizon": H, "variant": variant, "seed": seed, "held_out_sym": held,
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
    ap.add_argument("--variant", default="baseline",
                    choices=("baseline", "aug_a", "aug_b", "aug_c", "aug_abc"))
    ap.add_argument("--seeds", default="42")
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
    ap.add_argument("--tag", default=None,
                    help="optional summary file tag, defaults to f'h{H}_{variant}'")
    ap.add_argument("--wandb", action="store_true", default=False)
    ap.add_argument("--wandb-project", default="liangwenbei")
    args = ap.parse_args()

    H = args.horizon
    target_syms = [int(x) for x in args.syms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"unknown seed {s}, valid={list(SEED_CONFIGS)}")

    tag = args.tag or f"h{H}_{args.variant}"
    print(
        f"=== T35 Scheme K LOSO h={H} variant={args.variant} seeds={seeds} syms={target_syms} ===",
        flush=True,
    )

    # Build feat_names
    base_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(base_names_path) as f:
        base_names = [line.strip() for line in f]
    extra_names_path = os.path.join(T35_CACHE, "schemeK_extra_feat_names.txt")
    with open(extra_names_path) as f:
        extra_names = [line.strip() for line in f]
    feat_names = base_names + extra_names
    print(f"feat_names: {len(feat_names)} (base {len(base_names)} + extra {len(extra_names)})",
          flush=True)
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features: {leak}"

    progress("loading_caches", horizon=H, variant=args.variant, seeds=seeds)
    t0 = time.time()
    train_full = load_combined("train")
    val_full = load_combined("val")
    test_full = load_combined("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )
    assert train_full["X"].shape[1] == len(feat_names), \
        f"feat dim mismatch: X {train_full['X'].shape[1]} vs names {len(feat_names)}"

    all_results = []
    n_total = len(seeds) * len(target_syms)
    done = 0
    for s in seeds:
        cfg = SEED_CONFIGS[s]
        print(f"\n{'='*78}\n=== SEED {s} cfg={cfg} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            tagf = f"h{H}_{args.variant}_seed{s}_held{held}"
            print(f"\n  [{done}/{n_total}] fold {tagf}", flush=True)
            progress("training", horizon=H, variant=args.variant, seed=s, held=held,
                     done=done, total=n_total)
            r = train_one(H, s, held, args.variant,
                          train_full, val_full, test_full,
                          feat_names, args, cfg)
            all_results.append(r)

    # Aggregate per seed
    df = pd.DataFrame(all_results)
    print(f"\n{'='*78}\n=== AGGREGATE h={H} variant={args.variant} ===\n{'='*78}", flush=True)
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

    if args.wandb and HAS_WANDB:
        try:
            run = wandb.init(
                project=args.wandb_project,
                entity="cjxh21-Tsinghua University",
                name=f"T35-{tag}",
                config=vars(args),
            )
            for r in all_results:
                wandb.log({
                    f"held{r['held_out_sym']}_pnl": r["held_cum_pnl"],
                    f"held{r['held_out_sym']}_acc": r["held_acc"],
                })
            wandb.summary["loso_sum_pnl"] = sum(r["held_cum_pnl"] for r in all_results)
            wandb.summary["loso_mean_pnl"] = sum(r["held_cum_pnl"] for r in all_results) / max(len(all_results), 1)
            wandb.summary["n_features"] = len(feat_names)
            wandb.finish()
        except Exception as e:
            print(f"  WANDB FAILED: {e}", flush=True)

    summary_path = os.path.join(HERE, f"loso_summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json({
            "task": f"T35 Scheme K LOSO {tag} (raw argmax)",
            "horizon": H, "variant": args.variant, "seeds": seeds,
            "params": vars(args),
            "n_features": len(feat_names),
            "results": all_results,
            "aggregate": agg,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    progress("loso_done", n_runs=len(all_results), seeds=seeds, variant=args.variant)


if __name__ == "__main__":
    main()
