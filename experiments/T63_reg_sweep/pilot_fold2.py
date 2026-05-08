"""T63 Step 2: 5 strong-reg configs pilot on fold 2 (sym=2 holdout) at seed=42.

Same data setup as T53 LOSO (340-d after drop-fail+drop-tail) + aug_a + GPU LightGBM.
Each R config gets a single training; reports test cum_pnl on sym=2.
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
from build_aug import build_train_for_variant, class_balanced_weight  # noqa: E402

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
NUM_CLASS = 3

# Strong regularization variants (delta from T59 baseline seed=42 of feat=0.8/bag=0.8/leaves=127/l2=1)
REG_CONFIGS = {
    "R1": {
        "feature_fraction": 0.5, "bagging_fraction": 0.6, "num_leaves": 63,
        "lambda_l2": 5.0, "min_data_in_leaf": 200,
    },
    "R2": {
        "feature_fraction": 0.4, "bagging_fraction": 0.5, "num_leaves": 63,
        "lambda_l2": 10.0, "min_data_in_leaf": 300,
    },
    "R3": {
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "num_leaves": 63,
        "lambda_l1": 1.0, "lambda_l2": 5.0, "min_data_in_leaf": 300,
    },
    "R4": {
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "num_leaves": 15,
        "lambda_l2": 3.0, "min_data_in_leaf": 200, "max_depth": 4,
    },
    "R5": {
        "feature_fraction": 0.5, "bagging_fraction": 0.7, "num_leaves": 127,
        "lambda_l2": 1.0, "min_data_in_leaf": 100, "extra_trees": True,
    },
}


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running", "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"schemeN_{split}.npz")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--held", type=int, default=2)
    ap.add_argument("--variant", default="aug_a")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-drop-tail", type=int, default=3)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--configs", default="R1,R2,R3,R4,R5")
    args = ap.parse_args()

    H = args.horizon
    held = args.held
    cfgs_to_run = [c.strip() for c in args.configs.split(",")]
    print(f"=== T63 PILOT h={H} held={held} configs={cfgs_to_run} seed={args.seed} ===", flush=True)

    progress("loading_caches", held=held)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    base_dim = 226
    total_dim = train_full["X"].shape[1]
    n_extras = total_dim - base_dim
    extras_names_path = os.path.join(CACHE_DIR, "schemeN_extra_feat_names.txt")
    with open(extras_names_path) as f:
        extra_names = [line.strip() for line in f]
    assert len(extra_names) == n_extras

    report_path = os.path.join(HERE, "sym_invariance_report.json")
    with open(report_path) as f:
        report = json.load(f)
    fail_names = report["summary"]["fail_names"]
    fail_local = [extra_names.index(n) for n in fail_names]
    fail_global = [base_dim + i for i in fail_local]
    drop_set = set(fail_global)
    for i in range(args.n_drop_tail):
        drop_set.add(base_dim - 1 - i)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_set], dtype=np.int64)
    feat_dim = len(keep_idx)
    print(f"  base_dim={base_dim} n_extras={n_extras} feat_dim={feat_dim}", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeN_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN feature leak: {leak}"
    print(f"  no-leak check OK (feat_dim={feat_dim})", flush=True)

    X_tr_full = train_full["X"][:, keep_idx].astype(np.float32)
    X_va_full = val_full["X"][:, keep_idx].astype(np.float32)
    X_te_full = test_full["X"][:, keep_idx].astype(np.float32)
    y_tr_full = train_full[f"y{H}"]
    y_va_full = val_full[f"y{H}"]
    y_te_full = test_full[f"y{H}"]
    sym_tr = train_full["sym"]
    sym_va = val_full["sym"]
    sym_te = test_full["sym"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    mp_t_va = val_full["mp_t"]
    mp_th_va = val_full[f"mp_t{H}"]

    m_tr = sym_tr != held
    m_va = sym_va != held
    m_te = sym_te == held
    X_tr_orig = X_tr_full[m_tr]
    y_tr_orig = y_tr_full[m_tr].astype(np.int64)
    X_va = X_va_full[m_va]
    y_va = y_va_full[m_va].astype(np.int64)
    X_te = X_te_full[m_te]
    y_te = y_te_full[m_te].astype(np.int64)
    mp_t_te_k = mp_t_te[m_te]
    mp_th_te_k = mp_th_te[m_te]
    mp_t_va_k = mp_t_va[m_va]
    mp_th_va_k = mp_th_va[m_va]

    print(f"  n_tr={len(X_tr_orig):,} n_va={len(X_va):,} n_te={len(X_te):,}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T63-PILOT-h{H}-held{held}-seed{args.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T63 reg sweep pilot",
                    "horizon": H, "held": held, "seed": args.seed,
                    "n_features": feat_dim,
                },
                tags=["T63", "PILOT", f"held{held}", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    feat_mean, feat_std = None, None
    seed = args.seed

    seed_rng = np.random.default_rng(seed * 7919 + 1)
    X_tr_aug, y_tr_aug, sw_tr = build_train_for_variant(
        args.variant, X_tr_orig, y_tr_orig,
        feat_std, feat_mean, seed_rng,
        aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_aug):,}", flush=True)

    pilot_results = {}

    for cfg_name in cfgs_to_run:
        rcfg = REG_CONFIGS[cfg_name]
        print(f"\n{'='*78}\n=== PILOT cfg={cfg_name} {rcfg} ===\n{'='*78}", flush=True)
        progress("training", cfg=cfg_name, held=held)

        dtrain = lgb.Dataset(
            X_tr_aug, label=y_tr_aug, weight=sw_tr,
            feature_name=feat_names, free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_va, label=y_va, weight=sw_va,
            feature_name=feat_names, reference=dtrain, free_raw_data=False,
        )

        params = {
            "objective": "multiclass",
            "num_class": NUM_CLASS,
            "metric": "multi_logloss",
            "learning_rate": args.learning_rate,
            "bagging_freq": args.bagging_freq,
            "num_threads": args.num_threads,
            "seed": seed,
            "feature_fraction_seed": seed + 1,
            "bagging_seed": seed + 2,
            "data_random_seed": seed + 3,
            "verbose": -1,
            "device": "gpu",
            "gpu_use_dp": False,
        }
        # Apply reg config (overwrites defaults)
        for k, v in rcfg.items():
            params[k] = v

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
        best_iter = booster.best_iteration
        print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

        # val metric (early-stop set)
        va_prob = predict_proba(booster, X_va)
        va_pred = va_prob.argmax(axis=1).astype(np.int8)
        va_metrics = evaluate(
            f"VAL (early-stop) cfg={cfg_name}", va_pred, y_va, mp_t_va_k, mp_th_va_k,
        )
        # test fold-2 metric
        te_prob = predict_proba(booster, X_te)
        te_pred = te_prob.argmax(axis=1).astype(np.int8)
        te_metrics = evaluate(
            f"TEST (sym={held}) cfg={cfg_name}", te_pred, y_te, mp_t_te_k, mp_th_te_k,
        )

        pilot_results[cfg_name] = {
            "config": rcfg,
            "best_iter": int(best_iter),
            "train_time_sec": float(train_time),
            "val": va_metrics,
            "test_held2": te_metrics,
        }
        if use_wandb:
            wandb.log({
                f"{cfg_name}/best_iter": int(best_iter),
                f"{cfg_name}/test_held2_cum_pnl": te_metrics["cum_pnl"],
                f"{cfg_name}/test_held2_acc": te_metrics["accuracy"],
                f"{cfg_name}/val_cum_pnl": va_metrics["cum_pnl"],
                f"{cfg_name}/test_held2_n_active": te_metrics["n_predictions_active"],
            })

    # rank
    ranking = sorted(
        pilot_results.items(),
        key=lambda kv: kv[1]["test_held2"]["cum_pnl"],
        reverse=True,
    )
    print(f"\n{'='*78}\n=== PILOT RANKING (test sym={held} cum_pnl) ===\n{'='*78}", flush=True)
    for name, r in ranking:
        print(
            f"  {name}: cum_pnl={r['test_held2']['cum_pnl']:+.4f}  "
            f"acc={r['test_held2']['accuracy']:.4f}  "
            f"n_active={r['test_held2']['n_predictions_active']}  "
            f"best_iter={r['best_iter']}",
            flush=True,
        )

    summary = {
        "task": f"T63 pilot on fold {held} seed={seed}",
        "params": vars(args),
        "horizon": H,
        "configs": REG_CONFIGS,
        "results": pilot_results,
        "ranking": [
            {
                "name": n,
                "cum_pnl_held2": r["test_held2"]["cum_pnl"],
                "accuracy": r["test_held2"]["accuracy"],
                "n_active": r["test_held2"]["n_predictions_active"],
                "best_iter": r["best_iter"],
            }
            for n, r in ranking
        ],
        "winner": ranking[0][0] if ranking else None,
    }
    out_path = os.path.join(HERE, f"pilot_h{H}_held{held}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.finish()
    progress("pilot_done", winner=summary["winner"])


if __name__ == "__main__":
    main()
