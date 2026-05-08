"""T63 Step 3: 5-seed full-sym (T59 style) training with winner reg config.

Same data layout as T59:
  - Train on syms 0-4 (date 0-79) + aug_a
  - Early-stop on val syms 0-4 (date 80-95)
  - Test on syms 0-4 (date 96-119) — full 442k

5 seeds use the SAME regularization config (the winner from pilot).
Saves predictions per seed; downstream de_eval averages across seeds.
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
SYMS = (0, 1, 2, 3, 4)


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
    ap.add_argument("--seeds", default="1,7,13,42,100")
    ap.add_argument("--variant", default="aug_a")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--n-drop-tail", type=int, default=3)
    ap.add_argument("--out-tag", default="winner")
    ap.add_argument("--no-wandb", action="store_true")
    # Winner reg config (set via CLI)
    ap.add_argument("--feature-fraction", type=float, required=True)
    ap.add_argument("--bagging-fraction", type=float, required=True)
    ap.add_argument("--num-leaves", type=int, required=True)
    ap.add_argument("--lambda-l1", type=float, default=0.0)
    ap.add_argument("--lambda-l2", type=float, default=0.0)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--extra-trees", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T63 STEP3 5SEED-WINNER h={H} seeds={seeds} variant={args.variant} ===", flush=True)
    print(f"  winner cfg: feat={args.feature_fraction} bag={args.bagging_fraction} "
          f"leaves={args.num_leaves} l1={args.lambda_l1} l2={args.lambda_l2} "
          f"min_leaf={args.min_data_in_leaf} max_depth={args.max_depth} extra_trees={args.extra_trees}",
          flush=True)

    progress("loading_caches", seeds=seeds)
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

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T63-STEP3-h{H}-{args.out_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T63 winner 5seed full-sym",
                    "horizon": H, "seeds": seeds,
                    "n_features": feat_dim,
                    "winner": vars(args),
                },
                tags=["T63", "STEP3", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    feat_mean, feat_std = None, None

    X_tr = train_full["X"][:, keep_idx].astype(np.float32)
    X_va = val_full["X"][:, keep_idx].astype(np.float32)
    X_te = test_full["X"][:, keep_idx].astype(np.float32)
    y_tr = train_full[f"y{H}"].astype(np.int64)
    y_va = val_full[f"y{H}"].astype(np.int64)
    y_te = test_full[f"y{H}"].astype(np.int64)
    sym_te = test_full["sym"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    mp_t_va = val_full["mp_t"]
    mp_th_va = val_full[f"mp_t{H}"]

    print(f"  n_train={len(X_tr):,}  n_val={len(X_va):,}  n_test={len(X_te):,}", flush=True)
    print(f"  test sym dist: {dict(zip(*np.unique(sym_te, return_counts=True)))}", flush=True)

    all_results = {}
    for seed in seeds:
        print(f"\n{'='*78}\n=== seed={seed} ===\n{'='*78}", flush=True)
        progress("training_seed", seed=seed)

        seed_rng = np.random.default_rng(seed * 7919 + 1)
        X_tr_aug, y_tr_aug, sw_tr = build_train_for_variant(
            args.variant, X_tr, y_tr, feat_std, feat_mean, seed_rng,
            aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
        )
        sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)
        print(f"  n_train_used={len(X_tr_aug):,}", flush=True)

        dtrain = lgb.Dataset(X_tr_aug, label=y_tr_aug, weight=sw_tr,
                             feature_name=feat_names, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                           feature_name=feat_names, reference=dtrain, free_raw_data=False)

        params = {
            "objective": "multiclass",
            "num_class": NUM_CLASS,
            "metric": "multi_logloss",
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": args.bagging_freq,
            "lambda_l1": args.lambda_l1,
            "lambda_l2": args.lambda_l2,
            "max_depth": args.max_depth,
            "extra_trees": args.extra_trees,
            "num_threads": args.num_threads,
            "seed": seed,
            "feature_fraction_seed": seed + 1,
            "bagging_seed": seed + 2,
            "data_random_seed": seed + 3,
            "verbose": -1,
            "device": "gpu",
            "gpu_use_dp": False,
        }

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
        print(f"  trained in {train_time:.1f}s, best_iter={booster.best_iteration}", flush=True)

        model_path = os.path.join(HERE, f"winner_model_h{H}_seed{seed}.txt")
        booster.save_model(model_path, num_iteration=booster.best_iteration)

        va_prob = predict_proba(booster, X_va)
        va_pred = va_prob.argmax(axis=1).astype(np.int8)
        va_metrics = evaluate(f"VAL seed={seed}", va_pred, y_va, mp_t_va, mp_th_va)
        te_prob = predict_proba(booster, X_te)
        te_pred = te_prob.argmax(axis=1).astype(np.int8)
        te_metrics = evaluate(f"TEST seed={seed}", te_pred, y_te, mp_t_te, mp_th_te)

        per_sym_metrics = {}
        for k in SYMS:
            m = sym_te == k
            if m.sum() == 0:
                continue
            mk = _per_horizon_metrics(te_pred[m], y_te[m], mp_t_te[m], mp_th_te[m], fee_rate=0.0001)
            per_sym_metrics[int(k)] = {
                "n": int(m.sum()),
                "accuracy": float(mk["accuracy"]),
                "cum_pnl": float(mk["cum_pnl"]),
                "single_pnl": float(mk["single_pnl"]),
                "n_active": int(mk["n_predictions_active"]),
                "f0_5_macro": float(mk["f0_5_macro"]),
            }
        loso_equiv_sum_argmax = sum(v["cum_pnl"] for v in per_sym_metrics.values())
        print(f"  per-sym argmax cum_pnl: {[round(per_sym_metrics[k]['cum_pnl'],3) for k in SYMS]}  sum={loso_equiv_sum_argmax:+.4f}", flush=True)

        sess_map = {0: "am", 1: "pm"}
        te_df = pd.DataFrame({
            "sym": test_full["sym"].astype(np.int8),
            "date": test_full["date"].astype(np.int16),
            "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
            "t": test_full["t"].astype(np.int16),
            "true_label": y_te.astype(np.int8),
            "pred_label": te_pred.astype(np.int8),
            "prob_0": te_prob[:, 0].astype(np.float32),
            "prob_1": te_prob[:, 1].astype(np.float32),
            "prob_2": te_prob[:, 2].astype(np.float32),
            "midprice_t": mp_t_te.astype(np.float32),
            "midprice_th": mp_th_te.astype(np.float32),
        })
        pred_path = os.path.join(HERE, f"winner_pred_h{H}_seed{seed}.parquet")
        te_df.to_parquet(pred_path, index=False)
        print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

        all_results[seed] = {
            "horizon": H, "seed": seed,
            "n_train_orig": int(len(X_tr)), "n_train_used": int(len(X_tr_aug)),
            "n_val": int(len(X_va)), "n_test": int(len(X_te)),
            "best_iter": int(booster.best_iteration),
            "train_time_sec": float(train_time),
            "val": va_metrics, "test_full": te_metrics,
            "per_sym_test_argmax": per_sym_metrics,
            "loso_equiv_sum_argmax": float(loso_equiv_sum_argmax),
        }
        if use_wandb:
            wandb.log({
                f"seed{seed}/best_iter": int(booster.best_iteration),
                f"seed{seed}/train_time_sec": float(train_time),
                f"seed{seed}/test_cum_pnl": te_metrics["cum_pnl"],
                f"seed{seed}/test_acc": te_metrics["accuracy"],
                f"seed{seed}/loso_equiv_sum": float(loso_equiv_sum_argmax),
            })

    print(f"\n{'='*78}\n=== Aggregate ===\n{'='*78}", flush=True)
    sums = []
    pnls = []
    for seed in seeds:
        r = all_results[seed]
        sums.append(r["loso_equiv_sum_argmax"])
        pnls.append(r["test_full"]["cum_pnl"])
        print(f"seed={seed}: test_full_cum_pnl={r['test_full']['cum_pnl']:+.4f}  "
              f"loso_equiv_sum_argmax={r['loso_equiv_sum_argmax']:+.4f}  "
              f"per_sym={[round(r['per_sym_test_argmax'][k]['cum_pnl'],3) for k in SYMS]}", flush=True)

    summary = {
        "task": "T63 winner 5-seed full-sym training",
        "params": vars(args),
        "horizon": H,
        "seed_results": {f"seed{s}": all_results[s] for s in seeds},
        "agg": {
            "test_full_cum_pnl_per_seed": pnls,
            "loso_equiv_sum_argmax_per_seed": sums,
            "test_full_cum_pnl_mean": float(np.mean(pnls)) if pnls else 0.0,
            "loso_equiv_sum_argmax_mean": float(np.mean(sums)) if sums else 0.0,
        },
    }
    out_path = os.path.join(HERE, f"winner_summary_h{H}_{args.out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.finish()
    progress("step3_done", seeds=seeds)


if __name__ == "__main__":
    main()
