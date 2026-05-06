"""T9 LOSO training for Scheme E (T2 Scheme A 154 + F1-F4 sym-agnostic features).

Same protocol as T4: for each held_out_sym k in {0..4}:
    train: sym != k, dates 0..79
    val:   sym != k, dates 80..95   (early stopping)
    test:  sym == k, dates 96..119  (held-out)

Hyperparameters identical to T2 Scheme A (lr=0.05, num_leaves=127, etc.).
Sym-agnostic by construction. class_weight='balanced'.

`--features` controls which F-groups are included (always includes baseline A=154 cols):
    A          : 154 cols only (baseline reference)
    AF1        : A + F1 (amount z within window)
    AF12       : A + F1 + F2
    AF123      : A + F1 + F2 + F3
    AF1234     : A + F1 + F2 + F3 + F4   (default — full Scheme E)

Usage:
    python train_loso.py --features AF1234
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402
from build_features import get_feature_group_indices  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)

FEATURE_VARIANTS = {
    "A":      ["A"],
    "AF1":    ["A", "F1"],
    "AF12":   ["A", "F1", "F2"],
    "AF123":  ["A", "F1", "F2", "F3"],
    "AF1234": ["A", "F1", "F2", "F3", "F4"],
}


def load_split(split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"schemeE_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def predict_to_class_and_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    probs = np.concatenate(chunks, axis=0)
    preds = probs.argmax(axis=1).astype(np.int8)
    return preds, probs


def evaluate(name: str, preds: np.ndarray, y: np.ndarray, mp_t: np.ndarray, mp_t60: np.ndarray):
    m = _per_horizon_metrics(preds, y, mp_t, mp_t60, fee_rate=0.0001)
    m["pred_distribution"] = dict(m["pred_distribution"])
    m["label_distribution"] = dict(m["label_distribution"])
    print(f"  --- {name} ---", flush=True)
    for k in ("accuracy", "cum_pnl", "single_pnl", "n_predictions_active",
              "precision_up", "precision_down", "recall_up", "recall_down",
              "f0_5_up", "f0_5_down", "f0_5_macro"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"    {k:25s}: {v:.6g}", flush=True)
        else:
            print(f"    {k:25s}: {v}", flush=True)
    print(f"    pred_distribution        : {m['pred_distribution']}", flush=True)
    return m


def select_features(X: np.ndarray, group_indices: dict, variant: str) -> tuple[np.ndarray, list[int]]:
    """Slice columns according to variant. Returns (sliced_X, kept_col_indices)."""
    keep_groups = FEATURE_VARIANTS[variant]
    kept = []
    for g in keep_groups:
        kept.extend(group_indices[g])
    kept = sorted(kept)  # preserve original order
    return X[:, kept], kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(FEATURE_VARIANTS.keys()), default="AF1234")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--save-models", action="store_true",
                    help="save per-fold model (default off for ablation runs)")
    args = ap.parse_args()

    target_syms = [int(x) for x in args.syms.split(",")]
    variant = args.features
    print(f"=== T9 LOSO scheme E variant={variant}, target held-out syms = {target_syms} ===",
          flush=True)

    # ---- Load caches once ----
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded caches in {time.time() - t0:.1f}s; "
          f"train X={train_full['X'].shape}, val X={val_full['X'].shape}, "
          f"test X={test_full['X'].shape}", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeE_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    group_indices = get_feature_group_indices(all_feat_names)

    # Slice columns according to variant
    X_train, kept = select_features(train_full["X"], group_indices, variant)
    X_val, _ = select_features(val_full["X"], group_indices, variant)
    X_test, _ = select_features(test_full["X"], group_indices, variant)
    feat_names = [all_feat_names[i] for i in kept]
    print(f"variant {variant}: {len(feat_names)} features", flush=True)
    train_full["X"] = X_train  # replace; original gc'd
    val_full["X"] = X_val
    test_full["X"] = X_test

    # ---- WandB ----
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T9-LOSO-schemeE-{variant}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "E",
                    "variant": variant,
                    "n_features": len(feat_names),
                    "horizon": "label_60",
                    "loso_strategy": "train sym!=k date 0..79; val sym!=k date 80..95; "
                                     "test sym==k date 96..119",
                    **{k: v for k, v in vars(args).items() if k not in ("no_wandb",)},
                },
                tags=["T9-schemeE", variant, "label_60"],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback to personal", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    # ---- Per-fold ----
    fold_results = []
    for held_out_sym in target_syms:
        print(f"\n{'='*78}\n=== Fold: held_out_sym = {held_out_sym} ({variant}) ===\n{'='*78}",
              flush=True)
        m_tr = train_full["sym"] != held_out_sym
        m_va = val_full["sym"] != held_out_sym
        m_te = test_full["sym"] == held_out_sym

        X_tr = train_full["X"][m_tr]
        y_tr = train_full["y60"][m_tr].astype(np.int64)
        X_va = val_full["X"][m_va]
        y_va = val_full["y60"][m_va].astype(np.int64)
        mp_t_va = val_full["mp_t"][m_va]
        mp_t60_va = val_full["mp_t60"][m_va]
        X_te = test_full["X"][m_te]
        y_te = test_full["y60"][m_te].astype(np.int64)
        mp_t_te = test_full["mp_t"][m_te]
        mp_t60_te = test_full["mp_t60"][m_te]

        print(f"  n_train={len(X_tr):,}  n_val={len(X_va):,}  n_test={len(X_te):,}", flush=True)

        sw_tr = class_balanced_weight(y_tr)
        sw_va = class_balanced_weight(y_va)

        dtrain = lgb.Dataset(
            X_tr, label=y_tr, weight=sw_tr,
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
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": args.bagging_freq,
            "lambda_l2": args.lambda_l2,
            "num_threads": args.num_threads,
            "seed": args.seed,
            "verbose": -1,
        }

        t_train_start = time.time()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=args.num_boost_round,
            valid_sets=[dtrain, dval], valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )
        train_time = time.time() - t_train_start
        print(f"  fold {held_out_sym} trained in {train_time:.1f}s, "
              f"best_iter={booster.best_iteration}", flush=True)

        val_pred, _ = predict_to_class_and_proba(booster, X_va)
        te_pred, te_prob = predict_to_class_and_proba(booster, X_te)
        val_metrics = evaluate(f"VAL (sym != {held_out_sym}, dates 80..95)",
                               val_pred, y_va, mp_t_va, mp_t60_va)
        te_metrics = evaluate(f"HELD-OUT TEST (sym == {held_out_sym}, dates 96..119)",
                              te_pred, y_te, mp_t_te, mp_t60_te)

        # Save predictions for this fold (so threshold sweep can re-tune)
        sess_map = {0: "am", 1: "pm"}
        te_df = pd.DataFrame({
            "sym": test_full["sym"][m_te].astype(np.int8),
            "date": test_full["date"][m_te].astype(np.int16),
            "session": np.array([sess_map[s] for s in test_full["sess_idx"][m_te]],
                                dtype=object),
            "t": test_full["t"][m_te].astype(np.int16),
            "true_label_60": y_te.astype(np.int8),
            "pred_label_60": te_pred.astype(np.int8),
            "prob_0": te_prob[:, 0].astype(np.float32),
            "prob_1": te_prob[:, 1].astype(np.float32),
            "prob_2": te_prob[:, 2].astype(np.float32),
            "midprice_t": mp_t_te.astype(np.float32),
            "midprice_t60": mp_t60_te.astype(np.float32),
        })
        pred_path = os.path.join(HERE, f"loso_pred_schemeE_{variant}_held{held_out_sym}.parquet")
        te_df.to_parquet(pred_path, index=False)

        if args.save_models:
            model_path = os.path.join(HERE, f"loso_model_schemeE_{variant}_held{held_out_sym}.txt")
            booster.save_model(model_path, num_iteration=booster.best_iteration)
            print(f"  model saved -> {model_path}", flush=True)

        fold_results.append({
            "held_out_sym": held_out_sym,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "best_iter": int(booster.best_iteration),
            "train_time_sec": float(train_time),
            "val": val_metrics,
            "held_out": te_metrics,
        })

        if use_wandb:
            wandb.log({
                "fold/held_out_sym": held_out_sym,
                "fold/best_iter": int(booster.best_iteration),
                "fold/train_time_sec": float(train_time),
                "fold/val_cum_pnl": val_metrics["cum_pnl"],
                "fold/held_out_cum_pnl": te_metrics["cum_pnl"],
                "fold/held_out_single_pnl": te_metrics["single_pnl"],
                "fold/held_out_accuracy": te_metrics["accuracy"],
            })

    # ---- Aggregate ----
    print(f"\n{'='*78}\n=== LOSO scheme E variant={variant} aggregate ===\n{'='*78}", flush=True)
    print(f"{'sym':>5s}  {'best_it':>7s}  {'val_cum':>9s}  {'TEST_cum':>10s}  "
          f"{'TEST_single':>12s}  {'TEST_acc':>9s}", flush=True)
    cums, sing, accs, f05s = [], [], [], []
    for r in fold_results:
        v = r["val"]; t = r["held_out"]
        cums.append(t["cum_pnl"]); sing.append(t["single_pnl"])
        accs.append(t["accuracy"]); f05s.append(t["f0_5_macro"])
        print(f"{r['held_out_sym']:>5d}  {r['best_iter']:>7d}  "
              f"{v['cum_pnl']:>+9.4f}  {t['cum_pnl']:>+10.4f}  "
              f"{t['single_pnl']:>+12.6f}  {t['accuracy']:>9.4f}", flush=True)
    cums_a = np.array(cums); sing_a = np.array(sing); accs_a = np.array(accs)
    print(f"sum cum_pnl = {cums_a.sum():+.4f}  mean = {cums_a.mean():+.4f}  "
          f"std = {cums_a.std(ddof=0):.4f}", flush=True)
    print(f"folds with cum_pnl > 0: {(cums_a > 0).sum()}/{len(cums_a)}", flush=True)

    summary = {
        "scheme": "E",
        "variant": variant,
        "n_features": len(feat_names),
        "horizon": "label_60",
        "params": {**vars(args)},
        "folds": fold_results,
        "aggregate": {
            "cum_pnl_mean": float(cums_a.mean()),
            "cum_pnl_std": float(cums_a.std(ddof=0)),
            "cum_pnl_sum": float(cums_a.sum()),
            "single_pnl_mean": float(sing_a.mean()),
            "accuracy_mean": float(accs_a.mean()),
            "f0_5_macro_mean": float(np.array(f05s).mean()),
            "n_positive_folds": int((cums_a > 0).sum()),
            "n_total_folds": int(len(cums_a)),
        },
    }
    out_path = os.path.join(HERE, f"loso_results_schemeE_{variant}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.summary.update({
            f"agg/cum_pnl_mean": float(cums_a.mean()),
            f"agg/cum_pnl_std": float(cums_a.std(ddof=0)),
            f"agg/cum_pnl_sum": float(cums_a.sum()),
            f"agg/single_pnl_mean": float(sing_a.mean()),
            f"agg/accuracy_mean": float(accs_a.mean()),
            f"agg/n_positive_folds": int((cums_a > 0).sum()),
        })
        wandb.finish()


if __name__ == "__main__":
    main()
