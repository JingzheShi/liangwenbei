"""Train + evaluate LightGBM on label_60 for scheme A or B.

Pipeline:
  1. Load cached .npz built by build_features.py
  2. Train LightGBM with class-balanced sample weights and early stopping on val
  3. Evaluate on val + test:
       - argmax prediction -> per-horizon metrics for label_60
       - cum_pnl, single_pnl, accuracy, precision/recall/F0.5
  4. Save model, test predictions parquet, feature importance
  5. Log to WandB
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(EXP_DIR, "cache")
NUM_CLASS = 3


def load_split(scheme: str, split: str):
    p = os.path.join(CACHE_DIR, f"scheme{scheme}_{split}.npz")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    """sklearn 'balanced' style: weight_c = n_total / (num_class * count_c)."""
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def predict_to_class_and_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    """Predict probabilities then take argmax. Batched to keep memory in check."""
    probs_chunks = []
    for s in range(0, len(X), batch):
        p = booster.predict(X[s : s + batch])
        probs_chunks.append(p.astype(np.float32))
    probs = np.concatenate(probs_chunks, axis=0)  # (N, 3)
    preds = probs.argmax(axis=1).astype(np.int8)
    return preds, probs


def evaluate(name: str, preds: np.ndarray, y: np.ndarray, mp_t: np.ndarray, mp_t60: np.ndarray):
    metrics = _per_horizon_metrics(preds, y, mp_t, mp_t60, fee_rate=0.0001)
    metrics["pred_distribution"] = dict(metrics["pred_distribution"])
    metrics["label_distribution"] = dict(metrics["label_distribution"])
    print(f"\n--- {name} (label_60) ---", flush=True)
    for k in (
        "accuracy", "cum_pnl", "single_pnl", "n_predictions_active",
        "precision_up", "precision_down", "recall_up", "recall_down",
        "f0_5_up", "f0_5_down", "f0_5_macro",
    ):
        v = metrics.get(k)
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.6g}", flush=True)
        else:
            print(f"  {k:25s}: {v}", flush=True)
    print(f"  pred_distribution        : {metrics['pred_distribution']}", flush=True)
    print(f"  label_distribution       : {metrics['label_distribution']}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=("A", "B"), required=True)
    ap.add_argument("--num-boost-round", type=int, default=2000)
    ap.add_argument("--early-stopping", type=int, default=50)
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
    args = ap.parse_args()

    scheme = args.scheme
    print(f"=== Training scheme {scheme} ===", flush=True)

    # ---- WandB ----
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T2-gbdt-lgbm-scheme{scheme}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": scheme,
                    "horizon": "label_60",
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T2-gbdt", f"scheme-{scheme}", "label_60"],
            )
            # Prefer team entity if available; otherwise fall back to personal.
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team entity rejected ({e_team!r}); "
                      f"falling back to personal entity.", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); continuing without it.", flush=True)
            use_wandb = False

    # ---- Load cached features ----
    t0 = time.time()
    train = load_split(scheme, "train")
    val = load_split(scheme, "val")
    test = load_split(scheme, "test")
    print(f"loaded splits in {time.time()-t0:.1f}s; "
          f"train={train['X'].shape}, val={val['X'].shape}, test={test['X'].shape}",
          flush=True)

    feat_names_path = os.path.join(CACHE_DIR, f"scheme{scheme}_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]

    y_train = train["y60"].astype(np.int64)
    y_val = val["y60"].astype(np.int64)

    # Class balance
    cls_counts = np.bincount(y_train, minlength=3)
    print(f"train label_60 distribution: {dict(enumerate(cls_counts.tolist()))} "
          f"(ratios {cls_counts/cls_counts.sum()})", flush=True)
    sw_train = class_balanced_weight(y_train)
    sw_val = class_balanced_weight(y_val)

    # ---- LightGBM Datasets ----
    t0 = time.time()
    dtrain = lgb.Dataset(
        train["X"], label=y_train, weight=sw_train,
        feature_name=feat_names, free_raw_data=False,
    )
    dval = lgb.Dataset(
        val["X"], label=y_val, weight=sw_val,
        feature_name=feat_names, reference=dtrain, free_raw_data=False,
    )
    print(f"built lgb.Datasets in {time.time()-t0:.1f}s", flush=True)

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

    eval_log: dict = {"train": [], "val": []}

    def cb_log(env):
        # env.evaluation_result_list: list of (data_name, eval_name, value, is_higher_better)
        for r in env.evaluation_result_list:
            data_name, metric_name, value, _ = r[0], r[1], r[2], r[3]
            eval_log.setdefault(data_name, []).append({
                "iter": env.iteration, "metric": metric_name, "value": float(value),
            })
            if use_wandb:
                wandb.log({f"{data_name}/{metric_name}": float(value),
                           "iter": env.iteration})

    t0 = time.time()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
            lgb.log_evaluation(period=20),
            cb_log,
        ],
    )
    train_time = time.time() - t0
    print(f"\ntraining done in {train_time:.1f}s, best_iter={booster.best_iteration}",
          flush=True)

    # ---- Save model ----
    model_path = os.path.join(EXP_DIR, f"model_scheme{scheme}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)
    print(f"model saved -> {model_path}", flush=True)

    # ---- Predict val + test ----
    t0 = time.time()
    val_pred, val_proba = predict_to_class_and_proba(booster, val["X"])
    val_time = time.time() - t0
    print(f"val predict {len(val_pred)} samples in {val_time:.2f}s "
          f"({val_time/len(val_pred)*1e6:.2f} us/sample)", flush=True)

    t0 = time.time()
    test_pred, test_proba = predict_to_class_and_proba(booster, test["X"])
    test_time = time.time() - t0
    print(f"test predict {len(test_pred)} samples in {test_time:.2f}s "
          f"({test_time/len(test_pred)*1e6:.2f} us/sample)", flush=True)

    val_metrics = evaluate("VAL", val_pred, val["y60"], val["mp_t"], val["mp_t60"])
    test_metrics = evaluate("TEST", test_pred, test["y60"], test["mp_t"], test["mp_t60"])

    # ---- Save test predictions parquet ----
    sess_map = {0: "am", 1: "pm"}
    test_df = pd.DataFrame({
        "sym": test["sym"].astype(np.int8),
        "date": test["date"].astype(np.int16),
        "session": np.array([sess_map[s] for s in test["sess_idx"]], dtype=object),
        "t": test["t"].astype(np.int16),
        "true_label_60": test["y60"].astype(np.int8),
        "pred_label_60": test_pred.astype(np.int8),
        "prob_0": test_proba[:, 0].astype(np.float32),
        "prob_1": test_proba[:, 1].astype(np.float32),
        "prob_2": test_proba[:, 2].astype(np.float32),
        "midprice_t": test["mp_t"].astype(np.float32),
        "midprice_t60": test["mp_t60"].astype(np.float32),
    })
    pred_path = os.path.join(EXP_DIR, f"test_predictions_label60_scheme{scheme}.parquet")
    test_df.to_parquet(pred_path, index=False)
    print(f"test predictions saved -> {pred_path}", flush=True)

    # ---- Feature importance ----
    fi_split = booster.feature_importance(importance_type="split", iteration=booster.best_iteration)
    fi_gain = booster.feature_importance(importance_type="gain", iteration=booster.best_iteration)
    fi_df = (
        pd.DataFrame({"feature": feat_names, "split": fi_split, "gain": fi_gain})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    fi_path = os.path.join(EXP_DIR, f"feature_importance_scheme{scheme}.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"feature importance saved -> {fi_path}; top 10 by gain:", flush=True)
    print(fi_df.head(10).to_string(), flush=True)

    # ---- Summary results.json ----
    speed = {
        "train_time_sec": train_time,
        "val_predict_time_sec": val_time,
        "test_predict_time_sec": test_time,
        "val_us_per_sample": float(val_time / len(val_pred) * 1e6),
        "test_us_per_sample": float(test_time / len(test_pred) * 1e6),
        "best_iteration": int(booster.best_iteration),
    }
    summary = {
        "scheme": scheme,
        "horizon": "label_60",
        "params": params,
        "n_features": train["X"].shape[1],
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(test["y60"])),
        "speed": speed,
        "val": val_metrics,
        "test": test_metrics,
    }
    summary_path = os.path.join(EXP_DIR, f"results_scheme{scheme}.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"summary saved -> {summary_path}", flush=True)

    if use_wandb:
        wandb.summary.update({
            "val/cum_pnl": val_metrics["cum_pnl"],
            "val/single_pnl": val_metrics["single_pnl"],
            "val/accuracy": val_metrics["accuracy"],
            "test/cum_pnl": test_metrics["cum_pnl"],
            "test/single_pnl": test_metrics["single_pnl"],
            "test/accuracy": test_metrics["accuracy"],
            "test/n_predictions_active": test_metrics["n_predictions_active"],
            "speed/test_us_per_sample": speed["test_us_per_sample"],
            "speed/best_iteration": speed["best_iteration"],
            "n_features": train["X"].shape[1],
        })
        wandb.finish()

    print("\n=== DONE ===", flush=True)
    print(f"VAL  cum_pnl = {val_metrics['cum_pnl']:.6f}", flush=True)
    print(f"TEST cum_pnl = {test_metrics['cum_pnl']:.6f}", flush=True)


if __name__ == "__main__":
    main()
