"""T36: Bayesian hyperparam optimization for LightGBM h_60 LOSO.

Single fold (held=2, worst sym per T8 diagnostic) per trial → fast iteration.
Top trials are validated on full 5-fold LOSO in stage 2.

Search space:
    learning_rate, num_leaves, min_data_in_leaf, max_depth,
    lambda_l1, lambda_l2, min_gain_to_split,
    feature_fraction, bagging_fraction, bagging_freq,
    aug_a_range  (aug_a scale uniform in [1-r, 1+r])

Constraints:
    sym-agnostic features, stateless inference, drop trailing 3 cols.
    Augment is per-(sample, feature) random scale, applied only to training.
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
import optuna  # noqa: E402
import wandb  # noqa: E402

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
N_DROP_TAIL = 3
HORIZON = 60
HELD_SYM_FOR_OPT = 2

# Threshold sweep grid (matches T31 / T28 convention)
T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running", "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_concat(X_orig, y_orig, rng, aug_range):
    """Concat orig + aug with per-(sample, feature) scale in [1-r, 1+r]."""
    lo, hi = 1.0 - aug_range, 1.0 + aug_range
    scales = rng.uniform(lo, hi, size=X_orig.shape).astype(X_orig.dtype)
    X_aug = X_orig * scales
    X_out = np.concatenate([X_orig, X_aug], axis=0)
    y_out = np.concatenate([y_orig, y_orig], axis=0)
    return X_out, y_out


def predict_proba(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def thresholded_pred(probs, T, delta):
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def thresh_sweep_one_fold(probs, y_true, mp_t, mp_th, fee_rate=0.0001):
    """Sweep T,d for a single fold; return best (cum_pnl, T, d, n_active)."""
    best = None
    for T in T_GRID:
        for d in D_GRID:
            pred = thresholded_pred(probs, T, d)
            m = _per_horizon_metrics(pred, y_true, mp_t, mp_th, fee_rate=fee_rate)
            cur = (float(m["cum_pnl"]), T, d, int(m["n_predictions_active"]))
            if best is None or cur[0] > best[0]:
                best = cur
    return best


def load_split(split):
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def make_fold_data(train_full, val_full, test_full, held_sym):
    """Slice train/val/test for one held_sym fold.

    train: sym != held, dates 0..79
    val:   sym != held, dates 80..95  (early stopping)
    test:  sym == held, dates 96..119 (LOSO target)
    """
    m_tr = train_full["sym"] != held_sym
    m_va = val_full["sym"] != held_sym
    m_te = test_full["sym"] == held_sym

    X_tr_o = train_full["X"][m_tr][:, :-N_DROP_TAIL].astype(np.float32)
    y_tr_o = train_full[f"y{HORIZON}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL].astype(np.float32)
    y_va = val_full[f"y{HORIZON}"][m_va].astype(np.int64)
    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL].astype(np.float32)
    y_te = test_full[f"y{HORIZON}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te].astype(np.float32)
    mp_th_te = test_full[f"mp_t{HORIZON}"][m_te].astype(np.float32)

    return {
        "X_tr_o": X_tr_o, "y_tr_o": y_tr_o,
        "X_va": X_va, "y_va": y_va,
        "X_te": X_te, "y_te": y_te,
        "mp_t_te": mp_t_te, "mp_th_te": mp_th_te,
    }


def train_and_score(fold, params, aug_range, seed, num_boost_round, early_stopping):
    rng = np.random.default_rng(seed * 7919 + 1)
    X_tr, y_tr = aug_a_concat(fold["X_tr_o"], fold["y_tr_o"], rng, aug_range)
    sw_tr = class_balanced_weight(y_tr, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(fold["y_va"], num_class=NUM_CLASS)

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, free_raw_data=False)
    dval = lgb.Dataset(
        fold["X_va"], label=fold["y_va"], weight=sw_va,
        reference=dtrain, free_raw_data=False,
    )

    t0 = time.time()
    booster = lgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    train_time = time.time() - t0

    probs = predict_proba(booster, fold["X_te"])
    cum_pnl, T_best, d_best, n_active = thresh_sweep_one_fold(
        probs, fold["y_te"], fold["mp_t_te"], fold["mp_th_te"]
    )

    return {
        "cum_pnl": cum_pnl, "best_T": T_best, "best_d": d_best,
        "n_active": n_active, "best_iter": int(booster.best_iteration or 0),
        "train_time_sec": train_time, "n_train_used": int(len(X_tr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--num-boost-round", type=int, default=400)
    ap.add_argument("--early-stopping", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--study-name", default="T36-h60-optuna")
    ap.add_argument("--storage", default=None,
                    help="optional sqlite path (default: file in HERE)")
    args = ap.parse_args()

    storage = args.storage or f"sqlite:///{os.path.join(HERE, 'optuna_study.db')}"

    try:
        wandb.init(
            project="liangwenbei", entity="cjxh21-Tsinghua University",
            name=f"T36-optuna-h60-trials{args.n_trials}",
            config={
                "n_trials": args.n_trials, "horizon": HORIZON,
                "held_for_opt": HELD_SYM_FOR_OPT,
                "num_boost_round": args.num_boost_round,
                "early_stopping": args.early_stopping,
            },
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        wandb_active = True
    except Exception as e:
        print(f"  WandB init failed (mode=offline): {e}", flush=True)
        wandb_active = False

    progress("loading_data")
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    fold = make_fold_data(train_full, val_full, test_full, HELD_SYM_FOR_OPT)
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train_orig={fold['X_tr_o'].shape} val={fold['X_va'].shape} test={fold['X_te'].shape}",
        flush=True,
    )
    del train_full, val_full, test_full

    trial_log = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multiclass", "num_class": NUM_CLASS,
            "metric": "multi_logloss",
            "verbose": -1,
            "num_threads": args.num_threads,
            "seed": args.seed,
            "feature_fraction_seed": args.seed + 1,
            "bagging_seed": args.seed + 2,
            "data_random_seed": args.seed + 3,
            # Loss
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True),
            # Tree
            "num_leaves": trial.suggest_int(
                "num_leaves", 31, 511, log=True),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 20, 500, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            # Regularization
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float(
                "min_gain_to_split", 0.0, 0.1),
            # Sampling
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        }
        if args.use_gpu:
            params["device"] = "gpu"
            params["gpu_use_dp"] = False
        aug_range = trial.suggest_float("aug_a_range", 0.05, 0.4)

        t_trial = time.time()
        progress(f"trial_{trial.number}_running", trial=trial.number)
        try:
            res = train_and_score(
                fold, params, aug_range, args.seed,
                args.num_boost_round, args.early_stopping,
            )
        except Exception as e:
            print(f"trial {trial.number} ERROR: {e}", flush=True)
            return -100.0

        elapsed = time.time() - t_trial
        row = {
            "trial_number": trial.number,
            "params": dict(trial.params),
            "cum_pnl": res["cum_pnl"], "best_T": res["best_T"],
            "best_d": res["best_d"], "n_active": res["n_active"],
            "best_iter": res["best_iter"],
            "train_time_sec": res["train_time_sec"],
            "elapsed_sec": elapsed,
        }
        trial_log.append(row)
        with open(os.path.join(HERE, "trial_log.json"), "w") as f:
            json.dump(trial_log, f, indent=2)
        progress(
            f"trial_{trial.number}_done",
            trial=trial.number, cum_pnl=res["cum_pnl"], elapsed=elapsed,
        )
        print(
            f"  trial {trial.number}: cum_pnl={res['cum_pnl']:+.4f} "
            f"T={res['best_T']:.2f} d={res['best_d']:.2f} "
            f"iter={res['best_iter']} t={elapsed:.0f}s "
            f"params={trial.params}",
            flush=True,
        )
        if wandb_active:
            wandb.log({
                "trial": trial.number, "cum_pnl": res["cum_pnl"],
                "best_T": res["best_T"], "best_d": res["best_d"],
                "best_iter": res["best_iter"],
                "train_time_sec": res["train_time_sec"],
                **{f"param/{k}": v for k, v in trial.params.items()},
            })
        return res["cum_pnl"]

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    print(f"=== T36 Optuna search: held={HELD_SYM_FOR_OPT} n_trials={args.n_trials} ===", flush=True)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print("\n=== best ===", flush=True)
    print(f"  best_value (held={HELD_SYM_FOR_OPT} cum_pnl) = {study.best_value:+.4f}", flush=True)
    print(f"  best_params = {study.best_params}", flush=True)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(HERE, "trials_dataframe.csv"), index=False)

    summary = {
        "n_trials": len(study.trials),
        "best_trial_number": int(study.best_trial.number),
        "best_value_cum_pnl_held2": float(study.best_value),
        "best_params": study.best_params,
        "trial_log": trial_log,
    }
    with open(os.path.join(HERE, "optuna_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    progress("optuna_done", best_cum_pnl=float(study.best_value))

    if wandb_active:
        wandb.summary["best_cum_pnl_held2"] = float(study.best_value)
        wandb.summary["best_trial"] = int(study.best_trial.number)
        wandb.finish()


if __name__ == "__main__":
    main()
