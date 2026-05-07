"""T36 stage 2: take Optuna top-K hyperparam configs and validate
on full 5-fold LOSO (h_60). Per config: train 5 folds, sweep T,d on
the concatenated 5-fold OOF preds, report best LOSO sum cum_pnl.

Compares against iter_006 baseline (LOSO sum +13.61 with 5-seed asym DE).
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

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
N_DROP_TAIL = 3
HORIZON = 60
SYMS = (0, 1, 2, 3, 4)

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


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_concat(X_orig, y_orig, rng, aug_range):
    lo, hi = 1.0 - aug_range, 1.0 + aug_range
    scales = rng.uniform(lo, hi, size=X_orig.shape).astype(X_orig.dtype)
    X_aug = X_orig * scales
    return (np.concatenate([X_orig, X_aug], axis=0),
            np.concatenate([y_orig, y_orig], axis=0))


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


def thresh_sweep_full(fold_dfs, fee_rate=0.0001):
    """Sweep T,d across all 5 folds; return best (sum_cum_pnl, T, d, per_fold).

    Symmetric (T, d) grid search.
    """
    best = None
    for T in T_GRID:
        for d in D_GRID:
            per_fold = []
            sum_pnl = 0.0
            n_active_total = 0
            for held, df in fold_dfs.items():
                probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
                pred = thresholded_pred(probs, T, d)
                m = _per_horizon_metrics(
                    pred, df["true_label"].to_numpy(np.int64),
                    df["midprice_t"].to_numpy(np.float32),
                    df["midprice_th"].to_numpy(np.float32),
                    fee_rate=fee_rate,
                )
                pnl = float(m["cum_pnl"])
                per_fold.append({"held": held, "cum_pnl": pnl,
                                 "n_active": int(m["n_predictions_active"])})
                sum_pnl += pnl
                n_active_total += int(m["n_predictions_active"])
            cur = {
                "T": T, "delta": d,
                "sum_cum_pnl": sum_pnl,
                "n_pos_folds": sum(1 for r in per_fold if r["cum_pnl"] > 0),
                "n_active": n_active_total,
                "per_fold": per_fold,
            }
            if best is None or cur["sum_cum_pnl"] > best["sum_cum_pnl"]:
                best = cur
    return best


def load_split(split):
    d = np.load(os.path.join(T5B_CACHE, f"schemeC_{split}.npz"))
    return {k: d[k] for k in d.files}


def make_fold_data(train_full, val_full, test_full, held_sym):
    m_tr = train_full["sym"] != held_sym
    m_va = val_full["sym"] != held_sym
    m_te = test_full["sym"] == held_sym
    return {
        "X_tr_o": train_full["X"][m_tr][:, :-N_DROP_TAIL].astype(np.float32),
        "y_tr_o": train_full[f"y{HORIZON}"][m_tr].astype(np.int64),
        "X_va": val_full["X"][m_va][:, :-N_DROP_TAIL].astype(np.float32),
        "y_va": val_full[f"y{HORIZON}"][m_va].astype(np.int64),
        "X_te": test_full["X"][m_te][:, :-N_DROP_TAIL].astype(np.float32),
        "y_te": test_full[f"y{HORIZON}"][m_te].astype(np.int64),
        "mp_t_te": test_full["mp_t"][m_te].astype(np.float32),
        "mp_th_te": test_full[f"mp_t{HORIZON}"][m_te].astype(np.float32),
    }


def train_one_fold(fold, params, aug_range, seed,
                   num_boost_round, early_stopping):
    rng = np.random.default_rng(seed * 7919 + 1)
    X_tr, y_tr = aug_a_concat(fold["X_tr_o"], fold["y_tr_o"], rng, aug_range)
    sw_tr = class_balanced_weight(y_tr, NUM_CLASS)
    sw_va = class_balanced_weight(fold["y_va"], NUM_CLASS)
    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr, free_raw_data=False)
    dval = lgb.Dataset(fold["X_va"], label=fold["y_va"], weight=sw_va,
                       reference=dtrain, free_raw_data=False)

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
    return booster, probs, train_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", required=True,
                    help="json file with list of {name, params, aug_a_range}")
    ap.add_argument("--num-boost-round", type=int, default=400)
    ap.add_argument("--early-stopping", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--out", default="top5_results.json")
    args = ap.parse_args()

    with open(args.config_json) as f:
        configs = json.load(f)

    progress("loading_data")
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")

    folds = {h: make_fold_data(train_full, val_full, test_full, h) for h in SYMS}
    print(f"loaded data; train_orig per fold sizes:", flush=True)
    for h, f in folds.items():
        print(f"  held={h}: train={f['X_tr_o'].shape} val={f['X_va'].shape} test={f['X_te'].shape}", flush=True)
    del train_full, val_full, test_full

    out = []
    for ci, cfg in enumerate(configs):
        name = cfg["name"]
        params = dict(cfg["params"])
        params["objective"] = "multiclass"
        params["num_class"] = NUM_CLASS
        params["metric"] = "multi_logloss"
        params["verbose"] = -1
        params["num_threads"] = args.num_threads
        params["seed"] = args.seed
        params["feature_fraction_seed"] = args.seed + 1
        params["bagging_seed"] = args.seed + 2
        params["data_random_seed"] = args.seed + 3
        if args.use_gpu:
            params["device"] = "gpu"; params["gpu_use_dp"] = False
        aug_range = float(cfg["aug_a_range"])

        print(f"\n=== config {ci}: {name} ===", flush=True)
        print(f"  params={params}", flush=True)
        print(f"  aug_a_range={aug_range:.4f}", flush=True)
        progress(f"config_{ci}_{name}_running")

        fold_dfs = {}
        cfg_train_meta = []
        cfg_t0 = time.time()
        for held in SYMS:
            print(f"  -> training held={held} ...", flush=True)
            booster, probs, ttime = train_one_fold(
                folds[held], params, aug_range, args.seed,
                args.num_boost_round, args.early_stopping,
            )
            cfg_train_meta.append({"held": held,
                                   "best_iter": int(booster.best_iteration or 0),
                                   "train_time_sec": ttime})
            df = pd.DataFrame({
                "true_label": folds[held]["y_te"].astype(np.int8),
                "prob_0": probs[:, 0], "prob_1": probs[:, 1], "prob_2": probs[:, 2],
                "midprice_t": folds[held]["mp_t_te"],
                "midprice_th": folds[held]["mp_th_te"],
            })
            fold_dfs[held] = df
            # Save OOF parquet for downstream asym threshold optim
            safe_name = name.replace("+", "p").replace("/", "_")
            parq = os.path.join(
                HERE, f"oof_{safe_name}_held{held}.parquet"
            )
            df.to_parquet(parq, index=False)
            print(f"     held={held} best_iter={booster.best_iteration} t={ttime:.1f}s -> {os.path.basename(parq)}", flush=True)

        best = thresh_sweep_full(fold_dfs)
        cfg_elapsed = time.time() - cfg_t0
        out.append({
            "config_idx": ci, "name": name,
            "params": cfg["params"], "aug_a_range": aug_range,
            "loso_sum_cum_pnl": best["sum_cum_pnl"],
            "best_T": best["T"], "best_delta": best["delta"],
            "n_pos_folds": best["n_pos_folds"],
            "n_active": best["n_active"],
            "per_fold": best["per_fold"],
            "train_meta": cfg_train_meta,
            "elapsed_sec": cfg_elapsed,
        })
        print(
            f"  RESULT cfg {ci} ({name}): LOSO sum_pnl={best['sum_cum_pnl']:+.4f} "
            f"T={best['T']:.2f} d={best['delta']:.2f} pos_folds={best['n_pos_folds']}/5 "
            f"n_active={best['n_active']} elapsed={cfg_elapsed:.0f}s",
            flush=True,
        )
        with open(os.path.join(HERE, args.out), "w") as fout:
            json.dump(out, fout, indent=2)

    print("\n=== summary ===", flush=True)
    for r in sorted(out, key=lambda x: -x["loso_sum_cum_pnl"]):
        print(f"  {r['name']}: sum_pnl={r['loso_sum_cum_pnl']:+.4f}  "
              f"T={r['best_T']:.2f} d={r['best_delta']:.2f}  "
              f"pos_folds={r['n_pos_folds']}/5", flush=True)
    progress("top5_done")


if __name__ == "__main__":
    main()
