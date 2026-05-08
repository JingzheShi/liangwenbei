"""R5v2: Optuna Bayes optimization for LGB Huber alpha=1e-3 family.

Per trial:
  1. Sample hyperparams (Bayes TPE)
  2. Train 3 LGB Huber models on V4 split (train 0-75 / val 76-79), seeds (1, 7, 42)
  3. Predict test set, average 3 preds
  4. DE asym threshold optimization on aggregated test (442k pts, sym 0..4)
  5. Return DE-LOSO sum_per_sym PnL (objective for Optuna to MAXIMIZE)

Designed to run on a remote GPU (RTX 3080) with `device='gpu'` LGB.

Cache layout (in /root/lwb_remote_pkg/cache):
  - schemeP_train.npz: X, y60, mp_t, mp_t60, sym, date, ...
  - schemeP_test.npz: same fields (442k rows total, 5-sym)
  - schemeP_feat_names.txt: 369 feature names
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
import optuna
import lightgbm as lgb
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.environ.get("LWB_PKG_ROOT", "/root/lwb_remote_pkg")
CACHE_DIR = os.path.join(PKG_ROOT, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
H = 60

# ---- inline aug from T26/build_aug.py (avoid imports) ----
def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_scale(X, rng, lo=0.8, hi=1.2):
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return X * scales


# ---- features to drop (T59 + Stage5 fail) ----
T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def per_sym_pnl(pred, sym_arr, mp_t, mp_th, thr_up, thr_dn):
    per = []
    for k in SYMS:
        m = sym_arr == k
        if m.sum() == 0:
            per.append(0.0); continue
        a = ev_gate(pred[m], thr_up, thr_dn)
        per.append(float(vectorized_pnl(a, mp_t[m], mp_th[m]).sum()))
    return per


def de_loso(pred, sym_arr, mp_t, mp_th, bounds_hi=0.0040, de_seeds=(0,),
            maxiter=40, popsize=15):
    fold_pred, fold_mpt, fold_mpth = [], [], []
    for k in SYMS:
        m = sym_arr == k
        fold_pred.append(pred[m])
        fold_mpt.append(mp_t[m])
        fold_mpth.append(mp_th[m])

    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(SYMS)):
            a = ev_gate(fold_pred[i], thr_up, thr_dn)
            s += vectorized_pnl(a, fold_mpt[i], fold_mpth[i]).sum()
        return -float(s)

    best = None
    for sd in de_seeds:
        result = differential_evolution(
            f, bounds=[(0.0, bounds_hi), (0.0, bounds_hi)], seed=sd,
            maxiter=maxiter, popsize=popsize, polish=True, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7, updating="deferred",
            workers=1, init="sobol",
        )
        run = {"seed": int(sd), "thr_up": float(result.x[0]),
               "thr_dn": float(result.x[1]), "obj_val": float(-result.fun)}
        if best is None or run["obj_val"] > best["obj_val"]:
            best = run
    return best


# ---- Globals: load once ----
print("=== R5v2 Bayes opt: loading caches ===", flush=True)
t0 = time.time()
train_full = {k: v for k, v in np.load(os.path.join(CACHE_DIR, "schemeP_train.npz")).items()}
test_full = {k: v for k, v in np.load(os.path.join(CACHE_DIR, "schemeP_test.npz")).items()}
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
    all_feat_names = [line.strip() for line in f]
total_dim = train_full["X"].shape[1]
assert len(all_feat_names) == total_dim, (len(all_feat_names), total_dim)

name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
drop_idx = set()
for n in DROP_NAMES:
    if n in name_to_idx:
        drop_idx.add(name_to_idx[n])
keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
slicer = keep_idx
feat_dim = len(keep_idx)
feat_names = [all_feat_names[i] for i in slicer]
forbidden = {"date", "sym", "time"}
leak = forbidden & set(feat_names)
assert not leak, f"feature leak: {leak}"
print(f"  feat_dim={feat_dim}", flush=True)

# Build V4 train/val
date_tr = train_full["date"]
m_va = date_tr >= 76
m_t = ~m_va

X_tr_base = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
y_cls_tr_base = train_full["y60"][m_t].astype(np.int64)
y_regr_tr_base = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
y_cls_va = train_full["y60"][m_va].astype(np.int64)
sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)

# Test
X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
mp_t_te = test_full["mp_t"]
mp_th_te = test_full[f"mp_t{H}"]
sym_te = test_full["sym"].astype(np.int64)
print(f"  n_train={len(X_tr_base):,}  n_val={len(X_va):,}  n_test={len(X_te):,}", flush=True)


def train_one_seed(params_base, seed, num_boost_round=600, early_stopping=40,
                   aug_lo=0.80, aug_hi=1.20, aug_ratio=1.0):
    """Train one LGB Huber model with given hyperparams + seed; return test preds."""
    rng = np.random.default_rng(seed * 7919 + 1)
    n_aug = int(round(aug_ratio * len(X_tr_base)))
    if n_aug > 0:
        if n_aug == len(X_tr_base):
            aug_idx = np.arange(len(X_tr_base))
        else:
            aug_idx = rng.integers(0, len(X_tr_base), size=n_aug)
        X_aug = aug_a_scale(X_tr_base[aug_idx], rng, lo=aug_lo, hi=aug_hi)
        y_cls_aug = y_cls_tr_base[aug_idx]
        y_regr_aug = y_regr_tr_base[aug_idx]
        X_tr_full = np.concatenate([X_tr_base, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr_base, y_cls_aug], axis=0)
        y_regr_tr_full = np.concatenate([y_regr_tr_base, y_regr_aug], axis=0)
    else:
        X_tr_full = X_tr_base
        y_cls_tr_full = y_cls_tr_base
        y_regr_tr_full = y_regr_tr_base

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = dict(params_base)
    params["seed"] = seed
    params["feature_fraction_seed"] = seed + 1
    params["bagging_seed"] = seed + 2
    params["data_random_seed"] = seed + 3

    booster = lgb.train(
        params=params, train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping, verbose=False)],
    )
    best_iter = int(booster.best_iteration)
    # predict test
    chunks = []
    for s in range(0, len(X_te), 200_000):
        chunks.append(booster.predict(X_te[s:s+200_000]).astype(np.float32))
    te_pred = np.concatenate(chunks, axis=0)
    return te_pred, best_iter


# ---- Optuna objective ----
TRIAL_HISTORY = []
TRIAL_DIR = os.path.join(PKG_ROOT, "out_v2_lgb")
os.makedirs(TRIAL_DIR, exist_ok=True)


def make_objective(seeds, num_boost_round, early_stopping, de_maxiter, de_popsize, use_gpu):
    def objective(trial: optuna.Trial):
        params = {
            "objective": "huber",
            "alpha": trial.suggest_float("alpha", 3e-4, 3e-3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [0, 5, 10]),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 20.0, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "metric": "l1",
            "verbose": -1,
            "num_threads": 4,
        }
        if use_gpu:
            params["device"] = "gpu"
            params["gpu_use_dp"] = False

        t0 = time.time()
        te_preds = []
        best_iters = []
        for sd in seeds:
            try:
                pred, bi = train_one_seed(params, sd,
                                          num_boost_round=num_boost_round,
                                          early_stopping=early_stopping)
            except Exception as e:
                print(f"  trial {trial.number} seed {sd} FAIL: {e}", flush=True)
                raise optuna.TrialPruned() from e
            te_preds.append(pred)
            best_iters.append(bi)
        train_time = time.time() - t0

        avg_pred = np.mean(te_preds, axis=0)
        # DE eval
        t1 = time.time()
        best = de_loso(avg_pred, sym_te, mp_t_te, mp_th_te,
                       bounds_hi=0.0040, de_seeds=(0,),
                       maxiter=de_maxiter, popsize=de_popsize)
        eval_time = time.time() - t1
        score = float(best["obj_val"])

        # Per-sym
        per_de = per_sym_pnl(avg_pred, sym_te, mp_t_te, mp_th_te,
                             best["thr_up"], best["thr_dn"])

        rec = {
            "trial": trial.number,
            "score": score,
            "thr_up": best["thr_up"],
            "thr_dn": best["thr_dn"],
            "best_iters": best_iters,
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "per_sym": per_de,
            "params": params,
        }
        TRIAL_HISTORY.append(rec)
        with open(os.path.join(TRIAL_DIR, "trial_history.json"), "w") as f:
            json.dump(TRIAL_HISTORY, f, indent=2)
        print(f"  trial {trial.number:3d}  score={score:+8.4f} "
              f"[train {train_time:.1f}s, eval {eval_time:.1f}s, "
              f"best_iters={best_iters}]  alpha={params['alpha']:.4f} "
              f"nl={params['num_leaves']} lr={params['learning_rate']:.3f}",
              flush=True)
        return score
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--seeds", default="1,7,42")
    ap.add_argument("--num-boost-round", type=int, default=400)
    ap.add_argument("--early-stopping", type=int, default=30)
    ap.add_argument("--de-maxiter", type=int, default=40)
    ap.add_argument("--de-popsize", type=int, default=15)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--study-name", default="r5v2_lgb_huber")
    ap.add_argument("--storage", default=None)
    ap.add_argument("--time-budget-sec", type=int, default=4200)  # 70 min default
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== R5v2 Optuna Bayes LGB Huber: {args.n_trials} trials, "
          f"seeds={seeds}, gpu={args.use_gpu} ===", flush=True)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=10)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=args.study_name, storage=args.storage)
    objective = make_objective(seeds, args.num_boost_round, args.early_stopping,
                               args.de_maxiter, args.de_popsize, args.use_gpu)
    t_start = time.time()
    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.time_budget_sec)
    except KeyboardInterrupt:
        print("Interrupted", flush=True)

    t_elapsed = time.time() - t_start
    print(f"\n=== DONE: {len(study.trials)} trials in {t_elapsed:.0f}s ===", flush=True)

    best_trial = study.best_trial
    print(f"BEST: trial {best_trial.number}, score={best_trial.value:+.4f}", flush=True)
    print(f"  params: {json.dumps(best_trial.params, indent=2)}", flush=True)

    out = {
        "study_name": args.study_name,
        "n_trials_done": len(study.trials),
        "best_trial": best_trial.number,
        "best_score": float(best_trial.value),
        "best_params": dict(best_trial.params),
        "elapsed_sec": t_elapsed,
        "history": TRIAL_HISTORY,
        "all_trials": [{"n": t.number, "v": t.value, "p": t.params}
                       for t in study.trials if t.value is not None],
    }
    out_path = os.path.join(TRIAL_DIR, "bayes_lgb_huber_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
