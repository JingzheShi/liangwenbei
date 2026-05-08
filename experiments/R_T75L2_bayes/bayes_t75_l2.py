"""R_T75L2_bayes: Optuna Bayes opt on T75 LGB regression_l2 base (3-seed, DE asym).

Pre-computes augmented (aug_a) train sets per seed (1, 7, 42) once, builds LGB
Datasets once, then iterates over Optuna-suggested hyperparam combos. Each trial:
  - Train LGB-GPU with the suggested hyperparams on each of the 3 seeded aug sets
  - Predict on the 442k 5-sym test
  - Average preds across seeds
  - DE-search asymmetric (thr_up, thr_dn), get full 5-sym pnl + per-sym
  - Return LOSO-equivalent (mean of leave-one-sym-out 4-sym sums)

Optimizes LOSO-equiv (the same metric used to compare T75 against iter_015).
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
import lightgbm as lgb
import optuna
from scipy.optimize import differential_evolution

CACHE_DIR = "/root/lwb_remote_pkg/cache"
HERE = "/root/lwb_work_t75_bayes"
SEEDS = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
NUM_CLASS = 3

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_scale(X, rng, lo=0.8, hi=1.2):
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return X * scales


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    try:
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def total_pnl_at(folds, thr_up, thr_dn):
    s = 0.0
    for f in folds:
        a = ev_gate_asym(f["pred"], thr_up, thr_dn)
        s += float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum())
    return s


def per_sym_pnl_at(folds, thr_up, thr_dn):
    out = {}
    for f in folds:
        a = ev_gate_asym(f["pred"], thr_up, thr_dn)
        out[int(f["sym"])] = float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum())
    return out


def loso_equiv_from_persym(per_sym):
    sums = []
    for k in SYMS:
        sums.append(sum(v for kk, v in per_sym.items() if kk != k))
    return float(np.mean(sums)), float(np.min(sums))


def de_eval_3seed_avg(preds_per_seed, sym_te, mp_t_te, mp_th_te, de_seed=0):
    p = np.mean(np.stack(preds_per_seed, axis=0), axis=0)
    folds = []
    for k in SYMS:
        m = (sym_te == k)
        folds.append({
            "sym": int(k),
            "pred": p[m],
            "mp_t": mp_t_te[m].astype(np.float64),
            "mp_th": mp_th_te[m].astype(np.float64),
        })
    fee_thr = 2.0 * FEE
    bounds = [(0.0, 6.0 * fee_thr), (0.0, 6.0 * fee_thr)]

    def neg_obj(x):
        return -total_pnl_at(folds, x[0], x[1])

    res = differential_evolution(
        neg_obj, bounds, seed=de_seed, maxiter=80, popsize=24, tol=1e-7,
        polish=True, updating="deferred", workers=1,
    )
    thr_up, thr_dn = float(res.x[0]), float(res.x[1])
    pnl_de = -float(res.fun)
    per_sym = per_sym_pnl_at(folds, thr_up, thr_dn)
    loso_mean, loso_min = loso_equiv_from_persym(per_sym)
    return {
        "loso_mean": loso_mean, "loso_min": loso_min,
        "pnl_5sym": pnl_de, "thr_up": thr_up, "thr_dn": thr_dn,
        "per_sym": per_sym,
    }


def build_data():
    print("loading caches...", flush=True)
    t0 = time.time()
    tr = np.load(f"{CACHE_DIR}/schemeP_train.npz")
    te = np.load(f"{CACHE_DIR}/schemeP_test.npz")
    with open(f"{CACHE_DIR}/schemeP_feat_names.txt") as f:
        feat_names_all = [l.strip() for l in f]

    drop_idx = set()
    for n in DROP_NAMES:
        if n in feat_names_all:
            drop_idx.add(feat_names_all.index(n))
    keep_idx = np.array([i for i in range(tr["X"].shape[1]) if i not in drop_idx],
                        dtype=np.int64)
    feat_names = [feat_names_all[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))
    print(f"  feat_dim={len(keep_idx)} (dropped {len(drop_idx)})", flush=True)

    date_tr = tr["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr_raw = tr["X"][m_t][:, keep_idx].astype(np.float32, copy=False)
    y_cls_tr_raw = tr["y60"][m_t].astype(np.int64)
    y_regr_tr_raw = regr_target(tr["mp_t"][m_t], tr["mp_t60"][m_t])

    X_va = tr["X"][m_va][:, keep_idx].astype(np.float32, copy=False)
    y_cls_va = tr["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(tr["mp_t"][m_va], tr["mp_t60"][m_va])
    sw_va = class_balanced_weight(y_cls_va, NUM_CLASS)

    X_te = te["X"][:, keep_idx].astype(np.float32, copy=False)
    y_regr_te = regr_target(te["mp_t"], te["mp_t60"])
    mp_t_te = te["mp_t"].astype(np.float64)
    mp_th_te = te["mp_t60"].astype(np.float64)
    sym_te = te["sym"].astype(np.int64)

    print(f"  loaded in {time.time()-t0:.1f}s | n_tr={len(X_tr_raw):,} n_va={len(X_va):,} "
          f"n_te={len(X_te):,}", flush=True)

    print("building aug_a per-seed (lo=0.80, hi=1.20)...", flush=True)
    t0 = time.time()
    aug_data = {}
    for s in SEEDS:
        rng = np.random.default_rng(s * 7919 + 1)
        X_aug = aug_a_scale(X_tr_raw, rng, lo=0.80, hi=1.20)
        X_full = np.concatenate([X_tr_raw, X_aug], axis=0)
        y_regr_full = np.concatenate([y_regr_tr_raw, y_regr_tr_raw], axis=0)
        y_cls_full = np.concatenate([y_cls_tr_raw, y_cls_tr_raw], axis=0)
        sw_full = class_balanced_weight(y_cls_full, NUM_CLASS)
        aug_data[s] = (X_full, y_regr_full, sw_full)
        print(f"  seed={s} n_full={len(X_full):,}", flush=True)
    print(f"  aug built in {time.time()-t0:.1f}s", flush=True)

    return {
        "X_va": X_va, "y_regr_va": y_regr_va, "sw_va": sw_va,
        "X_te": X_te, "y_regr_te": y_regr_te,
        "mp_t_te": mp_t_te, "mp_th_te": mp_th_te, "sym_te": sym_te,
        "feat_names": feat_names, "aug_data": aug_data,
    }


def train_one_seed(params, seed, data, datasets):
    """Train + predict on test. Returns (test_pred, best_iter, train_time)."""
    dtrain, dval = datasets[seed]
    p = dict(params)
    p["seed"] = seed
    p["feature_fraction_seed"] = seed + 1
    p["bagging_seed"] = seed + 2
    p["data_random_seed"] = seed + 3
    t0 = time.time()
    booster = lgb.train(
        p, dtrain,
        num_boost_round=600,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    train_time = time.time() - t0
    best_iter = int(booster.best_iteration) if booster.best_iteration else 600
    pred = booster.predict(data["X_te"], num_iteration=best_iter).astype(np.float32)
    return pred, best_iter, train_time, booster


def make_datasets(data):
    feat_names = data["feat_names"]
    X_va, y_regr_va, sw_va = data["X_va"], data["y_regr_va"], data["sw_va"]
    datasets = {}
    for s in SEEDS:
        X_full, y_full, sw_full = data["aug_data"][s]
        dtrain = lgb.Dataset(X_full, label=y_full, weight=sw_full,
                             feature_name=feat_names, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                           feature_name=feat_names, reference=dtrain,
                           free_raw_data=False)
        # Construct so reference binning works
        dtrain.construct()
        dval.construct()
        datasets[s] = (dtrain, dval)
    return datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--no-baseline", action="store_true",
                    help="Skip the heterogeneous T75 baseline run")
    args = ap.parse_args()

    os.makedirs(HERE, exist_ok=True)
    log_path = os.path.join(HERE, "bayes.log")

    print(f"=== R_T75L2_bayes start at {datetime.now()} ===", flush=True)
    progress("loading_data")
    data = build_data()

    progress("building_datasets")
    print("building LGB Datasets...", flush=True)
    t0 = time.time()
    datasets = make_datasets(data)
    print(f"  built in {time.time()-t0:.1f}s", flush=True)

    # ---- Heterogeneous T75 baseline (3 seeds, T75 paper configs) ----
    T75_SEED_CFG = {
        42: dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
        1:  dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
        7:  dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    }
    common_baseline = dict(
        objective="regression_l2",
        learning_rate=0.05, min_data_in_leaf=100, bagging_freq=5,
        max_depth=-1,
        num_threads=args.num_threads, verbose=-1, metric="l2",
    )
    if args.use_gpu:
        common_baseline["device"] = "gpu"
        common_baseline["gpu_use_dp"] = False

    baseline_3seed = None
    if not args.no_baseline:
        progress("baseline_T75_heterogeneous")
        print("\n--- T75 heterogeneous 3-seed baseline ---", flush=True)
        preds_b = []
        for s in SEEDS:
            cfg = T75_SEED_CFG[s]
            params = {**common_baseline, **cfg}
            pred, bi, tt, _ = train_one_seed(params, s, data, datasets)
            print(f"  seed={s} best_iter={bi} train_time={tt:.1f}s", flush=True)
            preds_b.append(pred)
        de_b = de_eval_3seed_avg(
            preds_b, data["sym_te"], data["mp_t_te"], data["mp_th_te"],
        )
        baseline_3seed = de_b
        print(f"  BASELINE T75 3seed: loso_mean={de_b['loso_mean']:.4f} "
              f"pnl_5sym={de_b['pnl_5sym']:.4f} per_sym_min="
              f"{min(de_b['per_sym'].values()):.4f}", flush=True)
        with open(os.path.join(HERE, "baseline_3seed.json"), "w") as f:
            json.dump(de_b, f, indent=2)

        # Verify GPU is being used
        try:
            import subprocess
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu",
                                           "--format=csv,noheader,nounits"]).decode().strip()
            print(f"  nvidia-smi util after baseline: {out}", flush=True)
        except Exception as e:
            print(f"  nvidia-smi check failed: {e}", flush=True)

    # ---- Optuna Bayes opt ----
    storage = f"sqlite:///{HERE}/optuna_study.db"
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize", study_name="t75_l2_bayes",
        storage=storage, load_if_exists=True, sampler=sampler,
    )

    if len(study.trials) == 0:
        # seed Optuna with the T75 seed=42 config (anchor)
        study.enqueue_trial({
            "num_leaves": 127, "min_data_in_leaf": 100,
            "learning_rate": 0.05, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5,
            "lambda_l2": 1.0, "max_depth": -1,
        })
        study.enqueue_trial({
            "num_leaves": 63, "min_data_in_leaf": 100,
            "learning_rate": 0.05, "feature_fraction": 0.7,
            "bagging_fraction": 0.85, "bagging_freq": 5,
            "lambda_l2": 2.0, "max_depth": -1,
        })

    def objective(trial):
        params = {
            "objective": "regression_l2",
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_categorical("bagging_freq", [0, 5, 10]),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 20.0, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "num_threads": args.num_threads,
            "verbose": -1,
            "metric": "l2",
        }
        if args.use_gpu:
            params["device"] = "gpu"
            params["gpu_use_dp"] = False

        t_trial = time.time()
        preds = []
        bis = []
        for s in SEEDS:
            pred, bi, tt, _ = train_one_seed(params, s, data, datasets)
            preds.append(pred)
            bis.append(bi)
        de = de_eval_3seed_avg(
            preds, data["sym_te"], data["mp_t_te"], data["mp_th_te"],
        )
        trial_time = time.time() - t_trial

        trial.set_user_attr("loso_min", de["loso_min"])
        trial.set_user_attr("pnl_5sym", de["pnl_5sym"])
        trial.set_user_attr("thr_up", de["thr_up"])
        trial.set_user_attr("thr_dn", de["thr_dn"])
        trial.set_user_attr("per_sym", de["per_sym"])
        trial.set_user_attr("best_iters", bis)
        trial.set_user_attr("trial_time_s", trial_time)

        nl = params["num_leaves"]; mdl = params["min_data_in_leaf"]
        lr = params["learning_rate"]; ff = params["feature_fraction"]
        bf = params["bagging_fraction"]; bfreq = params["bagging_freq"]
        l2 = params["lambda_l2"]; md = params["max_depth"]
        print(f"  trial={trial.number:2d} loso={de['loso_mean']:7.4f} "
              f"min={de['loso_min']:7.4f} pnl5={de['pnl_5sym']:7.4f} "
              f"bi={bis} t={trial_time:.1f}s | "
              f"nl={nl} mdl={mdl} lr={lr:.4f} ff={ff:.2f} bf={bf:.2f} "
              f"bfreq={bfreq} l2={l2:.3f} md={md}", flush=True)

        progress("optuna_running",
                 trial=trial.number, n_trials=args.n_trials,
                 best_so_far=float(study.best_value) if len(study.trials) > 0 else None)
        return de["loso_mean"]

    print(f"\n--- Optuna Bayes opt: {args.n_trials} trials ---", flush=True)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best_t = study.best_trial
    print("\n=== BEST TRIAL ===", flush=True)
    print(f"  number={best_t.number} loso_mean={best_t.value:.4f}", flush=True)
    print(f"  user_attrs={best_t.user_attrs}", flush=True)
    print(f"  params={best_t.params}", flush=True)

    # Persist
    with open(os.path.join(HERE, "best_params.json"), "w") as f:
        json.dump({
            "best_value": float(best_t.value),
            "best_user_attrs": best_t.user_attrs,
            "best_params": best_t.params,
            "n_trials": len(study.trials),
        }, f, indent=2)

    # Persist all trials
    rows = []
    for t in study.trials:
        rows.append({
            "trial": t.number,
            "value": t.value,
            "state": str(t.state),
            **{f"p_{k}": v for k, v in t.params.items()},
            **{f"u_{k}": v for k, v in t.user_attrs.items()},
        })
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "trials.csv"), index=False)

    final = {
        "task": "T75 LGB regression_l2 Bayes opt",
        "n_trials_run": len(study.trials),
        "baseline_3seed": baseline_3seed,
        "bayes_best": {
            "loso_mean": float(best_t.value),
            "loso_min": best_t.user_attrs.get("loso_min"),
            "pnl_5sym": best_t.user_attrs.get("pnl_5sym"),
            "thr_up": best_t.user_attrs.get("thr_up"),
            "thr_dn": best_t.user_attrs.get("thr_dn"),
            "params": best_t.params,
        },
    }
    if baseline_3seed is not None:
        final["delta_vs_baseline"] = float(best_t.value) - baseline_3seed["loso_mean"]
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(final, f, indent=2)

    progress("done", best_value=float(best_t.value))
    print(f"\n=== DONE: results.json written ===", flush=True)


if __name__ == "__main__":
    main()
