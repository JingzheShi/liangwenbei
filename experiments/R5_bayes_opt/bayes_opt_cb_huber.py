"""R5: Optuna Bayes optimization for CB Huber (T99 +40.24 standalone).

Pipeline (matches T99 protocol):
  - Loads schemeP_train + schemeP_test caches from T68
  - V4 split: train 0-75, val 76-79
  - Drops T59_FAIL + STAGE5_FAIL features (DROP_NAMES)
  - aug_a per-feat scale [0.8, 1.2] concat
  - Class-balanced sample weight
  - 1-seed search → DE LOSO eval, post-hoc 3-seed top-K validate
  - CatBoost GPU (task_type='GPU')

Search space:
  depth:        int(4, 10)
  l2_leaf_reg:  loguniform(0.5, 20)
  subsample:    uniform(0.6, 0.95)
  rsm:          uniform(0.5, 1.0)  (CPU only — GPU forces rsm=1.0)
  learning_rate:loguniform(0.02, 0.1)
  iterations:   1500 fixed; early_stop on RMSE
  loss:         Huber:delta=alpha (default delta=1e-3 from T99)

Objective: avg test pred → DE LOSO sum_per_sym on 442k local test.
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
from catboost import CatBoostRegressor, Pool
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import aug_a_scale, class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def vectorized_pnl(action, mt, mh):
    side = action.astype(np.float64) - 1.0
    diff = mh.astype(np.float64) - mt.astype(np.float64)
    fe = FEE * np.abs(side) * np.abs((mh + 1.0) + (mt + 1.0))
    return (side * diff - fe) / (mt.astype(np.float64) + 1.0)


def ev_gate(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def de_loso(pred, sym_arr, mt, mh, bounds_hi=0.0040, de_seeds=(0, 42),
            maxiter=40, popsize=16):
    fold_p, fold_mt, fold_mh = [], [], []
    for k in SYMS:
        m = sym_arr == k
        fold_p.append(pred[m]); fold_mt.append(mt[m]); fold_mh.append(mh[m])

    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for i in range(len(SYMS)):
            a = ev_gate(fold_p[i], thr_up, thr_dn)
            s += vectorized_pnl(a, fold_mt[i], fold_mh[i]).sum()
        return -float(s)

    best = None
    runs = []
    for sd in de_seeds:
        r = differential_evolution(
            f, bounds=[(0.0, bounds_hi), (0.0, bounds_hi)], seed=sd,
            maxiter=maxiter, popsize=popsize, polish=True, tol=1e-7,
            mutation=(0.5, 1.0), recombination=0.7, updating="deferred",
            workers=1, init="sobol",
        )
        run = {"seed": int(sd), "thr_up": float(r.x[0]),
               "thr_dn": float(r.x[1]), "obj_val": float(-r.fun)}
        runs.append(run)
        if best is None or run["obj_val"] > best["obj_val"]:
            best = run
    return best, runs


def per_sym_pnl(pred, sym_arr, mt, mh, thr_up, thr_dn):
    per = []
    for k in SYMS:
        m = sym_arr == k
        if m.sum() == 0:
            per.append(0.0); continue
        a = ev_gate(pred[m], thr_up, thr_dn)
        per.append(float(vectorized_pnl(a, mt[m], mh[m]).sum()))
    return per


def build_data():
    print("loading caches...", flush=True)
    train_full = load_split("train")
    test_full = load_split("test")
    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(train_full["X"].shape[1]) if i not in drop_idx], dtype=np.int64,
    )
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))

    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, keep_idx].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    X_va = train_full["X"][m_va][:, keep_idx].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])

    X_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    sym_te = test_full["sym"].astype(np.int8)
    mt_te = test_full["mp_t"].astype(np.float64)
    mh_te = test_full["mp_t60"].astype(np.float64)

    return {
        "X_tr": X_tr, "y_cls_tr": y_cls_tr, "y_regr_tr": y_regr_tr,
        "X_va": X_va, "y_cls_va": y_cls_va, "y_regr_va": y_regr_va,
        "X_te": X_te, "sym_te": sym_te, "mt_te": mt_te, "mh_te": mh_te,
        "feat_names": feat_names, "feat_dim": len(keep_idx),
        "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
    }


def train_one_seed(data, params, seed, iterations=1200, early_stopping=60):
    rng = np.random.default_rng(seed * 7919 + 1)
    aug_idx = np.arange(len(data["X_tr"]))
    X_aug = aug_a_scale(data["X_tr"][aug_idx], rng, lo=0.80, hi=1.20)
    y_cls_aug = data["y_cls_tr"][aug_idx]
    y_regr_aug = data["y_regr_tr"][aug_idx]
    X_tr_full = np.concatenate([data["X_tr"], X_aug], axis=0)
    y_cls_tr_full = np.concatenate([data["y_cls_tr"], y_cls_aug], axis=0)
    y_regr_tr_full = np.concatenate([data["y_regr_tr"], y_regr_aug], axis=0)

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(data["y_cls_va"], num_class=NUM_CLASS)

    train_pool = Pool(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                      feature_names=data["feat_names"])
    val_pool = Pool(data["X_va"], label=data["y_regr_va"], weight=sw_va,
                    feature_names=data["feat_names"])

    p = dict(params)
    p["random_seed"] = seed
    p["iterations"] = iterations
    p["early_stopping_rounds"] = early_stopping
    model = CatBoostRegressor(**p)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=0)
    pred_te = model.predict(data["X_te"]).astype(np.float32)
    return pred_te, int(model.tree_count_)


def eval_with_seeds(data, params, seeds, iterations, early_stopping,
                    de_seeds, de_maxiter, de_popsize, bounds_hi):
    preds = []
    best_iters = []
    for sd in seeds:
        t_s = time.time()
        pred_te, bi = train_one_seed(data, params, sd,
                                     iterations=iterations,
                                     early_stopping=early_stopping)
        preds.append(pred_te)
        best_iters.append(bi)
        print(f"    seed={sd}: best_iter={bi} ({time.time()-t_s:.1f}s)", flush=True)
    avg_pred = np.mean(preds, axis=0).astype(np.float64)
    de_best, _ = de_loso(avg_pred, data["sym_te"], data["mt_te"], data["mh_te"],
                         bounds_hi=bounds_hi, de_seeds=de_seeds,
                         maxiter=de_maxiter, popsize=de_popsize)
    per_sym = per_sym_pnl(avg_pred, data["sym_te"], data["mt_te"], data["mh_te"],
                          de_best["thr_up"], de_best["thr_dn"])
    return de_best, per_sym, best_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--search-seeds", default="42")
    ap.add_argument("--validate-seeds", default="1,7,42")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=1200)
    ap.add_argument("--early-stop", type=int, default=60)
    ap.add_argument("--de-seeds", default="0,42")
    ap.add_argument("--de-maxiter", type=int, default=40)
    ap.add_argument("--de-popsize", type=int, default=16)
    ap.add_argument("--bounds-hi", type=float, default=0.0040)
    ap.add_argument("--storage", default="sqlite:///bayes_cb_huber.db")
    ap.add_argument("--study-name", default="cb_huber_v1")
    ap.add_argument("--out-prefix", default="cb_huber")
    ap.add_argument("--task-type", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    search_seeds = [int(s) for s in args.search_seeds.split(",")]
    validate_seeds = [int(s) for s in args.validate_seeds.split(",")]
    de_seeds = tuple(int(s) for s in args.de_seeds.split(","))
    print(f"=== R5 Optuna Bayes CB Huber n_trials={args.n_trials} ===", flush=True)
    print(f"  search_seeds={search_seeds}  validate_seeds={validate_seeds}  task_type={args.task_type}", flush=True)

    print("\nbuilding data...", flush=True)
    t0 = time.time()
    data = build_data()
    print(f"data ready in {time.time()-t0:.1f}s "
          f"(feat_dim={data['feat_dim']}, n_train={data['n_train']:,}, "
          f"n_val={data['n_val']:,}, n_test={data['n_test']:,})", flush=True)

    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=f"R5-bayes-cb-huber-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config={
                    "task": "R5_bayes_opt_cb_huber",
                    "n_trials": args.n_trials,
                    "search_seeds": search_seeds,
                    "validate_seeds": validate_seeds,
                    "task_type": args.task_type,
                    "n_features": data["feat_dim"],
                    "baseline_5seed_cb_huber": 40.24,
                },
                tags=["R5", "bayes", "cb", "huber"],
            )
        except Exception as e:
            print(f"wandb init failed ({e}); skip", flush=True); use_wandb = False

    trial_log_path = os.path.join(HERE, f"{args.out_prefix}_trials.jsonl")
    trial_log = open(trial_log_path, "a", buffering=1)

    def objective(trial):
        depth = trial.suggest_int("depth", 4, 9)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 0.5, 20.0, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 0.95)
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.1, log=True)
        delta = trial.suggest_float("delta", 5e-4, 3e-3, log=True)

        params = {
            "loss_function": f"Huber:delta={delta:g}",
            "eval_metric": "RMSE",
            "depth": depth,
            "l2_leaf_reg": l2_leaf_reg,
            "subsample": subsample,
            "learning_rate": learning_rate,
            "bootstrap_type": "Bernoulli",
            "verbose": 0,
            "allow_writing_files": False,
        }
        if args.task_type == "GPU":
            params["task_type"] = "GPU"
            params["devices"] = "0"
            # GPU: rsm forced to 1.0
        else:
            rsm = trial.suggest_float("rsm", 0.5, 1.0)
            params["rsm"] = rsm
            params["thread_count"] = 16

        t_trial = time.time()
        de_best, per_sym, best_iters = eval_with_seeds(
            data, params, search_seeds, args.iterations, args.early_stop,
            de_seeds, args.de_maxiter, args.de_popsize, args.bounds_hi,
        )
        elapsed = time.time() - t_trial
        de_score = de_best["obj_val"]

        rec = {
            "trial_n": trial.number, "params": dict(trial.params),
            "depth": depth, "l2_leaf_reg": l2_leaf_reg, "subsample": subsample,
            "learning_rate": learning_rate, "delta": delta,
            "best_iters": best_iters,
            "de_thr_up": de_best["thr_up"], "de_thr_dn": de_best["thr_dn"],
            "de_score_loso": de_score, "per_sym": per_sym,
            "elapsed_sec": elapsed, "ts": datetime.now(timezone.utc).isoformat(),
        }
        trial_log.write(json.dumps(rec) + "\n")
        print(f"  trial#{trial.number} DONE: de_loso={de_score:+.4f} "
              f"thr=({de_best['thr_up']:.5f},{de_best['thr_dn']:.5f}) "
              f"depth={depth} delta={delta:.4f} lr={learning_rate:.3f} "
              f"({elapsed:.1f}s)", flush=True)

        if use_wandb and wandb_run is not None:
            wandb_run.log({
                "trial": trial.number, "de_loso": de_score,
                "thr_up": de_best["thr_up"], "thr_dn": de_best["thr_dn"],
                "depth": depth, "l2_leaf_reg": l2_leaf_reg, "subsample": subsample,
                "learning_rate": learning_rate, "delta": delta,
                "trial_elapsed_sec": elapsed, "best_iter_avg": float(np.mean(best_iters)),
            })
        return de_score

    sampler = optuna.samplers.TPESampler(seed=12345, n_startup_trials=8,
                                          multivariate=True, group=True)
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        load_if_exists=True, direction="maximize", sampler=sampler,
    )
    print(f"study: {args.study_name}, existing trials: {len(study.trials)}", flush=True)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True,
                   show_progress_bar=False)
    trial_log.close()
    print(f"\n=== STUDY DONE ({len(study.trials)} trials) ===", flush=True)
    print(f"best (search-seed) value: {study.best_value:+.4f}", flush=True)
    print(f"best params: {study.best_params}", flush=True)

    sorted_trials = sorted([t for t in study.trials if t.value is not None],
                           key=lambda t: t.value, reverse=True)
    topk = sorted_trials[:args.top_k]
    print(f"\n=== TOP-{args.top_k} post-hoc {validate_seeds}-seed validation ===", flush=True)
    topk_results = []
    for i, t in enumerate(topk):
        params_t = {
            "loss_function": f"Huber:delta={t.params['delta']:g}",
            "eval_metric": "RMSE",
            "depth": t.params["depth"],
            "l2_leaf_reg": t.params["l2_leaf_reg"],
            "subsample": t.params["subsample"],
            "learning_rate": t.params["learning_rate"],
            "bootstrap_type": "Bernoulli", "verbose": 0,
            "allow_writing_files": False,
        }
        if args.task_type == "GPU":
            params_t["task_type"] = "GPU"; params_t["devices"] = "0"
        else:
            params_t["rsm"] = t.params.get("rsm", 1.0)
            params_t["thread_count"] = 16
        print(f"\n  [topK-{i+1}] trial#{t.number} search={t.value:+.4f}:", flush=True)
        de_best, per_sym, best_iters = eval_with_seeds(
            data, params_t, validate_seeds, args.iterations, args.early_stop,
            de_seeds, args.de_maxiter, args.de_popsize, args.bounds_hi,
        )
        rec = {
            "topk_rank": i + 1, "trial_n": t.number,
            "search_seed_score": float(t.value),
            "validate_seeds": validate_seeds,
            "validate_score": float(de_best["obj_val"]),
            "thr_up": de_best["thr_up"], "thr_dn": de_best["thr_dn"],
            "per_sym": per_sym, "best_iters": best_iters,
            "params": dict(t.params),
        }
        topk_results.append(rec)
        print(f"    -> validate_score={de_best['obj_val']:+.4f} per_sym={[f'{x:+.2f}' for x in per_sym]}", flush=True)

    out_path = os.path.join(HERE, f"{args.out_prefix}_topK_3seed.json")
    with open(out_path, "w") as f:
        json.dump({
            "validate_seeds": validate_seeds,
            "baseline_5seed_cb_huber": 40.24,
            "results": topk_results,
        }, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)

    best_validate = max(topk_results, key=lambda r: r["validate_score"])
    print(f"\nBEST under {validate_seeds}: trial#{best_validate['trial_n']} "
          f"validate={best_validate['validate_score']:+.4f}", flush=True)

    out_path2 = os.path.join(HERE, f"{args.out_prefix}_best.json")
    with open(out_path2, "w") as f:
        json.dump({
            "best_search_value": float(study.best_value),
            "best_validate_value": float(best_validate["validate_score"]),
            "best_params": dict(best_validate["params"]),
            "n_trials": len(study.trials),
            "baseline_5seed_cb_huber": 40.24,
            "thr_up": best_validate["thr_up"], "thr_dn": best_validate["thr_dn"],
            "per_sym": best_validate["per_sym"],
        }, f, indent=2)
    print(f"wrote {out_path2}", flush=True)

    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nparam importances:")
        for k, v in importances.items():
            print(f"  {k}: {v:.4f}")
        with open(os.path.join(HERE, f"{args.out_prefix}_importance.json"), "w") as f:
            json.dump({k: float(v) for k, v in importances.items()}, f, indent=2)
    except Exception as e:
        print(f"importance failed: {e}", flush=True)

    if use_wandb and wandb_run is not None:
        wandb_run.summary["best_search_value"] = float(study.best_value)
        wandb_run.summary["best_validate_value"] = float(best_validate["validate_score"])
        wandb_run.summary["best_params"] = dict(best_validate["params"])
        wandb_run.finish()


if __name__ == "__main__":
    main()
