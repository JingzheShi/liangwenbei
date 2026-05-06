"""T6: Multi-seed LOSO trainer for LightGBM (Scheme A or B).

For each (seed, held_out_sym) pair:
    train: (sym != held_out, dates 0..79)
    val:   (sym != held_out, dates 80..95)  -- early stopping
    test:  (sym == held_out, dates 96..119) -- held-out predictions

Outputs (per seed):
    workdir/T6_ensemble_multiseed/loso_pred_scheme{A,B}_seed{S}_held{K}.parquet
    workdir/T6_ensemble_multiseed/loso_model_scheme{A,B}_seed{S}_held{K}.txt

Usage:
    python loso_train_seeds.py --scheme A --seeds 1,7,13,100
    python loso_train_seeds.py --scheme B --seeds 1,7
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
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "experiments", "T2_gbdt_lgbm", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)


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


def load_split(scheme: str, split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"scheme{scheme}_{split}.npz")
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
              "f0_5_macro"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"    {k:25s}: {v:.6g}", flush=True)
        else:
            print(f"    {k:25s}: {v}", flush=True)
    return m


def train_one_fold(scheme, seed, held_out_sym, train_full, val_full, test_full,
                   feat_names, args):
    m_tr = train_full["sym"] != held_out_sym
    m_va = val_full["sym"] != held_out_sym
    m_te = test_full["sym"] == held_out_sym

    X_tr = train_full["X"][m_tr]
    y_tr = train_full["y60"][m_tr].astype(np.int64)

    X_va = val_full["X"][m_va]
    y_va = val_full["y60"][m_va].astype(np.int64)

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
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
    }

    t_train_start = time.time()
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    train_time = time.time() - t_train_start
    print(f"  fold seed={seed} held={held_out_sym} trained in {train_time:.1f}s, "
          f"best_iter={booster.best_iteration}", flush=True)

    te_pred, te_prob = predict_to_class_and_proba(booster, X_te)
    te_metrics = evaluate(f"HELD-OUT TEST (seed={seed} sym=={held_out_sym})",
                          te_pred, y_te, mp_t_te, mp_t60_te)

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"][m_te].astype(np.int8),
        "date": test_full["date"][m_te].astype(np.int16),
        "session": np.array([sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object),
        "t": test_full["t"][m_te].astype(np.int16),
        "true_label_60": y_te.astype(np.int8),
        "pred_label_60": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_t60": mp_t60_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"loso_pred_scheme{scheme}_seed{seed}_held{held_out_sym}.parquet")
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(HERE, f"loso_model_scheme{scheme}_seed{seed}_held{held_out_sym}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "seed": seed,
        "held_out_sym": held_out_sym,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "held_out_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_out_acc": float(te_metrics["accuracy"]),
        "held_out_n_active": int(te_metrics["n_predictions_active"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=("A", "B"), required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated seeds, e.g. 1,7,13,100")
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
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--syms", default="0,1,2,3,4")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]
    scheme = args.scheme
    print(f"=== T6 multi-seed LOSO scheme {scheme}, seeds = {seeds}, syms = {target_syms} ===", flush=True)

    progress("loading_caches", scheme=scheme, seeds=seeds)
    t0 = time.time()
    train_full = load_split(scheme, "train")
    val_full = load_split(scheme, "val")
    test_full = load_split(scheme, "test")
    print(f"loaded caches in {time.time() - t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, f"scheme{scheme}_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    print(f"feat dim = {len(feat_names)}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T6-multiseed-scheme{scheme}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": scheme,
                    "horizon": "label_60",
                    "seeds": seeds,
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T6-multiseed", f"scheme-{scheme}", "label_60"],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback to personal", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    all_results = []
    n_total = len(seeds) * len(target_syms)
    done = 0
    for seed in seeds:
        for held in target_syms:
            done += 1
            print(f"\n{'='*78}", flush=True)
            print(f"=== [{done}/{n_total}] seed={seed} held_out_sym={held} ===", flush=True)
            print(f"{'='*78}", flush=True)
            progress("training", scheme=scheme, seed=seed, held=held, done=done, total=n_total)
            r = train_one_fold(scheme, seed, held, train_full, val_full, test_full, feat_names, args)
            all_results.append(r)
            if use_wandb:
                wandb.log({
                    f"fold/seed_{seed}_held_{held}_cum_pnl": r["held_out_cum_pnl"],
                    f"fold/seed_{seed}_held_{held}_acc": r["held_out_acc"],
                    f"fold/seed_{seed}_held_{held}_best_iter": r["best_iter"],
                })

    summary_path = os.path.join(HERE, f"loso_train_seeds_scheme{scheme}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"scheme": scheme, "seeds": seeds, "results": all_results,
                   "elapsed_sec": time.time() - t0}, f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)
    progress("done", scheme=scheme, n_runs=len(all_results))

    if use_wandb:
        df = pd.DataFrame(all_results)
        for seed in seeds:
            sub = df[df.seed == seed]
            wandb.summary[f"seed_{seed}_loso_sum_argmax"] = float(sub["held_out_cum_pnl"].sum())
        wandb.finish()


if __name__ == "__main__":
    main()
