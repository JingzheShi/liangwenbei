"""T64 time-based val for early stopping — 4 val strategies on R34 Stage 3 schemeN cache.

Strategies:
  V1: val=date 80-95 all 5 syms, metric=multi_logloss        (T59 baseline; can be reused)
  V2: val=date 80-95 all 5 syms, metric=cum_pnl  (custom feval, monitor PnL directly)
  V3: val=date 88-95 (last 8 days) all 5 syms, metric=multi_logloss (tighter time-OOD val)
  V4: val=date 76-79 (last 5% of train range) all 5 syms, train=date 0-75 (walk-forward)

For each strategy: train one seed, save model + full-test predictions parquet.
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

T59_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train")
CACHE_DIR = os.path.join(T59_DIR, "cache")  # reuse T59 cache

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeN_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_proba(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def evaluate(name, preds, y, mp_t, mp_th, fee_rate=FEE):
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


def make_feval_cum_pnl(mp_t, mp_th):
    """LightGBM feval for multiclass: returns ('val_cum_pnl', value, is_higher_better=True).

    LightGBM multiclass preds layout in feval: shape (num_class * num_data,),
    class-major: preds[c*n + i] = score for row i, class c.
    """
    n_data = int(mp_t.shape[0])
    mp_t64 = mp_t.astype(np.float64)
    mp_th64 = mp_th.astype(np.float64)
    diff_const = mp_th64 - mp_t64
    fee_const = FEE * np.abs((mp_th64 + 1.0) + (mp_t64 + 1.0))
    denom_const = mp_t64 + 1.0

    def feval(preds, dataset):
        # LightGBM multiclass feval layout = ROW-MAJOR: preds[i*K + c] is row i, class c.
        # Verified empirically: preds.reshape(n, K).argmax(1) matches booster.predict().argmax(1).
        scores = np.asarray(preds).reshape(n_data, NUM_CLASS)
        pred_cls = np.argmax(scores, axis=1)
        side = pred_cls.astype(np.float64) - 1.0
        abs_side = np.abs(side)
        pnl = (side * diff_const - abs_side * fee_const) / denom_const
        return ("val_cum_pnl", float(pnl.sum()), True)

    return feval


def build_strategy_split(strategy, train_full, val_full, slicer):
    """Returns (X_tr, y_tr, X_va, y_va, mp_t_va, mp_th_va, info)."""
    if isinstance(slicer, slice):
        sl = lambda X: X[:, slicer]
    else:
        sl = lambda X: X[:, slicer]

    if strategy in ("V1", "V2"):
        X_tr = sl(train_full["X"]).astype(np.float32, copy=False)
        y_tr = train_full["y60"].astype(np.int64)
        X_va = sl(val_full["X"]).astype(np.float32, copy=False)
        y_va = val_full["y60"].astype(np.int64)
        mp_t_va = val_full["mp_t"]
        mp_th_va = val_full["mp_t60"]
        info = {"strategy": strategy, "train_dates": "0-79",
                "val_dates": "80-95", "n_train": int(len(X_tr)),
                "n_val": int(len(X_va))}

    elif strategy == "V3":
        date_va = val_full["date"]
        m_va = date_va >= 88
        X_tr = sl(train_full["X"]).astype(np.float32, copy=False)
        y_tr = train_full["y60"].astype(np.int64)
        X_va = sl(val_full["X"][m_va]).astype(np.float32, copy=False)
        y_va = val_full["y60"][m_va].astype(np.int64)
        mp_t_va = val_full["mp_t"][m_va]
        mp_th_va = val_full["mp_t60"][m_va]
        info = {"strategy": strategy, "train_dates": "0-79",
                "val_dates": "88-95 (last 8 days, tighter)",
                "n_train": int(len(X_tr)), "n_val": int(len(X_va))}

    elif strategy == "V4":
        # Walk-forward inside train: last 5% of date range = date 76-79
        date_tr = train_full["date"]
        m_va = date_tr >= 76  # 4 days = 5% of 80
        m_t = ~m_va
        X_tr = sl(train_full["X"][m_t]).astype(np.float32, copy=False)
        y_tr = train_full["y60"][m_t].astype(np.int64)
        X_va = sl(train_full["X"][m_va]).astype(np.float32, copy=False)
        y_va = train_full["y60"][m_va].astype(np.int64)
        mp_t_va = train_full["mp_t"][m_va]
        mp_th_va = train_full["mp_t60"][m_va]
        info = {"strategy": strategy, "train_dates": "0-75",
                "val_dates": "76-79 (walk-forward, last 5% of train)",
                "n_train": int(len(X_tr)), "n_val": int(len(X_va))}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return X_tr, y_tr, X_va, y_va, mp_t_va, mp_th_va, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=["V1", "V2", "V3", "V4"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--variant", default="aug_a")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--n-drop-tail", type=int, default=3)
    ap.add_argument("--drop-fail-extras", action="store_true", default=True)
    ap.add_argument("--out-tag", default="pilot")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    strategy = args.strategy
    print(f"=== T64 strategy={strategy} seed={seed} h={H} ===", flush=True)

    progress("loading_caches", seed=seed, strategy_=strategy)
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

    # R34 Stage 3 default: drop-fail-extras + drop-tail (replicate T59)
    report_path = os.path.join(T59_DIR, "sym_invariance_report.json")
    with open(report_path) as f:
        report = json.load(f)
    fail_names = report["summary"]["fail_names"]
    fail_local = [extra_names.index(n) for n in fail_names]
    fail_global = [base_dim + i for i in fail_local]
    drop_set = set(fail_global)
    for i in range(args.n_drop_tail):
        drop_set.add(base_dim - 1 - i)
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_set], dtype=np.int64,
    )
    slicer = keep_idx
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim} (drop-fail+drop-tail)", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeN_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    feat_names = [all_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak

    # Build strategy-specific train/val
    X_tr, y_tr, X_va, y_va, mp_t_va, mp_th_va, info = build_strategy_split(
        strategy, train_full, val_full, slicer,
    )
    print(f"  {info}", flush=True)

    # Test set always = test_full (date 96-119)
    X_te = train_full["X"][:0, slicer]  # placeholder, computed below
    X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
    y_te = test_full[f"y{H}"].astype(np.int64)
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T64-{strategy}-h{H}-seed{seed}-{args.out_tag}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "strategy": strategy,
                    "split_info": info,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T64", strategy, f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, split_info=info)

    seed_rng = np.random.default_rng(seed * 7919 + 1)
    X_tr_aug, y_tr_aug, sw_tr = build_train_for_variant(
        args.variant, X_tr, y_tr,
        None, None,
        seed_rng,
        aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_aug):,}  n_val={len(X_va):,}", flush=True)

    dtrain = lgb.Dataset(X_tr_aug, label=y_tr_aug, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASS,
        "learning_rate": args.learning_rate,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": args.bagging_freq,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": args.num_threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
    }
    if args.use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    feval = None
    if strategy == "V2":
        params["metric"] = "None"  # disable built-in; rely on feval for early stop
        feval = make_feval_cum_pnl(mp_t_va, mp_th_va)
        print(f"  V2: using custom feval cum_pnl on val ({len(mp_t_va):,} rows)", flush=True)
    else:
        params["metric"] = "multi_logloss"

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        feval=feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t_start
    best_iter = int(booster.best_iteration)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    # Save model
    model_path = os.path.join(HERE, f"model_{strategy}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Eval on val
    va_prob = predict_proba(booster, X_va)
    va_pred = va_prob.argmax(axis=1).astype(np.int8)
    va_metrics = evaluate(f"VAL ({strategy}) seed={seed}", va_pred, y_va, mp_t_va, mp_th_va)

    # Eval on full test set (442k, 5 syms, date 96-119) — same comparison surface as T59
    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(f"TEST full 5-sym ({strategy}) seed={seed}",
                          te_pred, y_te, mp_t_te, mp_th_te)

    # Per-sym test metrics
    per_sym = {}
    for k in SYMS:
        m = sym_te == k
        if m.sum() == 0:
            continue
        mk = _per_horizon_metrics(te_pred[m], y_te[m], mp_t_te[m], mp_th_te[m], fee_rate=FEE)
        per_sym[int(k)] = {
            "n": int(m.sum()),
            "accuracy": float(mk["accuracy"]),
            "cum_pnl": float(mk["cum_pnl"]),
            "single_pnl": float(mk["single_pnl"]),
            "n_active": int(mk["n_predictions_active"]),
            "f0_5_macro": float(mk["f0_5_macro"]),
        }
    loso_equiv = sum(v["cum_pnl"] for v in per_sym.values())
    print(f"  per-sym argmax cum_pnl: "
          f"{[round(per_sym[k]['cum_pnl'],3) for k in SYMS]}  sum={loso_equiv:+.4f}",
          flush=True)

    # Save predictions parquet
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
    pred_path = os.path.join(HERE, f"pred_{strategy}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    summary = {
        "task": f"T64 strategy={strategy} seed={seed}",
        "strategy": strategy,
        "seed": seed,
        "horizon": H,
        "split_info": info,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": va_metrics,
        "test_full": te_metrics,
        "per_sym_test": per_sym,
        "loso_equiv_sum_argmax": float(loso_equiv),
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_{strategy}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_iter": best_iter,
            "train_time_sec": float(train_time),
            "val_cum_pnl": va_metrics["cum_pnl"],
            "val_acc": va_metrics["accuracy"],
            "test_full_cum_pnl": te_metrics["cum_pnl"],
            "test_full_acc": te_metrics["accuracy"],
            "loso_equiv_sum": float(loso_equiv),
        })
        wandb.finish()
    progress("done", seed=seed, strategy_=strategy,
             best_iter=best_iter, test_cum_pnl=te_metrics["cum_pnl"])


if __name__ == "__main__":
    main()
