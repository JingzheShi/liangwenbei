"""T22: 5-fold LOSO LightGBM trainer for Scheme I (alphas + raw).

3 sub-schemes selected via --scheme flag:
  Ia  alpha-only:  cols [226:380] = 33 a101 + 124 a191 = 157 cols  (drops T3 time)
  Ib  alpha+raw:   cols [0:380] = full 380-d (raw 154 + T3 69 + alpha 157)
  Ic  curated:     load `curated_features.txt` (relative to HERE) and select.

Single seed (42) like T21b. h=10 default.
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

T22_CACHE = os.path.join(HERE, "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL_TIME = 3  # last 3 cols are time encoding (drop them)

SEED_CFG = dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8,
                num_leaves=127, lambda_l2=1.0)


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running", "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(T22_CACHE, f"schemeI_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def predict_proba(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s:s + batch]).astype(np.float32))
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


def select_columns(scheme, feat_names_full, layout, curated_path=None):
    """Return (col_indices, feat_names_used)."""
    a101_lo, a101_hi = layout["alpha101_33"]
    a191_lo, a191_hi = layout["alpha191_124"]
    raw_lo, raw_hi = layout["raw_154"]
    t3_lo, t3_hi = layout["t3_no_time_69"]

    if scheme == "Ia":
        # alpha only
        idx = list(range(a101_lo, a101_hi)) + list(range(a191_lo, a191_hi))
    elif scheme == "Ib":
        # alpha + raw + T3 (no time enc)
        idx = list(range(raw_lo, a191_hi))
    elif scheme == "Ic":
        if curated_path is None:
            raise ValueError("Ic requires --curated <path>")
        with open(curated_path) as f:
            curated = [line.strip() for line in f if line.strip()]
        name_to_idx = {n: i for i, n in enumerate(feat_names_full)}
        idx = [name_to_idx[n] for n in curated if n in name_to_idx]
    else:
        raise ValueError(scheme)
    feat_names_used = [feat_names_full[i] for i in idx]
    return np.array(idx, dtype=np.int64), feat_names_used


def train_one(H, held, train_full, val_full, test_full, col_idx, feat_names, args):
    seed = int(SEED_CFG["seed"])
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr = train_full["X"][m_tr][:, col_idx]
    y_tr = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va][:, col_idx]
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te][:, col_idx]
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[f"mp_t{H}"][m_te]

    print(f"    n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]}",
          flush=True)
    sw_tr = class_balanced_weight(y_tr)
    sw_va = class_balanced_weight(y_va)

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASS,
        "metric": "multi_logloss",
        "learning_rate": args.learning_rate,
        "num_leaves": int(SEED_CFG["num_leaves"]),
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": float(SEED_CFG["feature_fraction"]),
        "bagging_fraction": float(SEED_CFG["bagging_fraction"]),
        "bagging_freq": args.bagging_freq,
        "lambda_l2": float(SEED_CFG["lambda_l2"]),
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
        params["max_bin"] = 63

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
    print(f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}", flush=True)

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(f"HELD-OUT TEST h={H} {args.scheme} held={held}",
                          te_pred, y_te, mp_t_te, mp_th_te)

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"][m_te].astype(np.int8),
        "date": test_full["date"][m_te].astype(np.int16),
        "session": np.array([sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object),
        "t": test_full["t"][m_te].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        "pred_label": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"loso_pred_{args.scheme}_h{H}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)
    model_path = os.path.join(HERE, f"loso_model_{args.scheme}_h{H}_held{held}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    fi_gain = booster.feature_importance(importance_type="gain")
    fi_split = booster.feature_importance(importance_type="split")
    return {
        "scheme": args.scheme,
        "horizon": H,
        "seed": seed,
        "held_out_sym": held,
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "held_out_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_out_acc": float(te_metrics["accuracy"]),
        "held_out_n_active": int(te_metrics["n_predictions_active"]),
        "feat_imp_gain": fi_gain.tolist(),
        "feat_imp_split": fi_split.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=("Ia", "Ib", "Ic"), required=True)
    ap.add_argument("--curated", default=None,
                    help="path to curated feature list (one name per line) — required for Ic")
    ap.add_argument("--horizons", default="10")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    if args.no_gpu:
        args.use_gpu = False

    horizons = [int(x) for x in args.horizons.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]

    feat_names_path = os.path.join(T22_CACHE, "schemeI_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names_full = [line.strip() for line in f]
    layout = json.load(open(os.path.join(T22_CACHE, "schemeI_layout.json")))

    col_idx, feat_names = select_columns(args.scheme, feat_names_full, layout, args.curated)

    forbidden = {"date", "sym"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features: {leak}"
    print(f"=== T22 LOSO Scheme {args.scheme} {len(feat_names)}-d, GPU={args.use_gpu}, "
          f"horizons={horizons}, syms={target_syms} ===", flush=True)

    progress("loading_caches", scheme=args.scheme, horizons=horizons)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time() - t0:.1f}s; "
          f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
          flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T22-{args.scheme}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            try:
                wandb.init(
                    project="liangwenbei",
                    entity="cjxh21-Tsinghua University",
                    name=run_name,
                    config={
                        "scheme": args.scheme,
                        "n_features": len(feat_names),
                        "horizons": horizons,
                        "n_a101": layout["n_a101"],
                        "n_a191": layout["n_a191"],
                        **{k: v for k, v in vars(args).items() if k not in ("no_wandb",)},
                    },
                    tags=["T22", "alpha101", "alpha191", args.scheme],
                )
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback personal", flush=True)
                wandb.init(project="liangwenbei", name=run_name)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    all_results = []
    n_total = len(horizons) * len(target_syms)
    done = 0
    for H in horizons:
        print(f"\n{'=' * 78}\n=== HORIZON h={H} ===\n{'=' * 78}", flush=True)
        for held in target_syms:
            done += 1
            tag = f"{args.scheme}_h{H}_held{held}"
            print(f"\n  [{done}/{n_total}] fold {tag}", flush=True)
            progress("training", scheme=args.scheme, horizon=H, held=held,
                     done=done, total=n_total)
            r = train_one(H, held, train_full, val_full, test_full, col_idx, feat_names, args)
            all_results.append(r)
            if use_wandb:
                wandb.log({
                    f"{args.scheme}/h{H}/fold{held}/cum_pnl": r["held_out_cum_pnl"],
                    f"{args.scheme}/h{H}/fold{held}/acc": r["held_out_acc"],
                    f"{args.scheme}/h{H}/fold{held}/best_iter": r["best_iter"],
                })

    summary_path = os.path.join(HERE, f"loso_summary_{args.scheme}.json")
    with open(summary_path, "w") as f:
        compact = []
        for r in all_results:
            cr = {k: v for k, v in r.items() if not k.startswith("feat_imp_")}
            compact.append(cr)
        json.dump(sanitize_for_json({
            "task": f"T22 LOSO Scheme {args.scheme}",
            "scheme": args.scheme,
            "n_features": len(feat_names),
            "horizons": horizons,
            "target_syms": target_syms,
            "results": compact,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    if all_results:
        gain_arr = np.stack([np.array(r["feat_imp_gain"]) for r in all_results], axis=0)
        split_arr = np.stack([np.array(r["feat_imp_split"]) for r in all_results], axis=0)
        fi_df = pd.DataFrame({
            "feat_name": feat_names,
            "gain_sum_mean": gain_arr.mean(axis=0),
            "gain_sum_std": gain_arr.std(axis=0),
            "split_count_mean": split_arr.mean(axis=0),
        }).sort_values("gain_sum_mean", ascending=False)
        fi_path = os.path.join(HERE, f"feature_importance_{args.scheme}.csv")
        fi_df.to_csv(fi_path, index=False)
        print(f"feature importance -> {fi_path}", flush=True)
        print(f"\n=== TOP 30 BY GAIN (scheme {args.scheme}) ===", flush=True)
        for i, row in fi_df.head(30).iterrows():
            print(f"  {row['feat_name']:38s}  gain={row['gain_sum_mean']:.0f}  "
                  f"split={row['split_count_mean']:.0f}", flush=True)

    progress("loso_done", scheme=args.scheme, n_runs=len(all_results))
    if use_wandb:
        df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("feat_imp_")}
                           for r in all_results])
        for H in horizons:
            sub = df[df.horizon == H]
            wandb.summary[f"{args.scheme}_h{H}_sum_cum_pnl"] = float(sub["held_out_cum_pnl"].sum())
        wandb.finish()


if __name__ == "__main__":
    main()
