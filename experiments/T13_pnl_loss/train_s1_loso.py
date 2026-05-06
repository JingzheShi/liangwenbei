"""T13 S1 — Sample-weighted CE by |Δp|: 5-fold LOSO on Scheme C cache, h=10.

Identical to T5b/train_loso.py except:
  - Weight per sample = clip(|Δp_h|/(1+mp_t), 0, q99(.))  computed from the
    *training* fold only (no leakage across folds, no use of val/test stats).
  - We replace the class_balanced_weight with the |Δp| weight.

Usage:
    python3 train_s1_loso.py --horizon 10
    python3 train_s1_loso.py --horizon 10 --weight-mode raw_abs   # |Δp| only
    python3 train_s1_loso.py --horizon 10 --weight-mode normalized   # (default)
    python3 train_s1_loso.py --horizon 10 --weight-mode hybrid      # cls_balance × |Δp|
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

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "T5b_features_multihorizon", "cache"))
NUM_CLASS = 3


def load_split(split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def abs_dp_weight(mp_t: np.ndarray, mp_th: np.ndarray, mode: str = "normalized",
                  q: float = 0.99, eps: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Compute per-sample weight from |Δp_h|.

    Modes:
      - raw_abs:    weight = clip(|Δp|, 0, q99)
      - normalized: weight = clip(|Δp|/(1+mp_t), 0, q99)   <-- aligned with PnL eval
      - hybrid:     cls_balanced × normalized
    """
    diff = (mp_th.astype(np.float64) - mp_t.astype(np.float64))
    if mode == "raw_abs":
        w = np.abs(diff)
    elif mode in ("normalized", "hybrid"):
        denom = mp_t.astype(np.float64) + 1.0
        denom = np.where(denom <= 0, 1.0, denom)  # safety; midprice should never be ≤ -1
        w = np.abs(diff) / denom
    else:
        raise ValueError(f"unknown mode {mode}")
    cap = float(np.quantile(w, q))
    w_clipped = np.minimum(w, cap)
    # Floor: avoid weight=0 collapse on flat samples. Use small floor = cap * 1e-3
    # (so training can still partially learn flats, ~1000x downweighted vs tail).
    floor = cap * 1e-3
    w_final = np.maximum(w_clipped, floor).astype(np.float32)
    info = {
        "mode": mode,
        "cap_q99": cap,
        "floor": floor,
        "weight_mean": float(w_final.mean()),
        "weight_std": float(w_final.std()),
        "weight_max": float(w_final.max()),
        "weight_pct_at_floor": float((w_final == floor).mean()),
    }
    return w_final, info


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--weight-mode", default="normalized",
                    choices=["raw_abs", "normalized", "hybrid"])
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
    ap.add_argument("--out-tag", default="s1")
    args = ap.parse_args()

    H = args.horizon
    target_syms = [int(x) for x in args.syms.split(",")]
    print(f"=== T13 S1 weighted-CE LOSO  H={H}  weight={args.weight_mode}  syms={target_syms} ===",
          flush=True)

    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeC_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features in feat_names: {leak}"
    print(f"feat_dim={len(feat_names)}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T13-S1-h{H}-{args.weight_mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "C1",
                    "experiment": "T13-S1-pnl-weighted-ce",
                    "weight_mode": args.weight_mode,
                    "horizon": H,
                    "n_features": len(feat_names),
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T13", "S1", "pnl-loss", f"h{H}", args.weight_mode],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback personal", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    y_key = f"y{H}"
    mp_th_key = f"mp_t{H}"
    fold_results = {}

    for held in target_syms:
        tag = f"h{H}_held{held}"
        print(f"\n  --- fold {tag} ---", flush=True)
        m_tr = train_full["sym"] != held
        m_va = val_full["sym"] != held
        m_te = test_full["sym"] == held

        X_tr = train_full["X"][m_tr]
        y_tr = train_full[y_key][m_tr].astype(np.int64)
        mp_t_tr = train_full["mp_t"][m_tr]
        mp_th_tr = train_full[mp_th_key][m_tr]

        X_va = val_full["X"][m_va]
        y_va = val_full[y_key][m_va].astype(np.int64)
        mp_t_va = val_full["mp_t"][m_va]
        mp_th_va = val_full[mp_th_key][m_va]

        X_te = test_full["X"][m_te]
        y_te = test_full[y_key][m_te].astype(np.int64)
        mp_t_te = test_full["mp_t"][m_te]
        mp_th_te = test_full[mp_th_key][m_te]

        print(f"    n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(X_te):,}",
              flush=True)

        # Compute |Δp| weights from training fold only (no leak)
        sw_tr_dp, info_tr = abs_dp_weight(mp_t_tr, mp_th_tr, mode=args.weight_mode)
        # Val: same cap from train fold (informative early-stopping needs same dist)
        # Use train cap on val (to avoid val leakage — though val cap differs little)
        train_cap = info_tr["cap_q99"]
        if args.weight_mode == "raw_abs":
            wv_raw = np.abs(mp_th_va - mp_t_va)
        else:
            denom_va = mp_t_va.astype(np.float64) + 1.0
            denom_va = np.where(denom_va <= 0, 1.0, denom_va)
            wv_raw = np.abs((mp_th_va - mp_t_va).astype(np.float64)) / denom_va
        sw_va_dp = np.maximum(np.minimum(wv_raw, train_cap),
                              train_cap * 1e-3).astype(np.float32)

        if args.weight_mode == "hybrid":
            cls_w_tr = class_balanced_weight(y_tr)
            cls_w_va = class_balanced_weight(y_va)
            sw_tr = (sw_tr_dp * cls_w_tr).astype(np.float32)
            sw_va = (sw_va_dp * cls_w_va).astype(np.float32)
        else:
            sw_tr = sw_tr_dp
            sw_va = sw_va_dp

        print(f"    weight cap_q99={info_tr['cap_q99']:.4g} mean={info_tr['weight_mean']:.4g} "
              f"frac_at_floor={info_tr['weight_pct_at_floor']:.3f}", flush=True)

        dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                             feature_name=feat_names, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                           feature_name=feat_names, reference=dtrain, free_raw_data=False)

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

        t_start = time.time()
        booster = lgb.train(
            params, dtrain,
            num_boost_round=args.num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        train_time = time.time() - t_start
        print(f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
              flush=True)

        te_prob = predict_proba(booster, X_te)
        te_pred = te_prob.argmax(axis=1).astype(np.int8)
        te_metrics = evaluate(f"HELD-OUT TEST h={H} held={held}",
                              te_pred, y_te, mp_t_te, mp_th_te)
        va_prob = predict_proba(booster, X_va)
        va_pred = va_prob.argmax(axis=1).astype(np.int8)
        va_metrics = evaluate(f"VAL h={H} held={held}",
                              va_pred, y_va, mp_t_va, mp_th_va)

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
        pred_path = os.path.join(HERE, f"s1_pred_{tag}_{args.weight_mode}.parquet")
        te_df.to_parquet(pred_path, index=False)

        model_path = os.path.join(HERE, f"s1_model_{tag}_{args.weight_mode}.txt")
        booster.save_model(model_path, num_iteration=booster.best_iteration)

        fold_results[held] = {
            "horizon": H,
            "held_out_sym": held,
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "best_iter": int(booster.best_iteration),
            "train_time_sec": float(train_time),
            "weight_info": info_tr,
            "val": va_metrics,
            "held_out": te_metrics,
        }
        if use_wandb:
            wandb.log({
                f"h{H}/fold{held}/best_iter": int(booster.best_iteration),
                f"h{H}/fold{held}/train_time_sec": float(train_time),
                f"h{H}/fold{held}/val_cum_pnl": va_metrics["cum_pnl"],
                f"h{H}/fold{held}/val_acc": va_metrics["accuracy"],
                f"h{H}/fold{held}/held_cum_pnl": te_metrics["cum_pnl"],
                f"h{H}/fold{held}/held_single_pnl": te_metrics["single_pnl"],
                f"h{H}/fold{held}/held_acc": te_metrics["accuracy"],
                f"h{H}/fold{held}/held_f0_5_macro": te_metrics["f0_5_macro"],
            })

    print(f"\n{'='*78}\n=== Aggregate (raw argmax) ===\n{'='*78}", flush=True)
    cums = [fold_results[k]["held_out"]["cum_pnl"] for k in sorted(fold_results)]
    cums_a = np.array(cums)
    agg = {
        "horizon": H,
        "weight_mode": args.weight_mode,
        "n_folds": len(cums_a),
        "cum_pnl_per_fold": cums_a.tolist(),
        "cum_pnl_sum": float(cums_a.sum()),
        "cum_pnl_mean": float(cums_a.mean()) if len(cums_a) > 0 else 0.0,
        "cum_pnl_std": float(cums_a.std(ddof=0)) if len(cums_a) > 0 else 0.0,
        "n_pos_folds": int((cums_a > 0).sum()) if len(cums_a) > 0 else 0,
    }
    print(f"h={H} ({args.weight_mode}): sum={cums_a.sum():+.4f} mean={cums_a.mean():+.4f} "
          f"std={cums_a.std(ddof=0):.4f} pos={int((cums_a>0).sum())}/{len(cums_a)} "
          f"cums={[f'{x:+.3f}' for x in cums_a]}", flush=True)

    summary = {
        "scheme": "T13-S1",
        "weight_mode": args.weight_mode,
        "params": vars(args),
        "horizon": H,
        "fold_results": fold_results,
        "aggregate": agg,
    }
    out_path = os.path.join(HERE, f"loso_results_s1_h{H}_{args.weight_mode}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.summary.update({
            f"agg_h{H}/cum_pnl_sum": agg["cum_pnl_sum"],
            f"agg_h{H}/cum_pnl_mean": agg["cum_pnl_mean"],
            f"agg_h{H}/cum_pnl_std": agg["cum_pnl_std"],
            f"agg_h{H}/n_pos_folds": agg["n_pos_folds"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
