"""T47: LOSO LightGBM training across 4 horizons {30, 60, 120, 240}.

For each (horizon h, held_out_sym k):
    train: (sym != k, dates 0..79)  + aug_a (per-sample [0.80, 1.20] scaling, 1x ratio)
    val:   (sym != k, dates 80..95) -- early stopping (NO augment)
    test:  (sym == k, dates 96..119) -- held-out (NO augment)

All models share 223-d features (matches iter_006 / T26 aug_a). For h=120/240,
rows where t+H exceeds session length (y=-1) are filtered from training; test
predictions are still produced for ALL test rows so that stacking can use them.

Outputs (per (h, k)):
    loso_pred_h{H}_held{K}.parquet   -- held-out test predictions (probs)
    loso_pred_h{H}_val_held{K}.parquet -- val predictions (for stacking input)
    loso_model_h{H}_held{K}.txt
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
sys.path.insert(0, os.path.join(ROOT, "experiments", "T26_domain_randomization"))

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

from build_aug import (  # noqa: E402
    build_train_for_variant,
    class_balanced_weight,
    compute_feat_stats,
)

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
T47_CACHE = os.path.join(HERE, "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # 226 -> 223 to match iter_006 / T26 aug_a
HORIZONS = (30, 60, 120, 240)


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


def load_split(split: str) -> dict:
    base_p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    extra_p = os.path.join(T47_CACHE, f"schemeC_{split}_extras.npz")
    print(f"  loading {base_p} + {extra_p} ...", flush=True)
    base = np.load(base_p)
    extras = np.load(extra_p)
    out = {k: base[k] for k in base.files}
    for k in extras.files:
        out[k] = extras[k]
    return out


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def evaluate(name, preds, y, mp_t, mp_th, fee_rate=0.0001):
    valid = y != -1
    if valid.sum() == 0:
        print(f"  --- {name} (no valid samples) ---", flush=True)
        return {"cum_pnl": 0.0, "single_pnl": 0.0, "accuracy": 0.0,
                "n_predictions_active": 0, "f0_5_macro": 0.0,
                "pred_distribution": {}, "label_distribution": {}}
    m = _per_horizon_metrics(preds[valid], y[valid], mp_t[valid], mp_th[valid],
                             fee_rate=fee_rate)
    m["pred_distribution"] = dict(m["pred_distribution"])
    m["label_distribution"] = dict(m["label_distribution"])
    print(f"  --- {name} ---", flush=True)
    for k in ("accuracy", "cum_pnl", "single_pnl", "n_predictions_active"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"    {k:25s}: {v:+.6g}", flush=True)
        else:
            print(f"    {k:25s}: {v}", flush=True)
    return m


def train_one(H, held, train_full, val_full, test_full, feat_names, args):
    seed = args.seed
    y_key = f"y{H}"
    mp_th_key = f"mp_t{H}"

    m_tr_sym = train_full["sym"] != held
    m_va_sym = val_full["sym"] != held
    m_te_sym = test_full["sym"] == held

    # Filter -1 rows (only h=120,240 may have them)
    m_tr_valid = train_full[y_key] != -1
    m_va_valid = val_full[y_key] != -1

    m_tr = m_tr_sym & m_tr_valid
    m_va = m_va_sym & m_va_valid
    m_te = m_te_sym  # do NOT filter test by valid - keep all rows for stacking

    X_tr_o = train_full["X"][m_tr][:, :-N_DROP_TAIL]
    y_tr_o = train_full[y_key][m_tr].astype(np.int64)

    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL]
    y_va = val_full[y_key][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[mp_th_key][m_va]

    # All val rows where sym != held (for stacking we need predictions on ALL val rows
    # whose y_60 is valid, regardless of y_h validity)
    m_va_all = m_va_sym  # all val rows in non-held syms
    X_va_all = val_full["X"][m_va_all][:, :-N_DROP_TAIL]

    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL]
    y_te = test_full[y_key][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[mp_th_key][m_te]

    feat_mean, feat_std = compute_feat_stats(X_tr_o)

    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + H * 31)

    X_tr, y_tr, sw_tr = build_train_for_variant(
        "aug_a", X_tr_o, y_tr_o, feat_std, feat_mean,
        fold_rng, aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)

    print(
        f"    h={H} held={held} n_train_orig={len(X_tr_o):,} n_train_used={len(X_tr):,} "
        f"n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]}",
        flush=True,
    )

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
    if args.use_gpu:
        params.update({"device": "gpu", "gpu_use_dp": False})

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
    print(f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
          flush=True)

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(f"HELD-OUT TEST h={H} held={held}",
                          te_pred, y_te, mp_t_te, mp_th_te)

    # Predict on ALL val rows (sym != held), regardless of y_h validity
    va_all_prob = predict_proba(booster, X_va_all)
    va_all_pred = va_all_prob.argmax(axis=1).astype(np.int8)

    # For evaluation print, use only valid rows
    va_subset_prob = va_all_prob[val_full[y_key][m_va_all] != -1]
    va_subset_pred = va_subset_prob.argmax(axis=1).astype(np.int8)
    va_metrics = evaluate(f"VAL h={H} held={held}",
                          va_subset_pred, y_va, mp_t_va, mp_th_va)

    sess_map = {0: "am", 1: "pm"}

    # Save test predictions
    te_df = pd.DataFrame({
        "sym": test_full["sym"][m_te].astype(np.int8),
        "date": test_full["date"][m_te].astype(np.int16),
        "session": np.array([sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object),
        "t": test_full["t"][m_te].astype(np.int16),
        "true_label_h": y_te.astype(np.int8),
        "true_label_60": test_full["y60"][m_te].astype(np.int8),
        "pred_label": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
        "midprice_t60": test_full["mp_t60"][m_te].astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"loso_pred_h{H}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)

    # Save val predictions (for stacking input). Aligned with m_va_all.
    va_df = pd.DataFrame({
        "sym": val_full["sym"][m_va_all].astype(np.int8),
        "date": val_full["date"][m_va_all].astype(np.int16),
        "session": np.array([sess_map[s] for s in val_full["sess_idx"][m_va_all]], dtype=object),
        "t": val_full["t"][m_va_all].astype(np.int16),
        "true_label_h": val_full[y_key][m_va_all].astype(np.int8),
        "true_label_60": val_full["y60"][m_va_all].astype(np.int8),
        "pred_label": va_all_pred.astype(np.int8),
        "prob_0": va_all_prob[:, 0].astype(np.float32),
        "prob_1": va_all_prob[:, 1].astype(np.float32),
        "prob_2": va_all_prob[:, 2].astype(np.float32),
        "midprice_t": val_full["mp_t"][m_va_all].astype(np.float32),
        "midprice_t60": val_full["mp_t60"][m_va_all].astype(np.float32),
    })
    va_path = os.path.join(HERE, f"loso_pred_h{H}_val_held{held}.parquet")
    va_df.to_parquet(va_path, index=False)

    model_path = os.path.join(HERE, f"loso_model_h{H}_held{held}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "horizon": H,
        "held_out_sym": held,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "val_cum_pnl": float(va_metrics["cum_pnl"]),
        "val_acc": float(va_metrics["accuracy"]),
        "held_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
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
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-tag", default="T47")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]
    print(f"=== T47 multi-horizon aux LOSO horizons={horizons}, syms={target_syms} ===",
          flush=True)

    progress("loading_caches", horizons=horizons)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time() - t0:.1f}s; "
          f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
          flush=True)

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    print(f"feat dim = {len(feat_names)}", flush=True)
    assert len(feat_names) == train_full["X"].shape[1] - N_DROP_TAIL
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features in feat_names: {leak}"
    print("  OK: no date/sym/time in features", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T47-multihorizon-aux-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "C-223d",
                    "experiment": "T47_multihorizon_aux_stacking",
                    "horizons": horizons,
                    "n_features": len(feat_names),
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T47", "multihorizon", "stacking", "aug_a"],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback personal", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    all_results = []
    n_total = len(horizons) * len(target_syms)
    done = 0
    for H in horizons:
        print(f"\n{'='*78}\n=== HORIZON h={H} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            print(f"\n  [{done}/{n_total}] fold h{H}_held{held}", flush=True)
            progress("training", horizon=H, held=held, done=done, total=n_total)
            r = train_one(H, held, train_full, val_full, test_full, feat_names, args)
            all_results.append(r)
            if use_wandb:
                wandb.log({
                    f"h{H}/fold{held}/best_iter": r["best_iter"],
                    f"h{H}/fold{held}/train_time_sec": r["train_time_sec"],
                    f"h{H}/fold{held}/val_cum_pnl": r["val_cum_pnl"],
                    f"h{H}/fold{held}/held_cum_pnl": r["held_cum_pnl"],
                    f"h{H}/fold{held}/held_acc": r["held_acc"],
                })

    # Aggregate per horizon
    df = pd.DataFrame(all_results)
    aggregate = []
    print(f"\n{'='*78}\n=== AGGREGATE ===\n{'='*78}", flush=True)
    for H in horizons:
        sub = df[df.horizon == H]
        cums = sub["held_cum_pnl"].values
        agg = {
            "horizon": H,
            "n_folds": int(len(cums)),
            "cum_pnl_per_fold": cums.tolist(),
            "cum_pnl_sum": float(cums.sum()),
            "cum_pnl_mean": float(cums.mean()) if len(cums) else 0.0,
            "n_pos_folds": int((cums > 0).sum()),
        }
        aggregate.append(agg)
        print(f"  h{H}: sum={agg['cum_pnl_sum']:+.4f} per_fold={[f'{x:+.3f}' for x in cums]}",
              flush=True)
        if use_wandb:
            wandb.summary[f"agg/h{H}/cum_pnl_sum"] = agg["cum_pnl_sum"]

    summary_path = os.path.join(HERE, f"loso_train_{args.out_tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json({
            "task": "T47 multi-horizon aux stacking LOSO",
            "horizons": horizons,
            "target_syms": target_syms,
            "params": vars(args),
            "results": all_results,
            "aggregate": aggregate,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    progress("done", n_runs=len(all_results), horizons=horizons)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
