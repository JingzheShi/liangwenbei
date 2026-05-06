"""T26: LOSO LightGBM training with domain-randomization augments at h=60.

For each (variant, held_out_sym):
    train: (sym != held, dates 0..79)   + variant-specific augment
    val:   (sym != held, dates 80..95)  -- early stopping (NO augment)
    test:  (sym == held, dates 96..119) -- held-out sym (NO augment)

Variants:
    baseline | aug_a (scale) | aug_b (noise) | aug_c (dropout) | aug_abc

Constraint compliance:
    * Scheme C 223-d (drops trailing 3 time-encoding cols, matches iter_002)
    * sym only used to slice folds (NEVER as feature)
    * augment only applied to train data, never to val/test

Outputs (per (variant, held)):
    loso_pred_h60_{variant}_held{K}.parquet
    loso_model_h60_{variant}_held{K}.txt
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

from build_aug import (  # noqa: E402
    build_train_for_variant,
    class_balanced_weight,
    compute_feat_stats,
    VARIANTS,
)

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # match iter_002 / T11 (drop time-encoding tail to keep 223-d)


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
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


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


def train_one(H, variant, held, train_full, val_full, test_full, feat_names, args):
    seed = args.seed
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    # 226 -> 223 (drop trailing time-encoding)
    X_tr_o = train_full["X"][m_tr][:, :-N_DROP_TAIL]
    y_tr_o = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL]
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL]
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[f"mp_t{H}"][m_te]

    # Per-feature stats from THIS fold's train (not global): keeps stats LOSO-clean
    feat_mean, feat_std = compute_feat_stats(X_tr_o)

    # Fold-specific RNG so different folds see different aug realizations
    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)

    X_tr, y_tr, sw_tr = build_train_for_variant(
        variant, X_tr_o, y_tr_o, feat_std, feat_mean,
        fold_rng, aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)

    print(
        f"    n_train_orig={len(X_tr_o):,} n_train_used={len(X_tr):,} "
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
    print(
        f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
        flush=True,
    )

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(
        f"HELD-OUT TEST h={H} variant={variant} held={held}",
        te_pred, y_te, mp_t_te, mp_th_te,
    )
    va_prob = predict_proba(booster, X_va)
    va_pred = va_prob.argmax(axis=1).astype(np.int8)
    va_metrics = evaluate(
        f"VAL h={H} variant={variant} held={held}",
        va_pred, y_va, mp_t_va, mp_th_va,
    )

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"][m_te].astype(np.int8),
        "date": test_full["date"][m_te].astype(np.int16),
        "session": np.array(
            [sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object
        ),
        "t": test_full["t"][m_te].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        "pred_label": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"loso_pred_h{H}_{variant}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(HERE, f"loso_model_h{H}_{variant}_held{held}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    # Feature importance (for later analysis)
    fi_gain = booster.feature_importance(importance_type="gain")
    fi_split = booster.feature_importance(importance_type="split")

    return {
        "horizon": H,
        "variant": variant,
        "held_out_sym": held,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "val_cum_pnl": float(va_metrics["cum_pnl"]),
        "val_acc": float(va_metrics["accuracy"]),
        "val_n_active": int(va_metrics["n_predictions_active"]),
        "held_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_single_pnl": float(te_metrics["single_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
        "held_f0_5_macro": float(te_metrics["f0_5_macro"]),
        "fi_gain": fi_gain.tolist(),
        "fi_split": fi_split.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--variants", default="baseline,aug_a,aug_b,aug_c,aug_abc")
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
    ap.add_argument("--use-gpu", action="store_true",
                    help="enable LightGBM GPU device (recommended)")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-tag", default="T26")
    args = ap.parse_args()

    H = args.horizon
    target_syms = [int(x) for x in args.syms.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"unknown variant '{v}', valid={VARIANTS}")

    print(
        f"=== T26 domain-randomization LOSO h={H}, variants={variants}, syms={target_syms} ===",
        flush=True,
    )

    progress("loading_caches", horizon=H, variants=variants)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )

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
            run_name = f"T26-DR-h{H}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "C-223d",
                    "experiment": "domain_randomization",
                    "variants": variants,
                    "horizon": H,
                    "n_features": len(feat_names),
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T26", "domain-randomization", "schemeC", f"h{H}"],
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
    n_total = len(variants) * len(target_syms)
    done = 0
    for v in variants:
        print(f"\n{'='*78}\n=== VARIANT {v} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            tag = f"h{H}_{v}_held{held}"
            print(f"\n  [{done}/{n_total}] fold {tag}", flush=True)
            progress("training", horizon=H, variant=v, held=held,
                     done=done, total=n_total)
            r = train_one(H, v, held, train_full, val_full, test_full,
                          feat_names, args)
            all_results.append(r)
            if use_wandb:
                wandb.log({
                    f"{v}/fold{held}/best_iter": r["best_iter"],
                    f"{v}/fold{held}/train_time_sec": r["train_time_sec"],
                    f"{v}/fold{held}/val_cum_pnl": r["val_cum_pnl"],
                    f"{v}/fold{held}/val_acc": r["val_acc"],
                    f"{v}/fold{held}/held_cum_pnl": r["held_cum_pnl"],
                    f"{v}/fold{held}/held_single_pnl": r["held_single_pnl"],
                    f"{v}/fold{held}/held_acc": r["held_acc"],
                    f"{v}/fold{held}/held_f0_5_macro": r["held_f0_5_macro"],
                })

    # Aggregate per variant
    df = pd.DataFrame(all_results)
    aggregate = []
    print(f"\n{'='*78}\n=== AGGREGATE h={H} ===\n{'='*78}", flush=True)
    for v in variants:
        sub = df[df.variant == v]
        cums = sub["held_cum_pnl"].values
        agg = {
            "variant": v,
            "horizon": H,
            "n_folds": int(len(cums)),
            "cum_pnl_per_fold": cums.tolist(),
            "cum_pnl_sum": float(cums.sum()),
            "cum_pnl_mean": float(cums.mean()) if len(cums) else 0.0,
            "cum_pnl_std": float(cums.std(ddof=0)) if len(cums) else 0.0,
            "n_pos_folds": int((cums > 0).sum()),
        }
        aggregate.append(agg)
        print(
            f"  {v:10s}: sum={agg['cum_pnl_sum']:+.4f} "
            f"mean={agg['cum_pnl_mean']:+.4f} std={agg['cum_pnl_std']:.4f} "
            f"pos={agg['n_pos_folds']}/{agg['n_folds']} "
            f"per_fold={[f'{x:+.3f}' for x in cums]}",
            flush=True,
        )
        if use_wandb:
            wandb.summary[f"agg/{v}/cum_pnl_sum"] = agg["cum_pnl_sum"]
            wandb.summary[f"agg/{v}/cum_pnl_mean"] = agg["cum_pnl_mean"]
            wandb.summary[f"agg/{v}/cum_pnl_std"] = agg["cum_pnl_std"]
            wandb.summary[f"agg/{v}/n_pos_folds"] = agg["n_pos_folds"]

    summary_path = os.path.join(HERE, f"loso_train_{args.out_tag}_summary.json")
    with open(summary_path, "w") as f:
        # Strip the bulky fi_gain/fi_split arrays from results dump (keep separate)
        slim = []
        for r in all_results:
            r_slim = {k: v for k, v in r.items() if k not in ("fi_gain", "fi_split")}
            slim.append(r_slim)
        json.dump(sanitize_for_json({
            "task": "T26 domain randomization LOSO h=60",
            "horizon": H, "variants": variants,
            "target_syms": target_syms,
            "params": vars(args),
            "results": slim,
            "aggregate": aggregate,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    # Save feature importances separately
    fi_path = os.path.join(HERE, f"feat_importance_{args.out_tag}.json")
    fi_dump = {}
    for r in all_results:
        key = f"{r['variant']}_held{r['held_out_sym']}"
        fi_dump[key] = {
            "fi_gain": r["fi_gain"],
            "fi_split": r["fi_split"],
            "feat_names": feat_names,
        }
    with open(fi_path, "w") as f:
        json.dump(fi_dump, f)
    print(f"feature importances -> {fi_path}", flush=True)

    progress("done", n_runs=len(all_results), variants=variants)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
