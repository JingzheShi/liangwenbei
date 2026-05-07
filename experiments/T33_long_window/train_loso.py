"""T33 Scheme J LOSO LightGBM training, h=60.

Modes:
  baseline single seed=42 (default)
  --variants aug_a   (single or multiple seeds)
  --seeds 42,1,7,13,100  (multi-seed for ensemble)

For each (variant, seed, held_out_sym):
  train: (sym != held, dates 0..79) + variant aug
  val:   (sym != held, dates 80..95)  no aug, used for early stop
  test:  (sym == held, dates 96..119) no aug

Outputs (per (variant, seed, held)):
  loso_pred_h60_{variant}_seed{S}_held{K}.parquet
  loso_model_h60_{variant}_seed{S}_held{K}.txt
And summary:
  loso_train_T33_summary.json
  feat_importance_T33.json

GPU LightGBM by default (matches T26).
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

sys.path.insert(0, os.path.join(ROOT, "experiments", "T26_domain_randomization"))
from build_aug import (  # noqa: E402
    build_train_for_variant,
    class_balanced_weight,
    compute_feat_stats,
    VARIANTS,
)

T33_CACHE = os.path.join(ROOT, "experiments", "T33_long_window", "cache")
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


def load_split(split: str) -> dict:
    p = os.path.join(T33_CACHE, f"schemeJ_{split}.npz")
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


def train_one(H, variant, seed, held, train_full, val_full, test_full, feat_names, args):
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr_o = train_full["X"][m_tr]
    y_tr_o = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va]
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te]
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[f"mp_t{H}"][m_te]

    feat_mean, feat_std = compute_feat_stats(X_tr_o)
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
        f"HELD-OUT TEST h={H} variant={variant} seed={seed} held={held}",
        te_pred, y_te, mp_t_te, mp_th_te,
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
    pred_path = os.path.join(HERE, f"loso_pred_h{H}_{variant}_seed{seed}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(HERE, f"loso_model_h{H}_{variant}_seed{seed}_held{held}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    fi_gain = booster.feature_importance(importance_type="gain")
    fi_split = booster.feature_importance(importance_type="split")

    return {
        "horizon": H,
        "variant": variant,
        "seed": seed,
        "held_out_sym": held,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
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
    ap.add_argument("--variants", default="baseline")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="42")
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
    ap.add_argument("--use-gpu", action="store_true",
                    help="enable LightGBM GPU device")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-tag", default="T33")
    args = ap.parse_args()

    H = args.horizon
    target_syms = [int(x) for x in args.syms.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"unknown variant '{v}', valid={VARIANTS}")

    print(
        f"=== T33 Scheme J LOSO h={H}, variants={variants}, seeds={seeds}, "
        f"syms={target_syms} ===",
        flush=True,
    )

    progress("loading_caches", horizon=H, variants=variants, seeds=seeds)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )

    feat_names_path = os.path.join(T33_CACHE, "schemeJ_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    print(f"feat dim = {len(feat_names)}", flush=True)
    assert len(feat_names) == train_full["X"].shape[1], \
        f"mismatch {len(feat_names)} vs {train_full['X'].shape[1]}"
    forbidden = {"date", "sym", "time", "time_minutes_since_session_start",
                 "time_session_progress", "time_is_pm"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features in feat_names: {leak}"
    print("  OK: no date/sym/time in features", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T33-schemeJ-h{H}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "J-583d",
                    "experiment": "long_window",
                    "variants": variants,
                    "seeds": seeds,
                    "horizon": H,
                    "n_features": len(feat_names),
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T33", "long-window", "schemeJ", f"h{H}"],
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
    n_total = len(variants) * len(seeds) * len(target_syms)
    done = 0
    for v in variants:
        for s in seeds:
            print(f"\n{'='*78}\n=== VARIANT {v} SEED {s} ===\n{'='*78}", flush=True)
            for held in target_syms:
                done += 1
                tag = f"h{H}_{v}_seed{s}_held{held}"
                print(f"\n  [{done}/{n_total}] fold {tag}", flush=True)
                progress("training", horizon=H, variant=v, seed=s, held=held,
                         done=done, total=n_total)
                r = train_one(H, v, s, held, train_full, val_full, test_full,
                              feat_names, args)
                all_results.append(r)
                if use_wandb:
                    wandb.log({
                        f"{v}/seed{s}/fold{held}/best_iter": r["best_iter"],
                        f"{v}/seed{s}/fold{held}/train_time_sec": r["train_time_sec"],
                        f"{v}/seed{s}/fold{held}/held_cum_pnl": r["held_cum_pnl"],
                        f"{v}/seed{s}/fold{held}/held_acc": r["held_acc"],
                    })

    # Aggregate per (variant, seed)
    df = pd.DataFrame(all_results)
    aggregate = []
    print(f"\n{'='*78}\n=== AGGREGATE h={H} ===\n{'='*78}", flush=True)
    for v in variants:
        for s in seeds:
            sub = df[(df.variant == v) & (df.seed == s)]
            cums = sub["held_cum_pnl"].values
            agg = {
                "variant": v,
                "seed": s,
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
                f"  {v:10s} seed={s:3d}: sum={agg['cum_pnl_sum']:+.4f} "
                f"mean={agg['cum_pnl_mean']:+.4f} std={agg['cum_pnl_std']:.4f} "
                f"pos={agg['n_pos_folds']}/{agg['n_folds']} "
                f"per_fold={[f'{x:+.3f}' for x in cums]}",
                flush=True,
            )
            if use_wandb:
                wandb.summary[f"agg/{v}/seed{s}/cum_pnl_sum"] = agg["cum_pnl_sum"]
                wandb.summary[f"agg/{v}/seed{s}/cum_pnl_mean"] = agg["cum_pnl_mean"]
                wandb.summary[f"agg/{v}/seed{s}/cum_pnl_std"] = agg["cum_pnl_std"]

    summary_path = os.path.join(HERE, f"loso_train_{args.out_tag}_summary.json")
    with open(summary_path, "w") as f:
        slim = []
        for r in all_results:
            r_slim = {k: v for k, v in r.items() if k not in ("fi_gain", "fi_split")}
            slim.append(r_slim)
        json.dump(sanitize_for_json({
            "task": f"T33 Scheme J LOSO h={H}",
            "horizon": H, "variants": variants, "seeds": seeds,
            "target_syms": target_syms,
            "params": vars(args),
            "results": slim,
            "aggregate": aggregate,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    fi_path = os.path.join(HERE, f"feat_importance_{args.out_tag}.json")
    fi_dump = {}
    for r in all_results:
        key = f"{r['variant']}_seed{r['seed']}_held{r['held_out_sym']}"
        fi_dump[key] = {
            "fi_gain": r["fi_gain"],
            "fi_split": r["fi_split"],
            "feat_names": feat_names,
        }
    with open(fi_path, "w") as f:
        json.dump(fi_dump, f)
    print(f"feature importances -> {fi_path}", flush=True)

    progress("done", n_runs=len(all_results), variants=variants, seeds=seeds)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
