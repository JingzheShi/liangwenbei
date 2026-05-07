"""T29: aug_a on h_40 + scale-range ablation on h_60.

Runs 4 experiments back-to-back using one data load:
  - h_40 aug_a [0.8, 1.2]  (5-fold LOSO, seed=42)
  - h_60 aug_a [0.7, 1.3]  (5-fold LOSO, seed=42)
  - h_60 aug_a [0.6, 1.4]  (5-fold LOSO, seed=42)
  - h_60 aug_a [0.9, 1.1]  (5-fold LOSO, seed=42)

[0.8, 1.2] on h_60 is *not* re-run — its OOF predictions are reused from
T26 (see threshold_sweep_all.py).

Constraint compliance: matches T26 exactly (sym never feature, date dropped via
N_DROP_TAIL=3, augment train-only).
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
T26 = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26)

import lightgbm as lgb  # noqa: E402
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402
from build_aug import (  # noqa: E402  (T26's build_aug)
    aug_a_scale,
    class_balanced_weight,
    compute_feat_stats,
)

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3


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


def predict_proba(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def evaluate(name, preds, y, mp_t, mp_th, fee_rate=0.0001):
    m = _per_horizon_metrics(preds, y, mp_t, mp_th, fee_rate=fee_rate)
    print(f"  --- {name} ---", flush=True)
    for k in ("accuracy", "cum_pnl", "single_pnl", "n_predictions_active", "f0_5_macro"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"    {k:25s}: {v:+.6g}", flush=True)
        else:
            print(f"    {k:25s}: {v}", flush=True)
    return m


def build_train_aug_a(X_orig, y_orig, rng, scale_lo, scale_hi, num_class=NUM_CLASS):
    """Same shape as T26's build_train_for_variant but with explicit lo/hi."""
    X_aug = aug_a_scale(X_orig, rng, lo=scale_lo, hi=scale_hi)
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_orig], axis=0).astype(np.int64)
    w_merged = class_balanced_weight(y_merged, num_class)
    return X_merged, y_merged, w_merged


def train_one(H, scale_lo, scale_hi, held, train_full, val_full, test_full,
              feat_names, args, exp_tag):
    seed = args.seed
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

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

    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
    X_tr, y_tr, sw_tr = build_train_aug_a(
        X_tr_o, y_tr_o, fold_rng, scale_lo, scale_hi
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
        f"HELD-OUT TEST h={H} {exp_tag} held={held}",
        te_pred, y_te, mp_t_te, mp_th_te,
    )
    va_prob = predict_proba(booster, X_va)
    va_pred = va_prob.argmax(axis=1).astype(np.int8)
    va_metrics = evaluate(
        f"VAL h={H} {exp_tag} held={held}",
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
    pred_path = os.path.join(HERE, f"loso_pred_h{H}_{exp_tag}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(HERE, f"loso_model_h{H}_{exp_tag}_held{held}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "horizon": H,
        "exp_tag": exp_tag,
        "scale_lo": scale_lo,
        "scale_hi": scale_hi,
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
        "held_single_pnl": float(te_metrics["single_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
        "held_f0_5_macro": float(te_metrics["f0_5_macro"]),
    }


# Experiment plan: (horizon, scale_lo, scale_hi, exp_tag)
EXPERIMENTS = [
    (40, 0.8, 1.2, "aug_a_s08_12"),   # h_40 baseline aug_a
    (60, 0.7, 1.3, "aug_a_s07_13"),   # h_60 stronger aug
    (60, 0.6, 1.4, "aug_a_s06_14"),   # h_60 strongest aug
    (60, 0.9, 1.1, "aug_a_s09_11"),   # h_60 weak aug
]


def main():
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--experiments", default="all",
                    help="comma list of indices into EXPERIMENTS, or 'all'")
    args = ap.parse_args()

    if args.experiments == "all":
        chosen = list(range(len(EXPERIMENTS)))
    else:
        chosen = [int(x) for x in args.experiments.split(",")]

    print(
        f"=== T29 aug_a h_40 + scale ablation, experiments={chosen} ===",
        flush=True,
    )

    progress("loading_caches", chosen=chosen)
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
            run_name = f"T29-aug_a-ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "C-223d",
                    "experiment": "T29_aug_a_h40_ablation",
                    "experiments_planned": [
                        {"horizon": h, "scale_lo": lo, "scale_hi": hi, "tag": tag}
                        for (h, lo, hi, tag) in EXPERIMENTS
                    ],
                    "n_features": len(feat_names),
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T29", "aug_a", "ablation", "schemeC"],
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
    n_total = len(chosen) * 5
    done = 0
    per_exp_summary = []
    for ei in chosen:
        H, lo, hi, tag = EXPERIMENTS[ei]
        print(f"\n{'='*78}\n=== EXPERIMENT [{ei}] h={H} {tag} scale=[{lo}, {hi}] ===\n{'='*78}",
              flush=True)
        exp_results = []
        for held in range(5):
            done += 1
            print(f"\n  [{done}/{n_total}] fold held={held}", flush=True)
            progress("training", horizon=H, exp_tag=tag, held=held,
                     done=done, total=n_total)
            r = train_one(H, lo, hi, held, train_full, val_full, test_full,
                          feat_names, args, tag)
            all_results.append(r)
            exp_results.append(r)
            if use_wandb:
                wandb.log({
                    f"{tag}/fold{held}/best_iter": r["best_iter"],
                    f"{tag}/fold{held}/train_time_sec": r["train_time_sec"],
                    f"{tag}/fold{held}/val_cum_pnl": r["val_cum_pnl"],
                    f"{tag}/fold{held}/held_cum_pnl": r["held_cum_pnl"],
                    f"{tag}/fold{held}/held_single_pnl": r["held_single_pnl"],
                    f"{tag}/fold{held}/held_acc": r["held_acc"],
                    f"{tag}/fold{held}/held_f0_5_macro": r["held_f0_5_macro"],
                })

        cums = np.array([r["held_cum_pnl"] for r in exp_results])
        agg = {
            "exp_tag": tag, "horizon": H, "scale_lo": lo, "scale_hi": hi,
            "n_folds": int(len(cums)),
            "cum_pnl_per_fold": cums.tolist(),
            "cum_pnl_sum": float(cums.sum()),
            "cum_pnl_mean": float(cums.mean()),
            "cum_pnl_std": float(cums.std(ddof=0)),
            "n_pos_folds": int((cums > 0).sum()),
        }
        per_exp_summary.append(agg)
        print(
            f"  AGG {tag}: sum={agg['cum_pnl_sum']:+.4f} "
            f"mean={agg['cum_pnl_mean']:+.4f} std={agg['cum_pnl_std']:.4f} "
            f"pos={agg['n_pos_folds']}/{agg['n_folds']} "
            f"per_fold={[f'{x:+.3f}' for x in cums]}",
            flush=True,
        )
        if use_wandb:
            wandb.summary[f"agg/{tag}/cum_pnl_sum"] = agg["cum_pnl_sum"]
            wandb.summary[f"agg/{tag}/cum_pnl_mean"] = agg["cum_pnl_mean"]
            wandb.summary[f"agg/{tag}/cum_pnl_std"] = agg["cum_pnl_std"]
            wandb.summary[f"agg/{tag}/n_pos_folds"] = agg["n_pos_folds"]

    summary_path = os.path.join(HERE, "loso_train_T29_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json({
            "task": "T29 aug_a h_40 + scale ablation",
            "experiments": [
                {"idx": ei, "horizon": EXPERIMENTS[ei][0],
                 "scale_lo": EXPERIMENTS[ei][1], "scale_hi": EXPERIMENTS[ei][2],
                 "tag": EXPERIMENTS[ei][3]} for ei in chosen
            ],
            "params": vars(args),
            "results": all_results,
            "aggregate": per_exp_summary,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    progress("done", n_runs=len(all_results), experiments=chosen)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
