"""T11: Multi-seed LOSO LightGBM trainer for Scheme C (223-d, multi-horizon).

For each (seed, horizon, held_out_sym) trip:
    train: (sym != held, dates 0..79)
    val:   (sym != held, dates 80..95)  -- early stopping
    test:  (sym == held, dates 96..119) -- held-out predictions

Forces diversity across seeds via different (feature_fraction, bagging_fraction,
num_leaves, lambda_l2). T6b lesson: same hyperparam multi-seed = no improvement.

Outputs (per (seed, horizon, fold)):
    loso_pred_h{H}_seed{S}_held{K}.parquet  -- prob_0/1/2 + meta
    loso_model_h{H}_seed{S}_held{K}.txt     -- model on disk

Uses the 226-d schemeC cache and slices to 223-d (drops time-encoding tail) so
the LOSO feature space matches the iter_004 submission feature space exactly.
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

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # last 3 cols are time-encoding (drop to match iter_002/iter_004)

# T11 diversity-forcing hyperparam configs per seed.
# Each varies feature_fraction / bagging_fraction / capacity to maximize variance.
SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63, lambda_l2=1.0),
    13:  dict(seed=13,  feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=2.0),
    100: dict(seed=100, feature_fraction=0.5, bagging_fraction=0.6, num_leaves=255, lambda_l2=1.0),
}


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


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


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


def train_one(H, seed_cfg, held, train_full, val_full, test_full, feat_names, args):
    seed = int(seed_cfg["seed"])
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr = train_full["X"][m_tr][:, :-N_DROP_TAIL]
    y_tr = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL]
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL]
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
        "num_leaves": int(seed_cfg["num_leaves"]),
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": float(seed_cfg["feature_fraction"]),
        "bagging_fraction": float(seed_cfg["bagging_fraction"]),
        "bagging_freq": args.bagging_freq,
        "lambda_l2": float(seed_cfg["lambda_l2"]),
        "num_threads": args.num_threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
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
            lgb.log_evaluation(period=200),
        ],
    )
    train_time = time.time() - t_start
    print(f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
          flush=True)

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(f"HELD-OUT TEST h={H} seed={seed} held={held}",
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
    pred_path = os.path.join(HERE, f"loso_pred_h{H}_seed{seed}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)
    model_path = os.path.join(HERE, f"loso_model_h{H}_seed{seed}_held{held}.txt")
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "horizon": H,
        "seed": seed,
        "held_out_sym": held,
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "held_out_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_out_acc": float(te_metrics["accuracy"]),
        "held_out_n_active": int(te_metrics["n_predictions_active"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="10",
                    help="comma-separated horizons (default 10)")
    ap.add_argument("--seeds", default="42,1,7,13,100",
                    help="comma-separated seeds matching SEED_CONFIGS")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-tag", default="T11")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"seed {s} not in SEED_CONFIGS")

    print(f"=== T11 LOSO Scheme C 223-d, horizons={horizons}, seeds={seeds}, syms={target_syms} ===",
          flush=True)

    progress("loading_caches", horizons=horizons, seeds=seeds)
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
    assert not leak, f"FORBIDDEN features: {leak}"
    print("  OK: no date/sym/time in features", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T11-schemeC-multiseed-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "C-223d",
                    "n_features": len(feat_names),
                    "horizons": horizons,
                    "seeds": seeds,
                    "diversity_strategy": "per-seed (feature_fraction, bagging_fraction, num_leaves, lambda_l2)",
                    "seed_configs": {str(s): SEED_CONFIGS[s] for s in seeds},
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T11", "multiseed-ensemble", "schemeC", "223d"],
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
    n_total = len(horizons) * len(seeds) * len(target_syms)
    done = 0
    for H in horizons:
        print(f"\n{'='*78}\n=== HORIZON h={H} ===\n{'='*78}", flush=True)
        for s in seeds:
            seed_cfg = SEED_CONFIGS[s]
            print(f"\n  --- seed {s} cfg={seed_cfg} ---", flush=True)
            for held in target_syms:
                done += 1
                tag = f"h{H}_seed{s}_held{held}"
                print(f"\n  [{done}/{n_total}] fold {tag}", flush=True)
                progress("training", horizon=H, seed=s, held=held,
                         done=done, total=n_total)
                r = train_one(H, seed_cfg, held, train_full, val_full, test_full,
                              feat_names, args)
                all_results.append(r)
                if use_wandb:
                    wandb.log({
                        f"h{H}/seed{s}/fold{held}/cum_pnl": r["held_out_cum_pnl"],
                        f"h{H}/seed{s}/fold{held}/acc": r["held_out_acc"],
                        f"h{H}/seed{s}/fold{held}/best_iter": r["best_iter"],
                        f"h{H}/seed{s}/fold{held}/train_time_sec": r["train_time_sec"],
                    })

    summary_path = os.path.join(HERE, f"loso_train_{args.out_tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json({
            "task": "T11 multi-seed LOSO Scheme C 223-d",
            "horizons": horizons, "seeds": seeds, "target_syms": target_syms,
            "seed_configs": {str(s): SEED_CONFIGS[s] for s in seeds},
            "results": all_results,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)
    progress("loso_done", n_runs=len(all_results))

    # Per-seed argmax-baseline aggregate
    if use_wandb:
        df = pd.DataFrame(all_results)
        for H in horizons:
            for s in seeds:
                sub = df[(df.horizon == H) & (df.seed == s)]
                wandb.summary[f"h{H}_seed{s}_argmax_sum_cum_pnl"] = float(sub["held_out_cum_pnl"].sum())
        wandb.finish()


if __name__ == "__main__":
    main()
