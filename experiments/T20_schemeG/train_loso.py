"""T20 Scheme G LOSO h_10 trainer.

Same hyperparams as T11 (5-seed Scheme C) — but trained on Scheme G feature set.
We train ONE seed (=42) for now — the goal is feature importance analysis,
and to compare h_10 LOSO sum_cum_pnl to iter_003's +22.22 (5-seed ensemble).

If single-seed argmax_sum > +20 (single-seed LightGBM Scheme C42 was ~+18 LOSO),
we'll see whether the new features add signal.

Outputs:
    loso_pred_h10_seed{S}_held{K}.parquet -- per fold predictions
    loso_model_h10_seed{S}_held{K}.txt    -- per fold model
    feat_importance_h10_seed42.csv        -- gain + split importance
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

CACHE = os.path.join(HERE, "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # drop time-encoding (matches iter_003 contract; G feats follow it)

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
    p = os.path.join(CACHE, f"schemeG_{split}.npz")
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


def train_one(H, seed_cfg, held, train_full, val_full, test_full, feat_names, args, fi_acc):
    seed = int(seed_cfg["seed"])
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    # cache X with 395-d, drop last 3 time-encoding cols (positions 224, 225, 226 within base 226 → after the base block).
    # Layout: [0..153] raw, [154..225] T3 (last 3 = time), [226..394] G.
    # We drop indices [223, 224, 225] = time cols (in 0-based) → BUT time is at the END of the 226 base block.
    # Actually base 226 = 154 raw + 72 T3 (and last 3 of T3 are time cols) → indices 223..225 are time.
    # G follows at indices 226..394.
    # So drop_idx = [223, 224, 225].
    n_total = train_full["X"].shape[1]
    keep_mask = np.ones(n_total, dtype=bool)
    keep_mask[223:226] = False
    X_tr = train_full["X"][m_tr][:, keep_mask]
    y_tr = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va][:, keep_mask]
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te][:, keep_mask]
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

    # accumulate feature importance
    imp_split = booster.feature_importance(importance_type="split")
    imp_gain = booster.feature_importance(importance_type="gain")
    fi_acc["split"] += imp_split
    fi_acc["gain"] += imp_gain
    fi_acc["n_models"] += 1

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
    ap.add_argument("--horizons", default="10")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-tag", default="T20")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"seed {s} not in SEED_CONFIGS")

    print(f"=== T20 LOSO Scheme G, horizons={horizons}, seeds={seeds}, syms={target_syms} ===",
          flush=True)

    progress("loading_caches", horizons=horizons, seeds=seeds)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time() - t0:.1f}s; "
          f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
          flush=True)

    feat_names_path = os.path.join(CACHE, "schemeG_223base_feat_names.txt")
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
            run_name = f"T20-schemeG-feat-explosion-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm",
                    "scheme": "G",
                    "n_features": len(feat_names),
                    "horizons": horizons,
                    "seeds": seeds,
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T20", "schemeG", "feat-explosion"],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback personal", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    fi_acc = {"split": np.zeros(len(feat_names), dtype=np.float64),
              "gain": np.zeros(len(feat_names), dtype=np.float64),
              "n_models": 0}

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
                              feat_names, args, fi_acc)
                all_results.append(r)
                if use_wandb:
                    wandb.log({
                        f"h{H}/seed{s}/fold{held}/cum_pnl": r["held_out_cum_pnl"],
                        f"h{H}/seed{s}/fold{held}/acc": r["held_out_acc"],
                        f"h{H}/seed{s}/fold{held}/best_iter": r["best_iter"],
                        f"h{H}/seed{s}/fold{held}/train_time_sec": r["train_time_sec"],
                    })

    # Save feat importance averaged across folds
    fi_df = pd.DataFrame({
        "feature": feat_names,
        "split": fi_acc["split"] / max(1, fi_acc["n_models"]),
        "gain": fi_acc["gain"] / max(1, fi_acc["n_models"]),
    })
    fi_df = fi_df.sort_values("gain", ascending=False).reset_index(drop=True)
    fi_path = os.path.join(HERE, f"feat_importance_h{horizons[0]}_seed{seeds[0]}.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"\nfeat importance -> {fi_path}", flush=True)
    print("\nTop 30 by gain:")
    print(fi_df.head(30).to_string(index=False))

    summary_path = os.path.join(HERE, f"loso_train_{args.out_tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json({
            "task": "T20 Scheme G LOSO",
            "horizons": horizons, "seeds": seeds, "target_syms": target_syms,
            "n_features": len(feat_names),
            "results": all_results,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)
    progress("loso_done", n_runs=len(all_results))

    if use_wandb:
        df = pd.DataFrame(all_results)
        for H in horizons:
            for s in seeds:
                sub = df[(df.horizon == H) & (df.seed == s)]
                wandb.summary[f"h{H}_seed{s}_argmax_sum_cum_pnl"] = float(sub["held_out_cum_pnl"].sum())
        wandb.finish()


if __name__ == "__main__":
    main()
