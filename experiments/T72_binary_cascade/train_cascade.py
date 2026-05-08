"""T72: Binary cascade replacing 3-class softmax.

Stage A: binary `active_or_flat`  -> y_a = 1 if label in {0,2} else 0
         trained on FULL train data
Stage B: conditional binary `up_or_down`  -> y_b = 1 if label == 2 else 0
         trained ONLY on active rows (label in {0,2})

Both: LightGBM objective='binary', metric='binary_logloss'.
5 seeds (42,1,7,13,100), aug_a [0.80, 1.20], GPU.

Reuses T68 schemeP cache (~370 dims), drops T59 fail (10) + Stage5 fail (1).
V4 walk-forward val: train=date 0-75, val=date 76-79.

Usage:
  python train_cascade.py --stage A --seed 42 --use-gpu
  python train_cascade.py --stage B --seed 42 --use-gpu
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
from build_aug import aug_a_scale  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_BINARY = 2
SYMS = (0, 1, 2, 3, 4)

# Same seed configs as T70 (drop num_class — reuse for binary)
SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def progress(stage, seed, step, **extra):
    p = os.path.join(HERE, f"worker-progress_stage{stage}_seed{seed}.json")
    payload = {"status": "running", "stage": stage, "seed": seed, "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_proba_bin(booster, X, batch=200_000):
    """Binary booster.predict returns 1D array of P(class=1)."""
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def class_balanced_weight_binary(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (2 * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def build_v4_split(train_full, slicer):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_tr = train_full["y60"][m_t].astype(np.int64)
    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_va = train_full["y60"][m_va].astype(np.int64)
    info = {
        "strategy": "V4",
        "train_dates": "0-75",
        "val_dates": "76-79 (walk-forward last 5%)",
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }
    return X_tr, y_tr, X_va, y_va, info


def build_aug_a(X_orig, y_orig, rng, aug_ratio=1.0):
    """Concatenate X_orig with X_aug (same y); class-balanced weights on merged."""
    n_aug = int(round(aug_ratio * len(X_orig)))
    if n_aug <= 0:
        sw = class_balanced_weight_binary(y_orig)
        return X_orig, y_orig, sw
    if n_aug == len(X_orig):
        idx = np.arange(len(X_orig))
    else:
        idx = rng.integers(0, len(X_orig), size=n_aug)
    X_aug = aug_a_scale(X_orig[idx], rng)
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_orig[idx]], axis=0).astype(np.int64)
    sw = class_balanced_weight_binary(y_merged)
    return X_merged, y_merged, sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    stage = args.stage
    print(f"=== T72 cascade Stage {stage} seed={seed} h={H} ===", flush=True)

    progress(stage, seed, "loading_caches")
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim, (len(all_feat_names), total_dim)

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_idx], dtype=np.int64,
    )
    slicer = keep_idx
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)} fail features)", flush=True)

    feat_names = [all_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak

    X_tr, y_tr, X_va, y_va, info = build_v4_split(train_full, slicer)
    print(f"  {info}", flush=True)

    X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
    y_te = test_full[f"y{H}"].astype(np.int64)

    # Build binary labels per stage
    if stage == "A":
        # Stage A: active vs flat — full data
        yb_tr = (y_tr != 1).astype(np.int64)
        yb_va = (y_va != 1).astype(np.int64)
        Xb_tr, Xb_va = X_tr, X_va
        print(f"  Stage A: full train n={len(Xb_tr):,}  pos_rate={yb_tr.mean():.4f}", flush=True)
        print(f"           full val   n={len(Xb_va):,}  pos_rate={yb_va.mean():.4f}", flush=True)
    else:
        # Stage B: up vs down — only active rows
        m_act_tr = (y_tr != 1)
        m_act_va = (y_va != 1)
        Xb_tr = X_tr[m_act_tr]
        yb_tr = (y_tr[m_act_tr] == 2).astype(np.int64)
        Xb_va = X_va[m_act_va]
        yb_va = (y_va[m_act_va] == 2).astype(np.int64)
        print(f"  Stage B: active-only train n={len(Xb_tr):,}  up_rate={yb_tr.mean():.4f}", flush=True)
        print(f"           active-only val   n={len(Xb_va):,}  up_rate={yb_va.mean():.4f}", flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T72-cascade-stage{stage}-seed{seed}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu-binary",
                    "task": f"T72_cascade_stage{stage}",
                    "split_info": info,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    "stage": stage,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T72", "cascade", f"stage{stage}", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress(stage, seed, "training", split_info=info, n_train=int(len(Xb_tr)))

    seed_rng = np.random.default_rng(seed * 7919 + (1 if stage == "A" else 2))
    Xb_tr_aug, yb_tr_aug, sw_tr = build_aug_a(Xb_tr, yb_tr, seed_rng,
                                              aug_ratio=args.aug_ratio)
    sw_va = class_balanced_weight_binary(yb_va)
    print(f"  n_train_used={len(Xb_tr_aug):,}  n_val={len(Xb_va):,}", flush=True)

    dtrain = lgb.Dataset(Xb_tr_aug, label=yb_tr_aug, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(Xb_va, label=yb_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "binary",
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
        "metric": "binary_logloss",
    }
    if args.use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t_start
    best_iter = int(booster.best_iteration)
    best_score = booster.best_score
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)
    print(f"  best_score={dict(best_score['val'])}", flush=True)

    # Save model
    model_path = os.path.join(HERE, f"model_stage{stage}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Predict on val and test (probabilities)
    va_p = predict_proba_bin(booster, Xb_va)
    te_p = predict_proba_bin(booster, X_te)  # always full test (442k)

    val_logloss = float(best_score["val"]["binary_logloss"])
    val_acc = float(((va_p > 0.5).astype(np.int64) == yb_va).mean())
    print(f"  val_acc={val_acc:.4f}  val_logloss={val_logloss:.4f}", flush=True)

    # Save test predictions parquet (for cascade DE)
    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        f"prob_stage{stage}": te_p.astype(np.float32),
        "midprice_t": test_full["mp_t"].astype(np.float32),
        "midprice_th": test_full[f"mp_t{H}"].astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_stage{stage}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    summary = {
        "task": f"T72 cascade stage={stage} seed={seed}",
        "stage": stage,
        "seed": seed,
        "horizon": H,
        "split_info": info,
        "n_features": feat_dim,
        "drop_names": list(DROP_NAMES),
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val_logloss": val_logloss,
        "val_acc": val_acc,
        "n_train_active_only": int(len(Xb_tr)) if stage == "B" else None,
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_stage{stage}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_iter": best_iter,
            "train_time_sec": float(train_time),
            "val_logloss": val_logloss,
            "val_acc": val_acc,
        })
        wandb.finish()
    progress(stage, seed, "done", best_iter=best_iter,
             val_logloss=val_logloss, val_acc=val_acc)


if __name__ == "__main__":
    main()
