"""T71: V4 walk-forward val + Stage 5 features (schemeP cache) — h_40 horizon.

Mirrors T70 train_v4_stage5.py exactly EXCEPT trains on horizon=40 instead of 60.

Subversive idea: platform score = max over (h_5/10/20/40/60). iter_006+ only trained
h_60. iter_002 had a weak h_40 (226-d, +2.02). If h_40 with the new 359-d Stage 5
features beats iter_012 h_60 (+26.44) — then h_40 IS the new winner; if it's between
+20 and +26.44 we keep both active so the platform's max picks whichever is stronger
in any given fold profile.

Drop set:
  - T59 sym-invariance FAIL names (10): present in baseline 196 extras
  - Stage 5 FAIL: liq_asym_top5_W5 (1)
Total kept ~ 359 dims.

V4 walk-forward val: train=date 0-75 (95% of 0-79), val=date 76-79 (5%).
Test = date 96-119 full 5-sym, label = y40.

5-seed full-data, GPU LightGBM, num_boost=600, early_stop=40, aug_a [0.80, 1.20].
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

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
CACHE_DIR = os.path.join(T68_DIR, "cache")

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

T59_FAIL_NAMES = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
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


def build_v4_split(train_full, slicer, H):
    """V4 walk-forward: train=date 0-75 (95%), val=date 76-79 (5%)."""
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_tr = train_full[f"y{H}"][m_t].astype(np.int64)
    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_va = train_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full[f"mp_t{H}"][m_va]
    info = {
        "strategy": "V4",
        "horizon": H,
        "train_dates": "0-75",
        "val_dates": "76-79 (walk-forward last 5%)",
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }
    return X_tr, y_tr, X_va, y_va, mp_t_va, mp_th_va, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=40)
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
    ap.add_argument("--out-tag", default="t71_h40")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    print(f"=== T71 V4 + Stage 5 seed={seed} h={H} ===", flush=True)

    progress("loading_caches", seed=seed, horizon=H)
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
        else:
            print(f"  WARN drop name not found: {n}", flush=True)
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

    X_tr, y_tr, X_va, y_va, mp_t_va, mp_th_va, info = build_v4_split(train_full, slicer, H)
    print(f"  {info}", flush=True)

    X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
    y_te = test_full[f"y{H}"].astype(np.int64)
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T71-V4-stage5-h{H}-seed{seed}-{args.out_tag}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "task": "T71_h40_v4_stage5",
                    "split_info": info,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T71", "V4", "stage5", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, horizon=H, split_info=info)

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
        "metric": "multi_logloss",
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
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    model_path = os.path.join(HERE, f"model_T71_h{H}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    va_prob = predict_proba(booster, X_va)
    va_pred = va_prob.argmax(axis=1).astype(np.int8)
    va_metrics = evaluate(f"VAL V4 h{H} seed={seed}", va_pred, y_va, mp_t_va, mp_th_va)

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(f"TEST full 5-sym h{H} seed={seed}",
                          te_pred, y_te, mp_t_te, mp_th_te)

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
    pred_path = os.path.join(HERE, f"pred_T71_h{H}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    summary = {
        "task": f"T71 V4 stage5 h{H} seed={seed}",
        "seed": seed,
        "horizon": H,
        "split_info": info,
        "n_features": feat_dim,
        "drop_names": list(DROP_NAMES),
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": va_metrics,
        "test_full": te_metrics,
        "per_sym_test": per_sym,
        "loso_equiv_sum_argmax": float(loso_equiv),
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T71_h{H}_seed{seed}.json")
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
    progress("done", seed=seed, horizon=H,
             best_iter=best_iter, test_cum_pnl=te_metrics["cum_pnl"],
             loso_equiv=float(loso_equiv))


if __name__ == "__main__":
    main()
