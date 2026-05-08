"""T133: Alt boosters on T75 LGB L2 baseline.

Variants:
  dart:    boosting_type=dart, drop_rate=0.10, max_drop=50, 1500 rounds, NO early stopping
  tweedie: objective=tweedie, tweedie_variance_power=1.5
           (target shifted to be non-negative, then unshifted at predict time)

T75 baseline params: regression_l2, schemeP 359-d, no HYD/lag/monotone,
3 seeds (1, 7, 42), aug_a + class-balanced sample weights, V4 split (0-75 tr, 76-79 va).
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
from build_aug import aug_a_scale, class_balanced_weight  # noqa: E402

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
SCHEME_CACHE = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
H = 60

SEED_CONFIGS = {
    42: dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0),
    1:  dict(seed=1,  feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=1.0),
    7:  dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63, lambda_l2=2.0),
}

T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL + STAGE5_FAIL

# Tweedie target shift constant — ensures y_shifted >= 0
# y_regr = (mp_th - mp_t) / (mp_t + 1) — for 60-tick horizon, |y| << 0.01
# Use 0.01 as safe shift (should cover all cases)
TWEEDIE_SHIFT = 0.01


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_scheme(split):
    p = os.path.join(SCHEME_CACHE, f"schemeP_{split}.npz")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s:s+batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", choices=("dart", "tweedie"), required=True)
    ap.add_argument("--num-boost-round", type=int, default=None)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    variant = args.variant
    is_dart = variant == "dart"
    is_tweedie = variant == "tweedie"

    if args.num_boost_round is None:
        args.num_boost_round = 1500 if is_dart else 600

    tag = f"{variant}_seed{seed}"
    print(f"=== T133 {variant} seed={seed} ===", flush=True)
    progress("loading", variant=variant, seed=seed)

    t0 = time.time()
    train_full = load_scheme("train")
    test_full = load_scheme("test")
    print(f"  loaded schemeP in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(SCHEME_CACHE, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f if l.strip()]
    total_dim = train_full["X"].shape[1]
    assert len(all_names) == total_dim, (len(all_names), total_dim)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_names = [all_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"leak: {leak}"

    feat_names_full = feat_names

    # build train/val split (V4: train 0-75, val 76-79)
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, keep_idx].astype(np.float32, copy=False)
    X_va = train_full["X"][m_va][:, keep_idx].astype(np.float32, copy=False)

    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    print(f"  X_tr{X_tr.shape}  X_va{X_va.shape}  feat_dim={X_tr.shape[1]}", flush=True)
    print(f"  y_regr_tr range=[{y_regr_tr.min():.6f}, {y_regr_tr.max():.6f}]", flush=True)

    X_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_regr_te = regr_target(test_full["mp_t"], test_full["mp_t60"])
    sym_te = test_full["sym"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]

    # Tweedie target shift
    if is_tweedie:
        y_min_tr = float(y_regr_tr.min())
        y_min_va = float(y_regr_va.min())
        shift = max(TWEEDIE_SHIFT, abs(y_min_tr) * 1.5, abs(y_min_va) * 1.5)
        print(f"  tweedie shift={shift:.6f} (y_min_tr={y_min_tr:.6f}, y_min_va={y_min_va:.6f})",
              flush=True)
    else:
        shift = 0.0

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T133-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                       name=run_name,
                       config={"variant": variant, "seed": seed,
                               "n_features": X_tr.shape[1],
                               "tweedie_shift": shift,
                               "horizon": H},
                       tags=["T133", "alt_boosters", variant, f"seed{seed}"])
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", variant=variant, seed=seed)

    # aug_a per-feat scale [0.8, 1.2], ratio=1.0
    rng = np.random.default_rng(seed * 7919 + 1)
    aug_idx = np.arange(len(X_tr))
    X_aug = aug_a_scale(X_tr[aug_idx], rng, lo=0.8, hi=1.2)
    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
    del X_tr, X_aug

    # Apply tweedie shift to labels
    y_train_lgb = (y_regr_tr_full + shift).astype(np.float32) if is_tweedie else y_regr_tr_full
    y_val_lgb = (y_regr_va + shift).astype(np.float32) if is_tweedie else y_regr_va

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)

    dtrain = lgb.Dataset(X_tr_full, label=y_train_lgb, weight=sw_tr,
                         feature_name=feat_names_full, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_val_lgb, weight=sw_va,
                       feature_name=feat_names_full, reference=dtrain, free_raw_data=False)

    if is_dart:
        params = {
            "objective": "regression_l2",
            "metric": "l2",
            "boosting_type": "dart",
            "drop_rate": 0.10,
            "max_drop": 50,
            "skip_drop": 0.5,
            "uniform_drop": False,
            "learning_rate": args.learning_rate,
            "num_leaves": int(cfg["num_leaves"]),
            "min_data_in_leaf": 100,
            "feature_fraction": float(cfg["feature_fraction"]),
            "bagging_fraction": float(cfg["bagging_fraction"]),
            "bagging_freq": 5,
            "lambda_l2": float(cfg["lambda_l2"]),
            "num_threads": args.num_threads,
            "seed": seed,
            "feature_fraction_seed": seed + 1,
            "bagging_seed": seed + 2,
            "data_random_seed": seed + 3,
            "device": "gpu", "gpu_use_dp": False,
            "verbose": -1,
        }
    else:  # tweedie
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "metric": "tweedie",
            "learning_rate": args.learning_rate,
            "num_leaves": int(cfg["num_leaves"]),
            "min_data_in_leaf": 100,
            "feature_fraction": float(cfg["feature_fraction"]),
            "bagging_fraction": float(cfg["bagging_fraction"]),
            "bagging_freq": 5,
            "lambda_l2": float(cfg["lambda_l2"]),
            "num_threads": args.num_threads,
            "seed": seed,
            "feature_fraction_seed": seed + 1,
            "bagging_seed": seed + 2,
            "data_random_seed": seed + 3,
            "device": "gpu", "gpu_use_dp": False,
            "verbose": -1,
        }

    callbacks = [lgb.log_evaluation(period=100)]
    if not is_dart:
        callbacks.insert(0, lgb.early_stopping(args.early_stopping, verbose=True))

    t1 = time.time()
    booster = lgb.train(
        params=params, train_set=dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=callbacks,
    )
    train_time = time.time() - t1
    if is_dart:
        best_iter = args.num_boost_round
    else:
        best_iter = int(booster.best_iteration) if booster.best_iteration else args.num_boost_round
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    model_path = os.path.join(HERE, f"model_T133_{tag}.txt")
    if is_dart:
        booster.save_model(model_path)
    else:
        booster.save_model(model_path, num_iteration=best_iter)

    va_pred_raw = predict_chunked(booster, X_va)
    va_pred = (va_pred_raw - shift).astype(np.float32) if is_tweedie else va_pred_raw
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)

    te_pred_raw = predict_chunked(booster, X_te)
    te_pred = (te_pred_raw - shift).astype(np.float32) if is_tweedie else te_pred_raw
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  TEST mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    sess_map = {0: "am", 1: "pm"}
    df = pd.DataFrame({
        "sym": sym_te.astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]],
                            dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T133_{tag}.parquet")
    df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(df):,} rows)", flush=True)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    per_sym = []
    for s in SYMS:
        m = (sym_te == s)
        per_sym.append(float(pnl[m].sum()))
    print(f"  EV-gate k=1 cum_pnl={cum_pnl:+.4f} n_active={n_active:,} "
          f"per_sym={[round(x,2) for x in per_sym]}", flush=True)

    summary = {
        "task": f"T133 LGB {variant} seed={seed}",
        "variant": variant, "seed": seed, "horizon": H, "tag": tag,
        "tweedie_shift": shift,
        "n_features": X_te.shape[1],
        "best_iter": best_iter, "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl,
                 "ev_gate_k1_n_active": n_active,
                 "per_sym_ev_gate": per_sym},
    }
    out_path = os.path.join(HERE, f"summary_T133_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    if use_wandb and wandb is not None:
        wandb.log({"best_iter": best_iter, "train_time_sec": train_time,
                   "val_mse": va_mse, "val_mae": va_mae, "val_corr": va_corr,
                   "test_mae": te_mae, "test_corr": te_corr,
                   "test_ev_gate_k1_cum_pnl": cum_pnl,
                   "test_ev_gate_k1_n_active": n_active})
        wandb.finish()
    progress("done", variant=variant, seed=seed,
             best_iter=best_iter, test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
