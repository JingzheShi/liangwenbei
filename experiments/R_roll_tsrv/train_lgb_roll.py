"""R_roll_tsrv: Roll filter / TSRV inspired variants on T75 LGB L2 baseline.

Variants:
  V0 (baseline): standard schemeP 359-d + standard target Δmid_norm
                 (NOT retrained here; reuse existing /root/lwb_remote_pkg/preds/pred_T75_seed{S}.parquet)
  V1 (trickA-clean-target): standard schemeP 359-d + OLS-slope target through
       (mp_t, mp_t5, mp_t10, mp_t20, mp_t40, mp_t60) projected to h=60.
       Less noisy target → model uses capacity for signal not bid-ask bounce.
  V2 (trickB-meta-features): standard schemeP 359-d + 4 extra noise meta-features:
       - ep_var_proxy   = (0.5 * roll_eff_spr_ratio_W50 * (ask1-bid1)/(mp+1))^2
       - signal_var_proxy = max(rv_w50 - 5*ep_var_proxy, 1e-20)
       - noise_share    = ep_var / (ep_var + signal_var)
       - log_snr        = log(signal_var / ep_var)

Usage:
  python3 train_lgb_roll.py --variant V1 --seed 1
  python3 train_lgb_roll.py --variant V2 --seed 42
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb

CACHE_DIR = "/root/lwb_remote_pkg/cache"
WORK_DIR = "/root/lwb_work_roll_tsrv"
os.makedirs(WORK_DIR, exist_ok=True)

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
H = 60

# Same drop list as T75 baseline (line up with existing baseline for fair compare)
T59_FAIL = [
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL + STAGE5_FAIL

# Same per-seed configs as T75 baseline
SEED_CONFIGS = {
    42: dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:  dict(seed=1,  feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:  dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
}

# OLS slope weights for [t=0,5,10,20,40,60]; tbar = 22.5
OLS_T_DEV = np.array([-22.5, -17.5, -12.5, -2.5, 17.5, 37.5], dtype=np.float64)
OLS_T_VAR = float((OLS_T_DEV ** 2).sum())  # 2687.5


def progress(seed, variant, step, **extra):
    p = os.path.join(WORK_DIR, f"progress_{variant}_seed{seed}.json")
    with open(p, "w") as f:
        json.dump({"status": "running", "step": step, "variant": variant, "seed": seed,
                   "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   **extra}, f, indent=2)


def load_split(split):
    return {k: v for k, v in np.load(os.path.join(CACHE_DIR, f"schemeP_{split}.npz")).items()}


def baseline_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def clean_target(mp_t, mp_t5, mp_t10, mp_t20, mp_t40, mp_t60):
    """OLS slope * 60 / (mp_t + 1) — averages noise across forward 6 mid points."""
    y = np.stack([
        mp_t.astype(np.float64), mp_t5.astype(np.float64),
        mp_t10.astype(np.float64), mp_t20.astype(np.float64),
        mp_t40.astype(np.float64), mp_t60.astype(np.float64),
    ], axis=1)  # (n, 6)
    slope = (y * OLS_T_DEV).sum(axis=1) / OLS_T_VAR  # (n,)
    return (slope * 60.0 / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def add_meta_features(X, all_feat_names):
    """Append 4 noise meta-features computed from existing cols."""
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    bid1_i = name_to_idx["bid1"]
    ask1_i = name_to_idx["ask1"]
    roll_w50_i = name_to_idx["roll_eff_spr_ratio_W50"]
    rv_w50_i = name_to_idx["rv_w50"]

    bid1 = X[:, bid1_i].astype(np.float64)
    ask1 = X[:, ask1_i].astype(np.float64)
    roll = X[:, roll_w50_i].astype(np.float64)
    rv_w50 = X[:, rv_w50_i].astype(np.float64)

    mp = (bid1 + ask1) * 0.5
    quoted_rel = (ask1 - bid1) / (mp + 1.0)  # relative quoted spread
    eff_spr_rel = roll * quoted_rel
    ep_var = (0.5 * eff_spr_rel) ** 2  # noise variance (relative scale)
    sig_var = np.maximum(rv_w50 - 5.0 * ep_var, 1e-20)
    total = ep_var + sig_var + 1e-20
    noise_share = ep_var / total
    log_snr = np.log(sig_var / (ep_var + 1e-20))

    extra = np.stack([ep_var, sig_var, noise_share, log_snr], axis=1).astype(np.float32)
    return np.concatenate([X, extra], axis=1)


def aug_a_scale(X, rng, lo=0.80, hi=1.20):
    s = rng.uniform(lo, hi, size=X.shape).astype(np.float32)
    return X * s


def class_balanced_weight(y_cls, num_class=3):
    cnt = np.bincount(y_cls, minlength=num_class).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    inv = 1.0 / cnt
    inv = inv * (num_class / inv.sum())
    return inv[y_cls].astype(np.float32)


def predict_chunked(booster, X, batch=200_000):
    out = []
    for s in range(0, len(X), batch):
        out.append(booster.predict(X[s:s + batch]).astype(np.float32))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V0", "V1", "V2"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    variant = args.variant
    print(f"=== R_roll_tsrv {variant} seed={seed} h={H} ===", flush=True)
    progress(seed, variant, "loading_caches")

    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    with open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt")) as f:
        all_feat_names = [l.strip() for l in f if l.strip()]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim

    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names))

    # Slice features
    Xtr_raw = train_full["X"][:, keep_idx].astype(np.float32, copy=False)
    Xte_raw = test_full["X"][:, keep_idx].astype(np.float32, copy=False)

    # V2: append meta-features
    if variant == "V2":
        Xtr_proc = add_meta_features(Xtr_raw, feat_names)
        Xte_proc = add_meta_features(Xte_raw, feat_names)
        feat_names_used = feat_names + ["ep_var_proxy_W50", "signal_var_proxy_W50",
                                        "noise_share_W50", "log_snr_W50"]
    else:
        Xtr_proc = Xtr_raw
        Xte_proc = Xte_raw
        feat_names_used = feat_names

    feat_dim = Xtr_proc.shape[1]
    print(f"  feat_dim={feat_dim} (variant={variant})", flush=True)

    # Targets
    if variant == "V1":
        y_regr_tr_all = clean_target(
            train_full["mp_t"], train_full["mp_t5"], train_full["mp_t10"],
            train_full["mp_t20"], train_full["mp_t40"], train_full["mp_t60"])
        y_regr_te = clean_target(
            test_full["mp_t"], test_full["mp_t5"], test_full["mp_t10"],
            test_full["mp_t20"], test_full["mp_t40"], test_full["mp_t60"])
    else:
        y_regr_tr_all = baseline_target(train_full["mp_t"], train_full["mp_t60"])
        y_regr_te = baseline_target(test_full["mp_t"], test_full["mp_t60"])

    y_cls_tr_all = train_full["y60"].astype(np.int64)
    y_cls_te = test_full["y60"].astype(np.int64)
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]

    # V4 walk-forward split: train=date 0-75, val=date 76-79
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = Xtr_proc[m_t]
    X_va = Xtr_proc[m_va]
    y_regr_tr = y_regr_tr_all[m_t]
    y_regr_va = y_regr_tr_all[m_va]
    y_cls_tr = y_cls_tr_all[m_t]
    y_cls_va = y_cls_tr_all[m_va]

    print(f"  n_train={len(X_tr):,} n_val={len(X_va):,} n_test={len(Xte_proc):,}", flush=True)
    print(f"  y_regr_tr mean={y_regr_tr.mean():.6f} std={y_regr_tr.std():.6f}", flush=True)
    print(f"  y_regr_te mean={y_regr_te.mean():.6f} std={y_regr_te.std():.6f}", flush=True)

    cfg = SEED_CONFIGS[seed]
    rng = np.random.default_rng(seed * 7919 + 1)

    # aug_a_scale 100% ratio
    aug_idx = np.arange(len(X_tr))
    X_aug = aug_a_scale(X_tr, rng, 0.80, 1.20)
    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)

    sw_tr = class_balanced_weight(y_cls_tr_full, NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, NUM_CLASS)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names_used, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names_used, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression_l2",
        "learning_rate": args.learning_rate,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": 100,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": 5,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": 18,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
        "metric": "l2",
        "device": "gpu",
        "gpu_use_dp": False,
    }
    progress(seed, variant, "training")

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wb
            wandb = wb
            wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                       name=f"R_roll_tsrv-{variant}-seed{seed}",
                       config={"variant": variant, "seed": seed, "feat_dim": feat_dim,
                               **cfg},
                       tags=["R_roll_tsrv", variant, f"seed{seed}", "L2", "gpu"])
        except Exception as e:
            print(f"  wandb init failed: {e!r}", flush=True)
            use_wandb = False

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
    print(f"  trained {train_time:.1f}s best_iter={best_iter}", flush=True)

    model_path = os.path.join(WORK_DIR, f"model_{variant}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Eval
    va_pred = predict_chunked(booster, X_va)
    te_pred = predict_chunked(booster, Xte_proc)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  VAL mse={va_mse:.7f} corr={va_corr:.4f}", flush=True)
    print(f"  TEST mse={te_mse:.7f} corr={te_corr:.4f}", flush=True)

    # IMPORTANT: For LOSO eval downstream, we need pred_dmid_norm in the SAME scale
    # as standard target (since DE thresh is calibrated on Δmid_norm scale).
    # For V1 the model predicts clean-slope-target; we re-scale predictions to behave
    # as Δmid_norm by interpreting them as slope*60/(mp+1) = projected Δmid_norm.
    # That IS the same scale as baseline target by construction (slope*60 = projected
    # 60-tick move). So we can use te_pred directly as "pred_dmid_norm".
    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": baseline_target(mp_t_te, mp_th_te).astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(WORK_DIR, f"pred_{variant}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    # Quick raw EV-gate sanity at k=1
    fee_thr = 2.0 * FEE
    a = np.full(len(te_pred), 1, dtype=np.int8)
    a[te_pred > fee_thr] = 2
    a[te_pred < -fee_thr] = 0
    side = a.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((a != 1).sum())
    print(f"  TEST EV-gate k=1: cum_pnl={cum_pnl:+.4f} n_active={n_active}", flush=True)

    summary = {
        "variant": variant, "seed": seed, "horizon": H,
        "feat_dim": feat_dim, "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "corr": va_corr},
        "test": {"mse": te_mse, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
    }
    with open(os.path.join(WORK_DIR, f"summary_{variant}_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        wandb.log({"val_mse": va_mse, "val_corr": va_corr,
                   "test_mse": te_mse, "test_corr": te_corr,
                   "test_ev_gate_k1_cum_pnl": cum_pnl,
                   "test_ev_gate_k1_n_active": n_active,
                   "best_iter": best_iter, "train_time_sec": train_time})
        wandb.finish()

    progress(seed, variant, "done", best_iter=best_iter, val_mse=va_mse, test_mse=te_mse,
             test_corr=te_corr, ev_gate_k1_cum_pnl=cum_pnl, ev_gate_k1_n_active=n_active)
    print(f"DONE {variant} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
