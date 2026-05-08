"""R2 Group-DRO over sym + LGB Huber.

Mirrors T99 train_lgb_huber.py pipeline, adds 2-stage Group-DRO reweighting:
  Stage 1: train baseline LGB Huber alpha=0.001 with class-balanced weights
  Stage 2: predict on training set, compute per-sym MAE = L_s
           sym_mult_s = softmax(eta * z(L_s)) * 5  (mean=1)
           new_weight[i] = base_w[i] * sym_mult[ sym[i] ]
  Stage 3: retrain from scratch with reweighted samples
For eta=0 this reduces to the T99 baseline.

Outputs to /root/lwb_work/:
  model_R2_eta{eta}_seed{S}.txt
  pred_R2_eta{eta}_seed{S}.parquet  (sym/t/midprice_t/midprice_th/pred_dmid_norm)
  summary_R2_eta{eta}_seed{S}.json

Usage:
  python3 train_dro.py --seed 42 --eta 0     # baseline
  python3 train_dro.py --seed 42 --eta 1.0   # group_dro eta=1.0
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
import lightgbm as lgb
from scipy.special import softmax

CACHE_DIR = "/root/lwb_remote_pkg/cache"
OUT_DIR = "/root/lwb_work"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
FEE_EFF = 2.0 * FEE

SEED_CONFIGS = {
    42:  dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8, num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,  feature_fraction=0.6, bagging_fraction=0.7, num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63, lambda_l2=2.0),
}

T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread",
            "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP = T59_FAIL + STAGE5_FAIL


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_scale(X, rng, lo=0.8, hi=1.2):
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return X * scales


def regr_target(mp_t, mp_th):
    mp_t_d = mp_t.astype(np.float64)
    mp_th_d = mp_th.astype(np.float64)
    return ((mp_th_d - mp_t_d) / (mp_t_d + 1.0)).astype(np.float32)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p}...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s:s+batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def build_v4(train_full, slicer):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y60_tr = train_full["y60"][m_t].astype(np.int64)
    sym_tr = train_full["sym"][m_t].astype(np.int8)
    yreg_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y60_va = train_full["y60"][m_va].astype(np.int64)
    sym_va = train_full["sym"][m_va].astype(np.int8)
    yreg_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])

    return (X_tr, y60_tr, yreg_tr, sym_tr,
            X_va, y60_va, yreg_va, sym_va)


def train_lgb(X_tr_full, yreg_tr_full, sw_tr,
              X_va, yreg_va, sw_va,
              feat_names, cfg, seed,
              num_boost_round=600, early_stopping=40, lr=0.05,
              use_gpu=True, alpha=0.001, verbose_period=200):
    dtrain = lgb.Dataset(X_tr_full, label=yreg_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=yreg_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)
    params = {
        "objective": "huber", "alpha": alpha, "metric": "l1",
        "learning_rate": lr,
        "num_leaves": int(cfg["num_leaves"]),
        "min_data_in_leaf": 100,
        "feature_fraction": float(cfg["feature_fraction"]),
        "bagging_fraction": float(cfg["bagging_fraction"]),
        "bagging_freq": 5,
        "lambda_l2": float(cfg["lambda_l2"]),
        "num_threads": 18, "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
    }
    if use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False
    booster = lgb.train(
        params=params, train_set=dtrain,
        num_boost_round=num_boost_round, valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=verbose_period),
        ],
    )
    return booster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--eta", type=float, default=0.0,
                    help="Group-DRO temperature; eta=0 → baseline (no DRO).")
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--no-wandb", action="store_true", default=True)
    args = ap.parse_args()

    seed = args.seed
    eta = args.eta
    is_baseline = (abs(eta) < 1e-9)
    tag = "baseline" if is_baseline else f"eta{eta:g}"
    print(f"=== R2 LGB Huber DRO seed={seed} eta={eta} ({'BASELINE' if is_baseline else 'TRICK'}) ===", flush=True)
    t_total0 = time.time()

    train_full = load_split("train")
    test_full = load_split("test")

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    total_dim = train_full["X"].shape[1]
    assert len(all_feat_names) == total_dim
    name2i = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = {name2i[n] for n in DROP if n in name2i}
    keep = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_names = [all_feat_names[i] for i in keep]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak

    (X_tr, y60_tr, yreg_tr, sym_tr,
     X_va, y60_va, yreg_va, sym_va) = build_v4(train_full, keep)
    print(f"  feat_dim={len(keep)} n_train={len(X_tr):,} n_val={len(X_va):,}", flush=True)

    X_te = test_full["X"][:, keep].astype(np.float32, copy=False)
    yreg_te = regr_target(test_full["mp_t"], test_full["mp_t60"])
    sym_te = test_full["sym"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)

    rng = np.random.default_rng(seed * 7919 + 1)
    aug_idx = np.arange(len(X_tr))  # full 1.0 aug ratio (concat 1× scaled copy)
    X_aug = aug_a_scale(X_tr[aug_idx], rng, lo=0.8, hi=1.2)
    sym_aug = sym_tr[aug_idx]
    y60_aug = y60_tr[aug_idx]
    yreg_aug = yreg_tr[aug_idx]
    X_full = np.concatenate([X_tr, X_aug], axis=0)
    sym_full = np.concatenate([sym_tr, sym_aug], axis=0)
    y60_full = np.concatenate([y60_tr, y60_aug], axis=0)
    yreg_full = np.concatenate([yreg_tr, yreg_aug], axis=0)
    print(f"  augmented n_train_full={len(X_full):,}", flush=True)

    base_w_tr = class_balanced_weight(y60_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y60_va, num_class=NUM_CLASS)

    # --- Stage 1: baseline train ---
    print(f"  [Stage 1] training baseline LGB (eta={eta})...", flush=True)
    t0 = time.time()
    booster1 = train_lgb(X_full, yreg_full, base_w_tr,
                          X_va, yreg_va, sw_va, feat_names, cfg, seed,
                          num_boost_round=args.num_boost_round,
                          alpha=args.alpha, verbose_period=200)
    t1 = time.time() - t0
    best_iter_1 = int(booster1.best_iteration)
    print(f"    Stage 1 done in {t1:.1f}s, best_iter={best_iter_1}", flush=True)

    if is_baseline:
        booster_final = booster1
        sym_mult = np.ones(5, dtype=np.float64)
        per_sym_train_mae = None
        best_iter_final = best_iter_1
        train_time_total = t1
    else:
        # --- Stage 2: per-sym training MAE ---
        print(f"  [Stage 2] per-sym training MAE (predict on full train set)...", flush=True)
        t2 = time.time()
        tr_pred = predict_chunked(booster1, X_full, batch=400_000)
        per_sym_mae = np.zeros(5, dtype=np.float64)
        for s in SYMS:
            m = (sym_full == s)
            per_sym_mae[s] = float(np.abs(tr_pred[m] - yreg_full[m]).mean())
        z_mae = (per_sym_mae - per_sym_mae.mean()) / (per_sym_mae.std() + 1e-12)
        sym_mult = softmax(eta * z_mae) * 5.0  # mean = 1.0
        new_w = base_w_tr * sym_mult[sym_full].astype(np.float32)
        print(f"    per_sym_mae={per_sym_mae.round(7).tolist()}", flush=True)
        print(f"    z_mae={z_mae.round(3).tolist()}", flush=True)
        print(f"    sym_mult(eta={eta})={sym_mult.round(3).tolist()}", flush=True)
        t_stage2 = time.time() - t2
        print(f"    Stage 2 done in {t_stage2:.1f}s", flush=True)

        # --- Stage 3: retrain ---
        print(f"  [Stage 3] retraining with DRO weights...", flush=True)
        t3 = time.time()
        booster2 = train_lgb(X_full, yreg_full, new_w,
                              X_va, yreg_va, sw_va, feat_names, cfg, seed,
                              num_boost_round=args.num_boost_round,
                              alpha=args.alpha, verbose_period=200)
        t_stage3 = time.time() - t3
        best_iter_2 = int(booster2.best_iteration)
        print(f"    Stage 3 done in {t_stage3:.1f}s, best_iter={best_iter_2}", flush=True)
        booster_final = booster2
        per_sym_train_mae = per_sym_mae.tolist()
        best_iter_final = best_iter_2
        train_time_total = t1 + t_stage2 + t_stage3

    # --- Predict + save ---
    model_path = os.path.join(OUT_DIR, f"model_R2_{tag}_seed{seed}.txt")
    booster_final.save_model(model_path, num_iteration=best_iter_final)

    va_pred = predict_chunked(booster_final, X_va)
    va_mae = float(np.abs(va_pred - yreg_va).mean())
    va_corr = float(np.corrcoef(va_pred, yreg_va)[0, 1])

    te_pred = predict_chunked(booster_final, X_te)
    te_mae = float(np.abs(te_pred - yreg_te).mean())
    te_corr = float(np.corrcoef(te_pred, yreg_te)[0, 1])

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": sym_te.astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_dmid_norm": yreg_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(OUT_DIR, f"pred_R2_{tag}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    # symmetric EV-gate k=1 PnL with fee_thr
    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    side = pa.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    # per-sym EV-gate
    per_sym_pnl = []
    for s in SYMS:
        m = (sym_te == s)
        per_sym_pnl.append(float(pnl[m].sum()))
    print(f"  TEST EV-gate k=1 cum_pnl={cum_pnl:+.4f} per_sym={[round(x,2) for x in per_sym_pnl]}", flush=True)

    summary = {
        "task": f"R2 LGB Huber DRO seed={seed} eta={eta}",
        "seed": seed, "eta": eta, "is_baseline": is_baseline, "tag": tag,
        "alpha": args.alpha,
        "best_iter_final": best_iter_final,
        "train_time_sec_total": float(train_time_total),
        "per_sym_train_mae": per_sym_train_mae,
        "sym_mult": sym_mult.tolist(),
        "val": {"mae": va_mae, "corr": va_corr},
        "test": {"mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active,
                 "per_sym_ev_gate": per_sym_pnl},
    }
    out_path = os.path.join(OUT_DIR, f"summary_R2_{tag}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}  total={time.time()-t_total0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
