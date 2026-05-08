"""T122: Retest 3 winning tricks on T75 LGB L2 baseline.

Variants (controlled by --variant):
  baseline:  T75 LGB L2 (regression_l2), schemeP 359-d, no HYD, no lag, no monotone
  hyd:       baseline + 16 HYD interaction features (R3) -> 375 dims
  lag:       baseline + 12 multi-scale logret lag features (R4) -> 371 dims
  monotone:  baseline + monotone constraints (T117 list) on the 18 features
  all3:      HYD + lag + monotone (375 + 12 = 387 dims, monotone on subset)

Objective: regression_l2 (NOT huber) — match T75 baseline.

Per (variant, seed): saves model_T122_{variant}_seed{S}.txt + pred_T122_{variant}_seed{S}.parquet.
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
HYD_CACHE = os.path.join(ROOT, "experiments", "R3_hyd_quartet", "cache")
LAG_CACHE = os.path.join(ROOT, "experiments", "R4_logret_lag", "cache")

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

# Monotone constraint dictionary (T117 list)
MONOTONE_FEATS = {
    "imbalance": 1, "wmp_balance_12": 1,
    "lb_intst": 1, "la_intst": -1,
    "mb_intst": 1, "ma_intst": -1,
    "cb_intst": -1, "ca_intst": 1,
    "ewma_ofi_a0.05_lvl1": 1, "ewma_ofi_a0.1_lvl1": 1,
    "ewma_ofi_a0.3_lvl1": 1, "ewma_ofi_a0.5_lvl1": 1,
    "signed_rv_W20": 1, "signed_rv_W50": 1, "signed_rv_W100": 1,
    "signed_bv_W20": 1, "signed_bv_W50": 1, "signed_bv_W100": 1,
}


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
    ap.add_argument("--variant", choices=("baseline", "hyd", "lag", "monotone", "all3"),
                    required=True)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    variant = args.variant
    use_hyd = variant in ("hyd", "all3")
    use_lag = variant in ("lag", "all3")
    use_monotone = variant in ("monotone", "all3")

    tag = f"{variant}_seed{seed}"
    print(f"=== T122 {variant} seed={seed} (hyd={use_hyd} lag={use_lag} mono={use_monotone}) ===",
          flush=True)
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

    hyd_train = hyd_test = None
    lag_train = lag_test = None
    extra_names = []
    if use_hyd:
        hyd_train = np.load(os.path.join(HYD_CACHE, "hyd_train.npy"))
        hyd_test = np.load(os.path.join(HYD_CACHE, "hyd_test.npy"))
        with open(os.path.join(HYD_CACHE, "hyd_feat_names.txt")) as f:
            hyd_names = [l.strip() for l in f if l.strip()]
        assert hyd_train.shape[0] == train_full["X"].shape[0]
        assert hyd_test.shape[0] == test_full["X"].shape[0]
        extra_names += hyd_names
        print(f"  +HYD: {hyd_train.shape[1]} dims", flush=True)
    if use_lag:
        d_lt = np.load(os.path.join(LAG_CACHE, "lagret_train.npz"), allow_pickle=True)
        d_le = np.load(os.path.join(LAG_CACHE, "lagret_test.npz"), allow_pickle=True)
        lag_train = d_lt["lag"].astype(np.float32, copy=False)
        lag_test = d_le["lag"].astype(np.float32, copy=False)
        lag_names = d_lt["feat_names"].tolist()
        assert lag_train.shape[0] == train_full["X"].shape[0]
        assert lag_test.shape[0] == test_full["X"].shape[0]
        extra_names += list(lag_names)
        print(f"  +LAG: {lag_train.shape[1]} dims", flush=True)

    feat_names_full = feat_names + extra_names

    # build train/val split (V4: train 0-75, val 76-79)
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    Xb_tr = train_full["X"][m_t][:, keep_idx].astype(np.float32, copy=False)
    Xb_va = train_full["X"][m_va][:, keep_idx].astype(np.float32, copy=False)
    parts_tr = [Xb_tr]
    parts_va = [Xb_va]
    if use_hyd:
        parts_tr.append(hyd_train[m_t])
        parts_va.append(hyd_train[m_va])
    if use_lag:
        parts_tr.append(lag_train[m_t])
        parts_va.append(lag_train[m_va])
    X_tr = np.concatenate(parts_tr, axis=1) if len(parts_tr) > 1 else Xb_tr
    X_va = np.concatenate(parts_va, axis=1) if len(parts_va) > 1 else Xb_va
    del Xb_tr, Xb_va, parts_tr, parts_va

    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    print(f"  X_tr{X_tr.shape}  X_va{X_va.shape}  feat_dim={X_tr.shape[1]}", flush=True)

    Xb_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    parts_te = [Xb_te]
    if use_hyd:
        parts_te.append(hyd_test)
    if use_lag:
        parts_te.append(lag_test)
    X_te = np.concatenate(parts_te, axis=1) if len(parts_te) > 1 else Xb_te
    del Xb_te, parts_te
    y_regr_te = regr_target(test_full["mp_t"], test_full["mp_t60"])
    sym_te = test_full["sym"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]

    # Build monotone constraints aligned to feat_names_full
    mc = None
    n_constrained = 0
    if use_monotone:
        mc = [0] * len(feat_names_full)
        for fn, sign in MONOTONE_FEATS.items():
            if fn in feat_names_full:
                idx = feat_names_full.index(fn)
                mc[idx] = sign
                n_constrained += 1
        print(f"  monotone-constrained features: {n_constrained}/{len(feat_names_full)}",
              flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T122-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                       name=run_name,
                       config={"variant": variant, "seed": seed,
                               "use_hyd": use_hyd, "use_lag": use_lag,
                               "use_monotone": use_monotone,
                               "n_features": X_tr.shape[1],
                               "objective": "regression_l2",
                               "n_constrained": n_constrained,
                               "horizon": H},
                       tags=["T122", "lgb_l2", variant, f"seed{seed}"])
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

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names_full, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names_full, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression_l2",
        "metric": "l2",
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
    if use_monotone:
        params["monotone_constraints"] = mc
        params["monotone_constraints_method"] = "advanced"
        params["monotone_penalty"] = 0.0

    t1 = time.time()
    booster = lgb.train(
        params=params, train_set=dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=True),
                   lgb.log_evaluation(period=50)],
    )
    train_time = time.time() - t1
    best_iter = int(booster.best_iteration)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    model_path = os.path.join(HERE, f"model_T122_{tag}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)

    te_pred = predict_chunked(booster, X_te)
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
    pred_path = os.path.join(HERE, f"pred_T122_{tag}.parquet")
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
        "task": f"T122 LGB L2 {variant} seed={seed}",
        "variant": variant, "seed": seed, "horizon": H, "tag": tag,
        "objective": "regression_l2",
        "use_hyd": use_hyd, "use_lag": use_lag, "use_monotone": use_monotone,
        "n_constrained": n_constrained,
        "n_features": X_te.shape[1],
        "best_iter": best_iter, "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl,
                 "ev_gate_k1_n_active": n_active,
                 "per_sym_ev_gate": per_sym},
    }
    out_path = os.path.join(HERE, f"summary_T122_{tag}.json")
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
