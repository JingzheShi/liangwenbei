"""R_Pairwise_ALL — train baseline (schemeP-359) vs trick (+top30 pairs +7 globals).

Mirrors R3 / T117 recipe:
  Huber alpha=1e-3, GPU LightGBM, 600 rounds, early stop 40,
  num_leaves/feature_fraction/bagging/lambda_l2 per SEED_CONFIGS,
  aug_a per-feature scale [0.8, 1.2] ratio=1.0, class-balanced sample weight,
  V4 train/val (dates 0-75 / 76-79).

Trick = baseline + 30 top pairwise imbalances (selected by screen_topk.py)
       + 7 train-set global mid-price stats (broadcast).
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
HERE_CACHE = os.path.join(HERE, "cache")

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


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_scheme(split):
    p = os.path.join(SCHEME_CACHE, f"schemeP_{split}.npz")
    return {k: np.load(p)[k] for k in np.load(p).files}


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
    ap.add_argument("--arm", choices=("baseline", "trick"), required=True)
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    arm = args.arm
    tag = f"{arm}_seed{seed}"
    print(f"=== R_Pairwise_ALL {arm} seed={seed} ===", flush=True)
    progress("loading", arm=arm, seed=seed)

    t0 = time.time()
    train_full = load_scheme("train")
    test_full = load_scheme("test")
    print(f"  loaded schemeP in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(SCHEME_CACHE, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f if l.strip()]
    total_dim = train_full["X"].shape[1]
    assert len(all_names) == total_dim
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx],
                        dtype=np.int64)
    feat_names = [all_names[i] for i in keep_idx]
    assert "date" not in feat_names and "sym" not in feat_names

    if arm == "trick":
        pair_train_all = np.load(os.path.join(HERE_CACHE, "pairwise_train.npy"))
        pair_test_all = np.load(os.path.join(HERE_CACHE, "pairwise_test.npy"))
        top_idx = np.load(os.path.join(HERE_CACHE, "top_pairs_idx.npy"))
        with open(os.path.join(HERE_CACHE, "top_pairs_names.txt")) as f:
            top_pair_names = [l.strip() for l in f if l.strip()]
        pair_train = pair_train_all[:, top_idx]
        pair_test = pair_test_all[:, top_idx]
        del pair_train_all, pair_test_all

        glob = np.load(os.path.join(HERE_CACHE, "globals_consts.npz"))
        glob_vals = glob["values"].astype(np.float32)
        glob_names = [str(n) for n in glob["names"]]

        feat_names = feat_names + top_pair_names + glob_names
        print(f"  trick: top_pairs={pair_train.shape}  globals={glob_vals.shape}", flush=True)

    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr_base = train_full["X"][m_t][:, keep_idx].astype(np.float32, copy=False)
    X_va_base = train_full["X"][m_va][:, keep_idx].astype(np.float32, copy=False)
    if arm == "trick":
        n_tr = X_tr_base.shape[0]
        n_va = X_va_base.shape[0]
        glob_tr = np.broadcast_to(glob_vals, (n_tr, len(glob_vals))).astype(np.float32)
        glob_va = np.broadcast_to(glob_vals, (n_va, len(glob_vals))).astype(np.float32)
        X_tr = np.concatenate([X_tr_base, pair_train[m_t], glob_tr], axis=1)
        X_va = np.concatenate([X_va_base, pair_train[m_va], glob_va], axis=1)
        del glob_tr, glob_va
    else:
        X_tr = X_tr_base
        X_va = X_va_base
    del X_tr_base, X_va_base

    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    print(f"  X_tr{X_tr.shape}  X_va{X_va.shape}  feat_dim={X_tr.shape[1]}", flush=True)

    X_te_base = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    if arm == "trick":
        n_te = X_te_base.shape[0]
        glob_te = np.broadcast_to(glob_vals, (n_te, len(glob_vals))).astype(np.float32)
        X_te = np.concatenate([X_te_base, pair_test, glob_te], axis=1)
        del glob_te
    else:
        X_te = X_te_base
    del X_te_base
    y_regr_te = regr_target(test_full["mp_t"], test_full["mp_t60"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"R_Pair_ALL-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                       name=run_name, config={"arm": arm, "seed": seed,
                                                 "n_features": X_tr.shape[1],
                                                 "alpha": args.alpha,
                                                 "horizon": H},
                       tags=["R_Pairwise_ALL", arm])
        except Exception as e:
            print(f"  wandb init failed: {e!r}", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
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
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "huber", "alpha": args.alpha, "metric": "l1",
        "learning_rate": 0.05,
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
        "device": "gpu", "gpu_use_dp": False,
        "verbose": -1,
    }

    progress("training", arm=arm, seed=seed)
    t1 = time.time()
    booster = lgb.train(params=params, train_set=dtrain,
                        num_boost_round=args.num_boost_round,
                        valid_sets=[dval], valid_names=["val"],
                        callbacks=[lgb.early_stopping(args.early_stopping, verbose=True),
                                   lgb.log_evaluation(period=50)])
    train_time = time.time() - t1
    best_iter = int(booster.best_iteration)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    booster.save_model(os.path.join(HERE, f"model_RP_{tag}.txt"),
                       num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)

    te_pred = predict_chunked(booster, X_te)
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  TEST corr={te_corr:.4f}", flush=True)

    sess_map = {0: "am", 1: "pm"}
    df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]],
                            dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_RP_{tag}.parquet")
    df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(df):,} rows)", flush=True)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    print(f"  EV-gate k=1 cum_pnl={cum_pnl:+.4f} n_active={n_active:,}", flush=True)

    summary = {
        "arm": arm, "seed": seed, "horizon": H, "tag": tag,
        "n_features": X_te.shape[1], "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl,
                 "ev_gate_k1_n_active": n_active},
    }
    out_path = os.path.join(HERE, f"summary_RP_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    if use_wandb and wandb is not None:
        wandb.log({"best_iter": best_iter, "train_time_sec": train_time,
                   "val_mse": va_mse, "val_mae": va_mae, "val_corr": va_corr,
                   "test_corr": te_corr,
                   "test_ev_gate_k1_cum_pnl": cum_pnl,
                   "test_ev_gate_k1_n_active": n_active})
        wandb.finish()
    progress("done", arm=arm, seed=seed, best_iter=best_iter,
             test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
