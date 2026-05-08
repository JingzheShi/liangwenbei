"""T91 = T35 ReVol+Savgol revival under L2-regression target.

Concatenates T68 schemeP (359-d after T59+Stage5 drops) with T35 schemeK_extra
(32-d: 12 ReVol + 20 Savgol features) → 391-d feature space.

Then trains the **identical T75 5-seed LGB regression protocol**:
  - target  = y_regr = (mp_t60 - mp_t)/(mp_t + 1)
  - V4 walk-forward (train date 0-75, val 76-79, test 96-119 5 sym)
  - 5 seeds {1, 7, 13, 42, 100} with same param table as T75
  - aug_a (per-(sample,feat) scale [0.80, 1.20])
  - sample weight = class_balanced on 3-class
  - GPU LightGBM, num_boost=600, ES=40 on val MSE

Saves model_T91_seed{S}.txt + pred_T91_seed{S}.parquet (same schema as T75).
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
T35_DIR = os.path.join(ROOT, "experiments", "T35_revol_savgol")
P_CACHE = os.path.join(T68_DIR, "cache")
K_CACHE = os.path.join(T35_DIR, "cache")

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


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def load_split(split, extra_select=None):
    """Return dict with the schemeP fields + concatenated X (P + ReVol/SG).

    extra_select: optional bool index over the 32 schemeK_extra columns to keep.
    """
    p_path = os.path.join(P_CACHE, f"schemeP_{split}.npz")
    k_path = os.path.join(K_CACHE, f"schemeK_extra_{split}.npz")
    print(f"  loading {p_path} ...", flush=True)
    P = np.load(p_path)
    print(f"  loading {k_path} ...", flush=True)
    K = np.load(k_path)
    # alignment guard
    assert np.array_equal(P["sym"], K["sym"])
    assert np.array_equal(P["date"], K["date"])
    assert np.array_equal(P["t"], K["t"])
    assert np.array_equal(P["sess_idx"], K["sess_idx"])
    Xp = P["X"]
    Xk = K["X_extra"]
    if extra_select is not None:
        Xk = Xk[:, extra_select]
    X = np.concatenate([Xp, Xk], axis=1)
    out = {k: P[k] for k in P.files}
    out["X"] = X
    out["X_p_dim"] = Xp.shape[1]
    out["X_k_dim"] = Xk.shape[1]
    return out


def build_v4_split(train_full, slicer):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    y_regr_va = regr_target(train_full["mp_t"][m_va], train_full["mp_t60"][m_va])
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]

    info = {
        "strategy": "V4",
        "train_dates": "0-75",
        "val_dates": "76-79 (walk-forward last 5%)",
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
    }
    return (X_tr, y_cls_tr, y_regr_tr,
            X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info)


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--objective", default="regression_l2",
                    choices=["regression_l2", "regression_l1"])
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--variant", default="v1_all32",
                    choices=["v1_all32", "v2_revol12", "v3_savgol20", "v4_revol_only6"],
                    help="which subset of T35 schemeK_extra (32 columns) to add")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    print(f"=== T91 ReVol regr seed={seed} h={H} obj={args.objective} variant={args.variant} ===",
          flush=True)

    # ---- Decide which schemeK_extra columns to keep ----
    with open(os.path.join(K_CACHE, "schemeK_extra_feat_names.txt")) as f:
        k_names_all = [line.strip() for line in f if line.strip()]
    assert len(k_names_all) == 32, len(k_names_all)
    if args.variant == "v1_all32":
        keep_k_mask = np.ones(32, dtype=bool)
    elif args.variant == "v2_revol12":
        keep_k_mask = np.array([n.startswith("revol_") for n in k_names_all])
        assert keep_k_mask.sum() == 12
    elif args.variant == "v3_savgol20":
        keep_k_mask = np.array([n.startswith("sg_") for n in k_names_all])
        assert keep_k_mask.sum() == 20
    elif args.variant == "v4_revol_only6":
        # only the 6 wmp1 ReVol features (the highest-gain T35 candidates: wmp1_sigma_hat=#1)
        keep_k_mask = np.array([n.startswith("revol_wmp1_") for n in k_names_all])
        assert keep_k_mask.sum() == 6
    else:
        raise ValueError(args.variant)
    k_keep_names = [n for n, m in zip(k_names_all, keep_k_mask) if m]
    print(f"  schemeK extra columns kept ({keep_k_mask.sum()}): {k_keep_names}", flush=True)

    progress("loading_caches", seed=seed, variant=args.variant)
    t0 = time.time()
    train_full = load_split("train", extra_select=keep_k_mask)
    test_full = load_split("test", extra_select=keep_k_mask)
    print(f"loaded in {time.time()-t0:.1f}s X shape train={train_full['X'].shape} "
          f"P_dim={train_full['X_p_dim']} K_dim={train_full['X_k_dim']}", flush=True)

    feat_names_path = os.path.join(P_CACHE, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_p_feat_names = [line.strip() for line in f]
    total_dim_p = train_full["X_p_dim"]
    assert len(all_p_feat_names) == total_dim_p, (len(all_p_feat_names), total_dim_p)
    full_feat_names = all_p_feat_names + k_keep_names

    name_to_idx = {n: i for i, n in enumerate(full_feat_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
        else:
            print(f"  WARN drop name not found: {n}", flush=True)
    keep_idx = np.array(
        [i for i in range(len(full_feat_names)) if i not in drop_idx], dtype=np.int64,
    )
    slicer = keep_idx
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim} (P: {total_dim_p} - {len(drop_idx)} drops + K: {keep_k_mask.sum()})",
          flush=True)

    feat_names = [full_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info) = build_v4_split(train_full, slicer)
    print(f"  {info}", flush=True)
    print(f"  y_regr_tr stats: mean={y_regr_tr.mean():.6f} std={y_regr_tr.std():.6f}",
          flush=True)
    print(f"  y_regr_va stats: mean={y_regr_va.mean():.6f} std={y_regr_va.std():.6f}",
          flush=True)

    # Test
    X_te = test_full["X"][:, slicer].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T91-revol-regr-{args.variant}-seed{seed}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "task": "T91_revol_regr_revival",
                    "variant": args.variant,
                    "objective": args.objective,
                    "split_info": info,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    "k_columns": k_keep_names,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T91", "regression", "revol", "revival", f"h{H}", args.variant],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, split_info=info, n_features=feat_dim, variant=args.variant)

    seed_rng = np.random.default_rng(seed * 7919 + 1)
    # Free train_full now to keep memory low; we still hold X_tr/X_va views.
    del train_full
    import gc; gc.collect()

    n_aug = int(round(args.aug_ratio * len(X_tr)))
    if n_aug > 0:
        if n_aug == len(X_tr):
            aug_idx = np.arange(len(X_tr))
        else:
            aug_idx = seed_rng.integers(0, len(X_tr), size=n_aug)
        X_aug = aug_a_scale(X_tr[aug_idx], seed_rng,
                            lo=args.aug_lo, hi=args.aug_hi)
        y_cls_aug = y_cls_tr[aug_idx]
        y_regr_aug = y_regr_tr[aug_idx]
        X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
        y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)
        del X_aug, X_tr, y_cls_aug, y_regr_aug, y_cls_tr, y_regr_tr
        gc.collect()
    else:
        X_tr_full = X_tr
        y_cls_tr_full = y_cls_tr
        y_regr_tr_full = y_regr_tr

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}  n_val={len(X_va):,}", flush=True)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=True)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=True)
    del y_cls_tr_full
    gc.collect()

    params = {
        "objective": args.objective,
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
        "metric": "l2" if args.objective == "regression_l2" else "l1",
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

    model_path = os.path.join(HERE, f"model_T91_{args.variant}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL: mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)

    te_pred = predict_chunked(booster, X_te)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  TEST: mse={te_mse:.7f} mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"].astype(np.int8),
        "date": test_full["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
        "t": test_full["t"].astype(np.int16),
        "true_label": y_cls_te.astype(np.int8),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(HERE, f"pred_T91_{args.variant}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    fee_thr = 2.0 * FEE
    pred_action = np.full(len(te_pred), 1, dtype=np.int8)
    pred_action[te_pred > fee_thr] = 2
    pred_action[te_pred < -fee_thr] = 0
    side = pred_action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th_te.astype(np.float64) - mp_t_te.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th_te + 1.0) + (mp_t_te + 1.0))
    denom = mp_t_te.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  TEST EV-gate k=1 (thr={fee_thr:.5f}): cum_pnl={cum_pnl:+.4f} n_active={n_active:,}",
          flush=True)

    summary = {
        "task": f"T91 ReVol regr {args.variant} seed={seed}",
        "seed": seed,
        "horizon": H,
        "objective": args.objective,
        "variant": args.variant,
        "k_columns_kept": k_keep_names,
        "split_info": info,
        "n_features": feat_dim,
        "drop_names": list(DROP_NAMES),
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T91_{args.variant}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_iter": best_iter,
            "train_time_sec": float(train_time),
            "val_mse": va_mse, "val_mae": va_mae, "val_corr": va_corr,
            "test_mse": te_mse, "test_mae": te_mae, "test_corr": te_corr,
            "test_ev_gate_k1_cum_pnl": cum_pnl,
            "test_ev_gate_k1_n_active": n_active,
        })
        wandb.finish()
    progress("done", seed=seed, variant=args.variant,
             best_iter=best_iter, val_mse=va_mse, test_corr=te_corr,
             test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
