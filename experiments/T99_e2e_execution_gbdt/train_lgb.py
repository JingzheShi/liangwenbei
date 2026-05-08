"""T99: LightGBM regression on Δmid_norm with various built-in losses for end-to-end
execution-aware training (or robust regression as a proxy).

Supported objectives:
  - regression_l2  (baseline = T75)
  - regression_l1  (MAE)
  - huber          (alpha = transition threshold; ~ noise scale)
  - quantile       (alpha = quantile level; 0.5 = median)
  - custom_spoplus (custom SPO+ surrogate gradient — see below)

Pipeline matches T75:
  Target: y_regr = (mp_th - mp_t)/(mp_t + 1)
  Features: schemeP cache 359 dims (T59+Stage5 fail dropped)
  V4 split: train date 0-75, val date 76-79, test 96-119 5-sym
  aug_a per-feat scale [0.8, 1.2] concat; sample weight = class-balanced 3-class

For custom_spoplus:
  L_spo+(p, y) = max(|2p - y| - fee_eff, 0) - z*(y)·(2p-y) + fee_eff·|z*(y)|
  z*(y) = sign(y) if |y|>fee_eff else 0
  ∂L/∂p = 2 · 1[|2p-y|>fee_eff] · sign(2p-y) - 2 · z*(y)
  Hess = const 1.0 (LGB requires positive — Newton step degenerate to gradient descent)
  fee_eff is approximated as 2*FEE = 2e-4 (close enough on this scale).
  Add λ_l2 * (p - y)^2 / 2 anchor term to keep gradients bounded:
    ∂(L_anchor)/∂p = λ_l2 * (p - y), Hess = λ_l2.

Saves model_T99_<obj>_seed{S}.txt and pred_T99_<obj>_seed{S}.parquet.
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
CACHE_DIR = os.path.join(T68_DIR, "cache")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
FEE_EFF_APPROX = 2.0 * FEE

# Same seed configs as T75
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


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


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


def make_spoplus_obj(fee_eff, lambda_l2_anchor=10.0):
    """Custom SPO+ subgradient objective (closure over fee_eff scalar).

    Implements (LightGBM custom_obj signature):
        fn(preds, train_data) -> (grad, hess)

    Loss per sample (weighted automatically by LightGBM via Dataset.weight):
        L = max(|2p - y| - fe, 0) - z*·(2p - y) + fe·|z*|
            + λ/2 · (p - y)^2  (anchor)

    Subgradient w.r.t. p:
        dL/dp = 2 · indicator(|2p-y|>fe) · sign(2p-y) - 2·z* + λ·(p-y)

    Hess (regularization to enable LGB Newton step):
        d²L/dp² = λ  (positive, since the SPO+ piecewise-linear part contributes 0)

    fee_eff is treated as a constant (sample-mean ≈ 2*FEE). Per-sample fee_eff
    is roughly 2*FEE on this scale and the SPO+ formulation tolerates a small
    constant shift here.
    """
    fe = float(fee_eff)
    lam = float(lambda_l2_anchor)

    def obj(preds, train_data):
        y = train_data.get_label().astype(np.float64)
        p = preds.astype(np.float64)
        z_star = np.where(y > fe, 1.0, np.where(y < -fe, -1.0, 0.0))
        spread = 2.0 * p - y
        active = (np.abs(spread) > fe).astype(np.float64)
        sign_spread = np.where(spread >= 0.0, 1.0, -1.0)
        grad_spo = 2.0 * active * sign_spread - 2.0 * z_star
        grad_anchor = lam * (p - y)
        grad = (grad_spo + grad_anchor).astype(np.float64)
        hess = np.full_like(grad, lam)
        return grad, hess
    return obj


def make_pnl_eval(fee_thr=FEE_EFF_APPROX):
    """Custom eval metric: MAE of pred against y (placeholder).
    Actual PnL evaluation is done offline (need mp_t/mp_th)."""
    def eval_fn(preds, dataset):
        y = dataset.get_label()
        return ("mae", float(np.mean(np.abs(preds - y))), False)  # lower=better
    return eval_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--objective", default="regression_l2",
                    choices=["regression_l2", "regression_l1", "huber",
                             "quantile", "custom_spoplus"])
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="For huber: transition threshold; for quantile: q.")
    ap.add_argument("--spo-lambda-l2", type=float, default=10.0,
                    help="For custom_spoplus: anchor L2 coefficient.")
    ap.add_argument("--spo-fee-eff", type=float, default=FEE_EFF_APPROX,
                    help="For custom_spoplus: fee_eff approximation.")
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
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    obj_str = args.objective
    if args.tag:
        tag = args.tag
    else:
        if obj_str in ("huber", "quantile"):
            tag = f"{obj_str}_a{args.alpha:g}"
        elif obj_str == "custom_spoplus":
            tag = f"spop_lam{args.spo_lambda_l2:g}_fe{args.spo_fee_eff:g}"
        else:
            tag = obj_str
    print(f"=== T99 LGB seed={seed} h={H} obj={obj_str} tag={tag} ===", flush=True)

    progress("loading_caches", seed=seed, tag=tag)
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
    feat_names = [all_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info) = build_v4_split(train_full, slicer)
    print(f"  feat_dim={feat_dim}, {info}", flush=True)
    print(f"  y_regr_tr stats: mean={y_regr_tr.mean():.6e} std={y_regr_tr.std():.6e}",
          flush=True)

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
            run_name = (f"T99-lgb-{tag}-seed{seed}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "task": "T99_e2e_execution_gbdt",
                    "objective": obj_str,
                    "alpha": args.alpha,
                    "split_info": info,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T99", "lgb", obj_str, f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, tag=tag)

    seed_rng = np.random.default_rng(seed * 7919 + 1)
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
    else:
        X_tr_full = X_tr
        y_cls_tr_full = y_cls_tr
        y_regr_tr_full = y_regr_tr

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}  n_val={len(X_va):,}", flush=True)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    custom_obj = None
    if obj_str == "custom_spoplus":
        custom_obj = make_spoplus_obj(args.spo_fee_eff, args.spo_lambda_l2)

    params = {
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
    }
    if obj_str in ("huber", "quantile"):
        params["objective"] = obj_str
        params["alpha"] = args.alpha
        params["metric"] = "l1" if obj_str == "huber" else "quantile"
    elif obj_str == "regression_l1":
        params["objective"] = "regression_l1"
        params["metric"] = "l1"
    elif obj_str == "regression_l2":
        params["objective"] = "regression_l2"
        params["metric"] = "l2"
    elif obj_str == "custom_spoplus":
        # No "objective" entry — passed via fobj
        params["metric"] = "l2"  # for early-stopping monitoring
    else:
        raise RuntimeError(f"unknown obj {obj_str}")

    if args.use_gpu and obj_str != "custom_spoplus":
        # GPU + custom obj is brittle in some LGB builds — use CPU for safety.
        params["device"] = "gpu"
        params["gpu_use_dp"] = False
    if custom_obj is not None:
        # In LGB 4.x, custom obj is passed via params["objective"]
        params["objective"] = custom_obj

    t_start = time.time()
    booster = lgb.train(
        params=params, train_set=dtrain,
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

    model_path = os.path.join(HERE, f"model_T99_{tag}_seed{seed}.txt")
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
    pred_path = os.path.join(HERE, f"pred_T99_{tag}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    print(f"  TEST EV-gate k=1: cum_pnl={cum_pnl:+.4f} n_active={n_active:,}",
          flush=True)

    summary = {
        "task": f"T99 LGB {obj_str} seed={seed}",
        "seed": seed,
        "horizon": H,
        "objective": obj_str,
        "tag": tag,
        "alpha": args.alpha,
        "n_features": feat_dim,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T99_{tag}_seed{seed}.json")
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
    progress("done", seed=seed, tag=tag,
             best_iter=best_iter, val_mse=va_mse,
             test_corr=te_corr, test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
