"""T92: Regression on Δmid_norm + magnitude-based sample weight (T57 idea revived under L2).

Hypothesis (R44):
  T57 used PnL-aware weights on CE — failed (-3 ~ -11) because CE collapses |Δmid|
  so the weight has nothing to grip. Under L2 regression, weight directly multiplies
  (pred - Δmid)^2 → upweighted samples drive the fit toward larger |Δmid| residuals,
  which is exactly where PnL is made.

Variants:
  A_linear : w = |Δmid_norm|
  B_sqrt   : w = sqrt(|Δmid_norm|)
  C_capped : w = min(|Δmid_norm|, 95th percentile)

All weights have a small floor `eps_w` to keep tiny-|Δmid| samples in (LightGBM
ignores w=0 rows). Weights are mean-normalized to 1.0.

Reuses T75/iter_013 setup: T68 schemeP cache (drop the 11 KS-fail features → 359-d),
V4 walk-forward val (train=date 0-75, val=date 76-79), aug_a per-(sample,feat)
[0.80, 1.20], same SEED_CONFIGS as T64/T70/T75, GPU LightGBM regression_l2.
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


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def magnitude_weight(y_regr: np.ndarray, y_cls: np.ndarray, scheme: str,
                     eps_w: float, q_cap: float = 0.95,
                     train_cap_value: float = None,
                     train_q50_value: float = None,
                     alpha: float = 1.0,
                     num_class: int = 3) -> tuple[np.ndarray, dict]:
    """Compute sample weight from |y_regr| under one of several schemes.

    train_cap_value, train_q50_value: if provided (val/test), reuse train statistics
        to avoid val/test distribution leak.
    """
    abs_ret = np.abs(y_regr.astype(np.float64))
    info = {"scheme": scheme, "eps_w": float(eps_w),
            "abs_ret_mean": float(abs_ret.mean()),
            "abs_ret_std": float(abs_ret.std()),
            "abs_ret_q50": float(np.quantile(abs_ret, 0.50)),
            "abs_ret_q95": float(np.quantile(abs_ret, 0.95))}
    q50 = float(train_q50_value) if train_q50_value is not None else info["abs_ret_q50"]
    q95 = info["abs_ret_q95"]
    info["q50_used"] = q50

    if scheme == "A_linear":
        w = abs_ret
    elif scheme == "B_sqrt":
        w = np.sqrt(abs_ret)
    elif scheme == "C_capped":
        if train_cap_value is None:
            cap = float(np.quantile(abs_ret, q_cap))
        else:
            cap = float(train_cap_value)
        info["q_cap"] = q_cap
        info["cap"] = cap
        w = np.minimum(abs_ret, cap)
    elif scheme == "D_offset":
        # w = |y| + q50 (additive offset → low-|y| samples retain ~q50 weight)
        w = abs_ret + q50
        info["alpha"] = alpha
    elif scheme == "E_floor_q50":
        # w = max(|y|, q50) — floor at median, no upper bound
        w = np.maximum(abs_ret, q50)
    elif scheme == "F_class_mag":
        # class_balanced(y_cls) * (1 + alpha * |y|/q95)
        cb = class_balanced_weight_local(y_cls, num_class=num_class)
        mag_factor = 1.0 + alpha * (abs_ret / max(q95, 1e-12))
        w = cb.astype(np.float64) * mag_factor
        info["alpha"] = alpha
    elif scheme == "G_sqrt_offset":
        # sqrt(|y| + q50)  — milder than B
        w = np.sqrt(abs_ret + q50)
    else:
        raise ValueError(f"unknown scheme {scheme}")

    w = np.maximum(w, eps_w)
    mean_w = float(w.mean())
    info["mean_pre_norm"] = mean_w
    w = w * (1.0 / mean_w)
    info["mean_post_norm"] = float(w.mean())
    info["weight_q01"] = float(np.quantile(w, 0.01))
    info["weight_q50"] = float(np.quantile(w, 0.50))
    info["weight_q99"] = float(np.quantile(w, 0.99))
    return w.astype(np.float32), info


def class_balanced_weight_local(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--objective", default="regression_l2",
                    choices=["regression_l2", "regression_l1"])
    ap.add_argument("--weight-variant", required=True,
                    choices=["A_linear", "B_sqrt", "C_capped",
                             "D_offset", "E_floor_q50", "F_class_mag", "G_sqrt_offset"])
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="alpha multiplier for F_class_mag")
    ap.add_argument("--eps-w", type=float, default=1e-6,
                    help="weight floor — small enough so tiny-|Δmid| effectively excluded "
                         "from gradient but not literally w=0 (LightGBM drops w=0 rows)")
    ap.add_argument("--q-cap", type=float, default=0.95)
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
    variant = args.weight_variant
    print(f"=== T92 regr-Δmid + mag-weight seed={seed} h={H} variant={variant} ===", flush=True)

    progress("loading_caches", seed=seed, variant=variant)
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

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info) = build_v4_split(train_full, slicer)
    print(f"  {info}", flush=True)
    print(f"  y_regr_tr stats: mean={y_regr_tr.mean():.6f} std={y_regr_tr.std():.6f} "
          f"|y| q95={np.quantile(np.abs(y_regr_tr), 0.95):.6f}", flush=True)
    print(f"  y_regr_va stats: mean={y_regr_va.mean():.6f} std={y_regr_va.std():.6f}",
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
            run_name = (f"T92-{variant}-seed{seed}-{args.objective}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "task": "T92_magnitude_weight_regr",
                    "objective": args.objective,
                    "weight_variant": variant,
                    "split_info": info,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T92", "regression", "mag-weight", variant, f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, variant=variant, split_info=info)

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

    sw_tr, w_info_tr = magnitude_weight(y_regr_tr_full, y_cls_tr_full, variant,
                                        eps_w=args.eps_w, q_cap=args.q_cap,
                                        alpha=args.alpha, num_class=NUM_CLASS)
    train_cap_value = w_info_tr.get("cap")
    train_q50_value = w_info_tr.get("q50_used")
    sw_va, w_info_va = magnitude_weight(y_regr_va, y_cls_va, variant,
                                        eps_w=args.eps_w, q_cap=args.q_cap,
                                        train_cap_value=train_cap_value,
                                        train_q50_value=train_q50_value,
                                        alpha=args.alpha, num_class=NUM_CLASS)
    print(f"  train weight info: {w_info_tr}", flush=True)
    print(f"  val   weight info: {w_info_va}", flush=True)
    print(f"  n_train_used={len(X_tr_full):,}  n_val={len(X_va):,}", flush=True)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

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

    tag = f"T92_{variant}_seed{seed}"
    if args.tag:
        tag += f"_{args.tag}"
    model_path = os.path.join(HERE, f"model_{tag}.txt")
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
    pred_path = os.path.join(HERE, f"pred_{tag}.parquet")
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

    pnl_per_sym = []
    for sym_id in SYMS:
        m = test_full["sym"] == sym_id
        if m.sum() > 0:
            pnl_per_sym.append(float(pnl[m].sum()))
    sum_per_sym = float(sum(pnl_per_sym))
    print(f"  TEST EV-gate k=1 (thr={fee_thr:.5f}): cum_pnl={cum_pnl:+.4f} sum_per_sym={sum_per_sym:+.4f} "
          f"n_active={n_active:,}", flush=True)
    print(f"    per_sym = {[f'{x:+.4f}' for x in pnl_per_sym]}", flush=True)

    summary = {
        "task": f"T92 mag-weight regr seed={seed} variant={variant}",
        "seed": seed,
        "variant": variant,
        "horizon": H,
        "objective": args.objective,
        "split_info": info,
        "n_features": feat_dim,
        "drop_names": list(DROP_NAMES),
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl,
                 "ev_gate_k1_sum_per_sym": sum_per_sym,
                 "ev_gate_k1_per_sym": pnl_per_sym,
                 "ev_gate_k1_n_active": n_active},
        "weight_info_train": w_info_tr,
        "weight_info_val": w_info_va,
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_{tag}.json")
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
            "test_ev_gate_k1_sum_per_sym": sum_per_sym,
            "test_ev_gate_k1_n_active": n_active,
        })
        wandb.finish()
    progress("done", seed=seed, variant=variant,
             best_iter=best_iter, val_mse=va_mse, test_corr=te_corr,
             test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
