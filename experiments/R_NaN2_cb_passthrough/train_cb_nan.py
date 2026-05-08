"""R-NaN2: CatBoost Huber on Δmid_norm, comparing 3 NaN-handling variants.

Variants (--variant):
  - zero        : np.nan_to_num(X, nan=0.0) before training (counterfactual baseline)
  - min         : NaN passthrough + nan_mode='Min' (CB default; matches current pipeline)
  - max         : NaN passthrough + nan_mode='Max'

Same hero loss as T99 winner: Huber:delta=0.001, GPU.

Cache layout on remote:
  /root/lwb_remote_pkg/cache/schemeP_{train,val,test}.npz
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

from catboost import CatBoostRegressor, Pool

CACHE_DIR = "/root/lwb_remote_pkg/cache"
OUT_DIR = "/root/lwb_work_v3"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

SEED_CONFIGS = {
    42: dict(seed=42, depth=6, l2_leaf_reg=3.0, rsm=0.8, subsample=0.8),
    1:  dict(seed=1,  depth=6, l2_leaf_reg=1.0, rsm=0.6, subsample=0.7),
    7:  dict(seed=7,  depth=7, l2_leaf_reg=2.0, rsm=0.7, subsample=0.85),
}

T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP = T59_FAIL + STAGE5_FAIL


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_scale(X, rng, lo=0.8, hi=1.2):
    """Scale aug — preserves NaN (NaN * float = NaN)."""
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return X * scales


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def vectorized_pnl(pred_action, mp_t, mp_th):
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeP_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def progress(step, **extra):
    p = os.path.join(OUT_DIR, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


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
    return (X_tr, y_cls_tr, y_regr_tr,
            X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=["zero", "min", "max"], required=True)
    ap.add_argument("--alpha", type=float, default=0.001, help="Huber delta")
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--early-stopping", type=int, default=80)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    variant = args.variant
    H = 60
    tag = f"cb_huber_a{args.alpha:g}_{variant}"
    print(f"=== R-NaN2 CB Huber seed={seed} variant={variant} ===", flush=True)

    progress("loading_cache", seed=seed, variant=variant)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    feat_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_path) as f:
        all_feat_names = [line.strip() for line in f]
    name_to_idx = {n: i for i, n in enumerate(all_feat_names)}
    drop_idx = set()
    for n in DROP:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
    keep_idx = np.array(
        [i for i in range(train_full["X"].shape[1]) if i not in drop_idx],
        dtype=np.int64,
    )
    feat_dim = len(keep_idx)
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(feat_names)), "sym/date/time in features!"

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va) = build_v4_split(train_full, keep_idx)
    print(f"  feat_dim={feat_dim}, n_train={len(X_tr):,}, n_val={len(X_va):,}", flush=True)

    X_te = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    # NaN counts before treatment
    nan_tr_pre = int(np.isnan(X_tr).sum())
    nan_va_pre = int(np.isnan(X_va).sum())
    nan_te_pre = int(np.isnan(X_te).sum())
    print(f"  NaN counts (pre): train={nan_tr_pre:,}  val={nan_va_pre:,}  test={nan_te_pre:,}",
          flush=True)

    # Variant treatment
    if variant == "zero":
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
        cb_nan_mode = "Min"  # default; irrelevant since no NaN
    elif variant == "min":
        cb_nan_mode = "Min"
    elif variant == "max":
        cb_nan_mode = "Max"
    print(f"  variant={variant}  cb_nan_mode={cb_nan_mode}", flush=True)
    print(f"  NaN counts (post-treat): train={int(np.isnan(X_tr).sum()):,}  "
          f"val={int(np.isnan(X_va).sum()):,}  test={int(np.isnan(X_te).sum()):,}", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"R-NaN2-cb-{variant}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project="liangwenbei",
                       entity="cjxh21-Tsinghua University",
                       name=run_name,
                       config={"task": "R-NaN2", "model": "catboost",
                               "loss_function": "Huber",
                               "alpha": args.alpha, "variant": variant,
                               "cb_nan_mode": cb_nan_mode,
                               "n_features": feat_dim, "horizon": H, "seed": seed,
                               "nan_train_pre": nan_tr_pre},
                       tags=["R-NaN2", "cb", "huber", variant, f"seed{seed}"])
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)

    progress("augmenting", seed=seed, variant=variant)
    seed_rng = np.random.default_rng(seed * 7919 + 1)
    n_aug = len(X_tr)  # aug_ratio=1.0
    aug_idx = np.arange(len(X_tr))
    X_aug = aug_a_scale(X_tr[aug_idx], seed_rng, lo=0.80, hi=1.20)
    y_cls_aug = y_cls_tr[aug_idx]
    y_regr_aug = y_regr_tr[aug_idx]
    X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,}  NaN(post-aug)={int(np.isnan(X_tr_full).sum()):,}",
          flush=True)

    train_pool = Pool(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                      feature_names=feat_names)
    val_pool = Pool(X_va, label=y_regr_va, weight=sw_va, feature_names=feat_names)

    cb_params = dict(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=cfg["depth"],
        l2_leaf_reg=cfg["l2_leaf_reg"],
        loss_function=f"Huber:delta={args.alpha:g}",
        eval_metric="RMSE",
        random_seed=seed,
        rsm=1.0,  # GPU mode requires rsm=1
        bootstrap_type="Bernoulli",
        subsample=cfg["subsample"],
        verbose=200,
        early_stopping_rounds=args.early_stopping,
        allow_writing_files=False,
        nan_mode=cb_nan_mode,
        task_type="GPU",
        devices="0",
    )

    progress("training", seed=seed, variant=variant)
    model = CatBoostRegressor(**cb_params)
    t_start = time.time()
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    train_time = time.time() - t_start
    best_iter = int(model.tree_count_)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    model_path = os.path.join(OUT_DIR, f"model_{tag}_seed{seed}.cbm")
    model.save_model(model_path)

    va_pred = model.predict(X_va).astype(np.float32)
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1]) if va_pred.std() > 0 else 0.0
    print(f"  VAL: mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)

    te_pred = model.predict(X_te).astype(np.float32)
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1]) if te_pred.std() > 0 else 0.0

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
    pred_path = os.path.join(OUT_DIR, f"pred_{tag}_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    print(f"  TEST EV-gate k=1: cum_pnl={cum_pnl:+.4f} n_active={n_active:,}", flush=True)
    print(f"  TEST: mse={te_mse:.7f} mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    summary = {
        "task": f"R-NaN2 CB Huber seed={seed} variant={variant}",
        "seed": seed,
        "horizon": H,
        "variant": variant,
        "cb_nan_mode": cb_nan_mode,
        "alpha": args.alpha,
        "tag": tag,
        "n_features": feat_dim,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "nan_counts_pre": {"train": nan_tr_pre, "val": nan_va_pre, "test": nan_te_pre},
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
        "cb_params": {k: v for k, v in cb_params.items() if k != "verbose"},
    }
    out_path = os.path.join(OUT_DIR, f"summary_{tag}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({"best_iter": best_iter, "train_time_sec": float(train_time),
                   "val_mse": va_mse, "val_mae": va_mae, "val_corr": va_corr,
                   "test_mse": te_mse, "test_mae": te_mae, "test_corr": te_corr,
                   "test_ev_gate_k1_cum_pnl": cum_pnl,
                   "test_ev_gate_k1_n_active": n_active})
        wandb.finish()
    progress("done", seed=seed, variant=variant, cum_pnl=cum_pnl)
    print(f"\nRESULT_LINE: variant={variant} seed={seed} "
          f"cum_pnl={cum_pnl:+.4f} best_iter={best_iter}", flush=True)


if __name__ == "__main__":
    main()
