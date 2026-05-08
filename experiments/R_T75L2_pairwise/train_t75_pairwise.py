"""R_T75L2_pairwise: T75 LightGBM regression_l2 + pairwise imbalance + train-set globals.

Variants:
  - baseline : schemeP 359-d
  - trick    : schemeP 359-d + KS-filtered pairwise (~30) + 7 train-set globals
               (globals are mp_t row-wise deviations to train mean/std/percentiles)

Cache: /root/lwb_remote_pkg/cache/schemeP_{train,val,test}.npz
Out  : /root/lwb_work_t75_pairwise/

Constraints honored:
  - sym/date/time NOT features
  - all globals are train-only stats (no per-sym, no leak)
  - pairwise are in-window (per-row)
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

CACHE_DIR = "/root/lwb_remote_pkg/cache"
OUT_DIR = "/root/lwb_work_t75_pairwise"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# T75-style seed configs
SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
}

T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP = T59_FAIL + STAGE5_FAIL

# 20 price names → indices 6..15 (bid), 26..35 (ask)
PRICE_NAMES = [f"bid{i}" for i in range(1, 11)] + [f"ask{i}" for i in range(1, 11)]


def class_balanced_weight(y, num_class=3):
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_scale(X, rng, lo=0.8, hi=1.2):
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


def build_pairwise(X_prices, eps=1e-9):
    """X_prices: [N, 20] -> pairwise imbalance [N, 190]"""
    N, P = X_prices.shape
    n_pairs = P * (P - 1) // 2
    out = np.empty((N, n_pairs), dtype=np.float32)
    pair_idx = []
    k = 0
    for a in range(P):
        for b in range(a + 1, P):
            pa = X_prices[:, a].astype(np.float64)
            pb = X_prices[:, b].astype(np.float64)
            num = pa - pb
            den = np.abs(pa) + np.abs(pb) + eps
            out[:, k] = (num / den).astype(np.float32)
            pair_idx.append((a, b))
            k += 1
    return out, pair_idx


def ks_stat(a, b):
    """Two-sample KS statistic between arrays a, b."""
    # Subsample for speed if too large
    rng = np.random.default_rng(0)
    if len(a) > 100_000:
        a = rng.choice(a, 100_000, replace=False)
    if len(b) > 100_000:
        b = rng.choice(b, 100_000, replace=False)
    a = np.sort(a)
    b = np.sort(b)
    cdfs_a = np.searchsorted(a, np.concatenate([a, b]), side='right') / len(a)
    cdfs_b = np.searchsorted(b, np.concatenate([a, b]), side='right') / len(b)
    return float(np.abs(cdfs_a - cdfs_b).max())


def select_pairwise_by_ks(pw_train, pw_val, top_k=30):
    """Pick top_k pairs whose train vs val distribution is most stable (lowest KS)."""
    n_pairs = pw_train.shape[1]
    ks_scores = np.empty(n_pairs, dtype=np.float64)
    for k in range(n_pairs):
        # Drop NaN/inf
        a = pw_train[:, k]
        b = pw_val[:, k]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 100 or len(b) < 100:
            ks_scores[k] = 1.0
            continue
        ks_scores[k] = ks_stat(a, b)
    # ascending — most stable first
    order = np.argsort(ks_scores)
    return order[:top_k], ks_scores


def build_globals_rowwise(mp_t_query, train_stats):
    """7 row-wise deviations to train mp_t globals.
    train_stats: dict with mean,std,p10,p25,p50,p75,p90.
    Returns [N, 7] features, plus the global stats themselves as constants (for log).
    """
    m = train_stats["mean"]
    s = train_stats["std"] + 1e-12
    f0 = (mp_t_query - m) / s
    f1 = mp_t_query - train_stats["p10"]
    f2 = mp_t_query - train_stats["p25"]
    f3 = mp_t_query - train_stats["p50"]
    f4 = mp_t_query - train_stats["p75"]
    f5 = mp_t_query - train_stats["p90"]
    # rank = where in [p10,p90] this mp falls (clipped)
    # bucket index (0..4) reused as feature
    f6 = np.searchsorted(
        np.array([train_stats["p10"], train_stats["p25"], train_stats["p50"],
                  train_stats["p75"], train_stats["p90"]], dtype=np.float64),
        mp_t_query.astype(np.float64),
        side="right",
    ).astype(np.float32)
    return np.stack([f0, f1, f2, f3, f4, f5, f6], axis=1).astype(np.float32)


def build_v4_split(train_full, slicer):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t][:, slicer].astype(np.float32, copy=False)
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    mp_t_tr = train_full["mp_t"][m_t]
    mp_th_tr = train_full["mp_t60"][m_t]
    y_regr_tr = regr_target(mp_t_tr, mp_th_tr)

    X_va = train_full["X"][m_va][:, slicer].astype(np.float32, copy=False)
    y_cls_va = train_full["y60"][m_va].astype(np.int64)
    mp_t_va = train_full["mp_t"][m_va]
    mp_th_va = train_full["mp_t60"][m_va]
    y_regr_va = regr_target(mp_t_va, mp_th_va)

    return (X_tr, y_cls_tr, y_regr_tr, mp_t_tr,
            X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, m_t, m_va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", choices=["baseline", "trick"], required=True)
    ap.add_argument("--top-k-pairs", type=int, default=30)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    seed = args.seed
    variant = args.variant
    H = 60
    tag = f"T75L2_{variant}"
    print(f"=== R_T75L2_pairwise seed={seed} variant={variant} ===", flush=True)

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
    base_feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    assert not (forbidden & set(base_feat_names)), "leak in base!"
    base_dim = len(base_feat_names)
    print(f"  base_dim={base_dim}", flush=True)

    (X_tr_base, y_cls_tr, y_regr_tr, mp_t_tr,
     X_va_base, y_cls_va, y_regr_va, mp_t_va, mp_th_va,
     m_tr_mask, m_va_mask) = build_v4_split(train_full, keep_idx)
    print(f"  n_train={len(X_tr_base):,}  n_val={len(X_va_base):,}", flush=True)

    # Test
    X_te_base = test_full["X"][:, keep_idx].astype(np.float32, copy=False)
    y_cls_te = test_full[f"y{H}"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full[f"mp_t{H}"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sym_te = test_full["sym"]
    print(f"  n_test={len(X_te_base):,}", flush=True)

    # Build extras for trick variant
    extra_feat_names = []
    if variant == "trick":
        progress("building_extras", seed=seed)
        # Price columns from raw X
        price_idx = np.array([name_to_idx[n] for n in PRICE_NAMES], dtype=np.int64)
        prices_tr = train_full["X"][m_tr_mask][:, price_idx]
        prices_va = train_full["X"][m_va_mask][:, price_idx]
        prices_te = test_full["X"][:, price_idx]

        print(f"  building 190 pairwise from 20 prices ...", flush=True)
        t1 = time.time()
        pw_tr, pair_idx_list = build_pairwise(prices_tr)
        pw_va, _ = build_pairwise(prices_va)
        pw_te, _ = build_pairwise(prices_te)
        print(f"    done in {time.time()-t1:.1f}s", flush=True)

        # KS-select top_k
        print(f"  KS train-vs-val select top {args.top_k_pairs} pairs ...", flush=True)
        t1 = time.time()
        sel, ks_scores = select_pairwise_by_ks(pw_tr, pw_va, top_k=args.top_k_pairs)
        print(f"    done in {time.time()-t1:.1f}s ks_min={ks_scores[sel[0]]:.4f} ks_max(top{args.top_k_pairs})={ks_scores[sel[-1]]:.4f}",
              flush=True)
        # Save selection (only for seed=42 to avoid duplication)
        if seed == 42 and variant == "trick":
            sel_info = {
                "top_k": int(args.top_k_pairs),
                "selected": [{"k": int(k),
                              "pair": [PRICE_NAMES[pair_idx_list[k][0]], PRICE_NAMES[pair_idx_list[k][1]]],
                              "ks": float(ks_scores[k])}
                             for k in sel],
                "all_ks_summary": {
                    "min": float(ks_scores.min()),
                    "median": float(np.median(ks_scores)),
                    "max": float(ks_scores.max()),
                    "p90": float(np.percentile(ks_scores, 90)),
                },
            }
            with open(os.path.join(OUT_DIR, "pairwise_selection.json"), "w") as f:
                json.dump(sel_info, f, indent=2)

        pw_tr_sel = pw_tr[:, sel]
        pw_va_sel = pw_va[:, sel]
        pw_te_sel = pw_te[:, sel]
        for k in sel:
            a, b = pair_idx_list[k]
            extra_feat_names.append(f"pwimb_{PRICE_NAMES[a]}_{PRICE_NAMES[b]}")

        # Globals (train-set mp_t stats, applied row-wise)
        train_mp_for_stats = mp_t_tr  # use train-only (excludes val)
        train_stats = {
            "mean": float(np.mean(train_mp_for_stats)),
            "std": float(np.std(train_mp_for_stats)),
            "p10": float(np.percentile(train_mp_for_stats, 10)),
            "p25": float(np.percentile(train_mp_for_stats, 25)),
            "p50": float(np.percentile(train_mp_for_stats, 50)),
            "p75": float(np.percentile(train_mp_for_stats, 75)),
            "p90": float(np.percentile(train_mp_for_stats, 90)),
        }
        print(f"  train mp stats: {train_stats}", flush=True)
        gb_tr = build_globals_rowwise(mp_t_tr.astype(np.float32), train_stats)
        gb_va = build_globals_rowwise(mp_t_va.astype(np.float32), train_stats)
        gb_te = build_globals_rowwise(mp_t_te.astype(np.float32), train_stats)
        for n in ("mpz", "mpDp10", "mpDp25", "mpDp50", "mpDp75", "mpDp90", "mpqbucket"):
            extra_feat_names.append(f"glob_{n}")

        # Save train_stats for reproducibility
        if seed == 42:
            with open(os.path.join(OUT_DIR, "train_globals.json"), "w") as f:
                json.dump(train_stats, f, indent=2)

        # Concat
        X_tr = np.concatenate([X_tr_base, pw_tr_sel, gb_tr], axis=1)
        X_va = np.concatenate([X_va_base, pw_va_sel, gb_va], axis=1)
        X_te = np.concatenate([X_te_base, pw_te_sel, gb_te], axis=1)
        del pw_tr, pw_va, pw_te
    else:
        X_tr = X_tr_base
        X_va = X_va_base
        X_te = X_te_base

    feat_names = base_feat_names + extra_feat_names
    feat_dim = len(feat_names)
    assert X_tr.shape[1] == feat_dim
    print(f"  feat_dim={feat_dim} (base={base_dim} + extras={len(extra_feat_names)})", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"R-T75L2-{variant}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project="liangwenbei",
                       entity="cjxh21-Tsinghua University",
                       name=run_name,
                       config={"task": "R_T75L2_pairwise",
                               "model": "lightgbm-gpu",
                               "objective": "regression_l2",
                               "variant": variant,
                               "seed": seed,
                               "n_features": feat_dim,
                               "base_dim": base_dim,
                               "extras": len(extra_feat_names),
                               "horizon": H},
                       tags=["R-T75L2", "pairwise", variant, f"seed{seed}"])
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, variant=variant)

    seed_rng = np.random.default_rng(seed * 7919 + 1)

    # aug_a (full ratio)
    n_aug = len(X_tr)
    if n_aug > 0:
        X_aug = aug_a_scale(X_tr, seed_rng, lo=0.80, hi=1.20)
        y_cls_aug = y_cls_tr.copy()
        y_regr_aug = y_regr_tr.copy()
        X_tr_full = np.concatenate([X_tr, X_aug], axis=0)
        y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_aug], axis=0)
        y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_aug], axis=0)
        del X_aug
    else:
        X_tr_full = X_tr
        y_cls_tr_full = y_cls_tr
        y_regr_tr_full = y_regr_tr

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
    print(f"  n_train_used={len(X_tr_full):,} (with aug)", flush=True)

    dtrain = lgb.Dataset(X_tr_full, label=y_regr_tr_full, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_regr_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "regression_l2",
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
        "metric": "l2",
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

    model_path = os.path.join(OUT_DIR, f"model_{tag}_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Eval val
    va_pred = np.concatenate([booster.predict(X_va[s:s+200_000]).astype(np.float32)
                              for s in range(0, len(X_va), 200_000)])
    va_mse = float(((va_pred - y_regr_va) ** 2).mean())
    va_mae = float(np.abs(va_pred - y_regr_va).mean())
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    print(f"  VAL: mse={va_mse:.7f} mae={va_mae:.7f} corr={va_corr:.4f}", flush=True)

    # Eval test
    te_pred = np.concatenate([booster.predict(X_te[s:s+200_000]).astype(np.float32)
                              for s in range(0, len(X_te), 200_000)])
    te_mse = float(((te_pred - y_regr_te) ** 2).mean())
    te_mae = float(np.abs(te_pred - y_regr_te).mean())
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    print(f"  TEST: mse={te_mse:.7f} mae={te_mae:.7f} corr={te_corr:.4f}", flush=True)

    # Save predictions
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

    # k=1 quick sanity
    fee_thr = 2.0 * FEE
    pred_action = np.full(len(te_pred), 1, dtype=np.int8)
    pred_action[te_pred > fee_thr] = 2
    pred_action[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pred_action, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pred_action != 1).sum())
    print(f"  TEST k=1 cum_pnl={cum_pnl:+.4f} n_active={n_active:,}", flush=True)

    summary = {
        "task": f"R_T75L2_pairwise variant={variant} seed={seed}",
        "variant": variant, "seed": seed, "horizon": H,
        "objective": "regression_l2",
        "n_features": feat_dim, "base_dim": base_dim,
        "n_extras": len(extra_feat_names),
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "k1_cum_pnl": cum_pnl, "k1_n_active": n_active},
    }
    out_path = os.path.join(OUT_DIR, f"summary_{tag}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {out_path}", flush=True)

    if use_wandb:
        wandb.log({
            "best_iter": best_iter,
            "train_time_sec": float(train_time),
            "val_mse": va_mse, "val_mae": va_mae, "val_corr": va_corr,
            "test_mse": te_mse, "test_mae": te_mae, "test_corr": te_corr,
            "test_k1_cum_pnl": cum_pnl,
            "test_k1_n_active": n_active,
        })
        wandb.finish()
    progress("done", seed=seed, variant=variant,
             best_iter=best_iter, val_mse=va_mse, test_corr=te_corr,
             test_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
