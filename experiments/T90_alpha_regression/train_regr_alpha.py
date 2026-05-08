"""T90: Alpha101+Alpha191 regression revival under T75 protocol.

Feature space: schemeP keep (359-d after dropping T59+Stage5 fails)
            ⊕ alpha101 (33) ⊕ alpha191 (124)  = 516-d total.

Target & training otherwise identical to T75:
    y_regr = (mp_t60 - mp_t) / (mp_t + 1)
    V4 walk-forward (train=date 0-75, val=date 76-79, test=date 96-119 5-sym)
    aug_a (per-(sample,feat) scale [0.80, 1.20]) at ratio 1.0
    sample weight = class-balanced on the original 3-class label
    LightGBM regression_l2 GPU, num_boost=600, early_stop=40 on val MSE
    seed configs identical to T75/T64/T70

Inputs:
    - experiments/T68_stage5_features/cache/schemeP_{train,test}.npz   (370-d)
    - experiments/T22_alpha101_alpha191/cache/schemeI_{train,test}.npz (383-d)
    Row alignment between the two caches has been verified externally.

Outputs:
    - model_T90_seed{S}.txt
    - pred_T90_seed{S}.parquet  (same schema as T75 pred_T75_seed{S}.parquet)
    - summary_T90_seed{S}.json
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

P_CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
A_CACHE_DIR = os.path.join(ROOT, "experiments", "T22_alpha101_alpha191", "cache")

# T22 layout: raw_154 [0:154], t3_no_time_69 [154:223],
#             alpha101_33 [223:256], alpha191_124 [256:380], t3_time_3 [380:383]
ALPHA_SLICE = slice(223, 380)   # 157 alphas

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# Identical to T75 / T64 / T70
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


def predict_chunked(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def regr_target(mp_t, mp_th):
    return ((mp_th.astype(np.float64) - mp_t.astype(np.float64))
            / (mp_t.astype(np.float64) + 1.0)).astype(np.float32)


def load_combined(split, p_slicer, a_slicer):
    """Load schemeP[:,p_slicer] and schemeI[:,a_slicer], concat horizontally.

    Asserts alignment by checking sym/date/t arrays match (same row order).
    Returns dict with keys: X, mp_t, sym, date, sess_idx, t, y60, mp_t60.
    """
    p_path = os.path.join(P_CACHE_DIR, f"schemeP_{split}.npz")
    a_path = os.path.join(A_CACHE_DIR, f"schemeI_{split}.npz")
    print(f"  loading {p_path}", flush=True)
    dp = np.load(p_path)
    print(f"  loading {a_path}", flush=True)
    da = np.load(a_path)

    assert np.array_equal(dp["sym"], da["sym"]), f"{split} sym mismatch"
    assert np.array_equal(dp["date"], da["date"]), f"{split} date mismatch"
    assert np.array_equal(dp["t"], da["t"]), f"{split} t mismatch"

    Xp = dp["X"][:, p_slicer].astype(np.float32, copy=True)
    Xa = da["X"][:, a_slicer].astype(np.float32, copy=True)
    X = np.concatenate([Xp, Xa], axis=1)
    print(f"  {split}: schemeP {Xp.shape} ⊕ alphas {Xa.shape} → {X.shape}", flush=True)
    del Xp, Xa

    out = {
        "X": X,
        "mp_t": np.asarray(dp["mp_t"]),
        "sym": np.asarray(dp["sym"]),
        "date": np.asarray(dp["date"]),
        "sess_idx": np.asarray(dp["sess_idx"]),
        "t": np.asarray(dp["t"]),
        "y60": np.asarray(dp["y60"]),
        "mp_t60": np.asarray(dp["mp_t60"]),
    }
    return out


def build_v4_split(train_full):
    date_tr = train_full["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = train_full["X"][m_t]
    y_cls_tr = train_full["y60"][m_t].astype(np.int64)
    y_regr_tr = regr_target(train_full["mp_t"][m_t], train_full["mp_t60"][m_t])

    X_va = train_full["X"][m_va]
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
    return X_tr, y_cls_tr, y_regr_tr, X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info


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
    ap.add_argument("--no-aug", action="store_true",
                    help="disable aug (saves ~2x train RAM; for memory-tight runs)")
    args = ap.parse_args()

    H = args.horizon
    seed = args.seed
    print(f"=== T90 alpha-regr seed={seed} h={H} obj={args.objective} ===", flush=True)

    progress("loading_caches", seed=seed)
    t0 = time.time()

    # Build feature slicers
    feat_names_path = os.path.join(P_CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        p_all_names = [line.strip() for line in f]
    name_to_idx = {n: i for i, n in enumerate(p_all_names)}
    drop_idx = set()
    for n in DROP_NAMES:
        if n in name_to_idx:
            drop_idx.add(name_to_idx[n])
        else:
            print(f"  WARN drop name not found: {n}", flush=True)
    p_keep_idx = np.array(
        [i for i in range(len(p_all_names)) if i not in drop_idx], dtype=np.int64
    )
    p_keep_names = [p_all_names[i] for i in p_keep_idx]
    print(f"  schemeP keep {len(p_keep_idx)}-d (dropped {len(drop_idx)} fails)", flush=True)

    a_feat_names_path = os.path.join(A_CACHE_DIR, "schemeI_feat_names.txt")
    with open(a_feat_names_path) as f:
        a_all_names = [line.strip() for line in f]
    a_keep_idx = np.arange(ALPHA_SLICE.start, ALPHA_SLICE.stop, dtype=np.int64)
    a_keep_names = [a_all_names[i] for i in a_keep_idx]
    print(f"  alphas keep {len(a_keep_idx)}-d (alpha101+alpha191)", flush=True)

    feat_names = p_keep_names + a_keep_names
    feat_dim = len(feat_names)
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, leak

    train_full = load_combined("train", p_keep_idx, a_keep_idx)
    test_full = load_combined("test", p_keep_idx, a_keep_idx)
    print(f"  loaded combined caches in {time.time()-t0:.1f}s", flush=True)

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info) = build_v4_split(train_full)
    print(f"  {info}", flush=True)
    print(f"  feat_dim={feat_dim}", flush=True)
    print(f"  y_regr_tr stats: mean={y_regr_tr.mean():.6f} std={y_regr_tr.std():.6f}",
          flush=True)
    print(f"  y_regr_va stats: mean={y_regr_va.mean():.6f} std={y_regr_va.std():.6f}",
          flush=True)

    X_te = test_full["X"]
    y_cls_te = test_full[f"y{H}"].astype(np.int64) if f"y{H}" in test_full else test_full["y60"].astype(np.int64)
    y_regr_te = regr_target(test_full["mp_t"], test_full["mp_t60"])
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full["mp_t60"]
    sym_te = test_full["sym"]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (f"T90-alpha-regr-seed{seed}-{args.objective}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "task": "T90_alpha_regression",
                    "objective": args.objective,
                    "split_info": info,
                    "n_features": feat_dim,
                    "n_alphas": len(a_keep_idx),
                    "n_schemeP_kept": len(p_keep_idx),
                    "horizon": H,
                    "seed": seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T90", "alpha101", "alpha191", "regression", "EV-gate",
                      f"h{H}", args.objective],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)
    progress("training", seed=seed, split_info=info)

    seed_rng = np.random.default_rng(seed * 7919 + 1)

    # aug_a (default ratio 1.0)
    if args.no_aug or args.aug_ratio <= 0.0:
        X_tr_full = X_tr
        y_regr_tr_full = y_regr_tr
        y_cls_tr_full = y_cls_tr
        print(f"  aug disabled; n_train={len(X_tr_full):,}", flush=True)
    else:
        n_aug = int(round(args.aug_ratio * len(X_tr)))
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
        del X_aug

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)
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

    model_path = os.path.join(HERE, f"model_T90_seed{seed}.txt")
    booster.save_model(model_path, num_iteration=best_iter)

    # Eval val/test
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

    # Save preds parquet (T75 schema)
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
    pred_path = os.path.join(HERE, f"pred_T90_seed{seed}.parquet")
    te_df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

    # Quick raw EV-gate sanity at k=1
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

    # Per-sym cum_pnl at k=1 (sanity, the 5-sym test set ≈ LOSO single-eval)
    per_sym_pnl = []
    for s in SYMS:
        m = sym_te == s
        per_sym_pnl.append(float(pnl[m].sum()))
    sum_per_sym = float(sum(per_sym_pnl))
    print(f"  per-sym cum_pnl k=1: {per_sym_pnl}  sum={sum_per_sym:+.4f}", flush=True)

    # Top-30 alpha gain (for downstream feature-selection variant)
    fi = booster.feature_importance(importance_type="gain")
    fn = booster.feature_name()
    fi_pairs = sorted(zip(fn, fi.tolist()), key=lambda kv: -kv[1])
    top30 = fi_pairs[:30]
    alpha_top10 = [(n, g) for (n, g) in fi_pairs if n.startswith("a101_") or n.startswith("a191_")][:10]
    print(f"  top-10 features by gain:", flush=True)
    for nm, g in fi_pairs[:10]:
        print(f"    {nm:<40s} {g:>14.2f}", flush=True)
    print(f"  top-10 alphas by gain:", flush=True)
    for nm, g in alpha_top10:
        print(f"    {nm:<40s} {g:>14.2f}", flush=True)

    summary = {
        "task": f"T90 alpha-regr seed={seed}",
        "seed": seed,
        "horizon": H,
        "objective": args.objective,
        "split_info": info,
        "n_features": feat_dim,
        "n_alphas": len(a_keep_idx),
        "n_schemeP_kept": len(p_keep_idx),
        "drop_names": list(DROP_NAMES),
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active,
                 "per_sym_pnl_k1": per_sym_pnl, "sum_per_sym_k1": sum_per_sym},
        "top10_features_by_gain": [{"name": nm, "gain": g} for nm, g in fi_pairs[:10]],
        "top10_alphas_by_gain": [{"name": nm, "gain": g} for nm, g in alpha_top10],
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T90_seed{seed}.json")
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
            "test_sum_per_sym_k1": sum_per_sym,
        })
        wandb.finish()
    progress("done", seed=seed,
             best_iter=best_iter, val_mse=va_mse, test_corr=te_corr,
             test_ev_gate_k1_cum_pnl=cum_pnl,
             test_sum_per_sym_k1=sum_per_sym)


if __name__ == "__main__":
    main()
