"""T117: LightGBM Huber + monotone_constraints on theoretically monotone microstructure features.

Replicates T99 huber alpha=1e-3 setup; adds monotone_constraints on 18 features
where Cont 2014 / standard market-microstructure theory predicts a known signed
relationship to Δmid:
  +1 monotone (price-rising):  imbalance, wmp_balance_12, lb_intst, mb_intst,
                                ca_intst, ewma_ofi_*_lvl1, signed_rv_*, signed_bv_*
  -1 monotone (price-falling): la_intst, ma_intst, cb_intst

Set --no-monotone to recover the T99 baseline (sanity).
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

# Monotone constraint dictionary: feature name -> +1/-1
MONOTONE_FEATS = {
    # Order-book imbalance / weighted mid: bid surplus -> up
    "imbalance": 1,
    "wmp_balance_12": 1,
    # Order intensities: limit-buy / market-buy push price up; limit-ask / market-ask push down
    "lb_intst": 1,
    "la_intst": -1,
    "mb_intst": 1,
    "ma_intst": -1,
    # Cancel intensities: cancel-buy weakens support (down); cancel-ask removes resistance (up)
    "cb_intst": -1,
    "ca_intst": 1,
    # OFI (Cont 2014): positive OFI -> Delta mid up
    "ewma_ofi_a0.05_lvl1": 1,
    "ewma_ofi_a0.1_lvl1": 1,
    "ewma_ofi_a0.3_lvl1": 1,
    "ewma_ofi_a0.5_lvl1": 1,
    # Signed realized vol / bipower variation: sign indicates trend direction
    "signed_rv_W20": 1,
    "signed_rv_W50": 1,
    "signed_rv_W100": 1,
    "signed_bv_W20": 1,
    "signed_bv_W50": 1,
    "signed_bv_W100": 1,
}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=1e-3,
                    help="Huber transition threshold.")
    ap.add_argument("--monotone-method", default="advanced",
                    choices=["basic", "intermediate", "advanced"])
    ap.add_argument("--monotone-penalty", type=float, default=0.0)
    ap.add_argument("--no-monotone", action="store_true",
                    help="Disable monotone constraints (recover T99 baseline).")
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
    use_monotone = not args.no_monotone
    base_tag = f"huber_a{args.alpha:g}"
    if args.tag:
        tag = args.tag
    else:
        tag = base_tag + ("_mono" if use_monotone else "_nomono")
    print(f"=== T117 LGB seed={seed} h={H} tag={tag} monotone={use_monotone} ===", flush=True)

    progress("loading_caches", seed=seed, tag=tag)
    t0 = time.time()
    train_full = load_split("train")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeP_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f if line.strip()]
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

    # Build monotone_constraints aligned with kept features
    mc = [0] * feat_dim
    n_constrained = 0
    constrained_list = []
    for fn, sign in MONOTONE_FEATS.items():
        if fn in feat_names:
            i = feat_names.index(fn)
            mc[i] = sign
            n_constrained += 1
            constrained_list.append((fn, sign, i))
        else:
            print(f"  WARN: monotone feat {fn} not in kept features", flush=True)
    print(f"  monotone-constrained features: {n_constrained}", flush=True)
    for fn, sign, i in constrained_list:
        print(f"    {fn:30s} idx={i:3d}  sign={sign:+d}", flush=True)

    (X_tr, y_cls_tr, y_regr_tr,
     X_va, y_cls_va, y_regr_va, mp_t_va, mp_th_va, info) = build_v4_split(train_full, slicer)
    print(f"  feat_dim={feat_dim}, {info}", flush=True)
    print(f"  y_regr_tr stats: mean={y_regr_tr.mean():.6e} std={y_regr_tr.std():.6e}", flush=True)

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
            run_name = (f"T117-{tag}-seed{seed}-"
                        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "task": "T117_lgb_monotone",
                    "objective": "huber",
                    "alpha": args.alpha,
                    "monotone": use_monotone,
                    "monotone_method": args.monotone_method,
                    "monotone_penalty": args.monotone_penalty,
                    "n_constrained": n_constrained,
                    "split_info": info,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seed": seed,
                    **{k: v for k, v in vars(args).items()},
                },
                tags=["T117", "lgb", "monotone" if use_monotone else "baseline", f"h{H}"],
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

    params = {
        "objective": "huber",
        "alpha": args.alpha,
        "metric": "l1",
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

    if use_monotone:
        params["monotone_constraints"] = mc
        params["monotone_constraints_method"] = args.monotone_method
        params["monotone_penalty"] = float(args.monotone_penalty)

    if args.use_gpu:
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

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

    model_path = os.path.join(HERE, f"model_T117_{tag}_seed{seed}.txt")
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
    pred_path = os.path.join(HERE, f"pred_T117_{tag}_seed{seed}.parquet")
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

    # Feature importance: how often were monotone-constrained features used?
    fi_split = booster.feature_importance(importance_type="split")
    fi_gain = booster.feature_importance(importance_type="gain")
    mono_split = sum(int(fi_split[feat_names.index(fn)]) for fn in MONOTONE_FEATS if fn in feat_names)
    mono_gain = float(sum(fi_gain[feat_names.index(fn)] for fn in MONOTONE_FEATS if fn in feat_names))
    total_split = int(fi_split.sum())
    total_gain = float(fi_gain.sum())
    print(f"  monotone feat usage: split={mono_split}/{total_split} ({100*mono_split/max(total_split,1):.2f}%)  "
          f"gain={mono_gain:.4e}/{total_gain:.4e} ({100*mono_gain/max(total_gain,1e-12):.2f}%)", flush=True)

    summary = {
        "task": f"T117 LGB monotone={use_monotone} seed={seed}",
        "seed": seed,
        "horizon": H,
        "objective": "huber",
        "alpha": args.alpha,
        "monotone": use_monotone,
        "monotone_method": args.monotone_method,
        "monotone_penalty": args.monotone_penalty,
        "n_constrained": n_constrained,
        "constrained_list": [{"name": n, "sign": s} for n, s, _ in constrained_list],
        "tag": tag,
        "n_features": feat_dim,
        "best_iter": best_iter,
        "train_time_sec": float(train_time),
        "val": {"mse": va_mse, "mae": va_mae, "corr": va_corr},
        "test": {"mse": te_mse, "mae": te_mae, "corr": te_corr,
                 "ev_gate_k1_cum_pnl": cum_pnl, "ev_gate_k1_n_active": n_active},
        "feat_importance": {
            "monotone_split": mono_split,
            "monotone_gain": mono_gain,
            "total_split": total_split,
            "total_gain": total_gain,
            "monotone_split_pct": 100*mono_split/max(total_split, 1),
            "monotone_gain_pct": 100*mono_gain/max(total_gain, 1e-12),
        },
        "params": vars(args),
    }
    out_path = os.path.join(HERE, f"summary_T117_{tag}_seed{seed}.json")
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
            "monotone_split_pct": 100*mono_split/max(total_split, 1),
            "monotone_gain_pct": 100*mono_gain/max(total_gain, 1e-12),
        })
        wandb.finish()
    progress("done", seed=seed, tag=tag,
             best_iter=best_iter, val_mse=va_mse,
             test_corr=te_corr, test_ev_gate_k1_cum_pnl=cum_pnl)


if __name__ == "__main__":
    main()
