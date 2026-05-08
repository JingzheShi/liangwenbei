"""R_Hawkes_OFI: baseline (schemeP) vs +Hawkes-OFI (schemeP + 6 Hawkes features).

3 seeds × 2 variants = 6 LGB Huber GPU trains. Identical recipe to T117/R3:
LGB Huber alpha=1e-3, 600 rounds, early-stopping=40, lr=0.05, V4 split (date>=76 → val).

Output per train:
  model_<variant>_seed<S>.txt
  pred_<variant>_seed<S>.parquet  (date, sym, sess, t, true, pred, mp_t, mp_th)
  summary_<variant>_seed<S>.json
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

HERE = os.path.dirname(os.path.abspath(__file__))

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
H = 60

SEED_CONFIGS = {
    42: dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:  dict(seed=1,  feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:  dict(seed=7,  feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
}

T59_FAIL = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
            "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
            "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
            "roll_eff_spr_ratio_W100"]
STAGE5_FAIL = ["liq_asym_top5_W5"]
DROP_NAMES = T59_FAIL + STAGE5_FAIL


def class_balanced_weight(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    counts = np.bincount(y.astype(np.int64), minlength=num_class).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    cw = len(y) / (num_class * counts)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_scale(X: np.ndarray, rng: np.random.Generator,
                lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    scales = rng.uniform(lo, hi, size=X.shape).astype(X.dtype)
    return X * scales


def progress(progress_path, step, **extra):
    p = {"status": "running", "step": step,
         "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}
    with open(progress_path, "w") as f:
        json.dump(p, f, indent=2)


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


def build_features(scheme_train, scheme_test, feat_names_path,
                   hawkes_train_path, hawkes_test_path, hawkes_names_path,
                   variant):
    with open(feat_names_path) as f:
        all_names = [line.strip() for line in f if line.strip()]
    total_dim = scheme_train["X"].shape[1]
    assert len(all_names) == total_dim, (len(all_names), total_dim)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    drop_idx = {name_to_idx[n] for n in DROP_NAMES if n in name_to_idx}
    keep_idx = np.array([i for i in range(total_dim) if i not in drop_idx], dtype=np.int64)
    feat_names = [all_names[i] for i in keep_idx]
    assert "date" not in feat_names and "sym" not in feat_names

    sP_tr = scheme_train["X"][:, keep_idx].astype(np.float32, copy=False)
    sP_te = scheme_test["X"][:, keep_idx].astype(np.float32, copy=False)

    if variant == "baseline":
        return sP_tr, sP_te, feat_names

    # +Hawkes-OFI features
    hk_tr = np.load(hawkes_train_path).astype(np.float32, copy=False)
    hk_te = np.load(hawkes_test_path).astype(np.float32, copy=False)
    with open(hawkes_names_path) as f:
        hk_names = [line.strip() for line in f if line.strip()]
    assert hk_tr.shape[0] == sP_tr.shape[0], (hk_tr.shape, sP_tr.shape)
    assert hk_te.shape[0] == sP_te.shape[0], (hk_te.shape, sP_te.shape)
    assert hk_tr.shape[1] == len(hk_names), (hk_tr.shape, len(hk_names))

    X_tr = np.concatenate([sP_tr, hk_tr], axis=1)
    X_te = np.concatenate([sP_te, hk_te], axis=1)
    feat_names = feat_names + hk_names
    return X_tr, X_te, feat_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["baseline", "hawkes"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--scheme-train", required=True)
    ap.add_argument("--scheme-test", required=True)
    ap.add_argument("--scheme-feat-names", required=True)
    ap.add_argument("--hawkes-train", default="")
    ap.add_argument("--hawkes-test", default="")
    ap.add_argument("--hawkes-names", default="")
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--out-dir", default=HERE)
    args = ap.parse_args()

    seed = args.seed
    variant = args.variant
    tag = f"{variant}_seed{seed}"
    print(f"=== R_Hawkes {variant} seed={seed} ===", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    progress_path = os.path.join(args.out_dir, f"worker-progress_{tag}.json")
    progress(progress_path, "loading", variant=variant, seed=seed)

    t0 = time.time()
    scheme_train = {k: np.load(args.scheme_train)[k] for k in np.load(args.scheme_train).files}
    scheme_test = {k: np.load(args.scheme_test)[k] for k in np.load(args.scheme_test).files}
    print(f"  loaded schemeP in {time.time()-t0:.1f}s "
          f"train{scheme_train['X'].shape} test{scheme_test['X'].shape}", flush=True)

    X_tr_full, X_te, feat_names = build_features(
        scheme_train, scheme_test, args.scheme_feat_names,
        args.hawkes_train, args.hawkes_test, args.hawkes_names,
        variant)
    print(f"  feat_dim={X_tr_full.shape[1]} (variant={variant})", flush=True)

    # V4 split
    date_tr = scheme_train["date"]
    m_va = date_tr >= 76
    m_t = ~m_va
    X_tr = X_tr_full[m_t]
    X_va = X_tr_full[m_va]

    y_cls_tr = scheme_train["y60"][m_t].astype(np.int64)
    y_cls_va = scheme_train["y60"][m_va].astype(np.int64)
    y_regr_tr = regr_target(scheme_train["mp_t"][m_t], scheme_train["mp_t60"][m_t])
    y_regr_va = regr_target(scheme_train["mp_t"][m_va], scheme_train["mp_t60"][m_va])
    print(f"  X_tr{X_tr.shape}  X_va{X_va.shape}", flush=True)

    y_regr_te = regr_target(scheme_test["mp_t"], scheme_test["mp_t60"])
    mp_t_te = scheme_test["mp_t"]
    mp_th_te = scheme_test["mp_t60"]

    # WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            run_name = f"R_Hawkes-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project="liangwenbei", entity="cjxh21-Tsinghua University",
                       name=run_name,
                       config={"variant": variant, "seed": seed,
                               "n_features": X_tr.shape[1], "alpha": args.alpha,
                               "horizon": H, "trick": "hawkes_ofi"},
                       tags=["R_Hawkes_OFI", variant])
        except Exception as e:
            print(f"  wandb init failed: {e!r}", flush=True)
            use_wandb = False

    cfg = SEED_CONFIGS[seed]
    print(f"  cfg={cfg}", flush=True)

    rng = np.random.default_rng(seed * 7919 + 1)
    aug_idx = np.arange(len(X_tr))
    X_aug = aug_a_scale(X_tr[aug_idx], rng, lo=0.8, hi=1.2)
    X_tr_full2 = np.concatenate([X_tr, X_aug], axis=0)
    y_cls_tr_full = np.concatenate([y_cls_tr, y_cls_tr], axis=0)
    y_regr_tr_full = np.concatenate([y_regr_tr, y_regr_tr], axis=0)
    del X_aug, X_tr, X_tr_full

    sw_tr = class_balanced_weight(y_cls_tr_full, num_class=NUM_CLASS)
    sw_va = class_balanced_weight(y_cls_va, num_class=NUM_CLASS)

    dtrain = lgb.Dataset(X_tr_full2, label=y_regr_tr_full, weight=sw_tr,
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
        "num_threads": args.num_threads,
        "seed": seed, "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2, "data_random_seed": seed + 3,
        "device": "gpu", "gpu_use_dp": False,
        "verbose": -1,
    }

    progress(progress_path, "training", variant=variant, seed=seed)
    t1 = time.time()
    booster = lgb.train(
        params=params, train_set=dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=True),
                   lgb.log_evaluation(period=50)])
    train_time = time.time() - t1
    best_iter = int(booster.best_iteration)
    print(f"  trained in {train_time:.1f}s, best_iter={best_iter}", flush=True)

    booster.save_model(os.path.join(args.out_dir, f"model_{tag}.txt"),
                       num_iteration=best_iter)

    va_pred = predict_chunked(booster, X_va)
    va_corr = float(np.corrcoef(va_pred, y_regr_va)[0, 1])
    va_l1 = float(np.abs(va_pred - y_regr_va).mean())

    te_pred = predict_chunked(booster, X_te)
    te_corr = float(np.corrcoef(te_pred, y_regr_te)[0, 1])
    te_l1 = float(np.abs(te_pred - y_regr_te).mean())
    print(f"  VAL corr={va_corr:.4f} l1={va_l1:.6e}  TEST corr={te_corr:.4f} l1={te_l1:.6e}",
          flush=True)

    sess_map = {0: "am", 1: "pm"}
    df = pd.DataFrame({
        "sym": scheme_test["sym"].astype(np.int8),
        "date": scheme_test["date"].astype(np.int16),
        "session": np.array([sess_map[int(s)] for s in scheme_test["sess_idx"]],
                            dtype=object),
        "t": scheme_test["t"].astype(np.int16),
        "true_dmid_norm": y_regr_te.astype(np.float32),
        "pred_dmid_norm": te_pred.astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(args.out_dir, f"pred_{tag}.parquet")
    df.to_parquet(pred_path, index=False)
    print(f"  saved {pred_path}", flush=True)

    fee_thr = 2.0 * FEE
    pa = np.full(len(te_pred), 1, dtype=np.int8)
    pa[te_pred > fee_thr] = 2
    pa[te_pred < -fee_thr] = 0
    pnl = vectorized_pnl(pa, mp_t_te, mp_th_te)
    cum_pnl = float(pnl.sum())
    n_active = int((pa != 1).sum())
    print(f"  TEST EV-gate cum_pnl={cum_pnl:+.4f} n_active={n_active:,}", flush=True)

    # Feature importance for Hawkes features
    imp_gain = booster.feature_importance(importance_type="gain")
    imp_split = booster.feature_importance(importance_type="split")
    fi_dict = {}
    for fn, ig, isp in zip(feat_names, imp_gain, imp_split):
        if "hawkes_" in fn:
            fi_dict[fn] = {"gain": float(ig), "split": int(isp)}

    summary = {
        "task": f"R_Hawkes {variant} seed={seed}",
        "variant": variant, "seed": seed,
        "best_iter": best_iter, "train_time_sec": train_time,
        "feat_dim": int(X_te.shape[1]),
        "n_train_aug": int(X_tr_full2.shape[0]),
        "n_val": int(X_va.shape[0]),
        "val_corr": va_corr, "val_l1": va_l1,
        "test_corr": te_corr, "test_l1": te_l1,
        "test_fixed_thr_pnl": cum_pnl,
        "test_n_active": n_active,
        "hawkes_feature_importance": fi_dict,
    }
    with open(os.path.join(args.out_dir, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if use_wandb:
        try:
            import wandb
            wandb.log({**{f"final/{k}": v for k, v in summary.items() if isinstance(v, (int, float))}})
            wandb.finish()
        except Exception:
            pass

    progress(progress_path, "done", variant=variant, seed=seed,
             cum_pnl=cum_pnl)
    print(f"DONE {tag} cum_pnl={cum_pnl:+.4f} val_corr={va_corr:.4f} test_corr={te_corr:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
