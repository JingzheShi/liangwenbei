"""T57 LOSO h=60 LightGBM with PnL-aware sample weights.

R33 #10: replace class-balanced sample weight with weight = f(|future_return|)
where future_return = (mp_t60 - mp_t) / (mp_t + 1).

Three schemes:
  A_linear:  w = |ret|
  B_sqrt:    w = |ret|^0.5
  C_capped:  w = min(|ret|, 95th-percentile of |ret| over fold's train set)

A small floor `eps_w` is added so zero-|ret| samples still receive a tiny weight
(avoids LightGBM dropping samples entirely).  Weights are NOT class-rebalanced
on top — the whole point is to let PnL magnitude dominate.

Reuses the T53 R34 Stage 3 schemeN cache (353-d).  Same aug_a [0.80, 1.20],
same drop-tail (drop trailing 3 dims of base 226 → 223 base + 127 extras = 350-d).

Constraint compliance:
  * sample weight depends only on labels (mp_t, mp_t60) — these are available at
    training time, NOT used at predictor inference.
  * sym-agnostic: weight does NOT depend on sym/date/time.
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

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)

SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split: str) -> dict:
    p = os.path.join(CACHE_DIR, f"schemeN_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def evaluate(name, preds, y, mp_t, mp_th, fee_rate=0.0001):
    m = _per_horizon_metrics(preds, y, mp_t, mp_th, fee_rate=fee_rate)
    m["pred_distribution"] = dict(m["pred_distribution"])
    m["label_distribution"] = dict(m["label_distribution"])
    print(f"  --- {name} ---", flush=True)
    for k in ("accuracy", "cum_pnl", "single_pnl", "n_predictions_active", "f0_5_macro"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"    {k:25s}: {v:+.6g}", flush=True)
        else:
            print(f"    {k:25s}: {v}", flush=True)
    return m


def compute_pnl_weight(mp_t: np.ndarray, mp_th: np.ndarray, scheme: str,
                       eps_w: float, q_cap: float = 0.95) -> np.ndarray:
    """Weight from |future return| under one of three schemes."""
    abs_ret = np.abs((mp_th.astype(np.float64) - mp_t.astype(np.float64)) /
                     (mp_t.astype(np.float64) + 1.0))
    if scheme == "A_linear":
        w = abs_ret
    elif scheme == "B_sqrt":
        w = np.sqrt(abs_ret)
    elif scheme == "C_capped":
        cap = float(np.quantile(abs_ret, q_cap))
        w = np.minimum(abs_ret, cap)
    else:
        raise ValueError(f"unknown scheme {scheme}")
    # Floor — avoid weight 0 sample (LightGBM ignores w=0 rows).
    w = np.maximum(w, eps_w).astype(np.float32)
    # Normalize so mean weight = 1.0 (scale-invariant for LightGBM logloss).
    w = w * (1.0 / float(w.mean()))
    return w.astype(np.float32)


def build_aug_with_weight(X_orig: np.ndarray, y_orig: np.ndarray, w_orig: np.ndarray,
                          rng: np.random.Generator, aug_ratio: float = 1.0):
    """Mirror T26 aug_a but propagate the user-supplied weight (no class rebalance)."""
    n_aug = int(round(aug_ratio * len(X_orig)))
    if n_aug <= 0:
        return X_orig, y_orig.astype(np.int64), w_orig.astype(np.float32)
    if n_aug == len(X_orig):
        idx = np.arange(len(X_orig))
    else:
        idx = rng.integers(0, len(X_orig), size=n_aug)
    X_src = X_orig[idx]
    y_src = y_orig[idx]
    w_src = w_orig[idx]
    X_aug = aug_a_scale(X_src, rng)
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_src], axis=0).astype(np.int64)
    w_merged = np.concatenate([w_orig, w_src], axis=0).astype(np.float32)
    return X_merged, y_merged, w_merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--scheme", default="B_sqrt",
                    choices=["A_linear", "B_sqrt", "C_capped"])
    ap.add_argument("--eps-w", type=float, default=1e-5,
                    help="floor on |ret| weight to avoid zero-weight samples")
    ap.add_argument("--q-cap", type=float, default=0.95,
                    help="quantile cap for scheme C_capped")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--n-drop-tail", type=int, default=3)
    ap.add_argument("--no-drop-tail", action="store_true")
    ap.add_argument("--drop-fail-extras", action="store_true")
    ap.add_argument("--out-tag", default=None,
                    help="defaults to scheme name")
    ap.add_argument("--pred-prefix", default="loso_pred_h60")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    if args.out_tag is None:
        args.out_tag = args.scheme

    H = args.horizon
    seeds = [int(x) for x in args.seeds.split(",")]
    syms = [int(x) for x in args.syms.split(",")]
    print(f"=== T57 PnL-aware LOSO h={H} scheme={args.scheme} seeds={seeds} "
          f"syms={syms} ===", flush=True)

    progress("loading_caches", scheme=args.scheme, seeds=seeds, syms=syms)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    base_dim = 226
    total_dim = train_full["X"].shape[1]
    n_extras = total_dim - base_dim
    extras_names_path = os.path.join(CACHE_DIR, "schemeN_extra_feat_names.txt")
    with open(extras_names_path) as f:
        extra_names = [line.strip() for line in f]
    assert len(extra_names) == n_extras, f"{len(extra_names)} vs {n_extras}"

    if args.no_drop_tail:
        slicer = slice(0, total_dim)
        feat_dim = total_dim
    else:
        keep_idx = np.concatenate([
            np.arange(0, base_dim - args.n_drop_tail),
            np.arange(base_dim, total_dim),
        ])
        slicer = keep_idx
        feat_dim = len(keep_idx)
    print(f"  base_dim={base_dim} n_extras={n_extras} feat_dim={feat_dim}", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeN_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    if isinstance(slicer, slice):
        feat_names = all_feat_names[slicer]
    else:
        feat_names = [all_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN feature leak: {leak}"
    assert len(feat_names) == feat_dim
    print(f"  no-leak check OK (feat_dim={feat_dim})", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T57-PnLW-h{H}-{args.scheme}-{args.out_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "scheme": args.scheme,
                    "n_features": feat_dim,
                    "horizon": H,
                    "seeds": seeds,
                    "syms": syms,
                    **{k: v for k, v in vars(args).items() if k not in ("no_wandb",)},
                },
                tags=["T57", "PnL-aware-w", "LOSO", f"h{H}", args.scheme],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    if isinstance(slicer, slice):
        X_tr_full = train_full["X"][:, slicer].astype(np.float32, copy=False)
        X_va_full = val_full["X"][:, slicer].astype(np.float32, copy=False)
        X_te_full = test_full["X"][:, slicer].astype(np.float32, copy=False)
    else:
        X_tr_full = train_full["X"][:, slicer].astype(np.float32)
        X_va_full = val_full["X"][:, slicer].astype(np.float32)
        X_te_full = test_full["X"][:, slicer].astype(np.float32)

    y_tr_full = train_full[f"y{H}"]
    y_va_full = val_full[f"y{H}"]
    y_te_full = test_full[f"y{H}"]

    sym_tr = train_full["sym"]
    sym_va = val_full["sym"]
    sym_te = test_full["sym"]

    mp_t_tr = train_full["mp_t"]
    mp_th_tr = train_full[f"mp_t{H}"]
    mp_t_va = val_full["mp_t"]
    mp_th_va = val_full[f"mp_t{H}"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    all_results = {}
    for seed in seeds:
        all_results[seed] = {}
        cfg = SEED_CONFIGS[seed]
        print(f"\n{'='*78}\n=== seed={seed} cfg={cfg} ===\n{'='*78}", flush=True)
        for held in syms:
            tag = f"h{H}_seed{seed}_held{held}"
            print(f"\n  --- fold {tag} ---", flush=True)

            m_tr = sym_tr != held
            m_va = sym_va != held
            m_te = sym_te == held

            X_tr_orig = X_tr_full[m_tr]
            y_tr_orig = y_tr_full[m_tr].astype(np.int64)
            mp_t_tr_k = mp_t_tr[m_tr]
            mp_th_tr_k = mp_th_tr[m_tr]
            X_va = X_va_full[m_va]
            y_va = y_va_full[m_va].astype(np.int64)
            X_te = X_te_full[m_te]
            y_te = y_te_full[m_te].astype(np.int64)
            mp_t_te_k = mp_t_te[m_te]
            mp_th_te_k = mp_th_te[m_te]
            mp_t_va_k = mp_t_va[m_va]
            mp_th_va_k = mp_th_va[m_va]

            print(f"    n_train_orig={len(X_tr_orig):,} n_val={len(X_va):,} "
                  f"n_test={len(X_te):,}", flush=True)

            w_tr_orig = compute_pnl_weight(mp_t_tr_k, mp_th_tr_k, args.scheme,
                                           eps_w=args.eps_w, q_cap=args.q_cap)
            w_va = compute_pnl_weight(mp_t_va_k, mp_th_va_k, args.scheme,
                                      eps_w=args.eps_w, q_cap=args.q_cap)
            print(f"    weight stats: mean={w_tr_orig.mean():.4f} "
                  f"min={w_tr_orig.min():.4e} max={w_tr_orig.max():.4f} "
                  f"p99={np.quantile(w_tr_orig, 0.99):.4f}", flush=True)

            seed_rng = np.random.default_rng(seed * 7919 + 1 + held * 31)
            X_tr, y_tr, sw_tr = build_aug_with_weight(
                X_tr_orig, y_tr_orig, w_tr_orig,
                seed_rng, aug_ratio=args.aug_ratio,
            )
            print(f"    n_train_used={len(X_tr):,}", flush=True)

            dtrain = lgb.Dataset(
                X_tr, label=y_tr, weight=sw_tr,
                feature_name=feat_names, free_raw_data=False,
            )
            dval = lgb.Dataset(
                X_va, label=y_va, weight=w_va,
                feature_name=feat_names, reference=dtrain, free_raw_data=False,
            )

            params = {
                "objective": "multiclass",
                "num_class": NUM_CLASS,
                "metric": "multi_logloss",
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
            if args.use_gpu:
                params["device"] = "gpu"
                params["gpu_use_dp"] = False

            t_start = time.time()
            booster = lgb.train(
                params, dtrain,
                num_boost_round=args.num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
                    lgb.log_evaluation(period=200),
                ],
            )
            train_time = time.time() - t_start
            print(f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
                  flush=True)

            te_prob = predict_proba(booster, X_te)
            te_pred = te_prob.argmax(axis=1).astype(np.int8)
            te_metrics = evaluate(
                f"HELD-OUT TEST seed={seed} held={held} h={H}",
                te_pred, y_te, mp_t_te_k, mp_th_te_k,
            )

            sess_map = {0: "am", 1: "pm"}
            te_df = pd.DataFrame({
                "sym": test_full["sym"][m_te].astype(np.int8),
                "date": test_full["date"][m_te].astype(np.int16),
                "session": np.array(
                    [sess_map[int(s)] for s in test_full["sess_idx"][m_te]],
                    dtype=object,
                ),
                "t": test_full["t"][m_te].astype(np.int16),
                "true_label": y_te.astype(np.int8),
                "pred_label": te_pred.astype(np.int8),
                "prob_0": te_prob[:, 0].astype(np.float32),
                "prob_1": te_prob[:, 1].astype(np.float32),
                "prob_2": te_prob[:, 2].astype(np.float32),
                "midprice_t": mp_t_te_k.astype(np.float32),
                "midprice_th": mp_th_te_k.astype(np.float32),
            })
            pred_path = os.path.join(HERE, f"{args.pred_prefix}_seed{seed}_held{held}.parquet")
            te_df.to_parquet(pred_path, index=False)

            model_path = os.path.join(HERE, f"loso_model_{tag}.txt")
            booster.save_model(model_path, num_iteration=booster.best_iteration)

            all_results[seed][held] = {
                "horizon": H,
                "held": held,
                "seed": seed,
                "scheme": args.scheme,
                "n_train": int(len(X_tr_orig)),
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te)),
                "best_iter": int(booster.best_iteration),
                "train_time_sec": float(train_time),
                "test": te_metrics,
            }
            if use_wandb:
                wandb.log({
                    f"seed{seed}/fold{held}/best_iter": int(booster.best_iteration),
                    f"seed{seed}/fold{held}/train_time_sec": float(train_time),
                    f"seed{seed}/fold{held}/test_cum_pnl": te_metrics["cum_pnl"],
                    f"seed{seed}/fold{held}/test_single_pnl": te_metrics["single_pnl"],
                    f"seed{seed}/fold{held}/test_acc": te_metrics["accuracy"],
                    f"seed{seed}/fold{held}/test_f0_5": te_metrics["f0_5_macro"],
                })

    print(f"\n{'='*78}\n=== Aggregate ===\n{'='*78}", flush=True)
    agg = {}
    for seed in seeds:
        cums = []
        accs = []
        for held in sorted(all_results[seed].keys()):
            r = all_results[seed][held]
            cums.append(r["test"]["cum_pnl"])
            accs.append(r["test"]["accuracy"])
        cums_a = np.array(cums)
        accs_a = np.array(accs)
        agg[seed] = {
            "horizon": H,
            "seed": seed,
            "scheme": args.scheme,
            "n_folds": len(cums_a),
            "cum_pnl_per_fold": cums_a.tolist(),
            "acc_per_fold": accs_a.tolist(),
            "cum_pnl_sum": float(cums_a.sum()),
            "cum_pnl_mean": float(cums_a.mean()),
            "n_pos_folds": int((cums_a > 0).sum()),
        }
        print(
            f"seed={seed}: cum_pnl_sum={cums_a.sum():+.4f}  "
            f"per-fold={[f'{x:+.3f}' for x in cums_a]}  "
            f"acc={[f'{a:.3f}' for a in accs_a]}",
            flush=True,
        )
        if use_wandb:
            wandb.summary.update({
                f"agg_seed{seed}/cum_pnl_sum": float(cums_a.sum()),
                f"agg_seed{seed}/cum_pnl_mean": float(cums_a.mean()),
                f"agg_seed{seed}/fold2_acc": float(accs_a[2]) if len(accs_a) > 2 else 0.0,
            })

    summary = {
        "task": f"T57 PnL-aware sample weight LOSO ({args.scheme})",
        "params": vars(args),
        "horizon": H,
        "scheme": args.scheme,
        "fold_results": {f"seed{s}": all_results[s] for s in seeds},
        "aggregate": {f"seed{s}": agg[s] for s in seeds},
    }
    out_path = os.path.join(HERE, f"loso_summary_h{H}_{args.out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.finish()
    progress("loso_done", scheme=args.scheme, seeds=seeds)


if __name__ == "__main__":
    main()
