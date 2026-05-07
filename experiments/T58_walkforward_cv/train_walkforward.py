"""T58 Walk-forward CV using iter_008 setup (R34 stage3 schemeN, aug_a, GPU LightGBM, h60).

Combines train (date 0-79) + val (date 80-95) caches → contiguous dataset with date 0-95.
Test cache (date 96-119) reserved as platform-like held-out.

Schemes:
  W1: fixed-window walk-forward, train=32d, val=8d, test=8d, slide=8d
       7 folds: F0 train=0-31 val=32-39 test=40-47 ... F6 train=48-79 val=80-87 test=88-95
  W2: expanding-window walk-forward, val=last 8d of train_full, test=8d, slide=8d
       8 folds: F0 train_full=0-31 (val=24-31) test=32-39 ... F7 train_full=0-87 (val=80-87) test=88-95
  W3: expanding-window large-block test
       3 folds:
       F0 train_full=0-39 (val=32-39) test=40-87
       F1 train_full=0-55 (val=48-55) test=56-87
       F2 train_full=0-71 (val=64-71) test=72-87

For each fold:
  - Use ALL 5 syms (no LOSO).
  - Apply aug_a [0.80, 1.20] to the train portion only (val/test untouched).
  - GPU LightGBM, multiclass, h60.
  - SeedConfig same as T53.
  - Compute per-fold cum_pnl using src.eval.pnl._per_horizon_metrics.

Aggregation:
  - cum_pnl_sum = sum of per-fold cum_pnl
  - sym_days_total = total tested sym-days
  - normalized_cum_pnl = cum_pnl_sum * (120 / sym_days_total)  [LOSO 5-seed comparable: 120 sym-days]
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
from build_aug import build_train_for_variant, class_balanced_weight  # noqa: E402

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
CACHE_DIR = os.path.join(T53_DIR, "cache")
SYM_REPORT_PATH = os.path.join(T53_DIR, "sym_invariance_report.json")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)

# Reuse exact T53 seed configs to keep iter_008 recipe identical
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


def load_combined() -> dict:
    """Load train + val + test caches and concat them.

    Returns dict with concatenated arrays + 'date_split' marker (0=train,1=val,2=test).
    Test (date 96-119) is included only for sym-equivalence sanity but NOT used in CV here.
    We return only train+val combined (date 0-95).
    """
    parts = {}
    for split in ("train", "val"):
        p = os.path.join(CACHE_DIR, f"schemeN_{split}.npz")
        d = np.load(p)
        parts[split] = {k: d[k] for k in d.files}

    combined = {}
    keys = parts["train"].keys()
    for k in keys:
        combined[k] = np.concatenate([parts["train"][k], parts["val"][k]], axis=0)
    return combined


def get_keep_idx(total_dim: int, base_dim: int, n_extras: int,
                 extra_names: list[str], n_drop_tail: int = 3) -> tuple[np.ndarray, list[str]]:
    """Recreate the slicer used in T53 LOSO: drop tail (time-encoding) + sym_invariance FAIL extras."""
    with open(SYM_REPORT_PATH) as f:
        report = json.load(f)
    fail_names = report["summary"]["fail_names"]
    fail_local = [extra_names.index(n) for n in fail_names]
    fail_global = [base_dim + i for i in fail_local]
    drop_set = set(fail_global)
    for i in range(n_drop_tail):
        drop_set.add(base_dim - 1 - i)
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop_set],
        dtype=np.int64,
    )
    feat_names_path = os.path.join(CACHE_DIR, "schemeN_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    feat_names = [all_feat_names[i] for i in keep_idx]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN feature leak: {leak}"
    return keep_idx, feat_names


def make_folds(scheme: str) -> list[dict]:
    """Return list of fold specs with date ranges (inclusive)."""
    folds: list[dict] = []
    if scheme == "W1":
        # train=32d val=8d test=8d, slide 8d, dates 0-95
        # Fold 0: train 0-31, val 32-39, test 40-47
        # Fold k: train k*8 .. k*8+31, val k*8+32 .. k*8+39, test k*8+40 .. k*8+47
        for k in range(7):
            tr_lo, tr_hi = k * 8, k * 8 + 31
            va_lo, va_hi = k * 8 + 32, k * 8 + 39
            te_lo, te_hi = k * 8 + 40, k * 8 + 47
            if te_hi > 95:
                break
            folds.append({
                "fold": k,
                "train_dates": (tr_lo, tr_hi),
                "val_dates":   (va_lo, va_hi),
                "test_dates":  (te_lo, te_hi),
            })
    elif scheme == "W2":
        # expanding train to k*8+31, val=last 8d of train, test=8d
        # Fold 0: train_full 0-31, val=24-31, train=0-23, test=32-39
        # Fold k: train_full 0..k*8+31, val=k*8+24 .. k*8+31, train=0 .. k*8+23, test=k*8+32 .. k*8+39
        for k in range(8):
            tf_lo, tf_hi = 0, k * 8 + 31
            va_lo, va_hi = k * 8 + 24, k * 8 + 31
            tr_lo, tr_hi = 0, va_lo - 1
            te_lo, te_hi = k * 8 + 32, k * 8 + 39
            if te_hi > 95:
                break
            folds.append({
                "fold": k,
                "train_dates": (tr_lo, tr_hi),
                "val_dates":   (va_lo, va_hi),
                "test_dates":  (te_lo, te_hi),
            })
    elif scheme == "W3":
        # 3 folds, expanding train, large test blocks ending at 87
        # F0: train_full=0-39 (val=32-39, train=0-31), test=40-87
        # F1: train_full=0-55 (val=48-55, train=0-47), test=56-87
        # F2: train_full=0-71 (val=64-71, train=0-63), test=72-87
        specs = [
            (0, 31, 32, 39, 40, 87),
            (0, 47, 48, 55, 56, 87),
            (0, 63, 64, 71, 72, 87),
        ]
        for k, (tr_lo, tr_hi, va_lo, va_hi, te_lo, te_hi) in enumerate(specs):
            folds.append({
                "fold": k,
                "train_dates": (tr_lo, tr_hi),
                "val_dates":   (va_lo, va_hi),
                "test_dates":  (te_lo, te_hi),
            })
    else:
        raise ValueError(scheme)
    return folds


def run_fold(fold_spec: dict, scheme: str, seed: int,
             X_full: np.ndarray, y_full: np.ndarray,
             date_full: np.ndarray, sym_full: np.ndarray,
             mp_t_full: np.ndarray, mp_th_full: np.ndarray,
             feat_names: list[str], args, use_wandb: bool, wandb=None) -> dict:
    cfg = SEED_CONFIGS[seed]
    tr_lo, tr_hi = fold_spec["train_dates"]
    va_lo, va_hi = fold_spec["val_dates"]
    te_lo, te_hi = fold_spec["test_dates"]
    tag = f"{scheme}_seed{seed}_fold{fold_spec['fold']}_tr{tr_lo}-{tr_hi}_va{va_lo}-{va_hi}_te{te_lo}-{te_hi}"
    print(f"\n  --- {tag} ---", flush=True)

    m_tr = (date_full >= tr_lo) & (date_full <= tr_hi)
    m_va = (date_full >= va_lo) & (date_full <= va_hi)
    m_te = (date_full >= te_lo) & (date_full <= te_hi)

    X_tr_orig = X_full[m_tr]
    y_tr_orig = y_full[m_tr].astype(np.int64)
    X_va = X_full[m_va]
    y_va = y_full[m_va].astype(np.int64)
    X_te = X_full[m_te]
    y_te = y_full[m_te].astype(np.int64)

    mp_t_te = mp_t_full[m_te]
    mp_th_te = mp_th_full[m_te]

    n_tr_days = tr_hi - tr_lo + 1
    n_va_days = va_hi - va_lo + 1
    n_te_days = te_hi - te_lo + 1
    print(f"    n_train={len(X_tr_orig):,} ({n_tr_days}d)  n_val={len(X_va):,} ({n_va_days}d)  "
          f"n_test={len(X_te):,} ({n_te_days}d)", flush=True)

    # aug_a (matches iter_008): [0.80, 1.20] per (sample,feature)
    seed_rng = np.random.default_rng(seed * 7919 + 1 + fold_spec["fold"] * 31)
    X_tr, y_tr, sw_tr = build_train_for_variant(
        args.variant, X_tr_orig, y_tr_orig,
        None, None,  # aug_a doesn't need feat_std/feat_mean
        seed_rng,
        aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)
    print(f"    n_train_used={len(X_tr):,}", flush=True)

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                        feature_name=feat_names, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                      feature_name=feat_names, reference=dtrain, free_raw_data=False)

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

    t0 = time.time()
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
    train_time = time.time() - t0
    print(f"    trained {train_time:.1f}s, best_iter={booster.best_iteration}", flush=True)

    # Predict test
    te_prob = booster.predict(X_te).astype(np.float32)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)

    metrics = _per_horizon_metrics(te_pred, y_te, mp_t_te, mp_th_te, fee_rate=0.0001)
    metrics["pred_distribution"] = dict(metrics["pred_distribution"])
    metrics["label_distribution"] = dict(metrics["label_distribution"])
    print(f"    cum_pnl={metrics['cum_pnl']:+.4f}  acc={metrics['accuracy']:.4f}  "
          f"f0_5={metrics['f0_5_macro']:.4f}  n_active={metrics['n_predictions_active']}", flush=True)

    if use_wandb and wandb is not None:
        wandb.log({
            f"{scheme}/seed{seed}/fold{fold_spec['fold']}/cum_pnl": metrics["cum_pnl"],
            f"{scheme}/seed{seed}/fold{fold_spec['fold']}/acc": metrics["accuracy"],
            f"{scheme}/seed{seed}/fold{fold_spec['fold']}/f0_5": metrics["f0_5_macro"],
            f"{scheme}/seed{seed}/fold{fold_spec['fold']}/best_iter": booster.best_iteration,
            f"{scheme}/seed{seed}/fold{fold_spec['fold']}/train_time": train_time,
            f"{scheme}/seed{seed}/fold{fold_spec['fold']}/n_test": len(X_te),
        })

    # Optionally save model (skip by default to save disk)
    if args.save_models:
        model_path = os.path.join(HERE, f"model_{tag}.txt")
        booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "fold": fold_spec["fold"],
        "scheme": scheme,
        "seed": seed,
        "train_dates": fold_spec["train_dates"],
        "val_dates": fold_spec["val_dates"],
        "test_dates": fold_spec["test_dates"],
        "n_train": int(len(X_tr_orig)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "n_test_days": n_te_days,
        "n_test_sym_days": int(n_te_days * len(SYMS)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "test": metrics,
    }


def aggregate(scheme: str, fold_results: list[dict]) -> dict:
    cums = [r["test"]["cum_pnl"] for r in fold_results]
    accs = [r["test"]["accuracy"] for r in fold_results]
    sym_days = [r["n_test_sym_days"] for r in fold_results]
    cums_a = np.array(cums)
    sym_days_a = np.array(sym_days)
    cum_pnl_sum = float(cums_a.sum())
    total_sym_days = int(sym_days_a.sum())
    # normalize to LOSO-comparable 120 sym-days (5 syms × 24 days, the platform-like test cache size)
    norm = float(cum_pnl_sum * (120 / total_sym_days)) if total_sym_days > 0 else 0.0
    return {
        "scheme": scheme,
        "n_folds": len(fold_results),
        "cum_pnl_per_fold": cums,
        "acc_per_fold": accs,
        "sym_days_per_fold": sym_days,
        "cum_pnl_sum": cum_pnl_sum,
        "cum_pnl_mean_per_fold": float(cums_a.mean()),
        "total_sym_days": total_sym_days,
        "cum_pnl_per_120_sym_days": norm,
        "n_pos_folds": int((cums_a > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="W1,W2,W3", help="comma-separated subset of W1,W2,W3")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--variant", default="aug_a", choices=["aug_a", "baseline"])
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
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--out-tag", default="pilot_seed42")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T58 walk-forward CV  schemes={schemes}  seeds={seeds}  variant={args.variant} ===", flush=True)

    progress("loading_caches", schemes=schemes, seeds=seeds)
    t_load = time.time()
    combined = load_combined()
    print(f"loaded combined train+val in {time.time()-t_load:.1f}s", flush=True)

    base_dim = 226
    total_dim = combined["X"].shape[1]
    n_extras = total_dim - base_dim
    extras_names_path = os.path.join(CACHE_DIR, "schemeN_extra_feat_names.txt")
    with open(extras_names_path) as f:
        extra_names = [line.strip() for line in f]
    assert len(extra_names) == n_extras

    keep_idx, feat_names = get_keep_idx(total_dim, base_dim, n_extras, extra_names,
                                         n_drop_tail=args.n_drop_tail)
    feat_dim = len(keep_idx)
    print(f"  base_dim={base_dim} n_extras={n_extras} feat_dim={feat_dim}", flush=True)

    X_full = combined["X"][:, keep_idx].astype(np.float32, copy=False)
    y_full = combined[f"y{H}"]
    date_full = combined["date"]
    sym_full = combined["sym"]
    mp_t_full = combined["mp_t"]
    mp_th_full = combined[f"mp_t{H}"]

    print(f"  combined dates {date_full.min()}..{date_full.max()}, syms {np.unique(sym_full)}, "
          f"X.shape={X_full.shape}", flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T58-walkforward-h{H}-{args.out_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "scheme_set": "+".join(schemes),
                    "n_features": feat_dim,
                    "horizon": H,
                    "variant": args.variant,
                    "seeds": seeds,
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T58", "walkforward", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    summary: dict = {
        "task": "T58 walk-forward CV (R34 stage3 schemeN, aug_a, h60)",
        "params": vars(args),
        "horizon": H,
        "feat_dim": feat_dim,
        "schemes": {},
    }

    for scheme in schemes:
        folds = make_folds(scheme)
        print(f"\n{'='*78}\n=== scheme={scheme}  n_folds={len(folds)} ===", flush=True)
        for f in folds:
            print(f"  fold {f['fold']}: train {f['train_dates']}  val {f['val_dates']}  "
                  f"test {f['test_dates']}", flush=True)

        scheme_blob: dict = {"per_seed": {}, "aggregate_by_seed": {}}
        for seed in seeds:
            print(f"\n--- {scheme} seed={seed} ---", flush=True)
            progress(f"running_{scheme}_seed{seed}", scheme=scheme, seed=seed)
            fold_results = []
            for f in folds:
                r = run_fold(f, scheme, seed,
                            X_full, y_full, date_full, sym_full,
                            mp_t_full, mp_th_full,
                            feat_names, args, use_wandb, wandb)
                fold_results.append(r)
            agg = aggregate(scheme, fold_results)
            scheme_blob["per_seed"][seed] = fold_results
            scheme_blob["aggregate_by_seed"][seed] = agg
            print(f"\n  >>> {scheme} seed={seed}: cum_pnl_sum={agg['cum_pnl_sum']:+.4f} "
                  f"normalized_120sd={agg['cum_pnl_per_120_sym_days']:+.4f} "
                  f"n_pos_folds={agg['n_pos_folds']}/{agg['n_folds']} "
                  f"per_fold={[f'{x:+.3f}' for x in agg['cum_pnl_per_fold']]}", flush=True)
            if use_wandb:
                wandb.summary.update({
                    f"{scheme}_seed{seed}/cum_pnl_sum": agg["cum_pnl_sum"],
                    f"{scheme}_seed{seed}/cum_pnl_per_120sd": agg["cum_pnl_per_120_sym_days"],
                    f"{scheme}_seed{seed}/n_pos_folds": agg["n_pos_folds"],
                })

        # multi-seed aggregate per scheme: average normalized cum_pnl across seeds
        agg_seeds = list(scheme_blob["aggregate_by_seed"].values())
        norm_arr = np.array([a["cum_pnl_per_120_sym_days"] for a in agg_seeds])
        sum_arr = np.array([a["cum_pnl_sum"] for a in agg_seeds])
        scheme_blob["multi_seed_summary"] = {
            "n_seeds": len(seeds),
            "seeds": seeds,
            "cum_pnl_per_120_sym_days_per_seed": norm_arr.tolist(),
            "cum_pnl_sum_per_seed": sum_arr.tolist(),
            "mean_normalized": float(norm_arr.mean()),
            "std_normalized": float(norm_arr.std(ddof=0)),
            "mean_sum": float(sum_arr.mean()),
        }
        summary["schemes"][scheme] = scheme_blob

    out_path = os.path.join(HERE, f"results_{args.out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    # quick comparison line
    print("\n=== SUMMARY ===", flush=True)
    for scheme, blob in summary["schemes"].items():
        ms = blob["multi_seed_summary"]
        print(f"  {scheme}: cum_pnl_per_120sd  mean={ms['mean_normalized']:+.4f} "
              f"std={ms['std_normalized']:.4f} (n_seeds={ms['n_seeds']})", flush=True)

    if use_wandb:
        wandb.finish()
    progress("done", schemes=schemes, seeds=seeds)


if __name__ == "__main__":
    main()
