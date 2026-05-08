"""T66 weighted training: time-decay sample-weight + optional adversarial reweighting.

Loads schemeN cache (R34 Stage 3, 340-d), trains LightGBM full-sym multiclass
with per-row sample weight = class_balanced_weight * decay_w * adv_w.

decay_w(i) = exp(-alpha * (latest_date - sample_date_i))
  alpha=0 reproduces T59 baseline.

adv_w(i) = clip(p_train(i) / p_train.mean(), w_lo, w_hi)
  Uses adv_prob_train.npy from Step 1; off by default.

Outputs (per seed):
  weighted_model_h{H}_seed{S}_a{tag}.txt
  weighted_pred_h{H}_seed{S}_a{tag}.parquet  (442k rows)
  weighted_summary_h{H}_a{tag}.json          (aggregate over seeds)

Constraints:
  * date is NOT a feature (only used to compute weight at training time)
  * sym is NOT a feature
  * weight is per-row scalar; no per-sym statistics
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

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train", "cache")
T59_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train")

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)

SEED_CONFIGS = {
    42:  dict(seed=42,  feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(seed=1,   feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(seed=7,   feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(seed=13,  feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(seed=100, feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split):
    p = os.path.join(CACHE_DIR, f"schemeN_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def get_slicer():
    base_dim = 226
    extras_path = os.path.join(CACHE_DIR, "schemeN_extra_feat_names.txt")
    with open(extras_path) as f:
        extra_names = [line.strip() for line in f]
    n_extras = len(extra_names)
    total_dim = base_dim + n_extras
    report_path = os.path.join(T59_DIR, "sym_invariance_report.json")
    with open(report_path) as f:
        report = json.load(f)
    fail_names = report["summary"]["fail_names"]
    fail_local = [extra_names.index(n) for n in fail_names]
    fail_global = [base_dim + i for i in fail_local]
    drop = set(fail_global)
    for i in range(3):
        drop.add(base_dim - 1 - i)
    keep_idx = np.array(
        [i for i in range(total_dim) if i not in drop],
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


def evaluate(name, preds, y, mp_t, mp_th):
    m = _per_horizon_metrics(preds, y, mp_t, mp_th, fee_rate=0.0001)
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


def predict_proba(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def time_decay_weight(date_arr, alpha, latest_date):
    """exp(-alpha * (latest_date - date_arr)). alpha=0 -> all 1."""
    if alpha == 0.0:
        return np.ones_like(date_arr, dtype=np.float32)
    delta = (latest_date - date_arr).astype(np.float32)
    w = np.exp(-alpha * delta).astype(np.float32)
    # Normalize to mean 1 so absolute scale doesn't change overall regularization
    w = w * (len(w) / w.sum())
    return w


def adversarial_weight(p_train, w_lo=0.5, w_hi=5.0, mode="ratio_clip"):
    """Convert P(is_test|x) per train row into a sample weight."""
    if mode == "ratio_clip":
        w = np.clip(p_train / p_train.mean(), w_lo, w_hi).astype(np.float32)
    elif mode == "rank":
        # Rank-normalize: lowest p -> w_lo, highest p -> w_hi
        order = np.argsort(p_train)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(p_train))
        w = (w_lo + (w_hi - w_lo) * rank / max(len(p_train) - 1, 1)).astype(np.float32)
    else:
        raise ValueError(mode)
    # Normalize to mean 1
    w = w * (len(w) / w.sum())
    return w


def build_train_aug_a(X_orig, y_orig, w_orig, rng, num_class=3):
    """aug_a augment with carry-through sample weight.

    Concatenate original + scaled copy (lo=0.8, hi=1.2). Each row's weight is
    duplicated so the augmented copy inherits the same (decay * adv) base.
    Then class_balance is applied over the merged y.
    """
    X_aug = aug_a_scale(X_orig, rng)
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_orig], axis=0).astype(np.int64)
    w_base = np.concatenate([w_orig, w_orig], axis=0).astype(np.float32)
    # Class-balance, then multiply with the per-row base
    cb = class_balanced_weight(y_merged, num_class)
    w_merged = (cb * w_base).astype(np.float32)
    return X_merged, y_merged, w_merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--decay-alpha", type=float, default=0.0,
                    help="exp(-alpha * (latest_date - date)). 0 = uniform.")
    ap.add_argument("--latest-date", type=int, default=95,
                    help="reference date for decay (defaults to last val date).")
    ap.add_argument("--use-adv-weight", action="store_true", default=False)
    ap.add_argument("--adv-mode", default="ratio_clip", choices=["ratio_clip", "rank"])
    ap.add_argument("--adv-w-lo", type=float, default=0.5)
    ap.add_argument("--adv-w-hi", type=float, default=5.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--out-tag", default="a0", help="tag for output files")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T66 WEIGHTED h={H} alpha={args.decay_alpha} adv={args.use_adv_weight} seeds={seeds} ===", flush=True)

    progress("loading_caches", alpha=args.decay_alpha, seeds=seeds)
    t0 = time.time()
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    keep_idx, feat_names = get_slicer()
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim}", flush=True)

    X_tr = train["X"][:, keep_idx].astype(np.float32, copy=False)
    X_va = val["X"][:, keep_idx].astype(np.float32, copy=False)
    X_te = test["X"][:, keep_idx].astype(np.float32, copy=False)
    y_tr = train[f"y{H}"].astype(np.int64)
    y_va = val[f"y{H}"].astype(np.int64)
    y_te = test[f"y{H}"].astype(np.int64)
    date_tr = train["date"]
    sym_te = test["sym"]
    mp_t_te = test["mp_t"]
    mp_th_te = test[f"mp_t{H}"]
    mp_t_va = val["mp_t"]
    mp_th_va = val[f"mp_t{H}"]

    # Per-row decay weight (train only; val/test get class_balanced only)
    w_decay = time_decay_weight(date_tr, args.decay_alpha, args.latest_date)
    print(f"  decay_w stats: alpha={args.decay_alpha} min={w_decay.min():.4f} mean={w_decay.mean():.4f} max={w_decay.max():.4f}", flush=True)

    if args.use_adv_weight:
        adv_path = os.path.join(HERE, "adv_prob_train.npy")
        if not os.path.exists(adv_path):
            raise RuntimeError(f"missing {adv_path} - run adversarial_validation.py first")
        p_train = np.load(adv_path)
        w_adv = adversarial_weight(p_train, args.adv_w_lo, args.adv_w_hi, mode=args.adv_mode)
        print(f"  adv_w stats: mode={args.adv_mode} min={w_adv.min():.4f} mean={w_adv.mean():.4f} max={w_adv.max():.4f}", flush=True)
        w_base = (w_decay * w_adv).astype(np.float32)
    else:
        w_base = w_decay.astype(np.float32)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T66-WEIGHTED-h{H}-{args.out_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "task": "T66 weighted training",
                    "horizon": H,
                    "seeds": seeds,
                    "feat_dim": feat_dim,
                    **{k: v for k, v in vars(args).items() if k not in ("no_wandb",)},
                },
                tags=["T66", "WEIGHTED", f"h{H}", f"alpha{args.decay_alpha}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    all_results = {}
    for seed in seeds:
        cfg = SEED_CONFIGS[seed]
        print(f"\n{'='*78}\n=== seed={seed} cfg={cfg} ===\n{'='*78}", flush=True)
        progress("training_seed", seed=seed, alpha=args.decay_alpha)

        seed_rng = np.random.default_rng(seed * 7919 + 1)
        X_tr_aug, y_tr_aug, sw_tr = build_train_aug_a(
            X_tr, y_tr, w_base, seed_rng, num_class=NUM_CLASS,
        )
        sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)
        print(f"  n_train_used={len(X_tr_aug):,}  sw_tr stats: min={sw_tr.min():.4f} mean={sw_tr.mean():.4f} max={sw_tr.max():.4f}", flush=True)

        dtrain = lgb.Dataset(
            X_tr_aug, label=y_tr_aug, weight=sw_tr,
            feature_name=feat_names, free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_va, label=y_va, weight=sw_va,
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
        if not args.no_gpu:
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
        print(f"  trained in {train_time:.1f}s, best_iter={booster.best_iteration}", flush=True)

        model_path = os.path.join(HERE, f"weighted_model_h{H}_seed{seed}_{args.out_tag}.txt")
        booster.save_model(model_path, num_iteration=booster.best_iteration)

        va_prob = predict_proba(booster, X_va)
        va_pred = va_prob.argmax(axis=1).astype(np.int8)
        va_metrics = evaluate(f"VAL seed={seed} h={H}", va_pred, y_va, mp_t_va, mp_th_va)

        te_prob = predict_proba(booster, X_te)
        te_pred = te_prob.argmax(axis=1).astype(np.int8)
        te_metrics = evaluate(f"TEST (full) seed={seed} h={H}", te_pred, y_te, mp_t_te, mp_th_te)

        per_sym_metrics = {}
        for k in SYMS:
            m = sym_te == k
            if m.sum() == 0:
                continue
            mk = _per_horizon_metrics(te_pred[m], y_te[m], mp_t_te[m], mp_th_te[m], fee_rate=0.0001)
            per_sym_metrics[int(k)] = {
                "n": int(m.sum()),
                "accuracy": float(mk["accuracy"]),
                "cum_pnl": float(mk["cum_pnl"]),
                "single_pnl": float(mk["single_pnl"]),
                "n_active": int(mk["n_predictions_active"]),
                "f0_5_macro": float(mk["f0_5_macro"]),
            }
        loso_equiv_sum_argmax = sum(v["cum_pnl"] for v in per_sym_metrics.values())
        print(f"  per-sym argmax cum_pnl: {[round(per_sym_metrics[k]['cum_pnl'],3) for k in SYMS]}  sum={loso_equiv_sum_argmax:+.4f}", flush=True)

        sess_map = {0: "am", 1: "pm"}
        te_df = pd.DataFrame({
            "sym": test["sym"].astype(np.int8),
            "date": test["date"].astype(np.int16),
            "session": np.array([sess_map[int(s)] for s in test["sess_idx"]], dtype=object),
            "t": test["t"].astype(np.int16),
            "true_label": y_te.astype(np.int8),
            "pred_label": te_pred.astype(np.int8),
            "prob_0": te_prob[:, 0].astype(np.float32),
            "prob_1": te_prob[:, 1].astype(np.float32),
            "prob_2": te_prob[:, 2].astype(np.float32),
            "midprice_t": mp_t_te.astype(np.float32),
            "midprice_th": mp_th_te.astype(np.float32),
        })
        pred_path = os.path.join(HERE, f"weighted_pred_h{H}_seed{seed}_{args.out_tag}.parquet")
        te_df.to_parquet(pred_path, index=False)
        print(f"  saved {pred_path} ({len(te_df):,} rows)", flush=True)

        all_results[seed] = {
            "horizon": H,
            "seed": seed,
            "decay_alpha": args.decay_alpha,
            "use_adv_weight": args.use_adv_weight,
            "best_iter": int(booster.best_iteration),
            "train_time_sec": float(train_time),
            "val": va_metrics,
            "test_full": te_metrics,
            "per_sym_test_argmax": per_sym_metrics,
            "loso_equiv_sum_argmax": float(loso_equiv_sum_argmax),
        }
        if use_wandb:
            wandb.log({
                f"seed{seed}/best_iter": int(booster.best_iteration),
                f"seed{seed}/test_cum_pnl": te_metrics["cum_pnl"],
                f"seed{seed}/test_single_pnl": te_metrics["single_pnl"],
                f"seed{seed}/test_acc": te_metrics["accuracy"],
                f"seed{seed}/loso_equiv_sum": float(loso_equiv_sum_argmax),
            })

    print(f"\n{'='*78}\n=== Aggregate ===\n{'='*78}", flush=True)
    sums = []
    pnls = []
    for seed in seeds:
        r = all_results[seed]
        sums.append(r["loso_equiv_sum_argmax"])
        pnls.append(r["test_full"]["cum_pnl"])
        print(
            f"seed={seed}: test_full_cum_pnl={r['test_full']['cum_pnl']:+.4f}  "
            f"loso_equiv_sum_argmax={r['loso_equiv_sum_argmax']:+.4f}",
            flush=True,
        )

    summary = {
        "task": f"T66 weighted alpha={args.decay_alpha} adv={args.use_adv_weight}",
        "params": vars(args),
        "horizon": H,
        "seed_results": {f"seed{s}": all_results[s] for s in seeds},
        "agg": {
            "test_full_cum_pnl_per_seed": pnls,
            "loso_equiv_sum_argmax_per_seed": sums,
            "test_full_cum_pnl_mean": float(np.mean(pnls)) if pnls else 0.0,
            "loso_equiv_sum_argmax_mean": float(np.mean(sums)) if sums else 0.0,
        },
    }
    out_path = os.path.join(HERE, f"weighted_summary_h{H}_{args.out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.finish()
    progress("done", alpha=args.decay_alpha, seeds=seeds)


if __name__ == "__main__":
    main()
