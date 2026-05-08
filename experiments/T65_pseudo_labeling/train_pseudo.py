"""T65 pseudo-labeling: retrain T59 5-seed setup on train+val with pseudo data
appended to train (sample_weight = 0.5 vs 1.0).

Setup mirrors T59 (R34 Stage 3 340-d, aug_a, GPU LightGBM, multiclass logloss,
hyperparam-diverse 5 seeds). Differences:
  - Training set: original train (1.47M, dates 0-79) + pseudo subset of test
    (date 96-119; pseudo label = directional argmax(p0, p2) gated >0.55, |Δ|>0.10).
  - Pseudo rows have sample_weight = 0.5 * class_balanced.
  - Aug-a is applied to original train only; pseudo rows enter unaugmented.
  - Predictions saved on the *full* 442k test (so eval can later compare or
    explicitly report the masked-out (held-out) rows).

Outputs per seed:
  pseudo_model_h60_seed{S}.txt
  pseudo_pred_h60_seed{S}.parquet
  pseudo_summary_h60.json
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

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T59_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train")
T59_CACHE = os.path.join(T59_DIR, "cache")
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
    p = os.path.join(T59_CACHE, f"schemeN_{split}.npz")
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


def get_slicer(args, train_full):
    """Replicate T59's drop-fail-extras + drop-tail slicing -> 340-d feature set."""
    base_dim = 226
    total_dim = train_full["X"].shape[1]
    n_extras = total_dim - base_dim
    extras_names_path = os.path.join(T59_CACHE, "schemeN_extra_feat_names.txt")
    with open(extras_names_path) as f:
        extra_names = [line.strip() for line in f]
    assert len(extra_names) == n_extras

    if args.drop_fail_extras:
        report_path = os.path.join(T59_DIR, "sym_invariance_report.json")
        with open(report_path) as f:
            report = json.load(f)
        fail_names = report["summary"]["fail_names"]
        fail_local = [extra_names.index(n) for n in fail_names]
        fail_global = [base_dim + i for i in fail_local]
        drop_set = set(fail_global)
        if not args.no_drop_tail:
            for i in range(args.n_drop_tail):
                drop_set.add(base_dim - 1 - i)
        keep_idx = np.array(
            [i for i in range(total_dim) if i not in drop_set],
            dtype=np.int64,
        )
        return keep_idx
    if args.no_drop_tail:
        return np.arange(total_dim, dtype=np.int64)
    return np.concatenate([
        np.arange(0, base_dim - args.n_drop_tail),
        np.arange(base_dim, total_dim),
    ]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seeds", default="1,7,13,42,100")
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
    ap.add_argument("--no-drop-tail", action="store_true")
    ap.add_argument("--drop-fail-extras", action="store_true", default=True)
    ap.add_argument("--no-drop-fail-extras", dest="drop_fail_extras", action="store_false")
    ap.add_argument("--pseudo-weight", type=float, default=0.5,
                    help="multiplicative downweight on pseudo rows after class-balance")
    ap.add_argument("--out-tag", default="pseudo_v1")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== T65 PSEUDO h={H} variant={args.variant} seeds={seeds} pseudo_w={args.pseudo_weight} ===", flush=True)

    progress("loading_caches", seeds=seeds)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Slicer
    slicer = get_slicer(args, train_full)
    feat_dim = len(slicer)
    print(f"  feat_dim={feat_dim}", flush=True)

    feat_names_path = os.path.join(T59_CACHE, "schemeN_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    feat_names = [all_feat_names[i] for i in slicer]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN feature leak: {leak}"
    assert len(feat_names) == feat_dim

    # Load pseudo idx (into test_full row order)
    pseudo_path = os.path.join(HERE, "pseudo_idx_h60.npz")
    if not os.path.exists(pseudo_path):
        raise FileNotFoundError(f"missing {pseudo_path} — run gen_pseudo.py first")
    psd = np.load(pseudo_path)
    pseudo_idx = psd["idx"]
    pseudo_label = psd["label"]
    n_pseudo = len(pseudo_idx)
    print(f"  pseudo: n={n_pseudo:,} from test (subset of date 96-119)", flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T65-PSEUDO-h{H}-{args.variant}-{args.out_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "scheme": "N (R34 stage 3 340-d) + pseudo",
                    "n_features": feat_dim,
                    "horizon": H,
                    "variant": args.variant,
                    "seeds": seeds,
                    "n_pseudo": int(n_pseudo),
                    "pseudo_weight": float(args.pseudo_weight),
                    **{k: v for k, v in vars(args).items() if k not in ("no_wandb",)},
                },
                tags=["T65", "PSEUDO", f"h{H}", "iter1"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    import gc

    X_tr = train_full["X"][:, slicer].astype(np.float32)
    X_va = val_full["X"][:, slicer].astype(np.float32)
    X_te_full = test_full["X"][:, slicer].astype(np.float32)
    y_tr = train_full[f"y{H}"].astype(np.int64)
    y_va = val_full[f"y{H}"].astype(np.int64)
    y_te = test_full[f"y{H}"].astype(np.int64)
    sym_te = test_full["sym"].copy()
    date_te = test_full["date"].copy()
    sess_te = test_full["sess_idx"].copy()
    t_te = test_full["t"].copy()
    mp_t_te = test_full["mp_t"].copy()
    mp_th_te = test_full[f"mp_t{H}"].copy()
    mp_t_va = val_full["mp_t"].copy()
    mp_th_va = val_full[f"mp_t{H}"].copy()

    # Pseudo X (sliced)
    X_pseudo = X_te_full[pseudo_idx].copy()
    y_pseudo = pseudo_label.astype(np.int64)

    # Free large source arrays now that we've extracted the slices we need
    del train_full, val_full, test_full
    gc.collect()
    print(f"  n_train={len(X_tr):,}  n_val={len(X_va):,}  n_test={len(X_te_full):,}  n_pseudo={n_pseudo:,}", flush=True)

    feat_mean, feat_std = None, None  # aug_a doesn't use them

    all_results = {}
    for seed in seeds:
        cfg = SEED_CONFIGS[seed]
        print(f"\n{'='*78}\n=== seed={seed} cfg={cfg} ===\n{'='*78}", flush=True)
        progress("training_seed", seed=seed)

        model_path = os.path.join(HERE, f"pseudo_model_h{H}_seed{seed}.txt")
        pred_path = os.path.join(HERE, f"pseudo_pred_h{H}_seed{seed}.parquet")
        if os.path.exists(model_path) and os.path.exists(pred_path):
            print(f"  model+parquet already exist for seed={seed}, skipping training+pred", flush=True)
            continue
        if os.path.exists(model_path) and not os.path.exists(pred_path):
            print(f"  model exists but missing parquet — load and predict only", flush=True)
            booster = lgb.Booster(model_file=model_path)
            train_time = 0.0

            va_prob = predict_proba(booster, X_va)
            va_pred = va_prob.argmax(axis=1).astype(np.int8)
            va_metrics = evaluate(f"VAL seed={seed} h={H}", va_pred, y_va, mp_t_va, mp_th_va)
            te_prob = predict_proba(booster, X_te_full)
            te_pred = te_prob.argmax(axis=1).astype(np.int8)
            te_metrics = evaluate(
                f"TEST (full 5-sym) seed={seed} h={H}",
                te_pred, y_te, mp_t_te, mp_th_te,
            )
            per_sym_metrics = {}
            for k in SYMS:
                m = sym_te == k
                if m.sum() == 0:
                    continue
                mk = _per_horizon_metrics(
                    te_pred[m], y_te[m], mp_t_te[m], mp_th_te[m], fee_rate=0.0001,
                )
                per_sym_metrics[int(k)] = {
                    "n": int(m.sum()),
                    "accuracy": float(mk["accuracy"]),
                    "cum_pnl": float(mk["cum_pnl"]),
                    "single_pnl": float(mk["single_pnl"]),
                    "n_active": int(mk["n_predictions_active"]),
                    "f0_5_macro": float(mk["f0_5_macro"]),
                }
            loso_equiv_sum_argmax = sum(v["cum_pnl"] for v in per_sym_metrics.values())

            sess_map = {0: "am", 1: "pm"}
            is_pseudo_mask = np.zeros(len(X_te_full), dtype=bool)
            is_pseudo_mask[pseudo_idx] = True
            te_df = pd.DataFrame({
                "sym": sym_te.astype(np.int8),
                "date": date_te.astype(np.int16),
                "session": np.array([sess_map[int(s)] for s in sess_te], dtype=object),
                "t": t_te.astype(np.int16),
                "true_label": y_te.astype(np.int8),
                "pred_label": te_pred.astype(np.int8),
                "prob_0": te_prob[:, 0].astype(np.float32),
                "prob_1": te_prob[:, 1].astype(np.float32),
                "prob_2": te_prob[:, 2].astype(np.float32),
                "midprice_t": mp_t_te.astype(np.float32),
                "midprice_th": mp_th_te.astype(np.float32),
                "is_pseudo": is_pseudo_mask,
            })
            te_df.to_parquet(pred_path, index=False)
            print(f"  saved {pred_path}", flush=True)

            all_results[seed] = {
                "horizon": H,
                "seed": seed,
                "n_train_orig": int(len(X_tr)),
                "n_train_aug": -1,
                "n_pseudo": int(n_pseudo),
                "n_train_total": -1,
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te_full)),
                "best_iter": int(booster.best_iteration if booster.best_iteration is not None else 0),
                "train_time_sec": float(train_time),
                "val": va_metrics,
                "test_full": te_metrics,
                "per_sym_test_argmax": per_sym_metrics,
                "loso_equiv_sum_argmax": float(loso_equiv_sum_argmax),
                "resumed": True,
            }
            continue

        seed_rng = np.random.default_rng(seed * 7919 + 1)
        # Aug only the original train (not pseudo)
        X_tr_aug, y_tr_aug, sw_tr_aug = build_train_for_variant(
            args.variant, X_tr, y_tr, feat_std, feat_mean, seed_rng,
            aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
        )
        n_aug = len(X_tr_aug)

        # Append pseudo (no aug). Compute sample weights jointly so class balance
        # accounts for both original-aug rows and pseudo rows.
        X_full = np.concatenate([X_tr_aug, X_pseudo], axis=0)
        y_full = np.concatenate([y_tr_aug, y_pseudo], axis=0).astype(np.int64)
        del X_tr_aug, y_tr_aug, sw_tr_aug
        gc.collect()
        sw_full = class_balanced_weight(y_full, num_class=NUM_CLASS)
        # Apply pseudo downweight
        sw_full[n_aug:] *= float(args.pseudo_weight)
        print(f"  n_aug={n_aug:,} n_pseudo={n_pseudo:,} -> n_total={len(X_full):,}", flush=True)
        print(f"  sw stats: orig_mean={sw_full[:n_aug].mean():.4f}  pseudo_mean={sw_full[n_aug:].mean():.4f}", flush=True)

        sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)

        dtrain = lgb.Dataset(
            X_full, label=y_full, weight=sw_full,
            feature_name=feat_names, free_raw_data=True,
        )
        dval = lgb.Dataset(
            X_va, label=y_va, weight=sw_va,
            feature_name=feat_names, reference=dtrain, free_raw_data=True,
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
        print(f"  trained in {train_time:.1f}s, best_iter={booster.best_iteration}", flush=True)

        booster.save_model(model_path, num_iteration=booster.best_iteration)

        va_prob = predict_proba(booster, X_va)
        va_pred = va_prob.argmax(axis=1).astype(np.int8)
        va_metrics = evaluate(
            f"VAL seed={seed} h={H}", va_pred, y_va, mp_t_va, mp_th_va,
        )

        te_prob = predict_proba(booster, X_te_full)
        te_pred = te_prob.argmax(axis=1).astype(np.int8)
        te_metrics = evaluate(
            f"TEST (full 5-sym) seed={seed} h={H}",
            te_pred, y_te, mp_t_te, mp_th_te,
        )

        # Per-sym test metrics
        per_sym_metrics = {}
        for k in SYMS:
            m = sym_te == k
            if m.sum() == 0:
                continue
            mk = _per_horizon_metrics(
                te_pred[m], y_te[m], mp_t_te[m], mp_th_te[m], fee_rate=0.0001,
            )
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
        is_pseudo_mask = np.zeros(len(X_te_full), dtype=bool)
        is_pseudo_mask[pseudo_idx] = True
        te_df = pd.DataFrame({
            "sym": sym_te.astype(np.int8),
            "date": date_te.astype(np.int16),
            "session": np.array([sess_map[int(s)] for s in sess_te], dtype=object),
            "t": t_te.astype(np.int16),
            "true_label": y_te.astype(np.int8),
            "pred_label": te_pred.astype(np.int8),
            "prob_0": te_prob[:, 0].astype(np.float32),
            "prob_1": te_prob[:, 1].astype(np.float32),
            "prob_2": te_prob[:, 2].astype(np.float32),
            "midprice_t": mp_t_te.astype(np.float32),
            "midprice_th": mp_th_te.astype(np.float32),
            "is_pseudo": is_pseudo_mask,
        })
        te_df.to_parquet(pred_path, index=False)
        print(f"  saved {pred_path} ({len(te_df):,} rows; {is_pseudo_mask.sum():,} pseudo)", flush=True)

        all_results[seed] = {
            "horizon": H,
            "seed": seed,
            "n_train_orig": int(len(X_tr)),
            "n_train_aug": int(n_aug),
            "n_pseudo": int(n_pseudo),
            "n_train_total": int(len(X_full)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te_full)),
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
                f"seed{seed}/train_time_sec": float(train_time),
                f"seed{seed}/test_cum_pnl": te_metrics["cum_pnl"],
                f"seed{seed}/test_single_pnl": te_metrics["single_pnl"],
                f"seed{seed}/test_acc": te_metrics["accuracy"],
                f"seed{seed}/loso_equiv_sum": float(loso_equiv_sum_argmax),
            })

    print(f"\n{'='*78}\n=== Aggregate ===\n{'='*78}", flush=True)
    summary = {
        "task": "T65 R34 stage 3 + pseudo-labeling 5-seed",
        "params": vars(args),
        "horizon": H,
        "seed_results": all_results,
    }
    sum_path = os.path.join(HERE, f"pseudo_summary_h{H}.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  wrote -> {sum_path}", flush=True)
    progress("done")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
