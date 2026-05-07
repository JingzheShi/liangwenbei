"""T67 multi-split bagging on R34 Stage 3 schemeN cache.

Hypothesis: 5-seed ensemble (T59/iter_010 +24.52) only varied LightGBM
hyperparameters/seeds — every model saw the same 100% of training data. By
training each bag on a different 80% random subset of train (date 0..79),
we add data-level diversity → stronger variance reduction.

Pipeline per bag k = 0..K-1:
  1. Pool = train (date 0..79, all 5 syms) ~1.47M rows
  2. Sample 80% of pool with seed = bag_seed_base + k * 7919
     => ~1.18M training rows (matches LOSO single-fold magnitude for
        fair comparison)
  3. Apply aug_a [0.80, 1.20] to the 80% subset
  4. Early-stop val = full val (date 80..95) — SAME for all bags so that
     time-OOD signal is preserved (random val would never trigger early stop
     because train and OOB are i.i.d. mixed-date)
  5. LightGBM GPU multiclass logloss, hyperparams cycled across 20 distinct configs
  6. Predict on full 442K test set, save 442K-row parquet with probs + meta

After all bags trained:
  - Average probs across K bags
  - DE 4D threshold search per-sym (LOSO-equivalent objective = SUM of per-sym pnl)
  - DE 4D threshold search single-set (objective = total pnl)

Constraint check: model is sym-agnostic (no sym/date/time features). Verified
identical to T59 (uses same schemeN feature slicer + drop-fail/drop-tail).
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

T59_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train")
T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
CACHE_DIR = os.path.join(T59_DIR, "cache")  # reuse T59 cache (same R34 stage 3 features)

NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)

# 10 hyperparam configs, cycled per bag. For K>10 we cycle around.
BAG_CONFIGS = [
    dict(num_leaves=127, feature_fraction=0.80, bagging_fraction=0.80, lambda_l2=1.0),
    dict(num_leaves=127, feature_fraction=0.60, bagging_fraction=0.70, lambda_l2=1.0),
    dict(num_leaves=63,  feature_fraction=0.70, bagging_fraction=0.85, lambda_l2=2.0),
    dict(num_leaves=255, feature_fraction=0.50, bagging_fraction=0.60, lambda_l2=0.5),
    dict(num_leaves=127, feature_fraction=0.40, bagging_fraction=0.50, lambda_l2=3.0),
    dict(num_leaves=191, feature_fraction=0.65, bagging_fraction=0.75, lambda_l2=1.5),
    dict(num_leaves=95,  feature_fraction=0.55, bagging_fraction=0.65, lambda_l2=2.5),
    dict(num_leaves=159, feature_fraction=0.75, bagging_fraction=0.80, lambda_l2=0.8),
    dict(num_leaves=63,  feature_fraction=0.85, bagging_fraction=0.55, lambda_l2=1.2),
    dict(num_leaves=223, feature_fraction=0.45, bagging_fraction=0.90, lambda_l2=1.8),
    # bags 11..20 (extra configs for K=20 sweep)
    dict(num_leaves=127, feature_fraction=0.50, bagging_fraction=0.85, lambda_l2=2.2),
    dict(num_leaves=159, feature_fraction=0.60, bagging_fraction=0.55, lambda_l2=0.6),
    dict(num_leaves=95,  feature_fraction=0.80, bagging_fraction=0.70, lambda_l2=2.8),
    dict(num_leaves=255, feature_fraction=0.40, bagging_fraction=0.75, lambda_l2=1.0),
    dict(num_leaves=63,  feature_fraction=0.65, bagging_fraction=0.60, lambda_l2=3.5),
    dict(num_leaves=191, feature_fraction=0.55, bagging_fraction=0.85, lambda_l2=0.4),
    dict(num_leaves=127, feature_fraction=0.70, bagging_fraction=0.65, lambda_l2=1.6),
    dict(num_leaves=223, feature_fraction=0.85, bagging_fraction=0.50, lambda_l2=2.0),
    dict(num_leaves=159, feature_fraction=0.45, bagging_fraction=0.90, lambda_l2=0.7),
    dict(num_leaves=95,  feature_fraction=0.75, bagging_fraction=0.70, lambda_l2=1.4),
]


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


def build_keep_idx():
    """Replicates T59 train_loso slicer: drop sym-FAIL extras + drop last 3 (time encoding)."""
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
    drop_set = set(fail_global)
    for i in range(3):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--K", type=int, default=10, help="number of bags")
    ap.add_argument("--K-start", type=int, default=0, help="start bag index (for resume)")
    ap.add_argument("--bag-seed-base", type=int, default=2026)
    ap.add_argument("--subset-frac", type=float, default=0.80)
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
    ap.add_argument("--out-tag", default="bag")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    K = args.K
    print(f"=== T67 multi-split bagging h={H} K={K} subset_frac={args.subset_frac} ===", flush=True)

    progress("loading_caches", K=K)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    keep_idx, feat_names = build_keep_idx()
    feat_dim = len(keep_idx)
    print(f"  feat_dim={feat_dim}", flush=True)

    # Pool = train only (date 0..79, all 5 syms). Val (date 80..95) is held
    # FIXED across all bags as an OOD time-based early-stop signal.
    X_pool = train_full["X"][:, keep_idx].astype(np.float32)
    y_pool = train_full[f"y{H}"].astype(np.int64)
    n_pool = len(X_pool)
    n_subset = int(args.subset_frac * n_pool)
    X_va = val_full["X"][:, keep_idx].astype(np.float32)
    y_va = val_full[f"y{H}"].astype(np.int64)
    print(f"  pool: n_train={n_pool:,}  n_val_for_early_stop={len(X_va):,}", flush=True)
    print(f"  per-bag: n_subset={n_subset:,} (LOSO single-fold ≈ 1.18M)", flush=True)

    # Free the raw splits (keep test for predictions)
    del train_full, val_full

    X_te = test_full["X"][:, keep_idx].astype(np.float32)
    y_te = test_full[f"y{H}"].astype(np.int64)
    sym_te = test_full["sym"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    print(f"  n_test={len(X_te):,}", flush=True)
    print(f"  test sym dist: {dict(zip(*np.unique(sym_te, return_counts=True)))}", flush=True)

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T67-bagK{K}-h{H}-{args.variant}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "scheme": "N (R34 stage 3) multi-split bagging",
                    "n_features": feat_dim,
                    "horizon": H,
                    "K": K,
                    "subset_frac": args.subset_frac,
                    "variant": args.variant,
                    **{k: v for k, v in vars(args).items() if k not in ("no_wandb",)},
                },
                tags=["T67", "R34-stage3", "bagging", f"h{H}", f"K{K}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    feat_mean, feat_std = None, None  # aug_a doesn't need them

    bag_results = {}
    sess_map = {0: "am", 1: "pm"}

    for k in range(args.K_start, K):
        cfg = BAG_CONFIGS[k % len(BAG_CONFIGS)]
        bag_seed = args.bag_seed_base + k * 7919
        print(f"\n{'='*78}\n=== bag k={k} cfg={cfg} bag_seed={bag_seed} ===\n{'='*78}", flush=True)
        progress("training_bag", k=k, K=K, bag_seed=bag_seed)

        # Per-bag random subset of train pool (date 0..79)
        bag_rng = np.random.default_rng(bag_seed)
        all_idx = np.arange(n_pool)
        bag_rng.shuffle(all_idx)
        sub_idx = all_idx[:n_subset]

        X_sub = X_pool[sub_idx]
        y_sub = y_pool[sub_idx]

        # Apply aug_a (concat orig + scaled copy) on the 80% subset only
        aug_rng = np.random.default_rng(bag_seed + 1)
        X_tr_aug, y_tr_aug, sw_tr = build_train_for_variant(
            args.variant, X_sub, y_sub,
            feat_std, feat_mean,
            aug_rng,
            aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
        )
        sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)

        dtrain = lgb.Dataset(
            X_tr_aug, label=y_tr_aug, weight=sw_tr,
            feature_name=feat_names, free_raw_data=False,
        )
        # Shared time-OOD val (date 80..95) — same across bags
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
            "seed": bag_seed,
            "feature_fraction_seed": bag_seed + 11,
            "bagging_seed": bag_seed + 23,
            "data_random_seed": bag_seed + 37,
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

        model_path = os.path.join(HERE, f"bag_model_h{H}_k{k:02d}.txt")
        booster.save_model(model_path, num_iteration=booster.best_iteration)

        # Predict on test
        te_prob = predict_proba(booster, X_te)
        te_pred = te_prob.argmax(axis=1).astype(np.int8)
        # Quick raw cum_pnl on test (no thresh)
        from src.eval.pnl import _per_horizon_metrics
        te_metrics = _per_horizon_metrics(te_pred, y_te, mp_t_te, mp_th_te, fee_rate=0.0001)
        per_sym_pnl = []
        for ks in SYMS:
            m = sym_te == ks
            if m.sum() == 0:
                per_sym_pnl.append(0.0)
                continue
            mk = _per_horizon_metrics(te_pred[m], y_te[m], mp_t_te[m], mp_th_te[m], fee_rate=0.0001)
            per_sym_pnl.append(float(mk["cum_pnl"]))
        loso_equiv = float(sum(per_sym_pnl))
        print(f"  bag k={k}: argmax_total={te_metrics['cum_pnl']:+.4f}  per_sym={[round(p,3) for p in per_sym_pnl]}  loso_equiv={loso_equiv:+.4f}", flush=True)

        # Save predictions
        te_df = pd.DataFrame({
            "sym": test_full["sym"].astype(np.int8),
            "date": test_full["date"].astype(np.int16),
            "session": np.array([sess_map[int(s)] for s in test_full["sess_idx"]], dtype=object),
            "t": test_full["t"].astype(np.int16),
            "true_label": y_te.astype(np.int8),
            "pred_label": te_pred.astype(np.int8),
            "prob_0": te_prob[:, 0].astype(np.float32),
            "prob_1": te_prob[:, 1].astype(np.float32),
            "prob_2": te_prob[:, 2].astype(np.float32),
            "midprice_t": mp_t_te.astype(np.float32),
            "midprice_th": mp_th_te.astype(np.float32),
        })
        pred_path = os.path.join(HERE, f"bag_pred_h{H}_k{k:02d}.parquet")
        te_df.to_parquet(pred_path, index=False)

        bag_results[k] = {
            "k": k,
            "bag_seed": int(bag_seed),
            "cfg": {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in cfg.items()},
            "n_subset": int(n_subset),
            "n_pool": int(n_pool),
            "n_val_for_early_stop": int(len(X_va)),
            "best_iter": int(booster.best_iteration),
            "train_time_sec": float(train_time),
            "test_argmax_total_pnl": float(te_metrics["cum_pnl"]),
            "test_argmax_per_sym": per_sym_pnl,
            "test_argmax_loso_equiv": loso_equiv,
        }

        if use_wandb:
            wandb.log({
                f"bag{k}/best_iter": int(booster.best_iteration),
                f"bag{k}/train_time_sec": float(train_time),
                f"bag{k}/test_argmax_total_pnl": float(te_metrics["cum_pnl"]),
                f"bag{k}/test_argmax_loso_equiv": loso_equiv,
            })

        # Cleanup per-bag arrays
        del X_sub, y_sub, X_tr_aug, y_tr_aug, sw_tr, sw_va
        del dtrain, dval, booster, te_prob, te_pred, te_df

    print(f"\n{'='*78}\n=== Per-bag summary ===\n{'='*78}", flush=True)
    for k, r in bag_results.items():
        print(
            f"bag k={k}: argmax_total={r['test_argmax_total_pnl']:+.4f}  "
            f"loso_equiv={r['test_argmax_loso_equiv']:+.4f}  "
            f"best_iter={r['best_iter']}",
            flush=True,
        )

    summary = {
        "task": "T67 R34 stage 3 multi-split bagging",
        "params": vars(args),
        "horizon": H,
        "K": K,
        "n_pool": int(n_pool),
        "n_subset": int(n_subset),
        "feat_dim": feat_dim,
        "bag_results": bag_results,
    }
    out_path = os.path.join(HERE, f"bag_summary_h{H}_K{K}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.finish()
    progress("done_train", K=K)


if __name__ == "__main__":
    main()
