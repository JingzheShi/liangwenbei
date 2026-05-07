"""T52 LOSO h=60 LightGBM training on schemeC 226-d cache with cross-sym mixup
augment (Method B: per-pair 2 weighted rows) stacked on top of aug_a.

For each held_sym in {0..4}:
  train: schemeC_train[sym != held]  (1.18M)
  val:   schemeC_val[sym != held]    (235k)  - early stopping
  test:  schemeC_test[sym == held]   (88k)   - held-out evaluation

Outputs per (seed, held):
  loso_model_h60_{tag}_held{K}.txt
  loso_pred_h60_{tag}_held{K}.parquet
  loso_summary_h60_{tag}.json
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
from build_aug import class_balanced_weight  # noqa: E402

sys.path.insert(0, HERE)
from build_mixup import build_train_with_mixup  # noqa: E402

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)


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
    p = os.path.join(CACHE_DIR, f"schemeC_{split}.npz")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--use-aug-a", action="store_true", default=True)
    ap.add_argument("--no-aug-a", dest="use_aug_a", action="store_false")
    ap.add_argument("--aug-a-ratio", type=float, default=1.0)
    ap.add_argument("--mixup-ratio", type=float, default=0.5,
                    help="number of mixup PAIRS per orig (each pair -> 2 weighted rows)")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Beta(alpha, alpha) shape parameter")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--tag", default=None,
                    help="output tag; default auto = mr{mixup_ratio}_a{alpha}_seed{S}")
    ap.add_argument("--save-models", action="store_true", default=True)
    ap.add_argument("--no-save-models", dest="save_models", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    syms = [int(x) for x in args.syms.split(",")]
    if args.tag is None:
        mr_tag = f"{args.mixup_ratio:.2f}".replace(".", "p")
        a_tag = f"{args.alpha:.2f}".replace(".", "p")
        auga = "auga" if args.use_aug_a else "noaa"
        args.tag = f"mr{mr_tag}_a{a_tag}_{auga}_seed{args.seed}"
    print(f"=== T52 LOSO h={H} tag={args.tag} use_aug_a={args.use_aug_a} "
          f"aug_a_ratio={args.aug_a_ratio} mixup_ratio={args.mixup_ratio} "
          f"alpha={args.alpha} seed={args.seed} syms={syms} ===", flush=True)

    progress("loading_caches", tag=args.tag, syms=syms)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  train X: {train_full['X'].shape}", flush=True)

    feat_names_path = os.path.join(CACHE_DIR, "schemeC_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [l.strip() for l in f]
    feat_dim = len(feat_names)
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN feature leak: {leak}"
    assert feat_dim == train_full["X"].shape[1], \
        f"feat_dim mismatch: {feat_dim} vs {train_full['X'].shape[1]}"
    print(f"  feat_dim={feat_dim} (no date/sym/time leak)", flush=True)

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T52-LOSO-h{H}-{args.tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "scheme": "C 226-d + aug_a + cross_sym_mixup_methodB",
                    "n_features": feat_dim,
                    "horizon": H,
                    **{k: v for k, v in vars(args).items() if k not in ("no_wandb",)},
                },
                tags=["T52", "cross-sym-mixup", "LOSO", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    X_tr_full = train_full["X"].astype(np.float32, copy=False)
    X_va_full = val_full["X"].astype(np.float32, copy=False)
    X_te_full = test_full["X"].astype(np.float32, copy=False)
    y_tr_full = train_full[f"y{H}"]
    y_va_full = val_full[f"y{H}"]
    y_te_full = test_full[f"y{H}"]
    sym_tr = train_full["sym"]
    sym_va = val_full["sym"]
    sym_te = test_full["sym"]
    mp_t_va = val_full["mp_t"]
    mp_th_va = val_full[f"mp_t{H}"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]

    fold_results = {}
    for held in syms:
        tag_fold = f"h{H}_{args.tag}_held{held}"
        print(f"\n  --- fold {tag_fold} ---", flush=True)
        progress(f"training_fold_{held}", tag=args.tag)

        m_tr = sym_tr != held
        m_va = sym_va != held
        m_te = sym_te == held

        X_tr_orig = X_tr_full[m_tr]
        y_tr_orig = y_tr_full[m_tr].astype(np.int64)
        sym_tr_fold = sym_tr[m_tr]
        X_va = X_va_full[m_va]
        y_va = y_va_full[m_va].astype(np.int64)
        X_te = X_te_full[m_te]
        y_te = y_te_full[m_te].astype(np.int64)
        mp_t_te_k = mp_t_te[m_te]
        mp_th_te_k = mp_th_te[m_te]

        print(f"    n_train_orig={len(X_tr_orig):,} n_val={len(X_va):,} n_test={len(X_te):,}",
              flush=True)

        # IMPORTANT (per spec): mixup only across train syms (sym != held). The
        # held sym is already excluded by m_tr, so this is automatic.
        seed_fold = args.seed * 7919 + 1 + held * 31
        rng = np.random.default_rng(seed_fold)

        X_tr, y_tr, sw_tr = build_train_with_mixup(
            X_tr_orig, y_tr_orig, sym_tr_fold,
            rng=rng,
            use_aug_a=args.use_aug_a,
            aug_a_ratio=args.aug_a_ratio,
            mixup_ratio=args.mixup_ratio,
            alpha=args.alpha,
            num_class=NUM_CLASS,
        )
        sw_va = class_balanced_weight(y_va, num_class=NUM_CLASS)
        print(f"    n_train_used={len(X_tr):,}", flush=True)

        dtrain = lgb.Dataset(
            X_tr, label=y_tr, weight=sw_tr,
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
            "num_leaves": args.num_leaves,
            "min_data_in_leaf": args.min_data_in_leaf,
            "feature_fraction": args.feature_fraction,
            "bagging_fraction": args.bagging_fraction,
            "bagging_freq": args.bagging_freq,
            "lambda_l2": args.lambda_l2,
            "num_threads": args.num_threads,
            "seed": args.seed,
            "feature_fraction_seed": args.seed + 1,
            "bagging_seed": args.seed + 2,
            "data_random_seed": args.seed + 3,
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
            f"HELD-OUT TEST tag={args.tag} held={held} h={H}",
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
        pred_path = os.path.join(HERE, f"loso_pred_{tag_fold}.parquet")
        te_df.to_parquet(pred_path, index=False)

        if args.save_models:
            model_path = os.path.join(HERE, f"loso_model_{tag_fold}.txt")
            booster.save_model(model_path, num_iteration=booster.best_iteration)

        fold_results[held] = {
            "horizon": H,
            "held": held,
            "seed": args.seed,
            "n_train_orig": int(len(X_tr_orig)),
            "n_train_used": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_test": int(len(X_te)),
            "best_iter": int(booster.best_iteration),
            "train_time_sec": float(train_time),
            "test": te_metrics,
        }
        if use_wandb:
            wandb.log({
                f"fold{held}/best_iter": int(booster.best_iteration),
                f"fold{held}/train_time_sec": float(train_time),
                f"fold{held}/test_cum_pnl": te_metrics["cum_pnl"],
                f"fold{held}/test_single_pnl": te_metrics["single_pnl"],
                f"fold{held}/test_acc": te_metrics["accuracy"],
                f"fold{held}/test_f0_5": te_metrics["f0_5_macro"],
            })

        del X_tr, y_tr, sw_tr, dtrain, dval, X_tr_orig, y_tr_orig

    print(f"\n{'='*78}\n=== Aggregate ({args.tag}) ===\n{'='*78}", flush=True)
    cums, accs = [], []
    for held in sorted(fold_results.keys()):
        r = fold_results[held]
        cums.append(r["test"]["cum_pnl"])
        accs.append(r["test"]["accuracy"])
    cums_a = np.array(cums)
    print(f"raw_argmax_sum={cums_a.sum():+.4f}  per-fold={[f'{x:+.3f}' for x in cums_a]}",
          flush=True)
    print(f"acc_per_fold={[f'{a:.3f}' for a in accs]}", flush=True)

    summary = {
        "task": f"T52 cross-sym mixup tag={args.tag}",
        "params": vars(args),
        "horizon": H,
        "fold_results": fold_results,
        "aggregate": {
            "horizon": H,
            "tag": args.tag,
            "n_folds": int(len(cums_a)),
            "cum_pnl_per_fold": cums_a.tolist(),
            "acc_per_fold": accs,
            "raw_argmax_sum": float(cums_a.sum()),
            "n_pos_folds": int((cums_a > 0).sum()),
        },
    }
    out_path = os.path.join(HERE, f"loso_summary_h{H}_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.summary.update({
            "agg/raw_argmax_sum": float(cums_a.sum()),
            "agg/n_pos_folds": int((cums_a > 0).sum()),
            "alpha": args.alpha,
            "mixup_ratio": args.mixup_ratio,
            "use_aug_a": args.use_aug_a,
        })
        wandb.finish()
    progress("loso_done", tag=args.tag, sum=float(cums_a.sum()))


if __name__ == "__main__":
    main()
