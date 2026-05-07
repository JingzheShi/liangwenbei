"""T44 LOSO h=60 LightGBM training on schemeL cache (226 + 54 = 280 dim).

Reuses iter_005b's training pattern (T27 train_final_aug_a):
  - aug_a = per-(sample, feature) random scale [0.80, 1.20] applied to train only
  - GPU LightGBM (device=gpu, gpu_use_dp=False)
  - early-stopping on val split

LOSO setup (matches T5b train_loso.py):
  For each held_sym in {0..4}:
    train: schemeL_train[sym != held] (dates 0..79)
    val:   schemeL_val[sym != held]   (dates 80..95) - early stopping
    test:  schemeL_test[sym == held]  (dates 96..119) - held-out evaluation

Outputs per (seed, held):
  loso_model_h60_seed{S}_held{K}.txt
  loso_pred_h60_seed{S}_held{K}.parquet
  loso_summary_h60_seed{S}.json

Constraint compliance:
  - sym/date/time NEVER in feature vector (we slice off the last 3 time-encoding cols if requested)
  - aug only on train, never on val
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

# Reuse aug helpers (stateless functions)
T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import (  # noqa: E402
    build_train_for_variant,
    class_balanced_weight,
    compute_feat_stats,
)

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)

# Same as T27 SEED_CONFIGS.
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
    p = os.path.join(CACHE_DIR, f"schemeL_{split}.npz")
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
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--syms", default="0,1,2,3,4")
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
    ap.add_argument("--n-drop-tail", type=int, default=3,
                    help="drop trailing N feature dims (time encoding sits at base 224..226 in 280-d cache)")
    ap.add_argument("--no-drop-tail", action="store_true",
                    help="if set, do NOT drop trailing dims (keep all 280)")
    ap.add_argument("--drop-fail-extras", action="store_true",
                    help="drop 3 FAIL extras (dualz_ask_diff1, dualz_bid_diff5, dualz_ask_diff5); "
                         "keeps all 226 base + 51 extras = 277 dim. Overrides --n-drop-tail.")
    ap.add_argument("--out-tag", default="r34_stage1")
    ap.add_argument("--pred-prefix", default="loso_pred",
                    help="output parquet filename prefix; final path = HERE/{prefix}_h{H}_seed{S}_held{K}.parquet "
                         "or HERE/{prefix}_seed{S}_held{K}.parquet if --pred-no-h-tag is set")
    ap.add_argument("--pred-no-h-tag", action="store_true",
                    help="omit h{H} from prediction filename")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    H = args.horizon
    seeds = [int(x) for x in args.seeds.split(",")]
    syms = [int(x) for x in args.syms.split(",")]
    print(f"=== T44 LOSO h={H} variant={args.variant} seeds={seeds} syms={syms} ===", flush=True)

    progress("loading_caches", seeds=seeds, syms=syms)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    # Time-encoding cols are at 223,224,225 (0-indexed) of the base 226-d cache.
    # In schemeL with 226+54=280, the time-encoding is still at 223-225, then 54 new at 226-279.
    # We want to *drop time encoding only*, keep the new 54. So we explicitly slice [0:223] + [226:280].
    base_dim = 226
    n_extras = train_full["X"].shape[1] - base_dim  # 54
    if args.drop_fail_extras:
        # T44b spec: keep all 226 base + 51 extras (drop 3 FAIL columns)
        # FAIL extras (sym=2 std ~0): dualz_ask_diff1, dualz_bid_diff5, dualz_ask_diff5
        # Look up indices by name from extra-feat-names file.
        extras_names_path = os.path.join(CACHE_DIR, "schemeL_extra_feat_names.txt")
        with open(extras_names_path) as f:
            extra_names = [line.strip() for line in f]
        fail_names = ["dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5"]
        fail_local = [extra_names.index(n) for n in fail_names]
        fail_global = [base_dim + i for i in fail_local]
        keep_idx = np.array(
            [i for i in range(base_dim + n_extras) if i not in set(fail_global)],
            dtype=np.int64,
        )
        slicer = keep_idx
        feat_dim = len(keep_idx)
        print(f"  drop-fail-extras: dropping global idx {fail_global} ({fail_names})", flush=True)
    elif args.no_drop_tail:
        slicer = slice(0, train_full["X"].shape[1])
        feat_dim = train_full["X"].shape[1]
    else:
        # keep base[:-n_drop_tail] + extras
        keep_idx = np.concatenate([
            np.arange(0, base_dim - args.n_drop_tail),
            np.arange(base_dim, base_dim + n_extras),
        ])
        slicer = keep_idx
        feat_dim = len(keep_idx)
    print(f"  base_dim={base_dim} n_extras={n_extras} feat_dim={feat_dim}", flush=True)

    # Load feat_names and align to slicer
    feat_names_path = os.path.join(CACHE_DIR, "schemeL_feat_names.txt")
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

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = f"T44-LOSO-h{H}-{args.variant}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name=run_name,
                config={
                    "model": "lightgbm-gpu",
                    "scheme": "L (R34 stage 1)",
                    "n_features": feat_dim,
                    "horizon": H,
                    "variant": args.variant,
                    "seeds": seeds,
                    "syms": syms,
                    **{k: v for k, v in vars(args).items() if k not in ("no_wandb",)},
                },
                tags=["T44", "R34-stage1", "LOSO", f"h{H}"],
            )
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    feat_mean, feat_std = None, None  # only needed for aug_b/c/abc; aug_a doesn't use them

    # Pre-extract feature matrices once
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
            X_va = X_va_full[m_va]
            y_va = y_va_full[m_va].astype(np.int64)
            X_te = X_te_full[m_te]
            y_te = y_te_full[m_te].astype(np.int64)
            mp_t_te_k = mp_t_te[m_te]
            mp_th_te_k = mp_th_te[m_te]
            mp_t_va_k = mp_t_va[m_va]
            mp_th_va_k = mp_th_va[m_va]

            print(f"    n_train_orig={len(X_tr_orig):,} n_val={len(X_va):,} n_test={len(X_te):,}",
                  flush=True)

            seed_rng = np.random.default_rng(seed * 7919 + 1 + held * 31)
            X_tr, y_tr, sw_tr = build_train_for_variant(
                args.variant, X_tr_orig, y_tr_orig,
                feat_std, feat_mean,
                seed_rng,
                aug_ratio=args.aug_ratio, num_class=NUM_CLASS,
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
            if args.pred_no_h_tag:
                pred_tag = f"seed{seed}_held{held}"
            else:
                pred_tag = tag
            pred_path = os.path.join(HERE, f"{args.pred_prefix}_{pred_tag}.parquet")
            te_df.to_parquet(pred_path, index=False)

            model_path = os.path.join(HERE, f"loso_model_{tag}.txt")
            booster.save_model(model_path, num_iteration=booster.best_iteration)

            all_results[seed][held] = {
                "horizon": H,
                "held": held,
                "seed": seed,
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

    # Aggregate per seed
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
        "task": "T44 R34 stage 1 features LOSO",
        "params": vars(args),
        "horizon": H,
        "fold_results": {f"seed{s}": all_results[s] for s in seeds},
        "aggregate": {f"seed{s}": agg[s] for s in seeds},
    }
    out_path = os.path.join(HERE, f"loso_summary_h{H}_{args.out_tag}.json")
    with open(out_path, "w") as f:
        json.dump(sanitize_for_json(summary), f, indent=2)
    print(f"\nsummary -> {out_path}", flush=True)

    if use_wandb:
        wandb.finish()
    progress("loso_done", seeds=seeds)


if __name__ == "__main__":
    main()
