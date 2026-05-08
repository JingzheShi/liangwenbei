"""T56: XGBoost h_60 LOSO with R34 Stage 3 features (340-d after drop FAIL+tail).

Recipe mirrors T53 (LightGBM 5-seed) but swaps in XGBoost (multi:softprob, GPU
hist) to test whether a different tree-construction algorithm produces a
materially different decision boundary.

Per (seed, held_out_sym):
    train: (sym != held, dates 0..79) + per-(sample,feat) U[0.80,1.20] aug
           (orig + aug concat 1:1, class-balanced weights)
    val:   (sym != held, dates 80..95)  -- early stopping (NO augment)
    test:  (sym == held, dates 96..119) -- held-out predictions (NO augment)

Outputs (per (seed, fold)):
    xgb_pred_seed{S}_held{K}.parquet   -- full DE-thresh schema
    xgb_model_seed{S}_held{K}.json     -- XGBoost native model file

Compliance with CRITICAL_CONSTRAINTS:
  * sym/date NEVER in feature vector (only fold slicing)
  * Stateless per-sample augment, train-only
  * sym-agnostic model (no sym embedding / per-sym anything)
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

import xgboost as xgb  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

T53_CACHE = os.path.join(ROOT, "experiments", "T53_r34_stage3", "cache")
T53_REPORT = os.path.join(ROOT, "experiments", "T53_r34_stage3", "sym_invariance_report.json")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
H = 60

DEFAULT_SEEDS = [42, 1, 7, 13, 100]


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
    p = os.path.join(T53_CACHE, f"schemeN_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def evaluate_metrics(name, preds, y, mp_t, mp_th, fee_rate=0.0001):
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


def aug_uniform_concat(X_orig, y_orig, lo, hi, rng, num_class=3):
    scales = rng.uniform(lo, hi, size=X_orig.shape).astype(X_orig.dtype)
    X_aug = X_orig * scales
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_orig], axis=0).astype(np.int32)
    w_merged = class_balanced_weight(y_merged, num_class)
    return X_merged, y_merged, w_merged


def build_keep_idx(total_dim: int, base_dim: int, n_drop_tail: int):
    """Keep all extras except FAILs, drop trailing time-encoding columns of base."""
    extras_names_path = os.path.join(T53_CACHE, "schemeN_extra_feat_names.txt")
    with open(extras_names_path) as f:
        extra_names = [line.strip() for line in f]
    n_extras = total_dim - base_dim
    assert len(extra_names) == n_extras, f"{len(extra_names)} vs {n_extras}"

    with open(T53_REPORT) as f:
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

    feat_names_path = os.path.join(T53_CACHE, "schemeN_feat_names.txt")
    with open(feat_names_path) as f:
        all_feat_names = [line.strip() for line in f]
    feat_names = [all_feat_names[i] for i in keep_idx]

    return keep_idx, feat_names, fail_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-child-weight", type=float, default=100.0)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=2.0)
    ap.add_argument("--n-drop-tail", type=int, default=3)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--tag", default="xgb")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    target_syms = [int(x) for x in args.syms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    print(
        f"=== T56 XGBoost h_{H} LOSO: seeds={seeds} syms={target_syms} "
        f"aug=[{args.aug_lo},{args.aug_hi}] tag={args.tag} ===",
        flush=True,
    )

    progress("loading_caches", seeds=seeds, syms=target_syms)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )

    base_dim = 226
    total_dim = train_full["X"].shape[1]
    keep_idx, feat_names, fail_names = build_keep_idx(total_dim, base_dim, args.n_drop_tail)
    feat_dim = len(keep_idx)
    print(
        f"  total_dim={total_dim} keep_idx_dim={feat_dim} "
        f"(dropped tail {args.n_drop_tail} + {len(fail_names)} FAIL extras)",
        flush=True,
    )
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN feature leak: {leak}"
    print(f"  no-leak check OK (feat_dim={feat_dim})", flush=True)

    X_tr_all = train_full["X"][:, keep_idx].astype(np.float32)
    X_va_all = val_full["X"][:, keep_idx].astype(np.float32)
    X_te_all = test_full["X"][:, keep_idx].astype(np.float32)

    sym_tr = train_full["sym"]
    sym_va = val_full["sym"]
    sym_te = test_full["sym"]
    y_tr_all = train_full[f"y{H}"]
    y_va_all = val_full[f"y{H}"]
    y_te_all = test_full[f"y{H}"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sess_te = test_full["sess_idx"]
    date_te = test_full["date"]
    t_te = test_full["t"]

    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (
                f"T56-xgb-h{H}-{args.tag}-seeds{','.join(str(s) for s in seeds)}-"
                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "xgboost",
                    "scheme": "N (R34 stage 1+2+3 - FAILs)",
                    "n_features": feat_dim,
                    "horizon": H,
                    "seeds": seeds,
                    "aug_lo": args.aug_lo, "aug_hi": args.aug_hi,
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T56", "xgboost", "R34-stage3", "LOSO", f"h{H}"],
            )
            try:
                wandb.init(entity="cjxh21-Tsinghua University", **init_kwargs)
            except Exception as e_team:
                print(f"  WandB team rejected ({e_team!r}); fallback personal", flush=True)
                wandb.init(**init_kwargs)
        except Exception as e:
            print(f"  WandB init failed ({e!r}); skipping", flush=True)
            use_wandb = False

    all_results = []
    n_total = len(seeds) * len(target_syms)
    done = 0
    for s in seeds:
        print(f"\n{'='*78}\n=== SEED {s} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            tag_fold = f"seed{s}_held{held}"
            print(f"\n  [{done}/{n_total}] {tag_fold}", flush=True)
            progress("training", tag=args.tag, seed=s, held=held,
                     done=done, total=n_total,
                     elapsed_min=round((time.time() - t0) / 60.0, 2))

            m_tr = sym_tr != held
            m_va = sym_va != held
            m_te = sym_te == held

            X_tr_o = X_tr_all[m_tr]
            y_tr_o = y_tr_all[m_tr].astype(np.int32)
            X_va = X_va_all[m_va]
            y_va = y_va_all[m_va].astype(np.int32)
            X_te = X_te_all[m_te]
            y_te = y_te_all[m_te].astype(np.int64)
            mp_t_te_k = mp_t_te[m_te]
            mp_th_te_k = mp_th_te[m_te]

            fold_rng = np.random.default_rng(s * 7919 + held * 17 + 1)
            X_tr, y_tr, sw_tr = aug_uniform_concat(
                X_tr_o, y_tr_o, args.aug_lo, args.aug_hi, fold_rng, NUM_CLASS,
            )
            sw_va = class_balanced_weight(y_va, NUM_CLASS)
            print(
                f"    n_train_orig={len(X_tr_o):,} n_train_used={len(X_tr):,} "
                f"n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]}",
                flush=True,
            )

            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw_tr, feature_names=feat_names)
            dval = xgb.DMatrix(X_va, label=y_va, weight=sw_va, feature_names=feat_names)
            dtest = xgb.DMatrix(X_te, feature_names=feat_names)

            params = {
                "objective": "multi:softprob",
                "num_class": NUM_CLASS,
                "tree_method": "hist",
                "max_depth": args.max_depth,
                "learning_rate": args.learning_rate,
                "min_child_weight": args.min_child_weight,
                "subsample": args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "reg_lambda": args.reg_lambda,
                "seed": s,
                "verbosity": 0,
            }
            if args.use_gpu:
                params["device"] = "cuda"

            t_start = time.time()
            booster = xgb.train(
                params, dtrain,
                num_boost_round=args.num_boost_round,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=args.early_stopping,
                verbose_eval=200,
            )
            train_time = time.time() - t_start
            best_iter = int(booster.best_iteration)
            print(
                f"    trained in {train_time:.1f}s, best_iter={best_iter}",
                flush=True,
            )

            te_prob = booster.predict(
                dtest, iteration_range=(0, best_iter + 1),
            ).astype(np.float32)
            te_pred = te_prob.argmax(axis=1).astype(np.int8)
            te_metrics = evaluate_metrics(
                f"HELD-OUT TEST seed={s} held={held} h={H}",
                te_pred, y_te, mp_t_te_k, mp_th_te_k,
            )

            sess_map = {0: "am", 1: "pm"}
            te_df = pd.DataFrame({
                "sym": sym_te[m_te].astype(np.int8),
                "date": date_te[m_te].astype(np.int16),
                "session": np.array(
                    [sess_map[int(x)] for x in sess_te[m_te]], dtype=object
                ),
                "t": t_te[m_te].astype(np.int16),
                "true_label": y_te.astype(np.int8),
                "pred_label": te_pred.astype(np.int8),
                "prob_0": te_prob[:, 0].astype(np.float32),
                "prob_1": te_prob[:, 1].astype(np.float32),
                "prob_2": te_prob[:, 2].astype(np.float32),
                "midprice_t": mp_t_te_k.astype(np.float32),
                "midprice_th": mp_th_te_k.astype(np.float32),
            })
            pred_path = os.path.join(HERE, f"{args.tag}_pred_{tag_fold}.parquet")
            te_df.to_parquet(pred_path, index=False)

            model_path = os.path.join(HERE, f"{args.tag}_model_{tag_fold}.json")
            booster.save_model(model_path)

            r = {
                "tag": args.tag, "seed": s, "held_out_sym": held,
                "n_train_orig": int(len(X_tr_o)),
                "n_train_used": int(len(X_tr)),
                "n_val": int(len(X_va)),
                "n_test": int(len(X_te)),
                "best_iter": best_iter,
                "train_time_sec": float(train_time),
                "held_cum_pnl": float(te_metrics["cum_pnl"]),
                "held_single_pnl": float(te_metrics["single_pnl"]),
                "held_acc": float(te_metrics["accuracy"]),
                "held_n_active": int(te_metrics["n_predictions_active"]),
                "held_f0_5_macro": float(te_metrics.get("f0_5_macro", 0.0)),
            }
            all_results.append(r)
            if use_wandb:
                wandb.log({
                    f"seed{s}/fold{held}/cum_pnl": r["held_cum_pnl"],
                    f"seed{s}/fold{held}/acc": r["held_acc"],
                    f"seed{s}/fold{held}/best_iter": r["best_iter"],
                    f"seed{s}/fold{held}/train_time_sec": r["train_time_sec"],
                    f"seed{s}/fold{held}/n_active": r["held_n_active"],
                })

            interim = {
                "task": f"T56 XGBoost h_{H} LOSO (tag={args.tag})",
                "horizon": H,
                "seeds": seeds,
                "syms": target_syms,
                "feat_dim": feat_dim,
                "params": {k: v for k, v in vars(args).items() if k != "no_wandb"},
                "results": all_results,
                "elapsed_sec": time.time() - t0,
            }
            with open(os.path.join(HERE, f"results_{args.tag}.json"), "w") as f:
                json.dump(sanitize_for_json(interim), f, indent=2)

    # Per-seed aggregate
    df = pd.DataFrame(all_results)
    print(f"\n{'='*78}\n=== AGGREGATE h={H} (raw argmax per fold) ===\n{'='*78}", flush=True)
    agg = []
    for s in seeds:
        sub = df[df.seed == s]
        cums = sub["held_cum_pnl"].values
        rec = {
            "seed": s, "n_folds": int(len(cums)),
            "cum_pnl_per_fold": cums.tolist(),
            "cum_pnl_sum": float(cums.sum()),
            "cum_pnl_mean": float(cums.mean()) if len(cums) else 0.0,
            "n_pos_folds": int((cums > 0).sum()),
        }
        agg.append(rec)
        print(
            f"  seed{s}: sum={rec['cum_pnl_sum']:+.4f} "
            f"mean={rec['cum_pnl_mean']:+.4f} pos={rec['n_pos_folds']}/{rec['n_folds']} "
            f"per_fold={[f'{x:+.3f}' for x in cums]}",
            flush=True,
        )

    # Cross-seed aggregate (5-seed avg sum)
    overall_sum = float(df["held_cum_pnl"].sum())
    print(f"\n  ALL 25 folds raw-argmax SUM = {overall_sum:+.4f}", flush=True)

    out = {
        "task": f"T56 XGBoost h_{H} LOSO (tag={args.tag})",
        "horizon": H,
        "seeds": seeds,
        "syms": target_syms,
        "feat_dim": feat_dim,
        "params": {k: v for k, v in vars(args).items() if k != "no_wandb"},
        "results": all_results,
        "agg_by_seed": agg,
        "overall_argmax_sum_cum_pnl": overall_sum,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, f"results_{args.tag}.json"), "w") as f:
        json.dump(sanitize_for_json(out), f, indent=2)
    print(f"\nALL DONE in {out['elapsed_sec']:.1f}s. -> results_{args.tag}.json", flush=True)
    progress("loso_done", tag=args.tag, elapsed_sec=out["elapsed_sec"])

    if use_wandb:
        for s in seeds:
            sub = df[df.seed == s]
            wandb.summary[f"seed{s}_argmax_sum_cum_pnl"] = float(sub["held_cum_pnl"].sum())
        wandb.summary["overall_argmax_sum_cum_pnl"] = overall_sum
        wandb.finish()


if __name__ == "__main__":
    main()
