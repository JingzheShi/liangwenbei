"""T31 main runner: load data once, train many aug variants in sequence.

For each (variant, fold):
  - augment train + train LightGBM-GPU
  - save OOF preds for held-out sym
  - sweep thresholds on accumulated 5-fold OOF, pick best
  - log row to summary CSV

Variants are passed as a JSON config file with fields:
  [
    {"name": "aug_a_0.8_1.2", "kind": "uniform", "lo": 0.8, "hi": 1.2, "aug_ratio": 1.0, "seed": 42},
    {"name": "aug_pg_v1", "kind": "per_group", "group_ranges": {...}, "seed": 42},
    ...
  ]
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

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

from build_aug_v2 import (  # noqa: E402
    build_train_for_variant, class_balanced_weight,
    compute_feat_stats, build_group_indices,
)

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
SYMS = (0, 1, 2, 3, 4)
NUM_CLASS = 3
N_DROP_TAIL = 3  # drop trailing time-encoding cols for 226 -> 223
H = 60

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running", "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def load_split(split: str) -> dict:
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_proba(booster, X, batch=200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def thresholded_pred(probs, T, delta):
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def thresh_sweep(fold_dfs):
    """Sweep T_GRID x D_GRID across fold_dfs (dict[k]->df). Return best dict."""
    best = None
    for T in T_GRID:
        for d in D_GRID:
            per_fold = []
            for k, df in fold_dfs.items():
                probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
                pred = thresholded_pred(probs, T, d)
                m = _per_horizon_metrics(
                    pred, df["true_label"].to_numpy(np.int64),
                    df["midprice_t"].to_numpy(np.float32),
                    df["midprice_th"].to_numpy(np.float32),
                    fee_rate=0.0001,
                )
                per_fold.append({"sym": k, "cum_pnl": float(m["cum_pnl"]),
                                 "n_active": int(m["n_predictions_active"]),
                                 "accuracy": float(m["accuracy"])})
            sum_pnl = sum(r["cum_pnl"] for r in per_fold)
            n_pos = sum(1 for r in per_fold if r["cum_pnl"] > 0)
            row = {
                "T": T, "delta": d, "sum_pnl": sum_pnl,
                "n_pos": n_pos,
                "per_fold": [r["cum_pnl"] for r in per_fold],
                "n_active_total": sum(r["n_active"] for r in per_fold),
            }
            if best is None or sum_pnl > best["sum_pnl"]:
                best = row
    return best


def train_one_fold(cfg, held, train_full, val_full, test_full, group_indices,
                   feat_names, gbm_args, seed):
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr_o = train_full["X"][m_tr][:, :-N_DROP_TAIL]
    y_tr_o = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL]
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL]
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[f"mp_t{H}"][m_te]

    feat_mean, feat_std = compute_feat_stats(X_tr_o)
    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
    X_tr, y_tr, sw_tr = build_train_for_variant(
        cfg, X_tr_o, y_tr_o, feat_std, feat_mean, group_indices, fold_rng, NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, NUM_CLASS)

    dtrain = lgb.Dataset(X_tr, label=y_tr, weight=sw_tr,
                         feature_name=feat_names, free_raw_data=True)
    dval = lgb.Dataset(X_va, label=y_va, weight=sw_va,
                       feature_name=feat_names, reference=dtrain, free_raw_data=True)

    params = {
        "objective": "multiclass", "num_class": NUM_CLASS,
        "metric": "multi_logloss",
        "learning_rate": gbm_args.learning_rate,
        "num_leaves": gbm_args.num_leaves,
        "min_data_in_leaf": gbm_args.min_data_in_leaf,
        "feature_fraction": gbm_args.feature_fraction,
        "bagging_fraction": gbm_args.bagging_fraction,
        "bagging_freq": gbm_args.bagging_freq,
        "lambda_l2": gbm_args.lambda_l2,
        "num_threads": gbm_args.num_threads,
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbose": -1,
    }
    if gbm_args.use_gpu:
        params.update({"device": "gpu", "gpu_use_dp": False})

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=gbm_args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=gbm_args.early_stopping, verbose=False),
        ],
    )
    train_time = time.time() - t_start

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)

    sess_map = {0: "am", 1: "pm"}
    df = pd.DataFrame({
        "sym": test_full["sym"][m_te].astype(np.int8),
        "date": test_full["date"][m_te].astype(np.int16),
        "session": np.array([sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object),
        "t": test_full["t"][m_te].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        "pred_label": te_pred,
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    return df, booster, {
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "n_train_used": int(len(X_tr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-json", required=True,
                    help="Path to JSON file: list of variant configs (each with name, kind, ...)")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--feature-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument("--default-seed", type=int, default=42)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--save-models", action="store_true",
                    help="save booster models per fold (for 5-seed ensemble later)")
    ap.add_argument("--save-preds", action="store_true",
                    help="save OOF preds parquets per fold")
    ap.add_argument("--out-tag", default="phase1")
    args = ap.parse_args()

    with open(args.variants_json) as f:
        variants = json.load(f)
    print(f"=== T31 run_variants: {len(variants)} variants, h={H} ===", flush=True)
    progress("loading_data", n_variants=len(variants))

    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [ln.strip() for ln in f if ln.strip()]
    assert len(feat_names) == train_full["X"].shape[1] - N_DROP_TAIL
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features in feat_names: {leak}"
    group_indices = build_group_indices(feat_names)
    print(f"  loaded data in {time.time() - t0:.1f}s; "
          f"train={train_full['X'].shape}; "
          f"groups={len(group_indices)} ({sum(len(v) for v in group_indices.values())} feats)",
          flush=True)

    summary_rows = []
    summary_path = os.path.join(HERE, f"summary_{args.out_tag}.json")
    csv_path = os.path.join(HERE, f"summary_{args.out_tag}.csv")

    for vi, vcfg in enumerate(variants):
        vname = vcfg["name"]
        seed = int(vcfg.get("seed", args.default_seed))
        print(f"\n{'='*70}\n[{vi+1}/{len(variants)}] variant: {vname} (seed={seed})\n"
              f"  kind={vcfg.get('kind')}  cfg={vcfg}\n{'='*70}", flush=True)
        progress("training", variant_index=vi+1, n_variants=len(variants),
                 variant_name=vname)

        v_t0 = time.time()
        fold_dfs = {}
        per_fold_meta = []
        for held in SYMS:
            print(f"  fold held={held} ...", flush=True)
            df, booster, meta = train_one_fold(
                vcfg, held, train_full, val_full, test_full,
                group_indices, feat_names, args, seed,
            )
            fold_dfs[held] = df
            per_fold_meta.append({**meta, "held": held})
            if args.save_preds:
                pred_path = os.path.join(HERE, f"pred_{vname}_held{held}.parquet")
                df.to_parquet(pred_path, index=False)
            if args.save_models:
                model_path = os.path.join(HERE, f"model_{vname}_held{held}.txt")
                booster.save_model(model_path, num_iteration=booster.best_iteration)
            del booster
            print(f"    best_iter={meta['best_iter']} train_time={meta['train_time_sec']:.1f}s "
                  f"n_train_used={meta['n_train_used']:,}", flush=True)

        # Threshold sweep
        best_thr = thresh_sweep(fold_dfs)

        # Raw argmax sum (for ref)
        raw_pf = []
        for k, df in fold_dfs.items():
            pred = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32).argmax(1).astype(np.int8)
            m = _per_horizon_metrics(
                pred, df["true_label"].to_numpy(np.int64),
                df["midprice_t"].to_numpy(np.float32),
                df["midprice_th"].to_numpy(np.float32),
                fee_rate=0.0001,
            )
            raw_pf.append(float(m["cum_pnl"]))
        raw_sum = sum(raw_pf)

        v_elapsed = time.time() - v_t0
        row = {
            "variant_name": vname,
            "config": vcfg,
            "seed": seed,
            "horizon": H,
            "raw_argmax_sum_pnl": raw_sum,
            "raw_argmax_per_fold": raw_pf,
            "best_T": best_thr["T"],
            "best_delta": best_thr["delta"],
            "best_sum_pnl": best_thr["sum_pnl"],
            "best_n_pos_folds": best_thr["n_pos"],
            "best_per_fold": best_thr["per_fold"],
            "best_n_active": best_thr["n_active_total"],
            "per_fold_meta": per_fold_meta,
            "elapsed_sec": v_elapsed,
        }
        summary_rows.append(row)
        print(f"\n  -> {vname}: raw_sum={raw_sum:+.4f}  "
              f"BEST(T={best_thr['T']}, d={best_thr['delta']}) sum_pnl={best_thr['sum_pnl']:+.4f}  "
              f"per_fold={[f'{x:+.2f}' for x in best_thr['per_fold']]}  "
              f"npos={best_thr['n_pos']}/5  elapsed={v_elapsed:.0f}s", flush=True)

        # Persist incrementally
        with open(summary_path, "w") as f:
            json.dump({"variants": summary_rows, "elapsed_total_sec": time.time() - t0}, f, indent=2)
        with open(csv_path, "w") as f:
            f.write("variant_name,raw_sum_pnl,best_T,best_delta,best_sum_pnl,best_n_pos,"
                    "fold0,fold1,fold2,fold3,fold4,n_active,elapsed_sec\n")
            for r in summary_rows:
                pf = r["best_per_fold"]
                f.write(f"{r['variant_name']},{r['raw_argmax_sum_pnl']:.4f},"
                        f"{r['best_T']},{r['best_delta']},"
                        f"{r['best_sum_pnl']:.4f},{r['best_n_pos_folds']},"
                        f"{pf[0]:.4f},{pf[1]:.4f},{pf[2]:.4f},{pf[3]:.4f},{pf[4]:.4f},"
                        f"{r['best_n_active']},{r['elapsed_sec']:.0f}\n")
        progress("variant_done", variant_index=vi+1, last=vname,
                 best_sum_pnl=best_thr["sum_pnl"])

    print(f"\nALL DONE in {time.time() - t0:.1f}s. summary -> {summary_path}", flush=True)
    progress("done", n_variants=len(variants), elapsed_sec=time.time() - t0)


if __name__ == "__main__":
    main()
