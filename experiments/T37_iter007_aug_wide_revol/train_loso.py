"""T37 iter_007 candidate: 5-seed LOSO LightGBM with aug_a [0.75, 1.25] for h_60.

Two experiments share data loading:

  Step A: 223-d Scheme C features (matches iter_005b/iter_006), aug uniform[0.75, 1.25]
  Step B: 224-d (Step A features + revol_wmp1_sigma_hat from T35 cache),
          same aug uniform[0.75, 1.25]

For each (step, seed, held_sym):
    train: (sym != held, dates 0..79)   + per-feature uniform[0.75, 1.25] scale
    val:   (sym != held, dates 80..95)  -- early stopping (NO augment)
    test:  (sym == held, dates 96..119) -- held-out sym (NO augment)

Per-seed hyperparams from T25/T27 SEED_CONFIGS for diversity.
GPU LightGBM (RTX 3090).

Outputs (per (step, seed, held)):
    {step}_pred_seed{S}_held{K}.parquet
    {step}_model_seed{S}_held{K}.txt

Compliance with CRITICAL_CONSTRAINTS:
  * sym/date NEVER in feature vector (only used for fold slicing)
  * Augment is per-sample, stateless, only at training time
  * sym-agnostic model (LightGBM on raw feature columns)
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
from src.eval.pnl import _per_horizon_metrics, sanitize_for_json  # noqa: E402

# Reuse T26 build_aug helpers
T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import (  # noqa: E402
    aug_a_scale, class_balanced_weight, compute_feat_stats,
)

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
T35_CACHE = os.path.join(ROOT, "experiments", "T35_revol_savgol", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # drop trailing time-encoding cols for 226 -> 223
H = 60

# T27 SEED_CONFIGS (diversity of feature_fraction, bagging, num_leaves, lambda_l2)
SEED_CONFIGS = {
    42:  dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=1.0),
    1:   dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=1.0),
    7:   dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=2.0),
    13:  dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=0.5),
    100: dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=3.0),
}

# revol_wmp1_sigma_hat is at column index 11 in schemeK_extra cache
REVOL_WMP1_SIGMA_HAT_IDX = 11


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
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def load_extra_split(split: str) -> np.ndarray:
    """Return revol_wmp1_sigma_hat column for split (aligned with schemeC)."""
    p = os.path.join(T35_CACHE, f"schemeK_extra_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return d["X_extra"][:, REVOL_WMP1_SIGMA_HAT_IDX].astype(np.float32, copy=False)


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s : s + batch]).astype(np.float32))
    return np.concatenate(chunks, axis=0)


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
    """Per-(sample, feature) uniform[lo, hi] scale; concatenate orig + aug 1:1."""
    scales = rng.uniform(lo, hi, size=X_orig.shape).astype(X_orig.dtype)
    X_aug = X_orig * scales
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_orig], axis=0).astype(np.int64)
    w_merged = class_balanced_weight(y_merged, num_class)
    return X_merged, y_merged, w_merged


def train_one_fold(step_tag, seed, held, X_tr_full, y_tr_full, sym_tr,
                   X_va_full, y_va_full, sym_va,
                   X_te_full, y_te_full, sym_te,
                   mp_t_te_full, mp_th_te_full,
                   sess_te_full, date_te_full, t_te_full,
                   feat_names, args, cfg):
    m_tr = sym_tr != held
    m_va = sym_va != held
    m_te = sym_te == held

    X_tr_o = X_tr_full[m_tr]
    y_tr_o = y_tr_full[m_tr].astype(np.int64)
    X_va = X_va_full[m_va]
    y_va = y_va_full[m_va].astype(np.int64)
    X_te = X_te_full[m_te]
    y_te = y_te_full[m_te].astype(np.int64)
    mp_t_te = mp_t_te_full[m_te]
    mp_th_te = mp_th_te_full[m_te]

    # aug uniform [lo, hi]
    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
    X_tr, y_tr, sw_tr = aug_uniform_concat(
        X_tr_o, y_tr_o, args.aug_lo, args.aug_hi, fold_rng, NUM_CLASS,
    )
    sw_va = class_balanced_weight(y_va, NUM_CLASS)
    print(
        f"    n_train_orig={len(X_tr_o):,} n_train_used={len(X_tr):,} "
        f"n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]}",
        flush=True,
    )

    dtrain = lgb.Dataset(
        X_tr, label=y_tr, weight=sw_tr,
        feature_name=feat_names, free_raw_data=True,
    )
    dval = lgb.Dataset(
        X_va, label=y_va, weight=sw_va,
        feature_name=feat_names, reference=dtrain, free_raw_data=True,
    )

    params = {
        "objective": "multiclass", "num_class": NUM_CLASS,
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
        params.update({"device": "gpu", "gpu_use_dp": False})

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
        ],
    )
    train_time = time.time() - t_start
    print(
        f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
        flush=True,
    )

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate_metrics(
        f"HELD-OUT TEST {step_tag} seed={seed} held={held}",
        te_pred, y_te, mp_t_te, mp_th_te,
    )

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": sym_te[m_te].astype(np.int8),
        "date": date_te_full[m_te].astype(np.int16),
        "session": np.array(
            [sess_map[s] for s in sess_te_full[m_te]], dtype=object
        ),
        "t": t_te_full[m_te].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        "pred_label": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(
        HERE, f"{step_tag}_pred_seed{seed}_held{held}.parquet"
    )
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(
        HERE, f"{step_tag}_model_seed{seed}_held{held}.txt"
    )
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "step": step_tag, "seed": seed, "held_out_sym": held,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "held_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_single_pnl": float(te_metrics["single_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
    }


def run_step(step_tag, X_tr_full, X_va_full, X_te_full, feat_names,
             train_full, val_full, test_full, seeds, target_syms, args):
    print(f"\n{'#'*78}", flush=True)
    print(f"### STEP {step_tag}: feat_dim={X_tr_full.shape[1]}", flush=True)
    print(f"### {'#'*70}", flush=True)
    all_results = []
    n_total = len(seeds) * len(target_syms)
    done = 0
    sym_tr = train_full["sym"]
    sym_va = val_full["sym"]
    sym_te = test_full["sym"]
    y_tr = train_full[f"y{H}"]
    y_va = val_full[f"y{H}"]
    y_te = test_full[f"y{H}"]
    mp_t_te = test_full["mp_t"]
    mp_th_te = test_full[f"mp_t{H}"]
    sess_te = test_full["sess_idx"]
    date_te = test_full["date"]
    t_te = test_full["t"]

    for s in seeds:
        cfg = SEED_CONFIGS[s]
        print(f"\n{'='*78}\n=== {step_tag} SEED {s} cfg={cfg} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            print(f"\n  [{done}/{n_total}] {step_tag} seed={s} held={held}", flush=True)
            progress(f"training_{step_tag}",
                     step_tag=step_tag, seed=s, held=held,
                     done=done, total=n_total)
            r = train_one_fold(
                step_tag, s, held,
                X_tr_full, y_tr, sym_tr,
                X_va_full, y_va, sym_va,
                X_te_full, y_te, sym_te,
                mp_t_te, mp_th_te,
                sess_te, date_te, t_te,
                feat_names, args, cfg,
            )
            all_results.append(r)

    # per-seed aggregate
    df = pd.DataFrame(all_results)
    print(f"\n{'='*78}\n=== {step_tag} AGGREGATE h={H} (raw argmax per fold) ===\n{'='*78}", flush=True)
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
    return all_results, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="A,B",
                    help="which steps to run, comma list of A and/or B")
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--aug-lo", type=float, default=0.75)
    ap.add_argument("--aug-hi", type=float, default=1.25)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = ap.parse_args()

    target_syms = [int(x) for x in args.syms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"unknown seed {s}, valid={list(SEED_CONFIGS)}")
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    for st in steps:
        if st not in ("A", "B"):
            sys.exit(f"unknown step {st}, valid=A,B")

    print(f"=== T37 iter_007 candidate: steps={steps} seeds={seeds} "
          f"syms={target_syms} aug=[{args.aug_lo},{args.aug_hi}] ===", flush=True)

    progress("loading_caches", steps=steps, seeds=seeds)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )

    # 226 -> 223
    X_tr_223 = train_full["X"][:, :-N_DROP_TAIL].astype(np.float32, copy=False)
    X_va_223 = val_full["X"][:, :-N_DROP_TAIL].astype(np.float32, copy=False)
    X_te_223 = test_full["X"][:, :-N_DROP_TAIL].astype(np.float32, copy=False)

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names_223 = [ln.strip() for ln in f if ln.strip()]
    assert len(feat_names_223) == X_tr_223.shape[1]
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names_223)
    assert not leak, f"FORBIDDEN features: {leak}"

    # Step A inputs
    feat_names_A = feat_names_223
    X_tr_A, X_va_A, X_te_A = X_tr_223, X_va_223, X_te_223

    # Step B inputs (lazy load extra)
    feat_names_B = None
    X_tr_B = X_va_B = X_te_B = None
    if "B" in steps:
        rev_tr = load_extra_split("train").reshape(-1, 1)
        rev_va = load_extra_split("val").reshape(-1, 1)
        rev_te = load_extra_split("test").reshape(-1, 1)
        feat_names_B = feat_names_223 + ["revol_wmp1_sigma_hat"]
        X_tr_B = np.concatenate([X_tr_223, rev_tr], axis=1).astype(np.float32, copy=False)
        X_va_B = np.concatenate([X_va_223, rev_va], axis=1).astype(np.float32, copy=False)
        X_te_B = np.concatenate([X_te_223, rev_te], axis=1).astype(np.float32, copy=False)
        print(f"  Step B feat_dim={X_tr_B.shape[1]} (added revol_wmp1_sigma_hat)", flush=True)
        leak_B = forbidden & set(feat_names_B)
        assert not leak_B, f"FORBIDDEN features in B: {leak_B}"

    out = {
        "task": "T37 iter_007 candidate (5-seed aug[0.75,1.25] LOSO h_60, +/- revol_wmp1_sigma_hat)",
        "horizon": H,
        "seeds": seeds,
        "syms": target_syms,
        "aug_lo": args.aug_lo, "aug_hi": args.aug_hi,
        "params": vars(args),
        "results_by_step": {},
        "agg_by_step": {},
    }

    if "A" in steps:
        r_a, a_a = run_step(
            "stepA", X_tr_A, X_va_A, X_te_A, feat_names_A,
            train_full, val_full, test_full, seeds, target_syms, args,
        )
        out["results_by_step"]["A"] = r_a
        out["agg_by_step"]["A"] = a_a
        progress("stepA_done", n_results=len(r_a))
        # incremental save
        with open(os.path.join(HERE, "results.json"), "w") as f:
            json.dump(sanitize_for_json({**out, "elapsed_sec": time.time() - t0}), f, indent=2)

    if "B" in steps:
        r_b, a_b = run_step(
            "stepB", X_tr_B, X_va_B, X_te_B, feat_names_B,
            train_full, val_full, test_full, seeds, target_syms, args,
        )
        out["results_by_step"]["B"] = r_b
        out["agg_by_step"]["B"] = a_b
        progress("stepB_done", n_results=len(r_b))
        with open(os.path.join(HERE, "results.json"), "w") as f:
            json.dump(sanitize_for_json({**out, "elapsed_sec": time.time() - t0}), f, indent=2)

    out["elapsed_sec"] = time.time() - t0
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(sanitize_for_json(out), f, indent=2)
    print(f"\nALL DONE in {out['elapsed_sec']:.1f}s. summary -> {HERE}/results.json", flush=True)
    progress("loso_done", elapsed_sec=out["elapsed_sec"])


if __name__ == "__main__":
    main()
