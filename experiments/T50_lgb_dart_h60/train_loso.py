"""T50: LightGBM DART boosting LOSO Scheme C 223-d, h_60, 5 seeds x 5 folds, aug_a [0.80, 1.20].

Single-idea ablation: ONLY swap GBDT -> DART vs iter_005b (T27/T28). Everything
else (data, features, aug, fold split, seeds) matches iter_006 exactly.

DART = Dropouts meet Multiple Additive Regression Trees. Each new tree must
compensate for a randomly dropped subset of existing trees -> forces the model
to learn redundant / general patterns -> theoretically reduces OOD overfit.
R32+R35 expectation: +1~+4 LOSO over GBDT.

For each (seed, held_out_sym):
    train: (sym != held, dates 0..79)   + per-(sample,feat) U[0.80,1.20] scale
    val:   (sym != held, dates 80..95)  -- (NO aug, NO early stopping in DART)
    test:  (sym == held, dates 96..119) -- held-out predictions (NO aug)

Outputs (per (seed, fold)):
    dart_pred_seed{S}_held{K}.parquet
    dart_model_seed{S}_held{K}.txt

Compliance with CRITICAL_CONSTRAINTS:
  * sym/date NEVER in feature vector (only used for fold slicing)
  * Per-sample stateless aug, only at training time
  * sym-agnostic model (LightGBM on raw 223-d feature columns)
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

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
sys.path.insert(0, T26_DIR)
from build_aug import class_balanced_weight  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
SYMS = (0, 1, 2, 3, 4)
N_DROP_TAIL = 3  # 226 -> 223, drop trailing time-encoding cols (matches iter_005b)
H = 60

# Per-seed diversity (mirrors T27/T37 SEED_CONFIGS)
SEED_CONFIGS = {
    42:  dict(feature_fraction=0.8, bagging_fraction=0.8,  num_leaves=127, lambda_l2=2.0),
    1:   dict(feature_fraction=0.6, bagging_fraction=0.7,  num_leaves=127, lambda_l2=2.0),
    7:   dict(feature_fraction=0.7, bagging_fraction=0.85, num_leaves=63,  lambda_l2=3.0),
    13:  dict(feature_fraction=0.5, bagging_fraction=0.6,  num_leaves=255, lambda_l2=1.0),
    100: dict(feature_fraction=0.4, bagging_fraction=0.5,  num_leaves=127, lambda_l2=4.0),
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
    p = os.path.join(T5B_CACHE, f"schemeC_{split}.npz")
    print(f"  loading {p} ...", flush=True)
    d = np.load(p)
    return {k: d[k] for k in d.files}


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000):
    """For DART, predict at the final iter (full ensemble, no dropout at inference)."""
    chunks = []
    for s in range(0, len(X), batch):
        chunks.append(booster.predict(X[s: s + batch]).astype(np.float32))
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
    """Per-(sample, feature) U[lo,hi] scale; concat orig + aug 1:1."""
    scales = rng.uniform(lo, hi, size=X_orig.shape).astype(X_orig.dtype)
    X_aug = X_orig * scales
    X_merged = np.concatenate([X_orig, X_aug], axis=0)
    y_merged = np.concatenate([y_orig, y_orig], axis=0).astype(np.int64)
    w_merged = class_balanced_weight(y_merged, num_class)
    return X_merged, y_merged, w_merged


def train_one_fold(seed, held, X_tr_full, y_tr_full, sym_tr,
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
        # --- DART boosting (the key change from iter_005b) ---
        "boosting_type": "dart",
        "drop_rate": args.drop_rate,
        "max_drop": args.max_drop,
        "skip_drop": args.skip_drop,
        "uniform_drop": False,        # importance-weighted drop
        "xgboost_dart_mode": False,
        # --- Multi-class objective ---
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
        "drop_seed": seed + 4,
        "verbose": -1,
    }
    if args.use_gpu:
        params.update({"device": "gpu", "gpu_use_dp": False})

    t_start = time.time()
    # DART does NOT support early stopping -> fixed num_boost_round
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.log_evaluation(period=300)],
    )
    train_time = time.time() - t_start
    # DART has no best_iteration -> use full model
    final_iter = booster.current_iteration()
    print(
        f"    trained in {train_time:.1f}s, final_iter={final_iter}",
        flush=True,
    )

    te_prob = predict_proba(booster, X_te)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate_metrics(
        f"HELD-OUT TEST dart seed={seed} held={held}",
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
    pred_path = os.path.join(HERE, f"dart_pred_seed{seed}_held{held}.parquet")
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(HERE, f"dart_model_seed{seed}_held{held}.txt")
    booster.save_model(model_path)  # full model, no num_iteration cutoff (DART)

    return {
        "tag": "dart", "seed": seed, "held_out_sym": held,
        "n_train_orig": int(len(X_tr_o)),
        "n_train_used": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "final_iter": int(final_iter),
        "train_time_sec": float(train_time),
        "held_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_single_pnl": float(te_metrics["single_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--aug-lo", type=float, default=0.80)
    ap.add_argument("--aug-hi", type=float, default=1.20)
    ap.add_argument("--num-boost-round", type=int, default=1500,
                    help="DART has no early stopping; fix num_iter (PM-spec).")
    ap.add_argument("--learning-rate", type=float, default=0.04,
                    help="DART converges slower -> 0.04 (vs 0.05 GBDT).")
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=16)
    # DART-specific
    ap.add_argument("--drop-rate", type=float, default=0.10)
    ap.add_argument("--max-drop", type=int, default=50)
    ap.add_argument("--skip-drop", type=float, default=0.50)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    target_syms = [int(x) for x in args.syms.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    for s in seeds:
        if s not in SEED_CONFIGS:
            sys.exit(f"unknown seed {s}, valid={list(SEED_CONFIGS)}")

    print(
        f"=== T50 LightGBM DART h_{H} LOSO: seeds={seeds} syms={target_syms} "
        f"aug=[{args.aug_lo},{args.aug_hi}] num_iter={args.num_boost_round} "
        f"drop_rate={args.drop_rate} skip_drop={args.skip_drop} ===",
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

    # 226 -> 223 (drop trailing time-encoding cols, matches iter_005b)
    X_tr_223 = train_full["X"][:, :-N_DROP_TAIL].astype(np.float32, copy=False)
    X_va_223 = val_full["X"][:, :-N_DROP_TAIL].astype(np.float32, copy=False)
    X_te_223 = test_full["X"][:, :-N_DROP_TAIL].astype(np.float32, copy=False)

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [ln.strip() for ln in f if ln.strip()]
    assert len(feat_names) == X_tr_223.shape[1], (
        f"feat_names {len(feat_names)} != X cols {X_tr_223.shape[1]}"
    )
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features in feat_names: {leak}"
    print(f"  feat_dim={len(feat_names)}, no date/sym/time. OK.", flush=True)

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

    # WandB
    use_wandb = not args.no_wandb
    wandb = None
    if use_wandb:
        try:
            import wandb as wandb_mod
            wandb = wandb_mod
            run_name = (
                f"T50-lgb-dart-h{H}-seeds{','.join(str(s) for s in seeds)}-"
                f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            init_kwargs = dict(
                project="liangwenbei",
                name=run_name,
                config={
                    "model": "lightgbm-dart",
                    "scheme": "C-223d",
                    "horizon": H,
                    "seeds": seeds,
                    "aug_lo": args.aug_lo, "aug_hi": args.aug_hi,
                    **{k: v for k, v in vars(args).items() if k != "no_wandb"},
                },
                tags=["T50", "lightgbm", "dart", "schemeC", "223d", f"h{H}"],
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
        cfg = SEED_CONFIGS[s]
        print(f"\n{'='*78}\n=== SEED {s} (DART) cfg={cfg} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            print(f"\n  [{done}/{n_total}] tag=dart seed={s} held={held}", flush=True)
            progress("training", seed=s, held=held,
                     done=done, total=n_total,
                     elapsed_min=round((time.time() - t0) / 60.0, 2))
            r = train_one_fold(
                s, held,
                X_tr_223, y_tr, sym_tr,
                X_va_223, y_va, sym_va,
                X_te_223, y_te, sym_te,
                mp_t_te, mp_th_te,
                sess_te, date_te, t_te,
                feat_names, args, cfg,
            )
            all_results.append(r)
            if use_wandb:
                wandb.log({
                    f"seed{s}/fold{held}/cum_pnl": r["held_cum_pnl"],
                    f"seed{s}/fold{held}/acc": r["held_acc"],
                    f"seed{s}/fold{held}/final_iter": r["final_iter"],
                    f"seed{s}/fold{held}/train_time_sec": r["train_time_sec"],
                    f"seed{s}/fold{held}/n_active": r["held_n_active"],
                })
            interim = {
                "task": f"T50 LightGBM DART h_{H} LOSO",
                "horizon": H,
                "seeds": seeds,
                "syms": target_syms,
                "params": {k: v for k, v in vars(args).items() if k != "no_wandb"},
                "results": all_results,
                "elapsed_sec": time.time() - t0,
            }
            with open(os.path.join(HERE, "results.json"), "w") as f:
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

    overall_sum = float(df["held_cum_pnl"].sum())
    print(f"\n  OVERALL argmax sum across all 25 folds: {overall_sum:+.4f}", flush=True)

    out = {
        "task": f"T50 LightGBM DART h_{H} LOSO",
        "horizon": H,
        "seeds": seeds,
        "syms": target_syms,
        "boosting_type": "dart",
        "params": {k: v for k, v in vars(args).items() if k != "no_wandb"},
        "results": all_results,
        "agg_by_seed": agg,
        "overall_argmax_sum_cum_pnl": overall_sum,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(sanitize_for_json(out), f, indent=2)
    print(f"\nALL DONE in {out['elapsed_sec']:.1f}s. -> results.json", flush=True)
    progress("loso_done", elapsed_sec=out["elapsed_sec"], overall_argmax_sum=overall_sum)

    if use_wandb:
        for s in seeds:
            sub = df[df.seed == s]
            wandb.summary[f"seed{s}_argmax_sum_cum_pnl"] = float(sub["held_cum_pnl"].sum())
        wandb.summary["overall_argmax_sum_cum_pnl"] = overall_sum
        wandb.finish()


if __name__ == "__main__":
    main()
