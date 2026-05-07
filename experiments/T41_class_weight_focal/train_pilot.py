"""T41 pilot: compare class-weight schemes vs focal loss for h_60 LightGBM.

Each method × seed=42 × 5 LOSO folds. Trains the same architecture as T26/T27
aug_a (5-seed config[42]). The only thing that changes per method is the
sample weight (or the objective for focal).

Baseline ('balanced') = inv-frequency weights — already what iter_006 uses.
Other methods explore tighter / looser rebalancing and focal loss.

Outputs (per method, per held):
  loso_pred_h60_<method>_seed42_held<k>.parquet
  loso_model_h60_<method>_seed42_held<k>.txt

After training, run sweep_thresh.py to DE-search asymmetric thresholds on the
OOF concat for each method and pick the best sum_cum_pnl.
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
from build_aug import (  # noqa: E402
    aug_a_scale,
    compute_feat_stats,
)

sys.path.insert(0, HERE)
from focal_obj import MulticlassFocalObjective  # noqa: E402

T5B_CACHE = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache")
NUM_CLASS = 3
N_DROP_TAIL = 3
H = 60
SEED = 42

# T26/T27 seed=42 config (matches existing baseline iter_005b).
SEED42_CFG = dict(seed=42, feature_fraction=0.8, bagging_fraction=0.8,
                  num_leaves=127, lambda_l2=1.0)


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


def predict_proba(booster: lgb.Booster, X: np.ndarray, batch: int = 200_000,
                  is_focal: bool = False, num_class: int = NUM_CLASS):
    """Softmax-probability predictions. For focal (custom obj) the booster
    returns raw scores (n, K); we softmax here. For built-in multiclass the
    booster.predict already returns probs.
    """
    chunks = []
    for s in range(0, len(X), batch):
        out = booster.predict(X[s : s + batch])
        if is_focal:
            z = np.asarray(out, dtype=np.float64)
            if z.ndim == 1:
                z = z.reshape(num_class, -1).T
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            chunks.append(p.astype(np.float32))
        else:
            chunks.append(out.astype(np.float32))
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


def fixed_class_weights(y: np.ndarray, w_per_class: list, num_class: int = NUM_CLASS):
    cw = np.asarray(w_per_class, dtype=np.float64)
    assert cw.shape == (num_class,)
    return cw[y.astype(np.int64)].astype(np.float32)


def aug_a_for_train(X, y, feat_std, feat_mean, rng, aug_ratio=1.0):
    """Same aug_a augmentation pattern as T26 but without class-balanced weight
    (we'll attach the weight outside)."""
    n_aug = int(round(aug_ratio * len(X)))
    if n_aug <= 0:
        return X, y
    if n_aug == len(X):
        idx = np.arange(len(X))
    else:
        idx = rng.integers(0, len(X), size=n_aug)
    X_src = X[idx]
    y_src = y[idx]
    X_aug = aug_a_scale(X_src, rng)
    X_merged = np.concatenate([X, X_aug], axis=0)
    y_merged = np.concatenate([y, y_src], axis=0).astype(np.int64)
    return X_merged, y_merged


def train_one(method: str, held: int, train_full, val_full, test_full,
              feat_names, args):
    cfg = SEED42_CFG
    seed = SEED
    m_tr = train_full["sym"] != held
    m_va = val_full["sym"] != held
    m_te = test_full["sym"] == held

    X_tr_o = train_full["X"][m_tr][:, :-N_DROP_TAIL].astype(np.float32)
    y_tr_o = train_full[f"y{H}"][m_tr].astype(np.int64)
    X_va = val_full["X"][m_va][:, :-N_DROP_TAIL].astype(np.float32)
    y_va = val_full[f"y{H}"][m_va].astype(np.int64)
    mp_t_va = val_full["mp_t"][m_va]
    mp_th_va = val_full[f"mp_t{H}"][m_va]
    X_te = test_full["X"][m_te][:, :-N_DROP_TAIL].astype(np.float32)
    y_te = test_full[f"y{H}"][m_te].astype(np.int64)
    mp_t_te = test_full["mp_t"][m_te]
    mp_th_te = test_full[f"mp_t{H}"][m_te]

    feat_mean, feat_std = compute_feat_stats(X_tr_o)
    fold_rng = np.random.default_rng(seed * 7919 + held * 17 + 1)
    X_tr, y_tr = aug_a_for_train(X_tr_o, y_tr_o, feat_std, feat_mean, fold_rng,
                                  aug_ratio=args.aug_ratio)

    # Decide weight & objective per method.
    is_focal = method == "focal"
    if method == "balanced":
        # inv-freq on merged y
        counts = np.bincount(y_tr.astype(np.int64), minlength=NUM_CLASS).astype(np.float64)
        counts = np.where(counts == 0, 1.0, counts)
        cw = len(y_tr) / (NUM_CLASS * counts)
        sw_tr = cw[y_tr.astype(np.int64)].astype(np.float32)
        sw_va = (len(y_va) / (NUM_CLASS * np.maximum(np.bincount(y_va, minlength=NUM_CLASS), 1)))[y_va].astype(np.float32)
    elif method == "uniform":
        sw_tr = np.ones(len(y_tr), dtype=np.float32)
        sw_va = np.ones(len(y_va), dtype=np.float32)
    elif method.startswith("ratio_"):
        # method e.g. ratio_15_10_15  → [1.5, 1.0, 1.5]; ratio_30_10_30 → [3.0, 1.0, 3.0]
        parts = method.split("_")[1:]
        assert len(parts) == NUM_CLASS, f"ratio method needs {NUM_CLASS} parts, got {parts}"
        cw_list = [float(int(p)) / 10.0 for p in parts]
        sw_tr = fixed_class_weights(y_tr, cw_list)
        sw_va = fixed_class_weights(y_va, cw_list)
    elif method == "focal":
        # custom obj handles class imbalance via focal modulator; weight=1
        sw_tr = np.ones(len(y_tr), dtype=np.float32)
        sw_va = np.ones(len(y_va), dtype=np.float32)
    else:
        raise ValueError(f"unknown method {method}")

    sw_summary = {int(c): float(sw_tr[y_tr == c].mean()) if (y_tr == c).any() else 0.0
                  for c in range(NUM_CLASS)}
    print(
        f"    [{method}] n_train_orig={len(X_tr_o):,} n_train_used={len(X_tr):,} "
        f"n_val={len(X_va):,} n_test={len(X_te):,} feat_dim={X_tr.shape[1]} "
        f"sw_per_class={sw_summary}",
        flush=True,
    )

    dtrain = lgb.Dataset(
        X_tr, label=y_tr, weight=sw_tr,
        feature_name=feat_names, free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_va, label=y_va, weight=sw_va,
        feature_name=feat_names, reference=dtrain, free_raw_data=False,
    )

    params = {
        "num_class": NUM_CLASS,
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
    feval = None
    if is_focal:
        fobj = MulticlassFocalObjective(num_class=NUM_CLASS, alpha=args.focal_alpha,
                                         gamma=args.focal_gamma)
        params["objective"] = fobj
        params["metric"] = "None"

        def focal_eval(preds, eval_data):
            K = NUM_CLASS
            yv = eval_data.get_label().astype(np.int64)
            n = yv.shape[0]
            z = np.asarray(preds, dtype=np.float64)
            if z.ndim == 1:
                z = z.reshape(n, K)
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            eps = 1e-7
            p = np.clip(p, eps, 1.0 - eps)
            p_y = p[np.arange(n), yv]
            loss = -args.focal_alpha * (1.0 - p_y) ** args.focal_gamma * np.log(p_y)
            return ("focal_loss", float(loss.mean()), False)

        feval = focal_eval
    else:
        params["objective"] = "multiclass"
        params["metric"] = "multi_logloss"
        if args.use_gpu:
            params.update({"device": "gpu", "gpu_use_dp": False})

    t_start = time.time()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        feval=feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    train_time = time.time() - t_start
    print(
        f"    trained in {train_time:.1f}s, best_iter={booster.best_iteration}",
        flush=True,
    )

    te_prob = predict_proba(booster, X_te, is_focal=is_focal)
    te_pred = te_prob.argmax(axis=1).astype(np.int8)
    te_metrics = evaluate(
        f"HELD-OUT TEST h={H} method={method} held={held}",
        te_pred, y_te, mp_t_te, mp_th_te,
    )

    sess_map = {0: "am", 1: "pm"}
    te_df = pd.DataFrame({
        "sym": test_full["sym"][m_te].astype(np.int8),
        "date": test_full["date"][m_te].astype(np.int16),
        "session": np.array(
            [sess_map[s] for s in test_full["sess_idx"][m_te]], dtype=object
        ),
        "t": test_full["t"][m_te].astype(np.int16),
        "true_label": y_te.astype(np.int8),
        "pred_label": te_pred.astype(np.int8),
        "prob_0": te_prob[:, 0].astype(np.float32),
        "prob_1": te_prob[:, 1].astype(np.float32),
        "prob_2": te_prob[:, 2].astype(np.float32),
        "midprice_t": mp_t_te.astype(np.float32),
        "midprice_th": mp_th_te.astype(np.float32),
    })
    pred_path = os.path.join(
        HERE, f"loso_pred_h{H}_{method}_seed{seed}_held{held}.parquet"
    )
    te_df.to_parquet(pred_path, index=False)

    model_path = os.path.join(
        HERE, f"loso_model_h{H}_{method}_seed{seed}_held{held}.txt"
    )
    booster.save_model(model_path, num_iteration=booster.best_iteration)

    return {
        "method": method, "held_out_sym": held,
        "n_train_used": int(len(X_tr)),
        "best_iter": int(booster.best_iteration),
        "train_time_sec": float(train_time),
        "held_cum_pnl": float(te_metrics["cum_pnl"]),
        "held_acc": float(te_metrics["accuracy"]),
        "held_n_active": int(te_metrics["n_predictions_active"]),
        "held_f0_5_macro": float(te_metrics["f0_5_macro"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="uniform,ratio_15_10_15,ratio_20_10_20,ratio_30_10_30,focal",
                    help="comma list, ratio_<i>_<j>_<k> means class weights [i/10, j/10, k/10]")
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--aug-ratio", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument("--early-stopping", type=int, default=40)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=100)
    ap.add_argument("--bagging-freq", type=int, default=5)
    ap.add_argument("--num-threads", type=int, default=18)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    target_syms = [int(x) for x in args.syms.split(",")]

    print(f"=== T41 pilot methods={methods} seed={SEED} h={H} ===", flush=True)

    progress("loading_caches", methods=methods)
    t0 = time.time()
    train_full = load_split("train")
    val_full = load_split("val")
    test_full = load_split("test")
    print(
        f"loaded in {time.time() - t0:.1f}s; "
        f"train={train_full['X'].shape} val={val_full['X'].shape} test={test_full['X'].shape}",
        flush=True,
    )

    feat_names_path = os.path.join(T5B_CACHE, "schemeC_223d_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [line.strip() for line in f]
    assert len(feat_names) == train_full["X"].shape[1] - N_DROP_TAIL
    forbidden = {"date", "sym", "time"}
    leak = forbidden & set(feat_names)
    assert not leak, f"FORBIDDEN features: {leak}"

    all_results = []
    n_total = len(methods) * len(target_syms)
    done = 0
    for method in methods:
        print(f"\n{'='*78}\n=== METHOD {method} ===\n{'='*78}", flush=True)
        for held in target_syms:
            done += 1
            tag = f"{method}_held{held}"
            print(f"\n  [{done}/{n_total}] fold {tag}", flush=True)
            progress("training", method=method, held=held,
                     done=done, total=n_total)
            r = train_one(method, held, train_full, val_full, test_full,
                          feat_names, args)
            all_results.append(r)

    df = pd.DataFrame(all_results)
    print(f"\n{'='*78}\n=== AGGREGATE per-method (raw argmax, no thresh) ===\n{'='*78}",
          flush=True)
    agg = []
    for method in methods:
        sub = df[df.method == method]
        cums = sub["held_cum_pnl"].values
        rec = {
            "method": method,
            "n_folds": int(len(cums)),
            "cum_pnl_per_fold": cums.tolist(),
            "cum_pnl_sum_argmax": float(cums.sum()),
            "cum_pnl_mean_argmax": float(cums.mean()) if len(cums) else 0.0,
            "n_pos_folds_argmax": int((cums > 0).sum()),
            "n_active_total": int(sub["held_n_active"].sum()),
            "mean_acc": float(sub["held_acc"].mean()),
            "mean_f0_5_macro": float(sub["held_f0_5_macro"].mean()),
        }
        agg.append(rec)
        print(
            f"  {method:24s}: argmax_sum={rec['cum_pnl_sum_argmax']:+.4f} "
            f"mean={rec['cum_pnl_mean_argmax']:+.4f} pos={rec['n_pos_folds_argmax']}/{rec['n_folds']} "
            f"n_active={rec['n_active_total']:,} mean_acc={rec['mean_acc']:.4f} "
            f"mean_f0.5={rec['mean_f0_5_macro']:.4f}",
            flush=True,
        )

    summary_path = os.path.join(HERE, "pilot_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sanitize_for_json({
            "task": "T41 pilot class_weight_focal seed=42 h=60 (raw argmax)",
            "horizon": H, "seed": SEED, "methods": methods,
            "params": vars(args),
            "results": all_results,
            "aggregate": agg,
            "elapsed_sec": time.time() - t0,
        }), f, indent=2)
    print(f"\nsummary -> {summary_path}", flush=True)

    progress("pilot_done", n_runs=len(all_results), methods=methods)


if __name__ == "__main__":
    main()
