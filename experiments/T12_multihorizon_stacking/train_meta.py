"""T12: Multi-horizon stacking with Ridge / LogisticRegression meta-learner.

Setup:
    Base models (T5b iter_002 Scheme C 223-d, multi-horizon LightGBM) produced
    LOSO OOF predictions:
        loso_pred_h{H}_held{K}.parquet   for H in {5,10,20,40,60}, K in {0..4}
    Each file contains predictions on sym=K only; the underlying booster was
    trained on syms {0..4} \\ {K}.

Stacking:
    For each (sym, date, session, t) we have 5 horizons * 3 classes = 15-d
    base prob features. We train a meta-learner per fold k:
        meta_train rows  = OOF preds on sym != k (4 syms, valid OOF)
        meta_val   rows  = OOF preds on sym == k (1 sym, held-out)
    Target = label_10 (best-paying horizon, iter_002 baseline +21.86 LOSO sum).

    Models swept:
        Ridge classifier (manual one-vs-rest via sklearn Ridge regressor on
                          one-hot, since pure RidgeClassifier doesn't expose
                          calibrated probabilities) -> we use multinomial
                          LogisticRegression with l2 penalty (== L2-Logistic),
                          which is essentially Ridge on log-odds.
        LogisticRegression with l2:  C in {0.5, 1, 2, 5, 10}
        Ridge regressor on one-hot label, then softmax-normalized: alpha in
                          {0.1, 1, 5, 10, 50}

    Selection: per-fold val multi_logloss best (across both families combined).

Outputs:
    meta_oof_h10.parquet                merged across 5 folds (held=sym)
    fold{k}_meta_summary.json
    train_meta.log (printed)
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
T5B = os.path.join(HERE, "..", "T5b_features_multihorizon")
T5B = os.path.abspath(T5B)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

HORIZONS = [5, 10, 20, 40, 60]
KEY_COLS = ["sym", "date", "session", "t"]
LR_C_GRID = [0.5, 1.0, 2.0, 5.0, 10.0]
RIDGE_ALPHA_GRID = [0.1, 1.0, 5.0, 10.0, 50.0]
TARGET_HORIZON = 10


def load_oof_for_fold(k: int) -> pd.DataFrame:
    """Merge h_5/10/20/40/60 OOF predictions for held=k into one DataFrame.

    Result columns:
        KEY_COLS + ['true_label_5', ..., 'true_label_60',
                    'midprice_t', 'midprice_th_5', ..., 'midprice_th_60',
                    'p0_h5', 'p1_h5', 'p2_h5', ..., 'p2_h60']  (15 prob cols)
    """
    base = None
    for H in HORIZONS:
        path = os.path.join(T5B, f"loso_pred_h{H}_held{k}.parquet")
        df = pd.read_parquet(path)
        df = df.rename(columns={
            "true_label": f"true_label_{H}",
            "prob_0": f"p0_h{H}",
            "prob_1": f"p1_h{H}",
            "prob_2": f"p2_h{H}",
            "midprice_th": f"midprice_th_{H}",
        })
        df = df.drop(columns=["pred_label"])
        if base is None:
            base = df
        else:
            # midprice_t is the same across horizons; drop the duplicate
            df = df.drop(columns=["midprice_t"])
            base = base.merge(df, on=KEY_COLS, how="inner")
    return base


def fit_lr(X_tr: np.ndarray, y_tr: np.ndarray, C: float) -> LogisticRegression:
    # sklearn >= 1.5 dropped multi_class arg; lbfgs handles multinomial natively
    m = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs", max_iter=2000, n_jobs=1,
    )
    m.fit(X_tr, y_tr)
    return m


def fit_ridge_softmax(X_tr: np.ndarray, y_tr: np.ndarray,
                      alpha: float) -> Tuple[Ridge, np.ndarray]:
    """Fit Ridge regressor on one-hot label; return (model, classes_seen).

    Predict-time we apply softmax to the regressor outputs to get probs.
    """
    classes = np.array([0, 1, 2], dtype=np.int64)
    Y = np.zeros((len(y_tr), 3), dtype=np.float32)
    Y[np.arange(len(y_tr)), y_tr] = 1.0
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X_tr, Y)
    return m, classes


def predict_ridge_softmax(m: Ridge, X: np.ndarray) -> np.ndarray:
    raw = m.predict(X)  # (N, 3)
    raw = raw - raw.max(axis=1, keepdims=True)
    e = np.exp(raw)
    return e / e.sum(axis=1, keepdims=True)


def evaluate_meta_pnl(probs: np.ndarray, df_val: pd.DataFrame,
                      H: int, T_thr: float, delta: float) -> Dict[str, float]:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T_thr) & (side_max > p1 + delta)
    side_pred = np.where(p2 > p0, 2, 0)
    pred = np.where(take, side_pred, 1).astype(np.int64)
    m = _per_horizon_metrics(
        pred, df_val[f"true_label_{H}"].to_numpy(np.int64),
        df_val["midprice_t"].to_numpy(np.float32),
        df_val[f"midprice_th_{H}"].to_numpy(np.float32),
        fee_rate=0.0001,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "n_active": int(m["n_predictions_active"]),
        "accuracy": float(m["accuracy"]),
    }


def main():
    t0 = time.time()
    # 1) Load all 5 folds, merging horizons
    print("=== Loading OOF for 5 folds ===", flush=True)
    fold_dfs: Dict[int, pd.DataFrame] = {}
    for k in range(5):
        fold_dfs[k] = load_oof_for_fold(k)
        print(f"  fold{k}: {fold_dfs[k].shape} -- syms: {fold_dfs[k].sym.unique()}",
              flush=True)

    # 2) Build full OOF-across-syms (each row predicted by held-{its_sym} model)
    full_df = pd.concat([fold_dfs[k] for k in range(5)], ignore_index=True)
    print(f"\nFull OOF rows: {len(full_df):,}", flush=True)

    feat_cols = [f"p{c}_h{H}" for H in HORIZONS for c in [0, 1, 2]]
    assert len(feat_cols) == 15
    X_all = full_df[feat_cols].to_numpy(np.float32)
    y_all = full_df[f"true_label_{TARGET_HORIZON}"].to_numpy(np.int64)
    sym_all = full_df["sym"].to_numpy(np.int64)
    print(f"X shape: {X_all.shape}, target dist: "
          f"{np.bincount(y_all).tolist()}", flush=True)

    # 3) Per-fold meta train + val
    print("\n=== Per-fold meta-learner training ===", flush=True)
    meta_oof_rows = []
    fold_summaries = []
    for k in range(5):
        train_mask = sym_all != k
        val_mask = sym_all == k
        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_va, y_va = X_all[val_mask], y_all[val_mask]
        df_va = full_df[val_mask].reset_index(drop=True)

        # LR sweep
        results = []
        for C in LR_C_GRID:
            m = fit_lr(X_tr, y_tr, C)
            p_va = m.predict_proba(X_va)
            ll = log_loss(y_va, p_va, labels=[0, 1, 2])
            results.append({"family": "LR", "param": C, "logloss": ll,
                            "model": m, "probs": p_va})
        # Ridge sweep
        for alpha in RIDGE_ALPHA_GRID:
            m, _ = fit_ridge_softmax(X_tr, y_tr, alpha)
            p_va = predict_ridge_softmax(m, X_va)
            ll = log_loss(y_va, p_va, labels=[0, 1, 2])
            results.append({"family": "RidgeSoftmax", "param": alpha,
                            "logloss": ll, "model": m, "probs": p_va})

        results.sort(key=lambda r: r["logloss"])
        best = results[0]
        print(f"\n  fold{k} (val sym={k}): n_tr={len(y_tr):,} n_va={len(y_va):,}",
              flush=True)
        for r in results:
            print(f"    {r['family']:<13s} param={r['param']:>5}  "
                  f"val_logloss={r['logloss']:.4f}", flush=True)
        print(f"  -> best: {best['family']} param={best['param']} "
              f"logloss={best['logloss']:.4f}", flush=True)

        # Save best probs
        df_va["meta_p0"] = best["probs"][:, 0]
        df_va["meta_p1"] = best["probs"][:, 1]
        df_va["meta_p2"] = best["probs"][:, 2]
        df_va["meta_pred"] = best["probs"].argmax(axis=1).astype(np.int64)
        df_va["meta_family"] = best["family"]
        df_va["meta_param"] = best["param"]
        keep = KEY_COLS + [
            f"true_label_{TARGET_HORIZON}", "midprice_t",
            f"midprice_th_{TARGET_HORIZON}",
            "meta_p0", "meta_p1", "meta_p2", "meta_pred",
            "meta_family", "meta_param",
        ]
        meta_oof_rows.append(df_va[keep])
        fold_summaries.append({
            "fold": k, "best_family": best["family"],
            "best_param": float(best["param"]),
            "best_logloss": float(best["logloss"]),
            "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
            "all_results": [
                {"family": r["family"], "param": float(r["param"]),
                 "logloss": float(r["logloss"])}
                for r in results
            ],
        })

    meta_oof = pd.concat(meta_oof_rows, ignore_index=True)
    out_path = os.path.join(HERE, "meta_oof_h10.parquet")
    meta_oof.to_parquet(out_path, index=False)
    print(f"\nmeta OOF saved -> {out_path}  ({len(meta_oof):,} rows)", flush=True)

    # 4) Quick raw-argmax sanity check (no threshold)
    print("\n=== Meta raw-argmax baseline (no threshold) per fold ===", flush=True)
    for k in range(5):
        sub = meta_oof[meta_oof.sym == k]
        probs_va = sub[["meta_p0", "meta_p1", "meta_p2"]].to_numpy(np.float32)
        m = _per_horizon_metrics(
            probs_va.argmax(1).astype(np.int64),
            sub[f"true_label_{TARGET_HORIZON}"].to_numpy(np.int64),
            sub["midprice_t"].to_numpy(np.float32),
            sub[f"midprice_th_{TARGET_HORIZON}"].to_numpy(np.float32),
            fee_rate=0.0001,
        )
        print(f"  fold{k}: cum_pnl={m['cum_pnl']:+.4f}  "
              f"n_active={int(m['n_predictions_active'])}  "
              f"acc={m['accuracy']:.4f}", flush=True)

    # 5) Save summary
    elapsed = time.time() - t0
    summary = {
        "task": "T12 multi-horizon stacking with meta-learner",
        "target_horizon": TARGET_HORIZON,
        "feat_cols": feat_cols,
        "n_total_rows": int(len(meta_oof)),
        "fold_summaries": fold_summaries,
        "elapsed_sec": elapsed,
    }
    with open(os.path.join(HERE, "fold_summaries.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nfold_summaries.json saved", flush=True)
    print(f"\nelapsed: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
