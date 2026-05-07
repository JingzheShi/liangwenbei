"""T47: Stage 2 stacking + DE 4D threshold optimization.

For each fold k:
  1. Load val OOF (sym != k, dates 80-95) probs from 4 horizon models.
  2. Load held-out test (sym == k, dates 96-119) probs from 4 horizon models.
  3. Align rows by (sym, date, session, t) — so the 12-d stacking input is
     concatenated correctly.
  4. Filter to rows where y_60 is valid (always true for our cache).
  5. Fit Ridge regression (RidgeCV) on val with target = label_60_one_hot,
     12-d input (4 horizon × 3 prob).
  6. Apply ridge to test → 3-d "stacked logits"; clip to [0,1] and renormalize
     to make valid probabilities.

After all folds:
  7. DE 4D threshold optimization (T_up, T_dn, d_up, d_dn) on stacked probs
     across all 5 folds → maximize sum cum_pnl.

Outputs:
  stacking_results.json     -- per-fold stacked OOF and DE thresh result
  stacking_oof.parquet      -- concatenated stacked predictions across folds
  de_results.json           -- DE thresh + per-fold cum_pnl after thresh
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import RidgeCV

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "T30_threshold_optim"))

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

HORIZONS = (30, 60, 120, 240)
PROB_COLS = ["prob_0", "prob_1", "prob_2"]
N_FOLDS = 5
FEE = 0.0001


def load_pred_table(split: str, h: int, k: int) -> pd.DataFrame:
    """Load val/test pred parquet for horizon h, fold k."""
    if split == "test":
        p = os.path.join(HERE, f"loso_pred_h{h}_held{k}.parquet")
    else:
        p = os.path.join(HERE, f"loso_pred_h{h}_val_held{k}.parquet")
    return pd.read_parquet(p)


def merge_horizons(split: str, k: int) -> pd.DataFrame:
    """Merge predictions from 4 horizon models on row-aligned key."""
    keys = ["sym", "date", "session", "t"]
    dfs = {h: load_pred_table(split, h, k) for h in HORIZONS}

    # Use h=60 as the anchor (it has all rows valid)
    base = dfs[60][keys + PROB_COLS + ["true_label_60", "midprice_t", "midprice_t60"]].copy()
    base = base.rename(columns={c: f"{c}_h60" for c in PROB_COLS})

    for h in (30, 120, 240):
        right = dfs[h][keys + PROB_COLS].copy()
        right = right.rename(columns={c: f"{c}_h{h}" for c in PROB_COLS})
        base = base.merge(right, on=keys, how="left")

    # Sanity: stacking inputs must all be filled (val and test predict on all rows
    # regardless of y_h validity, so this should be fine).
    n_nulls = base[[f"prob_0_h{h}" for h in HORIZONS]].isnull().any(axis=1).sum()
    if n_nulls > 0:
        print(f"  WARN: split={split} fold={k}: {n_nulls} rows with NaN probs; will drop")
        base = base.dropna(subset=[f"prob_0_h{h}" for h in HORIZONS]).reset_index(drop=True)
    return base


def stack_features(df: pd.DataFrame) -> np.ndarray:
    """Extract 12-d stacking input from merged df."""
    cols = []
    for h in HORIZONS:
        cols += [f"prob_0_h{h}", f"prob_1_h{h}", f"prob_2_h{h}"]
    return df[cols].to_numpy(np.float32)


def one_hot(y: np.ndarray, num_class: int = 3) -> np.ndarray:
    n = len(y)
    out = np.zeros((n, num_class), dtype=np.float32)
    out[np.arange(n), y.astype(np.int64)] = 1.0
    return out


def fit_predict_ridge(X_tr, Y_tr, X_te, alphas=(0.1, 1.0, 10.0, 100.0)):
    """Fit RidgeCV on X_tr -> Y_tr (one-hot, 3-dim), predict X_te.

    Returns (X_te_pred (n_te, 3) projected to probability simplex).
    """
    # Fit one ridge per class column (RidgeCV doesn't natively support multi-output
    # CV for picking alpha; use independent fits).
    preds = np.zeros((len(X_te), 3), dtype=np.float32)
    for c in range(3):
        m = RidgeCV(alphas=alphas, fit_intercept=True)
        m.fit(X_tr, Y_tr[:, c])
        preds[:, c] = m.predict(X_te).astype(np.float32)
    # Project to probability simplex: clip negative, renormalize
    preds = np.clip(preds, 1e-6, None)
    preds = preds / preds.sum(axis=1, keepdims=True)
    return preds


def gate_asymmetric(probs: np.ndarray, T_up: float, T_dn: float,
                    d_up: float, d_dn: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def vectorized_pnl(pred: np.ndarray, label: np.ndarray,
                   mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    side = pred.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


def per_fold_pnl(pred, label, mp_t, mp_th):
    return float(vectorized_pnl(pred, label, mp_t, mp_th).sum())


def main():
    t0 = time.time()
    print("=== T47 stacking + DE thresh ===", flush=True)

    # Step 1: Per-fold ridge stacking
    print("\n[1] Fitting Ridge stacking per fold ...")
    fold_data = {}  # k -> dict of stacked test arrays
    val_baseline = {}  # for sanity: argmax of val stacked
    for k in range(N_FOLDS):
        df_va = merge_horizons("val", k)
        df_te = merge_horizons("test", k)

        X_va = stack_features(df_va)
        Y_va = one_hot(df_va["true_label_60"].to_numpy(np.int64))
        X_te = stack_features(df_te)
        y_te60 = df_te["true_label_60"].to_numpy(np.int64)

        # Inner CV is automatic via RidgeCV (LOOCV by default for small
        # alpha grid). For our val set ~300k rows, RidgeCV uses GCV which
        # is fast and avoids overfitting.
        stacked_te = fit_predict_ridge(X_va, Y_va, X_te)

        # Baseline argmax on stacked_te
        pred_te = stacked_te.argmax(axis=1).astype(np.int8)
        m = _per_horizon_metrics(
            pred_te, y_te60,
            df_te["midprice_t"].to_numpy(np.float32),
            df_te["midprice_t60"].to_numpy(np.float32),
            fee_rate=FEE,
        )
        print(f"  fold k={k} stacked argmax: cum_pnl={m['cum_pnl']:+.4f} "
              f"acc={m['accuracy']:.4f}", flush=True)

        fold_data[k] = {
            "stacked_probs": stacked_te,
            "true_label_60": y_te60,
            "midprice_t": df_te["midprice_t"].to_numpy(np.float64),
            "midprice_t60": df_te["midprice_t60"].to_numpy(np.float64),
            "df_te": df_te,
            "argmax_cum_pnl": float(m["cum_pnl"]),
            "argmax_accuracy": float(m["accuracy"]),
        }

    # Step 2: DE 4D threshold
    print("\n[2] DE 4D threshold optimization ...")

    label = [fold_data[k]["true_label_60"] for k in range(N_FOLDS)]
    mp_t = [fold_data[k]["midprice_t"] for k in range(N_FOLDS)]
    mp_th = [fold_data[k]["midprice_t60"] for k in range(N_FOLDS)]
    probs = [fold_data[k]["stacked_probs"] for k in range(N_FOLDS)]

    def objective(params):
        Tu, Td, du, dd = params
        total = 0.0
        for k in range(N_FOLDS):
            pred = gate_asymmetric(probs[k], Tu, Td, du, dd)
            total += vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()
        return -total  # minimize negative

    bounds = [(0.34, 0.55), (0.34, 0.55), (0.0, 0.30), (0.0, 0.30)]
    print(f"  bounds: {bounds}")
    print(f"  running scipy DE (popsize=30, maxiter=100, tol=1e-4) ...", flush=True)
    de_result = differential_evolution(
        objective, bounds,
        popsize=30, maxiter=100, tol=1e-4,
        mutation=(0.5, 1.5), recombination=0.7,
        seed=42, polish=True, workers=1, updating="deferred",
    )

    Tu_opt, Td_opt, du_opt, dd_opt = de_result.x
    sum_pnl_opt = -de_result.fun
    print(f"\n  DE result: T_up={Tu_opt:.4f} T_dn={Td_opt:.4f} "
          f"d_up={du_opt:.4f} d_dn={dd_opt:.4f}")
    print(f"  sum_cum_pnl_opt = {sum_pnl_opt:+.4f}")

    # Per-fold breakdown at opt
    per_fold_after = []
    for k in range(N_FOLDS):
        pred = gate_asymmetric(probs[k], Tu_opt, Td_opt, du_opt, dd_opt)
        pnl = float(vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum())
        per_fold_after.append(pnl)
        n_active = int(((pred == 0) | (pred == 2)).sum())
        print(f"  fold k={k}: cum_pnl={pnl:+.4f} n_active={n_active:,}", flush=True)

    # Save outputs
    out = {
        "task": "T47 multi-horizon aux stacking + DE 4D thresh",
        "horizons": list(HORIZONS),
        "n_folds": N_FOLDS,
        "stacking": {
            "method": "RidgeCV per-class on val OOF",
            "alphas": [0.1, 1.0, 10.0, 100.0],
            "input_dim": 12,
            "argmax_per_fold_pnl": [fold_data[k]["argmax_cum_pnl"] for k in range(N_FOLDS)],
            "argmax_sum_pnl": float(sum(fold_data[k]["argmax_cum_pnl"] for k in range(N_FOLDS))),
            "argmax_per_fold_acc": [fold_data[k]["argmax_accuracy"] for k in range(N_FOLDS)],
        },
        "de_thresh": {
            "T_up": float(Tu_opt),
            "T_dn": float(Td_opt),
            "d_up": float(du_opt),
            "d_dn": float(dd_opt),
            "sum_cum_pnl_after_thresh": float(sum_pnl_opt),
            "per_fold_pnl_after_thresh": per_fold_after,
            "n_pos_folds": int(sum(1 for x in per_fold_after if x > 0)),
            "de_n_iter": int(de_result.nit),
            "de_n_eval": int(de_result.nfev),
        },
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(HERE, "stacking_de_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults -> {out_path}")

    # Save concatenated stacked OOF parquet
    rows = []
    for k in range(N_FOLDS):
        df = fold_data[k]["df_te"].copy()
        sp = fold_data[k]["stacked_probs"]
        df["stack_prob_0"] = sp[:, 0]
        df["stack_prob_1"] = sp[:, 1]
        df["stack_prob_2"] = sp[:, 2]
        df["fold"] = np.int8(k)
        rows.append(df)
    full = pd.concat(rows, ignore_index=True)
    full.to_parquet(os.path.join(HERE, "stacking_oof.parquet"), index=False)
    print(f"OOF parquet -> {os.path.join(HERE, 'stacking_oof.parquet')}")
    print(f"\nelapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
