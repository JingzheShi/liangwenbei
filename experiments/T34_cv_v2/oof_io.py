"""
Common helpers for T34 CV system v2.

Loads OOF predictions from the various iter directories and exposes a
unified interface:

    load_oof(model, horizon) -> pd.DataFrame
        columns: sym, date, session, t, true_label, pred_label, prob_0, prob_1, prob_2,
                 midprice_t, midprice_th

Each row corresponds to one (sym, date, session, t) test point. The DataFrame
covers the FULL 5-sym OOF (concatenation of held{0..4}).

Supported models:
    - 'iter_002'        : T5b — h ∈ {5,10,20,40,60}, single seed argmax
    - 'iter_005b'       : T26 (seed42) + T27 (seeds 1,7,13,100) — h=60 ONLY,
                          5-seed prob average + (T=0.45, delta=0.10) threshold
    - 'iter_005b_h40'   : iter_002 h_40 (kept unchanged in iter_005b)
    - 't26_aug_a'       : T26 single-seed aug_a, h=60
    - 't26_baseline'    : T26 baseline (seed 42 no aug), h=60
    - 'iter_003_h10'    : T11 5-seed ensemble at h=10 (used for iter_003)

Per-trade alpha filter: see classify_active() and apply_threshold_rule().

NOTE: All sym-stratified analyses respect CRITICAL_CONSTRAINTS.md §3 (sym-agnostic).
We use sym only as a fold-id partition key (not as model feature).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HORIZONS = (5, 10, 20, 40, 60)
FEE_RATE = 0.0001


def _argmax_pred(df: pd.DataFrame) -> np.ndarray:
    """Take argmax over prob_0/1/2 columns."""
    p = df[["prob_0", "prob_1", "prob_2"]].to_numpy()
    return p.argmax(axis=1).astype(np.int64)


def _threshold_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    """Apply (T, delta) threshold rule on (N, 3) probs."""
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    side_pred = np.where(p2 > p0, 2, 0)
    return np.where(take, side_pred, 1).astype(np.int64)


def _load_concat_folds(pattern: str) -> pd.DataFrame:
    """Load and concat 5 hold-out folds matching pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no OOF files matching {pattern}")
    dfs = [pd.read_parquet(f) for f in files]
    out = pd.concat(dfs, axis=0, ignore_index=True)
    return out


def _ensemble_5seed(h: int, T: float, delta: float) -> pd.DataFrame:
    """Combine T26 aug_a (seed 42) + T27 aug_a (seeds 1,7,13,100) per fold,
    average probs, apply threshold rule."""
    # Per held-out sym (0..4), gather probs across 5 seeds
    out = []
    for held in range(5):
        # seed 42 = T26 aug_a
        p42 = pd.read_parquet(
            f"{ROOT}/experiments/T26_domain_randomization/loso_pred_h{h}_aug_a_held{held}.parquet"
        )
        # seeds 1,7,13,100 = T27
        seed_files = [
            f"{ROOT}/experiments/T27_iter005/loso_pred_h{h}_aug_a_seed{s}_held{held}.parquet"
            for s in (1, 7, 13, 100)
        ]
        ps = [pd.read_parquet(f) for f in seed_files]
        # All should align on (sym, date, session, t). Use seed42 as anchor.
        anchor = p42.copy()
        keys = ["sym", "date", "session", "t"]
        anchor = anchor.sort_values(keys).reset_index(drop=True)
        prob_acc = anchor[["prob_0", "prob_1", "prob_2"]].to_numpy().astype(np.float64)
        for ps_df in ps:
            ps_df = ps_df.sort_values(keys).reset_index(drop=True)
            assert (ps_df[keys].values == anchor[keys].values).all(), \
                f"mismatch keys held={held}"
            prob_acc += ps_df[["prob_0", "prob_1", "prob_2"]].to_numpy().astype(np.float64)
        prob_acc /= 5.0
        anchor = anchor.astype({"prob_0": np.float64, "prob_1": np.float64, "prob_2": np.float64})
        anchor.loc[:, "prob_0"] = prob_acc[:, 0]
        anchor.loc[:, "prob_1"] = prob_acc[:, 1]
        anchor.loc[:, "prob_2"] = prob_acc[:, 2]
        anchor["pred_label"] = _threshold_pred(prob_acc, T=T, delta=delta)
        out.append(anchor)
    return pd.concat(out, axis=0, ignore_index=True)


def load_oof(model: str, horizon: int) -> pd.DataFrame:
    """
    Returns OOF predictions for (model, horizon). Columns:
        sym, date, session, t, true_label, pred_label, prob_0, prob_1, prob_2,
        midprice_t, midprice_th

    `pred_label` reflects the deployed decision rule (argmax for raw, thresholded
    for threshold-based iterations).
    """
    if model == "iter_002":
        df = _load_concat_folds(
            f"{ROOT}/experiments/T5b_features_multihorizon/loso_pred_h{horizon}_held*.parquet"
        )
        # iter_002 uses raw argmax (no threshold rule per Predictor_iter_002)
        # Actually, iter_002 DID apply threshold rules per submission — see thresholds.json
        # But the raw OOF parquet stores argmax. We honor what was deployed.
        # From SUBMISSION_LOG iter_002 thresholds: per-horizon T,delta is applied.
        # Read iter_002 thresholds:
        # h_5: from sweep, look at submission/iter_002 dir for actual values
        thresh = _iter_002_thresholds().get(horizon)
        if thresh is not None:
            probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy()
            df["pred_label"] = _threshold_pred(probs, T=thresh[0], delta=thresh[1])
        else:
            df["pred_label"] = _argmax_pred(df)
        return df
    if model == "iter_005b" and horizon == 60:
        return _ensemble_5seed(h=60, T=0.45, delta=0.10)
    if model == "iter_005b" and horizon == 40:
        # iter_005b reuses iter_002's h_40 model with same threshold (T=0.5, delta=0.0)
        df = _load_concat_folds(
            f"{ROOT}/experiments/T5b_features_multihorizon/loso_pred_h40_held*.parquet"
        )
        probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy()
        df["pred_label"] = _threshold_pred(probs, T=0.50, delta=0.00)
        return df
    if model == "iter_005b" and horizon in (5, 10, 20):
        # Inactive in iter_005b — pred all 1
        df = _load_concat_folds(
            f"{ROOT}/experiments/T5b_features_multihorizon/loso_pred_h{horizon}_held*.parquet"
        )
        df["pred_label"] = 1
        return df
    if model == "t26_aug_a" and horizon == 60:
        df = _load_concat_folds(
            f"{ROOT}/experiments/T26_domain_randomization/loso_pred_h60_aug_a_held*.parquet"
        )
        probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy()
        # T27 best single-seed threshold: T=0.45, delta=0.10
        df["pred_label"] = _threshold_pred(probs, T=0.45, delta=0.10)
        return df
    if model == "t26_baseline" and horizon == 60:
        df = _load_concat_folds(
            f"{ROOT}/experiments/T26_domain_randomization/loso_pred_h60_baseline_held*.parquet"
        )
        df["pred_label"] = _argmax_pred(df)
        return df
    raise ValueError(f"unknown (model, horizon) = ({model}, {horizon})")


def _iter_002_thresholds() -> Dict[int, Tuple[float, float]]:
    """Per-horizon (T, delta) used in iter_002 deployment.

    Source: submission/iter_002_lgbm_schemeC/thresholds.json — each horizon
    has its own threshold from T5b LOSO sweep.
    """
    import json
    p = os.path.join(ROOT, "submission", "iter_002_lgbm_schemeC", "thresholds.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        cfg = json.load(f)
    out = {}
    for hcfg in cfg.get("horizons", []):
        if hcfg.get("active", True):
            out[int(hcfg["h"])] = (float(hcfg["T"]), float(hcfg["delta"]))
        else:
            # Inactive horizon: pred always 1 (no trade)
            out[int(hcfg["h"])] = (0.99, 0.99)
    return out


def compute_pnl_per_row(df: pd.DataFrame) -> np.ndarray:
    """Vectorized per-row PnL. Uses df.pred_label, df.midprice_t, df.midprice_th.

    Returns (N,) float64 array of single-row PnL contributions.
    """
    pred = df["pred_label"].to_numpy(dtype=np.int64)
    mp_t = df["midprice_t"].to_numpy(dtype=np.float64)
    mp_th = df["midprice_th"].to_numpy(dtype=np.float64)
    side = pred.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th - mp_t
    fee = FEE_RATE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    return (side * diff - fee) / denom


def cum_pnl_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute cum_pnl, n_active, per_trade_pnl on an OOF DataFrame."""
    pnl = compute_pnl_per_row(df)
    n_active = int((df["pred_label"].to_numpy() != 1).sum())
    cum = float(pnl.sum())
    per_trade = cum / n_active if n_active else 0.0
    return {
        "cum_pnl": cum,
        "n_active": n_active,
        "n_total": int(len(df)),
        "per_trade_pnl": per_trade,
    }


def cum_pnl_per_session(df: pd.DataFrame) -> pd.DataFrame:
    """Compute (sym, date, session) -> cum_pnl table."""
    pnl = compute_pnl_per_row(df)
    work = df[["sym", "date", "session", "pred_label"]].copy()
    work["pnl"] = pnl
    work["active"] = (work["pred_label"] != 1).astype(int)
    grp = work.groupby(["sym", "date", "session"], as_index=False).agg(
        cum_pnl=("pnl", "sum"),
        n_active=("active", "sum"),
        n_total=("active", "count"),
    )
    return grp


def cum_pnl_per_sym(df: pd.DataFrame) -> pd.DataFrame:
    """Per-sym aggregate."""
    pnl = compute_pnl_per_row(df)
    work = df[["sym", "pred_label"]].copy()
    work["pnl"] = pnl
    work["active"] = (work["pred_label"] != 1).astype(int)
    grp = work.groupby("sym", as_index=False).agg(
        cum_pnl=("pnl", "sum"),
        n_active=("active", "sum"),
        n_total=("active", "count"),
    )
    grp["per_trade_pnl"] = grp.apply(
        lambda r: r["cum_pnl"] / r["n_active"] if r["n_active"] else 0.0, axis=1
    )
    return grp
