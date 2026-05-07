"""Shared loader and gating utilities for T30 threshold optimization.

Loads the 5-seed aug_a ensemble OOF predictions for h_60 across all 5 LOSO
folds (held_sym 0..4), averages the per-seed softmax probabilities, and
returns one DataFrame per fold plus a concatenated "all-folds" DataFrame.

All predictors here are sym-agnostic / stateless / date-free per the workdir
hard constraints.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
T27_DIR = os.path.join(ROOT, "experiments", "T27_iter005")

SEEDS = [42, 1, 7, 13, 100]
N_FOLDS = 5
H = 60
FEE = 0.0001
PROB_COLS = ["prob_0", "prob_1", "prob_2"]


def _seed_path(seed: int, k: int) -> str:
    if seed == 42:
        return os.path.join(T26_DIR, f"loso_pred_h{H}_aug_a_held{k}.parquet")
    return os.path.join(T27_DIR, f"loso_pred_h{H}_aug_a_seed{seed}_held{k}.parquet")


def load_fold(k: int) -> pd.DataFrame:
    """Load one fold's OOF table with 5-seed-averaged probs."""
    base = pd.read_parquet(_seed_path(SEEDS[0], k))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in SEEDS[1:]:
        df_s = pd.read_parquet(_seed_path(s, k))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch fold={k} seed={s}: {len(df_s)} vs {n}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(len(SEEDS))
    out = base.drop(columns=PROB_COLS).copy()
    out["prob_0"] = p_avg[:, 0].astype(np.float32)
    out["prob_1"] = p_avg[:, 1].astype(np.float32)
    out["prob_2"] = p_avg[:, 2].astype(np.float32)
    out["fold"] = np.int8(k)
    return out


def load_all_folds() -> Dict[int, pd.DataFrame]:
    return {k: load_fold(k) for k in range(N_FOLDS)}


def load_concat() -> pd.DataFrame:
    return pd.concat([load_fold(k) for k in range(N_FOLDS)], ignore_index=True)


# ---------- gating ----------

def gate_symmetric(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def gate_asymmetric(probs: np.ndarray, T_up: float, T_dn: float,
                    d_up: float, d_dn: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


# ---------- pnl ----------

def per_fold_pnl(df_fold: pd.DataFrame, pred: np.ndarray) -> dict:
    m = _per_horizon_metrics(
        pred,
        df_fold["true_label"].to_numpy(np.int64),
        df_fold["midprice_t"].to_numpy(np.float32),
        df_fold["midprice_th"].to_numpy(np.float32),
        fee_rate=FEE,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
    }


def sum_pnl_over_folds(folds: Dict[int, pd.DataFrame], pred_per_fold) -> dict:
    """pred_per_fold can be a callable f(df_fold) -> ndarray pred, or dict[k]->ndarray."""
    per = []
    for k in range(N_FOLDS):
        df = folds[k]
        pred = pred_per_fold(df) if callable(pred_per_fold) else pred_per_fold[k]
        per.append(per_fold_pnl(df, pred))
    sums = sum(r["cum_pnl"] for r in per)
    return {
        "sum_cum_pnl": float(sums),
        "mean_cum_pnl": float(sums / N_FOLDS),
        "std_cum_pnl": float(np.std([r["cum_pnl"] for r in per], ddof=0)),
        "n_pos_folds": int(sum(1 for r in per if r["cum_pnl"] > 0)),
        "sum_n_active": int(sum(r["n_active"] for r in per)),
        "per_fold_pnl": [r["cum_pnl"] for r in per],
        "per_fold_active": [r["n_active"] for r in per],
        "mean_accuracy": float(np.mean([r["accuracy"] for r in per])),
    }


# ---------- vectorized PnL on concatenated frames ----------

def vectorized_pnl(pred: np.ndarray, label: np.ndarray,
                   mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    side = pred.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


@dataclass
class FoldArrays:
    fold: int
    probs: np.ndarray         # (n,3)
    label: np.ndarray         # (n,)
    mp_t: np.ndarray          # (n,)
    mp_th: np.ndarray         # (n,)


def fold_arrays(folds: Dict[int, pd.DataFrame]) -> List[FoldArrays]:
    out = []
    for k in range(N_FOLDS):
        df = folds[k]
        out.append(FoldArrays(
            fold=k,
            probs=df[PROB_COLS].to_numpy(np.float32),
            label=df["true_label"].to_numpy(np.int64),
            mp_t=df["midprice_t"].to_numpy(np.float64),
            mp_th=df["midprice_th"].to_numpy(np.float64),
        ))
    return out
