"""T44 R34 Stage 1 feature computations (sym-invariant).

Includes 4 feature families:
  #1 Dual window z-score (A9): z(W=20) - z(W=100), applied to ~37 raw cols
  #2 Signed Realized Vol (C9): (rv_pos - rv_neg) / (rv_pos + rv_neg) at W=20,50,100
  #3 Kyle-Obizhaeva bet size invariant (B4): amount / activity^(1/3) at W=50,100
  #6 EWMA-OFI multi-alpha (E3): EWMA on mlofi_W20_lvl_k for k in {1,5,10}, alpha in {0.05,0.1,0.3,0.5}

All operations are session-scoped (per (sym, date, sess) parquet); no cross-session shift.
W <= 100 to remain compatible with the platform's 100-tick inference window.

Constraint compliance:
  - sym never used (per-session series)
  - date never used
  - causal rolling (no look-ahead)
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

EPS = 1e-8

# Cols (must exist in the snapshot parquet) for dual-z #1.
DUAL_Z_COLS: List[str] = [
    # spread family (4)
    "spread1", "spread5", "spread10", "cumspread",
    # bid/ask price (4)
    "bid1", "ask1", "bid_mean", "ask_mean",
    # midprice family (4)
    "midprice1", "midprice2", "midprice5", "midprice10",
    # bid/ask depth (3 each)
    "bsize1", "bsize_mean", "totalbsize",
    "asize1", "asize_mean", "totalasize",
    # trade flow (2)
    "volume_delta", "amount_delta",
    # imbalance (1)
    "imbalance",
    # event intensities raw (6)
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    # event accumulations (6)
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    # bid/ask diffs (4)
    "bid_diff1", "ask_diff1", "bid_diff5", "ask_diff5",
]
# Total 37

# Param config
DUAL_Z_W_SHORT = 20
DUAL_Z_W_LONG = 100

SIGNED_RV_WINDOWS = (20, 50, 100)
KYLE_WINDOWS = (50, 100)
EWMA_OFI_LEVELS = (1, 5, 10)
EWMA_OFI_ALPHAS = (0.05, 0.1, 0.3, 0.5)


# --- #1 Dual-window z-score ---------------------------------------------

def _rolling_zscore(s: pd.Series, W: int) -> np.ndarray:
    """Causal rolling (x_t - mean) / std over [t-W+1 .. t]. NaN where invalid."""
    m = s.rolling(W, min_periods=W).mean()
    sd = s.rolling(W, min_periods=W).std(ddof=0)
    z = (s - m) / (sd + EPS)
    return z.to_numpy(dtype=np.float32)


def compute_dual_z_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Return (n_rows, len(DUAL_Z_COLS)) array of dual-z features.
    dual_z = z(W_short=20) - z(W_long=100). At each row, both z computed over
    last W ticks ending at t (causal). Pre-W_long ticks → NaN → set to 0.
    """
    n_out = valid_hi - valid_lo + 1
    F = len(DUAL_Z_COLS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    for i, col in enumerate(DUAL_Z_COLS):
        s = pd.Series(df[col].to_numpy(dtype=np.float64, copy=False))
        z_s = _rolling_zscore(s, DUAL_Z_W_SHORT)
        z_l = _rolling_zscore(s, DUAL_Z_W_LONG)
        dual = z_s - z_l
        dual = np.where(np.isfinite(dual), dual, 0.0).astype(np.float32)
        out[:, i] = dual[rows]
    return out


def dual_z_feature_names() -> List[str]:
    return [f"dualz_{c}" for c in DUAL_Z_COLS]


# --- #2 Signed Realized Vol --------------------------------------------

def compute_signed_rv_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """For W in SIGNED_RV_WINDOWS, return signed RV: (rv_pos - rv_neg) / (rv_pos + rv_neg + eps).
    r_t = log(midprice_t / midprice_{t-1}).
    """
    n_out = valid_hi - valid_lo + 1
    F = len(SIGNED_RV_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    # Guard against non-positive prices
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)

    r2_pos = (r * r) * (r > 0)
    r2_neg = (r * r) * (r < 0)
    s_pos = pd.Series(r2_pos)
    s_neg = pd.Series(r2_neg)

    for i, W in enumerate(SIGNED_RV_WINDOWS):
        rv_pos = s_pos.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)
        rv_neg = s_neg.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)
        sig = (rv_pos - rv_neg) / (rv_pos + rv_neg + EPS)
        sig = np.where(np.isfinite(sig), sig, 0.0).astype(np.float32)
        out[:, i] = sig[rows]
    return out


def signed_rv_feature_names() -> List[str]:
    return [f"signed_rv_W{W}" for W in SIGNED_RV_WINDOWS]


# --- #3 Kyle-Obizhaeva bet size invariant ------------------------------

def compute_kyle_invariant_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Kyle-Obizhaeva bet-size invariant: amount_delta / activity^(1/3),
    activity = |amount_delta| * sigma_log_mid (W ticks).

    Returns (n_out, len(KYLE_WINDOWS)) array. amount_delta is the raw signed value
    (NOT log1p-transformed; we want the original size for Kyle scaling).
    """
    n_out = valid_hi - valid_lo + 1
    F = len(KYLE_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)
    s_r = pd.Series(r)

    amt = df["amount_delta"].to_numpy(dtype=np.float64, copy=False)
    abs_amt = np.abs(amt)

    for i, W in enumerate(KYLE_WINDOWS):
        sigma_W = s_r.rolling(W, min_periods=W).std(ddof=0).to_numpy(dtype=np.float64) + EPS
        activity = abs_amt * sigma_W + EPS
        # cube-root activity; numpy cbrt handles negative gracefully but activity is non-negative
        activity_third_root = np.cbrt(activity)
        inv = amt / (activity_third_root + EPS)
        inv = np.where(np.isfinite(inv), inv, 0.0).astype(np.float32)
        out[:, i] = inv[rows]
    return out


def kyle_invariant_feature_names() -> List[str]:
    return [f"kyle_inv_W{W}" for W in KYLE_WINDOWS]


# --- #6 EWMA-OFI multi-alpha -------------------------------------------

def compute_ewma_ofi_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Apply EWMA(alpha) to mlofi_W20_lvl_k for k in {1,5,10}, alpha in {0.05,0.1,0.3,0.5}.

    Returns (n_out, 3*4=12) array. mlofi_W20_lvl_k from cache parquet has NaN at start
    of session (first W=20 ticks); we fill NaN with 0 then EWMA.
    """
    n_out = valid_hi - valid_lo + 1
    F = len(EWMA_OFI_LEVELS) * len(EWMA_OFI_ALPHAS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    col_idx = 0
    for k in EWMA_OFI_LEVELS:
        col = f"mlofi_W20_lvl{k}"
        x = df[col].to_numpy(dtype=np.float64, copy=False)
        x = np.where(np.isfinite(x), x, 0.0)
        s_x = pd.Series(x)
        for alpha in EWMA_OFI_ALPHAS:
            ewma = s_x.ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=np.float32)
            out[:, col_idx] = ewma[rows]
            col_idx += 1
    return out


def ewma_ofi_feature_names() -> List[str]:
    names: List[str] = []
    for k in EWMA_OFI_LEVELS:
        for alpha in EWMA_OFI_ALPHAS:
            names.append(f"ewma_ofi_a{alpha}_lvl{k}")
    return names


# --- All-in-one ---------------------------------------------------------

def compute_all_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    a = compute_dual_z_session(df, valid_lo, valid_hi)
    b = compute_signed_rv_session(df, valid_lo, valid_hi)
    c = compute_kyle_invariant_session(df, valid_lo, valid_hi)
    d = compute_ewma_ofi_session(df, valid_lo, valid_hi)
    return np.concatenate([a, b, c, d], axis=1)


def all_feature_names() -> List[str]:
    return (
        dual_z_feature_names()
        + signed_rv_feature_names()
        + kyle_invariant_feature_names()
        + ewma_ofi_feature_names()
    )


if __name__ == "__main__":
    # Quick self-test on a single session
    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    p = os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet")
    df = pd.read_parquet(p)
    print("session shape:", df.shape, "cols=", len(df.columns))
    valid_lo, valid_hi = 99, len(df) - 1 - 60
    feats = compute_all_session(df, valid_lo, valid_hi)
    names = all_feature_names()
    print(f"feats shape: {feats.shape}, names count={len(names)}")
    print("any nan?", np.isnan(feats).any())
    print("any inf?", np.isinf(feats).any())
    print("first row min/max/mean:", feats[0].min(), feats[0].max(), feats[0].mean())
    print("dual_z[0]:", feats[0, :5])
    print("signed_rv[0]:", feats[0, 37:40])
    print("kyle_inv[0]:", feats[0, 40:42])
    print("ewma_ofi[0]:", feats[0, 42:54])
