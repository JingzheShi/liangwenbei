"""T35: Causal Savitzky-Golay smooth features (R31 P10, Crypto LOB Better Inputs 2025).

For a length-L window (here L=9, polyorder=3), we use scipy.signal.savgol_coeffs
to derive the FIR weights for the LAST position of the window. Convolving the
weights with the signal (taking only "valid" positions) yields a strictly
causal smoothed series: smooth[t] only depends on x[t-L+1..t].

Per signal we emit 4 features at the output row time t:
    sg_smooth_last  : smoothed value at t
    sg_minus_raw    : smoothed value at t  -  raw value at t   (deviation)
    sg_slope        : 1st derivative estimated by SG at t      (per-tick)
    sg_accel        : 2nd derivative estimated by SG at t      (per-tick^2)

Constraint compliance:
  - Pure FIR over past values (no look-ahead, by construction)
  - Stateless / sym-agnostic
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_coeffs


SG_WIN = 9
SG_POLY = 3

# Pre-compute weights ONCE (immutable, shared).
_W_SMOOTH = savgol_coeffs(SG_WIN, SG_POLY, pos=SG_WIN - 1, deriv=0, use="dot").astype(np.float64)
_W_DERIV1 = savgol_coeffs(SG_WIN, SG_POLY, pos=SG_WIN - 1, deriv=1, use="dot").astype(np.float64)
_W_DERIV2 = savgol_coeffs(SG_WIN, SG_POLY, pos=SG_WIN - 1, deriv=2, use="dot").astype(np.float64)


def _causal_sg(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply causal SG with weights `w` (length SG_WIN) to series `x`.

    Returns array of same length; values for t < SG_WIN-1 are equal to x[t]
    (we just pass through; they are below WINDOW=100 anyway, so unused).
    """
    L = len(w)
    out = np.empty_like(x, dtype=np.float64)
    out[:L - 1] = x[:L - 1].astype(np.float64)
    if len(x) >= L:
        windows = sliding_window_view(x.astype(np.float64), L)  # (T-L+1, L)
        out[L - 1:] = windows @ w
    return out


def sg_feature_names(prefixes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in prefixes:
        out.extend([
            f"sg_{p}_smooth",
            f"sg_{p}_minus_raw",
            f"sg_{p}_slope",
            f"sg_{p}_accel",
        ])
    return out


def compute_sg_session(
    df: pd.DataFrame,
    signal_cols: Sequence[str],
    valid_lo: int,
    valid_hi: int,
) -> np.ndarray:
    """Return (n_rows, 4 * len(signal_cols)) SG feature matrix."""
    n_out = valid_hi - valid_lo + 1
    F = 4 * len(signal_cols)
    out = np.zeros((n_out, F), dtype=np.float32)

    rows = np.arange(valid_lo, valid_hi + 1)

    for i, col in enumerate(signal_cols):
        x = df[col].to_numpy(dtype=np.float32, copy=False)
        if not np.all(np.isfinite(x)):
            x = pd.Series(x).ffill().fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        smooth = _causal_sg(x, _W_SMOOTH)
        slope = _causal_sg(x, _W_DERIV1)
        accel = _causal_sg(x, _W_DERIV2)
        deviation = smooth - x.astype(np.float64)

        col_off = 4 * i
        out[:, col_off + 0] = smooth[rows].astype(np.float32)
        out[:, col_off + 1] = deviation[rows].astype(np.float32)
        out[:, col_off + 2] = slope[rows].astype(np.float32)
        out[:, col_off + 3] = accel[rows].astype(np.float32)

    return out


def compute_sg_window(
    df_window: pd.DataFrame,
    signal_cols: Sequence[str],
) -> np.ndarray:
    """Inference-time variant: takes a 100-tick window, returns 4*K features
    for the last tick. Equivalent to compute_sg_session at last row.
    """
    feats = compute_sg_session(
        df_window,
        signal_cols=signal_cols,
        valid_lo=len(df_window) - 1,
        valid_hi=len(df_window) - 1,
    )
    return feats[0]


# --- self test ---
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 2000
    mid = np.cumsum(rng.standard_normal(T) * 0.0005).astype(np.float32)
    df = pd.DataFrame({"midprice": mid, "wmp_lvl1": mid * 1.0001, "imbalance": rng.standard_normal(T).astype(np.float32) * 0.001})
    feats = compute_sg_session(df, ["midprice", "wmp_lvl1", "imbalance"], valid_lo=99, valid_hi=T - 1 - 60)
    print("feats shape:", feats.shape)
    print("any nan?:", np.isnan(feats).any())
    print("names:", sg_feature_names(["mid", "wmp1", "imb"]))
    print("first row:", feats[0])

    # Causality check: changing future values should NOT change SG features for any earlier row.
    df2 = df.copy()
    df2.iloc[1500:] = df2.iloc[1500:] + 999.0  # massive future shock
    feats2 = compute_sg_session(df2, ["midprice", "wmp_lvl1", "imbalance"], valid_lo=99, valid_hi=T - 1 - 60)
    # rows up to t=1499 should be untouched (last row in our slice = t-rel = 1499 - 99)
    diff = np.abs(feats[:1499 - 99 - 8] - feats2[:1499 - 99 - 8]).max()
    print(f"max diff for causal-safe rows: {diff:.6g} (should be 0)")
    assert diff == 0.0, "CAUSALITY VIOLATION"
    print("OK")
