"""T35: ReVol Normalization features (R31 P17, Lee 2025).

Causal rolling-window standardization of log returns, then summary stats over
last few ticks.

For a price series mid (length T), with mid being a "return-like" small
quantity (already normalized in our parquet to small +/- range), we compute:

  ret[t] = log((mid[t] + 1) / (mid[t-1] + 1))            t = 1..T-1
  mu_hat[t]    = mean(ret[t-W .. t-1])                   t = W..T-1
  sigma_hat[t] = std(ret[t-W .. t-1]) + 1e-6
  eps[t]       = (ret[t] - mu_hat[t]) / sigma_hat[t]

Using W=64 (causal, must NOT include current return).

Then per output row at time t (the "look-at-100-ticks-ending-at-t" frame),
we emit 6 features:
    eps_last      = eps[t]
    eps_mean5     = mean(eps[t-4..t])
    eps_mean20    = mean(eps[t-19..t])
    eps_std20     = std(eps[t-19..t])
    eps_skew20    = skew(eps[t-19..t])
    sigma_hat_t   = sigma_hat[t]   (rolling realised vol)

All computations are pure pandas/numpy on the full session series, then
indexed by output row positions. No look-ahead anywhere.

Constraint compliance:
  - sym never used (per-session series only)
  - date never used
  - Stateless: features at row t depend only on mid[t-W-19 .. t]
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


REVOL_W = 64       # window for rolling mu/sigma (returns)
REVOL_M = 20       # window for summary stats over last M epsilons
REVOL_M_SHORT = 5  # short summary window (mean5)
EPS = 1e-6


def _causal_eps_and_sigma(price: np.ndarray, W: int = REVOL_W) -> tuple[np.ndarray, np.ndarray]:
    """Return (eps, sigma_hat), both shape (T,), NaN where invalid.

    price : (T,) np.float32 of mid-like values (in our data, already small,
            "+1 then log" handles return-of-return interpretation).
    """
    T = len(price)
    out_eps = np.full(T, np.nan, dtype=np.float32)
    out_sigma = np.full(T, np.nan, dtype=np.float32)
    if T < W + 2:
        return out_eps, out_sigma

    ret = np.zeros(T, dtype=np.float64)
    ret[1:] = np.log((price[1:] + 1.0) / (price[:-1] + 1.0))

    s = pd.Series(ret)
    # Causal: estimator at t uses returns [t-W..t-1] -> shift(1) after rolling.
    mu = s.rolling(W).mean().shift(1).to_numpy()
    sg = s.rolling(W).std(ddof=1).shift(1).to_numpy()
    sg = np.where(np.isfinite(sg), sg, np.nan) + EPS

    eps = (ret - mu) / sg
    out_eps[:] = eps.astype(np.float32)
    out_sigma[:] = sg.astype(np.float32)
    return out_eps, out_sigma


def _rolling_skew(x: np.ndarray, win: int) -> np.ndarray:
    """Causal rolling skew (sample, bias=False) on numpy array; NaN where invalid."""
    s = pd.Series(x)
    return s.rolling(win, min_periods=win).skew().to_numpy().astype(np.float32)


def revol_feature_names(prefixes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for p in prefixes:
        out.extend([
            f"revol_{p}_eps_last",
            f"revol_{p}_eps_mean5",
            f"revol_{p}_eps_mean20",
            f"revol_{p}_eps_std20",
            f"revol_{p}_eps_skew20",
            f"revol_{p}_sigma_hat",
        ])
    return out


def compute_revol_session(
    df: pd.DataFrame,
    signal_cols: Sequence[str],
    valid_lo: int,
    valid_hi: int,
) -> np.ndarray:
    """Return (n_rows, 6 * len(signal_cols)) ReVol feature matrix for output rows
    in [valid_lo, valid_hi] inclusive.

    Each block of 6 cols corresponds to one signal in `signal_cols` (order
    matches `revol_feature_names`).

    Defensive fillna(0) for any leftover NaN (rare at t>=99, but possible if
    the underlying signal had NaN like wmp_lvl1 first ticks).
    """
    n_out = valid_hi - valid_lo + 1
    F = 6 * len(signal_cols)
    out = np.zeros((n_out, F), dtype=np.float32)

    rows = np.arange(valid_lo, valid_hi + 1)

    for i, col in enumerate(signal_cols):
        price = df[col].to_numpy(dtype=np.float32, copy=False)
        # Forward-fill any NaN inside the price series (rare at start),
        # then backfill leading NaN with 0.
        if not np.all(np.isfinite(price)):
            s = pd.Series(price)
            price = s.ffill().fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        eps, sigma = _causal_eps_and_sigma(price, W=REVOL_W)
        # Replace NaN with 0 (mostly only at very start of session)
        eps = np.where(np.isfinite(eps), eps, 0.0).astype(np.float32)
        sigma = np.where(np.isfinite(sigma), sigma, 0.0).astype(np.float32)

        s_eps = pd.Series(eps)
        eps_mean5 = s_eps.rolling(REVOL_M_SHORT, min_periods=1).mean().to_numpy().astype(np.float32)
        eps_mean20 = s_eps.rolling(REVOL_M, min_periods=1).mean().to_numpy().astype(np.float32)
        eps_std20 = s_eps.rolling(REVOL_M, min_periods=2).std(ddof=1).to_numpy().astype(np.float32)
        eps_skew20 = _rolling_skew(eps, REVOL_M)

        eps_std20 = np.where(np.isfinite(eps_std20), eps_std20, 0.0).astype(np.float32)
        eps_skew20 = np.where(np.isfinite(eps_skew20), eps_skew20, 0.0).astype(np.float32)

        col_off = 6 * i
        out[:, col_off + 0] = eps[rows]
        out[:, col_off + 1] = eps_mean5[rows]
        out[:, col_off + 2] = eps_mean20[rows]
        out[:, col_off + 3] = eps_std20[rows]
        out[:, col_off + 4] = eps_skew20[rows]
        out[:, col_off + 5] = sigma[rows]

    return out


def compute_revol_window(
    df_window: pd.DataFrame,
    signal_cols: Sequence[str],
) -> np.ndarray:
    """Inference-time variant: take a 100-tick window DataFrame, return a
    (6 * len(signal_cols),) feature vector for the *last* tick.

    Matches the offline `compute_revol_session` semantics (causal, no
    look-ahead, fillna defaults).
    """
    feats = compute_revol_session(
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
    # Simulate mid as small random walk
    inc = rng.standard_normal(T) * 0.0005
    mid = np.cumsum(inc).astype(np.float32)
    df = pd.DataFrame({"midprice": mid, "wmp_lvl1": mid + rng.standard_normal(T) * 1e-5})
    feats = compute_revol_session(df, ["midprice", "wmp_lvl1"], valid_lo=99, valid_hi=T - 1 - 60)
    print("feats shape:", feats.shape)
    print("any nan?:", np.isnan(feats).any())
    print("feature names:", revol_feature_names(["mid", "wmp1"]))
    print("first row:", feats[0])

    # Inference-time check on last-100 window
    last100 = df.iloc[-100:].reset_index(drop=True)
    last_feat = compute_revol_window(last100, ["midprice", "wmp_lvl1"])
    print("inference 1-row feat:", last_feat)
    print("OK")
