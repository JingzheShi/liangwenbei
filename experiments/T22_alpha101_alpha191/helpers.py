"""Time-series helpers used by Alpha101 + Alpha191 implementations.

All operations are vectorized (NumPy / Pandas rolling). Per-session,
single-stock, time-series only — sym-agnostic. NaN handling: helpers leave
leading NaNs; the caller is responsible for filling NaNs at the end.

Hard constraints:
  - 100-tick window only — windows used here cap at 50 (well under 100).
  - No cross-section operations — every helper takes a single Series.
  - Stateless within a session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


# ---------------------------------------------------------------------------
# Series → Series helpers
# ---------------------------------------------------------------------------

def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).mean()


def mean(x: pd.Series, n: int) -> pd.Series:
    return sma(x, n)


def stddev(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).std(ddof=0)


def ts_min(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).min()


def ts_max(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).max()


def ts_sum(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).sum()


def ts_prod(x: pd.Series, n: int) -> pd.Series:
    """Product over n; uses log-sum trick to avoid overflow.
    NaNs and non-positive entries get treated through abs+sign.
    """
    sgn = np.sign(x.to_numpy())
    abs_x = np.abs(x.to_numpy()) + EPS
    log_abs = np.log(abs_x)
    log_sum = pd.Series(log_abs, index=x.index).rolling(n, min_periods=n).sum()
    sgn_prod = pd.Series(sgn, index=x.index).rolling(n, min_periods=n).apply(
        np.prod, raw=True
    )
    return np.exp(log_sum) * sgn_prod


def delay(x: pd.Series, n: int) -> pd.Series:
    return x.shift(n)


def delta(x: pd.Series, n: int) -> pd.Series:
    return x - x.shift(n)


def returns(x: pd.Series, n: int = 1) -> pd.Series:
    """log returns (preserves sign for negative-priced normalized data via
    a fallback). For prices that may be negative, fall back to simple diff.
    """
    return x.pct_change(n)


def signedpower(x: pd.Series, p: float) -> pd.Series:
    arr = x.to_numpy()
    return pd.Series(np.sign(arr) * (np.abs(arr) ** p), index=x.index)


def log(x: pd.Series) -> pd.Series:
    arr = x.to_numpy()
    return pd.Series(np.sign(arr) * np.log1p(np.abs(arr)), index=x.index)


def abs_(x: pd.Series) -> pd.Series:
    return x.abs()


def sign(x: pd.Series) -> pd.Series:
    return pd.Series(np.sign(x.to_numpy()), index=x.index)


# ---------------------------------------------------------------------------
# Time-series rank / argmax / argmin (single-stock only — sym-agnostic)
# ---------------------------------------------------------------------------

def ts_rank(x: pd.Series, n: int) -> pd.Series:
    """Rank within rolling window of size n. Returns rank/n in [0,1].
    Uses pandas rolling apply with method='average'.
    Vectorized via numpy on numpy chunks for speed.
    """
    # Fast vectorized rolling rank using pandas
    return x.rolling(n, min_periods=n).rank(pct=True)


def ts_argmax(x: pd.Series, n: int) -> pd.Series:
    """Index of max within rolling window (1..n)."""
    # pandas .rolling(...).apply(np.argmax) is slow but correct
    arr = x.to_numpy()
    # Use a vectorized stride trick approach
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) >= n:
        # build a rolling window matrix (T-n+1, n)
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window_shape=n)  # shape (T-n+1, n)
        idx = np.argmax(sw, axis=1).astype(np.float64) + 1.0
        out[n - 1:] = idx
    return pd.Series(out, index=x.index)


def ts_argmin(x: pd.Series, n: int) -> pd.Series:
    arr = x.to_numpy()
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) >= n:
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window_shape=n)
        idx = np.argmin(sw, axis=1).astype(np.float64) + 1.0
        out[n - 1:] = idx
    return pd.Series(out, index=x.index)


def correlation(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).corr(y)


def covariance(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=n).cov(y)


def scale(x: pd.Series, a: float = 1.0) -> pd.Series:
    """Rescale so sum(|x|) = a. NOT cross-sectional — applied over the full
    session vector (sym-agnostic since one session = one stock).
    """
    s = x.abs().sum()
    if s < EPS:
        return x * 0.0
    return x * (a / s)


def decay_linear(x: pd.Series, n: int) -> pd.Series:
    """Linearly weighted moving average over the past n ticks.
    weights = (n, n-1, ..., 1) / sum(weights).
    """
    arr = x.to_numpy()
    w = np.arange(1, n + 1, dtype=np.float64)
    w = w / w.sum()
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) >= n:
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window_shape=n)
        out[n - 1:] = (sw * w[None, :]).sum(axis=1)
    return pd.Series(out, index=x.index)


def wma(x: pd.Series, n: int) -> pd.Series:
    """GTJA Alpha191's WMA (older weight slower decay):
    WMA(x, n) = sum_{i=1..n} 0.9^{i} * x_{n-i+1} / sum 0.9^i — i.e., heavier
    weight on recent. We follow the GTJA convention: weight = 0.9^(i-1)
    where i indexes from most recent (i=1) to oldest (i=n).
    """
    arr = x.to_numpy()
    w = 0.9 ** np.arange(n)  # weights[0] is most recent
    w = w / w.sum()
    out = np.full(len(arr), np.nan, dtype=np.float64)
    if len(arr) >= n:
        from numpy.lib.stride_tricks import sliding_window_view
        sw = sliding_window_view(arr, window_shape=n)
        # sliding_window_view returns oldest..newest in each row; reverse to newest..oldest
        sw_rev = sw[:, ::-1]
        out[n - 1:] = (sw_rev * w[None, :]).sum(axis=1)
    return pd.Series(out, index=x.index)


def sma_gtja(x: pd.Series, n: int, m: int = 1) -> pd.Series:
    """GTJA-style "SMA": Y[t] = (m * X[t] + (n-m) * Y[t-1]) / n
    This is an exponential smoother with alpha = m/n. Vectorized via ewm.
    """
    alpha = m / n
    return x.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


def regbeta(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    """Rolling regression beta cov(y, x, n) / var(x, n)."""
    cov = y.rolling(n, min_periods=n).cov(x)
    var = x.rolling(n, min_periods=n).var(ddof=0)
    return cov / (var + EPS)


def regresi(y: pd.Series, x: pd.Series, n: int) -> pd.Series:
    """Rolling regression residual at each point."""
    beta = regbeta(y, x, n)
    mean_y = y.rolling(n, min_periods=n).mean()
    mean_x = x.rolling(n, min_periods=n).mean()
    alpha = mean_y - beta * mean_x
    yhat = alpha + beta * x
    return y - yhat


def sumif(condition: pd.Series, x: pd.Series, n: int) -> pd.Series:
    """Sum of x over the window where condition is True."""
    masked = x.where(condition, 0.0)
    return masked.rolling(n, min_periods=n).sum()


def count_if(condition: pd.Series, n: int) -> pd.Series:
    """Count of True in window of size n."""
    return condition.astype(np.float64).rolling(n, min_periods=n).sum()


def highday(x: pd.Series, n: int) -> pd.Series:
    """How many days ago the highest value of x in the past n days occurred.
    GTJA: HIGHDAY(x, n) = n - 1 - argmax(x in last n days, where 0 is oldest).
    Equivalent: ts_argmax counted from newest = n - argmax_oldest + 1 - 1
    We return the number of ticks back from current: 0 = today, n-1 = oldest.
    """
    a = ts_argmax(x, n)  # 1..n, where n means current is the max
    return n - a


def lowday(x: pd.Series, n: int) -> pd.Series:
    a = ts_argmin(x, n)
    return n - a


# ---------------------------------------------------------------------------
# Convenience: extract OHLCV-ish vectors from session df
# ---------------------------------------------------------------------------

def base_series(df: pd.DataFrame) -> dict:
    """Returns dict of canonical price/volume Series for one session df.

    Uses bid1/ask1 to compute mid; volume_delta is the per-tick volume.
    OHLC come from columns directly (raw 154d).

    Note: prices/sizes here are PRE-NORMALIZED in the public dataset — they
    can be negative. helpers should not assume positivity.
    """
    b1 = df["bid1"]
    a1 = df["ask1"]
    bs1 = df["bsize1"]
    as1 = df["asize1"]
    mid = (a1 + b1) * 0.5

    # WMP level 1 (volume-weighted mid, level 1)
    denom = (bs1 + as1).replace(0, np.nan)
    vwap_l1 = (a1 * bs1 + b1 * as1) / denom
    vwap_l1 = vwap_l1.fillna(mid)

    # Multi-level VWAP across 10 levels (cleaner volume-weighted price proxy)
    bid_vols = sum(df[f"bsize{k}"] for k in range(1, 11))
    ask_vols = sum(df[f"asize{k}"] for k in range(1, 11))
    bid_dot = sum(df[f"bid{k}"] * df[f"bsize{k}"] for k in range(1, 11))
    ask_dot = sum(df[f"ask{k}"] * df[f"asize{k}"] for k in range(1, 11))
    total_v = (bid_vols + ask_vols).replace(0, np.nan)
    vwap = ((bid_dot + ask_dot) / total_v).fillna(mid)

    return {
        "open": df["open"].astype(np.float64),
        "high": df["high"].astype(np.float64),
        "low": df["low"].astype(np.float64),
        "close": df["close"].astype(np.float64),
        "volume": df["volume_delta"].astype(np.float64).abs(),  # |delta|, treat as size
        "amount": df["amount_delta"].astype(np.float64).abs(),
        "vwap": vwap.astype(np.float64),
        "vwap_l1": vwap_l1.astype(np.float64),
        "mid": mid.astype(np.float64),
        "bid1": b1.astype(np.float64),
        "ask1": a1.astype(np.float64),
        "bsize1": bs1.astype(np.float64),
        "asize1": as1.astype(np.float64),
        "spread": (a1 - b1).astype(np.float64),
        "ret": mid.pct_change().astype(np.float64),  # mid log-ish return
    }
