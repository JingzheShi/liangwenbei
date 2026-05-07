"""T3 feature_v1 computation: 72-d new features per session.

Per-session pure computation (input: DataFrame with 163 official cols, output:
DataFrame with 72 new feature cols). All operations are vectorized; no Python
for-loops over rows.

Feature groups (72 total):
  1. MLOFI: 10 levels x 3 windows = 30 cols
  2. WMP:   10 wmp_lvlk + 1 wmp_balance_12 = 11 cols
  3. RV:    4 windows = 4 cols
  4. EWMA intensities: 6 intst x 4 alphas = 24 cols
  5. Time encoding: 3 cols
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# ---- constants --------------------------------------------------------------

LOB_LEVELS = tuple(range(1, 11))  # 1..10
MLOFI_WINDOWS = (5, 20, 60)
RV_WINDOWS = (5, 10, 20, 50)
EWMA_ALPHAS = (0.05, 0.1, 0.3, 0.5)
INTST_COLS = ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst")

AM_START_SEC = 9 * 3600 + 40 * 60   # 34800
PM_START_SEC = 13 * 3600 + 10 * 60  # 47400
SESSION_DURATION_MIN = 100  # 100 minutes per session


# ---- 1. MLOFI ---------------------------------------------------------------

def compute_mlofi(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-Level Order Flow Imbalance (Cont-Stoikov-Kukanov 2014, Kolm 2023).

    Per level k=1..10:
        e_k(t) = I(b_k_t >= b_k_{t-1}) * bs_k_t - I(b_k_t <= b_k_{t-1}) * bs_k_{t-1}
               - I(a_k_t <= a_k_{t-1}) * as_k_t + I(a_k_t >= a_k_{t-1}) * as_k_{t-1}

    Then rolling sum over W in {5, 20, 60} → 10 x 3 = 30 features.
    """
    out = {}
    # raw per-tick e_k(t), level by level
    e_per_lvl: dict[int, pd.Series] = {}
    for k in LOB_LEVELS:
        b = df[f"bid{k}"]
        a = df[f"ask{k}"]
        bs = df[f"bsize{k}"]
        as_ = df[f"asize{k}"]
        b_prev = b.shift(1)
        a_prev = a.shift(1)
        bs_prev = bs.shift(1)
        as_prev = as_.shift(1)
        # boolean → cast to float for arithmetic; NaN survives via shift(1)
        ind_b_up = (b >= b_prev).astype(np.float64)
        ind_b_dn = (b <= b_prev).astype(np.float64)
        ind_a_dn = (a <= a_prev).astype(np.float64)
        ind_a_up = (a >= a_prev).astype(np.float64)
        e = ind_b_up * bs - ind_b_dn * bs_prev - ind_a_dn * as_ + ind_a_up * as_prev
        # The first row (b_prev = NaN) → e = NaN: pandas comparisons with NaN
        # return False, but multiplying with NaN bs_prev still gives NaN. Fine.
        e_per_lvl[k] = e

    # rolling sum
    for W in MLOFI_WINDOWS:
        for k in LOB_LEVELS:
            out[f"mlofi_W{W}_lvl{k}"] = (
                e_per_lvl[k].rolling(window=W, min_periods=W).sum()
            )

    return pd.DataFrame(out, index=df.index)


# ---- 2. WMP -----------------------------------------------------------------

def compute_wmp(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted Mid-Price (Stoikov microprice).

    For level k:
        WMP_k = a_k * bs_k / (bs_k + as_k) + b_k * as_k / (bs_k + as_k)
    Fallback: when (bs_k + as_k) == 0 → (a_k + b_k) / 2.

    Plus wmp_balance_12 = wmp_lvl1 - wmp_lvl2.
    """
    out = {}
    for k in LOB_LEVELS:
        b = df[f"bid{k}"].to_numpy()
        a = df[f"ask{k}"].to_numpy()
        bs = df[f"bsize{k}"].to_numpy()
        as_ = df[f"asize{k}"].to_numpy()
        denom = bs + as_
        # vectorized fallback
        wmp_normal = (a * bs + b * as_) / np.where(denom == 0, 1.0, denom)
        wmp_fallback = (a + b) / 2.0
        wmp = np.where(denom == 0, wmp_fallback, wmp_normal)
        out[f"wmp_lvl{k}"] = wmp

    out_df = pd.DataFrame(out, index=df.index)
    out_df["wmp_balance_12"] = out_df["wmp_lvl1"] - out_df["wmp_lvl2"]
    return out_df


# ---- 3. RV ------------------------------------------------------------------

def compute_rv(wmp_lvl1: pd.Series, windows: Iterable[int] = RV_WINDOWS) -> pd.DataFrame:
    """Multi-window Realized Volatility on log-returns of (WMP1 + 1).

    log_return(t) = log((WMP1(t) + 1) / (WMP1(t-1) + 1))
    RV_W(t) = sqrt(sum_{s=t-W+1..t} log_return(s)^2)
    """
    # (price+1) ∈ (0, 2) since price ∈ (-1, 1); guard against ≤0 with NaN
    safe = wmp_lvl1 + 1.0
    safe = safe.where(safe > 0, np.nan)
    log_ret = np.log(safe / safe.shift(1))
    sq = log_ret * log_ret
    out = {}
    for W in windows:
        # Rolling sum of squares can produce tiny negative values (~1e-22) from
        # floating-point roundoff when most returns are zero; clip before sqrt.
        rolled = sq.rolling(window=W, min_periods=W).sum().clip(lower=0.0)
        rv = np.sqrt(rolled)
        out[f"rv_w{W}"] = rv
    return pd.DataFrame(out, index=wmp_lvl1.index)


# ---- 4. EWMA intensities ----------------------------------------------------

def compute_ewma_intst(df: pd.DataFrame,
                       alphas: Iterable[float] = EWMA_ALPHAS,
                       cols: Iterable[str] = INTST_COLS) -> pd.DataFrame:
    """Multi-alpha EWMA over each *_intst column (Hawkes proxy)."""
    out = {}
    for alpha in alphas:
        for c in cols:
            ewma = df[c].ewm(alpha=alpha, adjust=False).mean()
            out[f"ewma_a{alpha}_{c}"] = ewma
    return pd.DataFrame(out, index=df.index)


# ---- 5. Time encoding -------------------------------------------------------

def _time_to_seconds(time_series: pd.Series) -> np.ndarray:
    """Convert a pandas Series of datetime.time → integer seconds since midnight (vectorized).

    Uses pd.to_datetime parsing on the str repr; ~1ms / 2000 rows.
    """
    s = time_series.astype(str)
    dt = pd.to_datetime(s, format="%H:%M:%S", errors="coerce")
    seconds = (dt.dt.hour.astype(np.int64) * 3600
               + dt.dt.minute.astype(np.int64) * 60
               + dt.dt.second.astype(np.int64))
    return seconds.to_numpy()


def detect_session(df: pd.DataFrame) -> str:
    """Detect session ('am' or 'pm') from the first time value."""
    first = df["time"].iloc[0]
    # datetime.time.hour
    h = first.hour if hasattr(first, "hour") else int(str(first).split(":")[0])
    return "am" if h < 12 else "pm"


def compute_time_encoding(df: pd.DataFrame, session: Optional[str] = None) -> pd.DataFrame:
    """Time-in-session encoding (3 features).

    - time_minutes_since_session_start: int 0..99
    - time_session_progress: float 0..1
    - time_is_pm: 0 or 1
    """
    if session is None:
        session = detect_session(df)
    if session == "am":
        start = AM_START_SEC
        is_pm = 0
    elif session == "pm":
        start = PM_START_SEC
        is_pm = 1
    else:
        raise ValueError(f"session must be 'am' or 'pm', got {session!r}")

    secs = _time_to_seconds(df["time"]) - start
    minutes = (secs // 60).astype(np.int64)
    minutes = np.clip(minutes, 0, SESSION_DURATION_MIN - 1)  # clip to [0, 99]
    progress = minutes.astype(np.float64) / (SESSION_DURATION_MIN - 1)

    return pd.DataFrame(
        {
            "time_minutes_since_session_start": minutes.astype(np.int32),
            "time_session_progress": progress.astype(np.float32),
            "time_is_pm": np.full(len(df), is_pm, dtype=np.int8),
        },
        index=df.index,
    )


# ---- aggregator -------------------------------------------------------------

def compute_all(df: pd.DataFrame, session: Optional[str] = None) -> pd.DataFrame:
    """Compute all 72 new features for one session DataFrame.

    Returns a DataFrame of 72 cols aligned to df.index.
    """
    mlofi = compute_mlofi(df)
    wmp = compute_wmp(df)
    rv = compute_rv(wmp["wmp_lvl1"])
    ewma = compute_ewma_intst(df)
    timef = compute_time_encoding(df, session=session)
    return pd.concat([mlofi, wmp, rv, ewma, timef], axis=1)


def feature_v1_columns() -> list[str]:
    """Return the 72 column names that compute_all produces (in order)."""
    cols: list[str] = []
    for W in MLOFI_WINDOWS:
        for k in LOB_LEVELS:
            cols.append(f"mlofi_W{W}_lvl{k}")
    for k in LOB_LEVELS:
        cols.append(f"wmp_lvl{k}")
    cols.append("wmp_balance_12")
    for W in RV_WINDOWS:
        cols.append(f"rv_w{W}")
    for alpha in EWMA_ALPHAS:
        for c in INTST_COLS:
            cols.append(f"ewma_a{alpha}_{c}")
    cols += ["time_minutes_since_session_start", "time_session_progress", "time_is_pm"]
    return cols
