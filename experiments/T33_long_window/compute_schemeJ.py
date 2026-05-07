"""Inference-time feature computation for iter_006 Scheme J.

Computes 583-d Scheme J features from a 100-tick window DataFrame:
  * 154 raw last-tick (with log1p amount_delta)
  * 69 T3 last-tick (no time encoding)
  * 360 long-window features (A/B/C/D/E/F)

This is a self-contained module included in the submission zip — does not
import from T3_features_v1 / src.

Constraint compliance:
  - sym/date never used
  - All computations stateless (per-call window only)
  - Window-based stats computed on-the-fly from the 100×154 DataFrame
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

WINDOW = 100

# T3 LOB / MLOFI / WMP / RV / EWMA params (matches experiments/T3_features_v1/compute.py)
LOB_LEVELS = tuple(range(1, 11))
MLOFI_WINDOWS = (5, 20, 60)
RV_WINDOWS = (5, 10, 20, 50)
EWMA_ALPHAS = (0.05, 0.1, 0.3, 0.5)
INTST_COLS = ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst")

# Scheme J A. core 30 columns
CORE_COLS_A = (
    "open", "high", "low", "close",
    "volume_delta", "amount_delta",
    "bid1", "ask1", "bsize1", "asize1",
    "bid2", "ask2", "bsize2", "asize2",
    "avgbid", "avgask", "totalbsize", "totalasize",
    "bid_mean", "ask_mean", "bsize_mean", "asize_mean",
    "imbalance", "cumspread",
    "midprice1", "spread1",
    "bid_diff1", "ask_diff1",
    "mb_intst", "ma_intst",
)


# ---- T3 feature_v1 (no time encoding) ------------------------------------

def compute_mlofi(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    e_per_lvl: dict[int, pd.Series] = {}
    for k in LOB_LEVELS:
        b = df[f"bid{k}"]; a = df[f"ask{k}"]
        bs = df[f"bsize{k}"]; as_ = df[f"asize{k}"]
        b_prev = b.shift(1); a_prev = a.shift(1)
        bs_prev = bs.shift(1); as_prev = as_.shift(1)
        ind_b_up = (b >= b_prev).astype(np.float64)
        ind_b_dn = (b <= b_prev).astype(np.float64)
        ind_a_dn = (a <= a_prev).astype(np.float64)
        ind_a_up = (a >= a_prev).astype(np.float64)
        e = ind_b_up * bs - ind_b_dn * bs_prev - ind_a_dn * as_ + ind_a_up * as_prev
        e_per_lvl[k] = e
    for W in MLOFI_WINDOWS:
        for k in LOB_LEVELS:
            out[f"mlofi_W{W}_lvl{k}"] = (
                e_per_lvl[k].rolling(window=W, min_periods=W).sum()
            )
    return pd.DataFrame(out, index=df.index)


def compute_wmp(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for k in LOB_LEVELS:
        b = df[f"bid{k}"].to_numpy()
        a = df[f"ask{k}"].to_numpy()
        bs = df[f"bsize{k}"].to_numpy()
        as_ = df[f"asize{k}"].to_numpy()
        denom = bs + as_
        wmp_normal = (a * bs + b * as_) / np.where(denom == 0, 1.0, denom)
        wmp_fallback = (a + b) / 2.0
        wmp = np.where(denom == 0, wmp_fallback, wmp_normal)
        out[f"wmp_lvl{k}"] = wmp
    df_out = pd.DataFrame(out, index=df.index)
    df_out["wmp_balance_12"] = df_out["wmp_lvl1"] - df_out["wmp_lvl2"]
    return df_out


def compute_rv(wmp_lvl1: pd.Series) -> pd.DataFrame:
    safe = wmp_lvl1 + 1.0
    safe = safe.where(safe > 0, np.nan)
    log_ret = np.log(safe / safe.shift(1))
    sq = log_ret * log_ret
    out = {}
    for W in RV_WINDOWS:
        rolled = sq.rolling(window=W, min_periods=W).sum().clip(lower=0.0)
        out[f"rv_w{W}"] = np.sqrt(rolled)
    return pd.DataFrame(out, index=wmp_lvl1.index)


def compute_ewma_intst(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for alpha in EWMA_ALPHAS:
        for c in INTST_COLS:
            out[f"ewma_a{alpha}_{c}"] = df[c].ewm(alpha=alpha, adjust=False).mean()
    return pd.DataFrame(out, index=df.index)


def feature_v1_columns_no_time() -> List[str]:
    cols = []
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
    return cols


# ---- vectorized rolling helpers (cumsum-based) ---------------------------

def _csum(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])


def rolling_sum(x: np.ndarray, W: int) -> np.ndarray:
    n = len(x)
    cs = _csum(x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n >= W:
        idx = np.arange(W - 1, n)
        y[idx] = cs[idx + 1] - cs[idx + 1 - W]
    return y


def rolling_mean(x: np.ndarray, W: int) -> np.ndarray:
    return rolling_sum(x, W) / W


def rolling_var(x: np.ndarray, W: int) -> np.ndarray:
    n = len(x)
    cs = _csum(x); cs2 = _csum(x * x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n >= W:
        idx = np.arange(W - 1, n)
        s1 = cs[idx + 1] - cs[idx + 1 - W]
        s2 = cs2[idx + 1] - cs2[idx + 1 - W]
        m = s1 / W
        y[idx] = np.maximum(s2 / W - m * m, 0.0)
    return y


def rolling_std(x: np.ndarray, W: int) -> np.ndarray:
    return np.sqrt(rolling_var(x, W))


def rolling_min(x: np.ndarray, W: int) -> np.ndarray:
    return pd.Series(x, dtype=np.float64).rolling(W, min_periods=W).min().to_numpy()


def rolling_max(x: np.ndarray, W: int) -> np.ndarray:
    return pd.Series(x, dtype=np.float64).rolling(W, min_periods=W).max().to_numpy()


def rolling_quantile(x: np.ndarray, W: int, q: float) -> np.ndarray:
    return pd.Series(x, dtype=np.float64).rolling(W, min_periods=W).quantile(q).to_numpy()


def rolling_autocorr_lag_k(x: np.ndarray, W: int, k: int) -> np.ndarray:
    n = len(x)
    L = W - k
    y = np.full(n, np.nan, dtype=np.float64)
    if L < 2 or n < W:
        return y
    x = x.astype(np.float64)
    p = np.zeros(n, dtype=np.float64)
    p[: n - k] = x[: n - k] * x[k:]
    cs = _csum(x); cs2 = _csum(x * x); csp = _csum(p)
    t_arr = np.arange(W - 1, n)
    a = t_arr - W + 1; b1 = t_arr - k
    c = t_arr - W + 1 + k; b2 = t_arr
    sumA = cs[b1 + 1] - cs[a]; sumB = cs[b2 + 1] - cs[c]
    sumA2 = cs2[b1 + 1] - cs2[a]; sumB2 = cs2[b2 + 1] - cs2[c]
    sumCross = csp[b1 + 1] - csp[a]
    meanA = sumA / L; meanB = sumB / L
    varA = sumA2 / L - meanA * meanA
    varB = sumB2 / L - meanB * meanB
    cov = sumCross / L - meanA * meanB
    denom = np.sqrt(np.maximum(varA, 0.0) * np.maximum(varB, 0.0))
    ac = np.where(denom > 1e-10, cov / np.where(denom > 1e-10, denom, 1.0), 0.0)
    y[t_arr] = ac
    return y


def rolling_slope_W(x: np.ndarray, W: int) -> np.ndarray:
    n = len(x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n < W:
        return y
    x = x.astype(np.float64)
    cs = _csum(x)
    csum_ix = _csum(np.arange(n, dtype=np.float64) * x)
    t_arr = np.arange(W - 1, n)
    a = t_arr - W + 1
    sum_x = cs[t_arr + 1] - cs[a]
    sum_ix = csum_ix[t_arr + 1] - csum_ix[a]
    sum_local = sum_ix - a * sum_x
    mean_i = (W - 1) / 2.0
    num = sum_local - mean_i * sum_x
    den = W * (W * W - 1) / 12.0
    y[t_arr] = num / den
    return y


def rolling_corr_with_time_W(x: np.ndarray, W: int) -> np.ndarray:
    n = len(x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n < W:
        return y
    x = x.astype(np.float64)
    cs = _csum(x); cs2 = _csum(x * x)
    csum_ix = _csum(np.arange(n, dtype=np.float64) * x)
    t_arr = np.arange(W - 1, n)
    a = t_arr - W + 1
    sum_x = cs[t_arr + 1] - cs[a]
    sum_x2 = cs2[t_arr + 1] - cs2[a]
    sum_ix = csum_ix[t_arr + 1] - csum_ix[a]
    sum_local = sum_ix - a * sum_x
    mean_x = sum_x / W
    var_x = sum_x2 / W - mean_x * mean_x
    mean_i = (W - 1) / 2.0
    var_i = (W * W - 1) / 12.0
    cov = sum_local / W - mean_i * mean_x
    denom = np.sqrt(np.maximum(var_x, 0.0) * var_i)
    y[t_arr] = np.where(denom > 1e-10, cov / np.where(denom > 1e-10, denom, 1.0), 0.0)
    return y


# ---- Long-window feature computation ------------------------------------

def _last_minus_first_W(x: np.ndarray, W: int) -> np.ndarray:
    n = len(x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n >= W:
        y[W - 1 :] = x[W - 1 :].astype(np.float64) - x[: n - W + 1].astype(np.float64)
    return y


def _ewma_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    return pd.Series(x, dtype=np.float64).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def _compute_mlofi_e_per_lvl(df: pd.DataFrame) -> dict:
    out = {}
    for k in LOB_LEVELS:
        b = df[f"bid{k}"].to_numpy(dtype=np.float64)
        a = df[f"ask{k}"].to_numpy(dtype=np.float64)
        bs = df[f"bsize{k}"].to_numpy(dtype=np.float64)
        as_ = df[f"asize{k}"].to_numpy(dtype=np.float64)
        b_prev = np.empty_like(b); b_prev[0] = np.nan; b_prev[1:] = b[:-1]
        a_prev = np.empty_like(a); a_prev[0] = np.nan; a_prev[1:] = a[:-1]
        bs_prev = np.empty_like(bs); bs_prev[0] = np.nan; bs_prev[1:] = bs[:-1]
        as_prev = np.empty_like(as_); as_prev[0] = np.nan; as_prev[1:] = as_[:-1]
        ind_b_up = (b >= b_prev).astype(np.float64)
        ind_b_dn = (b <= b_prev).astype(np.float64)
        ind_a_dn = (a <= a_prev).astype(np.float64)
        ind_a_up = (a >= a_prev).astype(np.float64)
        e = ind_b_up * bs - ind_b_dn * bs_prev - ind_a_dn * as_ + ind_a_up * as_prev
        e[0] = 0.0
        out[k] = e
    return out


def compute_long_window(df: pd.DataFrame, t3_df: pd.DataFrame) -> np.ndarray:
    """Returns a (T, 360) array. Rows where t < WINDOW-1 are NaN.

    Caller passes both df (raw 154 cols) and t3_df (T3 derived cols incl. wmp_lvl1).
    """
    T = len(df)
    out_arrs: List[np.ndarray] = []

    # ---- A. Summary stats per core feature (10 stats × 30) ----
    for col in CORE_COLS_A:
        x = df[col].to_numpy(dtype=np.float64, copy=False)
        if col == "amount_delta":
            x = np.sign(x) * np.log1p(np.abs(x))
        for W_ in (20, 50, 100):
            out_arrs.append(rolling_mean(x, W_))
        for W_ in (20, 100):
            out_arrs.append(rolling_std(x, W_))
        m100 = rolling_mean(x, 100)
        out_arrs.append(x - m100)
        out_arrs.append(_last_minus_first_W(x, 100))
        rmax = rolling_max(x, 100); rmin = rolling_min(x, 100)
        out_arrs.append(rmax - rmin)
        out_arrs.append(rolling_quantile(x, 100, 0.25))
        out_arrs.append(rolling_quantile(x, 100, 0.75))

    # ---- B. Long-window OFI (20) ----
    e_per_lvl = _compute_mlofi_e_per_lvl(df)
    for k in LOB_LEVELS:
        out_arrs.append(rolling_sum(e_per_lvl[k], 100))
    for k in LOB_LEVELS:
        e = e_per_lvl[k]
        mlofi_W50 = rolling_sum(e, 50)
        accel = np.full(T, np.nan, dtype=np.float64)
        if T >= 100:
            accel[99:] = mlofi_W50[99:] - mlofi_W50[49 : T - 50]
        out_arrs.append(accel)

    # ---- C. Cross-tick autocorrelation (13) ----
    midprice1 = df["midprice1"].to_numpy(dtype=np.float64)
    mid_diff = np.empty_like(midprice1)
    mid_diff[0] = 0.0
    mid_diff[1:] = midprice1[1:] - midprice1[:-1]
    for kk in (1, 5, 10, 50):
        out_arrs.append(rolling_autocorr_lag_k(mid_diff, 100, kk))
    bid1 = df["bid1"].to_numpy(dtype=np.float64)
    ask1 = df["ask1"].to_numpy(dtype=np.float64)
    vol = df["volume_delta"].to_numpy(dtype=np.float64)
    for kk in (1, 5, 10):
        out_arrs.append(rolling_autocorr_lag_k(bid1, 100, kk))
    for kk in (1, 5, 10):
        out_arrs.append(rolling_autocorr_lag_k(ask1, 100, kk))
    for kk in (1, 5, 10):
        out_arrs.append(rolling_autocorr_lag_k(vol, 100, kk))

    # ---- D. EWMA + cumulative intensity (18) ----
    for c in INTST_COLS:
        out_arrs.append(_ewma_alpha(df[c].to_numpy(dtype=np.float64), 0.01))
    for c in INTST_COLS:
        out_arrs.append(_ewma_alpha(df[c].to_numpy(dtype=np.float64), 0.014))
    for c in INTST_COLS:
        out_arrs.append(rolling_sum(df[c].to_numpy(dtype=np.float64), 100))

    # ---- E. WMP1 trend (4) ----
    wmp1 = t3_df["wmp_lvl1"].to_numpy(dtype=np.float64)
    out_arrs.append(rolling_slope_W(wmp1, 100))
    m100 = rolling_mean(wmp1, 100); sd100 = rolling_std(wmp1, 100)
    out_arrs.append((wmp1 - m100) / np.where(sd100 > 1e-10, sd100, 1.0))
    sma20 = rolling_mean(wmp1, 20)
    out_arrs.append(wmp1 - sma20)
    out_arrs.append(rolling_corr_with_time_W(wmp1, 100))

    # ---- F. Tick-rule signed volume (5) ----
    sign = np.sign(mid_diff).astype(np.float64)
    sign_series = pd.Series(np.where(sign == 0, np.nan, sign)).ffill().fillna(0).to_numpy()
    signed_vol = sign_series * vol
    for W_ in (5, 20, 50, 100):
        out_arrs.append(rolling_sum(signed_vol, W_))
    out_arrs.append(rolling_sum(sign_series, 100))

    return np.stack(out_arrs, axis=1).astype(np.float32)


def long_window_columns() -> List[str]:
    """Names of the 360 long-window features (in computation order)."""
    names: List[str] = []
    for col in CORE_COLS_A:
        for W_ in (20, 50, 100):
            names.append(f"A_mean_W{W_}_{col}")
        for W_ in (20, 100):
            names.append(f"A_std_W{W_}_{col}")
        names.append(f"A_lastMm_W100_{col}")
        names.append(f"A_lastMfirst_W100_{col}")
        names.append(f"A_range_W100_{col}")
        names.append(f"A_q25_W100_{col}")
        names.append(f"A_q75_W100_{col}")
    for k in LOB_LEVELS:
        names.append(f"B_mlofi_W100_lvl{k}")
    for k in LOB_LEVELS:
        names.append(f"B_mlofi_accel_lvl{k}")
    for kk in (1, 5, 10, 50):
        names.append(f"C_ac_mid_diff_lag{kk}_W100")
    for kk in (1, 5, 10):
        names.append(f"C_ac_bid1_lag{kk}_W100")
    for kk in (1, 5, 10):
        names.append(f"C_ac_ask1_lag{kk}_W100")
    for kk in (1, 5, 10):
        names.append(f"C_ac_vol_lag{kk}_W100")
    for c in INTST_COLS:
        names.append(f"D_ewma_a0.01_{c}")
    for c in INTST_COLS:
        names.append(f"D_ewma_a0.014_{c}")
    for c in INTST_COLS:
        names.append(f"D_cumsum_W100_{c}")
    names.append("E_wmp1_slope_W100")
    names.append("E_wmp1_break_W100")
    names.append("E_wmp1_minus_sma20")
    names.append("E_wmp1_corr_t_W100")
    for W_ in (5, 20, 50, 100):
        names.append(f"F_signed_vol_W{W_}")
    names.append("F_sign_cumsum_W100")
    return names
