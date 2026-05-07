"""T33 Scheme J: long-window features (target ~583-d).

Adds to base 223-d (Scheme C minus 3 time-encoding):
  A. Per-feature summary stats (10 stats × 30 core feats = 300)
       mean_W{20,50,100}, std_W{20,100}, last_minus_mean_W100,
       last_minus_first, range_W100, q25_W100, q75_W100
  B. Long-window OFI (20)
       mlofi_W100_lvl{1..10}, mlofi_accel_lvl{1..10}
  C. Cross-tick autocorrelation (13)
       mid_diff lag {1,5,10,50}, bid1/ask1 lag {1,5,10}, vol lag {1,5,10}
  D. Long-half-life EWMA + cumulative intensity (18)
       ewma_a0.01_*, ewma_a0.014_*, cumsum_W100_*  (× 6 intst cols)
  E. WMP1 trend (4)
       wmp1_slope_W100, wmp1_break, wmp1_minus_sma20, wmp1_corr_t_W100
  F. Tick-rule signed volume (5)
       signed_vol_W{5,20,50,100}, sign_cumsum_W100

All computations are vectorized via numpy cumulative sums (O(T) per feature
per session). All features use only data from within the 100-tick window
ending at t — sym-agnostic, stateless, no date.

Saves cache .npz keyed by split (train/val/test):
  X (N, F)  float32    F = 223 + 360 = 583
  y5,y10,y20,y40,y60   int8
  mp_t, mp_t{H}        float32
  sym, date, sess_idx, t  int

Also writes feat_names.txt (583 lines).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.data.split import get_split  # noqa: E402
from src.data.dataset import get_default_feature_cols  # noqa: E402

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}

# T3 derived features without time encoding (matches schemeC_223d_feat_names.txt)
T3_NON_TIME = (
    [f"mlofi_W{W}_lvl{k}" for W in (5, 20, 60) for k in range(1, 11)]
    + [f"wmp_lvl{k}" for k in range(1, 11)]
    + ["wmp_balance_12"]
    + [f"rv_w{W}" for W in (5, 10, 20, 50)]
    + [f"ewma_a{a}_{c}" for a in (0.05, 0.1, 0.3, 0.5)
       for c in ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst")]
)

# 30 core features for A (summary stats)
CORE_COLS = [
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
]
INTST_COLS = ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst")


# ---- vectorized rolling helpers (numpy cumsum-based) ----------------------

def _csum(x: np.ndarray) -> np.ndarray:
    """Return cumsum prepended with 0 so window sum = csum[b+1]-csum[a]."""
    return np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])


def rolling_sum(x: np.ndarray, W: int) -> np.ndarray:
    """y[t] = sum x[t-W+1..t]; t < W-1 -> NaN."""
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
    """Population variance over W ticks ending at t."""
    n = len(x)
    cs = _csum(x)
    cs2 = _csum(x * x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n >= W:
        idx = np.arange(W - 1, n)
        s1 = cs[idx + 1] - cs[idx + 1 - W]
        s2 = cs2[idx + 1] - cs2[idx + 1 - W]
        m = s1 / W
        v = s2 / W - m * m
        y[idx] = np.maximum(v, 0.0)
    return y


def rolling_std(x: np.ndarray, W: int) -> np.ndarray:
    return np.sqrt(rolling_var(x, W))


def rolling_min(x: np.ndarray, W: int) -> np.ndarray:
    """y[t] = min x[t-W+1..t]; via pandas (C-impl, deque-based)."""
    s = pd.Series(x, dtype=np.float64)
    return s.rolling(W, min_periods=W).min().to_numpy()


def rolling_max(x: np.ndarray, W: int) -> np.ndarray:
    s = pd.Series(x, dtype=np.float64)
    return s.rolling(W, min_periods=W).max().to_numpy()


def rolling_quantile(x: np.ndarray, W: int, q: float) -> np.ndarray:
    s = pd.Series(x, dtype=np.float64)
    return s.rolling(W, min_periods=W).quantile(q).to_numpy()


def rolling_autocorr_lag_k(x: np.ndarray, W: int, k: int) -> np.ndarray:
    """Pearson autocorr at lag k over W-tick window ending at t.

    Using only data in [t-W+1, t]. Pairs (x[i], x[i+k]) for i in [t-W+1, t-k]
    -> L = W - k pairs. Vectorized with cumulative sums.
    """
    n = len(x)
    L = W - k
    y = np.full(n, np.nan, dtype=np.float64)
    if L < 2 or n < W:
        return y
    x = x.astype(np.float64)
    # cross product p[i] = x[i] * x[i+k]; valid for i in [0, n-1-k]
    p = np.zeros(n, dtype=np.float64)
    p[: n - k] = x[: n - k] * x[k:]
    cs = _csum(x)
    cs2 = _csum(x * x)
    csp = _csum(p)
    t_arr = np.arange(W - 1, n)
    a = t_arr - W + 1     # start of left subset (length L)
    b1 = t_arr - k        # end of left subset
    c = t_arr - W + 1 + k # start of right subset (length L)
    b2 = t_arr            # end of right subset
    sumA = cs[b1 + 1] - cs[a]
    sumB = cs[b2 + 1] - cs[c]
    sumA2 = cs2[b1 + 1] - cs2[a]
    sumB2 = cs2[b2 + 1] - cs2[c]
    # cross sum_{i=a..b1} p[i] = sum x[i]*x[i+k] over that window
    sumCross = csp[b1 + 1] - csp[a]
    meanA = sumA / L
    meanB = sumB / L
    varA = sumA2 / L - meanA * meanA
    varB = sumB2 / L - meanB * meanB
    cov = sumCross / L - meanA * meanB
    denom = np.sqrt(np.maximum(varA, 0.0) * np.maximum(varB, 0.0))
    ac = np.where(denom > 1e-10, cov / np.where(denom > 1e-10, denom, 1.0), 0.0)
    y[t_arr] = ac
    return y


def rolling_slope_W(x: np.ndarray, W: int) -> np.ndarray:
    """Linear regression slope of x against position (0..W-1) over W-tick window.

    slope = sum (i_local - mean_i)(x_i - mean_x) / sum (i_local - mean_i)^2
          = (sum_{i_local=0..W-1} i_local * x_i - mean_i * sum x_i) / Const
    Const = W*(W^2-1)/12
    """
    n = len(x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n < W:
        return y
    x = x.astype(np.float64)
    cs = _csum(x)
    # csum_ix[i+1] = sum_{j=0..i} j * x[j]
    csum_ix = _csum(np.arange(n, dtype=np.float64) * x)
    t_arr = np.arange(W - 1, n)
    a = t_arr - W + 1
    sum_x = cs[t_arr + 1] - cs[a]
    sum_ix = csum_ix[t_arr + 1] - csum_ix[a]
    # sum_{i_local=0..W-1} i_local * x_i = sum (i - a) x_i = sum i*x_i - a*sum_x
    sum_local = sum_ix - a * sum_x
    mean_i = (W - 1) / 2.0
    num = sum_local - mean_i * sum_x
    den = W * (W * W - 1) / 12.0
    y[t_arr] = num / den
    return y


def rolling_corr_with_time_W(x: np.ndarray, W: int) -> np.ndarray:
    """Pearson corr between x[t-W+1..t] and time index (0..W-1) over W ticks."""
    n = len(x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n < W:
        return y
    x = x.astype(np.float64)
    cs = _csum(x)
    cs2 = _csum(x * x)
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


# ---- per-session feature build ------------------------------------------

def _last_minus_first_W(x: np.ndarray, W: int) -> np.ndarray:
    """y[t] = x[t] - x[t-W+1]; t < W-1 -> NaN."""
    n = len(x)
    y = np.full(n, np.nan, dtype=np.float64)
    if n >= W:
        y[W - 1 :] = x[W - 1 :].astype(np.float64) - x[: n - W + 1].astype(np.float64)
    return y


def _ewma_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """EWMA with adjust=False: y_t = alpha*x_t + (1-alpha)*y_{t-1}, y_0 = x_0."""
    s = pd.Series(x, dtype=np.float64).ewm(alpha=alpha, adjust=False).mean()
    return s.to_numpy()


def _compute_mlofi_e_per_lvl(df: pd.DataFrame) -> dict:
    """Per-tick e_k(t) for each LOB level k=1..10 (matches T3 compute_mlofi)."""
    out = {}
    for k in range(1, 11):
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
        # First row: NaN (matches T3); replace with 0 for cumsum stability
        e[0] = 0.0
        out[k] = e
    return out


def build_long_window_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """For one session, compute long-window features for every t in [WINDOW-1, T-1].

    Returns (X_new, feat_names) where X_new shape = (T, N_NEW_FEAT). Rows where
    t < WINDOW-1 are NaN (will be sliced out by caller).
    """
    T = len(df)
    out_arrs: List[np.ndarray] = []
    out_names: List[str] = []

    # ---- A. Per-feature summary stats over rolling W -----------------------
    for col in CORE_COLS:
        x = df[col].to_numpy(dtype=np.float64, copy=False)
        # log1p sign-preserving for amount_delta to keep scale sensible
        if col == "amount_delta":
            x = np.sign(x) * np.log1p(np.abs(x))
        for W_ in (20, 50, 100):
            out_arrs.append(rolling_mean(x, W_))
            out_names.append(f"A_mean_W{W_}_{col}")
        for W_ in (20, 100):
            out_arrs.append(rolling_std(x, W_))
            out_names.append(f"A_std_W{W_}_{col}")
        m100 = rolling_mean(x, 100)
        out_arrs.append(x - m100)
        out_names.append(f"A_lastMm_W100_{col}")
        out_arrs.append(_last_minus_first_W(x, 100))
        out_names.append(f"A_lastMfirst_W100_{col}")
        rmax = rolling_max(x, 100)
        rmin = rolling_min(x, 100)
        out_arrs.append(rmax - rmin)
        out_names.append(f"A_range_W100_{col}")
        out_arrs.append(rolling_quantile(x, 100, 0.25))
        out_names.append(f"A_q25_W100_{col}")
        out_arrs.append(rolling_quantile(x, 100, 0.75))
        out_names.append(f"A_q75_W100_{col}")

    # ---- B. Long-window OFI -----------------------------------------------
    e_per_lvl = _compute_mlofi_e_per_lvl(df)
    for k in range(1, 11):
        e = e_per_lvl[k]
        mlofi_W100 = rolling_sum(e, 100)
        out_arrs.append(mlofi_W100)
        out_names.append(f"B_mlofi_W100_lvl{k}")
    for k in range(1, 11):
        e = e_per_lvl[k]
        mlofi_W50 = rolling_sum(e, 50)
        # acceleration: 2nd half (last 50) minus 1st half (first 50 of 100 window)
        # = mlofi_W50(t) - mlofi_W50(t-50)
        accel = np.full(T, np.nan, dtype=np.float64)
        if T >= 100:
            accel[99:] = mlofi_W50[99:] - mlofi_W50[49 : T - 50]
        out_arrs.append(accel)
        out_names.append(f"B_mlofi_accel_lvl{k}")

    # ---- C. Cross-tick autocorrelation -----------------------------------
    midprice1 = df["midprice1"].to_numpy(dtype=np.float64)
    mid_diff = np.empty_like(midprice1)
    mid_diff[0] = 0.0
    mid_diff[1:] = midprice1[1:] - midprice1[:-1]
    for k in (1, 5, 10, 50):
        out_arrs.append(rolling_autocorr_lag_k(mid_diff, 100, k))
        out_names.append(f"C_ac_mid_diff_lag{k}_W100")
    bid1 = df["bid1"].to_numpy(dtype=np.float64)
    ask1 = df["ask1"].to_numpy(dtype=np.float64)
    vol = df["volume_delta"].to_numpy(dtype=np.float64)
    for k in (1, 5, 10):
        out_arrs.append(rolling_autocorr_lag_k(bid1, 100, k))
        out_names.append(f"C_ac_bid1_lag{k}_W100")
    for k in (1, 5, 10):
        out_arrs.append(rolling_autocorr_lag_k(ask1, 100, k))
        out_names.append(f"C_ac_ask1_lag{k}_W100")
    for k in (1, 5, 10):
        out_arrs.append(rolling_autocorr_lag_k(vol, 100, k))
        out_names.append(f"C_ac_vol_lag{k}_W100")

    # ---- D. Long-half-life EWMA + cumulative intensity --------------------
    for c in INTST_COLS:
        x = df[c].to_numpy(dtype=np.float64)
        out_arrs.append(_ewma_alpha(x, 0.01))
        out_names.append(f"D_ewma_a0.01_{c}")
    for c in INTST_COLS:
        x = df[c].to_numpy(dtype=np.float64)
        out_arrs.append(_ewma_alpha(x, 0.014))
        out_names.append(f"D_ewma_a0.014_{c}")
    for c in INTST_COLS:
        x = df[c].to_numpy(dtype=np.float64)
        out_arrs.append(rolling_sum(x, 100))
        out_names.append(f"D_cumsum_W100_{c}")

    # ---- E. WMP1 trend ---------------------------------------------------
    wmp1 = df["wmp_lvl1"].to_numpy(dtype=np.float64) if "wmp_lvl1" in df.columns \
           else 0.5 * (df["bid1"].to_numpy(dtype=np.float64) + df["ask1"].to_numpy(dtype=np.float64))
    slope = rolling_slope_W(wmp1, 100)
    out_arrs.append(slope)
    out_names.append("E_wmp1_slope_W100")
    m100 = rolling_mean(wmp1, 100)
    sd100 = rolling_std(wmp1, 100)
    break_ = (wmp1 - m100) / np.where(sd100 > 1e-10, sd100, 1.0)
    out_arrs.append(break_)
    out_names.append("E_wmp1_break_W100")
    sma20 = rolling_mean(wmp1, 20)
    out_arrs.append(wmp1 - sma20)
    out_names.append("E_wmp1_minus_sma20")
    out_arrs.append(rolling_corr_with_time_W(wmp1, 100))
    out_names.append("E_wmp1_corr_t_W100")

    # ---- F. Tick-rule signed volume --------------------------------------
    sign = np.sign(mid_diff).astype(np.float64)
    # carry-forward for zeros (pandas ffill on 0->NaN)
    sign_series = pd.Series(np.where(sign == 0, np.nan, sign)).ffill().fillna(0).to_numpy()
    signed_vol = sign_series * vol
    for W_ in (5, 20, 50, 100):
        out_arrs.append(rolling_sum(signed_vol, W_))
        out_names.append(f"F_signed_vol_W{W_}")
    out_arrs.append(rolling_sum(sign_series, 100))
    out_names.append("F_sign_cumsum_W100")

    X_new = np.stack(out_arrs, axis=1).astype(np.float32)
    return X_new, out_names


# ---- per-split build ------------------------------------------------------

def build_split(sym_dates, base_feat_cols: List[str], cache_dir: str, out_dir: str,
                split_name: str) -> Tuple[np.ndarray, List[str]]:
    Xs = []
    ys = {h: [] for h in HORIZONS}
    mps = {f"mp_t{h}": [] for h in HORIZONS}
    mp_t_lst = []
    syms, dates, sess_idxs, ts = [], [], [], []

    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()

    new_feat_names: List[str] = []
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(cache_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path)
        T = len(df)
        valid_lo = WINDOW - 1
        valid_hi = T - 1 - MAX_HORIZON
        if valid_hi < valid_lo:
            continue

        # Base 223-d: 154 raw + 69 T3 non-time
        X_base = df[base_feat_cols].iloc[valid_lo : valid_hi + 1].to_numpy(
            dtype=np.float32, copy=False
        )
        # log1p amount_delta (preserve sign), matches Scheme C
        if "amount_delta" in base_feat_cols:
            j = base_feat_cols.index("amount_delta")
            col = X_base[:, j].copy()
            X_base = X_base.copy()
            X_base[:, j] = np.sign(col) * np.log1p(np.abs(col))

        X_new_full, names_new = build_long_window_features(df)
        if not new_feat_names:
            new_feat_names = names_new
        X_new = X_new_full[valid_lo : valid_hi + 1]

        X_combined = np.concatenate([X_base, X_new], axis=1).astype(np.float32, copy=False)
        Xs.append(X_combined)

        midprice = df["midprice"].to_numpy(dtype=np.float32, copy=False)
        mp_t_lst.append(midprice[valid_lo : valid_hi + 1])

        for h in HORIZONS:
            y = df[f"label_{h}"].iloc[valid_lo : valid_hi + 1].to_numpy(
                dtype=np.int8, copy=False
            )
            ys[h].append(y)
            mp_h = midprice[valid_lo + h : valid_hi + 1 + h]
            mps[f"mp_t{h}"].append(mp_h)

        n = X_combined.shape[0]
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(np.arange(valid_lo, valid_hi + 1, dtype=np.int16))

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_sessions - i - 1)
            print(f"  [{i+1}/{n_sessions}] sessions; elapsed {elapsed:.1f}s ETA {eta:.0f}s",
                  flush=True)

    out = {
        "X": np.concatenate(Xs, axis=0),
        "mp_t": np.concatenate(mp_t_lst, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    for h in HORIZONS:
        out[f"y{h}"] = np.concatenate(ys[h], axis=0)
        out[f"mp_t{h}"] = np.concatenate(mps[f"mp_t{h}"], axis=0)
    print(f"  -> X shape {out['X'].shape}  mem {out['X'].nbytes/1e9:.2f}GB", flush=True)

    out_path = os.path.join(out_dir, f"schemeJ_{split_name}.npz")
    np.savez(out_path, **out)
    print(f"  saved -> {out_path}", flush=True)

    # NaN check
    n_nan = np.isnan(out["X"]).sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN in X (will fillna 0)", flush=True)
    return out["X"], new_feat_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--cache-dir", default="data/features_v1")
    ap.add_argument("--out-dir", default="experiments/T33_long_window/cache")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_cols = get_default_feature_cols()
    base_feat_cols = raw_cols + list(T3_NON_TIME)
    print(f"Base 223-d: 154 raw + 69 T3 non-time = {len(base_feat_cols)}", flush=True)

    # Quick smoke test: probe a single session, ensure no NaN at valid_lo and below
    df0 = pd.read_parquet(os.path.join(args.cache_dir, "snapshot_sym0_date0_am.parquet"))
    X_new, names_new = build_long_window_features(df0)
    print(f"Long-window new features: {X_new.shape[1]}-d, total Scheme J = "
          f"{len(base_feat_cols) + X_new.shape[1]}", flush=True)
    n_nan_base = df0[base_feat_cols].iloc[WINDOW - 1 :].isna().any().any()
    n_nan_new = np.isnan(X_new[WINDOW - 1 :]).any()
    print(f"  smoke test: base NaN={n_nan_base}  new NaN={n_nan_new} "
          f"(after t>={WINDOW-1})", flush=True)
    if n_nan_new:
        # pinpoint which feature has NaN
        idx_nan = np.where(np.isnan(X_new[WINDOW - 1 :]).any(axis=0))[0]
        print(f"  NaN in new feature indices: {idx_nan[:20]} (showing first 20)", flush=True)
        for j in idx_nan[:5]:
            print(f"     {names_new[j]}", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    saved_names: List[str] = []
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True); continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        _, names_new = build_split(split[key], base_feat_cols, args.cache_dir,
                                    args.out_dir, key)
        saved_names = names_new

    if saved_names:
        all_names = list(base_feat_cols) + list(saved_names)
        names_path = os.path.join(args.out_dir, "schemeJ_feat_names.txt")
        with open(names_path, "w") as f:
            for n in all_names:
                f.write(n + "\n")
        print(f"feature names ({len(all_names)} total) -> {names_path}", flush=True)


if __name__ == "__main__":
    main()
