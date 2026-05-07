"""T51 R34 Stage 2 feature computations (sym-invariant, complementary to T44 stage 1).

Includes 5 feature families:
  #4  Window quantile rank (A2):    20 cols x rank within trailing W=100
  #5  Realized skewness    (C1):    3 windows {20,50,100} on log-returns
  #7  Generalized OFI      (E4):    10 levels x 3 windows {5,20,60}
  #8  Kyle's lambda rolling (B2):   2 windows {50,100}
  #10 Vol-burst ratio      (F2):    {volume,|amount|} x {20,50} = 4

Total dim = 20 + 3 + 30 + 2 + 4 = 59.

All operations are session-scoped (per (sym, date, sess) parquet); no cross-session shift.
W <= 100 to remain compatible with the platform's 100-tick inference window.

Constraint compliance:
  - sym never used (per-session series)
  - date never used
  - causal rolling (no look-ahead)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-8


# ---- #4 Window Quantile Rank (A2) -------------------------------------

QRANK_COLS: List[str] = [
    "spread1", "spread5", "spread10", "cumspread",
    "amount_delta", "volume_delta",
    "imbalance", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_acc", "la_acc", "mb_acc", "ma_acc",
    "midprice",
]
QRANK_W = 100  # within-100 rank


def compute_qrank_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Quantile rank: percentile of x_t within trailing W=100 ticks (causal).

    For each col, rank ∈ [0, 1] is unitless. Exclusively depends on local order
    statistics → strictly sym-invariant given i.i.d. shape.
    """
    n_out = valid_hi - valid_lo + 1
    F = len(QRANK_COLS)
    out = np.zeros((n_out, F), dtype=np.float32)

    # vectorized: for each row r in [valid_lo, valid_hi]:
    #   rank = mean(x[r-W+1..r] <= x[r])
    # implementation via stride trick.
    for i, col in enumerate(QRANK_COLS):
        x = df[col].to_numpy(dtype=np.float32, copy=False)
        T = x.shape[0]
        if T < QRANK_W:
            continue
        # sliding_window_view: shape (T-W+1, W)
        sw = np.lib.stride_tricks.sliding_window_view(x, QRANK_W)
        # rank only on rows [valid_lo, valid_hi]; sw row index r corresponds
        # to window ending at r+W-1, so we need sw indices r' s.t. r' = t-W+1
        # for global row t in [valid_lo, valid_hi]:
        sw_lo = valid_lo - (QRANK_W - 1)
        sw_hi = valid_hi - (QRANK_W - 1)
        sw_slice = sw[sw_lo:sw_hi + 1]  # (n_out, W)
        last = sw_slice[:, -1:]
        rank = (sw_slice <= last).mean(axis=-1).astype(np.float32)
        out[:, i] = rank
    return out


def qrank_feature_names() -> List[str]:
    return [f"qrank_W{QRANK_W}_{c}" for c in QRANK_COLS]


# ---- #5 Realized Skewness (C1) ----------------------------------------

SKEW_WINDOWS = (20, 50, 100)


def compute_skew_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Rolling skewness of log-returns: E[(r-mu)^3]/sigma^3.

    Unitless (3rd moment / sigma^3). Sym-agnostic by construction.
    """
    n_out = valid_hi - valid_lo + 1
    F = len(SKEW_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)

    s_r = pd.Series(r)
    for i, W in enumerate(SKEW_WINDOWS):
        # pandas rolling.skew uses unbiased estimator; we use population formula
        # via mean of (x-mu)^3 / std^3 for numerical robustness.
        mu = s_r.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        sd = s_r.rolling(W, min_periods=W).std(ddof=0).to_numpy(dtype=np.float64) + EPS
        # m3: mean((r - mu)^3) within rolling W
        # Use pandas rolling apply only if needed; here use cumulative trick:
        # m3 = mean(r^3) - 3*mu*mean(r^2) + 3*mu^2*mean(r) - mu^3
        #    = mean(r^3) - 3*mu*var - mu^3   (via mean(r) = mu, mean(r^2)=var+mu^2)
        s_r2 = pd.Series(r * r)
        s_r3 = pd.Series(r * r * r)
        m_r2 = s_r2.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        m_r3 = s_r3.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        var = m_r2 - mu * mu
        m3 = m_r3 - 3.0 * mu * m_r2 + 2.0 * (mu ** 3)
        skew = m3 / (sd ** 3 + EPS)
        skew = np.where(np.isfinite(skew), skew, 0.0).astype(np.float32)
        # clip to [-10, 10] to prevent rare near-zero variance blowups
        skew = np.clip(skew, -10.0, 10.0)
        out[:, i] = skew[rows]
    return out


def skew_feature_names() -> List[str]:
    return [f"rskew_W{W}" for W in SKEW_WINDOWS]


# ---- #7 GOFI Generalized OFI (E4) -------------------------------------

GOFI_LEVELS = tuple(range(1, 11))   # 10 levels
GOFI_WINDOWS = (5, 20, 60)


def _gofi_per_tick(b: np.ndarray, bs: np.ndarray, a: np.ndarray, asz: np.ndarray) -> np.ndarray:
    """Per-tick GOFI contribution (Cao-Hansch-Wang 2008 generalized OFI).

    Standard MLOFI:
      e = I(b_t >= b_{t-1})*bs_t  - I(b_t <= b_{t-1})*bs_{t-1}
        - I(a_t <= a_{t-1})*as_t  + I(a_t >= a_{t-1})*as_{t-1}

    GOFI adds 'price unchanged, size changed' channel:
      + I(b_t == b_{t-1}) * (bs_t - bs_{t-1})
      - I(a_t == a_{t-1}) * (as_t - as_{t-1})
    """
    T = b.shape[0]
    e = np.zeros(T, dtype=np.float64)
    if T < 2:
        return e

    b_lag = b[:-1]
    a_lag = a[:-1]
    bs_lag = bs[:-1]
    as_lag = asz[:-1]
    bnow = b[1:]
    anow = a[1:]
    bsnow = bs[1:]
    asnow = asz[1:]

    # Bid contribution
    bid_up = (bnow > b_lag).astype(np.float64) * bsnow
    bid_dn = (bnow < b_lag).astype(np.float64) * bs_lag
    bid_eq = (bnow == b_lag).astype(np.float64) * (bsnow - bs_lag)
    bid_cont = bid_up - bid_dn + bid_eq

    # Ask contribution (note signs)
    ask_up = (anow > a_lag).astype(np.float64) * as_lag  # ask up == buyers consume → +
    ask_dn = (anow < a_lag).astype(np.float64) * asnow   # ask down == sellers add → -
    ask_eq = (anow == a_lag).astype(np.float64) * (asnow - as_lag)
    ask_cont = ask_up - ask_dn - ask_eq

    e_inc = bid_cont + ask_cont
    e[1:] = e_inc
    return e


def compute_gofi_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """For 10 levels and 3 windows: rolling sum of GOFI per-tick contribution.

    Returns (n_out, 30) array, ordered as [W=5: lvl1..lvl10, W=20: ..., W=60: ...].
    """
    n_out = valid_hi - valid_lo + 1
    F = len(GOFI_LEVELS) * len(GOFI_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    col_idx = 0
    # We compute per-level e first, then roll over each window.
    e_per_level = []
    for k in GOFI_LEVELS:
        b = df[f"bid{k}"].to_numpy(dtype=np.float64, copy=False)
        bs = df[f"bsize{k}"].to_numpy(dtype=np.float64, copy=False)
        a = df[f"ask{k}"].to_numpy(dtype=np.float64, copy=False)
        asz = df[f"asize{k}"].to_numpy(dtype=np.float64, copy=False)
        e = _gofi_per_tick(b, bs, a, asz)
        e_per_level.append(e)

    for W in GOFI_WINDOWS:
        for k_idx in range(len(GOFI_LEVELS)):
            e = e_per_level[k_idx]
            roll = pd.Series(e).rolling(W, min_periods=W).sum().to_numpy(dtype=np.float32)
            roll = np.where(np.isfinite(roll), roll, 0.0).astype(np.float32)
            out[:, col_idx] = roll[rows]
            col_idx += 1
    return out


def gofi_feature_names() -> List[str]:
    names: List[str] = []
    for W in GOFI_WINDOWS:
        for k in GOFI_LEVELS:
            names.append(f"gofi_W{W}_lvl{k}")
    return names


# ---- #8 Kyle's lambda rolling (B2) ------------------------------------

KYLE_LAMBDA_WINDOWS = (50, 100)


def compute_kyle_lambda_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Rolling Kyle lambda: OLS slope of delta_mid on signed sqrt-dollar-vol.

    delta_mid_t = midprice_t - midprice_{t-1}
    signed_dvol_t = sign(delta_mid_t) * sqrt(|amount_delta_t|)
    lambda_W = cov(delta_mid, signed_dvol) / var(signed_dvol) within rolling W.

    OLS slope is unit-less in 'price impact per sqrt(dollar)' but cross-sym
    same regime via Kyle 1985 invariance.
    """
    n_out = valid_hi - valid_lo + 1
    F = len(KYLE_LAMBDA_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    delta_mid = np.zeros_like(mid)
    delta_mid[1:] = mid[1:] - mid[:-1]
    delta_mid = np.where(np.isfinite(delta_mid), delta_mid, 0.0)

    amt = df["amount_delta"].to_numpy(dtype=np.float64, copy=False)
    sign_dm = np.sign(delta_mid)
    signed_dvol = sign_dm * np.sqrt(np.abs(amt) + EPS)

    sx = pd.Series(signed_dvol)
    sy = pd.Series(delta_mid)
    sx2 = pd.Series(signed_dvol * signed_dvol)
    sxy = pd.Series(signed_dvol * delta_mid)

    for i, W in enumerate(KYLE_LAMBDA_WINDOWS):
        mx = sx.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        my = sy.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        mxy = sxy.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        mx2 = sx2.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        cov = mxy - mx * my
        var = mx2 - mx * mx
        lam = cov / (var + EPS)
        lam = np.where(np.isfinite(lam), lam, 0.0).astype(np.float32)
        # robust clip; lambda can spike when var is tiny.
        lam = np.clip(lam, -1.0, 1.0)
        out[:, i] = lam[rows]
    return out


def kyle_lambda_feature_names() -> List[str]:
    return [f"kyle_lam_W{W}" for W in KYLE_LAMBDA_WINDOWS]


# ---- #10 Vol-Burst Ratio (F2) -----------------------------------------

VOL_BURST_WINDOWS = (20, 50)


def compute_vol_burst_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """For W in {20, 50}, compute:
      vol_burst_W = volume_delta_t / mean(volume_delta, W)
      amt_burst_W = |amount_delta_t| / mean(|amount_delta|, W)

    Both are unitless ratios (mean ~ 1 cross-sym by definition).
    """
    n_out = valid_hi - valid_lo + 1
    F = 2 * len(VOL_BURST_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    vol = df["volume_delta"].to_numpy(dtype=np.float64, copy=False)
    amt = np.abs(df["amount_delta"].to_numpy(dtype=np.float64, copy=False))
    s_vol = pd.Series(vol)
    s_amt = pd.Series(amt)

    col_idx = 0
    for W in VOL_BURST_WINDOWS:
        mean_v = s_vol.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64) + EPS
        mean_a = s_amt.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64) + EPS
        rv = vol / mean_v
        ra = amt / mean_a
        rv = np.where(np.isfinite(rv), rv, 0.0).astype(np.float32)
        ra = np.where(np.isfinite(ra), ra, 0.0).astype(np.float32)
        # Clip extremely large ratios that occur on near-zero rolling means.
        rv = np.clip(rv, -100.0, 100.0)
        ra = np.clip(ra, 0.0, 100.0)
        out[:, col_idx] = rv[rows]
        col_idx += 1
        out[:, col_idx] = ra[rows]
        col_idx += 1
    return out


def vol_burst_feature_names() -> List[str]:
    names: List[str] = []
    for W in VOL_BURST_WINDOWS:
        names.append(f"vol_burst_W{W}")
        names.append(f"amt_burst_W{W}")
    return names


# ---- All-in-one --------------------------------------------------------

def compute_all_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    a = compute_qrank_session(df, valid_lo, valid_hi)
    b = compute_skew_session(df, valid_lo, valid_hi)
    c = compute_gofi_session(df, valid_lo, valid_hi)
    d = compute_kyle_lambda_session(df, valid_lo, valid_hi)
    e = compute_vol_burst_session(df, valid_lo, valid_hi)
    return np.concatenate([a, b, c, d, e], axis=1)


def all_feature_names() -> List[str]:
    return (
        qrank_feature_names()
        + skew_feature_names()
        + gofi_feature_names()
        + kyle_lambda_feature_names()
        + vol_burst_feature_names()
    )


def feature_block_lengths() -> Tuple[int, int, int, int, int]:
    return (
        len(qrank_feature_names()),
        len(skew_feature_names()),
        len(gofi_feature_names()),
        len(kyle_lambda_feature_names()),
        len(vol_burst_feature_names()),
    )


if __name__ == "__main__":
    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    p = os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet")
    df = pd.read_parquet(p)
    print("session shape:", df.shape)
    valid_lo, valid_hi = 99, len(df) - 1 - 60
    feats = compute_all_session(df, valid_lo, valid_hi)
    names = all_feature_names()
    blocks = feature_block_lengths()
    print(f"feats shape: {feats.shape}, names count={len(names)}")
    print(f"blocks: qrank={blocks[0]}, skew={blocks[1]}, gofi={blocks[2]}, kyle_lam={blocks[3]}, vol_burst={blocks[4]}")
    print("any nan?", np.isnan(feats).any())
    print("any inf?", np.isinf(feats).any())
    print("first row:", feats[0])
    print("min/max/mean overall:", feats.min(), feats.max(), feats.mean())
    print(f"qrank[0,:5]={feats[0, :5]}")
    print(f"skew[0]={feats[0, blocks[0]:blocks[0]+blocks[1]]}")
    print(f"gofi[0,:5]={feats[0, blocks[0]+blocks[1]:blocks[0]+blocks[1]+5]}")
    print(f"kyle_lam[0]={feats[0, blocks[0]+blocks[1]+blocks[2]:blocks[0]+blocks[1]+blocks[2]+blocks[3]]}")
    print(f"vol_burst[0]={feats[0, blocks[0]+blocks[1]+blocks[2]+blocks[3]:]}")
