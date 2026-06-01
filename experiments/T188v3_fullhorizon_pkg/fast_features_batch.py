"""iter_012 batch-vectorized feature computation.

Operates on a 3D batch tensor (N, 100, K) where N = number of windows,
100 = window length (ticks), K = number of raw feature columns. Outputs
a (N, 216) feature block: T3-no-time(69) + Stage1(54) + Stage2(59) + Stage3(14) + Stage5(20).

The single-window reference is `fast_features.py`. This module produces
near-identical numbers (max diff < 1e-3 on float32 reference values),
but ~10-30x faster because all ops are vectorized across batch dimension.

Constraint compliance (CRITICAL_CONSTRAINTS.md):
  - sym never used
  - date never used
  - Each window in the batch is independent (sym/date-agnostic)
  - causal: only uses [t-W+1 .. t] in each window
  - stateless: no cross-call cache
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import lfilter

EPS = 1e-8

# === T3 constants ===
T3_LOB_LEVELS = tuple(range(1, 11))
T3_MLOFI_WINDOWS = (5, 20, 60)
T3_RV_WINDOWS = (5, 10, 20, 50)
T3_EWMA_ALPHAS = (0.05, 0.1, 0.3, 0.5)
T3_INTST_COLS = ("lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst")

# === Stage 1 constants ===
DUAL_Z_COLS: List[str] = [
    "spread1", "spread5", "spread10", "cumspread",
    "bid1", "ask1", "bid_mean", "ask_mean",
    "midprice1", "midprice2", "midprice5", "midprice10",
    "bsize1", "bsize_mean", "totalbsize",
    "asize1", "asize_mean", "totalasize",
    "volume_delta", "amount_delta",
    "imbalance",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    "bid_diff1", "ask_diff1", "bid_diff5", "ask_diff5",
]
DUAL_Z_W_SHORT = 20
DUAL_Z_W_LONG = 100
SIGNED_RV_WINDOWS = (20, 50, 100)
KYLE_WINDOWS = (50, 100)
EWMA_OFI_LEVELS = (1, 5, 10)
EWMA_OFI_ALPHAS = (0.05, 0.1, 0.3, 0.5)

# === Stage 2 constants ===
QRANK_COLS: List[str] = [
    "spread1", "spread5", "spread10", "cumspread",
    "amount_delta", "volume_delta",
    "imbalance", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_acc", "la_acc", "mb_acc", "ma_acc",
    "midprice",
]
QRANK_W = 100
SKEW_WINDOWS = (20, 50, 100)
GOFI_LEVELS = tuple(range(1, 11))
GOFI_WINDOWS = (5, 20, 60)
KYLE_LAMBDA_WINDOWS = (50, 100)
VOL_BURST_WINDOWS = (20, 50)

# === Stage 3 constants ===
EWMA_RES_ALPHA = 0.05
RV_RATIO_PAIRS = ((5, 50), (20, 100), (50, 100))
JSHARE_WINDOWS = (20, 30, 50, 100)
CANCEL_WINDOWS = (20, 50, 100)
ROLL_WINDOWS = (30, 50, 100)


# =============================================================================
# Batch helpers (all operate on (N, T) arrays)
# =============================================================================

def _ewma_last_batch(x: np.ndarray, alphas: Tuple[float, ...]) -> np.ndarray:
    """Run EWMA(adjust=False) along axis=-1 for each alpha; return last value.

    x: (N, T) float64.
    Returns: (N, len(alphas)) float64.

    EWMA recursion: y_0 = x_0; y_t = a*x_t + (1-a)*y_{t-1}.
    Equivalent IIR: lfilter(b=[a], a=[1, -(1-a)], zi=(1-a)*x[..., 0]).
    scipy.signal.lfilter supports broadcasting via axis.
    """
    N = x.shape[0]
    out = np.zeros((N, len(alphas)), dtype=np.float64)
    for i, a in enumerate(alphas):
        b = np.array([a], dtype=np.float64)
        a_filt = np.array([1.0, -(1.0 - a)], dtype=np.float64)
        zi = ((1.0 - a) * x[:, :1])  # (N, 1)
        y, _ = lfilter(b, a_filt, x, axis=-1, zi=zi)
        out[:, i] = y[:, -1]
    return out


def _rolling_sum_last_W(x: np.ndarray, W: int) -> np.ndarray:
    """Sum of last W elements along axis=-1. x: (N, T) → (N,)."""
    return x[:, -W:].sum(axis=-1)


def _rolling_sum_W_full(x: np.ndarray, W: int) -> np.ndarray:
    """Causal rolling sum of W along axis=-1, with NaN→0 fill before W-1.

    Returns (N, T): out[n, t] = sum(x[n, t-W+1..t]) for t >= W-1, else 0.
    Mirrors pandas rolling(W, min_periods=W).sum() with NaN→0 fill.
    """
    N, T = x.shape
    if W > T:
        return np.zeros_like(x)
    cs = np.concatenate([np.zeros((N, 1), dtype=x.dtype), np.cumsum(x, axis=-1)], axis=-1)
    # rolled[t] = cs[t+1] - cs[t+1-W] for t >= W-1
    out = np.zeros((N, T), dtype=x.dtype)
    out[:, W - 1:] = cs[:, W:] - cs[:, :T - W + 1]
    return out


def _rolling_mean_std_lastW(x: np.ndarray, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """Population mean & std (ddof=0) over the LAST W elements along axis=-1.

    x: (N, T) → (mean, std), each (N,).
    """
    seg = x[:, -W:]
    m = seg.mean(axis=-1)
    sd = seg.std(axis=-1, ddof=0)
    return m, sd


# =============================================================================
# T3 (batch)
# =============================================================================

def compute_t3_no_time_batch(X3d: np.ndarray, col_idx: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute T3 schemeC features at last tick for all N windows.

    X3d: (N, T=100, K) float64 array of raw features.
    col_idx: dict mapping raw column name -> int column index in X3d.

    Returns:
      out: (N, 69) float64 — mlofi(30) + wmp(11) + rv(4) + ewma_intst(24)
      derived: dict of (N, T) arrays needed by stage1
        - 'mlofi_W20_lvl1', 'mlofi_W20_lvl5', 'mlofi_W20_lvl10': (N, T) rolling-sum (W=20) with NaN→0
        - 'wmp_lvl1': (N, T) full per-tick wmp (used by RV computation)
        - 'midprice': (N, T) = X3d[..., midprice1_col]  (alias for stage 1/2/3 callers)
    """
    N, T, _ = X3d.shape
    out = np.zeros((N, 69), dtype=np.float64)
    col = 0
    derived: Dict[str, np.ndarray] = {}

    # === MLOFI per-tick e for each level ===
    e_per_lvl: Dict[int, np.ndarray] = {}
    for k in T3_LOB_LEVELS:
        b = X3d[:, :, col_idx[f"bid{k}"]]
        a = X3d[:, :, col_idx[f"ask{k}"]]
        bs = X3d[:, :, col_idx[f"bsize{k}"]]
        asz = X3d[:, :, col_idx[f"asize{k}"]]
        # b_prev[t] = b[t-1], with b_prev[0] = NaN
        b_prev = np.empty_like(b); b_prev[:, 0] = np.nan; b_prev[:, 1:] = b[:, :-1]
        a_prev = np.empty_like(a); a_prev[:, 0] = np.nan; a_prev[:, 1:] = a[:, :-1]
        bs_prev = np.empty_like(bs); bs_prev[:, 0] = np.nan; bs_prev[:, 1:] = bs[:, :-1]
        as_prev = np.empty_like(asz); as_prev[:, 0] = np.nan; as_prev[:, 1:] = asz[:, :-1]
        ind_b_up = (b >= b_prev).astype(np.float64)
        ind_b_dn = (b <= b_prev).astype(np.float64)
        ind_a_dn = (a <= a_prev).astype(np.float64)
        ind_a_up = (a >= a_prev).astype(np.float64)
        e = ind_b_up * bs - ind_b_dn * bs_prev - ind_a_dn * asz + ind_a_up * as_prev
        e_per_lvl[k] = e

    # mlofi at last tick for W=5,20,60 × 10 levels = 30
    for W in T3_MLOFI_WINDOWS:
        for k in T3_LOB_LEVELS:
            e = e_per_lvl[k]
            window = e[:, -W:]
            # NaN→0 before sum (W<=60, T=100, so window doesn't touch index 0 normally)
            window = np.where(np.isnan(window), 0.0, window)
            out[:, col] = window.sum(axis=-1)
            col += 1

    # Build full mlofi_W20_lvl{k} for k in 1,5,10 (needed by stage 1 EWMA-OFI)
    # pandas behavior: rolling(W=20, min_periods=20) on series with e[0]=NaN:
    #   - rolled[i<19] = NaN (insufficient data)
    #   - rolled[19] = sum(e[0..19]) but contains NaN → NaN
    #   - rolled[i>=20] = sum(e[i-19..i]) clean
    # Then iter_009 EWMA path does NaN→0 fill before EWMA, so:
    #   - rolled_clean[0..19] = 0; rolled_clean[20..99] = sum
    for k in (1, 5, 10):
        e = e_per_lvl[k]
        e_clean = np.where(np.isnan(e), 0.0, e)
        rolled_full = _rolling_sum_W_full(e_clean, 20)  # (N, T)
        # zero out idx < 20 to match the "NaN at i=19 due to NaN at e[0]" -> 0 fill
        rolled_full[:, :20] = 0.0
        derived[f"mlofi_W20_lvl{k}"] = rolled_full

    # === WMP per level + balance_12 (11) ===
    wmp_per_level: List[np.ndarray] = []
    for k in T3_LOB_LEVELS:
        b = X3d[:, :, col_idx[f"bid{k}"]]
        a = X3d[:, :, col_idx[f"ask{k}"]]
        bs = X3d[:, :, col_idx[f"bsize{k}"]]
        asz = X3d[:, :, col_idx[f"asize{k}"]]
        denom = bs + asz
        wmp_normal = (a * bs + b * asz) / np.where(denom == 0, 1.0, denom)
        wmp_fallback = (a + b) / 2.0
        wmp = np.where(denom == 0, wmp_fallback, wmp_normal)
        wmp_per_level.append(wmp)
        out[:, col] = wmp[:, -1]
        col += 1
    out[:, col] = wmp_per_level[0][:, -1] - wmp_per_level[1][:, -1]
    col += 1
    derived["wmp_lvl1"] = wmp_per_level[0]

    # === RV (4) over wmp_lvl1+1 ===
    safe = wmp_per_level[0] + 1.0
    safe = np.where(safe > 0, safe, np.nan)
    log_ret = np.zeros_like(safe)
    log_ret[:, 1:] = np.log(safe[:, 1:] / safe[:, :-1])
    log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)
    sq = log_ret * log_ret
    for W in T3_RV_WINDOWS:
        s = sq[:, -W:].sum(axis=-1)
        s = np.maximum(s, 0.0)
        out[:, col] = np.sqrt(s)
        col += 1

    # === EWMA intensities (6 cols × 4 alphas = 24) ===
    # iter_009 uses pandas .ewm(alpha, adjust=False).mean(). We replicate via lfilter.
    # ordering: for alpha in T3_EWMA_ALPHAS: for c in T3_INTST_COLS: -> 4*6 = 24
    for alpha in T3_EWMA_ALPHAS:
        for c in T3_INTST_COLS:
            x = X3d[:, :, col_idx[c]]
            x_safe = np.where(np.isfinite(x), x, 0.0)
            v = _ewma_last_batch(x_safe, (alpha,))[:, 0]
            out[:, col] = v
            col += 1

    # midprice alias for stage1/2/3 (= midprice1)
    derived["midprice"] = X3d[:, :, col_idx["midprice1"]].copy()
    return out, derived


def t3_no_time_feature_names() -> List[str]:
    cols: List[str] = []
    for W in T3_MLOFI_WINDOWS:
        for k in T3_LOB_LEVELS:
            cols.append(f"mlofi_W{W}_lvl{k}")
    for k in T3_LOB_LEVELS:
        cols.append(f"wmp_lvl{k}")
    cols.append("wmp_balance_12")
    for W in T3_RV_WINDOWS:
        cols.append(f"rv_w{W}")
    for alpha in T3_EWMA_ALPHAS:
        for c in T3_INTST_COLS:
            cols.append(f"ewma_a{alpha}_{c}")
    return cols


# =============================================================================
# Stage 1 (batch)
# =============================================================================

def compute_stage1_batch(X3d: np.ndarray, col_idx: Dict[str, int],
                          derived: Dict[str, np.ndarray]) -> np.ndarray:
    """Returns (N, 54)."""
    N, T, _ = X3d.shape
    out = np.zeros((N, 54), dtype=np.float64)
    col = 0

    # --- dual_z (37) ---
    # Stack the columns we need into (N, T, 37) then compute per-feature stats
    dz_idx = np.array([col_idx[c] for c in DUAL_Z_COLS], dtype=np.int64)
    X_dz = X3d[:, :, dz_idx]  # (N, T, 37)
    last = X_dz[:, -1, :]  # (N, 37)
    seg20 = X_dz[:, -DUAL_Z_W_SHORT:, :]
    seg100 = X_dz[:, -DUAL_Z_W_LONG:, :]
    m20 = seg20.mean(axis=1)  # (N, 37)
    s20 = seg20.std(axis=1, ddof=0)
    m100 = seg100.mean(axis=1)
    s100 = seg100.std(axis=1, ddof=0)
    z_s = (last - m20) / (s20 + EPS)
    z_l = (last - m100) / (s100 + EPS)
    dual = z_s - z_l
    dual = np.where(np.isfinite(dual), dual, 0.0)
    out[:, col:col + 37] = dual
    col += 37

    # --- signed_rv (3) ---
    mid = derived["midprice"]  # (N, T)
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[:, 1:] = np.log(mid_safe[:, 1:] / mid_safe[:, :-1])
    r = np.where(np.isfinite(r), r, 0.0)
    r2 = r * r
    r_pos_mask = (r > 0).astype(np.float64)
    r_neg_mask = (r < 0).astype(np.float64)
    r2_pos = r2 * r_pos_mask
    r2_neg = r2 * r_neg_mask
    for W in SIGNED_RV_WINDOWS:
        rp = r2_pos[:, -W:].sum(axis=-1)
        rn = r2_neg[:, -W:].sum(axis=-1)
        out[:, col] = (rp - rn) / (rp + rn + EPS)
        col += 1

    # --- kyle_inv (2) ---
    amt = X3d[:, :, col_idx["amount_delta"]]  # (N, T)
    amt_last = amt[:, -1]
    abs_amt_last = np.abs(amt_last)
    for W in KYLE_WINDOWS:
        sigma = r[:, -W:].std(axis=-1, ddof=0) + EPS
        activity = abs_amt_last * sigma + EPS
        a13 = np.cbrt(activity)
        v = amt_last / (a13 + EPS)
        v = np.where(np.isfinite(v), v, 0.0)
        out[:, col] = v
        col += 1

    # --- ewma_ofi (12) — 3 levels × 4 alphas ---
    for k in EWMA_OFI_LEVELS:
        x = derived[f"mlofi_W20_lvl{k}"]  # (N, T)
        x_safe = np.where(np.isfinite(x), x, 0.0)
        out[:, col:col + 4] = _ewma_last_batch(x_safe, EWMA_OFI_ALPHAS)
        col += 4

    return out


def stage1_feature_names() -> List[str]:
    names = [f"dualz_{c}" for c in DUAL_Z_COLS]
    names += [f"signed_rv_W{W}" for W in SIGNED_RV_WINDOWS]
    names += [f"kyle_inv_W{W}" for W in KYLE_WINDOWS]
    for k in EWMA_OFI_LEVELS:
        for a in EWMA_OFI_ALPHAS:
            names.append(f"ewma_ofi_a{a}_lvl{k}")
    return names


# =============================================================================
# Stage 2 (batch)
# =============================================================================

def compute_stage2_batch(X3d: np.ndarray, col_idx: Dict[str, int],
                          derived: Dict[str, np.ndarray]) -> np.ndarray:
    """Returns (N, 59)."""
    N, T, _ = X3d.shape
    out = np.zeros((N, 59), dtype=np.float64)
    col = 0

    # --- qrank (20) ---
    # rank = mean(x[-W:] <= x_last) across the last W ticks
    qr_idx = []
    for c in QRANK_COLS:
        if c == "midprice":
            qr_idx.append(-1)  # sentinel for derived
        else:
            qr_idx.append(col_idx[c])
    for i, c in enumerate(QRANK_COLS):
        if c == "midprice":
            x = derived["midprice"]
        else:
            x = X3d[:, :, qr_idx[i]]
        # match reference dtype: float32
        x32 = x.astype(np.float32, copy=False)
        last_val = x32[:, -1:]  # (N, 1)
        seg = x32[:, -QRANK_W:]
        out[:, col] = (seg <= last_val).mean(axis=-1)
        col += 1

    # --- skew (3) on log-returns of midprice ---
    mid = derived["midprice"]
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[:, 1:] = np.log(mid_safe[:, 1:] / mid_safe[:, :-1])
    r = np.where(np.isfinite(r), r, 0.0)
    for W in SKEW_WINDOWS:
        seg = r[:, -W:]  # (N, W)
        mu = seg.mean(axis=-1)
        sd = seg.std(axis=-1, ddof=0) + EPS
        seg2 = seg * seg
        seg3 = seg2 * seg
        m_r2 = seg2.mean(axis=-1)
        m_r3 = seg3.mean(axis=-1)
        m3 = m_r3 - 3.0 * mu * m_r2 + 2.0 * (mu ** 3)
        skew = m3 / (sd ** 3 + EPS)
        skew = np.where(np.isfinite(skew), skew, 0.0)
        skew = np.clip(skew, -10.0, 10.0)
        out[:, col] = skew
        col += 1

    # --- gofi (30) per (W, k) ---
    e_per_level: List[np.ndarray] = []
    for k in GOFI_LEVELS:
        b = X3d[:, :, col_idx[f"bid{k}"]]
        bs = X3d[:, :, col_idx[f"bsize{k}"]]
        a = X3d[:, :, col_idx[f"ask{k}"]]
        asz = X3d[:, :, col_idx[f"asize{k}"]]
        # gofi e[t]: (delta logic). e[0] = 0 by convention; e[t] for t>=1.
        b_lag = b[:, :-1]
        a_lag = a[:, :-1]
        bs_lag = bs[:, :-1]
        as_lag = asz[:, :-1]
        bnow = b[:, 1:]
        anow = a[:, 1:]
        bsnow = bs[:, 1:]
        asnow = asz[:, 1:]
        bid_up = (bnow > b_lag).astype(np.float64) * bsnow
        bid_dn = (bnow < b_lag).astype(np.float64) * bs_lag
        bid_eq = (bnow == b_lag).astype(np.float64) * (bsnow - bs_lag)
        bid_cont = bid_up - bid_dn + bid_eq
        ask_up = (anow > a_lag).astype(np.float64) * as_lag
        ask_dn = (anow < a_lag).astype(np.float64) * asnow
        ask_eq = (anow == a_lag).astype(np.float64) * (asnow - as_lag)
        ask_cont = ask_up - ask_dn - ask_eq
        e = np.zeros((N, T), dtype=np.float64)
        e[:, 1:] = bid_cont + ask_cont
        e_per_level.append(e)

    for W in GOFI_WINDOWS:
        for k_idx in range(len(GOFI_LEVELS)):
            seg = e_per_level[k_idx][:, -W:]
            v = seg.sum(axis=-1)
            v = np.where(np.isfinite(v), v, 0.0)
            out[:, col] = v
            col += 1

    # --- kyle_lambda (2) — rolling cov/var of (signed_dvol, delta_mid) over W ---
    delta_mid = np.zeros_like(mid)
    delta_mid[:, 1:] = mid[:, 1:] - mid[:, :-1]
    delta_mid = np.where(np.isfinite(delta_mid), delta_mid, 0.0)
    sign_dm = np.sign(delta_mid)
    amt = X3d[:, :, col_idx["amount_delta"]]
    signed_dvol = sign_dm * np.sqrt(np.abs(amt) + EPS)
    sxy = signed_dvol * delta_mid
    sx2 = signed_dvol * signed_dvol
    for W in KYLE_LAMBDA_WINDOWS:
        x_seg = signed_dvol[:, -W:]
        y_seg = delta_mid[:, -W:]
        mx = x_seg.mean(axis=-1)
        my = y_seg.mean(axis=-1)
        mxy = sxy[:, -W:].mean(axis=-1)
        mx2 = sx2[:, -W:].mean(axis=-1)
        cov = mxy - mx * my
        var = mx2 - mx * mx
        lam = cov / (var + EPS)
        lam = np.where(np.isfinite(lam), lam, 0.0)
        lam = np.clip(lam, -1.0, 1.0)
        out[:, col] = lam
        col += 1

    # --- vol_burst (4) ---
    vol = X3d[:, :, col_idx["volume_delta"]]
    amt_abs = np.abs(amt)
    for W in VOL_BURST_WINDOWS:
        mean_v = vol[:, -W:].mean(axis=-1) + EPS
        mean_a = amt_abs[:, -W:].mean(axis=-1) + EPS
        rv = vol[:, -1] / mean_v
        ra = amt_abs[:, -1] / mean_a
        rv = np.where(np.isfinite(rv), rv, 0.0)
        ra = np.where(np.isfinite(ra), ra, 0.0)
        out[:, col] = np.clip(rv, -100.0, 100.0)
        col += 1
        out[:, col] = np.clip(ra, 0.0, 100.0)
        col += 1

    return out


def stage2_feature_names() -> List[str]:
    names = [f"qrank_W{QRANK_W}_{c}" for c in QRANK_COLS]
    names += [f"rskew_W{W}" for W in SKEW_WINDOWS]
    for W in GOFI_WINDOWS:
        for k in GOFI_LEVELS:
            names.append(f"gofi_W{W}_lvl{k}")
    names += [f"kyle_lam_W{W}" for W in KYLE_LAMBDA_WINDOWS]
    for W in VOL_BURST_WINDOWS:
        names.append(f"vol_burst_W{W}")
        names.append(f"amt_burst_W{W}")
    return names


# =============================================================================
# Stage 3 (batch)
# =============================================================================

def compute_stage3_batch(X3d: np.ndarray, col_idx: Dict[str, int],
                          derived: Dict[str, np.ndarray]) -> np.ndarray:
    """Returns (N, 14)."""
    N, T, _ = X3d.shape
    out = np.zeros((N, 14), dtype=np.float64)
    col = 0

    mid = derived["midprice"]  # (N, T)

    # --- ewma_resid (1) ---
    ewma_mid = _ewma_last_batch(mid, (EWMA_RES_ALPHA,))[:, 0]
    resid = mid[:, -1] - ewma_mid
    resid = np.where(np.isfinite(resid), resid, 0.0)
    out[:, col] = np.clip(resid, -0.05, 0.05)
    col += 1

    # --- rv_ratio (3) ---
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[:, 1:] = np.log(mid_safe[:, 1:] / mid_safe[:, :-1])
    r = np.where(np.isfinite(r), r, 0.0)
    r2 = r * r
    rv_W: Dict[int, np.ndarray] = {}
    for W in (5, 20, 50, 100):
        rv_W[W] = r2[:, -W:].sum(axis=-1)  # (N,)
    for Wn, Wd in RV_RATIO_PAIRS:
        ratio = rv_W[Wn] / (rv_W[Wd] + EPS)
        ratio = np.where(np.isfinite(ratio), ratio, 0.0)
        out[:, col] = np.clip(ratio, 0.0, 100.0)
        col += 1

    # --- jshare (4) ---
    abs_r = np.abs(r)
    abs_r_lag = np.zeros_like(abs_r)
    abs_r_lag[:, 1:] = abs_r[:, :-1]
    prod = abs_r * abs_r_lag
    factor = np.pi / 2.0
    for W in JSHARE_WINDOWS:
        bv = factor * prod[:, -W:].sum(axis=-1)
        rv_w = r2[:, -W:].sum(axis=-1)
        j = (rv_w - bv) / (rv_w + EPS)
        j = np.where(np.isfinite(j), j, 0.0)
        out[:, col] = np.clip(j, 0.0, 1.0)
        col += 1

    # --- cancel_imb (3) ---
    lb = X3d[:, :, col_idx["lb_intst"]]
    mb = X3d[:, :, col_idx["mb_intst"]]
    cb = X3d[:, :, col_idx["cb_intst"]]
    la = X3d[:, :, col_idx["la_intst"]]
    ma = X3d[:, :, col_idx["ma_intst"]]
    ca = X3d[:, :, col_idx["ca_intst"]]
    bid_total = lb + mb + cb
    ask_total = la + ma + ca
    for W in CANCEL_WINDOWS:
        cb_sum = cb[:, -W:].sum(axis=-1)
        ca_sum = ca[:, -W:].sum(axis=-1)
        bt_sum = bid_total[:, -W:].sum(axis=-1)
        at_sum = ask_total[:, -W:].sum(axis=-1)
        cb_share = cb_sum / (bt_sum + EPS)
        ca_share = ca_sum / (at_sum + EPS)
        imb = cb_share - ca_share
        imb = np.where(np.isfinite(imb), imb, 0.0)
        out[:, col] = np.clip(imb, -1.0, 1.0)
        col += 1

    # --- roll_eff_spr (3) ---
    close = X3d[:, :, col_idx["close"]]
    dc = np.zeros_like(close)
    dc[:, 1:] = close[:, 1:] - close[:, :-1]
    dc = np.where(np.isfinite(dc), dc, 0.0)
    dc_lag = np.zeros_like(dc)
    dc_lag[:, 1:] = dc[:, :-1]
    qspr_last = X3d[:, -1, col_idx["spread1"]] + 1.0
    sxy_dc = dc * dc_lag
    for W in ROLL_WINDOWS:
        x_seg = dc[:, -W:]
        y_seg = dc_lag[:, -W:]
        mx = x_seg.mean(axis=-1)
        my = y_seg.mean(axis=-1)
        mxy = sxy_dc[:, -W:].mean(axis=-1)
        cov = mxy - mx * my
        eff = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
        ratio = eff / (qspr_last + EPS)
        ratio = np.where(np.isfinite(ratio), ratio, 0.0)
        out[:, col] = np.clip(ratio, 0.0, 10.0)
        col += 1

    return out


def stage3_feature_names() -> List[str]:
    names = [f"mid_ewma_resid_a{EWMA_RES_ALPHA}"]
    names += [f"rv_ratio_W{n}_W{d}" for (n, d) in RV_RATIO_PAIRS]
    names += [f"jshare_W{W}" for W in JSHARE_WINDOWS]
    names += [f"cancel_imb_W{W}" for W in CANCEL_WINDOWS]
    names += [f"roll_eff_spr_ratio_W{W}" for W in ROLL_WINDOWS]
    return names


# =============================================================================
# Stage 5 (batch) — 20 features in 6 families, all sym-agnostic, max W = 100
# =============================================================================

ADAPT_MOM_WINDOWS = (20, 50, 100)
OFI_TOX_PAIRS = ((1, 20), (1, 50), (5, 20), (5, 50))
SIGNED_BV_WINDOWS = (20, 50, 100)
SPREAD_REG_WINDOWS = (20, 50, 100)
TRADE_PERS_WINDOWS = (20, 50, 100)
LIQ_ASYM_WINDOWS = (5, 20, 50, 100)


def compute_stage5_batch(X3d: np.ndarray, col_idx: Dict[str, int]) -> np.ndarray:
    """Stage 5 batch feature extractor.

    X3d: (N, T=100, K) float64.
    Returns: (N, 20) float64.
    """
    N, T, _ = X3d.shape
    out = np.zeros((N, 20), dtype=np.float64)
    col = 0

    mid = X3d[:, :, col_idx["midprice1"]]  # (N, T)

    # --- A. Adaptive momentum (3) ---
    for W in ADAPT_MOM_WINDOWS:
        last = mid[:, -1]
        ref = mid[:, -W]
        sd = mid[:, -W:].std(axis=-1, ddof=0) + EPS
        v = (last - ref) / sd
        v = np.where(np.isfinite(v), v, 0.0)
        out[:, col] = np.clip(v, -10.0, 10.0)
        col += 1

    # precompute Δmid, log returns
    dmid = np.zeros_like(mid)
    dmid[:, 1:] = mid[:, 1:] - mid[:, :-1]
    dmid = np.where(np.isfinite(dmid), dmid, 0.0)

    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[:, 1:] = np.log(mid_safe[:, 1:] / mid_safe[:, :-1])
    r = np.where(np.isfinite(r), r, 0.0)

    # --- B. OFI toxicity (4) ---
    ofi_per_lvl: Dict[int, np.ndarray] = {}
    for k in (1, 5):
        b = X3d[:, :, col_idx[f"bid{k}"]]
        a = X3d[:, :, col_idx[f"ask{k}"]]
        bs = X3d[:, :, col_idx[f"bsize{k}"]]
        asz = X3d[:, :, col_idx[f"asize{k}"]]
        b_prev = np.empty_like(b); b_prev[:, 0] = b[:, 0]; b_prev[:, 1:] = b[:, :-1]
        a_prev = np.empty_like(a); a_prev[:, 0] = a[:, 0]; a_prev[:, 1:] = a[:, :-1]
        bs_prev = np.empty_like(bs); bs_prev[:, 0] = bs[:, 0]; bs_prev[:, 1:] = bs[:, :-1]
        as_prev = np.empty_like(asz); as_prev[:, 0] = asz[:, 0]; as_prev[:, 1:] = asz[:, :-1]
        bid_up = (b > b_prev).astype(np.float64) * bs
        bid_dn = (b < b_prev).astype(np.float64) * bs_prev
        bid_eq = (b == b_prev).astype(np.float64) * (bs - bs_prev)
        ask_up = (a > a_prev).astype(np.float64) * as_prev
        ask_dn = (a < a_prev).astype(np.float64) * asz
        ask_eq = (a == a_prev).astype(np.float64) * (asz - as_prev)
        e = (bid_up - bid_dn + bid_eq) - (ask_up - ask_dn - ask_eq)
        e = np.where(np.isfinite(e), e, 0.0)
        ofi_per_lvl[k] = e

    for (lvl, W) in OFI_TOX_PAIRS:
        x = ofi_per_lvl[lvl][:, -W:]
        y = dmid[:, -W:]
        mx = x.mean(axis=-1)
        my = y.mean(axis=-1)
        x_c = x - mx[:, None]
        y_c = y - my[:, None]
        sxy = (x_c * y_c).sum(axis=-1)
        sxx = (x_c * x_c).sum(axis=-1)
        syy = (y_c * y_c).sum(axis=-1)
        denom = np.sqrt(sxx * syy) + EPS
        v = sxy / denom
        v = np.where(np.isfinite(v), v, 0.0)
        out[:, col] = np.clip(v, -1.0, 1.0)
        col += 1

    # --- C. Multi-scale signed bipower (3) ---
    abs_r = np.abs(r)
    abs_r_lag = np.zeros_like(abs_r)
    abs_r_lag[:, 1:] = abs_r[:, :-1]
    sign_r = np.sign(r)
    bv_term = sign_r * abs_r * abs_r_lag
    r2 = r * r
    factor = np.pi / 2.0
    for W in SIGNED_BV_WINDOWS:
        sbv = factor * bv_term[:, -W:].sum(axis=-1)
        rv = r2[:, -W:].sum(axis=-1) + EPS
        v = sbv / rv
        v = np.where(np.isfinite(v), v, 0.0)
        out[:, col] = np.clip(v, -10.0, 10.0)
        col += 1

    # --- D. Stationary spread regime (3) ---
    spr = X3d[:, :, col_idx["spread1"]]
    for W in SPREAD_REG_WINDOWS:
        seg = spr[:, -W:]
        med = np.median(seg, axis=-1)
        q75 = np.quantile(seg, 0.75, axis=-1)
        q25 = np.quantile(seg, 0.25, axis=-1)
        iqr = q75 - q25
        denom = np.where(iqr > EPS, iqr, np.inf)
        v = (spr[:, -1] - med) / denom
        v = np.where(np.isfinite(v), v, 0.0)
        out[:, col] = np.clip(v, -10.0, 10.0)
        col += 1

    # --- E. Trade-direction persistence (3) ---
    sign_dm = np.sign(dmid)
    sign_lag = np.zeros_like(sign_dm)
    sign_lag[:, 1:] = sign_dm[:, :-1]
    for W in TRADE_PERS_WINDOWS:
        x = sign_dm[:, -W:]
        y = sign_lag[:, -W:]
        mx = x.mean(axis=-1)
        my = y.mean(axis=-1)
        x_c = x - mx[:, None]
        y_c = y - my[:, None]
        sxy = (x_c * y_c).sum(axis=-1)
        sxx = (x_c * x_c).sum(axis=-1)
        syy = (y_c * y_c).sum(axis=-1)
        denom = np.sqrt(sxx * syy) + EPS
        v = sxy / denom
        v = np.where(np.isfinite(v), v, 0.0)
        out[:, col] = np.clip(v, -1.0, 1.0)
        col += 1

    # --- F. Liquidity asymmetry (4) ---
    bid_top5 = np.zeros_like(mid)
    ask_top5 = np.zeros_like(mid)
    for k in range(1, 6):
        bid_top5 = bid_top5 + X3d[:, :, col_idx[f"bsize{k}"]]
        ask_top5 = ask_top5 + X3d[:, :, col_idx[f"asize{k}"]]
    for W in LIQ_ASYM_WINDOWS:
        sb = bid_top5[:, -W:].sum(axis=-1)
        sa = ask_top5[:, -W:].sum(axis=-1)
        v = np.log((1.0 + sb) / (1.0 + sa))
        v = np.where(np.isfinite(v), v, 0.0)
        out[:, col] = np.clip(v, -10.0, 10.0)
        col += 1

    assert col == 20, f"Stage5 column count mismatch: {col}"
    return out


def stage5_feature_names() -> List[str]:
    names: List[str] = []
    for W in ADAPT_MOM_WINDOWS:
        names.append(f"adapt_mom_W{W}")
    for (lvl, W) in OFI_TOX_PAIRS:
        names.append(f"ofi_tox_lvl{lvl}_W{W}")
    for W in SIGNED_BV_WINDOWS:
        names.append(f"signed_bv_W{W}")
    for W in SPREAD_REG_WINDOWS:
        names.append(f"spread_reg_W{W}")
    for W in TRADE_PERS_WINDOWS:
        names.append(f"trade_pers_W{W}")
    for W in LIQ_ASYM_WINDOWS:
        names.append(f"liq_asym_top5_W{W}")
    return names


# =============================================================================
# Master orchestrator (batch)
# =============================================================================

def all_feature_names() -> List[str]:
    return (t3_no_time_feature_names() + stage1_feature_names()
            + stage2_feature_names() + stage3_feature_names()
            + stage5_feature_names())


def compute_batch_features(X3d: np.ndarray, col_idx: Dict[str, int]) -> np.ndarray:
    """Compute the (T3+S1+S2+S3+S5) feature block for N windows.

    X3d: (N, 100, K) float64 — raw column tensor (must contain at least all
         column names referenced in DUAL_Z_COLS, QRANK_COLS, GOFI_LEVELS, etc.)
    col_idx: dict mapping raw column name -> int column in axis=2 of X3d.

    Returns: (N, 216) float64.
    """
    t3_v, derived = compute_t3_no_time_batch(X3d, col_idx)
    s1_v = compute_stage1_batch(X3d, col_idx, derived)
    s2_v = compute_stage2_batch(X3d, col_idx, derived)
    s3_v = compute_stage3_batch(X3d, col_idx, derived)
    s5_v = compute_stage5_batch(X3d, col_idx)
    return np.concatenate([t3_v, s1_v, s2_v, s3_v, s5_v], axis=1)
