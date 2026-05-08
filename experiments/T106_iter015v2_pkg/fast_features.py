"""iter_010 fast feature computation — last-tick only, no pandas rolling.

Equivalent to iter_009's compute.py + r34_features.py + r34_stage2_features.py
+ r34_stage3_features.py BUT operates on a single 100-tick window and only
computes the LAST-tick statistics (which is all that the predictor needs).

Speedup ~50x on a single window vs the reference session-rolling implementation,
because we avoid pandas Series construction and pandas rolling overhead.

Returns the 350-feature vector (T3 no-time 69 + Stage1 54 + Stage2 59 + Stage3 14
+ raw last 154 = 350); the caller drops 10 KS-fail extras to get 340 final features.

Constraint compliance (CRITICAL_CONSTRAINTS.md):
  - sym never used
  - date never used
  - causal: only uses [t-W+1 .. t] for each W <= 100
  - sym-agnostic: only window-local statistics
  - stateless: no cross-call cache (every call independent)
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
# T3 helpers
# =============================================================================

def _ewma_last_multi(x: np.ndarray, alphas: Tuple[float, ...]) -> np.ndarray:
    """For each alpha, return EWMA(adjust=False) final value over x.

    x: (T,) float64 array (NaNs already replaced)
    Returns (len(alphas),) float64.

    EWMA recursion: y_0 = x_0; y_t = a*x_t + (1-a)*y_{t-1}.
    Equivalent IIR filter: b=[a], a_filt=[1, -(1-a)], with init zi=(1-a)*x[0].
    Use scipy.signal.lfilter (C backend, fast).
    """
    out = np.zeros(len(alphas), dtype=np.float64)
    for i, a in enumerate(alphas):
        b = np.array([a], dtype=np.float64)
        a_filt = np.array([1.0, -(1.0 - a)], dtype=np.float64)
        zi = np.array([(1.0 - a) * x[0]], dtype=np.float64)
        y, _ = lfilter(b, a_filt, x, zi=zi)
        out[i] = y[-1]
    return out


def _wmp_lvl(b: np.ndarray, a: np.ndarray, bs: np.ndarray, asz: np.ndarray) -> np.ndarray:
    """Weighted mid-price for one level (vectorized over time)."""
    denom = bs + asz
    wmp_normal = (a * bs + b * asz) / np.where(denom == 0, 1.0, denom)
    wmp_fallback = (a + b) / 2.0
    return np.where(denom == 0, wmp_fallback, wmp_normal)


def compute_t3_no_time_last(df_arr: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute T3 schemeC features at last tick.

    Inputs:
      df_arr: dict mapping raw column name -> 1-D numpy array (length 100)
              must contain bid{k}, ask{k}, bsize{k}, asize{k} for k=1..10,
              plus the 6 *_intst cols.

    Returns:
      (out_vec, derived) where:
      - out_vec: (69,) — mlofi(30) + wmp(11) + rv(4) + ewma_intst(24)
      - derived: { 'mlofi_W20_lvl1', 'mlofi_W20_lvl5', 'mlofi_W20_lvl10', 'wmp_lvl1' }
                 (full 100-tick arrays, needed by stage 1 ewma_ofi & rv computation)

    Output ordering matches r34_stage2/3 expectations + iter_009's
    `_t3_no_time_cols` list (mlofi 30 → wmp 11 → rv 4 → ewma_intst 24).
    """
    out = np.zeros(69, dtype=np.float64)

    # === MLOFI per-tick e_k(t) for each level ===
    e_per_lvl: Dict[int, np.ndarray] = {}
    for k in T3_LOB_LEVELS:
        b = df_arr[f"bid{k}"].astype(np.float64, copy=False)
        a = df_arr[f"ask{k}"].astype(np.float64, copy=False)
        bs = df_arr[f"bsize{k}"].astype(np.float64, copy=False)
        asz = df_arr[f"asize{k}"].astype(np.float64, copy=False)
        b_prev = np.empty_like(b); b_prev[0] = np.nan; b_prev[1:] = b[:-1]
        a_prev = np.empty_like(a); a_prev[0] = np.nan; a_prev[1:] = a[:-1]
        bs_prev = np.empty_like(bs); bs_prev[0] = np.nan; bs_prev[1:] = bs[:-1]
        as_prev = np.empty_like(asz); as_prev[0] = np.nan; as_prev[1:] = asz[:-1]
        ind_b_up = (b >= b_prev).astype(np.float64)
        ind_b_dn = (b <= b_prev).astype(np.float64)
        ind_a_dn = (a <= a_prev).astype(np.float64)
        ind_a_up = (a >= a_prev).astype(np.float64)
        e = ind_b_up * bs - ind_b_dn * bs_prev - ind_a_dn * asz + ind_a_up * as_prev
        e_per_lvl[k] = e

    # mlofi rolling sum at last tick (W=5, 20, 60); pandas .rolling().sum() with
    # min_periods=W returns NaN if any value in window is NaN -> we replicate:
    # if any NaN in last W → result is NaN. (E[0] is NaN due to b_prev=NaN at i=0.)
    # For W=5,20: window doesn't touch index 0 (last 5 / 20 of length 100), so safe.
    # For W=60: also doesn't reach index 0.
    # In practice: only when T < 100 might W=60 reach index 0; we always have T=100
    # so all sums are well-defined. But be defensive: replace NaN -> 0 in e first.
    col_idx = 0
    derived: Dict[str, np.ndarray] = {}
    for W in T3_MLOFI_WINDOWS:
        for k in T3_LOB_LEVELS:
            e = e_per_lvl[k]
            window = e[-W:]
            if np.any(np.isnan(window)):
                # fall back to 0-filled NaN positions (shouldn't happen with W<=60 and T=100)
                window = np.where(np.isnan(window), 0.0, window)
            out[col_idx] = window.sum()
            col_idx += 1

    # We also need full-length mlofi_W20_lvl{1,5,10} for stage 1 ewma_ofi.
    # Reproduce r34's exact behavior: pandas rolling sum with NaN propagation,
    # then NaN→0 fill before EWMA in stage 1. So compute via numpy:
    for k in (1, 5, 10):
        e = e_per_lvl[k]
        # rolling sum with W=20: out[i] = sum(e[i-19..i])
        # Use cumsum trick.
        e_clean = np.where(np.isnan(e), 0.0, e)
        # However, pandas rolling with NaN propagates NaN within the window.
        # In r34: pandas.Series(x).ewm() is called with x = where(isfinite(x), x, 0.0),
        # so NaN at i=0..18 of mlofi_W20_lvl_k (where rolling W=20 hasn't filled) get 0.
        # Equivalently: pandas mlofi_W20[i] = sum(e[i-19..i]) for i>=19; NaN for i<19.
        # After NaN→0 fill: mlofi_W20[i<19] = 0; mlofi_W20[i>=19] = sum(e[i-19..i]).
        # Using e_clean (NaN→0) and cumsum:
        # Reference: pandas rolling(W=20, min_periods=20).sum() with e[0]=NaN.
        # Windows ending at rows 0..18 are short (NaN). Row 19's window [0..19]
        # includes the NaN, also NaN. Rows 20..99 are clean sums. After NaN→0
        # fill: rolled[0..19] = 0, rolled[20..99] = sum(e[i-19..i]).
        cs = np.concatenate(([0.0], np.cumsum(e_clean)))
        T = len(e)
        rolled = np.zeros(T, dtype=np.float64)
        if T >= 20:
            # rolled[i] = sum(e_clean[i-19..i]) for i in 20..T-1
            # cs has length T+1; cs[i+1] - cs[i+1-20] = sum(e_clean[i-19..i])
            # For i in [20, T-1]: cs slice [21..T], cs slice [1..T-20]
            rolled[20:] = cs[21:T + 1] - cs[1:T - 19]
        derived[f"mlofi_W20_lvl{k}"] = rolled

    # === WMP per level + balance_12 (11 cols) ===
    wmp_lvls = []
    for k in T3_LOB_LEVELS:
        b = df_arr[f"bid{k}"].astype(np.float64, copy=False)
        a = df_arr[f"ask{k}"].astype(np.float64, copy=False)
        bs = df_arr[f"bsize{k}"].astype(np.float64, copy=False)
        asz = df_arr[f"asize{k}"].astype(np.float64, copy=False)
        wmp = _wmp_lvl(b, a, bs, asz)
        wmp_lvls.append(wmp)
        out[col_idx] = wmp[-1]
        col_idx += 1
    out[col_idx] = wmp_lvls[0][-1] - wmp_lvls[1][-1]  # balance_12
    col_idx += 1
    derived["wmp_lvl1"] = wmp_lvls[0]

    # === RV (4 cols) — over wmp_lvl1 + 1 ===
    safe = wmp_lvls[0] + 1.0
    safe = np.where(safe > 0, safe, np.nan)
    log_ret = np.zeros_like(safe)
    log_ret[1:] = np.log(safe[1:] / safe[:-1])
    log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)
    sq = log_ret * log_ret
    for W in T3_RV_WINDOWS:
        rolled = sq[-W:].sum()
        rolled = max(rolled, 0.0)  # clip
        out[col_idx] = np.sqrt(rolled)
        col_idx += 1

    # === EWMA intensities (6 cols × 4 alphas = 24) ===
    # Pandas ewm with adjust=False default replaces NaN with x_prev (forward-fill).
    # In our data *_intst cols are non-negative non-NaN, so no NaN concern.
    # Order in r34: for alpha: for col in INTST_COLS.
    for alpha in T3_EWMA_ALPHAS:
        for c in T3_INTST_COLS:
            x = df_arr[c].astype(np.float64, copy=False)
            x_safe = np.where(np.isfinite(x), x, 0.0)
            v = _ewma_last_multi(x_safe, (alpha,))[0]
            out[col_idx] = v
            col_idx += 1

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
# Stage 1 (R34)
# =============================================================================

def compute_stage1_last(df_arr: Dict[str, np.ndarray], derived: Dict[str, np.ndarray]) -> np.ndarray:
    """Returns (54,) = dual_z(37) + signed_rv(3) + kyle_inv(2) + ewma_ofi(12)."""
    out = np.zeros(54, dtype=np.float64)
    col = 0

    # --- dual_z (37) ---
    X = np.column_stack([df_arr[c].astype(np.float64, copy=False) for c in DUAL_Z_COLS])
    last = X[-1]
    m20 = X[-DUAL_Z_W_SHORT:].mean(axis=0)
    s20 = X[-DUAL_Z_W_SHORT:].std(axis=0)
    m100 = X.mean(axis=0)
    s100 = X.std(axis=0)
    z_s = (last - m20) / (s20 + EPS)
    z_l = (last - m100) / (s100 + EPS)
    out[col:col+37] = z_s - z_l
    col += 37

    # --- signed_rv (3) ---
    mid = df_arr["midprice"].astype(np.float64, copy=False)
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)
    r2_pos = (r * r) * (r > 0)
    r2_neg = (r * r) * (r < 0)
    for W in SIGNED_RV_WINDOWS:
        rp = r2_pos[-W:].sum()
        rn = r2_neg[-W:].sum()
        out[col] = (rp - rn) / (rp + rn + EPS)
        col += 1

    # --- kyle_inv (2) ---
    amt = df_arr["amount_delta"].astype(np.float64, copy=False)
    abs_amt_last = abs(amt[-1])
    amt_last = amt[-1]
    for W in KYLE_WINDOWS:
        sigma = r[-W:].std() + EPS
        activity = abs_amt_last * sigma + EPS
        a13 = np.cbrt(activity)
        out[col] = amt_last / (a13 + EPS)
        col += 1

    # --- ewma_ofi (12) — 3 levels × 4 alphas ---
    for k in EWMA_OFI_LEVELS:
        x = derived[f"mlofi_W20_lvl{k}"].astype(np.float64, copy=False)
        x_safe = np.where(np.isfinite(x), x, 0.0)
        out[col:col+4] = _ewma_last_multi(x_safe, EWMA_OFI_ALPHAS)
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
# Stage 2 (R34)
# =============================================================================

def compute_stage2_last(df_arr: Dict[str, np.ndarray]) -> np.ndarray:
    """Returns (59,) = qrank(20) + skew(3) + gofi(30) + kyle_lam(2) + vol_burst(4)."""
    out = np.zeros(59, dtype=np.float64)
    col = 0

    # --- qrank (20) — rank of last value within window of size 100 ---
    for c in QRANK_COLS:
        x = df_arr[c].astype(np.float32, copy=False)  # match reference dtype
        last_val = x[-1]
        # rank = mean(x[-W:] <= last)
        out[col] = (x[-QRANK_W:] <= last_val).mean()
        col += 1

    # --- skew (3) on log-returns of midprice ---
    mid = df_arr["midprice"].astype(np.float64, copy=False)
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)
    for W in SKEW_WINDOWS:
        seg = r[-W:]
        mu = seg.mean()
        sd = seg.std() + EPS
        # population skew = mean((r-mu)^3) / sd^3
        # use formula: m3 = mean(r^3) - 3*mu*mean(r^2) + 2*mu^3
        seg2 = seg * seg
        seg3 = seg2 * seg
        m_r2 = seg2.mean()
        m_r3 = seg3.mean()
        m3 = m_r3 - 3.0 * mu * m_r2 + 2.0 * mu ** 3
        skew = m3 / (sd ** 3 + EPS)
        if not np.isfinite(skew):
            skew = 0.0
        out[col] = float(np.clip(skew, -10.0, 10.0))
        col += 1

    # --- gofi (30) per (W, k) ---
    e_per_level: List[np.ndarray] = []
    for k in GOFI_LEVELS:
        b = df_arr[f"bid{k}"].astype(np.float64, copy=False)
        bs = df_arr[f"bsize{k}"].astype(np.float64, copy=False)
        a = df_arr[f"ask{k}"].astype(np.float64, copy=False)
        asz = df_arr[f"asize{k}"].astype(np.float64, copy=False)
        T = len(b)
        e = np.zeros(T, dtype=np.float64)
        if T >= 2:
            b_lag = b[:-1]; a_lag = a[:-1]
            bs_lag = bs[:-1]; as_lag = asz[:-1]
            bnow = b[1:]; anow = a[1:]
            bsnow = bs[1:]; asnow = asz[1:]
            bid_up = (bnow > b_lag).astype(np.float64) * bsnow
            bid_dn = (bnow < b_lag).astype(np.float64) * bs_lag
            bid_eq = (bnow == b_lag).astype(np.float64) * (bsnow - bs_lag)
            bid_cont = bid_up - bid_dn + bid_eq
            ask_up = (anow > a_lag).astype(np.float64) * as_lag
            ask_dn = (anow < a_lag).astype(np.float64) * asnow
            ask_eq = (anow == a_lag).astype(np.float64) * (asnow - as_lag)
            ask_cont = ask_up - ask_dn - ask_eq
            e[1:] = bid_cont + ask_cont
        e_per_level.append(e)
    for W in GOFI_WINDOWS:
        for k_idx in range(len(GOFI_LEVELS)):
            seg = e_per_level[k_idx][-W:]
            v = seg.sum()
            if not np.isfinite(v):
                v = 0.0
            out[col] = v
            col += 1

    # --- kyle_lambda (2) — rolling OLS slope at last tick ---
    delta_mid = np.zeros_like(mid)
    delta_mid[1:] = mid[1:] - mid[:-1]
    delta_mid = np.where(np.isfinite(delta_mid), delta_mid, 0.0)
    sign_dm = np.sign(delta_mid)
    signed_dvol = sign_dm * np.sqrt(np.abs(df_arr["amount_delta"].astype(np.float64, copy=False)) + EPS)
    sxy = signed_dvol * delta_mid
    sx2 = signed_dvol * signed_dvol
    for W in KYLE_LAMBDA_WINDOWS:
        x_seg = signed_dvol[-W:]
        y_seg = delta_mid[-W:]
        mx = x_seg.mean()
        my = y_seg.mean()
        mxy = sxy[-W:].mean()
        mx2 = sx2[-W:].mean()
        cov = mxy - mx * my
        var = mx2 - mx * mx
        lam = cov / (var + EPS)
        if not np.isfinite(lam):
            lam = 0.0
        out[col] = float(np.clip(lam, -1.0, 1.0))
        col += 1

    # --- vol_burst (4) — for W in (20,50): last/mean(W), |amount| variant ---
    vol = df_arr["volume_delta"].astype(np.float64, copy=False)
    amt_abs = np.abs(df_arr["amount_delta"].astype(np.float64, copy=False))
    for W in VOL_BURST_WINDOWS:
        mean_v = vol[-W:].mean() + EPS
        mean_a = amt_abs[-W:].mean() + EPS
        rv = vol[-1] / mean_v
        ra = amt_abs[-1] / mean_a
        if not np.isfinite(rv): rv = 0.0
        if not np.isfinite(ra): ra = 0.0
        out[col] = float(np.clip(rv, -100.0, 100.0))
        col += 1
        out[col] = float(np.clip(ra, 0.0, 100.0))
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
# Stage 3 (R34)
# =============================================================================

def compute_stage3_last(df_arr: Dict[str, np.ndarray]) -> np.ndarray:
    """Returns (14,) = ewma_resid(1) + rv_ratio(3) + jshare(4) + cancel_imb(3) + roll_eff_spr(3)."""
    out = np.zeros(14, dtype=np.float64)
    col = 0

    mid = df_arr["midprice"].astype(np.float64, copy=False)

    # --- ewma_resid (1) ---
    ewma_mid = _ewma_last_multi(mid, (EWMA_RES_ALPHA,))[0]
    resid = mid[-1] - ewma_mid
    if not np.isfinite(resid):
        resid = 0.0
    out[col] = float(np.clip(resid, -0.05, 0.05))
    col += 1

    # --- rv_ratio (3) ---
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)
    r2 = r * r
    rv_W: Dict[int, float] = {}
    for W in (5, 20, 50, 100):
        rv_W[W] = r2[-W:].sum()
    for Wn, Wd in RV_RATIO_PAIRS:
        ratio = rv_W[Wn] / (rv_W[Wd] + EPS)
        if not np.isfinite(ratio): ratio = 0.0
        out[col] = float(np.clip(ratio, 0.0, 100.0))
        col += 1

    # --- jshare (4) ---
    abs_r = np.abs(r)
    abs_r_lag = np.zeros_like(abs_r)
    abs_r_lag[1:] = abs_r[:-1]
    prod = abs_r * abs_r_lag
    factor = np.pi / 2.0
    for W in JSHARE_WINDOWS:
        bv = factor * prod[-W:].sum()
        rv_w = r2[-W:].sum()
        j_share = (rv_w - bv) / (rv_w + EPS)
        if not np.isfinite(j_share): j_share = 0.0
        out[col] = float(np.clip(j_share, 0.0, 1.0))
        col += 1

    # --- cancel_imb (3) ---
    lb = df_arr["lb_intst"].astype(np.float64, copy=False)
    mb = df_arr["mb_intst"].astype(np.float64, copy=False)
    cb = df_arr["cb_intst"].astype(np.float64, copy=False)
    la = df_arr["la_intst"].astype(np.float64, copy=False)
    ma = df_arr["ma_intst"].astype(np.float64, copy=False)
    ca = df_arr["ca_intst"].astype(np.float64, copy=False)
    bid_total = lb + mb + cb
    ask_total = la + ma + ca
    for W in CANCEL_WINDOWS:
        cb_sum = cb[-W:].sum()
        ca_sum = ca[-W:].sum()
        bt_sum = bid_total[-W:].sum()
        at_sum = ask_total[-W:].sum()
        cb_share = cb_sum / (bt_sum + EPS)
        ca_share = ca_sum / (at_sum + EPS)
        imb = cb_share - ca_share
        if not np.isfinite(imb): imb = 0.0
        out[col] = float(np.clip(imb, -1.0, 1.0))
        col += 1

    # --- roll_eff_spr (3) ---
    close = df_arr["close"].astype(np.float64, copy=False)
    dc = np.zeros_like(close)
    dc[1:] = close[1:] - close[:-1]
    dc = np.where(np.isfinite(dc), dc, 0.0)
    dc_lag = np.zeros_like(dc)
    dc_lag[1:] = dc[:-1]
    qspr_last_arr = df_arr["spread1"].astype(np.float64, copy=False) + 1.0
    sxy_dc = dc * dc_lag
    for W in ROLL_WINDOWS:
        x_seg = dc[-W:]
        y_seg = dc_lag[-W:]
        mx = x_seg.mean(); my = y_seg.mean()
        mxy = sxy_dc[-W:].mean()
        cov = mxy - mx * my
        eff = 2.0 * np.sqrt(max(-cov, 0.0))
        ratio = eff / (qspr_last_arr[-1] + EPS)
        if not np.isfinite(ratio): ratio = 0.0
        out[col] = float(np.clip(ratio, 0.0, 10.0))
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
# Master orchestrator
# =============================================================================

def all_feature_names() -> List[str]:
    """T3 (69) + Stage1 (54) + Stage2 (59) + Stage3 (14) = 196 names."""
    return t3_no_time_feature_names() + stage1_feature_names() + stage2_feature_names() + stage3_feature_names()


def compute_window_features(df_arr: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute the 69 + 127 (= 54 + 59 + 14) feature block for one window.

    Returns (196,) numpy array: T3-no-time(69) + Stage1(54) + Stage2(59) + Stage3(14).
    """
    t3_v, derived = compute_t3_no_time_last(df_arr)
    s1_v = compute_stage1_last(df_arr, derived)
    s2_v = compute_stage2_last(df_arr)
    s3_v = compute_stage3_last(df_arr)
    return np.concatenate([t3_v, s1_v, s2_v, s3_v])
