"""R20 Top-15 micro-structure factors — Scheme H extension.

Implements the 15 highest-priority factors from `r20_factor_library.md` §10.
All operations are vectorized (NumPy / Pandas rolling). No Python for-loop over rows.

Hard constraints (CRITICAL_CONSTRAINTS.md §1):
  - per-session compute (each parquet is independent — no cross-session state)
  - sym-agnostic
  - stateless within a session: relies only on past 100 ticks before t

Output: a DataFrame of N rows × ~70 cols aligned to df.index.

Layout (in `r20_feature_columns()` order):
  1. Stoikov micro-price + micro_gap_to_mid                         2
  2. Roll's effective spread W in {20, 30, 50, 100}                 4
  3. Realized Skewness W in {10, 20, 50}                            3
  4. Realized signed RV components (RV_pos, RV_neg, RVS_signed) W in {20, 50}  6
  5. Bipower variation W in {10, 20, 50}                            3
  6. BNS jump z-stat W in {20, 50}                                  2
  7. Pairwise price differences (16 pairs)                          16
  8. Triplet price imbalances (5 triplets)                          5
  9. Multi-level imbalance slope k=1..10                            10
  10. EWMA-OFI total at α in {0.05, 0.1, 0.3, 0.5}                  4
  11. GOFI rolling sum W=20                                         1
  12. Tick-rule signed volume W in {10, 20, 30, 50}                 4
  13. Kyle's λ rolling W=50                                         1
  14. Amihud illiquidity W=50                                       1
  15. Book slope per side (bid, ask)                                2
  16. Depth concentration top-3 / top-5 per side (4)                4
  17. Depth skew bid/ask                                            1

Total = 2+4+3+6+3+2+16+5+10+4+1+4+1+1+2+4+1 = 69
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

EPS = 1e-9
LOB_LEVELS = tuple(range(1, 11))


# ============================================================================
# Helpers
# ============================================================================

def _safe_div(a, b):
    return a / (b + EPS)


def _rolling_cov(x: pd.Series, y: pd.Series, W: int) -> pd.Series:
    """Rolling cov(x, y) with min_periods=W (NaN before W-1)."""
    return x.rolling(W, min_periods=W).cov(y)


def _rolling_var(x: pd.Series, W: int) -> pd.Series:
    return x.rolling(W, min_periods=W).var()


def _signed_carry_fwd(sign: np.ndarray) -> np.ndarray:
    """Tick-rule sign carry-forward: 0s replaced by previous nonzero sign,
    leading 0s -> 0.

    Vectorized via pandas ffill on a Series with 0 → NaN trick.
    """
    s = pd.Series(sign).replace(0, np.nan).ffill().fillna(0.0)
    return s.to_numpy()


# ============================================================================
# 1. Stoikov micro-price + gap
# ============================================================================

def compute_stoikov(df: pd.DataFrame, rho: float = 0.5) -> pd.DataFrame:
    """Stoikov simplified micro-price.
        M = mid + (S/2) * (2I - 1) * ρ
    where S = a1 - b1, I = bs1 / (bs1 + as1), ρ = 0.5 (constant).
    Plus micro_gap_to_mid = M - mid.
    """
    b1 = df["bid1"].to_numpy()
    a1 = df["ask1"].to_numpy()
    bs1 = df["bsize1"].to_numpy()
    as1 = df["asize1"].to_numpy()
    mid = (a1 + b1) * 0.5
    S = a1 - b1
    denom = bs1 + as1
    I = np.where(denom > 0, bs1 / np.where(denom == 0, 1.0, denom), 0.5)
    micro = mid + 0.5 * S * (2.0 * I - 1.0) * rho
    gap = micro - mid
    return pd.DataFrame({
        "stoikov_micro": micro,
        "stoikov_micro_gap_mid": gap,
    }, index=df.index)


# ============================================================================
# 2. Roll's effective spread (rolling cov of Δp, Δp lagged)
# ============================================================================

def compute_roll_spread(df: pd.DataFrame, windows=(20, 30, 50)) -> pd.DataFrame:
    mid = (df["bid1"] + df["ask1"]) * 0.5
    dp = mid.diff()
    dp_lag = dp.shift(1)
    out = {}
    for W in windows:
        cov = _rolling_cov(dp, dp_lag, W)
        s = 2.0 * np.sqrt(np.clip(-cov.to_numpy(), 0.0, np.inf))
        out[f"roll_spread_W{W}"] = s
    return pd.DataFrame(out, index=df.index)


# ============================================================================
# 3 & 4. Realized Skewness + signed RV
# ============================================================================

def compute_realized_skew_and_signed_rv(df: pd.DataFrame) -> pd.DataFrame:
    """RSkew + signed RV components based on simple Δmid returns."""
    mid = (df["bid1"] + df["ask1"]) * 0.5
    r = mid.diff()
    r2 = r * r
    r3 = r ** 3

    out = {}
    # 3a Realized skew
    for W in (10, 20, 50):
        sum_r2 = r2.rolling(W, min_periods=W).sum()
        sum_r3 = r3.rolling(W, min_periods=W).sum()
        rv = np.sqrt(sum_r2.clip(lower=0.0))
        rv_pow_3 = rv ** 3
        rskew = np.where(rv_pow_3.to_numpy() > 0,
                         np.sqrt(W) * sum_r3.to_numpy() / np.where(rv_pow_3.to_numpy() > 0, rv_pow_3.to_numpy(), 1.0),
                         0.0)
        out[f"r_skew_W{W}"] = rskew

    # 3b/4 Signed RV
    pos_mask = (r > 0).astype(np.float64)
    neg_mask = (r < 0).astype(np.float64)
    r2_pos = (r2 * pos_mask)
    r2_neg = (r2 * neg_mask)
    for W in (20, 50):
        rv_p = r2_pos.rolling(W, min_periods=W).sum().clip(lower=0.0)
        rv_n = r2_neg.rolling(W, min_periods=W).sum().clip(lower=0.0)
        rv_total = (rv_p + rv_n).replace(0, np.nan)
        signed = ((rv_p - rv_n) / rv_total).fillna(0.0)
        out[f"rv_pos_W{W}"] = rv_p.to_numpy()
        out[f"rv_neg_W{W}"] = rv_n.to_numpy()
        out[f"rvs_signed_W{W}"] = signed.to_numpy()

    return pd.DataFrame(out, index=df.index)


# ============================================================================
# 5 & 6. Bipower variation + BNS jump z-stat
# ============================================================================

def compute_bipower_and_jump(df: pd.DataFrame) -> pd.DataFrame:
    mid = (df["bid1"] + df["ask1"]) * 0.5
    r = mid.diff()
    abs_r = r.abs()
    abs_r_lag = abs_r.shift(1)
    bipower_term = (abs_r * abs_r_lag) * (np.pi / 2.0)
    r2 = r * r
    r4 = r ** 4

    out = {}
    # BV
    for W in (10, 20, 50):
        out[f"bv_W{W}"] = bipower_term.rolling(W, min_periods=W).sum().to_numpy()

    # BNS jump z-stat: J = (RV - BV) / sqrt(((π²/4) + π - 5) * max(BV², RQ) / W)
    factor = (np.pi ** 2) / 4.0 + np.pi - 5.0
    for W in (20, 50):
        rv = r2.rolling(W, min_periods=W).sum()
        bv = bipower_term.rolling(W, min_periods=W).sum()
        rq = (W / 3.0) * r4.rolling(W, min_periods=W).sum()
        denom = np.sqrt(np.maximum(factor * np.maximum(bv ** 2, rq) / W, 0.0) + EPS)
        z = (rv - bv) / denom
        out[f"bns_jump_z_W{W}"] = z.to_numpy()
    return pd.DataFrame(out, index=df.index)


# ============================================================================
# 7. Pairwise price differences (16 pairs)
# ============================================================================

PAIR_NAMES = [
    ("b1", "a1"), ("b1", "mid"), ("a1", "mid"),
    ("b1", "wmp"), ("a1", "wmp"), ("wmp", "mid"),
    ("wmp", "vwap"), ("open", "close"), ("open", "mid"),
    ("close", "mid"), ("b1", "b5"), ("b1", "b10"),
    ("a1", "a5"), ("a1", "a10"), ("b5", "b10"), ("a5", "a10"),
]


def compute_pairwise_diffs(df: pd.DataFrame, wmp_lvl1: pd.Series) -> pd.DataFrame:
    """Pairwise (p_i - p_j). Avoids `p_i + p_j` denominator since prices can be
    negative (data is normalized). Raw diff is safe and informative — LightGBM
    handles scale.
    """
    b1 = df["bid1"].to_numpy()
    a1 = df["ask1"].to_numpy()
    b5 = df["bid5"].to_numpy()
    a5 = df["ask5"].to_numpy()
    b10 = df["bid10"].to_numpy()
    a10 = df["ask10"].to_numpy()
    mid = (a1 + b1) * 0.5
    wmp = wmp_lvl1.to_numpy()
    # vwap proxy: (open + close) / 2 (within tick OHLC)
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    vwap = (open_ + close) * 0.5

    p = {
        "b1": b1, "a1": a1, "mid": mid, "wmp": wmp, "vwap": vwap,
        "open": open_, "close": close, "b5": b5, "b10": b10, "a5": a5, "a10": a10,
    }
    out = {}
    for x, y in PAIR_NAMES:
        out[f"pair_{x}_{y}"] = p[x] - p[y]
    return pd.DataFrame(out, index=df.index)


# ============================================================================
# 8. Triplet price imbalances (5)
# ============================================================================

TRIPLETS = [
    ("b1", "mid", "a1"),
    ("b1", "b5", "b10"),
    ("a1", "a5", "a10"),
    ("b1", "wmp", "a1"),
    ("b5", "mid", "a5"),
]


def compute_triplet_imb(df: pd.DataFrame, wmp_lvl1: pd.Series) -> pd.DataFrame:
    """triplet_imb = (max - mid) / (mid - min + ε)
    where mid is the median of the three values.
    Vectorized via column-stack + np.sort.
    """
    b1 = df["bid1"].to_numpy()
    a1 = df["ask1"].to_numpy()
    b5 = df["bid5"].to_numpy()
    a5 = df["ask5"].to_numpy()
    b10 = df["bid10"].to_numpy()
    a10 = df["ask10"].to_numpy()
    mid = (a1 + b1) * 0.5
    wmp = wmp_lvl1.to_numpy()
    p = {
        "b1": b1, "a1": a1, "mid": mid, "wmp": wmp,
        "b5": b5, "b10": b10, "a5": a5, "a10": a10,
    }
    out = {}
    for tr in TRIPLETS:
        arr = np.stack([p[k] for k in tr], axis=1)
        sarr = np.sort(arr, axis=1)
        lo, mid_v, hi = sarr[:, 0], sarr[:, 1], sarr[:, 2]
        out[f"trip_{tr[0]}_{tr[1]}_{tr[2]}"] = (hi - mid_v) / (mid_v - lo + EPS)
    return pd.DataFrame(out, index=df.index)


# ============================================================================
# 9. Multi-level imbalance slope (10 cols)
# ============================================================================

def compute_imb_slope(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for k in LOB_LEVELS:
        bs = df[f"bsize{k}"].to_numpy()
        as_ = df[f"asize{k}"].to_numpy()
        out[f"imb_lvl{k}"] = (bs - as_) / (bs + as_ + EPS)
    return pd.DataFrame(out, index=df.index)


# ============================================================================
# 10. EWMA-OFI total at 4 alphas
# ============================================================================

def _per_tick_e_total(df: pd.DataFrame) -> pd.Series:
    """Σ_k of per-tick OFI e_k(t) (Cont-Stoikov-Kukanov 2014)."""
    e_total = pd.Series(0.0, index=df.index)
    for k in LOB_LEVELS:
        b = df[f"bid{k}"]
        a = df[f"ask{k}"]
        bs = df[f"bsize{k}"]
        as_ = df[f"asize{k}"]
        b_p = b.shift(1); a_p = a.shift(1); bs_p = bs.shift(1); as_p = as_.shift(1)
        ind_b_up = (b >= b_p).astype(np.float64)
        ind_b_dn = (b <= b_p).astype(np.float64)
        ind_a_dn = (a <= a_p).astype(np.float64)
        ind_a_up = (a >= a_p).astype(np.float64)
        e = ind_b_up * bs - ind_b_dn * bs_p - ind_a_dn * as_ + ind_a_up * as_p
        e_total = e_total.add(e, fill_value=0.0)
    return e_total


def compute_ewma_ofi_total(df: pd.DataFrame, alphas=(0.05, 0.1, 0.3, 0.5)) -> pd.DataFrame:
    e_total = _per_tick_e_total(df).fillna(0.0)
    out = {}
    for alpha in alphas:
        out[f"ewma_ofi_total_a{alpha}"] = e_total.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    return pd.DataFrame(out, index=df.index)


# ============================================================================
# 11. GOFI rolling sum W=20
# ============================================================================

def compute_gofi(df: pd.DataFrame, W: int = 20) -> pd.DataFrame:
    """Generalized OFI = OFI_classic + passive size adjustment when price is
    unchanged (Cao-Hansch-Wang 2008).
    """
    g_total = pd.Series(0.0, index=df.index)
    for k in LOB_LEVELS:
        b = df[f"bid{k}"]; a = df[f"ask{k}"]
        bs = df[f"bsize{k}"]; as_ = df[f"asize{k}"]
        b_p = b.shift(1); a_p = a.shift(1); bs_p = bs.shift(1); as_p = as_.shift(1)
        ind_b_up = (b >= b_p).astype(np.float64)
        ind_b_dn = (b <= b_p).astype(np.float64)
        ind_a_dn = (a <= a_p).astype(np.float64)
        ind_a_up = (a >= a_p).astype(np.float64)
        e_classic = ind_b_up * bs - ind_b_dn * bs_p - ind_a_dn * as_ + ind_a_up * as_p
        ind_b_eq = (b == b_p).astype(np.float64)
        ind_a_eq = (a == a_p).astype(np.float64)
        e_passive = ind_b_eq * (bs - bs_p) - ind_a_eq * (as_ - as_p)
        g_total = g_total.add(e_classic + e_passive, fill_value=0.0)
    return pd.DataFrame({
        f"gofi_W{W}": g_total.rolling(W, min_periods=W).sum().to_numpy(),
    }, index=df.index)


# ============================================================================
# 12. Tick-rule signed volume (Lee-Ready)
# ============================================================================

def compute_tick_rule_signed(df: pd.DataFrame, windows=(10, 20, 30, 50)) -> pd.DataFrame:
    mid = (df["bid1"] + df["ask1"]) * 0.5
    dmid = mid.diff().to_numpy()
    sign_raw = np.sign(dmid)  # may be 0 when no move
    sign = _signed_carry_fwd(sign_raw)
    vol = df["volume_delta"].abs().to_numpy()  # take absolute; volume_delta sign is rate-of-change
    signed_vol = sign * vol
    sv = pd.Series(signed_vol, index=df.index)
    v = pd.Series(vol, index=df.index)
    out = {}
    for W in windows:
        num = sv.rolling(W, min_periods=W).sum()
        den = v.rolling(W, min_periods=W).sum()
        out[f"tick_imb_W{W}"] = (num / (den + EPS)).to_numpy()
    return pd.DataFrame(out, index=df.index)


# ============================================================================
# 13. Kyle's λ rolling W=50
# ============================================================================

def compute_kyle_lambda(df: pd.DataFrame, W: int = 50) -> pd.DataFrame:
    mid = (df["bid1"] + df["ask1"]) * 0.5
    dmid = mid.diff()
    sgn = np.sign(dmid.to_numpy())
    sgn = _signed_carry_fwd(sgn)
    vol = df["volume_delta"].abs().to_numpy()
    x = pd.Series(sgn * np.sqrt(vol), index=df.index)  # signed sqrt volume
    cov = _rolling_cov(dmid, x, W)
    var = _rolling_var(x, W)
    lam = (cov / (var + EPS)).to_numpy()
    return pd.DataFrame({f"kyle_lambda_W{W}": lam}, index=df.index)


# ============================================================================
# 14. Amihud illiquidity W=50
# ============================================================================

def compute_amihud(df: pd.DataFrame, W: int = 50) -> pd.DataFrame:
    mid = (df["bid1"] + df["ask1"]) * 0.5
    r_abs = mid.diff().abs()
    vol = df["volume_delta"].abs()
    ratio = r_abs / (vol + EPS)
    return pd.DataFrame({
        f"amihud_W{W}": ratio.rolling(W, min_periods=W).mean().to_numpy(),
    }, index=df.index)


# ============================================================================
# 15. Book slope per side (bid + ask) [2]
# ============================================================================

def compute_book_slope(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over k=2..10 of (price_k - price_1) / (cumulative size to k + ε).
    Higher |slope| = thicker / steeper book on that side.
    """
    bid_prices = np.column_stack([df[f"bid{k}"].to_numpy() for k in LOB_LEVELS])  # (T, 10)
    ask_prices = np.column_stack([df[f"ask{k}"].to_numpy() for k in LOB_LEVELS])
    bid_sizes = np.column_stack([df[f"bsize{k}"].to_numpy() for k in LOB_LEVELS])
    ask_sizes = np.column_stack([df[f"asize{k}"].to_numpy() for k in LOB_LEVELS])
    # cumulative size up to k
    cum_bs = np.cumsum(bid_sizes, axis=1)  # cum_bs[:, k-1] is sum_{j=1..k} bs_j
    cum_as = np.cumsum(ask_sizes, axis=1)

    # for k=2..10, slope_k = (b_k - b_1) / (cum_bs[k]) ; cum_bs[:, k-1] in 0-based
    p1_b = bid_prices[:, 0:1]
    p1_a = ask_prices[:, 0:1]
    diff_b = bid_prices[:, 1:] - p1_b  # (T, 9)
    diff_a = ask_prices[:, 1:] - p1_a
    cum_b_2_10 = cum_bs[:, 1:]  # k=2..10 columns
    cum_a_2_10 = cum_as[:, 1:]
    slope_bid = np.mean(diff_b / (cum_b_2_10 + EPS), axis=1)
    slope_ask = np.mean(diff_a / (cum_a_2_10 + EPS), axis=1)
    return pd.DataFrame({
        "book_slope_bid": slope_bid,
        "book_slope_ask": slope_ask,
    }, index=df.index)


# ============================================================================
# 16. Depth concentration top-3 / top-5
# ============================================================================

def compute_depth_concentration(df: pd.DataFrame) -> pd.DataFrame:
    bid_sizes = np.column_stack([df[f"bsize{k}"].to_numpy() for k in LOB_LEVELS])
    ask_sizes = np.column_stack([df[f"asize{k}"].to_numpy() for k in LOB_LEVELS])
    bs_total = bid_sizes.sum(axis=1)
    as_total = ask_sizes.sum(axis=1)
    return pd.DataFrame({
        "bid_top3_concentration": bid_sizes[:, :3].sum(axis=1) / (bs_total + EPS),
        "ask_top3_concentration": ask_sizes[:, :3].sum(axis=1) / (as_total + EPS),
        "bid_top5_concentration": bid_sizes[:, :5].sum(axis=1) / (bs_total + EPS),
        "ask_top5_concentration": ask_sizes[:, :5].sum(axis=1) / (as_total + EPS),
    }, index=df.index)


# ============================================================================
# 17. Depth skew (1)
# ============================================================================

def compute_depth_skew(df: pd.DataFrame) -> pd.DataFrame:
    bs_total = sum(df[f"bsize{k}"] for k in LOB_LEVELS)
    as_total = sum(df[f"asize{k}"] for k in LOB_LEVELS)
    return pd.DataFrame({
        "depth_skew": ((bs_total - as_total) / (bs_total + as_total + EPS)).to_numpy(),
    }, index=df.index)


# ============================================================================
# Aggregator
# ============================================================================

def _wmp_lvl1_from_df(df: pd.DataFrame) -> pd.Series:
    """Recompute WMP level 1 (avoid depending on T3 cache columns)."""
    b = df["bid1"].to_numpy()
    a = df["ask1"].to_numpy()
    bs = df["bsize1"].to_numpy()
    as_ = df["asize1"].to_numpy()
    denom = bs + as_
    wmp_n = (a * bs + b * as_) / np.where(denom == 0, 1.0, denom)
    wmp_f = (a + b) / 2.0
    wmp = np.where(denom == 0, wmp_f, wmp_n)
    return pd.Series(wmp, index=df.index)


def compute_r20(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all R20 Top-15 features for one session DataFrame."""
    wmp1 = _wmp_lvl1_from_df(df)
    parts = [
        compute_stoikov(df),
        compute_roll_spread(df),
        compute_realized_skew_and_signed_rv(df),
        compute_bipower_and_jump(df),
        compute_pairwise_diffs(df, wmp1),
        compute_triplet_imb(df, wmp1),
        compute_imb_slope(df),
        compute_ewma_ofi_total(df),
        compute_gofi(df),
        compute_tick_rule_signed(df),
        compute_kyle_lambda(df),
        compute_amihud(df),
        compute_book_slope(df),
        compute_depth_concentration(df),
        compute_depth_skew(df),
    ]
    return pd.concat(parts, axis=1)


def r20_feature_columns() -> List[str]:
    """Return the exact ordered column names produced by compute_r20()."""
    cols: List[str] = []
    # 1. Stoikov
    cols += ["stoikov_micro", "stoikov_micro_gap_mid"]
    # 2. Roll
    for W in (20, 30, 50):
        cols.append(f"roll_spread_W{W}")
    # 3. RSkew
    for W in (10, 20, 50):
        cols.append(f"r_skew_W{W}")
    # 4. Signed RV
    for W in (20, 50):
        cols += [f"rv_pos_W{W}", f"rv_neg_W{W}", f"rvs_signed_W{W}"]
    # 5. BV
    for W in (10, 20, 50):
        cols.append(f"bv_W{W}")
    # 6. BNS jump
    for W in (20, 50):
        cols.append(f"bns_jump_z_W{W}")
    # 7. Pairwise
    for x, y in PAIR_NAMES:
        cols.append(f"pair_{x}_{y}")
    # 8. Triplet
    for tr in TRIPLETS:
        cols.append(f"trip_{tr[0]}_{tr[1]}_{tr[2]}")
    # 9. Imbalance slope
    for k in LOB_LEVELS:
        cols.append(f"imb_lvl{k}")
    # 10. EWMA OFI total
    for alpha in (0.05, 0.1, 0.3, 0.5):
        cols.append(f"ewma_ofi_total_a{alpha}")
    # 11. GOFI
    cols.append("gofi_W20")
    # 12. Tick-rule
    for W in (10, 20, 30, 50):
        cols.append(f"tick_imb_W{W}")
    # 13. Kyle's λ
    cols.append("kyle_lambda_W50")
    # 14. Amihud
    cols.append("amihud_W50")
    # 15. Book slope
    cols += ["book_slope_bid", "book_slope_ask"]
    # 16. Concentration
    cols += [
        "bid_top3_concentration", "ask_top3_concentration",
        "bid_top5_concentration", "ask_top5_concentration",
    ]
    # 17. Depth skew
    cols.append("depth_skew")
    return cols


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    df = pd.read_parquet("data/features_v1/snapshot_sym0_date0_am.parquet")
    feats = compute_r20(df)
    cols_expected = r20_feature_columns()
    print(f"Output cols: {feats.shape[1]}, expected: {len(cols_expected)}")
    assert list(feats.columns) == cols_expected, (
        f"order mismatch:\n  got: {feats.columns.tolist()[:5]}...\n  exp: {cols_expected[:5]}..."
    )
    nan_per_col = feats.iloc[100:].isna().sum()
    print(f"NaN count per col after row 100 (max): {int(nan_per_col.max())}")
    print(f"  cols with any NaN: {(nan_per_col > 0).sum()}")
    if (nan_per_col > 0).any():
        print(f"  examples: {nan_per_col[nan_per_col > 0].head().to_dict()}")
    print("first row of micro-price family:", feats[["stoikov_micro", "stoikov_micro_gap_mid"]].iloc[0].to_dict())
    print("Roll W=20 at row 100:", float(feats["roll_spread_W20"].iloc[100]))
    print("imb_lvl1 first 3:", feats["imb_lvl1"].iloc[:3].tolist())
