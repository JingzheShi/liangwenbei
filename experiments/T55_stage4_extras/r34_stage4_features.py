"""T55 R34 Stage 4 feature computations (sym-invariant extras).

3 feature families (R33 #5/#6 + a new spread regression residual):

  A. Triplet imbalance (R33 #5):
       For each triplet (a, b, c) of correlated columns, compute row-wise:
         sorted = sort([a, b, c])  -> [low, mid, high]
         imb = (high - mid) / (high - low + EPS)   in [0, 1]
       Bounded variant of Optiver's triplet imbalance — more stable across
       sym scales than the unbounded (max-mid)/(mid-min) form.
       24 triplets -> 24 dims

  B. Hull MA Fibonacci (R33 #6):
       HMA(n) = WMA(sqrt(n)) ( 2*WMA(n/2) - WMA(n) )
       Apply 4 base series x 5 Fibonacci windows {5, 13, 34, 55, 89}:
         - mid - HMA(mid)             (residual; pre-norm midprice ~ unitless)
         - HMA(log_return)            (smoothed return)
         - spread1 - HMA(spread1)     (residual)
         - imbalance - HMA(imbalance) (residual)
       4 x 5 = 20 dims

  C. Spread regression residual (new):
       Within W in {20, 50, 100}, OLS-fit  mid_diff[t] ~ a*spread[t] + b
       on rolling window of W observations, output residual at current t,
       standardized by rolling std of mid_diff.  3 dims.

Total = 24 + 20 + 3 = 47 dims.

All operations are session-scoped.  No look-ahead, no cross-session shift.
Max window in any computation = 100 (HMA(89) lookback ~96; W=100 OLS = 100; OK).

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


# =================================================================== #
#  A. Triplet imbalance                                               #
# =================================================================== #

# Each entry: (col_a, col_b, col_c) — 3 *comparable-scale* columns.
TRIPLETS: List[Tuple[str, str, str]] = [
    # quotes (skip bid1/mid/ask1 — mid is exactly average of bid1,ask1; constant)
    ("bid2", "midprice", "ask2"),
    ("bid3", "midprice", "ask3"),
    ("bid5", "midprice", "ask5"),
    # bid sizes
    ("bsize1", "bsize2", "bsize3"),
    ("bsize3", "bsize4", "bsize5"),
    ("bsize5", "bsize7", "bsize10"),
    # ask sizes
    ("asize1", "asize2", "asize3"),
    ("asize3", "asize4", "asize5"),
    ("asize5", "asize7", "asize10"),
    # bid_diff (level-to-level price gaps; all in price units)
    ("bid_diff1", "bid_diff2", "bid_diff3"),
    # ask_diff
    ("ask_diff1", "ask_diff2", "ask_diff3"),
    # spreads
    ("spread1", "spread2", "spread3"),
    ("spread1", "spread5", "spread10"),
    ("spread3", "spread5", "spread7"),
    # mlofi (multi-level order-flow imbalance, dimensionless)
    ("mlofi_W5_lvl1", "mlofi_W5_lvl2", "mlofi_W5_lvl3"),
    ("mlofi_W20_lvl1", "mlofi_W20_lvl2", "mlofi_W20_lvl3"),
    ("mlofi_W60_lvl1", "mlofi_W60_lvl2", "mlofi_W60_lvl3"),
    # weighted midprices per level (price units)
    ("wmp_lvl1", "wmp_lvl2", "wmp_lvl3"),
    ("wmp_lvl3", "wmp_lvl5", "wmp_lvl10"),
    # order intensities (bid / ask sides)
    ("lb_intst", "mb_intst", "cb_intst"),
    ("la_intst", "ma_intst", "ca_intst"),
    # realized variance windows
    ("rv_w5", "rv_w10", "rv_w20"),
    # bid/ask rates near top
    ("bid_rate1", "bid_rate2", "bid_rate3"),
    ("ask_rate1", "ask_rate2", "ask_rate3"),
]


def _triplet_imb(cols: np.ndarray) -> np.ndarray:
    """cols shape (T, 3). Returns (T,) with bounded triplet imbalance:
        sorted = sort(row)  ->  [lo, md, hi]
        out    = (hi - md) / (hi - lo + EPS)   in [0, 1]

    0  -> mid is at the top (concentration near max);
    1  -> mid is at the bottom (concentration near min);
    0.5 -> mid sits exactly in the middle of [lo, hi].

    Sym-invariant by construction (ratio of comparable quantities).
    Stays valid even when lo == hi (then output = 0; degenerate row).
    """
    sorted_arr = np.sort(cols, axis=1)
    lo = sorted_arr[:, 0]
    md = sorted_arr[:, 1]
    hi = sorted_arr[:, 2]
    rng = hi - lo
    out = (hi - md) / (rng + EPS)
    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def compute_triplet_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    n_out = valid_hi - valid_lo + 1
    F = len(TRIPLETS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    for i, (a, b, c) in enumerate(TRIPLETS):
        cols = np.stack([
            df[a].to_numpy(dtype=np.float64, copy=False),
            df[b].to_numpy(dtype=np.float64, copy=False),
            df[c].to_numpy(dtype=np.float64, copy=False),
        ], axis=1)  # (T, 3)
        imb = _triplet_imb(cols)
        out[:, i] = imb[rows]
    return out


def triplet_feature_names() -> List[str]:
    return [f"trip_{a}_{b}_{c}" for (a, b, c) in TRIPLETS]


# =================================================================== #
#  B. Hull MA Fibonacci                                               #
# =================================================================== #

HMA_WINDOWS = (5, 13, 34, 55, 89)
# All HMA windows: max actual lookback = 89 + sqrt(89)-1 = 89+8 = 97 < 100  OK


def _wma_causal(x: np.ndarray, n: int) -> np.ndarray:
    """Causal weighted moving average. Weights = 1, 2, ..., n (most recent = n).
    y[t] = sum_{i=0..n-1} w[i] * x[t - n + 1 + i] / sum(w)
    Returns array same shape as x; first (n-1) values are NaN.
    """
    n = int(n)
    if n <= 1:
        return x.astype(np.float64, copy=True)
    w = np.arange(1, n + 1, dtype=np.float64)
    w /= w.sum()
    # convolve flips the kernel internally -> use w[::-1] so most recent gets weight n
    out = np.full_like(x, np.nan, dtype=np.float64)
    if len(x) >= n:
        # 'valid': out len = len(x) - n + 1, first valid index in result is n-1 in x
        conv = np.convolve(x.astype(np.float64), w[::-1], mode="valid")
        out[n - 1:] = conv
    return out


def _hma(x: np.ndarray, n: int) -> np.ndarray:
    """HMA(n) = WMA(sqrt(n))( 2*WMA(n/2) - WMA(n) ). Causal."""
    n = int(n)
    half = max(2, n // 2)
    sq = max(2, int(round(np.sqrt(n))))
    a = _wma_causal(x, half)
    b = _wma_causal(x, n)
    inner = 2.0 * a - b  # NaN where b is NaN
    # WMA on a series with leading NaNs: convolve produces NaN-contaminated values
    # for the first (n-1)+(sq-1) outputs. We use a NaN-safe path: replace NaNs
    # with 0 in inner and rely on the fact that we'll discard the first valid_lo rows.
    inner_filled = np.where(np.isfinite(inner), inner, 0.0)
    out = _wma_causal(inner_filled, sq)
    # Set positions where the underlying b is NaN to NaN (they're invalid)
    invalid_mask = ~np.isfinite(b)
    out[invalid_mask] = np.nan
    return out


def compute_hma_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    n_out = valid_hi - valid_lo + 1
    F = 4 * len(HMA_WINDOWS)  # 4 base series, 5 windows = 20
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    spread = df["spread1"].to_numpy(dtype=np.float64, copy=False)
    imb = df["imbalance"].to_numpy(dtype=np.float64, copy=False)

    # log-return series (causal)
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)

    col = 0
    for W in HMA_WINDOWS:
        # 1) mid - HMA(mid): residual midprice
        h_mid = _hma(mid, W)
        resid_mid = mid - h_mid
        resid_mid = np.where(np.isfinite(resid_mid), resid_mid, 0.0)
        out[:, col] = np.clip(resid_mid[rows], -0.05, 0.05).astype(np.float32)
        col += 1

        # 2) HMA(log_return): smoothed return
        h_ret = _hma(r, W)
        h_ret = np.where(np.isfinite(h_ret), h_ret, 0.0)
        out[:, col] = np.clip(h_ret[rows], -0.005, 0.005).astype(np.float32)
        col += 1

        # 3) spread1 - HMA(spread1): residual spread
        h_sp = _hma(spread, W)
        resid_sp = spread - h_sp
        resid_sp = np.where(np.isfinite(resid_sp), resid_sp, 0.0)
        out[:, col] = np.clip(resid_sp[rows], -1.0, 1.0).astype(np.float32)
        col += 1

        # 4) imbalance - HMA(imbalance): residual imb
        h_imb = _hma(imb, W)
        resid_imb = imb - h_imb
        resid_imb = np.where(np.isfinite(resid_imb), resid_imb, 0.0)
        out[:, col] = np.clip(resid_imb[rows], -1.0, 1.0).astype(np.float32)
        col += 1

    assert col == F, f"col={col} F={F}"
    return out


def hma_feature_names() -> List[str]:
    names = []
    for W in HMA_WINDOWS:
        names.append(f"hma_resid_mid_W{W}")
        names.append(f"hma_logret_W{W}")
        names.append(f"hma_resid_sp1_W{W}")
        names.append(f"hma_resid_imb_W{W}")
    return names


# =================================================================== #
#  C. Spread regression residual                                      #
# =================================================================== #

REG_WINDOWS = (20, 50, 100)


def compute_spread_resid_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Rolling OLS  mid_diff ~ a * (spread1+1) + b   over W observations.
    Output:  residual / (sqrt(rolling_var(mid_diff)) + EPS)   (standardized).
    """
    n_out = valid_hi - valid_lo + 1
    F = len(REG_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    spread = df["spread1"].to_numpy(dtype=np.float64, copy=False) + 1.0  # actual quoted spread

    y = np.zeros_like(mid)
    y[1:] = mid[1:] - mid[:-1]
    y = np.where(np.isfinite(y), y, 0.0)
    x = np.where(np.isfinite(spread), spread, 0.0)

    s_x = pd.Series(x)
    s_y = pd.Series(y)
    s_xy = pd.Series(x * y)
    s_xx = pd.Series(x * x)
    s_yy = pd.Series(y * y)

    for i, W in enumerate(REG_WINDOWS):
        mx = s_x.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        my = s_y.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        mxy = s_xy.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        mxx = s_xx.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        myy = s_yy.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)

        var_x = mxx - mx * mx
        var_y = myy - my * my
        cov_xy = mxy - mx * my

        a = np.where(var_x > EPS, cov_xy / (var_x + EPS), 0.0)
        b = my - a * mx
        y_hat = a * x + b
        resid = y - y_hat
        std_y = np.sqrt(np.maximum(var_y, 0.0))
        z = resid / (std_y + EPS)
        z = np.where(np.isfinite(z), z, 0.0).astype(np.float32)
        z = np.clip(z, -10.0, 10.0)
        out[:, i] = z[rows]
    return out


def spread_resid_feature_names() -> List[str]:
    return [f"spr_reg_resid_W{W}" for W in REG_WINDOWS]


# =================================================================== #
#  All in one                                                         #
# =================================================================== #

def compute_all_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    a = compute_triplet_session(df, valid_lo, valid_hi)
    b = compute_hma_session(df, valid_lo, valid_hi)
    c = compute_spread_resid_session(df, valid_lo, valid_hi)
    return np.concatenate([a, b, c], axis=1)


def all_feature_names() -> List[str]:
    return triplet_feature_names() + hma_feature_names() + spread_resid_feature_names()


def feature_block_lengths() -> Tuple[int, int, int]:
    return (
        len(triplet_feature_names()),
        len(hma_feature_names()),
        len(spread_resid_feature_names()),
    )


if __name__ == "__main__":
    import os
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
    print(f"blocks: triplet={blocks[0]}, hma={blocks[1]}, resid={blocks[2]}")
    print("any nan?", np.isnan(feats).any())
    print("any inf?", np.isinf(feats).any())
    print(f"min/max/mean per col:")
    for j, n in enumerate(names):
        col = feats[:, j]
        print(f"  {j:3d} {n:38s} min={col.min():+.4g} max={col.max():+.4g} "
              f"mean={col.mean():+.4g} std={col.std():.4g}")
