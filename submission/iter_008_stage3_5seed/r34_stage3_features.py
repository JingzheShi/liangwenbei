"""T53 R34 Stage 3 feature computations (sym-invariant, complementary to stage 1+2).

Includes 5 feature families (rank #11..#15 from r34_top15_features.md):
  #11 EWMA-residual midprice (H4):       1 dim     (mid - ewma_mid)/ewma_mid, alpha=0.05
  #12 Multi-W RV ratio       (F1):       3 dims    rv5/rv50, rv20/rv100, rv50/rv100
  #13 Bipower variation      (C7):       4 dims    BV at W=20/50/100 + J_share at W=100
  #14 Cancel-pressure imb    (E7):       3 dims    cb_share - ca_share at W=20/50/100
  #15 Roll's effective spread(B1):       3 dims    2*sqrt(max(-cov(dmid),0))/spread1 at W=30/50/100

Total dim = 1 + 3 + 4 + 3 + 3 = 14.

All operations are session-scoped; **no cross-session shift**, no look-ahead.
Max window = 100 (compatible with platform 100-tick inference window).

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


# ---- #11 EWMA-Residual Midprice (H4) -----------------------------------
# NOTE: midprice in this dataset is pre-normalized and centered near 0,
# so dividing by ewma is numerically unstable. Use raw (mid - ewma)
# (already unitless across syms because midprice is pre-normalized).

EWMA_RES_ALPHA = 0.05  # half-life ~14 ticks


def compute_ewma_resid_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """(mid - ewm(mid, a)). Mid-price innovation; unitless across pre-normalized syms."""
    n_out = valid_hi - valid_lo + 1
    out = np.zeros((n_out, 1), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    ewma = pd.Series(mid).ewm(alpha=EWMA_RES_ALPHA, adjust=False).mean().to_numpy()
    resid = mid - ewma
    resid = np.where(np.isfinite(resid), resid, 0.0).astype(np.float32)
    # clamp tiny outliers; typical resid std ~1e-3 across all syms.
    resid = np.clip(resid, -0.05, 0.05)
    out[:, 0] = resid[rows]
    return out


def ewma_resid_feature_names() -> List[str]:
    return [f"mid_ewma_resid_a{EWMA_RES_ALPHA}"]


# ---- #12 Multi-W RV ratio (F1) -----------------------------------------

# We compute rv_W (sum of squared log-returns over W) for needed windows
# then form 3 ratios:
#   rv5/rv50, rv20/rv100, rv50/rv100
RV_WINDOWS = (5, 20, 50, 100)
RV_RATIO_PAIRS = ((5, 50), (20, 100), (50, 100))


def compute_rv_ratio_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    n_out = valid_hi - valid_lo + 1
    F = len(RV_RATIO_PAIRS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)
    r2 = r * r
    s_r2 = pd.Series(r2)

    rv = {}
    for W in RV_WINDOWS:
        rv[W] = s_r2.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)

    for i, (Wn, Wd) in enumerate(RV_RATIO_PAIRS):
        ratio = rv[Wn] / (rv[Wd] + EPS)
        ratio = np.where(np.isfinite(ratio), ratio, 0.0).astype(np.float32)
        # ratio is unbounded but heavy-tailed; clip extreme spikes
        ratio = np.clip(ratio, 0.0, 100.0)
        out[:, i] = ratio[rows]
    return out


def rv_ratio_feature_names() -> List[str]:
    return [f"rv_ratio_W{n}_W{d}" for (n, d) in RV_RATIO_PAIRS]


# ---- #13 Bipower variation (C7) ----------------------------------------

# BV is in vol units → scales with sym-specific volatility (FAIL on KS).
# We expose only the J_share variant: (RV - BV)/RV at multiple windows,
# strictly bounded in [0, 1] regardless of sym.
JSHARE_WINDOWS = (20, 30, 50, 100)


def compute_bv_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Jump-share J_W = max(0, (RV_W - BV_W) / RV_W) in [0, 1].

    BV_W = (pi/2) * sum_{t-W+1..t} |r_t| * |r_{t-1}|  (jump-robust vol)
    RV_W = sum r_t^2 within W
    Sym-agnostic by construction (ratio of vol estimators).
    """
    n_out = valid_hi - valid_lo + 1
    F = len(JSHARE_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    mid = df["midprice"].to_numpy(dtype=np.float64, copy=False)
    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[1:] = np.log(mid_safe[1:] / mid_safe[:-1])
    r = np.where(np.isfinite(r), r, 0.0)
    abs_r = np.abs(r)
    abs_r_lag = np.zeros_like(abs_r)
    abs_r_lag[1:] = abs_r[:-1]
    prod = abs_r * abs_r_lag

    s_prod = pd.Series(prod)
    s_r2 = pd.Series(r * r)
    factor = np.pi / 2.0

    for i, W in enumerate(JSHARE_WINDOWS):
        bv = factor * s_prod.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)
        rv = s_r2.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)
        j_share = (rv - bv) / (rv + EPS)
        j_share = np.where(np.isfinite(j_share), j_share, 0.0).astype(np.float32)
        j_share = np.clip(j_share, 0.0, 1.0)
        out[:, i] = j_share[rows]
    return out


def bv_feature_names() -> List[str]:
    return [f"jshare_W{W}" for W in JSHARE_WINDOWS]


# ---- #14 Cancel-Pressure Imbalance (E7) --------------------------------

CANCEL_WINDOWS = (20, 50, 100)


def compute_cancel_imb_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """rolling cancel-share imbalance:
        cb_share_W = sum(cb_intst, W) / sum(lb_intst+mb_intst+cb_intst, W)
        ca_share_W = sum(ca_intst, W) / sum(la_intst+ma_intst+ca_intst, W)
        cancel_imb_W = cb_share_W - ca_share_W   (in [-1, 1])
    """
    n_out = valid_hi - valid_lo + 1
    F = len(CANCEL_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    lb = df["lb_intst"].to_numpy(dtype=np.float64, copy=False)
    mb = df["mb_intst"].to_numpy(dtype=np.float64, copy=False)
    cb = df["cb_intst"].to_numpy(dtype=np.float64, copy=False)
    la = df["la_intst"].to_numpy(dtype=np.float64, copy=False)
    ma = df["ma_intst"].to_numpy(dtype=np.float64, copy=False)
    ca = df["ca_intst"].to_numpy(dtype=np.float64, copy=False)

    bid_total = lb + mb + cb
    ask_total = la + ma + ca

    s_cb = pd.Series(cb)
    s_ca = pd.Series(ca)
    s_bt = pd.Series(bid_total)
    s_at = pd.Series(ask_total)

    for i, W in enumerate(CANCEL_WINDOWS):
        cb_sum = s_cb.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)
        ca_sum = s_ca.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)
        bt_sum = s_bt.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)
        at_sum = s_at.rolling(W, min_periods=W).sum().to_numpy(dtype=np.float64)
        cb_share = cb_sum / (bt_sum + EPS)
        ca_share = ca_sum / (at_sum + EPS)
        imb = cb_share - ca_share
        imb = np.where(np.isfinite(imb), imb, 0.0).astype(np.float32)
        imb = np.clip(imb, -1.0, 1.0)
        out[:, i] = imb[rows]
    return out


def cancel_imb_feature_names() -> List[str]:
    return [f"cancel_imb_W{W}" for W in CANCEL_WINDOWS]


# ---- #15 Roll's effective spread (B1) ----------------------------------

ROLL_WINDOWS = (30, 50, 100)


def compute_roll_spread_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    """Roll 1984 effective spread via lag-1 autocov of trade-price changes.

    Uses `close` (last traded price within tick — captures bid-ask bounce);
    midprice is too smooth and yields zero signal.
    Normalized by quoted spread (spread1 + 1.0; spread1 is mean-shifted by -1
    in this dataset's pre-normalization → spread1+1.0 == ask1-bid1).

    For each W in {30, 50, 100}:
      cov_W = rolling cov(d_close, d_close_lag1)
      eff_W = 2 * sqrt(max(-cov_W, 0))
      ratio_W = eff_W / (quoted_spread + eps)   (unitless, ~ O(1) cross-sym)
    """
    n_out = valid_hi - valid_lo + 1
    F = len(ROLL_WINDOWS)
    out = np.zeros((n_out, F), dtype=np.float32)
    rows = np.arange(valid_lo, valid_hi + 1)

    close = df["close"].to_numpy(dtype=np.float64, copy=False)
    dc = np.zeros_like(close)
    dc[1:] = close[1:] - close[:-1]
    dc = np.where(np.isfinite(dc), dc, 0.0)
    dc_lag = np.zeros_like(dc)
    dc_lag[1:] = dc[:-1]

    qspr = df["spread1"].to_numpy(dtype=np.float64, copy=False) + 1.0  # actual quoted spread

    s_x = pd.Series(dc)
    s_y = pd.Series(dc_lag)
    s_xy = pd.Series(dc * dc_lag)

    for i, W in enumerate(ROLL_WINDOWS):
        mx = s_x.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        my = s_y.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        mxy = s_xy.rolling(W, min_periods=W).mean().to_numpy(dtype=np.float64)
        cov = mxy - mx * my
        eff = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
        ratio = eff / (qspr + EPS)
        ratio = np.where(np.isfinite(ratio), ratio, 0.0).astype(np.float32)
        # Effective spread typically O(1) * quoted spread; allow up to 10x for outliers
        ratio = np.clip(ratio, 0.0, 10.0)
        out[:, i] = ratio[rows]
    return out


def roll_spread_feature_names() -> List[str]:
    return [f"roll_eff_spr_ratio_W{W}" for W in ROLL_WINDOWS]


# ---- All-in-one --------------------------------------------------------

def compute_all_session(df: pd.DataFrame, valid_lo: int, valid_hi: int) -> np.ndarray:
    a = compute_ewma_resid_session(df, valid_lo, valid_hi)
    b = compute_rv_ratio_session(df, valid_lo, valid_hi)
    c = compute_bv_session(df, valid_lo, valid_hi)
    d = compute_cancel_imb_session(df, valid_lo, valid_hi)
    e = compute_roll_spread_session(df, valid_lo, valid_hi)
    return np.concatenate([a, b, c, d, e], axis=1)


def all_feature_names() -> List[str]:
    return (
        ewma_resid_feature_names()
        + rv_ratio_feature_names()
        + bv_feature_names()
        + cancel_imb_feature_names()
        + roll_spread_feature_names()
    )


def feature_block_lengths() -> Tuple[int, int, int, int, int]:
    return (
        len(ewma_resid_feature_names()),
        len(rv_ratio_feature_names()),
        len(bv_feature_names()),
        len(cancel_imb_feature_names()),
        len(roll_spread_feature_names()),
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
    print(f"blocks: ewma_resid={blocks[0]}, rv_ratio={blocks[1]}, bv={blocks[2]}, cancel_imb={blocks[3]}, roll_spr={blocks[4]}")
    print("any nan?", np.isnan(feats).any())
    print("any inf?", np.isinf(feats).any())
    print("first row:", feats[0])
    print(f"min/max/mean per col:")
    for j, n in enumerate(names):
        col = feats[:, j]
        print(f"  {n:30s} min={col.min():+.4g} max={col.max():+.4g} mean={col.mean():+.4g} std={col.std():.4g}")
