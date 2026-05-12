"""T68 Stage 5 features (batch-vectorized).

20 new features in 6 families, all sym-agnostic & W <= 100, causal:

  A. Adaptive momentum (3): (mid[-1] - mid[-W]) / std(mid[-W:]),
     for W in (20, 50, 100). NaN→0 clip [-10,10].
  B. OFI toxicity (4): corr(OFI_lvl[-W:], dmid[-W:]) for
     (lvl,W) in {(1,20),(1,50),(5,20),(5,50)}.
  C. Multi-scale signed bipower (3): signed_BV =
     (π/2) Σ_t sign(r_t)*|r_t|*|r_{t-1}|, normalised by
     RV_W (= Σ r²) -> ratio. For W in (20,50,100).
  D. Stationary spread regime (3): (spread1[-1] - median(spread1[-W:])) /
     IQR(spread1[-W:])  for W in (20,50,100). Clip [-10,10].
  E. Trade-direction persistence (3): lag-1 autocorr of sign(dmid)
     over W in (20,50,100). Clip [-1,1].
  F. Liquidity asymmetry (4): log((1+sum_top5_bid_W)/(1+sum_top5_ask_W))
     for W in (5,20,50,100).

All inputs available from the 100-tick LOB window. No date / sym used.

Inputs: X3d (N, T=100, K) raw column tensor, col_idx mapping.
Output: (N, 20) float64.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-8

ADAPT_MOM_WINDOWS = (20, 50, 100)
OFI_TOX_PAIRS = ((1, 20), (1, 50), (5, 20), (5, 50))
SIGNED_BV_WINDOWS = (20, 50, 100)
SPREAD_REG_WINDOWS = (20, 50, 100)
TRADE_PERS_WINDOWS = (20, 50, 100)
LIQ_ASYM_WINDOWS = (5, 20, 50, 100)


def _seg_mean(x: np.ndarray, W: int) -> np.ndarray:
    return x[:, -W:].mean(axis=-1)


def _seg_std(x: np.ndarray, W: int) -> np.ndarray:
    return x[:, -W:].std(axis=-1, ddof=0)


def compute_stage5_batch(X3d: np.ndarray, col_idx: Dict[str, int]) -> np.ndarray:
    """Stage 5 batch feature extractor.

    X3d: (N, T, K) float64.
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

    # --- precompute Δmid, log returns ---
    dmid = np.zeros_like(mid)
    dmid[:, 1:] = mid[:, 1:] - mid[:, :-1]
    dmid = np.where(np.isfinite(dmid), dmid, 0.0)

    mid_safe = np.where(mid > 0, mid, np.nan)
    r = np.zeros_like(mid_safe)
    r[:, 1:] = np.log(mid_safe[:, 1:] / mid_safe[:, :-1])
    r = np.where(np.isfinite(r), r, 0.0)

    # --- B. OFI toxicity (4) ---
    # Build per-tick OFI for needed levels (1, 5).
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
        # Level-OFI (Cont 2014): bid contributes when price up (+bs_now) or
        # down (-bs_prev) or equal (Δbs); ask is symmetric with sign flip.
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
        # Pearson correlation
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
    bv_term = sign_r * abs_r * abs_r_lag  # signed bipower term per tick
    r2 = r * r
    factor = np.pi / 2.0
    for W in SIGNED_BV_WINDOWS:
        sbv = factor * bv_term[:, -W:].sum(axis=-1)
        rv = r2[:, -W:].sum(axis=-1) + EPS
        v = sbv / rv  # ratio bounded loosely
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
        # If IQR == 0 the regime is undefined (constant) -> 0.
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
    # sum of top-5 bid sizes and top-5 ask sizes per tick
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
