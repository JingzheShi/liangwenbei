"""T88: Range-based volatility estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang).

OHLC fields in raw data are session-cumulative (high/low monotonic since session
open). Therefore, we DERIVE bar-level OHLC from `close` over a rolling window of
W ticks:
    O(W) = close[t-W+1]
    C(W) = close[t]
    H(W) = max(close[t-W+1..t])
    L(W) = min(close[t-W+1..t])

Since prices are normalized as 涨跌幅 (return-from-yesterday-close), all are on a
return scale. log(H/L) ≈ r_H - r_L (since (1+r) is close to 1).

Estimators:
  - Parkinson(1980)        : σ²_P = (r_H - r_L)^2 / (4 ln 2)
  - Garman-Klass(1980)     : σ²_GK = 0.5*(r_H - r_L)^2 - (2 ln 2 - 1)*(r_C - r_O)^2
  - Rogers-Satchell(1991)  : σ²_RS = (r_H - r_C)*(r_H - r_O) + (r_L - r_C)*(r_L - r_O)
  - Yang-Zhang (simplified): σ²_YZ ≈ k*σ²_OC + (1-k)*σ²_RS where σ²_OC = (r_C - r_O)^2,
                             k = 0.34/(1.34 + (W+1)/(W-1))  (canonical YZ weight, n=2 sub-bars)

Output (12 features per row):
  Raw σ at W=100 (4, σ × 1e4 scale): rv_park_W100, rv_gk_W100, rv_rs_W100, rv_yz_W100
  Vol regime ratios (8, log(σ_W / σ_W100)):
    rv_park_ratio_W{20,50}, rv_gk_ratio_W{20,50}, rv_rs_ratio_W{20,50}, rv_yz_ratio_W{20,50}

Design rationale:
  - W=100 raw σ: captures absolute vol level; KS across sym ~0.30-0.43 (within
    team-accepted range; close-only rv_w50 sits at 0.45).
  - Ratios short/long vol: sym-INVARIANT by construction (each sym's vol level
    cancels in numerator/denominator). Regime indicator: >0 means vol expanding,
    <0 means contracting. Captures vol clustering / mean reversion.

We use σ (not log σ²) for level features to avoid the EPS-floor bimodality at
small W when prices are discretized.

All features are sym-agnostic (no per-sym normalization or stats).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-14
LN2 = np.log(2.0)
SCALE = 1e4  # σ × 1e4 → basis-point-like magnitude
WINDOWS: Tuple[int, ...] = (20, 50, 100)
REF_W = 100  # baseline window for ratios
RATIO_WINDOWS: Tuple[int, ...] = (20, 50)
ESTIMATORS = ("park", "gk", "rs", "yz")


def feature_names() -> List[str]:
    names = []
    for est in ESTIMATORS:
        names.append(f"rv_{est}_W{REF_W}")
    for est in ESTIMATORS:
        for W in RATIO_WINDOWS:
            names.append(f"rv_{est}_ratio_W{W}")
    return names


def _to_sigma(x_var: np.ndarray) -> np.ndarray:
    """sqrt(max(x, 0)) * SCALE → bp-ish vol magnitude (no log artifact)."""
    return np.sqrt(np.maximum(x_var, 0.0)) * SCALE


def _sigma2_estimators(sub: np.ndarray) -> Dict[str, np.ndarray]:
    """All 4 σ² estimators for one (N, W) close window. Returns dict of (N,)."""
    O = sub[:, 0]
    C = sub[:, -1]
    H = sub.max(axis=-1)
    L = sub.min(axis=-1)
    rHL = H - L
    rCO = C - O

    sigma2_P = (rHL * rHL) / (4.0 * LN2)
    sigma2_GK = 0.5 * (rHL * rHL) - (2.0 * LN2 - 1.0) * (rCO * rCO)
    sigma2_RS = (H - C) * (H - O) + (L - C) * (L - O)
    n = 2
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    sigma2_YZ = k * (rCO * rCO) + (1.0 - k) * np.maximum(sigma2_RS, 0.0)

    return {
        "park": np.maximum(sigma2_P, 0.0),
        "gk": np.maximum(sigma2_GK, 0.0),
        "rs": np.maximum(sigma2_RS, 0.0),
        "yz": np.maximum(sigma2_YZ, 0.0),
    }


def compute_range_vol_batch(close_3d: np.ndarray) -> np.ndarray:
    """
    Args:
        close_3d: (N, T) array of close-prices (last T ticks ending at row's t).
                  T must be >= max(WINDOWS).

    Returns:
        (N, 12) float32 features:
          [4]  σ × 1e4 at W=100 for each of (park, gk, rs, yz)
          [8]  log(σ_W / σ_W=100) for W in (20, 50), each of 4 estimators
    """
    N, T = close_3d.shape
    assert T >= max(WINDOWS), f"need T >= {max(WINDOWS)}, got {T}"

    # Compute σ² for each window
    s2_per_W: Dict[int, Dict[str, np.ndarray]] = {}
    for W in WINDOWS:
        s2_per_W[W] = _sigma2_estimators(close_3d[:, -W:])

    out = np.zeros((N, 4 + 4 * len(RATIO_WINDOWS)), dtype=np.float64)
    col = 0

    # 1) Raw σ at W=REF_W (4 cols): vol level proxies
    for est in ESTIMATORS:
        s2_ref = s2_per_W[REF_W][est]
        out[:, col] = _to_sigma(s2_ref)
        col += 1

    # 2) Ratios log(σ_W / σ_REF_W) (8 cols): sym-invariant regime indicator
    for est in ESTIMATORS:
        s2_ref = s2_per_W[REF_W][est]
        # log(σ_W / σ_REF) = 0.5 * log((σ²_W + EPS) / (σ²_REF + EPS))
        denom = s2_ref + EPS
        for W in RATIO_WINDOWS:
            num = s2_per_W[W][est] + EPS
            ratio = 0.5 * np.log(num / denom)
            # Clip extreme: ratios > exp(5) or < exp(-5) are unstable; clip to ±5
            out[:, col] = np.clip(ratio, -5.0, 5.0)
            col += 1

    out = np.where(np.isfinite(out), out, 0.0).astype(np.float32)
    return out


if __name__ == "__main__":
    # Quick smoke test
    rng = np.random.default_rng(42)
    close = rng.normal(0.001, 0.0005, size=(1000, 100)).astype(np.float64)
    feats = compute_range_vol_batch(close)
    names = feature_names()
    print(f"close shape: {close.shape}")
    print(f"feats shape: {feats.shape}")
    assert feats.shape == (1000, 12), feats.shape
    assert len(names) == 12
    print("\nFeature stats:")
    for i, n in enumerate(names):
        v = feats[:, i]
        print(f"  {n:25s} mean={v.mean():+8.3f} std={v.std():.3f} "
              f"min={v.min():+.2f} max={v.max():+.2f}")
    print("\nOK")
