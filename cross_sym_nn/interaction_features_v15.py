"""V15 interaction features: V9 base (136 channels) + 20 NEW temporal-aggregation
derived features (channel 136..155 → output 156 total).

These 20 'temporal agg' features are derived from existing window-scaled L8
features (rv_w*, signed_rv_W*, mlofi_W{5,20}_lvl*, rskew_W*, kyle_inv_W50).
They expose volatility-regime, momentum-vs-volatility, OFI-window-expansion,
and cross-sym normalized variants that V9 does NOT include.

All formulas sym-agnostic and date-agnostic. Vectorized NumPy.
"""
from __future__ import annotations
import numpy as np

from interaction_features import _sup, _vec, csec_demean_vec, _init_names
from interaction_features_v8 import compute_v9_features, csec_signed_rank_vec, S

EPS = 1e-6


def compute_v15_temporal(X_grp: np.ndarray, verbose: bool = False):
    """20 temporal-aggregation derived features per sym.

    Returns:
      X_temp: (N, 5, 20) float32
      names:  list[str] len 20
    """
    if X_grp.ndim != 3 or X_grp.shape[1] != S:
        raise ValueError(f'Expected (N, 5, F), got {X_grp.shape}')
    if X_grp.dtype != np.float32:
        X_grp = X_grp.astype(np.float32, copy=False)
    X_grp = np.nan_to_num(X_grp, nan=0.0, posinf=0.0, neginf=0.0)
    _init_names()

    rv5  = _vec(X_grp, _sup('rv_w5'))
    rv20 = _vec(X_grp, _sup('rv_w20'))
    rv50 = _vec(X_grp, _sup('rv_w50'))
    srv20  = _vec(X_grp, _sup('signed_rv_W20'))
    srv50  = _vec(X_grp, _sup('signed_rv_W50'))
    srv100 = _vec(X_grp, _sup('signed_rv_W100'))
    rsk20 = _vec(X_grp, _sup('rskew_W20'))
    rsk50 = _vec(X_grp, _sup('rskew_W50'))
    ofi5_l1  = _vec(X_grp, _sup('mlofi_W5_lvl1'))
    ofi20_l1 = _vec(X_grp, _sup('mlofi_W20_lvl1'))
    sum5_15  = sum(_vec(X_grp, _sup(f'mlofi_W5_lvl{k}'))  for k in range(1, 6))
    sum20_15 = sum(_vec(X_grp, _sup(f'mlofi_W20_lvl{k}')) for k in range(1, 6))
    kyle = _vec(X_grp, _sup('kyle_inv_W50'))

    def safe_ratio(a, b):
        return (a / np.maximum(np.abs(b), EPS)).astype(np.float32)

    feats = []
    names = []

    # 1-3: rv ratios
    feats.append(safe_ratio(rv50, rv5));   names.append('T01_rv50_over_rv5')
    feats.append(safe_ratio(rv50, rv20));  names.append('T02_rv50_over_rv20')
    feats.append(safe_ratio(rv20, rv5));   names.append('T03_rv20_over_rv5')
    # 4: vol expansion abs
    feats.append((rv50 - rv20).astype(np.float32)); names.append('T04_rv50_minus_rv20')
    # 5-6: signed rv momentum delta (lookback expansion)
    feats.append((srv50 - srv20).astype(np.float32));   names.append('T05_srv50_minus_srv20')
    feats.append((srv100 - srv50).astype(np.float32));  names.append('T06_srv100_minus_srv50')
    # 7-8: skew delta + skew*vol product
    feats.append((rsk50 - rsk20).astype(np.float32));   names.append('T07_rsk50_minus_rsk20')
    feats.append((rsk50 * rv50).astype(np.float32));    names.append('T08_rsk50_x_rv50')
    # 9-10: OFI expansion (top-level + broad)
    feats.append((ofi20_l1 - ofi5_l1).astype(np.float32));  names.append('T09_ofi20l1_minus_ofi5l1')
    feats.append((sum20_15 - sum5_15).astype(np.float32));  names.append('T10_sumW20_minus_sumW5')
    # 11: momentum-vol product
    feats.append((srv50 * rv50).astype(np.float32));    names.append('T11_srv50_x_rv50')
    # 12: kyle illiquidity-vol composite
    feats.append((kyle * rv50).astype(np.float32));     names.append('T12_kyle_x_rv50')
    # 13: cross-sym demean of rv50 (vol-regime cross-sym)
    feats.append(csec_demean_vec(rv50).astype(np.float32));  names.append('T13_csec_demean_rv50')
    # 14: cross-sym demean of srv50 (momentum cross-sym)
    feats.append(csec_demean_vec(srv50).astype(np.float32)); names.append('T14_csec_demean_srv50')
    # 15: cross-sym demean of ofi20_l1
    feats.append(csec_demean_vec(ofi20_l1).astype(np.float32)); names.append('T15_csec_demean_ofi20l1')
    # 16-17: cross-sym signed rank of rv50, srv50
    feats.append(csec_signed_rank_vec(rv50).astype(np.float32));  names.append('T16_csec_rank_rv50')
    feats.append(csec_signed_rank_vec(srv50).astype(np.float32)); names.append('T17_csec_rank_srv50')
    # 18: directional ofi
    feats.append((ofi5_l1 * np.sign(srv50)).astype(np.float32));  names.append('T18_dir_ofi5l1')
    # 19: signed_rv difference momentum acceleration
    accel = (srv50 - srv20) - (srv20 - srv20)  # equivalent to srv50 - srv20; reuse different combination
    # use a different composite: signed_rv * vol_ratio
    feats.append((srv50 * safe_ratio(rv50, rv20)).astype(np.float32)); names.append('T19_srv50_x_volRatio')
    # 20: ofi acceleration (signed)
    feats.append((ofi20_l1 * (ofi20_l1 - ofi5_l1)).astype(np.float32)); names.append('T20_ofi_accel')

    X_temp = np.stack(feats, axis=2).astype(np.float32)  # (N, 5, 20)
    X_temp = np.nan_to_num(X_temp, nan=0.0, posinf=0.0, neginf=0.0)
    if verbose:
        for i, n in enumerate(names):
            arr = X_temp[:, :, i]
            print(f'  {n}: min={arr.min():.3f} max={arr.max():.3f} std={arr.std():.3f}')
    return X_temp, names


def compute_v15_features(X_grp: np.ndarray, verbose: bool = False):
    """V9 features (136 ch) + 20 new temporal-agg → (N, 5, 156)."""
    X_v9, names_old, names_new = compute_v9_features(X_grp, verbose=verbose)
    X_temp, names_temp = compute_v15_temporal(X_grp, verbose=verbose)
    X_combined = np.concatenate([X_v9, X_temp], axis=2).astype(np.float32, copy=False)
    return X_combined, names_old, names_new, names_temp


if __name__ == '__main__':
    np.random.seed(0)
    N = 16
    X = np.random.randn(N, 5, 359).astype(np.float32)
    X_combined, no, nn_, nt = compute_v15_features(X, verbose=False)
    print(f'X_combined shape: {X_combined.shape} (expect (16, 5, 156))')
    print(f'names_old={len(no)} names_new={len(nn_)} names_temp={len(nt)}')
    print(f'finite: {np.isfinite(X_combined).all()}')
    print(f'min={X_combined.min():.3f} max={X_combined.max():.3f}')
