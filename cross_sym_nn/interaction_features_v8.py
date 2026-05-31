"""V8 interaction feature engineering — Top 20 new features (NF*) added to existing Top 50.

This module computes (N, 5, 116 + 20) per-sym channels:
- Channels 0..115: original v7 116 channels from compute_top50_features
- Channels 116..135: 20 new "v8" per-sym scalar features (1 channel per feature × 5 syms)

Sym-agnostic: all formulas apply identical operations across sym axis.
Date-agnostic: no use of `date` field.
Vectorized NumPy throughout.

Sources: research_v8_features/TOP_30_NEW_RECOMMENDATIONS.json (Top 30 P1 high-gain features
from 56-paper literature survey). Selected 20 that are TRACTABLE given L8 z-scored inputs.
"""
from __future__ import annotations
import numpy as np

from interaction_features import (
    compute_top50_features, _sup, _vec, _NAME2IDX, _init_names,
    csec_demean, csec_demean_vec, _UNORD_PAIRS,
)

S = 5
EPS = 1e-6


# ---------- Cross-sym IQR z-score helper ----------------------------------

def csec_iqr_zscore_vec(b: np.ndarray, eps: float = EPS) -> np.ndarray:
    """(N, 5): (b - median) / IQR for cross-sym (per row)."""
    med = np.median(b, axis=1, keepdims=True)
    q25 = np.quantile(b, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(b, 0.75, axis=1, keepdims=True)
    iqr = q75 - q25
    return (b - med) / np.maximum(iqr, eps)


def csec_signed_rank_vec(b: np.ndarray) -> np.ndarray:
    """(N, 5): cross-sym signed rank in [-1, 1]."""
    order = np.argsort(b, axis=1)
    ranks = np.empty_like(b, dtype=np.float32)
    np.put_along_axis(ranks, order, np.arange(S, dtype=np.float32).reshape(1, S), axis=1)
    r = ranks / float(S - 1)
    return ((r - 0.5) * 2.0).astype(np.float32)


def _sum_mlofi_W5_lvl15(X):
    return sum(_vec(X, _sup(f'mlofi_W5_lvl{k}')) for k in range(1, 6))


def _sum_mlofi_W20_lvl15(X):
    return sum(_vec(X, _sup(f'mlofi_W20_lvl{k}')) for k in range(1, 6))


# ---------- 20 new feature generators -------------------------------------
# Each returns ((N, 5) per-sym values, list of 5 names).

def gen_NF067(X):
    """OFI × RV (W5 × W20)."""
    b = _vec(X, _sup('mlofi_W5_lvl1')) * _vec(X, _sup('rv_w20'))
    return b, [f'NF067_ofiW5_x_rvW20_s{i}' for i in range(S)]

def gen_NF068(X):
    """Imb × RV."""
    b = _vec(X, _sup('imbalance')) * _vec(X, _sup('rv_w20'))
    return b, [f'NF068_imb_x_rvW20_s{i}' for i in range(S)]

def gen_NF066(X):
    """OFI × Spread."""
    b = _vec(X, _sup('mlofi_W5_lvl1')) * _vec(X, _sup('spread1'))
    return b, [f'NF066_ofiW5_x_spread_s{i}' for i in range(S)]

def gen_NF113(X):
    """(microprice - mid) × OFI: combined informed-trade signal."""
    excess = _vec(X, _sup('wmp_lvl1')) - _vec(X, _sup('midprice1'))
    b = excess * _vec(X, _sup('mlofi_W5_lvl1'))
    return b, [f'NF113_microExcess_x_ofi_s{i}' for i in range(S)]

def gen_NF083(X):
    """Concave self-impact: sign(OFI) * sqrt(|OFI|)."""
    o = _vec(X, _sup('mlofi_W5_lvl1'))
    b = np.sign(o) * np.sqrt(np.abs(o))
    return b.astype(np.float32), [f'NF083_concaveOFI_s{i}' for i in range(S)]

def gen_NF040(X):
    """HAR vol composite: mean of rv_w5, rv_w20, rv_w50."""
    b = (_vec(X, _sup('rv_w5')) + _vec(X, _sup('rv_w20')) + _vec(X, _sup('rv_w50'))) / 3.0
    return b, [f'NF040_harComposite_s{i}' for i in range(S)]

def gen_NF019(X):
    """Microprice residual: wmp_lvl3 - wmp_lvl1 (deep book correction)."""
    b = _vec(X, _sup('wmp_lvl3')) - _vec(X, _sup('wmp_lvl1'))
    return b, [f'NF019_wmpResidual_s{i}' for i in range(S)]

def gen_NF116(X):
    """Deep microprice cross-sym demean: wmp_lvl3 - mean_cross_sym."""
    b = _vec(X, _sup('wmp_lvl3'))
    return csec_demean_vec(b), [f'NF116_demean_wmp3_s{i}' for i in range(S)]

def gen_NF022(X):
    """Integrated OFI W5: sum(mlofi_W5_lvl1..5) (uniform weights as PCA proxy)."""
    b = _sum_mlofi_W5_lvl15(X)
    return b, [f'NF022_intOFI_W5_s{i}' for i in range(S)]

def gen_NF094(X):
    """Integrated OFI W20: sum(mlofi_W20_lvl1..5)."""
    b = _sum_mlofi_W20_lvl15(X)
    return b, [f'NF094_intOFI_W20_s{i}' for i in range(S)]

def gen_NF027(X):
    """OFI idiosyncratic: ofi - cross-sym mean(ofi)."""
    b = _vec(X, _sup('mlofi_W5_lvl1'))
    return csec_demean_vec(b), [f'NF027_idioOFI_s{i}' for i in range(S)]

def gen_NF026(X):
    """OFI factor ratio: ofi / mean(|ofi|) cross-sym."""
    b = _vec(X, _sup('mlofi_W5_lvl1'))
    denom = np.mean(np.abs(b), axis=1, keepdims=True)
    return (b / (denom + EPS)).astype(np.float32), [f'NF026_ofiFactorRatio_s{i}' for i in range(S)]

def gen_NF059(X):
    """Cross-sym IQR-normalized signed_rv_W20."""
    b = _vec(X, _sup('signed_rv_W20'))
    return csec_iqr_zscore_vec(b), [f'NF059_iqrRet_s{i}' for i in range(S)]

def gen_NF060(X):
    """Cross-sym IQR-normalized OFI W5."""
    b = _vec(X, _sup('mlofi_W5_lvl1'))
    return csec_iqr_zscore_vec(b), [f'NF060_iqrOFI_s{i}' for i in range(S)]

def gen_NF052(X):
    """Time-series-like OFI normalization: use cross-sym signed rank (TS-norm proxy)."""
    b = _vec(X, _sup('mlofi_W5_lvl1'))
    return csec_signed_rank_vec(b), [f'NF052_signedRankOFI_s{i}' for i in range(S)]

def gen_NF103(X):
    """signed_bv_W20 cross-sym demean (signed book volume, market-neutral)."""
    b = _vec(X, _sup('signed_bv_W20'))
    return csec_demean_vec(b), [f'NF103_demean_sbvW20_s{i}' for i in range(S)]

def gen_NF055(X):
    """signed_bv_W20 × amount_delta (dollar-weighted signed imbalance proxy)."""
    b = _vec(X, _sup('signed_bv_W20')) * _vec(X, _sup('amount_delta'))
    return b, [f'NF055_sbv_x_amt_s{i}' for i in range(S)]

def gen_NF071(X):
    """Vol-gated OFI: ofi * tanh(-rv_w20). High-vol regimes dampen OFI signal."""
    o = _vec(X, _sup('mlofi_W5_lvl1'))
    rv = _vec(X, _sup('rv_w20'))
    b = o * np.tanh(-rv)
    return b.astype(np.float32), [f'NF071_volGatedOFI_s{i}' for i in range(S)]

def gen_NF067b(X):
    """OFI × Kyle_inv: OFI scaled by impact (illiquidity)."""
    b = _vec(X, _sup('mlofi_W5_lvl1')) * _vec(X, _sup('kyle_inv_W50'))
    return b, [f'NF067b_ofi_x_kyleInv_s{i}' for i in range(S)]

def gen_NF076(X):
    """Asymmetric-window delta cross-return:
    signed_rv_W20[i] - cross-sym mean(signed_rv_W50) (delta vs slow market signal)."""
    b_fast = _vec(X, _sup('signed_rv_W20'))
    b_slow = _vec(X, _sup('signed_rv_W50'))
    mkt_slow = b_slow.mean(axis=1, keepdims=True)
    b = b_fast - mkt_slow
    return b.astype(np.float32), [f'NF076_deltaCross_s{i}' for i in range(S)]


NEW_FEAT_ORDER = [
    'NF067', 'NF068', 'NF066', 'NF113', 'NF083',
    'NF040', 'NF019', 'NF116', 'NF022', 'NF094',
    'NF027', 'NF026', 'NF059', 'NF060', 'NF052',
    'NF103', 'NF055', 'NF071', 'NF067b', 'NF076',
]

_NEW_GEN = {
    'NF067': gen_NF067, 'NF068': gen_NF068, 'NF066': gen_NF066, 'NF113': gen_NF113,
    'NF083': gen_NF083, 'NF040': gen_NF040, 'NF019': gen_NF019, 'NF116': gen_NF116,
    'NF022': gen_NF022, 'NF094': gen_NF094, 'NF027': gen_NF027, 'NF026': gen_NF026,
    'NF059': gen_NF059, 'NF060': gen_NF060, 'NF052': gen_NF052, 'NF103': gen_NF103,
    'NF055': gen_NF055, 'NF071': gen_NF071, 'NF067b': gen_NF067b, 'NF076': gen_NF076,
}


def compute_new_features_v8(X_grp: np.ndarray, verbose: bool = False):
    """X_grp: (N, 5, 359) z-scored L8 features.

    Returns:
      X_new: (N, 5, 20) float32 — 20 per-sym scalar channels.
      feat_names: list[str], len 100 (5 syms × 20 features) — but we return per-channel name.
    """
    if X_grp.ndim != 3 or X_grp.shape[1] != S:
        raise ValueError(f'Expected X_grp shape (N, 5, F), got {X_grp.shape}')
    if X_grp.dtype != np.float32:
        X_grp = X_grp.astype(np.float32, copy=False)
    X_grp = np.nan_to_num(X_grp, nan=0.0, posinf=0.0, neginf=0.0)

    _init_names()
    chunks = []
    names = []
    for fid in NEW_FEAT_ORDER:
        gen = _NEW_GEN[fid]
        arr, n = gen(X_grp)
        if arr.shape != (X_grp.shape[0], S):
            raise ValueError(f'{fid}: expected (N, 5) got {arr.shape}')
        chunks.append(arr.astype(np.float32, copy=False)[:, :, None])  # (N, 5, 1)
        names.append(fid)
        if verbose:
            print(f'  {fid}: {arr.shape}  min={arr.min():.3f} max={arr.max():.3f} std={arr.std():.3f}')
    X_new = np.concatenate(chunks, axis=2)  # (N, 5, 20)
    X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
    return X_new, names


def compute_v9_features(X_grp: np.ndarray, verbose: bool = False):
    """Combined feature computation: 116 v7 channels + 20 new v8 channels.

    Returns:
      X_combined: (N, 5, 136) float32
      old_names:  list[str] len 116
      new_names:  list[str] len 20
    """
    X_old, names_old = compute_top50_features(X_grp, verbose=verbose)
    X_new, names_new = compute_new_features_v8(X_grp, verbose=verbose)
    X_combined = np.concatenate([X_old, X_new], axis=2).astype(np.float32, copy=False)
    return X_combined, names_old, names_new


# ---------- Smoke test -----------------------------------------------------

if __name__ == '__main__':
    np.random.seed(0)
    N = 8
    X = np.random.randn(N, 5, 359).astype(np.float32)
    X_combined, names_old, names_new = compute_v9_features(X, verbose=True)
    print(f'\nX_combined shape={X_combined.shape}  n_old={len(names_old)}  n_new={len(names_new)}')
    print(f'finite: {np.isfinite(X_combined).all()}')
    print(f'min={X_combined.min():.3f} max={X_combined.max():.3f}  '
          f'mean={X_combined.mean():.3f} std={X_combined.std():.3f}')
