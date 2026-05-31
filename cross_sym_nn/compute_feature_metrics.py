"""Compute IC/PSI/variance/collinearity metrics for the 116 per-sym interaction channels.

Methodology:
- Loads grouped train data (preprocessed: log1p + window-z + mirror aug already applied).
- Use the original train half (skip mirror) for IC / PSI computation so y semantics is clean.
- For each per-sym channel k:
    ic_global  = Spearman ρ between channel[:, :, k].flatten and y_reg.flatten
    ic_sym{i}  = Spearman ρ between channel[:, i, k] and y_reg[:, i]   (i=0..4)
    pearson_r  = Pearson r between channel.flat and y.flat
    psi_sym_max = max over 10 unordered (i,j) sym pairs of PSI(channel[:,i,k], channel[:,j,k])
    psi_train_val = PSI between train channel and val channel
    variance   = global std (channel.flat)
    n_collinear_above_0.95 = number of OTHER channels with |Pearson r| > 0.95
- Writes feature_metrics.csv

PSI bin count = 200 (per Worker A sonnet v1 standard).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_loader import load_grouped_data
from interaction_features import compute_top50_features

PSI_BINS = 200
PSI_EPS = 1e-6


def psi(a: np.ndarray, b: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population stability index. a is reference, b is target.
    Uses fixed quantile bins computed on a∪b to be symmetric.
    """
    finite_a = a[np.isfinite(a)]
    finite_b = b[np.isfinite(b)]
    if len(finite_a) < bins or len(finite_b) < bins:
        return 0.0
    combined = np.concatenate([finite_a, finite_b])
    qs = np.quantile(combined, np.linspace(0, 1, bins + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    a_h, _ = np.histogram(finite_a, bins=qs)
    b_h, _ = np.histogram(finite_b, bins=qs)
    a_p = a_h / max(a_h.sum(), 1)
    b_p = b_h / max(b_h.sum(), 1)
    a_p = np.where(a_p == 0, PSI_EPS, a_p)
    b_p = np.where(b_p == 0, PSI_EPS, b_p)
    return float(np.sum((b_p - a_p) * np.log(b_p / a_p)))


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; ignores NaN pairs."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return 0.0
    x = x[m]; y = y[m]
    rx = pd.Series(x).rank(method='average').to_numpy()
    ry = pd.Series(y).rank(method='average').to_numpy()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den < 1e-12:
        return 0.0
    return float((rx * ry).sum() / den)


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return 0.0
    x = x[m]; y = y[m]
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    if den < 1e-12:
        return 0.0
    return float((x * y).sum() / den)


def main():
    t0 = time.time()
    print('Loading grouped data ...', flush=True)
    train_data, val_data, _, stats = load_grouped_data(level='L8', verbose=True)
    # Train is augmented (orig + mirror) → take first half (orig)
    n_tr_aug = train_data['X_grp'].shape[0]
    n_tr = n_tr_aug // 2
    print(f'  n_tr_aug={n_tr_aug}  → taking orig half n_tr={n_tr}', flush=True)

    X_tr_359 = train_data['X_grp'][:n_tr]                  # (N_tr, 5, 359)
    y_tr     = train_data['y_reg'][:n_tr]                  # (N_tr, 5)
    X_va_359 = val_data['X_grp']

    print(f'  X_tr_359={X_tr_359.shape}  y_tr={y_tr.shape}  X_va_359={X_va_359.shape}', flush=True)

    # Compute interaction features
    print('Computing interaction features (train) ...', flush=True)
    t1 = time.time()
    X_inter_tr, names = compute_top50_features(X_tr_359)
    print(f'  X_inter_tr={X_inter_tr.shape}  n_channels={X_inter_tr.shape[-1]}  '
          f'n_names={len(names)}  dt={time.time()-t1:.1f}s', flush=True)
    print('Computing interaction features (val) ...', flush=True)
    t1 = time.time()
    X_inter_va, _ = compute_top50_features(X_va_359)
    print(f'  X_inter_va={X_inter_va.shape}  dt={time.time()-t1:.1f}s', flush=True)

    n_channels = X_inter_tr.shape[-1]

    # Per-channel z-normalization params (global, computed on train)
    flat_tr = X_inter_tr.reshape(-1, n_channels)
    mu = flat_tr.mean(axis=0).astype(np.float32)
    sd = np.maximum(flat_tr.std(axis=0), 1e-6).astype(np.float32)

    # Build feature_metrics dataframe
    print('Computing IC / PSI / variance / collinearity ...', flush=True)
    y_flat = y_tr.flatten()

    records = []
    # For collinearity, compute Pearson r matrix across channels on a SUBSAMPLE for speed
    n_sub = min(100_000, flat_tr.shape[0])
    sub_idx = np.random.default_rng(0).choice(flat_tr.shape[0], n_sub, replace=False)
    flat_sub = flat_tr[sub_idx]
    flat_sub_z = (flat_sub - flat_sub.mean(axis=0, keepdims=True)) / np.maximum(
        flat_sub.std(axis=0, keepdims=True), 1e-9)
    # cov / corr
    print(f'  computing channel-channel corr on {n_sub} samples ...', flush=True)
    corr = np.corrcoef(flat_sub_z.T).astype(np.float32)
    np.fill_diagonal(corr, 0.0)
    n_collin_above = (np.abs(corr) > 0.95).sum(axis=1)   # (n_channels,)
    print(f'  corr shape={corr.shape}  n_above_0.95 max={n_collin_above.max()}', flush=True)

    # Per-channel metrics
    for k in range(n_channels):
        ch_tr = X_inter_tr[:, :, k]   # (N_tr, 5)
        ch_va = X_inter_va[:, :, k]   # (N_va, 5)
        ch_flat = ch_tr.flatten()

        # IC global
        ic_g = spearman_r(ch_flat, y_flat)
        pr   = pearson_r(ch_flat, y_flat)
        # IC per sym
        ic_syms = []
        for i in range(5):
            ic_syms.append(spearman_r(ch_tr[:, i], y_tr[:, i]))
        # PSI cross-sym (10 unordered pairs, max)
        psi_pairs = []
        for i in range(5):
            for j in range(i + 1, 5):
                psi_pairs.append(psi(ch_tr[:, i], ch_tr[:, j]))
        psi_sym_max = max(psi_pairs) if psi_pairs else 0.0
        # PSI train vs val (channel flatten)
        psi_tv = psi(ch_flat, ch_va.flatten())
        var = float(np.var(ch_flat))

        records.append({
            'channel_id': k,
            'name': names[k] if k < len(names) else f'channel_{k}',
            'ic_global': ic_g,
            'ic_sym0': ic_syms[0],
            'ic_sym1': ic_syms[1],
            'ic_sym2': ic_syms[2],
            'ic_sym3': ic_syms[3],
            'ic_sym4': ic_syms[4],
            'pearson_r': pr,
            'psi_sym_max': psi_sym_max,
            'psi_train_val': psi_tv,
            'variance': var,
            'mu': float(mu[k]),
            'sd': float(sd[k]),
            'n_collinear_above_0.95': int(n_collin_above[k]),
        })

        if (k + 1) % 20 == 0:
            print(f'  ch {k+1}/{n_channels}  ic_g={ic_g:+.4f}  psi_sym={psi_sym_max:.3f}  psi_tv={psi_tv:.3f}',
                  flush=True)

    df = pd.DataFrame.from_records(records)
    out_csv = HERE / 'feature_metrics.csv'
    df.to_csv(out_csv, index=False)
    print(f'\nWrote {out_csv}  ({len(df)} rows)  dt_total={time.time()-t0:.1f}s', flush=True)

    # Also save per-channel mu/sd (for runtime z-norm)
    np.savez(HERE / 'inter_zstats.npz', mu=mu, sd=sd, names=np.array(names, dtype=object))
    print(f'Wrote inter_zstats.npz  (mu/sd shape={mu.shape})', flush=True)

    # Quick summary
    print('\n=== Summary ===')
    print(f"  |ic_global| > 0.005 : {(df['ic_global'].abs() > 0.005).sum()}")
    print(f"  |ic_global| > 0.01  : {(df['ic_global'].abs() > 0.01).sum()}")
    print(f"  |ic_global| > 0.02  : {(df['ic_global'].abs() > 0.02).sum()}")
    print(f"  psi_sym_max < 0.5   : {(df['psi_sym_max'] < 0.5).sum()}")
    print(f"  psi_train_val < 0.3 : {(df['psi_train_val'] < 0.3).sum()}")
    print(f"  variance > 1e-4     : {(df['variance'] > 1e-4).sum()}")


if __name__ == '__main__':
    main()
