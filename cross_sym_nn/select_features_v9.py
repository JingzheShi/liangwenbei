"""Compute z-stats over 136 combined channels (116 v7 + 20 v8 new),
save selected_features_v9_60.npy = [40 v7 selected + 20 new] + inter_zstats_v9.npz.

This re-uses the existing 40 channels selected in selected_features.npy and
appends the 20 new NF* channels (no filtering — research-prioritized).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_loader import load_grouped_data
from interaction_features_v8 import compute_v9_features


def main():
    t0 = time.time()
    print('Loading data ...', flush=True)
    train_data, val_data, _, _ = load_grouped_data(level='L8', verbose=True)
    n_tr_aug = train_data['X_grp'].shape[0]
    n_tr = n_tr_aug // 2
    X_tr_359 = train_data['X_grp'][:n_tr]
    y_tr     = train_data['y_reg'][:n_tr]
    X_va_359 = val_data['X_grp']
    print(f'  X_tr_359={X_tr_359.shape}  y_tr={y_tr.shape}  X_va_359={X_va_359.shape}', flush=True)

    print('Computing v9 features (train, 116+20=136 channels) ...', flush=True)
    t1 = time.time()
    X_tr_inter, names_old, names_new = compute_v9_features(X_tr_359)
    print(f'  X_tr_inter={X_tr_inter.shape}  dt={time.time()-t1:.1f}s', flush=True)
    print('Computing v9 features (val) ...', flush=True)
    t1 = time.time()
    X_va_inter, _, _ = compute_v9_features(X_va_359)
    print(f'  X_va_inter={X_va_inter.shape}  dt={time.time()-t1:.1f}s', flush=True)

    n_channels = X_tr_inter.shape[-1]
    assert n_channels == 136, f'Expected 136 channels, got {n_channels}'

    # Per-channel z-norm params over train
    flat_tr = X_tr_inter.reshape(-1, n_channels)
    mu = flat_tr.mean(axis=0).astype(np.float32)
    sd = np.maximum(flat_tr.std(axis=0), 1e-6).astype(np.float32)
    print(f'  mu range [{mu.min():.3f}, {mu.max():.3f}]  sd range [{sd.min():.3f}, {sd.max():.3f}]', flush=True)

    # Save zstats
    names_combined = np.array(['v7_ch' + str(i) for i in range(116)] + names_new, dtype=object)
    np.savez(HERE / 'inter_zstats_v9.npz', mu=mu, sd=sd, names=names_combined)
    print(f'Wrote inter_zstats_v9.npz (n={n_channels})', flush=True)

    # Build 60-channel selection: 40 v7 selected (channels 0..115) + 20 new (channels 116..135)
    sel_old = json.load(open(HERE / 'selected_features.json'))['selected_channel_ids']
    assert len(sel_old) == 40
    sel_old = np.array(sel_old, dtype=np.int64)
    sel_new = np.arange(116, 116 + 20, dtype=np.int64)
    sel_v9 = np.concatenate([sel_old, sel_new])
    assert len(sel_v9) == 60
    np.save(HERE / 'selected_features_v9_60.npy', sel_v9)
    print(f'Wrote selected_features_v9_60.npy ({len(sel_v9)} indices)', flush=True)

    # Quick IC analysis on the 20 new channels (Spearman correlation with y)
    print('\n=== IC analysis for 20 new NF channels ===', flush=True)
    import pandas as pd

    def spearman_safe(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 100:
            return 0.0
        rx = pd.Series(x[m]).rank(method='average').to_numpy()
        ry = pd.Series(y[m]).rank(method='average').to_numpy()
        rx = rx - rx.mean(); ry = ry - ry.mean()
        den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / den) if den >= 1e-12 else 0.0

    y_flat = y_tr.flatten()
    ic_global_list = []
    psi_train_val_list = []
    for k in range(20):
        ch_idx = 116 + k
        ch_tr = X_tr_inter[:, :, ch_idx].flatten()
        ch_va = X_va_inter[:, :, ch_idx].flatten()
        ic = spearman_safe(ch_tr, y_flat)
        # PSI quick
        combined = np.concatenate([ch_tr, ch_va])
        qs = np.quantile(combined, np.linspace(0, 1, 51))
        qs[0] -= 1e-9; qs[-1] += 1e-9
        a_h, _ = np.histogram(ch_tr, bins=qs)
        b_h, _ = np.histogram(ch_va, bins=qs)
        a_p = a_h / max(a_h.sum(), 1); b_p = b_h / max(b_h.sum(), 1)
        a_p = np.where(a_p == 0, 1e-6, a_p); b_p = np.where(b_p == 0, 1e-6, b_p)
        psi = float(np.sum((b_p - a_p) * np.log(b_p / a_p)))
        ic_global_list.append(ic)
        psi_train_val_list.append(psi)
        print(f'  NF[{k:2d}] {names_new[k]:18s} ic={ic:+.5f}  psi_tv={psi:.4f}', flush=True)

    ic_arr = np.array(ic_global_list); psi_arr = np.array(psi_train_val_list)
    print(f'\nNF channels: |ic|>0.005: {(np.abs(ic_arr) > 0.005).sum()}/20  |ic|>0.01: {(np.abs(ic_arr) > 0.01).sum()}/20',
          flush=True)
    print(f'  psi_train_val < 0.3: {(psi_arr < 0.3).sum()}/20', flush=True)
    print(f'  mean |ic| = {np.abs(ic_arr).mean():.5f}  max |ic| = {np.abs(ic_arr).max():.5f}', flush=True)

    out_json = {
        'n_total_channels': int(n_channels),
        'n_selected': 60,
        'sel_v9_indices': sel_v9.tolist(),
        'n_old_kept': 40, 'n_new_added': 20,
        'new_names': names_new,
        'new_ic_global': ic_global_list,
        'new_psi_train_val': psi_train_val_list,
    }
    (HERE / 'selected_features_v9_60.json').write_text(json.dumps(out_json, indent=2))
    print(f'Wrote selected_features_v9_60.json  total_dt={time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
