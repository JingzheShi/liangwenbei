"""Build selected_features_v15_80.npy and inter_zstats_v15.npz.

Strategy:
- 60 indices = same as selected_features_v9_60.npy (proven base set 0..135)
- 20 indices = new temporal-agg channels (indices 136..155 in v15 156-dim space)
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data_loader import load_grouped_data
from interaction_features_v15 import compute_v15_features


def main():
    sel_v9_60 = np.load(HERE / 'selected_features_v9_60.npy').astype(np.int64)
    assert sel_v9_60.shape == (60,)
    sel_temp_20 = np.arange(136, 156, dtype=np.int64)  # new temporal channels
    sel_80 = np.concatenate([sel_v9_60, sel_temp_20])
    assert sel_80.shape == (80,)
    assert sel_80.max() == 155 and sel_80.min() == sel_v9_60.min()

    np.save(HERE / 'selected_features_v15_80.npy', sel_80)
    (HERE / 'selected_features_v15_80.json').write_text(json.dumps({
        'n_total': 80,
        'n_v9_60_indices': sel_v9_60.tolist(),
        'n_temporal_20_indices': sel_temp_20.tolist(),
    }, indent=2))
    print(f'Wrote selected_features_v15_80.npy ({len(sel_80)} indices)')

    # Now compute z-stats over train set for the full 156-dim space
    print('\nLoading train data for zstats ...', flush=True)
    train_data, _, _, _ = load_grouped_data(level='L8', verbose=True)
    X_tr = train_data['X_grp']  # may include mirror aug
    # discard mirror half
    N_full = X_tr.shape[0]
    N_orig = N_full // 2
    X_tr = X_tr[:N_orig]
    print(f'  using {N_orig} train groups', flush=True)

    t0 = time.time()
    # Compute v15 features in chunks
    chunks = []
    batch = 200_000
    for s in range(0, X_tr.shape[0], batch):
        Xs = X_tr[s:s + batch]
        Xc, _, _, _ = compute_v15_features(Xs)
        chunks.append(Xc)
    X_v15 = np.concatenate(chunks, axis=0)  # (N, 5, 156)
    print(f'  v15 features computed: shape={X_v15.shape}  dt={time.time()-t0:.1f}s', flush=True)

    # Robust z-stats per channel
    X_flat = X_v15.reshape(-1, X_v15.shape[-1])  # (N*5, 156)
    mu = X_flat.mean(axis=0).astype(np.float32)
    sd = X_flat.std(axis=0).astype(np.float32)
    sd = np.maximum(sd, 1e-6)
    names = [f'ch_{i}' for i in range(X_v15.shape[-1])]

    np.savez_compressed(HERE / 'inter_zstats_v15.npz',
                        mu=mu, sd=sd, names=np.array(names))
    print(f'Wrote inter_zstats_v15.npz  mu[:5]={mu[:5]}  sd[:5]={sd[:5]}')
    print(f'  new temporal sd[136:156]={sd[136:156]}')


if __name__ == '__main__':
    main()
