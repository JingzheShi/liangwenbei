"""Verify that (date, sess_idx, t) groups have exactly 5 syms in the data cache."""
import numpy as np
from pathlib import Path

CACHE = Path('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p')

for split in ['train', 'val', 'test']:
    d = np.load(CACHE / f'schemeP_{split}.npz')
    date = d['date'].astype(np.int32)
    sess = d['sess_idx'].astype(np.int32)
    t = d['t'].astype(np.int32)
    sym = d['sym'].astype(np.int32)
    n_total = len(sym)
    d.close()

    # Sort by (date, sess_idx, t, sym)
    order = np.lexsort((sym, t, sess, date))
    date_s = date[order]
    sess_s = sess[order]
    t_s = t[order]
    sym_s = sym[order]

    # Find group boundaries
    keys = np.stack([date_s, sess_s, t_s], axis=1)
    change = np.any(keys[1:] != keys[:-1], axis=1)
    boundaries = np.concatenate([[0], np.where(change)[0] + 1, [n_total]])
    group_sizes = np.diff(boundaries)

    n_groups = len(group_sizes)
    n_complete = int((group_sizes == 5).sum())
    complete_mask = group_sizes == 5
    complete_starts = boundaries[:-1][complete_mask]

    # Verify sym ordering within complete groups
    row_idx = (complete_starts[:, None] + np.arange(5)[None, :]).flatten()
    sym_groups = sym_s[row_idx].reshape(-1, 5)
    sym_order_ok = (sym_groups == np.arange(5)[None, :]).all()

    print(f'=== {split} ===')
    print(f'  n_rows={n_total:,}, n_groups={n_groups:,}')
    print(f'  complete groups (size=5): {n_complete:,} ({100*n_complete/n_groups:.3f}%)')
    print(f'  rows in complete groups: {n_complete*5:,} ({100*n_complete*5/n_total:.3f}%)')
    uvals, ucounts = np.unique(group_sizes, return_counts=True)
    print(f'  group size distribution: {dict(zip(uvals.tolist(), ucounts.tolist()))}')
    print(f'  sym order within groups always [0,1,2,3,4]: {sym_order_ok}')
    print()
