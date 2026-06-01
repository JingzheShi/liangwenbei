"""Filter Top50 interaction channels by IC + PSI + collinearity.

Thresholds (final after tuning if needed):
  |ic_global| > 0.005        (Spearman IC: weak but non-zero signal)
  max |ic_sym_k| > 0.01      (at least one sym should show useful per-sym signal)
  psi_sym_max < 1.0          (cross-sym distribution drift allowed up to PSI=1)
  psi_train_val < 0.3        (train/val drift small)
  variance > 1e-4            (non-degenerate)
  collinear dedup: keep one per |r|>0.95 cluster, prefer lower psi_sym_max.

Writes:
  selected_features.npy      (K,) int64 channel indices kept
  selected_features.json     metadata
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METRICS_CSV = HERE / 'feature_metrics.csv'

# Thresholds — start permissive then tighten if too many features survive.
TH = dict(
    ic_global_abs=0.005,
    ic_sym_max_abs=0.01,
    psi_sym_max=1.0,
    psi_train_val=0.3,
    variance=1e-4,
    collinear_r=0.95,
)


def select():
    df = pd.read_csv(METRICS_CSV)
    n0 = len(df)
    print(f'Loaded {n0} channels')

    df['ic_sym_max_abs'] = df[[f'ic_sym{i}' for i in range(5)]].abs().max(axis=1)
    df['ic_global_abs']  = df['ic_global'].abs()
    df['pearson_r_abs']  = df['pearson_r'].abs()

    # Apply each filter and print survival
    masks = {
        'ic_global':    df['ic_global_abs']    > TH['ic_global_abs'],
        'ic_sym_max':   df['ic_sym_max_abs']   > TH['ic_sym_max_abs'],
        'psi_sym':      df['psi_sym_max']      < TH['psi_sym_max'],
        'psi_train_val':df['psi_train_val']    < TH['psi_train_val'],
        'variance':     df['variance']         > TH['variance'],
    }
    print('Filter survival:')
    for name, m in masks.items():
        print(f'  {name:14s} {m.sum():4d}/{n0}')

    overall = masks['ic_global'] & masks['ic_sym_max'] & masks['psi_sym'] & masks['psi_train_val'] & masks['variance']
    keep = df[overall].copy()
    print(f'\nAfter all filters: {len(keep)}/{n0}')

    # Sort by ic_global_abs descending; we'll keep the top representative of each collinearity cluster
    keep = keep.sort_values('ic_global_abs', ascending=False).reset_index(drop=True)

    # Collinearity dedup: need correlation matrix.
    # Recompute corr ON THE KEPT channels using stored data (re-import).
    # Compute corr only over kept channels using a sub-sample.
    print('\nComputing channel-channel corr on kept channels ...')
    from data_loader import load_grouped_data
    from interaction_features import compute_top50_features

    train_data, _, _, _ = load_grouped_data(level='L8', verbose=False)
    n_tr_aug = train_data['X_grp'].shape[0]
    n_tr = n_tr_aug // 2
    X_tr = train_data['X_grp'][:n_tr]
    X_inter, _ = compute_top50_features(X_tr)        # (N, 5, 116)
    flat = X_inter.reshape(-1, X_inter.shape[-1])
    rng = np.random.default_rng(0)
    sub = rng.choice(flat.shape[0], min(200_000, flat.shape[0]), replace=False)
    flat_sub = flat[sub]
    flat_sub_z = (flat_sub - flat_sub.mean(axis=0)) / np.maximum(flat_sub.std(axis=0), 1e-9)
    corr = np.corrcoef(flat_sub_z.T)
    np.fill_diagonal(corr, 0.0)

    # Dedup
    keep_ids = list(keep['channel_id'].astype(int))   # sorted by |ic_global| desc
    dropped = set()
    keep_after_dedup = []
    for cid in keep_ids:
        if cid in dropped:
            continue
        keep_after_dedup.append(cid)
        # Drop any other channel that's highly correlated with this one
        peers = np.where(np.abs(corr[cid]) > TH['collinear_r'])[0]
        for p in peers:
            if p == cid:
                continue
            # Only drop if p is also in the kept-by-filter set (else it's already discarded)
            if p in keep_ids:
                dropped.add(int(p))

    keep_after_dedup = sorted(keep_after_dedup)
    print(f'After collinearity dedup (|r|>{TH["collinear_r"]}): {len(keep_after_dedup)}')

    # If too few, relax and try again — keep simple: report and proceed regardless
    if len(keep_after_dedup) < 10:
        print('WARNING: too few features after filter (<10). Loosening thresholds...')
        # Loosen and re-run
        for k in ['ic_global_abs', 'ic_sym_max_abs']:
            TH[k] *= 0.5
        return select()

    sel_arr = np.array(keep_after_dedup, dtype=np.int64)
    np.save(HERE / 'selected_features.npy', sel_arr)

    out = {
        'n_total_channels': int(n0),
        'n_selected': int(len(sel_arr)),
        'thresholds': TH,
        'selected_channel_ids': sel_arr.tolist(),
        'selected_names': df.set_index('channel_id').loc[sel_arr, 'name'].tolist(),
        'top20_by_ic_global_abs': df.sort_values('ic_global_abs', ascending=False).head(20)[
            ['channel_id', 'name', 'ic_global', 'ic_sym_max_abs', 'psi_sym_max', 'psi_train_val',
             'variance', 'n_collinear_above_0.95']
        ].to_dict(orient='records'),
    }
    (HERE / 'selected_features.json').write_text(json.dumps(out, indent=2))
    print(f'\nWrote selected_features.npy ({len(sel_arr)} channels)')
    print(f'Wrote selected_features.json')

    # Print sample of selected
    sel_df = df.set_index('channel_id').loc[sel_arr].sort_values('ic_global_abs', ascending=False)
    print('\nTop 10 selected (by |ic_global|):')
    print(sel_df[['name', 'ic_global', 'ic_sym_max_abs', 'psi_sym_max', 'psi_train_val', 'variance']].head(10).to_string())


if __name__ == '__main__':
    select()
