"""Build selected_features_v9_100.npy = 80 v7 channels + 20 NF channels (100 total).

Strategy:
- Start from existing 80 selection (selected_features_v9_80.json) → 60 v7 + 20 NF
- Extend v7 from 60 → 80 by pulling 20 more channels from feature_metrics.csv
  (PSI < 0.6, |ic_global| sorted, exclude already-selected, ic_global rounded dedup)
- NF stays at the existing 20 (indices 116..135)

Also copies inter_zstats_v9.npz to inter_zstats_v9_100.npz for trainer compatibility
(zstats are over all 136 raw channels and apply BEFORE sel_idx slicing — so
the file is byte-identical; the suffix simply matches the sweep dispatcher).
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

OUT_NPY = HERE / 'selected_features_v9_100.npy'
OUT_JSON = HERE / 'selected_features_v9_100.json'
ZSTATS_IN  = HERE / 'inter_zstats_v9.npz'
ZSTATS_OUT = HERE / 'inter_zstats_v9_100.npz'


def main():
    sel_80 = json.load(open(HERE / 'selected_features_v9_80.json'))
    sel_v7_60 = [c for c in sel_80['sel_80_indices'] if c < 116]
    sel_nf_20 = [c for c in sel_80['sel_80_indices'] if c >= 116]
    assert len(sel_v7_60) == 60 and len(sel_nf_20) == 20

    df = pd.read_csv(HERE / 'feature_metrics.csv')
    df = df.copy()
    df['ic_abs'] = df['ic_global'].abs()

    df_extra = df[~df['channel_id'].isin(sel_v7_60)].copy()
    df_extra = df_extra[(df_extra['channel_id'] < 116)]  # v7 channels only
    df_extra = df_extra[(df_extra['psi_train_val'] < 0.6) & (df_extra['variance'] > 1e-4)]
    df_extra['ic_key'] = df_extra['ic_global'].round(6)
    df_extra = df_extra.sort_values('ic_abs', ascending=False).drop_duplicates('ic_key', keep='first')
    extra_20 = df_extra.head(20)['channel_id'].astype(int).tolist()
    assert len(extra_20) == 20, f'Got {len(extra_20)} extra channels'

    sel_v7_80 = sorted(sel_v7_60 + extra_20)
    sel_100 = np.array(sel_v7_80 + sel_nf_20, dtype=np.int64)
    assert len(sel_100) == 100
    assert len(set(sel_100.tolist())) == 100, 'duplicates'

    np.save(OUT_NPY, sel_100)
    OUT_JSON.write_text(json.dumps({
        'n_total': int(len(sel_100)),
        'n_v7_orig_60': 60,
        'n_v7_extra_20': 20,
        'n_v7_total': 80,
        'n_nf': 20,
        'sel_100_indices': sel_100.tolist(),
        'extra_20_v7_channels': sorted(extra_20),
        'extra_20_v7_names': [df.loc[df['channel_id'] == c, 'name'].values[0] for c in sorted(extra_20)],
    }, indent=2))
    print(f'Wrote {OUT_NPY} ({len(sel_100)} indices)')
    print(f'  v7 80 indices ({len(sel_v7_80)}): {sel_v7_80}')
    print(f'  NF 20 indices ({len(sel_nf_20)}): {sel_nf_20}')

    # Copy zstats (same 136-channel z-scoring, slicing happens after)
    shutil.copy(ZSTATS_IN, ZSTATS_OUT)
    print(f'Copied {ZSTATS_IN.name} → {ZSTATS_OUT.name}')


if __name__ == '__main__':
    main()
