"""Build selected_features_v9_80.npy = 60 v7 channels + 20 NF channels (80 total).

Strategy:
- Reuse existing 40 v7 channels from selected_features.json.
- Add 20 more v7 channels from feature_metrics.csv (looser PSI 0.6, top-|IC|, deduped).
- Append 20 NF channels (indices 116..135) — same as v9_60.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

OUT_NPY = HERE / 'selected_features_v9_80.npy'
OUT_JSON = HERE / 'selected_features_v9_80.json'


def main():
    df = pd.read_csv(HERE / 'feature_metrics.csv')
    sel_old = json.load(open(HERE / 'selected_features.json'))['selected_channel_ids']
    sel_old = list(sel_old)
    assert len(sel_old) == 40

    df = df.copy()
    df['ic_abs'] = df['ic_global'].abs()
    df_extra = df[~df['channel_id'].isin(sel_old)].copy()
    df_extra = df_extra[(df_extra['psi_train_val'] < 0.6) & (df_extra['variance'] > 1e-4)]
    df_extra['ic_key'] = df_extra['ic_global'].round(6)
    df_extra = df_extra.sort_values('ic_abs', ascending=False).drop_duplicates('ic_key', keep='first')
    extra_20 = df_extra.head(20)['channel_id'].astype(int).tolist()
    assert len(extra_20) == 20, f'Got {len(extra_20)} extra channels'

    sel_v7_60 = sorted(sel_old + extra_20)
    sel_nf_20 = list(range(116, 136))
    sel_80 = np.array(sel_v7_60 + sel_nf_20, dtype=np.int64)
    assert len(sel_80) == 80

    np.save(OUT_NPY, sel_80)
    OUT_JSON.write_text(json.dumps({
        'n_total': int(len(sel_80)),
        'n_v7_orig': 40,
        'n_v7_extra': 20,
        'n_nf': 20,
        'sel_80_indices': sel_80.tolist(),
        'extra_20_v7_channels': sorted(extra_20),
        'extra_20_v7_names': [df.loc[df['channel_id'] == c, 'name'].values[0] for c in sorted(extra_20)],
    }, indent=2))
    print(f'Wrote {OUT_NPY} ({len(sel_80)} indices)')
    print(f'  v7 60 indices: {sel_v7_60}')
    print(f'  NF 20 indices: {sel_nf_20}')


if __name__ == '__main__':
    main()
