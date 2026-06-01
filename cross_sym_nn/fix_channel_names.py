"""Rebuild per-channel names from TOP50 spec → 116 channel names."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(HERE))
from interaction_features import TOP50_ORDER

# n_channels_per_sym for each feature type:
#   ord_pair        -> 4 (4 partners)
#   unord_pair      -> 4 (4 signed partners)
#   csec_5          -> 1
#   attn_5          -> 1
#   broadcast_1     -> 1
FEATURE_CHANNEL_TYPES = {
    'F003': ('ord_pair',    'mp_t'),
    'F001': ('ord_pair',    'mid'),
    'F004': ('ord_pair',    'ofiW5'),
    'F009': ('ord_pair',    'imb1'),
    'F010': ('ord_pair',    'logretW5'),
    'F002': ('ord_pair',    'wmp'),
    'F015': ('ord_pair',    'microprice'),
    'F040': ('csec',        'zsc_imb1'),
    'F034': ('csec',        'zsc_ofi'),
    'F078': ('csec',        'demean_ofiW5'),
    'F079': ('csec',        'demean_ofiW20'),
    'F031': ('csec',        'demean_mid'),
    'F019': ('ord_pair',    'mlofiW5L3'),
    'F102': ('unord_pair',  'sumOFI1-5'),
    'F021': ('ord_pair',    'wap5'),
    'F157': ('unord_pair',  'imb1xOFI'),
    'F175': ('unord_pair',  'OFIximb1'),
    'F110': ('unord_pair',  'imb1xdepth'),
    'F195': ('unord_pair',  'ofiNormDepth'),
    'F179': ('unord_pair',  'microMinusMid'),
    'F193': ('csec',        'demean_micro_v2'),
    'F123': ('csec',        'rank_ofiW5'),
    'F124': ('csec',        'rank_imb1'),
    'F049': ('csec',        'rank_wmp'),
    'F047': ('csec',        'rank_logretW20'),
    'F062': ('csec',        'demean_logretW5'),
    'F145': ('csec',        'demean_logretW5_v2'),
    'F202': ('csec',        'tsnorm_imb1'),
    'F201': ('csec',        'tsnorm_ofi'),
    'F060': ('csec',        'share_topsize'),
    'F085': ('csec',        'demean_micro_v3'),
    'F108': ('unord_pair',  'midDiffXImbSum'),
    'F109': ('unord_pair',  'ofiDiffXSpreadSum'),
    'F063': ('csec',        'demean_instret'),
    'F147': ('csec',        'demean_microExcess'),
    'F168': ('broadcast',   'mkt_ofi'),
    'F166': ('broadcast',   'mkt_ret'),
    'F011': ('ord_pair',    'logretW20'),
    'F177': ('unord_pair',  'rvRatioShortLong'),
    'F067': ('unord_pair',  'ofiW20'),
    'F008': ('ord_pair',    'spread'),
    'F126': ('csec',        'rank_RV'),
    'F046': ('csec',        'rank_kyleinv'),
    'F165': ('csec',        'signedRank_logretW20'),
    'F139': ('attn',        'mid_tau1'),
    'F205': ('attn',        'mid_tau2'),
    'F056': ('broadcast',   'mkt_imb1'),
    'F081': ('csec',        'demean_RV'),
    'F104': ('unord_pair',  'mptMinusOld'),
    'F191': ('csec',        'demean_imbDepth'),
}


def build_channel_names():
    names = []
    feature_for_channel = []
    for fid in TOP50_ORDER:
        ftype, alias = FEATURE_CHANNEL_TYPES[fid]
        if ftype == 'ord_pair':
            for k in range(4):
                names.append(f'{fid}_{alias}_partner{k}')
                feature_for_channel.append(fid)
        elif ftype == 'unord_pair':
            for k in range(4):
                names.append(f'{fid}_{alias}_partner{k}_signed')
                feature_for_channel.append(fid)
        elif ftype in ('csec', 'attn'):
            names.append(f'{fid}_{alias}')
            feature_for_channel.append(fid)
        elif ftype == 'broadcast':
            names.append(f'{fid}_{alias}_bcast')
            feature_for_channel.append(fid)
        else:
            raise ValueError(ftype)
    return names, feature_for_channel


if __name__ == '__main__':
    names, fids = build_channel_names()
    print(f'Total channels: {len(names)}  (expected 116)')

    # Patch CSV
    df = pd.read_csv(HERE / 'feature_metrics.csv')
    df = df.sort_values('channel_id').reset_index(drop=True)
    assert len(df) == len(names), f'mismatch {len(df)} vs {len(names)}'
    df['name'] = names
    df['feature_id'] = fids
    df.to_csv(HERE / 'feature_metrics.csv', index=False)
    print('Patched feature_metrics.csv')

    # Save names for future use
    (HERE / 'channel_names.json').write_text(json.dumps({
        'names': names, 'feature_ids': fids,
    }, indent=2))
    print('Wrote channel_names.json')
