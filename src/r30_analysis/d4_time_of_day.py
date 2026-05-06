"""
Dim 4: Time-of-day regime analysis.
For each horizon, compute per-trade pnl AND active-rate by time bucket
using iter_002 LOSO OOF predictions across all 5 syms × 240 sessions.
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = 'data'; OUT = 'research_and_history/r30_figures'
HORIZONS = [5, 10, 20, 40, 60]
FEE = 0.000065  # one-side; calibrated: 2*fee=0.00013 reproduces LOSO sum h=60 +6.37 vs reported +6.30

# Load OOF
def load_oof(h):
    parts = []
    for held in range(5):
        df = pd.read_parquet(f'experiments/T5b_features_multihorizon/loso_pred_h{h}_held{held}.parquet')
        parts.append(df)
    return pd.concat(parts, ignore_index=True)

# Build (sym, date, session, t) -> time_str map by reading all test snapshots once
print('Loading time index from snapshots (sym 0..4 × dates 96..119 × {am,pm})...', flush=True)
time_map = {}  # (sym, date, sess) -> array indexed by t
for sym in range(5):
    for date in range(96, 120):
        for sess in ['am','pm']:
            f = f'{DATA}/snapshot_sym{sym}_date{date}_{sess}.parquet'
            if os.path.exists(f):
                df = pd.read_parquet(f, columns=['time'])
                # store HH*60+MM
                tarr = np.array([t.hour*60 + t.minute for t in df['time'].values], dtype=np.int32)
                time_map[(sym, date, sess)] = tarr
print(f'  loaded {len(time_map)} session time arrays', flush=True)

def bucket_min(m):
    if m < 600:   return 'open_09:40-10:00'
    elif m < 680: return 'mid_morn_10:00-11:20'
    elif m < 825: return 'pm_open_13:10-13:30'
    else:         return 'pm_late_13:30-14:50'

# Per-horizon analysis
all_results = {}
for h in HORIZONS:
    print(f'horizon {h}...', flush=True)
    oof = load_oof(h)
    # add bucket
    keys = list(zip(oof['sym'].values.astype(int), oof['date'].values.astype(int), oof['session'].values))
    times = []
    for k, t in zip(keys, oof['t'].values.astype(int)):
        ta = time_map.get(k)
        if ta is not None and 0 <= t < len(ta):
            times.append(int(ta[t]))
        else:
            times.append(-1)
    oof['min_of_day'] = times
    oof = oof[oof['min_of_day'] > 0]
    oof['bucket'] = oof['min_of_day'].map(bucket_min)
    # PNL: per-trade pnl assuming buy at midprice_t, sell at midprice_th, minus 2 fee
    # Active = pred != 1
    # If pred == 2: long, pnl = (mid_th - mid_t) - fee*2
    # If pred == 0: short, pnl = (mid_t - mid_th) - fee*2
    sgn = np.where(oof['pred_label'] == 2, 1.0, np.where(oof['pred_label'] == 0, -1.0, 0.0))
    raw = (oof['midprice_th'].values - oof['midprice_t'].values) * sgn
    # subtract 2 * fee for active trades. Assume fee = 0.00007
    pnl = np.where(sgn != 0, raw - 2*FEE, 0.0)
    oof['pnl'] = pnl
    oof['active'] = (sgn != 0).astype(int)
    oof['correct_active'] = ((oof['pred_label']==oof['true_label']) & (oof['active']==1)).astype(int)
    
    # Aggregate per bucket × per sym
    rows = []
    for bk, gb in oof.groupby('bucket'):
        for sym, g2 in gb.groupby('sym'):
            n = len(g2)
            n_act = int(g2['active'].sum())
            sum_pnl = float(g2['pnl'].sum())
            ppt = sum_pnl / n_act if n_act > 0 else 0
            hit = float(g2['correct_active'].sum() / n_act) if n_act > 0 else 0
            rows.append({'horizon': h, 'bucket': bk, 'sym': int(sym),
                         'n': n, 'n_active': n_act, 'active_rate': n_act/n,
                         'sum_pnl': sum_pnl, 'per_trade_pnl': ppt, 'hit_rate_active': hit})
        # also pooled (sym=-1)
        n = len(gb); n_act = int(gb['active'].sum())
        sum_pnl = float(gb['pnl'].sum())
        ppt = sum_pnl / n_act if n_act > 0 else 0
        hit = float(gb['correct_active'].sum() / n_act) if n_act > 0 else 0
        rows.append({'horizon': h, 'bucket': bk, 'sym': -1,
                     'n': n, 'n_active': n_act, 'active_rate': n_act/n,
                     'sum_pnl': sum_pnl, 'per_trade_pnl': ppt, 'hit_rate_active': hit})
    all_results[h] = rows

import pandas as pd
all_df = pd.DataFrame([r for h in HORIZONS for r in all_results[h]])
all_df.to_csv(f'{OUT}/d4_tod_summary.csv', index=False)

# --- Plot 1: per-trade pnl × bucket × horizon (pooled across syms) ---
buckets = ['open_09:40-10:00','mid_morn_10:00-11:20','pm_open_13:10-13:30','pm_late_13:30-14:50']
pooled = all_df[all_df.sym == -1]
fig, ax = plt.subplots(figsize=(12,6))
xs = np.arange(len(buckets)); W = 0.16
for i, h in enumerate(HORIZONS):
    vals = [pooled[(pooled.horizon==h)&(pooled.bucket==b)].per_trade_pnl.iloc[0] if len(pooled[(pooled.horizon==h)&(pooled.bucket==b)]) else 0 for b in buckets]
    ax.bar(xs+(i-2)*W, vals, W, label=f'h={h}')
ax.axhline(0, color='black', lw=0.5)
ax.axhline(-2*FEE, color='red', ls='--', label='-2×fee = -1.4bp')
ax.set_xticks(xs); ax.set_xticklabels(buckets, rotation=15)
ax.set_ylabel('per-trade pnl (raw - 2×fee)')
ax.set_title('iter_002 per-trade PnL by horizon × time-of-day  (pooled across syms)')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/d4_pnl_per_trade_tod.png', dpi=110, bbox_inches='tight'); plt.close()

# --- Plot 2: active rate × bucket × horizon ---
fig, ax = plt.subplots(figsize=(12,6))
for i, h in enumerate(HORIZONS):
    vals = [pooled[(pooled.horizon==h)&(pooled.bucket==b)].active_rate.iloc[0] if len(pooled[(pooled.horizon==h)&(pooled.bucket==b)]) else 0 for b in buckets]
    ax.bar(xs+(i-2)*W, vals, W, label=f'h={h}')
ax.set_xticks(xs); ax.set_xticklabels(buckets, rotation=15)
ax.set_ylabel('active rate (pred != 1)')
ax.set_title('iter_002 trading frequency by horizon × time-of-day')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/d4_active_rate_tod.png', dpi=110, bbox_inches='tight'); plt.close()

# --- Plot 3: cum pnl × bucket × horizon ---
fig, ax = plt.subplots(figsize=(12,6))
for i, h in enumerate(HORIZONS):
    vals = [pooled[(pooled.horizon==h)&(pooled.bucket==b)].sum_pnl.iloc[0] if len(pooled[(pooled.horizon==h)&(pooled.bucket==b)]) else 0 for b in buckets]
    ax.bar(xs+(i-2)*W, vals, W, label=f'h={h}')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(xs); ax.set_xticklabels(buckets, rotation=15)
ax.set_ylabel('sum LOSO pnl')
ax.set_title('iter_002 LOSO cumulative PnL by horizon × time-of-day')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/d4_sum_pnl_tod.png', dpi=110, bbox_inches='tight'); plt.close()

# Print overall summary
print()
print('=== Per-trade PnL by (horizon × bucket) — pooled all sym ===')
piv = pooled.pivot_table(index='bucket', columns='horizon', values='per_trade_pnl')
print(piv.round(6))
print()
print('=== Active rate by (horizon × bucket) — pooled ===')
piv2 = pooled.pivot_table(index='bucket', columns='horizon', values='active_rate')
print(piv2.round(4))
print()
print('=== Sum PnL by (horizon × bucket) — pooled ===')
piv3 = pooled.pivot_table(index='bucket', columns='horizon', values='sum_pnl')
print(piv3.round(2))

print('Dim4 done.')
