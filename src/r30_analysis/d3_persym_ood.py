"""
Dim 3: Per-sym differences (microstructure profile + activity intensity)
to characterize each sym so we can guess which one platform's unknown sym
most resembles.

Stats per sym:
  - midprice volatility (bp/tick)
  - bid-ask spread (bp)
  - bid1 size, ask1 size (tick-by-tick avg) — liquidity proxy
  - amount_delta (元) — trading activity
  - turnover rate (volume_delta) per tick
  - tick-arrival of OFI (pos vs neg)
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = 'data'; OUT = 'research_and_history/r30_figures'
np.random.seed(42)

sample = []
for sym in range(5):
    dates = list(range(120)); np.random.shuffle(dates)
    for d in dates[:24]:
        for s in ['am','pm']:
            f = f'{DATA}/snapshot_sym{sym}_date{d}_{s}.parquet'
            if os.path.exists(f): sample.append((sym,d,s,f))

# Per-sym time-of-day buckets too (for question 4)
def bucket(t):
    minutes = t.hour*60 + t.minute
    if minutes < 600:   return 'open'      # 09:40 - 10:00
    elif minutes < 680: return 'mid_morn'  # 10:00 - 11:20
    elif minutes < 815: return 'lunch'     # never; actually pm starts 13:00
    elif minutes < 825: return 'pm_open'   # 13:10 - 13:30
    else:               return 'pm_close'  # 13:30 - 14:50/15:00

stats = {sym: [] for sym in range(5)}
for sym, d, s, f in sample:
    df = pd.read_parquet(f, columns=['time','midprice','bid1','ask1','bsize1','asize1',
                                     'amount_delta','volume_delta','imbalance',
                                     'totalbsize','totalasize'])
    spread = (df['ask1'] - df['bid1']).values
    mid = df['midprice'].values
    md1 = np.diff(mid)
    rec = {
        'sym': sym, 'date': d, 'sess': s,
        'mid_std_bp': float(np.std(md1)*1e4),
        'spread_bp_mean': float(np.mean(spread)*1e4),
        'spread_bp_p95': float(np.percentile(spread, 95)*1e4),
        'bsize1_mean': float(df['bsize1'].mean()),
        'asize1_mean': float(df['asize1'].mean()),
        'amount_delta_mean': float(df['amount_delta'].mean()),
        'amount_delta_p99': float(df['amount_delta'].quantile(0.99)),
        'volume_delta_mean': float(df['volume_delta'].mean()),
        'imbalance_std': float(df['imbalance'].std()),
        'mid_drift_total_bp': float((mid[-1] - mid[0])*1e4),
    }
    stats[sym].append(rec)

# Aggregate
agg = []
for sym in range(5):
    d = pd.DataFrame(stats[sym])
    agg.append({
        'sym': sym,
        'n_sessions': len(d),
        'mid_std_bp_mean': d.mid_std_bp.mean(),
        'mid_std_bp_p50': d.mid_std_bp.median(),
        'spread_bp_mean': d.spread_bp_mean.mean(),
        'spread_bp_p95_mean': d.spread_bp_p95.mean(),
        'bsize1_mean': d.bsize1_mean.mean(),
        'asize1_mean': d.asize1_mean.mean(),
        'amount_delta_mean': d.amount_delta_mean.mean(),
        'amount_delta_p99_mean': d.amount_delta_p99.mean(),
        'volume_delta_mean': d.volume_delta_mean.mean(),
        'imbalance_std_mean': d.imbalance_std.mean(),
        'mid_drift_abs_mean_bp': d.mid_drift_total_bp.abs().mean(),
    })
adf = pd.DataFrame(agg)
adf.to_csv(f'{OUT}/d3_persym_summary.csv', index=False)
print(adf.to_string(index=False))

# --- Plot: per-sym profile radar-ish ---
metrics = ['mid_std_bp_mean','spread_bp_mean','bsize1_mean','amount_delta_mean','imbalance_std_mean']
labels  = ['mid σ (bp)','spread (bp)','bsize1 avg','amount/tick (元)','imbalance σ']
# Normalize to 0..1 across syms
norm = adf[metrics].copy()
for c in metrics:
    v = norm[c].values
    norm[c] = (v - v.min()) / (v.max() - v.min() + 1e-12)
fig, axes = plt.subplots(1, 5, figsize=(20,4))
for i, sym in enumerate(range(5)):
    ax = axes[i]
    vals = norm.iloc[i].values
    ax.bar(labels, vals, color='steelblue')
    ax.set_title(f'sym {sym}\nspread={adf.iloc[i].spread_bp_mean:.2f}bp σmid={adf.iloc[i].mid_std_bp_mean:.2f}bp')
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.savefig(f'{OUT}/d3_persym_profile.png', dpi=110, bbox_inches='tight'); plt.close()

# --- Plot: distribution of session-level mid σ per sym ---
fig, ax = plt.subplots(figsize=(10,5))
data = [pd.DataFrame(stats[sym]).mid_std_bp.values for sym in range(5)]
bp = ax.boxplot(data, labels=[f'sym {s}' for s in range(5)], showfliers=False, patch_artist=True)
for p in bp['boxes']: p.set_facecolor('lightsteelblue')
ax.set_ylabel('per-session σ(Δmid_1) [bp]')
ax.set_title('Mid-price tick volatility per sym (across sessions)')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/d3_persym_volatility.png', dpi=110, bbox_inches='tight'); plt.close()

# --- per-sym × time-of-day: σ(Δmid) ---
buckets = ['open','mid_morn','pm_open','pm_close']
tod_stats = {b: {sym: [] for sym in range(5)} for b in buckets}
for sym, d, s, f in sample[:80]:  # subsample
    df = pd.read_parquet(f, columns=['time','midprice'])
    df['bucket'] = df['time'].apply(bucket)
    md1 = np.diff(df['midprice'].values)
    bk = df['bucket'].values[1:]  # align with diff
    for b in buckets:
        mask = bk == b
        if mask.sum() > 50:
            tod_stats[b][sym].append(np.std(md1[mask])*1e4)

fig, ax = plt.subplots(figsize=(10,6))
xs = np.arange(len(buckets)); W = 0.16
for i, sym in enumerate(range(5)):
    means = [np.mean(tod_stats[b][sym]) if tod_stats[b][sym] else 0 for b in buckets]
    ax.bar(xs+(i-2)*W, means, W, label=f'sym {sym}')
ax.set_xticks(xs); ax.set_xticklabels(buckets)
ax.set_ylabel('σ(Δmid_1) [bp]'); ax.set_title('Tick volatility per sym × time-of-day')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/d3_persym_tod_volatility.png', dpi=110, bbox_inches='tight'); plt.close()

# Save
json.dump(adf.to_dict(orient='records'), open(f'{OUT}/d3_persym_summary.json','w'), indent=2, default=float)
print('Dim3 done.')
