"""
Dim 6: aug_a vs baseline feature importance & per-sym predictions diff.
- FI gain change: which features rise/fall after aug_a?
- Cross-sym prediction correlation: is aug_a more consistent across sym?
- Per-sym pnl change: where does aug_a help most?
"""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'research_and_history/r30_figures'
FEE = 0.000065

# Load FI
fi = json.load(open('experiments/T26_domain_randomization/feat_importance_T26.json'))
feat_names = fi['baseline_held0']['feat_names']
n_feat = len(feat_names)

# Average FI over 5 folds for baseline & aug_a
def avg_fi(prefix, kind='fi_gain'):
    arrs = [np.array(fi[f'{prefix}_held{h}'][kind]) for h in range(5)]
    return np.mean(np.stack(arrs), axis=0)

fi_base = avg_fi('baseline', 'fi_gain')
fi_a    = avg_fi('aug_a', 'fi_gain')
fi_b    = avg_fi('aug_b', 'fi_gain')
fi_c    = avg_fi('aug_c', 'fi_gain')

# Normalize to relative importance (sum=1) before comparing
def normalize(x): return x / (x.sum() + 1e-12)
nb, na = normalize(fi_base), normalize(fi_a)
delta = na - nb  # positive = more important under aug_a

df_fi = pd.DataFrame({'feat': feat_names,
                      'fi_base_norm': nb,
                      'fi_aug_a_norm': na,
                      'delta': delta,
                      'rank_base': np.argsort(-fi_base).argsort(),
                      'rank_aug_a': np.argsort(-fi_a).argsort(),
                      })
df_fi['rank_diff'] = df_fi['rank_base'] - df_fi['rank_aug_a']  # positive = climbed under aug_a
df_fi.to_csv(f'{OUT}/d6_fi_diff.csv', index=False)

# Top climbers / fallers
print('=== Top 15 features whose importance INCREASED under aug_a ===')
print(df_fi.sort_values('delta', ascending=False).head(15)[['feat','fi_base_norm','fi_aug_a_norm','delta','rank_base','rank_aug_a']].to_string(index=False))
print()
print('=== Top 15 features whose importance DECREASED under aug_a ===')
print(df_fi.sort_values('delta').head(15)[['feat','fi_base_norm','fi_aug_a_norm','delta','rank_base','rank_aug_a']].to_string(index=False))

# --- Plot 1: top 30 features by abs(delta) bar chart ---
top = df_fi.sort_values('delta', key=np.abs, ascending=False).head(30)
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['green' if d>0 else 'red' for d in top['delta']]
ax.barh(top['feat'][::-1], top['delta'].values[::-1], color=colors[::-1])
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('Δ (normalized FI under aug_a) − (under baseline)')
ax.set_title('Top 30 features by FI shift under aug_a (green=more important, red=less)')
plt.tight_layout(); plt.savefig(f'{OUT}/d6_fi_delta_top30.png', dpi=110, bbox_inches='tight'); plt.close()

# --- Plot 2: top 25 features by absolute importance, baseline vs aug_a side by side ---
top25 = df_fi.sort_values('fi_base_norm', ascending=False).head(25)
fig, ax = plt.subplots(figsize=(12, 7))
xs = np.arange(len(top25)); W = 0.4
ax.bar(xs-W/2, top25['fi_base_norm'].values, W, label='baseline', color='steelblue')
ax.bar(xs+W/2, top25['fi_aug_a_norm'].values, W, label='aug_a', color='orange')
ax.set_xticks(xs); ax.set_xticklabels(top25['feat'].values, rotation=75, ha='right', fontsize=8)
ax.legend(); ax.set_ylabel('normalized FI gain')
ax.set_title('Top 25 features by baseline FI: baseline vs aug_a')
plt.tight_layout(); plt.savefig(f'{OUT}/d6_fi_top25_compare.png', dpi=110, bbox_inches='tight'); plt.close()

# --- aug_a vs baseline OOF diff ---
def load_oof_T26(variant, h=60):
    parts = []
    for held in range(5):
        f = f'experiments/T26_domain_randomization/loso_pred_h{h}_{variant}_held{held}.parquet'
        if os.path.exists(f): parts.append(pd.read_parquet(f))
    return pd.concat(parts, ignore_index=True) if parts else None

oof_b = load_oof_T26('baseline')
oof_a = load_oof_T26('aug_a')
print(f'\nOOF baseline rows: {len(oof_b)}, aug_a rows: {len(oof_a)}')

# Per-sym pnl (with the SAME thresholding, hopefully OOF preds are already thresholded)
# Compute pnl for each
for tag, oof in [('baseline', oof_b), ('aug_a', oof_a)]:
    sgn = np.where(oof['pred_label']==2, 1, np.where(oof['pred_label']==0, -1, 0)).astype(np.float64)
    raw = (oof['midprice_th'].values - oof['midprice_t'].values) * sgn
    pnl = np.where(sgn != 0, raw - 2*FEE, 0.0)
    oof['pnl'] = pnl
    oof['active'] = (sgn != 0).astype(int)

# Per-sym pnl
fig, ax = plt.subplots(figsize=(10, 5))
xs = np.arange(5); W = 0.4
b_pnl = [float(oof_b[oof_b.sym==s].pnl.sum()) for s in range(5)]
a_pnl = [float(oof_a[oof_a.sym==s].pnl.sum()) for s in range(5)]
ax.bar(xs-W/2, b_pnl, W, label='baseline', color='steelblue')
ax.bar(xs+W/2, a_pnl, W, label='aug_a',    color='orange')
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(xs); ax.set_xticklabels([f'sym {s}' for s in range(5)])
ax.set_ylabel('LOSO sum PnL (h=60)')
ax.set_title(f'h=60 per-sym LOSO PnL: baseline ({sum(b_pnl):.2f}) vs aug_a ({sum(a_pnl):.2f})')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/d6_aug_a_persym_pnl.png', dpi=110, bbox_inches='tight'); plt.close()

print('\n=== Per-sym PnL diff (aug_a − baseline) ===')
for s in range(5):
    print(f'sym {s}: baseline {b_pnl[s]:+7.2f}, aug_a {a_pnl[s]:+7.2f}, Δ={a_pnl[s]-b_pnl[s]:+7.2f}')
print(f'TOTAL    : baseline {sum(b_pnl):+7.2f}, aug_a {sum(a_pnl):+7.2f}, Δ={sum(a_pnl)-sum(b_pnl):+7.2f}')

# --- Cross-sym prediction consistency ---
# How? Compute mean prob_2 - prob_0 per sym and look at variance across syms.
# Higher consistency = lower per-sym mean shift.
print('\n=== Average prob shift (prob_2 − prob_0) per sym ===')
for tag, oof in [('baseline', oof_b), ('aug_a', oof_a)]:
    shifts = [float((oof[oof.sym==s]['prob_2'] - oof[oof.sym==s]['prob_0']).mean()) for s in range(5)]
    spread = max(shifts) - min(shifts)
    print(f'{tag}:  per-sym shifts {[f"{x:+.4f}" for x in shifts]}  range={spread:.4f}')

# --- Plot: feature importance category aggregation ---
# Group features into categories
def categorize(name):
    name = name.lower()
    if 'spread' in name: return 'spread'
    if 'imbalance' in name: return 'imbalance'
    if name.startswith(('bid','ask')) and ('size' in name or 'rate' in name) and not name.startswith('asize'): pass
    if 'midprice' in name: return 'midprice'
    if name.startswith('bid') and 'size' not in name: return 'bid_price'
    if name.startswith('bsize') or name.startswith('totalbsize'): return 'bid_size'
    if name.startswith('ask') and 'size' not in name: return 'ask_price'
    if name.startswith('asize') or name.startswith('totalasize'): return 'ask_size'
    if 'avg' in name: return 'avg_lob'
    if name.startswith(('lb','la','mb','ma','cb','ca')): 
        if 'intst' in name: return 'orderflow_intst'
        if 'acc' in name: return 'orderflow_acc'
        if 'ind' in name: return 'orderflow_ind'
    if 'amount_delta' in name or 'volume_delta' in name: return 'volume'
    if 'open' in name or 'close' in name or 'high' in name or 'low' in name: return 'ohlc'
    return 'other'

df_fi['cat'] = df_fi['feat'].apply(categorize)
cat_agg = df_fi.groupby('cat').agg(
    fi_base=('fi_base_norm','sum'),
    fi_aug_a=('fi_aug_a_norm','sum'),
    n=('feat','size'),
).reset_index().sort_values('fi_base', ascending=False)
cat_agg['delta'] = cat_agg['fi_aug_a'] - cat_agg['fi_base']
print('\n=== Aggregated FI by category ===')
print(cat_agg.to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 5))
xs = np.arange(len(cat_agg)); W = 0.4
ax.bar(xs-W/2, cat_agg['fi_base'].values, W, label='baseline', color='steelblue')
ax.bar(xs+W/2, cat_agg['fi_aug_a'].values, W, label='aug_a', color='orange')
ax.set_xticks(xs); ax.set_xticklabels(cat_agg['cat'].values, rotation=30, ha='right')
ax.set_ylabel('summed normalized FI')
ax.set_title('FI aggregated by feature category — aug_a shifts emphasis toward...')
ax.legend(); ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/d6_fi_by_category.png', dpi=110, bbox_inches='tight'); plt.close()

cat_agg.to_csv(f'{OUT}/d6_fi_by_category.csv', index=False)
print('Dim6 done.')
