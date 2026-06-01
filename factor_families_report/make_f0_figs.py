import json, re
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

imp_list = json.load(open('importance.json'))
feat_names = open('/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p/schemeP_feat_names.txt').read().strip().splitlines()
RAW_N = 154
raw_names = feat_names[:RAW_N]

def cat(name):
    if name in ('open','high','low','close'): return 'OHLC'
    if name in ('volume_delta','amount_delta'): return 'volume/amount delta'
    if re.match(r'^bid\d+$', name): return 'bid price L1-L10'
    if re.match(r'^ask\d+$', name): return 'ask price L1-L10'
    if re.match(r'^bsize\d+$', name): return 'bid size L1-L10'
    if re.match(r'^asize\d+$', name): return 'ask size L1-L10'
    if name in ('avgbid','avgask','totalbsize','totalasize'): return 'LOB aggregate (avgbid/avgask/totalbsize/totalasize)'
    if re.match(r'^midprice\d+$', name): return 'midprice levels'
    if re.match(r'^spread\d+$', name): return 'spread levels'
    if re.match(r'^bid_diff\d+$', name): return 'bid_diff levels'
    if re.match(r'^ask_diff\d+$', name): return 'ask_diff levels'
    if name in ('bid_mean','ask_mean','bsize_mean','asize_mean','cumspread','imbalance'): return 'LOB simple aggregates'
    if re.match(r'^(bid|ask)_rate\d+$', name): return 'price rate'
    if re.match(r'^(bsize|asize)_rate\d+$', name): return 'size rate'
    if name.endswith('_intst'): return 'order flow intensity (intst)'
    if name.endswith('_ind'): return 'order flow indicator (ind)'
    if name.endswith('_acc'): return 'order flow acceleration (acc)'
    return 'OTHER'

gain = {it['name']: it.get('gain', 0.0) for it in imp_list}
total_gain = sum(gain.values())
raw_total = sum(gain.get(n, 0) for n in raw_names)
derived_total = total_gain - raw_total

cat_gain = defaultdict(float); cat_count = defaultdict(int)
for n in raw_names:
    cat_gain[cat(n)] += gain.get(n, 0.0)
    cat_count[cat(n)] += 1

items = sorted(cat_gain.items(), key=lambda x: x[1])  # ascending for barh

# === Figure 1: F0 horizontal bar of category gain (raw 154 only) ===
fig, ax = plt.subplots(figsize=(9.0, 5.6))
labels = [c for c,_ in items]
vals   = [v for _,v in items]
share_raw = [v/raw_total*100 for v in vals]
avg_per = [v/cat_count[c] for c,v in items]

bar_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(items)))
y = np.arange(len(items))
ax.barh(y, share_raw, color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Share of raw-154 total gain (%)', fontsize=10)
ax.set_title('F0: Raw 154-dim feature categories — LightGBM gain share (within raw)', fontsize=11)
for i, (s, a, c) in enumerate(zip(share_raw, avg_per, [cat_count[c] for c,_ in items])):
    ax.text(s + 0.4, i, f'{s:.1f}% (n={c}, avg={a:.3f})', va='center', fontsize=8)
ax.set_xlim(0, max(share_raw)*1.32)
ax.grid(axis='x', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('f0_raw_category_gain.pdf', bbox_inches='tight')
plt.close()
print('saved f0_raw_category_gain.pdf')

# === Figure 2: F0 raw vs derived (donut+bar) ===
fig, ax = plt.subplots(figsize=(7.5, 3.5))
parts = [('Raw 154 (F0)', raw_total/total_gain*100, '#2c6fbb'),
         ('Derived 216 (F1–F6 derived)', derived_total/total_gain*100, '#d4793b')]
y = [0]
left = 0
for lab, v, col in parts:
    ax.barh(y, v, left=left, color=col, edgecolor='black', linewidth=0.6, label=f'{lab}: {v:.1f}%')
    ax.text(left + v/2, 0, f'{lab}\n{v:.1f}%', ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    left += v
ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel('Share of total LGB gain (%)')
ax.set_title('F0 vs derived: raw 154 dims carry 70.9% of total LGB gain', fontsize=11)
ax.grid(axis='x', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig('f0_raw_vs_derived.pdf', bbox_inches='tight')
plt.close()
print('saved f0_raw_vs_derived.pdf')

# === Figure 3: 6x3 order-flow heatmap ===
prefixes = ['lb','la','mb','ma','cb','ca']
suffixes = ['intst','ind','acc']
mat = np.zeros((len(prefixes), len(suffixes)))
for i,p in enumerate(prefixes):
    for j,s in enumerate(suffixes):
        mat[i,j] = gain.get(f'{p}_{s}', 0.0)

fig, ax = plt.subplots(figsize=(5.0, 5.2))
im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(suffixes))); ax.set_xticklabels(suffixes)
ax.set_yticks(range(len(prefixes))); ax.set_yticklabels(['lb (limit-bid)','la (limit-ask)','mb (market-bid)','ma (market-ask)','cb (cancel-bid)','ca (cancel-ask)'])
for i in range(len(prefixes)):
    for j in range(len(suffixes)):
        color = 'white' if mat[i,j] > mat.max()*0.55 else 'black'
        ax.text(j, i, f'{mat[i,j]:.3f}', ha='center', va='center', color=color, fontsize=9)
ax.set_title('F0 order-flow 18 raw dims: gain heatmap\n(6 event types × 3 variants)', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='LGB gain')
plt.tight_layout()
plt.savefig('f0_orderflow_heatmap.pdf', bbox_inches='tight')
plt.close()
print('saved f0_orderflow_heatmap.pdf')

print('TOTALS: total_gain=%.4f raw_total=%.4f derived=%.4f raw_share=%.2f%%' %
      (total_gain, raw_total, derived_total, raw_total/total_gain*100))
