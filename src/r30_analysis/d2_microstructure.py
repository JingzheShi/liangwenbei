"""
Dim 2: Microstructure noise vs fundamental signal.
- mid_diff lag-1 autocorrelation (negative => bid-ask bounce dominates)
- Roll's spread implied = 2*sqrt(-Cov(Δp_t, Δp_{t-1}))   if Cov<0
- Variance ratio test: Var(Δmid_h) / (h * Var(Δmid_1)) — random walk = 1
- Order-flow imbalance vs future Δmid_h correlation
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = 'data'; OUT = 'research_and_history/r30_figures'
HORIZONS = [1, 5, 10, 20, 40, 60]
np.random.seed(42)

# Sample: 16 sessions per sym for speed
sample = []
for sym in range(5):
    dates = list(range(120)); np.random.shuffle(dates)
    for d in dates[:16]:
        for s in ['am','pm']:
            f = f'{DATA}/snapshot_sym{sym}_date{d}_{s}.parquet'
            if os.path.exists(f): sample.append((sym,d,s,f))
print(f'sessions: {len(sample)}', flush=True)

results = {sym: {'mid_diff_ac1': [], 'rolls_spread_bp': [], 'var_ratio': {h: [] for h in HORIZONS},
                 'ofi_corr_h': {h: [] for h in HORIZONS}, 'mid_d1_var': []}
           for sym in range(5)}

# Define OFI (Order Flow Imbalance, Cont 2014):
#   OFI_t = sum over levels of:
#     (bsize_t - bsize_{t-1} if bid_t == bid_{t-1};
#      +bsize_t if bid_t > bid_{t-1};
#      -bsize_{t-1} if bid_t < bid_{t-1}) ... etc for ask side (sign flipped).
# Use only level 1 for simplicity here.
def compute_ofi1(df):
    bid1 = df['bid1'].values
    ask1 = df['ask1'].values
    bs1 = df['bsize1'].values
    as1 = df['asize1'].values
    n = len(df)
    ofi = np.zeros(n)
    # bid contribution
    bid_chg = np.sign(bid1[1:] - bid1[:-1])
    bs_diff = bs1[1:] - bs1[:-1]
    # if bid up: +bs1[t]; if same: bs_diff; if down: -bs1[t-1]
    delta_b = np.where(bid_chg > 0, bs1[1:], np.where(bid_chg == 0, bs_diff, -bs1[:-1]))
    # ask contribution
    ask_chg = np.sign(ask1[1:] - ask1[:-1])
    as_diff = as1[1:] - as1[:-1]
    # if ask up: -as1[t-1]; if same: -as_diff; if down: +as1[t]
    delta_a = np.where(ask_chg > 0, -as1[:-1], np.where(ask_chg == 0, -as_diff, as1[1:]))
    ofi[1:] = delta_b + delta_a
    return ofi

n = 0
for sym, d, s, f in sample:
    df = pd.read_parquet(f, columns=['midprice','bid1','ask1','bsize1','asize1'])
    mid = df['midprice'].values
    md1 = np.diff(mid)
    if md1.std() < 1e-12: continue
    # AC lag-1
    x = md1 - md1.mean(); denom = (x**2).mean()
    ac1 = (x[:-1]*x[1:]).mean() / denom if denom > 0 else 0
    results[sym]['mid_diff_ac1'].append(ac1)
    # Roll's spread implied
    cov1 = np.cov(md1[:-1], md1[1:])[0,1]
    if cov1 < 0:
        rs = 2 * np.sqrt(-cov1)
    else:
        rs = 0.0
    results[sym]['rolls_spread_bp'].append(rs * 1e4)
    results[sym]['mid_d1_var'].append(md1.var())
    # variance ratio: Var(mid_h - mid_0) / (h * Var(mid_1 - mid_0))
    var1 = md1.var()
    for h in HORIZONS:
        if h == 1:
            results[sym]['var_ratio'][1].append(1.0)
        else:
            dh = mid[h:] - mid[:-h]
            results[sym]['var_ratio'][h].append(dh.var() / (h * var1) if var1 > 0 else np.nan)
    # OFI vs future Δmid corr
    ofi = compute_ofi1(df)
    for h in HORIZONS:
        if h == 1:
            future = np.diff(mid)
            x_ofi = ofi[1:]
        else:
            future = mid[h:] - mid[:-h]
            x_ofi = ofi[:-h+1] if h > 0 else ofi  # align at t
            x_ofi = ofi[1:len(mid)-h+1]  # use ofi[t] vs (mid[t+h]-mid[t]) starting t=1
            future = mid[h:len(mid)] - mid[:len(mid)-h]
            # align lengths
            L = min(len(x_ofi), len(future))
            x_ofi = x_ofi[:L]; future = future[:L]
        if x_ofi.std() < 1e-15 or future.std() < 1e-15:
            results[sym]['ofi_corr_h'][h].append(0.0)
        else:
            r = np.corrcoef(x_ofi, future)[0,1]
            results[sym]['ofi_corr_h'][h].append(r if not np.isnan(r) else 0.0)
    n += 1
print(f'  processed {n} sessions', flush=True)

# Aggregate
summary = []
for sym in range(5):
    summary.append({
        'sym': sym,
        'mid_diff_ac1_mean': float(np.mean(results[sym]['mid_diff_ac1'])),
        'rolls_spread_bp_mean': float(np.mean(results[sym]['rolls_spread_bp'])),
        'mid_d1_std_bp': float(np.sqrt(np.mean(results[sym]['mid_d1_var'])) * 1e4),
        'var_ratio_h60': float(np.mean(results[sym]['var_ratio'][60])),
        'ofi_corr_h1': float(np.mean(results[sym]['ofi_corr_h'][1])),
        'ofi_corr_h5': float(np.mean(results[sym]['ofi_corr_h'][5])),
        'ofi_corr_h20': float(np.mean(results[sym]['ofi_corr_h'][20])),
        'ofi_corr_h60': float(np.mean(results[sym]['ofi_corr_h'][60])),
    })
sdf = pd.DataFrame(summary)
sdf.to_csv(f'{OUT}/d2_microstructure_summary.csv', index=False)
print(sdf.to_string(index=False))

# --- Plot: variance ratio vs horizon ---
fig, ax = plt.subplots(figsize=(10,6))
for sym in range(5):
    means = [np.mean(results[sym]['var_ratio'][h]) for h in HORIZONS]
    ax.plot(HORIZONS, means, marker='o', label=f'sym {sym}')
ax.axhline(1.0, color='black', ls='--', label='random walk (=1)')
ax.set_xlabel('horizon h'); ax.set_ylabel('Var(Δmid_h) / (h · Var(Δmid_1))')
ax.set_title("Variance ratio test\n>1 = trending (momentum), <1 = mean-reverting (bounce)")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/d2_variance_ratio.png', dpi=110, bbox_inches='tight'); plt.close()

# --- Plot: AC1 of mid_diff per sym ---
fig, ax = plt.subplots(figsize=(8,5))
ac1s = [np.mean(results[sym]['mid_diff_ac1']) for sym in range(5)]
ax.bar(range(5), ac1s, color=['steelblue']*5)
ax.axhline(0, color='black', lw=0.5)
for i, v in enumerate(ac1s): ax.text(i, v+np.sign(v)*0.005, f'{v:.3f}', ha='center', fontsize=10)
ax.set_xlabel('sym'); ax.set_xticks(range(5))
ax.set_ylabel('lag-1 autocorrelation of Δmid_1')
ax.set_title('Microstructure: lag-1 autocorr of mid increments\nNegative = bid-ask bounce; near 0 = random walk')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/d2_mid_diff_ac1.png', dpi=110, bbox_inches='tight'); plt.close()

# --- Plot: OFI predictive correlation vs horizon ---
fig, ax = plt.subplots(figsize=(10,6))
for sym in range(5):
    corrs = [np.mean(results[sym]['ofi_corr_h'][h]) for h in HORIZONS]
    ax.plot(HORIZONS, corrs, marker='o', label=f'sym {sym}')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('horizon h'); ax.set_ylabel('Corr(OFI_t, Δmid_{t,t+h})')
ax.set_title('Order-flow imbalance predictive correlation across horizons')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/d2_ofi_corr.png', dpi=110, bbox_inches='tight'); plt.close()

json.dump(summary, open(f'{OUT}/d2_microstructure_summary.json','w'), indent=2)
print('Dim2 done.')
