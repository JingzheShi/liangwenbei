"""
Dim 1: Per-horizon label structure analysis.
For each horizon h ∈ {5,10,20,40,60}:
  - distribution of Δmid = mid_{t+h} - mid_t
  - P(|Δmid| > α_h)
  - signal-to-noise: var(Δmid) / fee²
  - per-sym × per-horizon stats
We sample 60 sessions (12 per sym from train dates 0..95) for speed.
"""
import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA = 'data'
OUT = 'research_and_history/r30_figures'
os.makedirs(OUT, exist_ok=True)

ALPHA = {5: 0.0005, 10: 0.0005, 20: 0.001, 40: 0.001, 60: 0.001}
HORIZONS = [5, 10, 20, 40, 60]
FEE = 0.00007  # one-side fee assumption (typical for HFT, also matches per-trade ≈ -0.0001)

# Sample sessions: 16 per sym, mix of train+test dates so we cover regime
np.random.seed(42)
sample_sessions = []
for sym in range(5):
    dates = list(range(0, 120))
    np.random.shuffle(dates)
    for d in dates[:16]:
        for sess in ['am', 'pm']:
            f = f'{DATA}/snapshot_sym{sym}_date{d}_{sess}.parquet'
            if os.path.exists(f):
                sample_sessions.append((sym, d, sess, f))
print(f'Sampling {len(sample_sessions)} sessions across 5 syms', flush=True)

# Compute Δmid for each horizon and aggregate
agg = {h: {sym: [] for sym in range(5)} for h in HORIZONS}
n_done = 0
for sym, d, sess, f in sample_sessions:
    df = pd.read_parquet(f, columns=['midprice'])
    mid = df['midprice'].values
    for h in HORIZONS:
        delta = mid[h:] - mid[:-h]
        agg[h][sym].append(delta)
    n_done += 1
    if n_done % 40 == 0:
        print(f'  loaded {n_done}/{len(sample_sessions)}', flush=True)

# Concatenate
flat = {h: {sym: np.concatenate(agg[h][sym]) for sym in range(5)} for h in HORIZONS}
print('aggregated.', flush=True)

# --- Stats table ---
stats_rows = []
for h in HORIZONS:
    a = ALPHA[h]
    for sym in range(5):
        d = flat[h][sym]
        stats_rows.append({
            'horizon': h, 'sym': sym, 'n': len(d),
            'std': float(np.std(d)),
            'mean_abs': float(np.mean(np.abs(d))),
            'p50_abs': float(np.percentile(np.abs(d), 50)),
            'p_active': float(np.mean(np.abs(d) > a)),
            'p_up': float(np.mean(d > a)),
            'p_down': float(np.mean(d < -a)),
            'snr_vs_fee': float(np.var(d) / (FEE**2)),
            'cond_mean_active': float(np.mean(np.abs(d[np.abs(d) > a]))) if np.any(np.abs(d) > a) else 0,
        })
    # global row
    d_all = np.concatenate([flat[h][s] for s in range(5)])
    stats_rows.append({
        'horizon': h, 'sym': -1, 'n': len(d_all),
        'std': float(np.std(d_all)),
        'mean_abs': float(np.mean(np.abs(d_all))),
        'p50_abs': float(np.percentile(np.abs(d_all), 50)),
        'p_active': float(np.mean(np.abs(d_all) > a)),
        'p_up': float(np.mean(d_all > a)),
        'p_down': float(np.mean(d_all < -a)),
        'snr_vs_fee': float(np.var(d_all) / (FEE**2)),
        'cond_mean_active': float(np.mean(np.abs(d_all[np.abs(d_all) > a]))) if np.any(np.abs(d_all) > a) else 0,
    })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(f'{OUT}/d1_label_stats.csv', index=False)
print(stats_df[stats_df.sym == -1].to_string(index=False))

# --- Plot 1: Δmid distribution (log scale) for all horizons (global, all sym pooled) ---
fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
for ax, h in zip(axes, HORIZONS):
    a = ALPHA[h]
    d_all = np.concatenate([flat[h][s] for s in range(5)])
    bins = np.linspace(-0.005, 0.005, 121)
    ax.hist(d_all, bins=bins, alpha=0.7, density=True, color='steelblue')
    ax.axvline(a, color='red', ls='--', lw=1, label=f'α={a*100:.2f}%')
    ax.axvline(-a, color='red', ls='--', lw=1)
    ax.axvline(FEE, color='orange', ls=':', lw=1, label='fee≈0.7bp')
    ax.axvline(-FEE, color='orange', ls=':', lw=1)
    p_act = np.mean(np.abs(d_all) > a)
    ax.set_title(f'h={h}  P(|Δ|>α)={p_act:.2%}\nσ={np.std(d_all)*1e4:.1f} bp')
    ax.set_xlabel('Δmidprice')
    ax.set_yscale('log')
    if h == 5: ax.set_ylabel('density (log)')
    ax.legend(fontsize=8, loc='upper left')
fig.suptitle('Δmidprice distribution across horizons (all sym pooled)')
plt.tight_layout()
plt.savefig(f'{OUT}/d1_delta_mid_dist.png', dpi=110, bbox_inches='tight')
plt.close()

# --- Plot 2: SNR vs fee, per sym, per horizon ---
fig, ax = plt.subplots(figsize=(10, 6))
syms = list(range(5))
xs = np.arange(len(HORIZONS))
W = 0.16
for i, sym in enumerate(syms):
    snrs = [stats_df[(stats_df.horizon==h) & (stats_df.sym==sym)].snr_vs_fee.iloc[0] for h in HORIZONS]
    ax.bar(xs + (i-2)*W, snrs, W, label=f'sym {sym}')
ax.set_xticks(xs); ax.set_xticklabels([f'h={h}' for h in HORIZONS])
ax.set_ylabel('Var(Δmid) / fee²  (signal/noise w.r.t. fee)')
ax.set_yscale('log')
ax.set_title('Signal-to-fee² ratio per (sym, horizon)\nHigher = more above-fee movement')
ax.axhline(1, color='red', ls='--', label='SNR=1 (Δ²==fee²)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/d1_snr_per_sym.png', dpi=110, bbox_inches='tight')
plt.close()

# --- Plot 3: P(|Δmid|>α) per (sym, horizon) ---
fig, ax = plt.subplots(figsize=(10, 6))
for i, sym in enumerate(syms):
    p_acts = [stats_df[(stats_df.horizon==h) & (stats_df.sym==sym)].p_active.iloc[0] for h in HORIZONS]
    ax.plot(HORIZONS, p_acts, marker='o', label=f'sym {sym}')
p_acts_g = [stats_df[(stats_df.horizon==h) & (stats_df.sym==-1)].p_active.iloc[0] for h in HORIZONS]
ax.plot(HORIZONS, p_acts_g, marker='s', color='black', lw=2.5, label='global')
ax.set_xlabel('horizon h'); ax.set_ylabel('P(|Δmid|>α_h)')
ax.set_title("Active label rate per (sym, horizon) — α=0.05% for h≤10, α=0.1% for h≥20")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/d1_p_active_per_sym.png', dpi=110, bbox_inches='tight')
plt.close()

# --- Plot 4: cond E[|Δmid| | |Δmid|>α] vs fee ---
# When the model says "trade", how big is the average move? must beat 2*fee for net pnl
fig, ax = plt.subplots(figsize=(10, 6))
for i, sym in enumerate(syms):
    cms = [stats_df[(stats_df.horizon==h) & (stats_df.sym==sym)].cond_mean_active.iloc[0] for h in HORIZONS]
    ax.plot(HORIZONS, cms, marker='o', label=f'sym {sym}')
cms_g = [stats_df[(stats_df.horizon==h) & (stats_df.sym==-1)].cond_mean_active.iloc[0] for h in HORIZONS]
ax.plot(HORIZONS, cms_g, marker='s', color='black', lw=2.5, label='global')
ax.axhline(2*FEE, color='red', ls='--', label='2×fee = 1.4bp (round-trip cost)')
ax.set_xlabel('horizon h'); ax.set_ylabel('E[|Δmid| | |Δmid|>α]')
ax.set_title('Conditional mean of magnitude when active — must exceed 2×fee for profit')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/d1_cond_mean_when_active.png', dpi=110, bbox_inches='tight')
plt.close()

# --- Plot 5: Δmid autocorrelation (lag-1) per sym, per horizon -----------------
# This tells us about predictable structure
def per_session_autocorr(deltas_list, lag):
    accs = []
    for d in deltas_list:
        if len(d) < lag+10:
            continue
        x = d - d.mean()
        denom = (x**2).mean()
        if denom > 1e-20:
            accs.append( (x[:-lag]*x[lag:]).mean() / denom )
    return np.mean(accs) if accs else 0.0

ac_rows = []
for h in HORIZONS:
    for sym in range(5):
        for lag in [1, 5, 20]:
            ac = per_session_autocorr(agg[h][sym], lag)
            ac_rows.append({'horizon': h, 'sym': sym, 'lag': lag, 'ac': ac})
ac_df = pd.DataFrame(ac_rows)
ac_df.to_csv(f'{OUT}/d1_label_autocorr.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, lag in zip(axes, [1, 5, 20]):
    for sym in range(5):
        vals = [ac_df[(ac_df.horizon==h)&(ac_df.sym==sym)&(ac_df.lag==lag)].ac.iloc[0] for h in HORIZONS]
        ax.plot(HORIZONS, vals, marker='o', label=f'sym {sym}')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('horizon h'); ax.set_title(f'lag={lag} autocorrelation of Δmid_h(t)')
    if lag == 1: ax.set_ylabel('autocorr')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('Autocorrelation of Δmid_h: positive = predictable (momentum), negative = mean-reverting')
plt.tight_layout()
plt.savefig(f'{OUT}/d1_label_autocorr.png', dpi=110, bbox_inches='tight')
plt.close()

print('Dim1 done')
