"""
Dim 5: iter_002 per-trade error analysis.
- Hit rate (correct active predictions / all active predictions)
- Sym-level per-trade pnl per horizon (already in OOF, but more detail)
- Loss-makers vs profit-makers: prob histogram, time of trade, sym
- Probability calibration: are high-prob preds more accurate?
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'research_and_history/r30_figures'
HORIZONS = [5, 10, 20, 40, 60]
FEE = 0.000065

def load_oof(h):
    parts = []
    for held in range(5):
        df = pd.read_parquet(f'experiments/T5b_features_multihorizon/loso_pred_h{h}_held{held}.parquet')
        parts.append(df)
    return pd.concat(parts, ignore_index=True)

# Detailed analysis
summary_rows = []
calib_curves = {}
prob_dist_data = {}

for h in HORIZONS:
    oof = load_oof(h)
    sgn = np.where(oof['pred_label']==2, 1, np.where(oof['pred_label']==0, -1, 0)).astype(np.float64)
    raw = (oof['midprice_th'].values - oof['midprice_t'].values) * sgn
    pnl = np.where(sgn != 0, raw - 2*FEE, 0.0)
    oof['pnl'] = pnl
    oof['active'] = (sgn != 0).astype(int)
    oof['correct'] = ((oof['pred_label'] == oof['true_label']) & (oof['active']==1)).astype(int)
    oof['profit'] = ((pnl > 0) & (oof['active']==1)).astype(int)
    # max prob over (0,2): the "active" confidence
    oof['conf'] = np.maximum(oof['prob_0'], oof['prob_2'])
    # margin over flat
    oof['margin'] = oof['conf'] - oof['prob_1']

    # per-sym breakdown
    for sym in range(5):
        g = oof[oof.sym == sym]
        n = len(g); n_act = int(g.active.sum())
        if n_act == 0:
            continue
        sum_pnl = float(g.pnl.sum())
        ppt = sum_pnl / n_act
        hit = float(g.correct.sum() / n_act)
        prof = float(g.profit.sum() / n_act)
        summary_rows.append({'horizon': h, 'sym': sym, 'n': n, 'n_active': n_act,
                             'active_rate': n_act/n, 'sum_pnl': sum_pnl,
                             'per_trade_pnl': ppt, 'hit_rate_label': hit,
                             'profit_rate': prof})
    # global
    n = len(oof); n_act = int(oof.active.sum())
    summary_rows.append({'horizon': h, 'sym': -1, 'n': n, 'n_active': n_act,
                         'active_rate': n_act/n, 'sum_pnl': float(oof.pnl.sum()),
                         'per_trade_pnl': float(oof.pnl.sum() / n_act) if n_act else 0,
                         'hit_rate_label': float(oof.correct.sum()/n_act) if n_act else 0,
                         'profit_rate': float(oof.profit.sum()/n_act) if n_act else 0})

    # Calibration: bin by confidence and measure hit rate / per-trade pnl
    active = oof[oof.active == 1].copy()
    bins = np.linspace(0.34, 0.95, 13)
    active['conf_bin'] = pd.cut(active['conf'], bins=bins)
    cb = active.groupby('conf_bin', observed=True).agg(
        n=('conf','size'),
        per_trade_pnl=('pnl','mean'),
        hit_rate=('correct','mean'),
        profit_rate=('profit','mean'),
    ).reset_index()
    cb['conf_mid'] = cb['conf_bin'].apply(lambda x: x.mid if hasattr(x,'mid') else None)
    calib_curves[h] = cb
    
    # Distribution of conf for profit vs loss
    prof_conf = active[active.pnl > 0]['conf'].values
    loss_conf = active[active.pnl <= 0]['conf'].values
    prob_dist_data[h] = (prof_conf, loss_conf)

sumdf = pd.DataFrame(summary_rows)
sumdf.to_csv(f'{OUT}/d5_sym_horizon_summary.csv', index=False)

print('=== Per-sym × horizon: per-trade pnl ===')
piv = sumdf[sumdf.sym>=0].pivot_table(index='sym', columns='horizon', values='per_trade_pnl')
print(piv.round(6))
print()
print('=== Per-sym × horizon: hit rate (correct label)  ===')
piv2 = sumdf[sumdf.sym>=0].pivot_table(index='sym', columns='horizon', values='hit_rate_label')
print(piv2.round(4))
print()
print('=== Per-sym × horizon: profit rate (pnl > 0) ===')
piv3 = sumdf[sumdf.sym>=0].pivot_table(index='sym', columns='horizon', values='profit_rate')
print(piv3.round(4))
print()
print('=== Per-sym × horizon: active rate ===')
piv4 = sumdf[sumdf.sym>=0].pivot_table(index='sym', columns='horizon', values='active_rate')
print(piv4.round(3))

# --- Plot 1: per-sym × horizon per-trade pnl heatmap ---
fig, axes = plt.subplots(1, 3, figsize=(20,5))
for ax, (m, t) in zip(axes, [('per_trade_pnl','per-trade PnL (after 2×fee)'),
                              ('hit_rate_label','hit rate (correct label)'),
                              ('profit_rate','profit rate (PnL > 0)')]):
    p = sumdf[sumdf.sym>=0].pivot_table(index='sym', columns='horizon', values=m)
    im = ax.imshow(p.values, aspect='auto', cmap='RdBu_r' if m=='per_trade_pnl' else 'viridis')
    ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels([f'h={h}' for h in HORIZONS])
    ax.set_yticks(range(5)); ax.set_yticklabels([f'sym {s}' for s in range(5)])
    ax.set_title(t)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            ax.text(j, i, f'{p.values[i,j]:.4f}' if m=='per_trade_pnl' else f'{p.values[i,j]:.3f}',
                    ha='center', va='center', fontsize=9, color='black')
    plt.colorbar(im, ax=ax, fraction=0.04)
plt.tight_layout(); plt.savefig(f'{OUT}/d5_per_sym_horizon.png', dpi=110, bbox_inches='tight'); plt.close()

# --- Plot 2: calibration curves per horizon ---
fig, axes = plt.subplots(1, 2, figsize=(14,5))
for h in HORIZONS:
    cb = calib_curves[h]
    axes[0].plot(cb['conf_mid'], cb['hit_rate'], marker='o', label=f'h={h}')
    axes[1].plot(cb['conf_mid'], cb['per_trade_pnl'], marker='o', label=f'h={h}')
axes[0].plot([0.34,0.95], [0.34,0.95], 'k--', alpha=0.5, label='perfect calibration')
axes[0].set_xlabel('predicted confidence (max prob_0/prob_2)')
axes[0].set_ylabel('hit rate (label correct)')
axes[0].set_title('Calibration: confidence vs label correctness')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].axhline(0, color='black', lw=0.5)
axes[1].axhline(-2*FEE, color='red', ls='--', label='-2×fee')
axes[1].set_xlabel('predicted confidence')
axes[1].set_ylabel('per-trade PnL (net of 2×fee)')
axes[1].set_title('Confidence vs per-trade PnL')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/d5_calibration.png', dpi=110, bbox_inches='tight'); plt.close()

# --- Plot 3: confidence distribution profit vs loss, per horizon ---
fig, axes = plt.subplots(1, 5, figsize=(20,4), sharey=True)
for ax, h in zip(axes, HORIZONS):
    prof, loss = prob_dist_data[h]
    bins = np.linspace(0.34, 0.95, 50)
    ax.hist(prof, bins=bins, alpha=0.55, density=True, label=f'profit ({len(prof)})', color='green')
    ax.hist(loss, bins=bins, alpha=0.55, density=True, label=f'loss ({len(loss)})', color='red')
    mp = np.mean(prof) if len(prof) else 0
    ml = np.mean(loss) if len(loss) else 0
    ax.set_title(f'h={h}\nμ_prof={mp:.3f}, μ_loss={ml:.3f}')
    ax.set_xlabel('confidence')
    ax.legend(fontsize=8)
fig.suptitle('Confidence distribution: profit vs loss trades (overlap = unable to discriminate)')
plt.tight_layout(); plt.savefig(f'{OUT}/d5_conf_dist_profit_loss.png', dpi=110, bbox_inches='tight'); plt.close()

print('Dim5 done.')
