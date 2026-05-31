"""Aggregate v10 LGBM results across seeds."""
import json
from pathlib import Path
import numpy as np

ROOT = Path('runs_v10_lgbm')

results = {}
for cfg in ['A', 'B', 'C', 'D']:
    pnls = []
    ics  = []
    iters = []
    for seed in range(5):
        p = ROOT / f'lgbm_{cfg}_s{seed}' / 'results.json'
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        pnls.append(d['metrics']['test_pnl'])
        ics.append(d['metrics']['test_ic'])
        iters.append(d['best_iter'])
    if pnls:
        results[cfg] = dict(
            n=len(pnls), pnls=pnls, ics=ics, iters=iters,
            mean_pnl=np.mean(pnls), std_pnl=np.std(pnls),
            mean_ic=np.mean(ics), std_ic=np.std(ics),
            mean_iter=np.mean(iters),
        )

print('Config | N | test_pnl (mean±std) | test_ic (mean±std) | best_iter')
print('-------|---|---------------------|--------------------|-----------')
for cfg, r in sorted(results.items()):
    print(f"  {cfg}  | {r['n']} | {r['mean_pnl']:+.2f} ± {r['std_pnl']:.2f}   | {r['mean_ic']:.4f} ± {r['std_ic']:.4f}   | {r['mean_iter']:.1f}")

print('\nNN SOTA:        +39.65 ± 0.92 (Pairformer v9, 60 features)')
print('LGB baseline:   +26.58 ± 0.96 (359d schemeP, early stop on val)')
