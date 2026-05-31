"""Aggregate v7 5-seed results vs v6_pairwise SOTA baseline."""
import json
import glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def load_seeds(pattern):
    rs = []
    for p in sorted(glob.glob(str(HERE / pattern))):
        try:
            r = json.load(open(p))
            r['_path'] = p
            rs.append(r)
        except Exception as e:
            print(f'skip {p}: {e}')
    return rs


def summarize(rs, label):
    if not rs:
        print(f'{label}: no runs')
        return
    test = np.array([r['test_pnl']  for r in rs])
    val  = np.array([r['val_pnl']   for r in rs])
    ic_t = np.array([r['test_ic']   for r in rs])
    bs   = np.array([r['best_step'] for r in rs])
    print(f'{label}: n_seeds={len(rs)}')
    print(f'  test_pnl: mean={test.mean():+.4f}  std={test.std():.4f}  '
          f'min={test.min():+.4f}  max={test.max():+.4f}')
    print(f'  val_pnl : mean={val.mean():+.4f}  std={val.std():.4f}')
    print(f'  test_ic : mean={ic_t.mean():+.5f} std={ic_t.std():.5f}')
    print(f'  best_step: mean={bs.mean():.0f}  range=[{bs.min()},{bs.max()}]')
    print(f'  per-seed test_pnl: {[f"{t:+.4f}" for t in test]}')


def main():
    v6 = load_seeds('runs_v6/v6_pairwise_pruned259_s*/results.json')
    v7 = load_seeds('runs_v7/v6_pairwise_v7inter40_s*/results.json')

    summarize(v6, 'v6_pairwise SOTA (259d)')
    print()
    summarize(v7, 'v7_pairwise + 40 inter feats (299d)')

    if v6 and v7:
        v6t = np.array([r['test_pnl'] for r in v6])
        v7t = np.array([r['test_pnl'] for r in v7])
        delta = v7t.mean() - v6t.mean()
        std_pool = np.sqrt(v6t.var() + v7t.var())
        print()
        print(f'Δ test_pnl (v7 - v6) = {delta:+.4f}')
        print(f'pooled std = {std_pool:.4f}')
        print(f'z-score    = {delta / max(std_pool, 1e-6):+.2f}')
        verdict = 'NEW SOTA 🚀' if delta > 0.30 else 'marginal/noise' if delta > -0.3 else 'REGRESSION'
        print(f'verdict    : {verdict}')


if __name__ == '__main__':
    main()
