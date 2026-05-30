"""Aggregate runs_v2/*/results.json → summary tables for REPORT_v2.md."""
from __future__ import annotations
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_v2"

def load_all():
    rows = []
    for d in sorted(RUNS.glob("*")):
        f = d / "results.json"
        if not f.exists():
            continue
        try:
            with open(f) as fh:
                r = json.load(fh)
            rows.append(r)
        except Exception as e:
            print(f"  failed to load {f}: {e}", file=sys.stderr)
    return rows

def fmt(x, w=8, p=4, sign=False):
    if x is None:
        return f"{'-':>{w}}"
    fmt_str = f"{{:{'+' if sign else ''}{w}.{p}f}}"
    try:
        return fmt_str.format(float(x))
    except Exception:
        return f"{str(x):>{w}}"

def main():
    rows = load_all()
    if not rows:
        print("no runs found in runs_v2/")
        return

    archs = sorted(set(r['arch'] for r in rows))
    print(f"Found {len(rows)} runs across {len(archs)} archs: {archs}\n")

    # Detailed table
    print("=" * 110)
    print(f"{'arch':<12}{'seed':>4}{'params':>10}{'best_step':>10}{'val_pnl':>12}{'test_pnl':>12}{'val_ic':>10}{'test_ic':>10}{'t_sec':>8}")
    print("-" * 110)
    for r in sorted(rows, key=lambda x: (x['arch'], x['seed'])):
        if r.get('status') != 'ok':
            print(f"{r['arch']:<12}{r['seed']:>4}  FAILED")
            continue
        print(f"{r['arch']:<12}{r['seed']:>4}{r['n_params']:>10}"
              f"{r['best_step']:>10}"
              f"{fmt(r['val_pnl'],12,4,True)}{fmt(r['test_pnl'],12,4,True)}"
              f"{fmt(r['val_ic'],10,5)}{fmt(r['test_ic'],10,5)}"
              f"{fmt(r['t_train_sec'],8,1)}")
    print("=" * 110)

    # Per-arch summary
    print(f"\nPer-arch summary:")
    print(f"{'arch':<12}{'n':>4}{'mean_val':>10}{'mean_test':>11}{'std_test':>10}{'best_test':>11}{'mean_best_step':>15}")
    print("-" * 75)
    for arch in archs:
        sub = [r for r in rows if r['arch'] == arch and r.get('status') == 'ok']
        if not sub:
            print(f"{arch:<12}{0:>4}   (no successful runs)")
            continue
        n = len(sub)
        import statistics
        mean_val   = statistics.mean(r['val_pnl']  for r in sub)
        mean_test  = statistics.mean(r['test_pnl'] for r in sub)
        std_test   = statistics.pstdev(r['test_pnl'] for r in sub) if n > 1 else 0.0
        best_test  = max(r['test_pnl'] for r in sub)
        mean_step  = statistics.mean(r['best_step'] for r in sub)
        print(f"{arch:<12}{n:>4}{fmt(mean_val,10,3,True)}{fmt(mean_test,11,3,True)}"
              f"{fmt(std_test,10,3)}{fmt(best_test,11,3,True)}{mean_step:>15.0f}")

    # Per-sym test PnL for each run
    print(f"\nPer-sym test PnL:")
    print(f"{'arch':<12}{'seed':>4}  {'sym0':>8}{'sym1':>8}{'sym2':>8}{'sym3':>8}{'sym4':>8}{'total':>9}")
    for r in sorted(rows, key=lambda x: (x['arch'], x['seed'])):
        if r.get('status') != 'ok':
            continue
        sps = r['test_sym_pnl']
        print(f"{r['arch']:<12}{r['seed']:>4}  "
              f"{fmt(sps.get('sym0'),8,2,True)}{fmt(sps.get('sym1'),8,2,True)}"
              f"{fmt(sps.get('sym2'),8,2,True)}{fmt(sps.get('sym3'),8,2,True)}"
              f"{fmt(sps.get('sym4'),8,2,True)}{fmt(r['test_pnl'],9,2,True)}")

    # Overfit diagnostic: best_step vs total_steps, val@best vs val@final
    print(f"\nOverfit diagnostic (best_step / final_step val_pnl / drop):")
    print(f"{'arch':<12}{'seed':>4}  {'best_step':>10}{'total_step':>11}{'val_best':>10}{'val_final':>10}{'drop':>8}")
    for r in sorted(rows, key=lambda x: (x['arch'], x['seed'])):
        if r.get('status') != 'ok':
            continue
        vb = r['val_pnl']; vf = r.get('final_step_val_pnl')
        drop = (vb - vf) if vf is not None else None
        print(f"{r['arch']:<12}{r['seed']:>4}  "
              f"{r['best_step']:>10}{r['total_steps']:>11}"
              f"{fmt(vb,10,3,True)}{fmt(vf,10,3,True)}{fmt(drop,8,3,True)}")

if __name__ == "__main__":
    main()
