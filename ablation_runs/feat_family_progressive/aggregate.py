"""Aggregate 30 NN results (L2..L7 x 5 seeds) into mean/std + write feat_family_table.tex + audit md."""
import glob, json, os
from pathlib import Path
import numpy as np

ROOT = Path('/root/projects/liangwenbei_workdir/ablation_runs/feat_family_progressive')
OUT_TEX = Path('/root/projects/liangwenbei_workdir/onepage_v2/feat_family_table.tex')
OUT_AUDIT = Path('/root/projects/liangwenbei_workdir/onepage_v2/feat_family_audit.md')

LEVELS = ['L2', 'L3', 'L4', 'L5', 'L6', 'L7']
SEEDS = [1, 2, 3, 4, 5]

LEVEL_LABEL = {
    'L1': '154 raw (= big-ab row 7)',
    'L2': '+ LOB 派生',
    'L3': '+ 多尺度 OFI',
    'L4': '+ 订单流强度',
    'L5': '+ 微结构波动',
    'L6': '+ 窗口统计',
    'L7': '+ 不对称性 (= 370-d, no drop)',
    'L8': '+ drop 11 (= 359-d) (= big-ab row 14)',
}

# Known anchors from prior 5-seed:
ANCHOR_NN_L1 = (26.05, 0.57)   # NN 154 + L2 + window-z
ANCHOR_NN_L8 = (34.34, 0.50)   # NN 359-d

def collect():
    rows = {}
    for L in LEVELS:
        tests = []
        feat_dim = None
        for s in SEEDS:
            f = ROOT / f"{L}_s{s}" / "results.json"
            if not f.exists():
                continue
            d = json.load(open(f))
            tests.append(d['test_pnl_h60_sym'])
            feat_dim = d['feat_dim']
        if tests:
            rows[L] = {
                'mean': float(np.mean(tests)),
                'std': float(np.std(tests, ddof=1)) if len(tests) > 1 else 0.0,
                'n': len(tests),
                'feat_dim': feat_dim,
                'all': tests,
            }
        else:
            rows[L] = None
    return rows

def main():
    rows = collect()
    print("=== Collected ===")
    for L in LEVELS:
        r = rows[L]
        if r is None:
            print(f"  {L}: NO DATA")
        else:
            print(f"  {L}: mean={r['mean']:.4f} std={r['std']:.4f} n={r['n']} d={r['feat_dim']} all={r['all']}")

    # Write standalone feat_family_table.tex
    tex_lines = [
        r"\begin{tabularx}{\linewidth}{@{}c >{\raggedright\arraybackslash}X r r@{}}",
        r"\toprule",
        r"\# & Feature set & NN Test PnL ($h{=}60$, 5-seed) & $\Delta$ vs prev \\",
        r"\midrule",
    ]
    prev_mean = ANCHOR_NN_L1[0]
    tex_lines.append(rf"L1 & 154 raw $+$ L2 $+$ window-$z$ (anchor) & $+{ANCHOR_NN_L1[0]:.2f}{{\scriptscriptstyle\,\pm {ANCHOR_NN_L1[1]:.2f}}}$ & --- \\")
    for L in LEVELS:
        r = rows[L]
        if r is None:
            tex_lines.append(rf"{L} & {LEVEL_LABEL[L]} & MISSING & --- \\")
            continue
        delta = r['mean'] - prev_mean
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        tex_lines.append(rf"{L} & {LEVEL_LABEL[L]} & $+{r['mean']:.2f}{{\scriptscriptstyle\,\pm {r['std']:.2f}}}$ & ${delta_str}$ \\")
        prev_mean = r['mean']
    # L8 anchor
    delta = ANCHOR_NN_L8[0] - prev_mean
    delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
    tex_lines.append(rf"L8 & {LEVEL_LABEL['L8']} & $+{ANCHOR_NN_L8[0]:.2f}{{\scriptscriptstyle\,\pm {ANCHOR_NN_L8[1]:.2f}}}$ & ${delta_str}$ \\")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabularx}")
    OUT_TEX.write_text("\n".join(tex_lines) + "\n")
    print(f"wrote {OUT_TEX}")

    # Audit md
    aud = ["# Feat-Family Progressive (NN only, 6 new levels, 5-seed)\n\n",
           "Train 0-79 / val 80-95 (ES + thr search) / test 96-119; window-z mandatory on input.\n\n",
           "## Means & Stds (test PnL, h=60)\n\n",
           "| Level | Feat dim | mean | std | seeds | Δ vs prev |\n",
           "|-------|----------|------|-----|-------|-----------|\n",
           f"| L1 (anchor) | 154 | {ANCHOR_NN_L1[0]:.2f} | {ANCHOR_NN_L1[1]:.2f} | 5 (prior) | — |\n"]
    prev = ANCHOR_NN_L1[0]
    for L in LEVELS:
        r = rows[L]
        if r is None:
            aud.append(f"| {L} | ? | MISSING | — | 0 | — |\n")
            continue
        d = r['mean'] - prev
        aud.append(f"| {L} | {r['feat_dim']} | {r['mean']:.2f} | {r['std']:.2f} | {r['n']} | {d:+.2f} |\n")
        prev = r['mean']
    aud.append(f"| L8 (anchor) | 359 | {ANCHOR_NN_L8[0]:.2f} | {ANCHOR_NN_L8[1]:.2f} | 5 (prior) | {ANCHOR_NN_L8[0]-prev:+.2f} |\n\n")
    aud.append("## Per-seed test PnL\n\n")
    for L in LEVELS:
        r = rows[L]
        if r is None:
            continue
        aud.append(f"- {L}: {r['all']}\n")
    OUT_AUDIT.write_text("".join(aud))
    print(f"wrote {OUT_AUDIT}")

    # Also dump a quick dict for tex insertion
    out = {L: {'mean': rows[L]['mean'], 'std': rows[L]['std']} if rows[L] else None for L in LEVELS}
    out['anchors'] = {'L1': list(ANCHOR_NN_L1), 'L8': list(ANCHOR_NN_L8)}
    json.dump(out, open(ROOT / 'aggregated.json', 'w'), indent=2)
    print(f"wrote {ROOT/'aggregated.json'}")

if __name__ == '__main__':
    main()
