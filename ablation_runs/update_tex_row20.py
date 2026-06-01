"""Add row 20 (50+50 NN(SPO) + LGB asym EV) to big ablation tex. Re-mark SOTA if row 20 beats row 19."""
from __future__ import annotations
import json, re, sys
from pathlib import Path

TEX = Path("/root/projects/liangwenbei_workdir/onepage_v2/one_page_summary.tex")
ENS_ASYM = Path("/root/projects/liangwenbei_workdir/ablation_runs/big50_ensemble/asym/results.json")
ENS_SYM = Path("/root/projects/liangwenbei_workdir/ablation_runs/big50_ensemble/sym/results.json")

asym = json.loads(ENS_ASYM.read_text())
sym = json.loads(ENS_SYM.read_text())
r20_test = float(asym["decision"]["test_pnl_h60"])
r20_val = float(asym["decision"]["val_pnl_h60"])
r20_sym_test = float(sym["decision"]["test_pnl_h60"])
r19_test = 37.73  # current row 19 in tex

delta_vs_5plus5 = r20_test - r19_test
print(f"row 20 test_pnl asym = {r20_test:.4f}  val={r20_val:.4f}")
print(f"row 20 test_pnl sym  = {r20_sym_test:.4f}")
print(f"delta vs row 19 = {delta_vs_5plus5:+.4f}")

tex = TEX.read_text()

# Determine SOTA placement
sota_threshold = 0.30  # if r20 beats r19 by > 0.3, mark r20 SOTA; else keep r19 SOTA
new_sota_row = 20 if delta_vs_5plus5 > sota_threshold else 19

# Format strings
sign20 = "+" if r20_test >= 0 else ""
sign_d = "+" if delta_vs_5plus5 >= 0 else ""

if new_sota_row == 20:
    # Remove (SOTA) tag and bold from row 19, add to row 20
    tex = re.sub(
        r"19 & \(5\$\+\$5\) NN\(SPO\) \$\+\$ LGB, asym EV\s+& \$\\mathbf\{\+37\.73\}\$ \(SOTA\)\s+& \$\+1\.54\$ \\\\",
        r"19 & (5$+$5) NN(SPO) $+$ LGB, asym EV                                         & $+37.73$ & $+1.54$ \\\\",
        tex,
    )
    row20_line = (
        f"20 & (50$+$50) NN(SPO) $+$ LGB, asym EV (\\textbf{{final SOTA recipe}})        "
        f"& $\\mathbf{{{sign20}{r20_test:.2f}}}$ (SOTA) & ${sign_d}{delta_vs_5plus5:.2f}$ \\\\"
    )
else:
    # Keep row 19 as SOTA; row 20 marked as ~public submission
    row20_line = (
        f"20 & (50$+$50) NN(SPO) $+$ LGB, asym EV ($\\approx$ project public submission) "
        f"& ${sign20}{r20_test:.2f}$ & ${sign_d}{delta_vs_5plus5:.2f}$ \\\\"
    )

# Insert row 20 before \bottomrule (after row 19)
old = "19 & (5$+$5) NN(SPO) $+$ LGB, asym EV                                         & $+37.73$ & $+1.54$ \\\\\n\\bottomrule"
old_with_sota = "19 & (5$+$5) NN(SPO) $+$ LGB, asym EV                                         & $\\mathbf{+37.73}$ (SOTA) & $+1.54$ \\\\\n\\bottomrule"
new = old.replace("\\bottomrule", row20_line + "\n\\bottomrule") if old in tex else None
new_alt = old_with_sota.replace("\\bottomrule", row20_line + "\n\\bottomrule") if old_with_sota in tex else None

if new is not None:
    tex = tex.replace(old, new)
elif new_alt is not None:
    tex = tex.replace(old_with_sota, new_alt)
else:
    print("[FAIL] could not find row 19 / bottomrule anchor")
    sys.exit(1)

# Update caption to reference row 20
# Find the big_ablation caption text and add a 50+50 mention.
old_cap_tail = (
    r"SPO 决策焦点 $+1.54$，累计 $\mathbf{+37.73}$（SOTA）。LGB 用项目 \texttt{HP\_CONFIGS[0]}。}"
)
if new_sota_row == 20:
    new_cap_tail = (
        r"SPO 决策焦点 $+1.54$，累计 $+37.73$。"
        f"\\textbf{{Row 20}}: 扩 (50$+$50) multi-seed + 5 HP$\\_$CONFIGS 跨族 → "
        f"\\textbf{{${sign20}{r20_test:.2f}$（V4 walk-forward 本队最终公榜 recipe；本表新 SOTA, ${sign_d}{delta_vs_5plus5:.2f}$ vs row 19）}}。"
        r"LGB 5 HP\_CONFIGS $\times$ 10 seed = 50；NN lr/dropout/batch 3$\times$4$\times$3 cycle $\times$ 10 seed = 50。}"
    )
else:
    new_cap_tail = (
        r"SPO 决策焦点 $+1.54$，累计 $\mathbf{+37.73}$（SOTA）。"
        f"\\textbf{{Row 20}}: (50$+$50) NN(SPO) + LGB asym EV "
        f"= ${sign20}{r20_test:.2f}$（${sign_d}{delta_vs_5plus5:.2f}$ vs row 19；$\\approx$ 本队公榜提交 V4-local repro）。"
        r"LGB 5 HP\_CONFIGS $\times$ 10 seed = 50；NN HP cycle $\times$ 10 seed = 50。}"
    )

if old_cap_tail in tex:
    tex = tex.replace(old_cap_tail, new_cap_tail)
    print("caption updated")
else:
    print("[WARN] caption tail not found verbatim — caption not updated automatically")

TEX.write_text(tex)
print(f"tex saved to {TEX}")
print(f"row 20: test={r20_test:.4f} delta={delta_vs_5plus5:+.4f}  SOTA row = {new_sota_row}")
