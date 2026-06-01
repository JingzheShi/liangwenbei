#!/usr/bin/env python3
"""
Compact 3-panel feature engineering figure for onepage_v2/one_page_summary.tex.
Output: Feature_Img.pdf (vector, ~9.8cm x 4.5cm, single \\linewidth)

Panels:
  P1 (left)   : 6 families - paired horizontal bars (#dim & gain%) with values
  P2 (middle) : Top-10 factors horizontal bars, colored by family
  P3 (right)  : 6x6 family block-corr heatmap
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------- font config ----------
import matplotlib.font_manager as fm

# explicit CJK font registration (ttc not auto-picked up by mpl)
_cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk_path)
_cjk_prop = fm.FontProperties(fname=_cjk_path)
_CJK_NAME = _cjk_prop.get_name()
mpl.rcParams["font.sans-serif"] = [_CJK_NAME, "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType
mpl.rcParams["ps.fonttype"] = 42

# ---------- load data ----------
REPORT = Path("/root/projects/liangwenbei_workdir/factor_families_report")
summary = json.loads((REPORT / "summary.json").read_text())

family_labels = [
    "F1 LOB派生", "F2 多尺度OFI", "F3 订单流强度",
    "F4 微结构波动", "F5 窗口统计", "F6 不对称性",
]
# compact single-line labels for left panel
short_labels = [
    "F1 LOB派生", "F2 多尺度OFI", "F3 订单流强度",
    "F4 微结构波动", "F5 窗口统计", "F6 不对称性",
]

dim_counts = [summary["family_counts"][k] for k in family_labels]
gain_pcts = [summary["family_gain_share_pct"][k] for k in family_labels]
total_dims = sum(dim_counts)
dim_pcts = [100.0 * c / total_dims for c in dim_counts]

corr = np.array(summary["block_corr"])
top10 = summary["top10_factors"]

# raw / 派生 split per family (from factor_families_report/results.json v3)
raw_v_der = {
    "F1 LOB派生":    {"n_raw": 94, "n_der": 11, "gain_raw": 5.327, "gain_der": 0.196},
    "F2 多尺度OFI":  {"n_raw":  0, "n_der": 50, "gain_raw": 0.000, "gain_der": 0.750},
    "F3 订单流强度": {"n_raw": 18, "n_der": 27, "gain_raw": 3.958, "gain_der": 0.835},
    "F4 微结构波动": {"n_raw":  2, "n_der": 27, "gain_raw": 0.006, "gain_der": 1.020},
    "F5 窗口统计":   {"n_raw":  0, "n_der": 67, "gain_raw": 0.000, "gain_der": 0.708},
    "F6 不对称性":   {"n_raw": 40, "n_der": 34, "gain_raw": 0.396, "gain_der": 0.468},
}
total_gain = sum(v["gain_raw"] + v["gain_der"] for v in raw_v_der.values())  # ≈ 13.664
dim_raw_pcts  = [raw_v_der[k]["n_raw"] / total_dims * 100 for k in family_labels]
dim_der_pcts  = [raw_v_der[k]["n_der"] / total_dims * 100 for k in family_labels]
gain_raw_pcts = [raw_v_der[k]["gain_raw"] / total_gain * 100 for k in family_labels]
gain_der_pcts = [raw_v_der[k]["gain_der"] / total_gain * 100 for k in family_labels]

# ---------- palette: F1 = onepage lblue, others tone-coordinated ----------
# lblue is RGB(40,120,210); use a custom 6-family palette
palette = {
    "F1 LOB派生":     "#2878D2",   # onepage lblue (primary, 40.4%)
    "F2 多尺度OFI":   "#A6CEE3",   # light blue
    "F3 订单流强度":   "#E07B39",   # accent orange (secondary, 35.1%)
    "F4 微结构波动":   "#9FD3B5",   # mint
    "F5 窗口统计":     "#C7B6D9",   # soft purple
    "F6 不对称性":     "#F4C26B",   # warm tan
}
colors = [palette[k] for k in family_labels]

# ---------- figure ----------
# 9.8 cm wide x 5.0 cm tall  -> inches
W_IN, H_IN = 9.8 / 2.54, 5.0 / 2.54
fig = plt.figure(figsize=(W_IN, H_IN), dpi=300)

gs = fig.add_gridspec(
    1, 3,
    width_ratios=[1.10, 0.85, 1.90],
    left=0.13, right=0.99, top=0.86, bottom=0.16,
    wspace=1.05,
)

FS_TITLE = 7.0
FS_LABEL = 6.3
FS_VAL = 5.8
FS_TICK = 6.0

# ============ Panel 1: paired bars (#dim & gain%) ============
ax1 = fig.add_subplot(gs[0, 0])
y = np.arange(len(family_labels))[::-1]  # F1 at top
bar_h = 0.36
# dim%: stacked [raw (alpha 0.65) | 派生 (alpha 0.20)]
ax1.barh(y + bar_h / 2, dim_raw_pcts, height=bar_h,
         color=colors, alpha=0.65, edgecolor="none")
ax1.barh(y + bar_h / 2, dim_der_pcts, left=dim_raw_pcts, height=bar_h,
         color=colors, alpha=0.20, edgecolor="none")
# gain%: stacked [raw (alpha 1.0) | 派生 (alpha 0.40)]
ax1.barh(y - bar_h / 2, gain_raw_pcts, height=bar_h,
         color=colors, alpha=1.0, edgecolor="none")
ax1.barh(y - bar_h / 2, gain_der_pcts, left=gain_raw_pcts, height=bar_h,
         color=colors, alpha=0.40, edgecolor="none")
# annotate "n_raw+n_der" at end of dim bar; "gain%" at end of gain bar
for i, yi in enumerate(y):
    fl = family_labels[i]
    nr, nd = raw_v_der[fl]["n_raw"], raw_v_der[fl]["n_der"]
    dim_tot = dim_raw_pcts[i] + dim_der_pcts[i]
    g_tot = gain_raw_pcts[i] + gain_der_pcts[i]
    # F1 row (top): prefix with label so encoding is explicit; F2-F6 stay compact
    if i == 0:
        dim_txt = f"维度 {nr}+{nd}"
        g_txt = f"重要性 {g_tot:.1f}%"
    else:
        dim_txt = f"{nr}+{nd}"
        g_txt = f"{g_tot:.1f}%"
    ax1.text(dim_tot + 1.0, yi + bar_h / 2, dim_txt,
             va="center", fontsize=FS_VAL, color="#333333")
    ax1.text(g_tot + 1.0, yi - bar_h / 2, g_txt,
             va="center", fontsize=FS_VAL, color="#333333",
             fontweight="bold" if g_tot > 30 else "normal")
ax1.set_yticks(y)
ax1.set_yticklabels(short_labels, fontsize=FS_TICK)
ax1.set_xlim(0, max(max(gain_pcts), max(dim_pcts)) * 1.55)
ax1.set_xticks([])
ax1.set_xlabel("上行 = 维度数 (raw+派生)  /  下行 = 重要性 (%)",
               fontsize=5.0, color="#555555", labelpad=2)
ax1.set_title("家族结构\n(深=raw, 浅=派生)", fontsize=FS_TITLE, pad=1.5, fontweight="bold")
for sp in ["top", "right", "bottom"]:
    ax1.spines[sp].set_visible(False)
ax1.spines["left"].set_color("#888888")
ax1.spines["left"].set_linewidth(0.6)
ax1.tick_params(axis="y", length=0, pad=1)
# tiny legend
from matplotlib.patches import Patch
# no explicit legend; raw/派生 encoding is described in the title
# (depth gradient is self-evident in the bars)

# ============ Panel 2: Top-10 factors ============
ax2 = fig.add_subplot(gs[0, 1])
names = [t["name"] for t in top10]
gains = [t["gain"] for t in top10]
fams = [t["family"] for t in top10]
fcolors = [palette[f] for f in fams]
yy = np.arange(len(names))[::-1]
ax2.barh(yy, gains, color=fcolors, edgecolor="none", height=0.74)
for i, (yi, g) in enumerate(zip(yy, gains)):
    ax2.text(g + 0.012, yi, f"{g:.1f}", va="center", ha="left",
             fontsize=FS_VAL, color="#333333")
ax2.set_xlim(0, max(gains) * 1.22)
ax2.set_yticks(yy)
ax2.set_yticklabels(names, fontsize=FS_VAL, family="monospace")
ax2.set_xticks([])
ax2.set_title("Top-10 因子\n(按家族着色)", fontsize=FS_TITLE, pad=1.5, fontweight="bold")
for sp in ["top", "right", "bottom"]:
    ax2.spines[sp].set_visible(False)
ax2.spines["left"].set_color("#888888")
ax2.spines["left"].set_linewidth(0.6)
ax2.tick_params(axis="y", length=0, pad=1)
# subtle annotation: F3 + F1 dominate top10
f3 = sum(1 for f in fams if f == "F3 订单流强度")
f1 = sum(1 for f in fams if f == "F1 LOB派生")
ax2.text(0.5, -0.045, f"F3×{f3} + F1×{f1} = Top10 全部",
         transform=ax2.transAxes, ha="center", va="top",
         fontsize=5.7, color="#333333", style="italic")

# ============ Panel 3: 6x6 family corr heatmap ============
ax3 = fig.add_subplot(gs[0, 2])
# use a clean sequential map; emphasize F5 dim row
cmap = mpl.colormaps["Blues"]
vmax = float(corr.max())
im = ax3.imshow(corr, cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal")
# annotate values (compact: no leading 0, 2 chars max)
def _fmt(v):
    s = f"{v:.2f}"
    return s[1:] if s.startswith("0.") else s  # "0.15" -> ".15"

for i in range(6):
    for j in range(6):
        v = corr[i, j]
        txt_color = "white" if v > vmax * 0.55 else "#222222"
        ax3.text(j, i, _fmt(v), ha="center", va="center",
                 fontsize=5.6, color=txt_color, family="monospace")
ax3.set_xticks(range(6))
ax3.set_yticks(range(6))
ax3.set_xticklabels([f"F{i+1}" for i in range(6)], fontsize=FS_TICK)
ax3.set_yticklabels([f"F{i+1}" for i in range(6)], fontsize=FS_TICK)
ax3.tick_params(axis="both", length=0, pad=1)
ax3.set_title("家族间 |corr|", fontsize=FS_TITLE, pad=2.5, fontweight="bold")
for sp in ax3.spines.values():
    sp.set_visible(False)

# ---------- save ----------
out_pdf = Path("/root/projects/liangwenbei_workdir/onepage_v2/Feature_Img.pdf")
fig.savefig(out_pdf, format="pdf", bbox_inches=None, pad_inches=0.01)
print(f"Saved: {out_pdf}  ({out_pdf.stat().st_size/1024:.1f} KB)")
plt.close(fig)
