#!/usr/bin/env python3
"""
Waterfall chart of the key ablation increments from one_page_summary.tex
(reduced from rows 1->20 to the 8 most informative milestones for a 5-min talk).

Output: waterfall.pdf
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# CJK font registration
_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk)
_name = fm.FontProperties(fname=_cjk).get_name()
mpl.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Milestones (label, delta).  Start = NN raw + L2 (row 2, +0.52) baseline.
# Pick the 8 most impactful tricks for a single-page visual.
START = 0.52
# These are the headline rows from one_page_summary.tex Table-1.
# We collapse the few small +/- rows (F4, F5, KS drop, sign-log1p, LGB-only path,
# single-pair ensemble) into one bridging step to keep the chart legible while
# still summing to the actual ensemble SOTA 41.58.
steps = [
    ("起点\nNN raw + L2", None),                                    # initial bar (=0.52)
    ("+ window-z\n(单 trick 跃升)",            +22.88),  # ->23.40
    ("+ 多尺度 OFI (F2)",                     +2.57),    # ->25.97
    ("+ 订单流强度派生 (F3)",                  +3.59),    # ->29.56
    ("+ 不对称性派生 (F6)",                    +4.64),    # ->34.20
    ("+ 镜像数据增强",                         +2.41),    # ->36.61
    ("→ NN×LGB 异质\n(5+5) 集成",              +0.51),    # ->37.12
    ("+ ETUD 非对称阈值",                      +0.78),    # ->37.90
    ("+ SPO+ DFL\n(决策焦点学习)",              +3.54),    # ->41.44
    ("+ 50×50 HP cycling",                    +0.14),    # ->41.58
]
labels = [s[0] for s in steps]
deltas = [s[1] for s in steps]

# cumulative values
cum = [START]
for d in deltas[1:]:
    cum.append(cum[-1] + d)

n = len(labels)
x = np.arange(n)

fig, ax = plt.subplots(figsize=(13.5, 5.3))

bar_w = 0.62
# colors
c_start = "#3a6ea5"
c_pos   = "#4aa3df"
c_total = "#cf3a3a"
c_neg   = "#c79a3a"

# start bar
ax.bar(x[0], cum[0], color=c_start, width=bar_w, edgecolor="white")
ax.text(x[0], cum[0] + 0.6, f"+{cum[0]:.2f}", ha="center", va="bottom", fontsize=11, color="#222")

# delta bars (waterfall floats)
for i in range(1, n):
    d = deltas[i]
    bottom = cum[i-1] if d >= 0 else cum[i]
    height = abs(d)
    color = c_pos if d >= 0 else c_neg
    ax.bar(x[i], height, bottom=bottom, color=color, width=bar_w, edgecolor="white")
    sign = "+" if d >= 0 else "-"
    ax.text(x[i], bottom + height + 0.45, f"{sign}{abs(d):.2f}",
            ha="center", va="bottom", fontsize=11, color="#222")
    # also annotate cumulative below bar
    ax.text(x[i], -2.6, f"={cum[i]:.2f}", ha="center", va="top",
            fontsize=9.5, color="#555", fontstyle="italic")

# final "total" overlay bar
total_x = n
ax.bar(total_x, cum[-1], color=c_total, width=bar_w, edgecolor="white")
ax.text(total_x, cum[-1] + 0.6, f"{cum[-1]:.2f}", ha="center", va="bottom",
        fontsize=13, color=c_total, fontweight="bold")

# connecting dashed lines
for i in range(n - 1):
    ax.hlines(cum[i], x[i] + bar_w/2, x[i+1] - bar_w/2,
              colors="#888", linestyles="dashed", linewidth=0.8)
ax.hlines(cum[-1], n - 1 + bar_w/2, total_x - bar_w/2,
          colors="#888", linestyles="dashed", linewidth=0.8)

# labels
labels_total = labels + ["最终\nTest PnL"]
ax.set_xticks(list(x) + [total_x])
ax.set_xticklabels(labels_total, rotation=0, fontsize=10.2, linespacing=1.15)

ax.set_ylabel("Test PnL  (h = 60)", fontsize=12)
ax.set_title("关键 trick 增益 waterfall —— 起点 NN raw+L2 (0.52) → 集成最终 41.58",
             fontsize=14, color="#001E5A", pad=10)

ax.set_ylim(-4.5, 49)
ax.set_xlim(-0.55, total_x + 0.55)
ax.axhline(0, color="#bbb", linewidth=0.7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=10)

# legend manual
from matplotlib.patches import Patch
leg = [
    Patch(facecolor=c_start, label="起点"),
    Patch(facecolor=c_pos,   label="正增益"),
    Patch(facecolor=c_total, label="最终累计"),
]
ax.legend(handles=leg, loc="upper left", frameon=False, fontsize=10.5)

plt.tight_layout()
plt.savefig("/root/projects/liangwenbei_workdir/slides/waterfall.pdf",
            bbox_inches="tight", pad_inches=0.05)
print("waterfall.pdf written")
