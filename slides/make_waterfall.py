#!/usr/bin/env python3
"""
Waterfall chart of the key ablation increments, GROUPED BY DOMAIN (v2):

  Feature domain  (blue):    window-z, 6-family deltas, mirror aug, sign-log1p
  Model domain    (orange):  NN->LGB switch, L2 loss, M7 retrain
  Ensemble & Exec (green):   5+5 ensemble, ETUD, SPO+ DFL, 50+50 HP cycling

Visual aids:
  - background tinted band per domain
  - per-bar PnL delta above the bar (no text overlap: y-offset + small fontsize)
  - cumulative value italicized below x-axis
  - final SOTA bar in red overlay
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.patches import Patch

# CJK font
_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk)
_name = fm.FontProperties(fname=_cjk).get_name()
mpl.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Domain color palette
C_FEAT = "#3b82c4"
C_MODEL = "#e89045"
C_ENS = "#4ea36d"
C_START = "#7a7a7a"
C_TOTAL = "#cf3a3a"

# Light background tints
BG_FEAT = "#e8f1f9"
BG_MODEL = "#fdefe0"
BG_ENS = "#e6f3eb"

START = 0.52
steps = [
    # label,             delta,   domain ('start','feat','model','ens')
    ("起点\nNN raw + L2",          None,    "start"),    # +0.52
    ("+ window-$z$\n(OOD)",       +22.88,  "feat"),     # ->23.40
    ("+ F1 LOB 派生",              -0.07,   "feat"),     # ->23.33
    ("+ F2 多尺度 OFI",            +2.57,   "feat"),     # ->25.90
    ("+ F3 订单流",                +3.59,   "feat"),     # ->29.49
    ("+ F4 微结构波动",            -0.51,   "feat"),     # ->28.98
    ("+ F5 窗口统计",              +0.70,   "feat"),     # ->29.68
    ("+ F6 不对称性",              +4.64,   "feat"),     # ->34.32
    ("+ 镜像增强",                 +0.83,   "feat"),     # ->35.15
    ("+ NN×LGB\n(5+5) 集成",       +1.97,   "ens"),      # ->37.12
    ("+ ETUD 阈值",                +0.78,   "ens"),      # ->37.90
    ("+ SPO+ DFL",                +3.54,   "ens"),      # ->41.44
    ("+ 50×50 HP cycling",        +0.14,   "ens"),      # ->41.58
]
labels = [s[0] for s in steps]
deltas = [s[1] for s in steps]
domains = [s[2] for s in steps]

cum = [START]
for d in deltas[1:]:
    cum.append(cum[-1] + d)

n = len(labels)
x = np.arange(n)

fig, ax = plt.subplots(figsize=(13.6, 5.4))

bar_w = 0.62

# ---- Domain background bands ----
# feature domain: indices 1..5
# (no model-domain step in this 9-step view; the NN->LGB joint switch / L2-loss / M7
#  retrain are *embedded* in the (5+5) ensemble step and inside SPO+ phase-2 finetune,
#  so we visualize 2 domains explicitly + Ensemble&Execution for clarity)
# ens domain: 6..9
def _band(x_left, x_right, color, label):
    ax.axvspan(x_left, x_right, color=color, alpha=0.55, zorder=0)

_band(-0.5,                       0 + bar_w/2 + 0.05, "#f4f4f4", "start")
_band(0 + bar_w/2 + 0.05,         8 + bar_w/2 + 0.05, BG_FEAT, "feature")
_band(8 + bar_w/2 + 0.05,        12 + bar_w/2 + 0.05, BG_ENS,  "ens")
_band(12 + bar_w/2 + 0.05,       13.6,                "#f4f4f4", "total")

# Domain top labels
ax.text((0 + bar_w/2 + 0.05 + 8 + bar_w/2 + 0.05) / 2, 47.2,
        "Feature 域", ha="center", va="center",
        fontsize=11.5, color=C_FEAT, fontweight="bold")
ax.text((8 + bar_w/2 + 0.05 + 12 + bar_w/2 + 0.05) / 2, 47.2,
        "Ensemble & Execution 域", ha="center", va="center",
        fontsize=11.5, color=C_ENS, fontweight="bold")

# Vertical separators
ax.axvline(0 + bar_w/2 + 0.05,  color="#888", lw=0.5, linestyle="--", zorder=1)
ax.axvline(8 + bar_w/2 + 0.05,  color="#888", lw=0.8, linestyle="--", zorder=1)
ax.axvline(12 + bar_w/2 + 0.05, color="#888", lw=0.8, linestyle="--", zorder=1)

# ---- Bars ----
# start
ax.bar(x[0], cum[0], color=C_START, width=bar_w, edgecolor="white", zorder=3)
ax.text(x[0], cum[0] + 0.7, f"+{cum[0]:.2f}",
        ha="center", va="bottom", fontsize=10, color="#222", zorder=4)

# delta bars
dom_color = {"feat": C_FEAT, "model": C_MODEL, "ens": C_ENS, "start": C_START}
# stagger annotation y-offset to prevent overlap on small deltas
y_offset_extra = {i: 0.0 for i in range(len(deltas))}
# small-delta annotations stagger
for i in (2, 5, 12):  # F1 -0.07, F4 -0.51, 50x50 +0.14
    y_offset_extra[i] = 1.4
for i in range(1, n):
    d = deltas[i]
    bottom = cum[i - 1] if d >= 0 else cum[i]
    height = abs(d)
    color = "#2a2a2a" if d < 0 else dom_color[domains[i]]
    ax.bar(x[i], height, bottom=bottom, color=color, width=bar_w,
           edgecolor="white", zorder=3)
    sign = "+" if d >= 0 else "-"
    ax.text(x[i], bottom + height + 0.55 + y_offset_extra[i],
            f"{sign}{abs(d):.2f}",
            ha="center", va="bottom", fontsize=9.5, color="#222", zorder=4)
    # cumulative below
    ax.text(x[i], -2.7, f"={cum[i]:.2f}", ha="center", va="top",
            fontsize=8.6, color="#555", fontstyle="italic", zorder=4)

# Final total overlay bar
total_x = n
ax.bar(total_x, cum[-1], color=C_TOTAL, width=bar_w,
       edgecolor="white", zorder=3)
ax.text(total_x, cum[-1] + 0.7, f"{cum[-1]:.2f}",
        ha="center", va="bottom", fontsize=12, color=C_TOTAL,
        fontweight="bold", zorder=4)

# Connecting dashed cum lines
for i in range(n - 1):
    ax.hlines(cum[i], x[i] + bar_w/2, x[i + 1] - bar_w/2,
              colors="#888", linestyles="dashed", linewidth=0.7, zorder=2)
ax.hlines(cum[-1], n - 1 + bar_w/2, total_x - bar_w/2,
          colors="#888", linestyles="dashed", linewidth=0.7, zorder=2)

# Labels
labels_total = labels + ["最终\nTest PnL"]
ax.set_xticks(list(x) + [total_x])
ax.set_xticklabels(labels_total, rotation=22, ha="right", fontsize=7.8, linespacing=1.12)

ax.set_ylabel("Test PnL  (h = 60)", fontsize=11.5)
ax.set_title("关键 trick 增益 waterfall —— 按领域上色：Feature → Ensemble & Execution",
             fontsize=13, color="#001E5A", pad=10)

ax.set_ylim(-4.5, 50)
ax.set_xlim(-0.55, total_x + 0.55)
ax.axhline(0, color="#bbb", linewidth=0.7, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=9.5)

# Legend
leg = [
    Patch(facecolor=C_START, label="起点"),
    Patch(facecolor=C_FEAT,  label="Feature 域"),
    Patch(facecolor=C_ENS,   label="Ensemble & Execution 域"),
    Patch(facecolor=C_TOTAL, label="最终累计"),
]
ax.legend(handles=leg, loc="upper left", frameon=False, fontsize=9.5,
          ncol=4, bbox_to_anchor=(0.0, 1.02))

plt.tight_layout()
plt.savefig("/root/projects/liangwenbei_workdir/slides/waterfall.pdf",
            bbox_inches="tight", pad_inches=0.05)
print("waterfall.pdf written (color-grouped)")
