#!/usr/bin/env python3
"""把因子工程 3 panel 拆成 3 张独立 PDF：Feature_family / Feature_top10 / Feature_corr."""

import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk_path)
_CJK = fm.FontProperties(fname=_cjk_path).get_name()
mpl.rcParams["font.sans-serif"] = [_CJK, "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

REPORT = Path("/root/projects/liangwenbei_workdir/factor_families_report")
HERE = Path(__file__).resolve().parent
summary = json.loads((REPORT / "summary.json").read_text())

family_labels = [
    "F1 LOB派生", "F2 多尺度OFI", "F3 订单流强度",
    "F4 微结构波动", "F5 窗口统计", "F6 不对称性",
]
dim_counts = [summary["family_counts"][k] for k in family_labels]
gain_pcts = [summary["family_gain_share_pct"][k] for k in family_labels]
total_dims = sum(dim_counts)
dim_pcts = [100.0 * c / total_dims for c in dim_counts]
corr = np.array(summary["block_corr"])
top10 = summary["top10_factors"]

palette = {
    "F1 LOB派生":   "#2878D2",
    "F2 多尺度OFI": "#A6CEE3",
    "F3 订单流强度": "#E07B39",
    "F4 微结构波动": "#9FD3B5",
    "F5 窗口统计":   "#C7B6D9",
    "F6 不对称性":   "#F4C26B",
}
colors = [palette[k] for k in family_labels]

raw_v_der = {
    "F1 LOB派生":    {"n_raw": 94, "n_der": 11, "gain_raw": 5.327, "gain_der": 0.196},
    "F2 多尺度OFI":  {"n_raw":  0, "n_der": 50, "gain_raw": 0.000, "gain_der": 0.750},
    "F3 订单流强度": {"n_raw": 18, "n_der": 27, "gain_raw": 3.958, "gain_der": 0.835},
    "F4 微结构波动": {"n_raw":  2, "n_der": 27, "gain_raw": 0.006, "gain_der": 1.020},
    "F5 窗口统计":   {"n_raw":  0, "n_der": 67, "gain_raw": 0.000, "gain_der": 0.708},
    "F6 不对称性":   {"n_raw": 40, "n_der": 34, "gain_raw": 0.396, "gain_der": 0.468},
}
total_gain = sum(v["gain_raw"] + v["gain_der"] for v in raw_v_der.values())
dim_raw_pcts  = [raw_v_der[k]["n_raw"]   / total_dims * 100 for k in family_labels]
dim_der_pcts  = [raw_v_der[k]["n_der"]   / total_dims * 100 for k in family_labels]
gain_raw_pcts = [raw_v_der[k]["gain_raw"] / total_gain * 100 for k in family_labels]
gain_der_pcts = [raw_v_der[k]["gain_der"] / total_gain * 100 for k in family_labels]


def style_axes(ax):
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#888")
    ax.spines["left"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, pad=1)


# ============ Sub 1: family bars ============
fig, ax = plt.subplots(figsize=(7.5/2.54, 5.0/2.54), dpi=200)
y = np.arange(len(family_labels))[::-1]
bar_h = 0.36
ax.barh(y + bar_h/2, dim_raw_pcts, height=bar_h, color=colors, alpha=0.65, edgecolor="none")
ax.barh(y + bar_h/2, dim_der_pcts, left=dim_raw_pcts, height=bar_h, color=colors, alpha=0.20, edgecolor="none")
ax.barh(y - bar_h/2, gain_raw_pcts, height=bar_h, color=colors, alpha=1.0, edgecolor="none")
ax.barh(y - bar_h/2, gain_der_pcts, left=gain_raw_pcts, height=bar_h, color=colors, alpha=0.40, edgecolor="none")

for i, yi in enumerate(y):
    fl = family_labels[i]
    nr, nd = raw_v_der[fl]["n_raw"], raw_v_der[fl]["n_der"]
    d_tot = dim_raw_pcts[i] + dim_der_pcts[i]
    g_tot = gain_raw_pcts[i] + gain_der_pcts[i]
    if i == 0:
        d_txt = f"维度 {nr}+{nd}"
        g_txt = f"重要性 {g_tot:.1f}%"
    else:
        d_txt = f"{nr}+{nd}"
        g_txt = f"{g_tot:.1f}%"
    ax.text(d_tot + 1.0, yi + bar_h/2, d_txt, va="center", fontsize=5.8, color="#333")
    ax.text(g_tot + 1.0, yi - bar_h/2, g_txt, va="center", fontsize=5.8, color="#333",
            fontweight="bold" if g_tot > 30 else "normal")
ax.set_yticks(y)
ax.set_yticklabels(family_labels, fontsize=6.5)
ax.set_xlim(0, max(max(gain_pcts), max(dim_pcts)) * 1.55)
ax.set_xticks([])
ax.set_title("家族结构（深=raw, 浅=派生）", fontsize=7.5, pad=2, fontweight="bold")
ax.set_xlabel("上 = 维度数(raw+派生) / 下 = 重要性(%)", fontsize=4.8, color="#555", labelpad=2)
style_axes(ax)
fig.tight_layout(pad=0.4)
fig.savefig(HERE / "Feature_family.pdf", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)


# ============ Sub 2: top10 factors ============
fig, ax = plt.subplots(figsize=(6.0/2.54, 5.0/2.54), dpi=200)
names = [t["name"] for t in top10]
gains = [t["gain"] for t in top10]
fams = [t["family"] for t in top10]
fcolors = [palette[f] for f in fams]
yy = np.arange(len(names))[::-1]
ax.barh(yy, gains, color=fcolors, edgecolor="none", height=0.74)
for i, (yi, g) in enumerate(zip(yy, gains)):
    ax.text(g + 0.012, yi, f"{g:.1f}", va="center", ha="left", fontsize=5.8, color="#333")
ax.set_xlim(0, max(gains) * 1.20)
ax.set_yticks(yy)
ax.set_yticklabels(names, fontsize=6.0, family="monospace")
ax.set_xticks([])
ax.set_title("Top-10 因子（按家族着色）", fontsize=7.5, pad=2, fontweight="bold")
ax.text(0.5, -0.05, "F3 × 4 + F1 × 6 = Top10 全部",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.7,
        color="#333", style="italic")
style_axes(ax)
fig.tight_layout(pad=0.4)
fig.savefig(HERE / "Feature_top10.pdf", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)


# ============ Sub 3: corr heatmap ============
fig, ax = plt.subplots(figsize=(6.0/2.54, 5.0/2.54), dpi=200)
cmap = mpl.colormaps["Blues"]
im = ax.imshow(corr, cmap=cmap, vmin=0, vmax=corr.max())
for i in range(6):
    for j in range(6):
        v = corr[i, j]
        col = "white" if v > corr.max() * 0.55 else "#222"
        ax.text(j, i, f".{int(round(v*100)):02d}", ha="center", va="center", fontsize=5.4, color=col)
fam_short = [f"F{i+1}" for i in range(6)]
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(fam_short, fontsize=6.0)
ax.set_yticklabels(fam_short, fontsize=6.0)
ax.set_title("家族间 |corr|", fontsize=7.5, pad=2, fontweight="bold")
for sp in ax.spines.values():
    sp.set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig(HERE / "Feature_corr.pdf", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print(f"Saved: Feature_family.pdf / Feature_top10.pdf / Feature_corr.pdf in {HERE}")
