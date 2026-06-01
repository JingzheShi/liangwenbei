#!/usr/bin/env python3
"""2 张额外的因子工程子图：
  (1) Feature_ablation.pdf —— 每家族 ablation 增益 Δ PnL（从 Table 1 rows 4-9 拆出）
  (2) Feature_rawder.pdf  —— 每家族内 raw vs derived gain 占比
"""
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

HERE = Path(__file__).resolve().parent

family_labels = ["F1 LOB派生", "F2 OFI", "F3 订单流", "F4 波动", "F5 窗口统计", "F6 不对称性"]
palette = ["#2878D2", "#A6CEE3", "#E07B39", "#9FD3B5", "#C7B6D9", "#F4C26B"]

# ===== (1) ablation 各家族 ΔPnL（来自 onepage Table 1 rows 4-9）=====
# row 3 (baseline NN 154 raw + L2 + window-z) = +23.40
# row 4 +LOB 派生 (+11 F1 派生): +23.33 → Δ = -0.07
# row 5 +多尺度 OFI (+50 F2 整族): +25.90 → Δ = +2.57
# row 6 +订单流强度 (+27 F3 派生): +29.49 → Δ = +3.59
# row 7 +微结构波动 (+27 F4 派生): +28.98 → Δ = -0.51
# row 8 +窗口统计 (+63 F5 整族): +29.68 → Δ = +0.70
# row 9 +不对称性 (+34 F6 派生): +34.32 → Δ = +4.64
deltas = [-0.07, +2.57, +3.59, -0.51, +0.70, +4.64]

fig, ax = plt.subplots(figsize=(6.0/2.54, 5.0/2.54), dpi=200)
yy = np.arange(len(family_labels))[::-1]
bar_colors = [palette[i] if deltas[i] >= 0 else "#bbbbbb" for i in range(len(deltas))]
ax.barh(yy, deltas, color=bar_colors, edgecolor="none", height=0.65)
for i, (yi, d) in enumerate(zip(yy, deltas)):
    fw = "bold" if d > 1.0 else "normal"
    ax.text(d + (0.15 if d >= 0 else -0.15), yi, f"{d:+.2f}",
            va="center", ha="left" if d >= 0 else "right",
            fontsize=5.8, color="#222", fontweight=fw)
ax.set_yticks(yy)
ax.set_yticklabels(family_labels, fontsize=6.0)
ax.axvline(0, color="#888", linewidth=0.4, linestyle="--")
ax.set_xlim(-1.2, max(deltas)*1.30)
ax.set_xticks([])
ax.set_title("加入各家族派生因子的 ΔPnL", fontsize=7.5, pad=2, fontweight="bold")
ax.set_xlabel("(NN V4, row 3 baseline +23.40 → 累加至 row 9)",
              fontsize=4.5, color="#555", labelpad=2)
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#888")
ax.spines["left"].set_linewidth(0.6)
ax.tick_params(axis="y", length=0, pad=1)
fig.tight_layout(pad=0.4)
fig.savefig(HERE / "Feature_ablation.pdf", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

# ===== (2) family raw vs derived gain 占比 =====
# gain (%) of total 13.664; raw + derived per family
raw_v_der = {
    "F1 LOB派生":    {"gain_raw": 5.327, "gain_der": 0.196},
    "F2 OFI":        {"gain_raw": 0.000, "gain_der": 0.750},
    "F3 订单流":      {"gain_raw": 3.958, "gain_der": 0.835},
    "F4 波动":        {"gain_raw": 0.006, "gain_der": 1.020},
    "F5 窗口统计":    {"gain_raw": 0.000, "gain_der": 0.708},
    "F6 不对称性":    {"gain_raw": 0.396, "gain_der": 0.468},
}
total_gain = sum(v["gain_raw"] + v["gain_der"] for v in raw_v_der.values())
raw_pct = [raw_v_der[k]["gain_raw"]/total_gain*100 for k in raw_v_der]
der_pct = [raw_v_der[k]["gain_der"]/total_gain*100 for k in raw_v_der]

fig, ax = plt.subplots(figsize=(6.0/2.54, 5.0/2.54), dpi=200)
y = np.arange(len(family_labels))[::-1]
ax.barh(y, raw_pct,  color=palette, alpha=1.0, edgecolor="none", height=0.6, label="raw 部分")
ax.barh(y, der_pct, left=raw_pct, color=palette, alpha=0.4, edgecolor="none", height=0.6, label="派生部分")
for i, yi in enumerate(y):
    r, d = raw_pct[i], der_pct[i]
    if r > 1: ax.text(r/2, yi, f"{r:.1f}", ha="center", va="center", fontsize=5.4, color="white" if r>3 else "#333", fontweight="bold")
    if d > 1: ax.text(r + d/2, yi, f"{d:.1f}", ha="center", va="center", fontsize=5.4, color="#333")
ax.set_yticks(y)
ax.set_yticklabels(family_labels, fontsize=6.0)
ax.set_xticks([])
ax.set_xlim(0, max([r+d for r, d in zip(raw_pct, der_pct)]) * 1.10)
ax.set_title("家族内 raw vs 派生 重要性占比 (% of 全局)", fontsize=7.5, pad=2, fontweight="bold")
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
ax.spines["left"].set_color("#888")
ax.spines["left"].set_linewidth(0.6)
ax.tick_params(axis="y", length=0, pad=1)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#888", alpha=1.0, label="raw"),
                   Patch(facecolor="#888", alpha=0.4, label="派生")],
          loc="lower right", fontsize=5.0, frameon=False,
          handlelength=0.9, handletextpad=0.3, borderpad=0.0)
fig.tight_layout(pad=0.4)
fig.savefig(HERE / "Feature_rawder.pdf", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)

print(f"Saved: Feature_ablation.pdf  +  Feature_rawder.pdf  in {HERE}")
