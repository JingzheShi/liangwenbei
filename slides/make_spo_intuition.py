#!/usr/bin/env python3
"""
SPO+ DFL intuition figure - VERTICAL stacked layout (v2).

Top panel: L2 loss (smooth quadratic) - the per-sample regression loss.
Bottom panel: SPO+ surrogate - kink at the decision boundary y_hat=+/-f,
              flat (zero gradient) inside the no-trade band when y aligns.

Both share the same x-axis (prediction y_hat) at fixed true y > +f.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(_cjk)
_name = fm.FontProperties(fname=_cjk).get_name()
mpl.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

f = 1.0
y_true = 1.6
y_hat = np.linspace(-3, 4, 1001)

# L2 (per-sample squared error)
l2 = (y_hat - y_true) ** 2

# SPO+ surrogate: ReLU(|2yhat - y| - f) - z*(y)(2yhat - y) + f|z*(y)|
z_star = 1.0
spo = (np.maximum(np.abs(2 * y_hat - y_true) - f, 0)
       - z_star * (2 * y_hat - y_true)
       + f * abs(z_star))

# Vertical layout: figure is taller than wide (column figure)
fig, axes = plt.subplots(2, 1, figsize=(5.3, 6.2), sharex=True,
                          gridspec_kw=dict(hspace=0.30))

# ---------- TOP : L2 ----------
ax = axes[0]
ax.plot(y_hat, l2, lw=2.6, color="#3a6ea5", label=r"L2 损失 $(\hat{y}-y)^2$")
ax.axvspan(-f, f, color="#dddddd", alpha=0.55, label=r"不出手区 $|\hat{y}|\leq f$")
ax.axvline(y_true, color="#001E5A", lw=1.0, linestyle=":")
ax.axhline(0, color="#999", lw=0.6)
ax.set_ylabel("L2 损失", fontsize=11)
ax.set_title("Phase 1：L2 回归预训练 —— 光滑二次型，逼近真值",
             fontsize=11, color="#001E5A", pad=6)
ax.set_ylim(-0.6, 12)
ax.set_xticks([-f, 0, f, y_true, 2 * f])
ax.set_xticklabels(["−f", "0", "+f", "y", "+2f"])
ax.text(y_true + 0.08, 0.5, "真实 y", color="#001E5A", fontsize=9)
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate(r"$\hat{y}\to y$ 时梯度→0," "\n但决策可能仍错",
            xy=(0.3, 1.69), xytext=(-2.7, 6.5),
            fontsize=9, color="#555",
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

# ---------- BOTTOM : SPO+ ----------
ax = axes[1]
ax.plot(y_hat, spo, lw=2.6, color="#cf3a3a", label="SPO+ 决策焦点损失")
ax.axvspan(-f, f, color="#dddddd", alpha=0.55, label=r"不出手区 $|\hat{y}|\leq f$")
ax.axvline(+f, color="#aa0000", lw=0.9, linestyle="--")
ax.axvline(-f, color="#aa0000", lw=0.9, linestyle="--")
ax.axvline(y_true, color="#001E5A", lw=1.0, linestyle=":")
ax.axhline(0, color="#999", lw=0.6)
ax.set_xlabel(r"预测 $\hat{y}$", fontsize=11)
ax.set_ylabel("SPO+ 损失", fontsize=11)
ax.set_title("Phase 2：SPO+ joint finetune —— 在决策边界处出现 kink",
             fontsize=11, color="#001E5A", pad=6)
ax.set_ylim(-0.6, 12)
ax.set_xlim(-3, 4)
ax.set_xticks([-f, 0, f, y_true, 2 * f])
ax.set_xticklabels(["−f", "0", "+f", "y", "+2f"])
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.annotate("过门槛 +f 后梯度才放缓\n→ 对齐离散动作期望收益",
            xy=(1.2, 0.65), xytext=(-2.7, 7.0),
            fontsize=9, color="#555",
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

plt.tight_layout()
plt.savefig("/root/projects/liangwenbei_workdir/slides/spo_intuition.pdf",
            bbox_inches="tight", pad_inches=0.05)
print("spo_intuition.pdf written (vertical)")
