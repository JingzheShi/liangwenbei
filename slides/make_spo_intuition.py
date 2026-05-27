#!/usr/bin/env python3
"""
SPO+ DFL intuition figure.

Left panel: oracle decision z*(y) over true return y given fee f
            -- step function {-1, 0, +1}
Right panel: SPO+ surrogate vs L2 loss as a function of prediction y_hat
             at a fixed true y, illustrating that L2 keeps shrinking gradient
             once |y_hat| < f even though decision is already wrong.
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

f = 1.0   # fee (normalized units)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

# ---------- LEFT : oracle decision ----------
ax = axes[0]
y = np.linspace(-3, 3, 1001)
z = np.where(y > f, 1, np.where(y < -f, -1, 0))

ax.plot(y, z, lw=2.6, color="#001E5A", drawstyle="steps-mid")
ax.axvspan(-f, f, color="#cccccc", alpha=0.35, label="不出手区 |y|≤f")
ax.axhline(0, color="#999", lw=0.7)
ax.axvline(0, color="#999", lw=0.7)
ax.axvline(+f, color="#aa0000", lw=0.9, linestyle="--")
ax.axvline(-f, color="#aa0000", lw=0.9, linestyle="--")

ax.set_xlabel("真实未来收益 y", fontsize=11)
ax.set_ylabel("oracle 离散动作 z*(y)", fontsize=11)
ax.set_title("决策只看 sign + 是否过费率门槛 f",
             fontsize=11.5, color="#001E5A", pad=8)
ax.set_xticks([-f, 0, f])
ax.set_xticklabels(["−f", "0", "+f"])
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels(["−1\n(卖)", "0\n(不出手)", "+1\n(买)"])
ax.set_ylim(-1.6, 1.6)
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# annotation
ax.annotate("L2 损失会逼近真实 y\n但决策只需越过 ±f",
            xy=(2.0, 1), xytext=(0.05, 1.35), fontsize=9.5,
            color="#444",
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

# ---------- RIGHT : SPO+ vs L2 ----------
ax = axes[1]
y_true = 1.6           # a positive-return sample (oracle = +1)
y_hat = np.linspace(-3, 4, 1001)

# L2 (per-sample squared error)
l2 = (y_hat - y_true) ** 2

# SPO+ surrogate: ReLU(|2yhat - y| - f) - z*(y)(2yhat - y) + f|z*(y)|
z_star = 1.0   # because y_true > f
spo = np.maximum(np.abs(2*y_hat - y_true) - f, 0) - z_star * (2*y_hat - y_true) + f * abs(z_star)

ax.plot(y_hat, l2, lw=2.4, color="#888888", label=r"L2  $(\hat{y}-y)^2$")
ax.plot(y_hat, spo, lw=2.6, color="#cf3a3a", label="SPO+ 决策焦点损失")
ax.axvspan(-f, f, color="#cccccc", alpha=0.35)
ax.axvline(y_true, color="#001E5A", lw=1.0, linestyle=":")
ax.set_xlabel(r"预测 $\hat{y}$", fontsize=11)
ax.set_ylabel("损失", fontsize=11)
ax.set_title("SPO+ 在「过/不过门槛」处给出强梯度",
             fontsize=11.5, color="#001E5A", pad=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_xticks([-f, 0, f, y_true, 2*f])
ax.set_xticklabels(["−f", "0", "+f", "y", "+2f"])
ax.set_ylim(-0.6, 11.5)
ax.set_xlim(-3, 4)

ax.text(y_true+0.08, 0.45, "y (真实)", color="#001E5A", fontsize=8.5)
ax.legend(loc="upper right", fontsize=9.2, frameon=False)

ax.annotate(r"L2 在 $\hat{y}\to y$ 时梯度→0，" "\n" "但决策可能仍错；\nSPO+ 越过门槛后才放缓",
            xy=(-0.2, 3.6), xytext=(-2.9, 8.0), fontsize=8.8, color="#444",
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

plt.tight_layout()
plt.savefig("/root/projects/liangwenbei_workdir/slides/spo_intuition.pdf",
            bbox_inches="tight", pad_inches=0.05)
print("spo_intuition.pdf written")
