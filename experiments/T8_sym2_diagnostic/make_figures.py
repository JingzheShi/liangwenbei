"""
T8 visualizations:
- midprice trajectories per sym (mean across dates)
- label distributions per sym
- rolling volatility per sym
- max_prob histogram on sym=2 OOF
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

WORKDIR = "/root/projects/liangwenbei_workdir"
OUT = f"{WORKDIR}/experiments/T8_sym2_diagnostic/figures"
os.makedirs(OUT, exist_ok=True)

# ---------- Load full data ----------
print("[fig] loading data ...")
keep = ["date", "sym", "midprice", "label_60", "amount_delta", "volume_delta", "spread1"]
files = sorted(glob(f"{WORKDIR}/data/snapshot_sym*.parquet"))
dfs = []
for f in files:
    base = Path(f).stem
    parts = base.split("_")
    df = pd.read_parquet(f, columns=keep)
    df["session"] = parts[-1]
    dfs.append(df)
big = pd.concat(dfs, ignore_index=True)
big["sym"] = big["sym"].astype(int)
big["date"] = big["date"].astype(int)
print(f"[fig] big: {big.shape}")

COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:red", 3: "tab:green", 4: "tab:purple"}

# ---------- Figure 1: midprice trajectory by date ----------
print("[fig] figure 1 (midprice traj) ...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for sym in range(5):
    sub = big[big.sym == sym]
    daily = sub.groupby(["date", "session"])["midprice"].mean().reset_index()
    am = daily[daily.session == "am"].sort_values("date")
    pm = daily[daily.session == "pm"].sort_values("date")
    label = f"sym={sym}" + (" (held in LOSO)" if sym == 2 else "")
    lw = 2.5 if sym == 2 else 1.0
    alpha = 1.0 if sym == 2 else 0.6
    axes[0].plot(am.date, am.midprice, label=label, color=COLORS[sym], lw=lw, alpha=alpha)
    axes[1].plot(pm.date, pm.midprice, label=label, color=COLORS[sym], lw=lw, alpha=alpha)
axes[0].set_title("Mean midprice (session-avg) by date — AM session")
axes[1].set_title("Mean midprice (session-avg) by date — PM session")
axes[1].set_xlabel("date")
for ax in axes:
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.axvline(96, color="gray", ls="--", alpha=0.5, label="_nolegend_")
plt.tight_layout()
plt.savefig(f"{OUT}/sym_midprice_traj.png", dpi=120)
plt.close()
print(f"  -> {OUT}/sym_midprice_traj.png")

# ---------- Figure 2: label_60 distribution ----------
print("[fig] figure 2 (label dist) ...")
fig, ax = plt.subplots(figsize=(10, 6))
syms = list(range(5))
labels = [0, 1, 2]
data = np.zeros((5, 3))
for i, sym in enumerate(syms):
    sub = big[big.sym == sym]
    counts = sub["label_60"].value_counts(normalize=True)
    for j, lab in enumerate(labels):
        data[i, j] = counts.get(lab, 0)

x = np.arange(5)
w = 0.25
ax.bar(x - w, data[:, 0], w, label="down (0)", color="tab:red")
ax.bar(x, data[:, 1], w, label="flat (1)", color="gray")
ax.bar(x + w, data[:, 2], w, label="up (2)", color="tab:green")
for i, sym in enumerate(syms):
    for j, lab in enumerate(labels):
        ax.text(i + (j-1)*w, data[i, j] + 0.005, f"{data[i, j]:.2f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"sym={s}" + (" *" if s == 2 else "") for s in syms])
ax.set_ylabel("fraction of ticks")
ax.set_title("label_60 class distribution per sym (* = held-out)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/sym_label_dist.png", dpi=120)
plt.close()
print(f"  -> {OUT}/sym_label_dist.png")

# ---------- Figure 3: per-tick volatility (mid_diff std) by date ----------
print("[fig] figure 3 (volatility traj) ...")
fig, ax = plt.subplots(figsize=(14, 6))
big_sorted = big.sort_values(["sym", "date", "session"])
big_sorted["mid_diff"] = big_sorted.groupby(["sym", "date", "session"])["midprice"].diff()
vol_per = big_sorted.groupby(["sym", "date"])["mid_diff"].apply(lambda s: s.abs().mean()).reset_index()
vol_per.columns = ["sym", "date", "abs_mid_diff"]
for sym in range(5):
    sub = vol_per[vol_per.sym == sym].sort_values("date")
    label = f"sym={sym}" + (" (held)" if sym == 2 else "")
    lw = 2.5 if sym == 2 else 1.0
    alpha = 1.0 if sym == 2 else 0.6
    ax.plot(sub.date, sub.abs_mid_diff, label=label, color=COLORS[sym], lw=lw, alpha=alpha)
ax.set_xlabel("date")
ax.set_ylabel("mean |Δmidprice| per tick (per-day)")
ax.set_title("Per-tick volatility (|Δmidprice| mean) over time — sym=2 lower than others")
ax.legend()
ax.grid(alpha=0.3)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig(f"{OUT}/sym_volatility_traj.png", dpi=120)
plt.close()
print(f"  -> {OUT}/sym_volatility_traj.png")

# ---------- Figure 4: amount_delta distribution log-scale ----------
print("[fig] figure 4 (amount_delta) ...")
fig, ax = plt.subplots(figsize=(10, 6))
for sym in range(5):
    sub = big[big.sym == sym]["amount_delta"]
    log_amt = np.log1p(sub.clip(lower=0))
    label = f"sym={sym}" + (" (held)" if sym == 2 else "")
    ax.hist(log_amt, bins=80, density=True, histtype="step",
            label=label, color=COLORS[sym], lw=2 if sym == 2 else 1, alpha=0.8)
ax.set_xlabel("log1p(amount_delta) — CNY")
ax.set_ylabel("density")
ax.set_title("Per-tick trade amount distribution (log) — sym=2 dominates")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/sym_amount_delta_hist.png", dpi=120)
plt.close()
print(f"  -> {OUT}/sym_amount_delta_hist.png")

# ---------- Figure 5: max_prob distribution on OOF for sym=2 ----------
print("[fig] figure 5 (oof max_prob) ...")
pred = pd.read_parquet(f"{WORKDIR}/experiments/T4_loso_validate/loso_pred_schemeB_held2.parquet")
pred["max_prob"] = pred[["prob_0", "prob_1", "prob_2"]].max(axis=1)
pred["correct"] = (pred.pred_label_60 == pred.true_label_60).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# (a) max_prob histogram split by correct/wrong
axes[0].hist(pred[pred.correct == 1]["max_prob"], bins=50, alpha=0.6, label="correct", color="tab:green")
axes[0].hist(pred[pred.correct == 0]["max_prob"], bins=50, alpha=0.6, label="wrong", color="tab:red")
axes[0].set_xlabel("max_prob (model confidence)")
axes[0].set_ylabel("count")
axes[0].set_title("OOF max_prob distribution — correct vs wrong (sym=2 LOSO)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# (b) predicted vs true label distribution
labs = [0, 1, 2]
true_d = [pred.true_label_60.eq(l).mean() for l in labs]
pred_d = [pred.pred_label_60.eq(l).mean() for l in labs]
x = np.arange(3)
w = 0.35
axes[1].bar(x - w/2, true_d, w, label="true", color="tab:blue")
axes[1].bar(x + w/2, pred_d, w, label="pred", color="tab:orange")
for i, (t, p) in enumerate(zip(true_d, pred_d)):
    axes[1].text(i - w/2, t + 0.01, f"{t:.2f}", ha="center", fontsize=10)
    axes[1].text(i + w/2, p + 0.01, f"{p:.2f}", ha="center", fontsize=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(["down (0)", "flat (1)", "up (2)"])
axes[1].set_ylabel("fraction")
axes[1].set_title("LOSO held=2: pred dist vs true dist (model massively over-calls UP)")
axes[1].legend()
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/sym2_oof_diagnosis.png", dpi=120)
plt.close()
print(f"  -> {OUT}/sym2_oof_diagnosis.png")

print("[done] all figures")
