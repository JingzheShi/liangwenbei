"""Pearson-correlation analysis + sample-session visualisation for feature_v1.

Outputs:
  - experiments/T3_features_v1/figures/*.png  (4 plots)
  - experiments/T3_features_v1/pearson_top20.csv (per label)

Run: python experiments/T3_features_v1/analyze.py
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from compute import feature_v1_columns  # noqa: E402

CACHE_DIR = os.path.join(REPO_ROOT, "data", "features_v1")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

LABEL_COLS = ("label_5", "label_10", "label_20", "label_40", "label_60")
NEW_COLS = feature_v1_columns()


def _load_subset(n_per_sym: int = 6) -> pd.DataFrame:
    """Load a random subset (~30 sessions) for correlation analysis. Avoids OOM."""
    files = sorted(os.listdir(CACHE_DIR))
    rng = np.random.RandomState(42)
    chosen = []
    by_sym = {0: [], 1: [], 2: [], 3: [], 4: []}
    for f in files:
        # snapshot_symX_dateY_{sess}.parquet
        sym = int(f.split("_sym")[1].split("_")[0])
        by_sym[sym].append(f)
    for sym in by_sym:
        idx = rng.choice(len(by_sym[sym]), size=n_per_sym, replace=False)
        chosen.extend([by_sym[sym][i] for i in idx])
    print(f"[analyze] loading {len(chosen)} sessions for Pearson analysis")
    parts = []
    for f in chosen:
        df = pd.read_parquet(os.path.join(CACHE_DIR, f))
        parts.append(df[list(NEW_COLS) + list(LABEL_COLS)])
    return pd.concat(parts, axis=0, ignore_index=True)


def pearson_top20(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Pearson r between each new feature and one label column.

    Labels are 0/1/2 categorical-as-int; map to (-1, 0, +1) so positive
    correlation = 'feature predicts up-move'.
    """
    y_raw = df[label].astype(np.float64).to_numpy()
    y = np.where(y_raw == 2, 1.0, np.where(y_raw == 0, -1.0, 0.0))
    rows = []
    for c in NEW_COLS:
        x = df[c].to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 100:
            rows.append((c, np.nan, mask.sum()))
            continue
        xm = x[mask]
        ym = y[mask]
        if xm.std() == 0 or ym.std() == 0:
            rows.append((c, 0.0, mask.sum()))
            continue
        r = np.corrcoef(xm, ym)[0, 1]
        rows.append((c, r, mask.sum()))
    out = pd.DataFrame(rows, columns=["feature", "pearson_r", "n"])
    out["abs_r"] = out["pearson_r"].abs()
    out = out.sort_values("abs_r", ascending=False).reset_index(drop=True)
    return out


def plot_traces(sample_path: str = "snapshot_sym0_date0_am.parquet"):
    """Visualise MLOFI / WMP / RV / EWMA on one sample session."""
    df = pd.read_parquet(os.path.join(CACHE_DIR, sample_path))
    n = len(df)
    # Sample 20 evenly-spaced indices for the dot overlay
    idx20 = np.linspace(50, n - 1, 20).astype(int)

    # Panel 1: MLOFI W=20 across levels 1-3 + level1 W∈{5,20,60}
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    for k in (1, 2, 3):
        ax.plot(df.index, df[f"mlofi_W20_lvl{k}"], label=f"mlofi_W20_lvl{k}", lw=0.8)
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_title("MLOFI W=20 across LOB levels 1-3 — should oscillate around 0")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("MLOFI flow")

    ax = axes[1]
    for W in (5, 20, 60):
        ax.plot(df.index, df[f"mlofi_W{W}_lvl1"], label=f"mlofi_W{W}_lvl1", lw=0.8)
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_title("MLOFI level 1 across windows {5,20,60} — longer W = smoother")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("tick")
    ax.set_ylabel("MLOFI flow")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "mlofi.png"), dpi=110)
    plt.close(fig)

    # Panel 2: WMP_lvl1, WMP_lvl2 vs midprice + wmp_balance_12
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    ax.plot(df.index, df["midprice"], label="midprice (official)", lw=0.7, color="k")
    ax.plot(df.index, df["wmp_lvl1"], label="wmp_lvl1", lw=0.7, alpha=0.8)
    ax.plot(df.index, df["wmp_lvl2"], label="wmp_lvl2", lw=0.7, alpha=0.8)
    ax.scatter(idx20, df["wmp_lvl1"].iloc[idx20], s=18, color="red", zorder=5, label="20 sample t")
    ax.set_title("WMP_lvl1 / WMP_lvl2 vs official midprice — WMP tracks mid w/ imbalance bias")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("price (chg %)")

    ax = axes[1]
    ax.plot(df.index, df["wmp_balance_12"], label="wmp_balance_12 = wmp_lvl1 - wmp_lvl2", lw=0.7)
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_title("WMP balance (Optiver-style microprice slope)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("tick")
    ax.set_ylabel("Δ price")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "wmp.png"), dpi=110)
    plt.close(fig)

    # Panel 3: RV_W trajectories
    fig, ax = plt.subplots(1, 1, figsize=(11, 4))
    for W in (5, 10, 20, 50):
        ax.plot(df.index, df[f"rv_w{W}"], label=f"rv_w{W}", lw=0.8)
    ax.set_title("Realized Volatility (multi-window) — short W spikes during turbulence")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("tick")
    ax.set_ylabel("RV (sqrt sum log_ret²)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rv.png"), dpi=110)
    plt.close(fig)

    # Panel 4: EWMA intensities for lb_intst across alphas
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes[0]
    ax.plot(df.index, df["lb_intst"], label="lb_intst (raw)", lw=0.5, color="k", alpha=0.5)
    for alpha in (0.05, 0.1, 0.3, 0.5):
        ax.plot(df.index, df[f"ewma_a{alpha}_lb_intst"], label=f"ewma_a{alpha}_lb", lw=0.8)
    ax.set_title("EWMA of lb_intst — smaller α → longer half-life → smoother")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("intensity")

    ax = axes[1]
    # Compare buy/sell pressure via fastest EWMA (α=0.5) on lb vs la
    ax.plot(df.index, df["ewma_a0.5_lb_intst"], label="ewma_a0.5_lb (limit BUY)", lw=0.8)
    ax.plot(df.index, df["ewma_a0.5_la_intst"], label="ewma_a0.5_la (limit SELL)", lw=0.8)
    ax.plot(df.index, df["ewma_a0.5_mb_intst"], label="ewma_a0.5_mb (mkt BUY)", lw=0.8, alpha=0.6)
    ax.plot(df.index, df["ewma_a0.5_ma_intst"], label="ewma_a0.5_ma (mkt SELL)", lw=0.8, alpha=0.6)
    ax.set_title("Order-class intensities (Hawkes proxy, α=0.5)")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("tick")
    ax.set_ylabel("intensity")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ewma_intst.png"), dpi=110)
    plt.close(fig)

    print(f"[analyze] wrote 4 figures to {FIG_DIR}")


def main():
    t0 = time.time()
    df = _load_subset(n_per_sym=6)
    print(f"[analyze] subset shape: {df.shape}; load time {time.time()-t0:.1f}s")

    # Pearson per label
    ranks = {}
    for label in LABEL_COLS:
        r = pearson_top20(df, label)
        ranks[label] = r
        out_path = os.path.join(HERE, f"pearson_top20_{label}.csv")
        r.head(20).to_csv(out_path, index=False)
        print(f"[analyze] {label} top1: {r.iloc[0]['feature']}={r.iloc[0]['pearson_r']:+.4f}")

    # Aggregate: max |r| across all 5 labels per feature
    agg = pd.DataFrame({"feature": NEW_COLS})
    for label in LABEL_COLS:
        agg = agg.merge(
            ranks[label][["feature", "pearson_r"]].rename(columns={"pearson_r": f"r_{label}"}),
            on="feature",
            how="left",
        )
    agg["max_abs_r"] = agg[[c for c in agg.columns if c.startswith("r_")]].abs().max(axis=1)
    agg = agg.sort_values("max_abs_r", ascending=False).reset_index(drop=True)
    agg.head(30).to_csv(os.path.join(HERE, "pearson_top20_aggregate.csv"), index=False)
    print(f"\n[analyze] Aggregate top 20 (by max |r| across 5 labels):")
    print(agg.head(20).to_string(index=False))

    plot_traces()
    print(f"\n[analyze] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
