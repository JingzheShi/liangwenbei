"""Generate threshold sweep heatmaps from T43 grid sweep."""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(HERE, "results.json")) as f:
        res = json.load(f)
    grid = res["grid_sweep"]
    df = pd.DataFrame(grid)
    Ts = sorted(df["T"].unique())
    deltas = sorted(df["delta"].unique())

    def _piv(col):
        return df.pivot(index="delta", columns="T", values=col).loc[deltas, Ts]

    heat_5 = _piv("sum_5fold").to_numpy()
    heat_4 = _piv("sum_4fold_no3").to_numpy()
    heat_actr = _piv("mean_active_rate_4fold").to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, heat, title in zip(axes, [heat_5, heat_4, heat_actr],
                                ["sum_5fold (incl sym=3)",
                                 "sum_4fold (drop sym=3)",
                                 "mean active rate (4 folds)"]):
        im = ax.imshow(heat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(Ts))); ax.set_xticklabels([f"{x:.2f}" for x in Ts])
        ax.set_yticks(range(len(deltas))); ax.set_yticklabels([f"{x:.2f}" for x in deltas])
        ax.set_xlabel("T (symmetric)")
        ax.set_ylabel("delta (symmetric)")
        ax.set_title(title)
        for i in range(len(deltas)):
            for j in range(len(Ts)):
                v = heat[i, j]
                ax.text(j, i, f"{v:.2f}" if title != "mean active rate (4 folds)" else f"{v:.3f}",
                        ha="center", va="center",
                        color="white" if v < (heat.max() + heat.min()) / 2 else "black",
                        fontsize=8)
        plt.colorbar(im, ax=ax)
    iter006 = res["iter_006_baseline"]["metrics"]
    fig.suptitle(
        f"T43 threshold sweep | iter_006 ref: 5fold={iter006['sum_5fold']:.2f} "
        f"4fold={iter006['sum_4fold_no3']:.2f} actr_4f={iter006['mean_active_rate_4fold']:.3f}",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(HERE, "thresh_sweep_heatmap.png")
    plt.savefig(out, dpi=110)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
