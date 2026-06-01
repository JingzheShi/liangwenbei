"""Analyze SchemeP factors: family classification, LGB importance, correlations, figures.

Outputs into /root/projects/liangwenbei_workdir/factor_families_report/:
  - family_assignment.json   (per-feature assignment)
  - importance.json          (per-feature LGB gain)
  - corr_within_F1.pdf .. corr_within_F6.pdf
  - corr_block_summary.pdf
  - importance_top40.pdf
  - importance_family_share.pdf
  - importance_family_share.csv
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = Path("/root/projects/liangwenbei_workdir/factor_families_report")
CACHE = Path("/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p")
LGB_MODEL = Path(
    "/root/projects/liangwenbei_workdir/ablation_runs/big50_lgb_m7/seed1/model.txt"
)

# --- Chinese font setup for matplotlib -------------------------------------
NOTO_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(NOTO_PATH):
    font_manager.fontManager.addfont(NOTO_PATH)
    # matplotlib only registers the first face of a .ttc -> "Noto Sans CJK JP"
    # which still contains the full CJK Unified Ideographs (works for Simplified Chinese)
    rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    rcParams["font.family"] = "sans-serif"
rcParams["axes.unicode_minus"] = False

# --- Family classification ------------------------------------------------
FAMILY_ORDER = ["F1", "F2", "F3", "F4", "F5", "F6"]
FAMILY_LABEL = {
    "F1": "F1 LOB派生",
    "F2": "F2 多尺度OFI",
    "F3": "F3 订单流强度",
    "F4": "F4 微结构波动",
    "F5": "F5 窗口统计",
    "F6": "F6 不对称性",
}
FAMILY_COLOR = {
    "F1": "#1f77b4",  # blue
    "F2": "#d62728",  # red
    "F3": "#2ca02c",  # green
    "F4": "#ff7f0e",  # orange
    "F5": "#9467bd",  # purple
    "F6": "#8c564b",  # brown
}


def classify(name: str) -> str:
    # F2 OFI family (multi-scale OFI / EWMA-OFI / Kyle / OFI toxicity)
    if name.startswith("mlofi_"):
        return "F2"
    if name.startswith("ewma_ofi_"):
        return "F2"
    if name.startswith("kyle_inv_"):
        return "F2"
    if name.startswith("kyle_lam_"):
        return "F2"
    if name.startswith("ofi_tox_"):
        return "F2"

    # F3 order-flow intensity raw + EWMA + cancel
    if name.startswith("ewma_a") and "_intst" in name:
        return "F3"
    if name.endswith(("_intst", "_ind", "_acc")) and any(
        name.startswith(p) for p in ("lb_", "la_", "mb_", "ma_", "cb_", "ca_")
    ):
        return "F3"
    if name.startswith("cancel_imb_"):
        return "F3"

    # F4 volatility
    if name.startswith("rv_w") or name.startswith("rv_ratio_"):
        return "F4"
    if name.startswith("signed_rv_") or name.startswith("signed_bv_"):
        return "F4"
    if name.startswith("rskew_"):
        return "F4"
    if name.startswith("vol_burst_") or name.startswith("amt_burst_"):
        return "F4"
    if name.startswith("jshare_"):
        return "F4"
    if name.startswith("roll_eff_spr_"):
        return "F4"
    if name in ("volume_delta", "amount_delta"):
        return "F4"

    # F5 window-statistics
    if name.startswith("dualz_"):
        return "F5"
    if name.startswith("qrank_"):
        return "F5"
    if name.startswith("mid_ewma_resid"):
        return "F5"
    if name.startswith("adapt_mom_"):
        return "F5"
    if name.startswith("spread_reg_"):
        return "F5"
    if name.startswith("trade_pers_"):
        return "F5"

    # F6 asymmetry / order-flow direction
    if name.startswith("gofi_"):
        return "F6"
    if name.startswith("liq_asym_"):
        return "F6"
    if name.startswith(("bid_rate", "ask_rate", "bsize_rate", "asize_rate")):
        return "F6"

    # F1 default: raw LOB & derived structure
    return "F1"


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    feat_names = (
        (CACHE / "schemeP_feat_names.txt").read_text().strip().splitlines()
    )
    assert len(feat_names) == 370, f"expected 370 feats, got {len(feat_names)}"

    # --- classify -------------------------------------------------------
    fam_of = OrderedDict()
    for n in feat_names:
        fam_of[n] = classify(n)
    counts = {f: 0 for f in FAMILY_ORDER}
    for f in fam_of.values():
        counts[f] += 1
    print("Family counts:", counts, flush=True)
    (HERE / "family_assignment.json").write_text(
        json.dumps(
            {"counts": counts, "assignment": fam_of, "labels": FAMILY_LABEL},
            ensure_ascii=False,
            indent=2,
        )
    )

    # --- LGB feature importance -----------------------------------------
    print(f"Loading LGB model {LGB_MODEL}", flush=True)
    booster = lgb.Booster(model_file=str(LGB_MODEL))
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    assert len(gain) == len(feat_names), (
        f"mismatch: {len(gain)} vs {len(feat_names)}"
    )
    importance = []
    for n, g, s in zip(feat_names, gain, split):
        importance.append(
            {"name": n, "family": fam_of[n], "gain": float(g), "split": int(s)}
        )
    (HERE / "importance.json").write_text(
        json.dumps(importance, ensure_ascii=False, indent=2)
    )
    total_gain = float(np.sum(gain))
    family_gain = {f: 0.0 for f in FAMILY_ORDER}
    for r in importance:
        family_gain[r["family"]] += r["gain"]
    family_share = {
        f: family_gain[f] / max(total_gain, 1e-9) for f in FAMILY_ORDER
    }
    print("Total gain:", total_gain)
    print("Family share:", family_share)

    # CSV of family share
    df_share = pd.DataFrame(
        [
            {
                "family": FAMILY_LABEL[f],
                "n_factors": counts[f],
                "gain_share_pct": round(100.0 * family_share[f], 2),
                "mean_gain_per_factor": (
                    family_gain[f] / max(counts[f], 1)
                ),
            }
            for f in FAMILY_ORDER
        ]
    )
    df_share.to_csv(HERE / "importance_family_share.csv", index=False)

    # --- LOAD a sample of data for correlation --------------------------
    print("Loading sample X from schemeP_train.npz...", flush=True)
    data = np.load(CACHE / "schemeP_train.npz")
    X_full = data["X"]  # (N, 370) float32
    rng = np.random.default_rng(0)
    N = X_full.shape[0]
    n_sample = min(100_000, N)
    idx = rng.choice(N, size=n_sample, replace=False)
    X = X_full[np.sort(idx)].astype(np.float32)
    print(f"  sample shape {X.shape}")

    # Compute global correlation matrix (370x370) once
    # Using float64 to avoid precision loss
    X64 = X.astype(np.float64)
    # subtract mean
    mu = X64.mean(axis=0, keepdims=True)
    sd = X64.std(axis=0, ddof=0, keepdims=True) + 1e-12
    Xn = (X64 - mu) / sd
    C = (Xn.T @ Xn) / X64.shape[0]
    # clip extreme NaN
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(HERE / "corr_full_370.npy", C.astype(np.float32))

    # --- Within-family heatmaps -----------------------------------------
    fam_to_idx = {f: [] for f in FAMILY_ORDER}
    for i, n in enumerate(feat_names):
        fam_to_idx[fam_of[n]].append(i)

    for f in FAMILY_ORDER:
        ids = np.array(fam_to_idx[f])
        sub = C[np.ix_(ids, ids)]
        n = len(ids)
        names = [feat_names[i] for i in ids]
        # If the family is large, omit tick labels for readability
        fig, ax = plt.subplots(figsize=(8.4, 7.2))
        im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(
            f"{FAMILY_LABEL[f]}：家族内 Pearson 相关性（{n} 因子）",
            fontsize=11,
        )
        if n <= 40:
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            ax.set_xticklabels(names, fontsize=5, rotation=90)
            ax.set_yticklabels(names, fontsize=5)
        else:
            tick_step = max(1, n // 25)
            ticks = np.arange(0, n, tick_step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([names[i] for i in ticks], fontsize=5, rotation=90)
            ax.set_yticklabels([names[i] for i in ticks], fontsize=5)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Pearson r", fontsize=9)
        plt.tight_layout()
        out = HERE / f"corr_within_{f}.pdf"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out.name}  (n={n}  mean|r|={np.mean(np.abs(sub)):.3f})")

    # --- Block-summary 6x6 heatmap (mean |corr| within blocks) ----------
    block = np.zeros((6, 6), dtype=np.float64)
    for i, fi in enumerate(FAMILY_ORDER):
        for j, fj in enumerate(FAMILY_ORDER):
            sub = C[np.ix_(fam_to_idx[fi], fam_to_idx[fj])]
            if i == j:
                # exclude diagonal of full corr (self correlations = 1)
                mask = ~np.eye(len(fam_to_idx[fi]), dtype=bool)
                v = np.mean(np.abs(sub[mask])) if mask.sum() > 0 else 0.0
            else:
                v = float(np.mean(np.abs(sub)))
            block[i, j] = v
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(block, cmap="YlOrRd", vmin=0, vmax=block.max())
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILY_ORDER], rotation=35,
                       ha="right", fontsize=9)
    ax.set_yticklabels([FAMILY_LABEL[f] for f in FAMILY_ORDER], fontsize=9)
    ax.set_title("家族间相关性总览：块内平均 |Pearson r|", fontsize=11)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{block[i,j]:.2f}", ha="center", va="center",
                    color="black" if block[i,j] < block.max() * 0.6 else "white",
                    fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.savefig(HERE / "corr_block_summary.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.save(HERE / "corr_block_summary.npy", block)
    print(f"  saved corr_block_summary.pdf")

    # --- Top-40 LGB importance, family-colored --------------------------
    imp_sorted = sorted(importance, key=lambda r: -r["gain"])
    top40 = imp_sorted[:40]
    fig, ax = plt.subplots(figsize=(8.0, 9.5))
    ys = np.arange(len(top40))[::-1]
    bars = ax.barh(
        ys,
        [r["gain"] for r in top40],
        color=[FAMILY_COLOR[r["family"]] for r in top40],
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([r["name"] for r in top40], fontsize=7)
    ax.set_xlabel("LGB gain (越大越重要)", fontsize=9)
    ax.set_title("LGB 特征重要性 Top 40（按家族着色）", fontsize=11)
    # legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=FAMILY_COLOR[f], edgecolor="black", label=FAMILY_LABEL[f])
        for f in FAMILY_ORDER
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(HERE / "importance_top40.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved importance_top40.pdf")

    # --- Family share bar -----------------------------------------------
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    xs = np.arange(6)
    counts_arr = np.array([counts[f] for f in FAMILY_ORDER])
    shares = np.array([100.0 * family_share[f] for f in FAMILY_ORDER])
    width = 0.38
    b1 = ax.bar(
        xs - width / 2,
        counts_arr / counts_arr.sum() * 100,
        width,
        label="因子数占比 (%)",
        color="lightgray",
        edgecolor="black",
    )
    b2 = ax.bar(
        xs + width / 2,
        shares,
        width,
        label="LGB gain 占比 (%)",
        color=[FAMILY_COLOR[f] for f in FAMILY_ORDER],
        edgecolor="black",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [FAMILY_LABEL[f] for f in FAMILY_ORDER], rotation=15, fontsize=9
    )
    ax.set_ylabel("百分比 (%)", fontsize=9)
    ax.set_title("各家族：因子数 vs LGB gain 占比", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    for x, c in zip(xs - width / 2, counts_arr):
        ax.text(x, c / counts_arr.sum() * 100 + 0.5, str(c),
                ha="center", va="bottom", fontsize=8)
    for x, s in zip(xs + width / 2, shares):
        ax.text(x, s + 0.5, f"{s:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(HERE / "importance_family_share.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved importance_family_share.pdf")

    # --- Top family/factor summary --------------------------------------
    top_fam = max(family_share.items(), key=lambda kv: kv[1])
    print(f"Top family: {top_fam[0]} share={top_fam[1]:.3f}")
    print("Top 10 factors:")
    for r in imp_sorted[:10]:
        print(f"  {r['name']:32s} {FAMILY_LABEL[r['family']]:12s} gain={r['gain']:.1f}")

    # Save aggregated summary for tex
    summary = {
        "n_factors_total": len(feat_names),
        "family_counts": {FAMILY_LABEL[f]: counts[f] for f in FAMILY_ORDER},
        "family_gain_share_pct": {
            FAMILY_LABEL[f]: round(100.0 * family_share[f], 2)
            for f in FAMILY_ORDER
        },
        "top_importance_family": FAMILY_LABEL[top_fam[0]],
        "top_importance_family_share_pct": round(100.0 * top_fam[1], 2),
        "top10_factors": [
            {"name": r["name"], "family": FAMILY_LABEL[r["family"]],
             "gain": round(r["gain"], 1)}
            for r in imp_sorted[:10]
        ],
        "block_corr": [[round(block[i, j], 3) for j in range(6)] for i in range(6)],
        "block_corr_labels": [FAMILY_LABEL[f] for f in FAMILY_ORDER],
    }
    (HERE / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print("ALL DONE")


if __name__ == "__main__":
    main()
