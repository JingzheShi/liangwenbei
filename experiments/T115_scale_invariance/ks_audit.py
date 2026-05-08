"""T115 Phase 1: Per-feature pairwise KS distance audit across 5 syms.

For each of 370 schemeP features, compute KS distance between every pair of syms
(10 pairs * 370 = 3700 KS values), take max across pairs as feature severity score.
Output:
  - top_ks_features.csv (sorted by max_ks desc)
  - all_ks_features.csv (full table with per-pair KS)
  - ks_audit_progress.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

SYMS = (0, 1, 2, 3, 4)
PAIRS = [(i, j) for i in SYMS for j in SYMS if i < j]


def main():
    print("=== T115 KS audit ===", flush=True)
    t0 = time.time()
    d = np.load(os.path.join(CACHE_DIR, "schemeP_train.npz"))
    X = d["X"]
    sym = d["sym"]
    n, F = X.shape
    print(f"  X: {X.shape}, sym counts: {[(s, int((sym==s).sum())) for s in SYMS]}",
          flush=True)

    feat_names = [l.strip() for l in open(os.path.join(CACHE_DIR, "schemeP_feat_names.txt"))]
    assert len(feat_names) == F

    # Subsample 20k per sym for KS speed (KS is O(n log n))
    rng = np.random.default_rng(0)
    NSAMP = 20000
    idx_per_sym = {}
    for s in SYMS:
        sym_idx = np.where(sym == s)[0]
        if len(sym_idx) > NSAMP:
            sym_idx = rng.choice(sym_idx, NSAMP, replace=False)
        idx_per_sym[s] = sym_idx

    print(f"  using NSAMP={NSAMP} per sym, {F} features × {len(PAIRS)} pairs = "
          f"{F*len(PAIRS)} KS tests", flush=True)

    # Stats per (feature, sym): mean, std
    stats_per_sym = {s: {} for s in SYMS}
    for s in SYMS:
        Xs = X[idx_per_sym[s]]
        stats_per_sym[s]["mean"] = Xs.mean(axis=0)
        stats_per_sym[s]["std"] = Xs.std(axis=0)
        stats_per_sym[s]["abs_mean"] = np.abs(Xs).mean(axis=0)

    rows = []
    t_start = time.time()
    for fi in range(F):
        ks_per_pair = np.zeros(len(PAIRS), dtype=np.float64)
        for k, (a, b) in enumerate(PAIRS):
            xa = X[idx_per_sym[a], fi]
            xb = X[idx_per_sym[b], fi]
            res = ks_2samp(xa, xb, mode="asymp")
            ks_per_pair[k] = float(res.statistic)
        max_ks = float(ks_per_pair.max())
        mean_ks = float(ks_per_pair.mean())
        # Magnitude statistic — log10 ratio of max abs_mean to min abs_mean across syms
        am = np.array([stats_per_sym[s]["abs_mean"][fi] for s in SYMS], dtype=np.float64)
        if am.min() > 1e-10:
            mag_ratio = float(am.max() / am.min())
        else:
            mag_ratio = float("inf") if am.max() > 1e-10 else 1.0
        # std ratio
        sds = np.array([stats_per_sym[s]["std"][fi] for s in SYMS], dtype=np.float64)
        if sds.min() > 1e-10:
            std_ratio = float(sds.max() / sds.min())
        else:
            std_ratio = float("inf") if sds.max() > 1e-10 else 1.0
        row = {
            "feat_idx": fi,
            "feat_name": feat_names[fi],
            "ks_max": max_ks,
            "ks_mean": mean_ks,
            "abs_mean_ratio_max_min": mag_ratio,
            "std_ratio_max_min": std_ratio,
        }
        for k, (a, b) in enumerate(PAIRS):
            row[f"ks_{a}_{b}"] = float(ks_per_pair[k])
        for s in SYMS:
            row[f"std_sym{s}"] = float(stats_per_sym[s]["std"][fi])
            row[f"absmean_sym{s}"] = float(stats_per_sym[s]["abs_mean"][fi])
        rows.append(row)
        if (fi + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"  [{fi+1}/{F}] KS done, {elapsed:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.sort_values("ks_max", ascending=False, inplace=True)
    out_full = os.path.join(HERE, "all_ks_features.csv")
    df.to_csv(out_full, index=False)
    print(f"  wrote {out_full}", flush=True)

    out_top = os.path.join(HERE, "top_ks_features.csv")
    df.head(50).to_csv(out_top, index=False)
    print(f"  wrote {out_top}", flush=True)

    print("\n=== TOP 30 worst KS features ===", flush=True)
    cols_show = ["feat_idx", "feat_name", "ks_max", "ks_mean",
                 "abs_mean_ratio_max_min", "std_ratio_max_min"]
    print(df[cols_show].head(30).to_string(index=False), flush=True)

    # Categorize
    target_substrs = ["amount_delta", "spread", "bid_diff", "ask_diff", "cumspread"]
    in_cat = df["feat_name"].apply(lambda n: any(s in n for s in target_substrs))
    in_cat_no_norm = df["feat_name"].apply(
        lambda n: any(s in n for s in target_substrs) and not any(p in n for p in ["dualz_", "qrank_", "_ratio"])
    )
    print(f"\nTarget category coverage in top 30: {int(in_cat.iloc[:30].sum())}/30",
          flush=True)
    print(f"Target category (raw, no dualz/qrank) coverage in top 30: "
          f"{int(in_cat_no_norm.iloc[:30].sum())}/30", flush=True)

    summary = {
        "n_features": F,
        "top30_target_count": int(in_cat.iloc[:30].sum()),
        "top30_raw_target_count": int(in_cat_no_norm.iloc[:30].sum()),
        "elapsed_sec": time.time() - t0,
        "top30": df[cols_show].head(30).to_dict(orient="records"),
    }
    with open(os.path.join(HERE, "ks_audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
