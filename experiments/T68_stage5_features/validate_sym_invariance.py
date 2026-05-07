"""KS sym-invariance test for the 20 Stage 5 features.

For each feature, run KS-2sample between syms within the train cache.
A feature is FAIL if the worst pairwise KS-stat exceeds threshold (0.10),
matching the convention used in T44/T51/T53.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from typing import List

import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))


def ks_test(arr_per_sym: List[np.ndarray]) -> dict:
    K = len(arr_per_sym)
    pairs = list(combinations(range(K), 2))
    stats = []
    for i, j in pairs:
        x = arr_per_sym[i]
        y = arr_per_sym[j]
        # Subsample to 50k to keep KS fast
        n = min(50_000, len(x), len(y))
        rng = np.random.default_rng(0)
        if len(x) > n:
            x = rng.choice(x, n, replace=False)
        if len(y) > n:
            y = rng.choice(y, n, replace=False)
        s, p = ks_2samp(x, y)
        stats.append((i, j, float(s), float(p)))
    worst = max(stats, key=lambda t: t[2])
    return {"pairs": stats, "worst_pair": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(HERE, "cache", "schemeP_train.npz"))
    ap.add_argument("--names", default=os.path.join(HERE, "cache", "schemeP_feat_names.txt"))
    ap.add_argument("--out", default=os.path.join(HERE, "sym_invariance_report.json"))
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="Fail if worst pairwise KS > threshold")
    args = ap.parse_args()

    print(f"Loading {args.cache}")
    d = np.load(args.cache)
    X = d["X"]
    sym = d["sym"]
    with open(args.names) as f:
        names = [l.strip() for l in f]

    # Stage 5 occupies last 20 cols
    s5_start = X.shape[1] - 20
    s5_end = X.shape[1]
    print(f"Stage 5 columns: {s5_start}..{s5_end}")
    print(f"Names: {names[s5_start:s5_end]}")

    syms = np.unique(sym)
    print(f"syms: {syms}")

    fail_features: List[str] = []
    report = {}
    for ci in range(s5_start, s5_end):
        arr_per_sym = [X[sym == s, ci].astype(np.float64) for s in syms]
        # drop infinities/NaN if any
        arr_per_sym = [a[np.isfinite(a)] for a in arr_per_sym]
        result = ks_test(arr_per_sym)
        worst_stat = result["worst_pair"][2]
        feat = names[ci]
        report[feat] = {
            "worst_ks": worst_stat,
            "worst_pair": result["worst_pair"][:2],
            "all_pairs": [(i, j, s) for (i, j, s, _) in result["pairs"]],
            "pass": worst_stat <= args.threshold,
        }
        marker = "PASS" if worst_stat <= args.threshold else "FAIL"
        print(f"  [{marker}] {feat}: worst_ks={worst_stat:.4f}")
        if worst_stat > args.threshold:
            fail_features.append(feat)

    report["_meta"] = {
        "threshold": args.threshold,
        "fail_features": fail_features,
        "n_total_tested": s5_end - s5_start,
        "n_fail": len(fail_features),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote {args.out}")
    print(f"FAIL: {len(fail_features)} / {s5_end - s5_start}")
    print(fail_features)


if __name__ == "__main__":
    main()
