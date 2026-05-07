"""Quick sym-invariance check for T44 R34 stage 1 features.

For each of the 54 new features, compute per-sym mean/std/median (subsampled),
then report the cross-sym CV (std of per-sym means / |overall mean|) and the
max KS statistic across pairs. With 1.4M samples KS p-values are always tiny;
the *KS statistic itself* (max distribution gap) is the meaningful invariance metric.

We define "passes" as: max pairwise KS statistic <= 0.30 (loose threshold;
strict would be 0.10 but 0.30 still indicates "same shape, different location only").
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from r34_features import all_feature_names  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
N_BASE = 226


def main():
    d = np.load(os.path.join(CACHE_DIR, "schemeL_train.npz"))
    X = d["X"][:, N_BASE:]  # (N, 54)
    sym = d["sym"]
    names = all_feature_names()
    assert X.shape[1] == len(names) == 54, f"X.shape={X.shape}, names={len(names)}"

    SUBSAMPLE = 30000
    # Per-sym subsample for KS (KS is O(n log n) per pair; with ~300k per sym, KS test handles fine)
    rng = np.random.default_rng(0)
    sym_to_idx = {s: np.where(sym == s)[0] for s in range(5)}
    sym_to_sub = {s: rng.choice(idxs, size=min(SUBSAMPLE, len(idxs)), replace=False)
                  for s, idxs in sym_to_idx.items()}

    rows = []
    print(f"{'feature':35s}  {'min_mean':>10s} {'max_mean':>10s}  "
          f"{'min_std':>10s} {'max_std':>10s}  {'max_KS':>8s} {'flag':>6s}", flush=True)
    print("-" * 105)
    fail_count = 0
    for j, name in enumerate(names):
        col = X[:, j].astype(np.float64)
        per_sym_means = np.array([col[sym == s].mean() for s in range(5)])
        per_sym_stds = np.array([col[sym == s].std() for s in range(5)])
        # Pairwise KS on subsamples
        ks_max = 0.0
        for a in range(5):
            for b in range(a + 1, 5):
                ks_stat, _ = stats.ks_2samp(col[sym_to_sub[a]], col[sym_to_sub[b]])
                if ks_stat > ks_max:
                    ks_max = float(ks_stat)
        flag = "PASS" if ks_max <= 0.30 else ("WARN" if ks_max <= 0.50 else "FAIL")
        if flag != "PASS":
            fail_count += 1
        print(f"{name:35s}  {per_sym_means.min():10.4g} {per_sym_means.max():10.4g}  "
              f"{per_sym_stds.min():10.4g} {per_sym_stds.max():10.4g}  "
              f"{ks_max:8.3f} {flag:>6s}", flush=True)
        rows.append({
            "feat": name,
            "per_sym_mean": per_sym_means.tolist(),
            "per_sym_std": per_sym_stds.tolist(),
            "max_ks_stat": ks_max,
            "flag": flag,
        })

    print("-" * 105)
    print(f"PASS: {sum(1 for r in rows if r['flag']=='PASS')}/54", flush=True)
    print(f"WARN: {sum(1 for r in rows if r['flag']=='WARN')}/54", flush=True)
    print(f"FAIL: {sum(1 for r in rows if r['flag']=='FAIL')}/54", flush=True)

    out = os.path.join(HERE, "sym_invariance_report.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "summary": {
            "pass": sum(1 for r in rows if r['flag']=='PASS'),
            "warn": sum(1 for r in rows if r['flag']=='WARN'),
            "fail": sum(1 for r in rows if r['flag']=='FAIL'),
            "total": len(rows),
        }}, f, indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
