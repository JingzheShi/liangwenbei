"""Sym-invariance KS test for T55 schemeP cache.

Layout: 226 base | 54 stage1 | 59 stage2 | 14 stage3 | 47 stage4 = 400 cols

For each extra feature (174 total):
  - per-sym (mean, std)
  - max pairwise KS statistic across 5 sym pairs (subsampled to 30k each)
  - flag: PASS (<=0.30), WARN (0.30-0.50), FAIL (>0.50)

Writes sym_invariance_report.json with full table + summary.
Stage 1/2/3 already validated; new FAIL list focuses on stage4.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CACHE_DIR = os.path.join(HERE, "cache")
N_BASE = 226
N_STAGE1 = 54
N_STAGE2 = 59
N_STAGE3 = 14
N_STAGE12 = N_STAGE1 + N_STAGE2  # 113
N_STAGE123 = N_STAGE12 + N_STAGE3  # 127


def stage_of(j: int) -> str:
    if j < N_STAGE1:
        return "stage1"
    if j < N_STAGE12:
        return "stage2"
    if j < N_STAGE123:
        return "stage3"
    return "stage4"


def main():
    extras_path = os.path.join(CACHE_DIR, "schemeP_extra_feat_names.txt")
    with open(extras_path) as f:
        names = [line.strip() for line in f]
    n_extra = len(names)
    print(f"validating {n_extra} extras (stage1+2+3+4)", flush=True)

    d = np.load(os.path.join(CACHE_DIR, "schemeP_train.npz"))
    X = d["X"][:, N_BASE:N_BASE + n_extra]
    sym = d["sym"]
    assert X.shape[1] == n_extra, f"X has {X.shape[1]} extras, expected {n_extra}"

    SUBSAMPLE = 30_000
    rng = np.random.default_rng(0)
    sym_to_idx = {s: np.where(sym == s)[0] for s in range(5)}
    sym_to_sub = {s: rng.choice(idxs, size=min(SUBSAMPLE, len(idxs)), replace=False)
                  for s, idxs in sym_to_idx.items()}

    rows = []
    print(f"{'feature':40s}  {'min_mean':>10s} {'max_mean':>10s}  "
          f"{'min_std':>10s} {'max_std':>10s}  {'max_KS':>8s} {'flag':>6s}", flush=True)
    print("-" * 110)
    for j, name in enumerate(names):
        col = X[:, j].astype(np.float64)
        per_sym_means = np.array([col[sym == s].mean() for s in range(5)])
        per_sym_stds = np.array([col[sym == s].std() for s in range(5)])
        ks_max = 0.0
        for a in range(5):
            for b in range(a + 1, 5):
                ks_stat, _ = stats.ks_2samp(col[sym_to_sub[a]], col[sym_to_sub[b]])
                if ks_stat > ks_max:
                    ks_max = float(ks_stat)
        flag = "PASS" if ks_max <= 0.30 else ("WARN" if ks_max <= 0.50 else "FAIL")
        st = stage_of(j)
        marker = f" <-- {st}" if st == "stage4" else ""
        print(f"{name:40s}  {per_sym_means.min():10.4g} {per_sym_means.max():10.4g}  "
              f"{per_sym_stds.min():10.4g} {per_sym_stds.max():10.4g}  "
              f"{ks_max:8.3f} {flag:>6s}{marker}", flush=True)
        rows.append({
            "feat": name,
            "stage": st,
            "per_sym_mean": per_sym_means.tolist(),
            "per_sym_std": per_sym_stds.tolist(),
            "max_ks_stat": ks_max,
            "flag": flag,
        })

    print("-" * 110)
    pass_n = sum(1 for r in rows if r['flag'] == 'PASS')
    warn_n = sum(1 for r in rows if r['flag'] == 'WARN')
    fail_n = sum(1 for r in rows if r['flag'] == 'FAIL')
    print(f"OVERALL: PASS={pass_n}/{n_extra} | WARN={warn_n}/{n_extra} | FAIL={fail_n}/{n_extra}",
          flush=True)
    pass4 = sum(1 for r in rows if r["stage"] == "stage4" and r['flag'] == 'PASS')
    warn4 = sum(1 for r in rows if r["stage"] == "stage4" and r['flag'] == 'WARN')
    fail4 = sum(1 for r in rows if r["stage"] == "stage4" and r['flag'] == 'FAIL')
    print(f"STAGE 4 ONLY: PASS={pass4} | WARN={warn4} | FAIL={fail4}", flush=True)

    fail_names = [r["feat"] for r in rows if r["flag"] == "FAIL"]
    print(f"\nFAIL list (will be dropped from training):", flush=True)
    for fn in fail_names:
        print(f"  - {fn}", flush=True)

    out = os.path.join(HERE, "sym_invariance_report.json")
    with open(out, "w") as f:
        json.dump({"rows": rows, "summary": {
            "pass": pass_n, "warn": warn_n, "fail": fail_n, "total": n_extra,
            "stage4": {"pass": pass4, "warn": warn4, "fail": fail4},
            "fail_names": fail_names,
        }}, f, indent=2)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
