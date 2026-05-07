"""Validate that the 6 time features are sym-invariant.

Each (date, sess, t) row is replicated for sym 0..4 with identical t/sess_idx,
so the time-feature distributions across syms must be **identical**. We confirm
this with a 2-sample KS test (p-value should be 1.0 for all pairs) plus a direct
elementwise equality check on aligned rows.

Output: sym_invariance_report.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

CACHE = os.path.join(HERE, "cache")
TIME_FEAT_NAMES = [
    "sec_norm",
    "sin_sess_pos",
    "cos_sess_pos",
    "is_open_30min",
    "is_close_30min",
    "is_pm",
]
SYMS = (0, 1, 2, 3, 4)


def main():
    p = os.path.join(CACHE, "schemeQ_train.npz")
    print(f"loading {p} ...", flush=True)
    d = np.load(p)
    X = d["X"]  # (N, 376)
    sym = d["sym"]
    t = d["t"]
    sess = d["sess_idx"]
    n_total_dim = X.shape[1]
    print(f"  X shape={X.shape}", flush=True)

    # 6 time features are last 6 columns
    tfeat = X[:, -6:]
    report = {"feat_names": TIME_FEAT_NAMES, "ks_pairwise": {}, "exact_match": {}}

    for fi, fname in enumerate(TIME_FEAT_NAMES):
        # Pairwise KS between syms (subsample 50k each for speed)
        rng = np.random.default_rng(42)
        sub = {}
        for k in SYMS:
            mask = sym == k
            arr = tfeat[mask, fi]
            n = min(len(arr), 50_000)
            idx = rng.choice(len(arr), size=n, replace=False)
            sub[k] = arr[idx]

        worst_p = 1.0
        for i in range(len(SYMS)):
            for j in range(i + 1, len(SYMS)):
                a = sub[SYMS[i]]; b = sub[SYMS[j]]
                _, p = ks_2samp(a, b)
                worst_p = min(worst_p, float(p))
        report["ks_pairwise"][fname] = {"worst_p": worst_p}
        print(f"  {fname:18s}  worst KS p-value = {worst_p:.4f}", flush=True)

    # Exact match: align rows by (sym, date, sess, t) and confirm features are identical
    # across syms for the same (date, sess, t).
    # Pick sym=0 baseline; for sym>0, sample 1000 rows and verify time features identical.
    # (By construction t and sess_idx fully determine the time features.)
    exact_ok = True
    sample_rng = np.random.default_rng(7)
    base_mask = sym == 0
    base_idx = np.where(base_mask)[0]
    sample = sample_rng.choice(base_idx, size=2000, replace=False)
    for k in SYMS[1:]:
        ok_count = 0
        for i in sample:
            t0 = t[i]; sess0 = sess[i]
            # find a row in sym=k with same (t, sess) (don't need same date)
            cand_mask = (sym == k) & (t == t0) & (sess == sess0)
            cand_idx = np.where(cand_mask)[0]
            if len(cand_idx) == 0:
                continue
            j = cand_idx[0]
            if np.allclose(tfeat[i], tfeat[j], rtol=1e-6, atol=1e-7):
                ok_count += 1
            else:
                exact_ok = False
        report["exact_match"][f"sym0_vs_sym{k}_n_match"] = int(ok_count)
        print(f"  exact match sym0 vs sym{k}: {ok_count}/2000", flush=True)
    report["exact_invariant"] = bool(exact_ok)

    out_path = os.path.join(HERE, "sym_invariance_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
