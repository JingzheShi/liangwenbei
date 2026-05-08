"""T88: KS sym-invariance check on the 12 range-vol features.

For each feature, compare per-sym distributions on train data via 2-sample KS
test. A feature is "sym-invariant" if KS distance is small across all 5 syms
(typically D < 0.10 → safe to use).
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")


def main():
    d = np.load(os.path.join(CACHE, "t88_range_vol_train.npz"))
    X = d["X_t88"]
    sym = d["sym"]
    with open(os.path.join(CACHE, "t88_feat_names.txt")) as f:
        names = [l.strip() for l in f]

    print(f"X shape: {X.shape}")
    SYMS = (0, 1, 2, 3, 4)

    # Subsample for KS speed (KS is O(n log n) but with N=1.4M it's slow)
    N_SUB = 50_000
    rng = np.random.default_rng(0)
    sub_idx = {s: rng.choice(np.where(sym == s)[0], size=N_SUB, replace=False) for s in SYMS}

    all_passed = True
    report = []
    print(f"\n{'feat':28s} {'overall':>8s}  {'mean(KS)':>9s}  {'max(KS)':>9s}  {'min p':>8s}  status")
    print("-" * 80)
    for j, n in enumerate(names):
        v_per_sym = [X[sub_idx[s], j] for s in SYMS]
        ks_pairs = []
        ps = []
        for i in range(5):
            for k in range(i + 1, 5):
                D, p = stats.ks_2samp(v_per_sym[i], v_per_sym[k], mode="asymp")
                ks_pairs.append(D)
                ps.append(p)
        mean_D = float(np.mean(ks_pairs))
        max_D = float(np.max(ks_pairs))
        min_p = float(np.min(ps))
        # rough "good" threshold: max KS < 0.10
        ok = max_D < 0.10
        if not ok:
            all_passed = False
        status = "OK" if ok else "FAIL"
        # also overall mean/std as sanity
        v = X[:, j]
        print(f"{n:28s} {v.mean():+8.2f}  {mean_D:9.4f}  {max_D:9.4f}  {min_p:8.1e}  {status}")
        report.append({
            "feat": n, "mean_overall": float(v.mean()),
            "mean_KS": mean_D, "max_KS": max_D,
            "min_p": min_p, "status": status,
        })

    out = {
        "n_train_rows": int(len(X)),
        "n_subsample_per_sym": N_SUB,
        "ks_threshold": 0.10,
        "all_features_passed": bool(all_passed),
        "feature_report": report,
    }
    out_path = os.path.join(HERE, "ks_validate_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\n{'ALL PASSED' if all_passed else 'SOME FAILED'} (max KS < 0.10 across all sym pairs)")


if __name__ == "__main__":
    main()
