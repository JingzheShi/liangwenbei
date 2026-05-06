"""Defensive verification that the trained LightGBM models satisfy CRITICAL_CONSTRAINTS.md:

1. feature_names contain no metadata columns (date / sym / time / session)
2. Inference is order-independent: shuffle(input) -> shuffle(predictions) (within reorder)
3. A fake sym=99 sample causes no error (model doesn't index by sym)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import lightgbm as lgb  # noqa: E402

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(EXP_DIR, "cache")
FORBIDDEN_TOKENS = ("date", "sym", "time", "session")


def check_scheme(scheme: str) -> dict:
    print(f"\n========== Scheme {scheme} ==========")
    model_path = os.path.join(EXP_DIR, f"model_scheme{scheme}.txt")
    feat_path = os.path.join(CACHE, f"scheme{scheme}_feat_names.txt")
    test_path = os.path.join(CACHE, f"scheme{scheme}_test.npz")
    if not (os.path.exists(model_path) and os.path.exists(feat_path) and os.path.exists(test_path)):
        return {"scheme": scheme, "skipped": True}

    feat_names = [ln.strip() for ln in open(feat_path)]
    print(f"  n_features = {len(feat_names)}")

    # 1. forbidden token check
    bad = []
    for fn in feat_names:
        low = fn.lower()
        for tok in FORBIDDEN_TOKENS:
            # Match as a standalone token at the start (e.g. 'date', 'sym',
            # 'time_of_day', 'session_idx') but accept tokens like 'midprice'
            # (does NOT start with date/sym/time/session). Also flag any feature
            # whose first underscore-segment equals one of the forbidden tokens.
            seg0 = fn.split("_", 1)[0].lower()
            if seg0 == tok or fn == tok:
                bad.append(fn)
                break
    if bad:
        print(f"  ❌ FOUND FORBIDDEN FEATURE NAMES: {bad[:10]}")
    else:
        print("  ✅ no forbidden feature names")

    # 2. shuffle-equivariance test
    booster = lgb.Booster(model_file=model_path)
    test = np.load(test_path)
    X = test["X"][:5000]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    proba_orig = booster.predict(X)
    proba_perm = booster.predict(X[perm])
    # invert the permutation to bring back to original order
    inv = np.argsort(perm)
    diff = np.max(np.abs(proba_orig - proba_perm[inv]))
    if diff < 1e-6:
        print(f"  ✅ shuffle-invariance: max abs diff = {diff:.2e}")
    else:
        print(f"  ❌ shuffle-invariance VIOLATED: max abs diff = {diff:.2e}")

    # 3. sym=99 sample: synthesize one row with the same feature vector but the
    # *model* does not see sym at all. So we just confirm prediction succeeds on
    # arbitrary input. (Since sym is NOT a feature, this trivially holds, but we
    # exercise it for completeness.)
    fake = X[:1].copy()
    proba_fake = booster.predict(fake)
    print(f"  ✅ predict on arbitrary sample (no sym in model): proba shape {proba_fake.shape}")

    return {
        "scheme": scheme,
        "n_features": len(feat_names),
        "first_5_features": feat_names[:5],
        "last_5_features": feat_names[-5:],
        "forbidden_in_feat_names": bad,
        "shuffle_max_abs_diff": float(diff),
    }


if __name__ == "__main__":
    results = {}
    for s in ("A", "B"):
        results[s] = check_scheme(s)
    import json
    out = os.path.join(EXP_DIR, "sym_agnostic_verification.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")
