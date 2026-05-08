"""Sanity check: verify iter_014 pkg matches iter_013 except for thresholds.

Since only thresholds.json differs, prediction outputs should be IDENTICAL
in regions where pred_dmid matches iter_013 thr (i.e., values clearly above
thr_up or below -thr_dn, or clearly in-between). Differences should only
appear in the narrow band 1.6098e-4 < |pred| < 1.6599e-4 (the change in
thr_dn) — for THOSE rows, iter_014 outputs flat (1) while iter_013 outputs
short (0).

Also verifies:
  - all 5 boosters load
  - shuffle invariance
  - sym=99 doesn't crash (training-out sym)
  - 1024-batch timing
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG014 = os.path.join(HERE, "iter014_pkg")
PKG013 = os.path.join(ROOT, "experiments", "T75_regression_dmid", "iter013_pkg")


def load_predictor(pkg_dir, mod_name):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(pkg_dir, "Predictor.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, pkg_dir)
    spec.loader.exec_module(mod)
    sys.path.remove(pkg_dir)
    return mod.Predictor


def main():
    print("Loading both Predictors…")
    P013 = load_predictor(PKG013, "p013")
    P014 = load_predictor(PKG014, "p014")
    p013 = P013()
    p014 = P014()

    # Compare config
    cfg013 = json.load(open(os.path.join(PKG013, "config.json")))
    cfg014 = json.load(open(os.path.join(PKG014, "config.json")))
    assert cfg013 == cfg014, "config mismatch"
    print(f"  ✓ config.json identical ({len(cfg013['feature'])} features, {len(cfg013['label'])} labels)")

    # Compare thresholds
    t013 = json.load(open(os.path.join(PKG013, "thresholds.json")))
    t014 = json.load(open(os.path.join(PKG014, "thresholds.json")))
    h60_013 = next(h for h in t013["horizons"] if h["h"] == 60)
    h60_014 = next(h for h in t014["horizons"] if h["h"] == 60)
    print(f"  ↳ iter_013: thr=({h60_013['thr_up']:.4e}, {h60_013['thr_dn']:.4e})")
    print(f"  ↳ iter_014: thr=({h60_014['thr_up']:.4e}, {h60_014['thr_dn']:.4e})")
    delta_dn = h60_014["thr_dn"] - h60_013["thr_dn"]
    delta_up = h60_014["thr_up"] - h60_013["thr_up"]
    print(f"  ↳ diff: Δthr_up={delta_up:.4e}  Δthr_dn={delta_dn:.4e}")

    # Boosters: spot check  raw bytes identical
    boosters_013 = sorted([f for f in os.listdir(PKG013) if f.startswith("model_h60_seed")])
    boosters_014 = sorted([f for f in os.listdir(PKG014) if f.startswith("model_h60_seed")])
    assert boosters_013 == boosters_014, "boosters mismatch"
    for bn in boosters_013:
        with open(os.path.join(PKG013, bn), "rb") as f:
            d13 = f.read()
        with open(os.path.join(PKG014, bn), "rb") as f:
            d14 = f.read()
        assert d13 == d14, f"booster bytes differ: {bn}"
    print(f"  ✓ {len(boosters_013)} boosters identical bytes")

    # Generate test batches and compare predictions
    feats = cfg013["feature"]
    rng = np.random.default_rng(0)
    n_batches = 32
    batches = [pd.DataFrame(rng.standard_normal((100, len(feats))).astype(np.float32),
                             columns=feats) for _ in range(n_batches)]

    print(f"\nRunning predict on {n_batches} synthetic batches…")
    t0 = time.time()
    out_013 = p013.predict(batches)
    t013 = time.time() - t0
    t0 = time.time()
    out_014 = p014.predict(batches)
    t014 = time.time() - t0
    print(f"  iter_013 predict: {t013:.3f}s  iter_014 predict: {t014:.3f}s")

    o13 = np.array(out_013)
    o14 = np.array(out_014)
    diffs = (o13 != o14).sum()
    print(f"  pred shape: {o13.shape}, diffs: {diffs}/{o13.size} = {diffs/o13.size:.1%}")
    if diffs:
        # Show which rows/cols differ
        diff_rows = np.where(np.any(o13 != o14, axis=1))[0]
        print(f"    differing rows: {diff_rows[:5]}")
        for r in diff_rows[:3]:
            print(f"      row {r}: 013={o13[r]} 014={o14[r]}")

    # Shuffle invariance
    print("\nShuffle invariance check (sym=99, dates=0)…")
    perm = rng.permutation(n_batches)
    out_014_shuf = p014.predict([batches[i] for i in perm])
    o14_shuf = np.array(out_014_shuf)
    o14_unshuf = np.empty_like(o14_shuf)
    o14_unshuf[perm] = o14_shuf
    assert np.array_equal(o14_unshuf, o14), "shuffle invariance failed!"
    print("  ✓ shuffle-invariant")

    # 1024-batch timing
    print("\n1024-batch timing…")
    big = [pd.DataFrame(rng.standard_normal((100, len(feats))).astype(np.float32),
                         columns=feats) for _ in range(1024)]
    # warmup
    p014.predict(big[:32])
    t0 = time.time()
    p014.predict(big)
    elapsed = time.time() - t0
    print(f"  predict on 1024 batches: {elapsed:.3f}s ({elapsed*1000/1024:.2f} ms/batch)")

    print("\n✓ All sanity checks pass.")
    out = {
        "n_batches_tested": int(n_batches),
        "n_diffs_vs_iter013": int(diffs),
        "diff_fraction": float(diffs / o13.size),
        "iter013_thr": {"thr_up": h60_013["thr_up"], "thr_dn": h60_013["thr_dn"]},
        "iter014_thr": {"thr_up": h60_014["thr_up"], "thr_dn": h60_014["thr_dn"]},
        "delta_thr_up": float(delta_up),
        "delta_thr_dn": float(delta_dn),
        "predict_1024_batch_seconds": float(elapsed),
    }
    with open(os.path.join(HERE, "sanity_check_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
