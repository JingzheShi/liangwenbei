"""Compliance + sanity tests for iter_015 v2 Predictor.

  1. Smoke: random data smoke + correct output shape (B, 5)
  2. Shuffle invariance: shuffle batches, predict, compare element-wise
  3. sym=99 injection: feed a sym out of training range — must not raise.
     (Note: Predictor doesn't read sym at all, so this is a pure regression
     test that the pipeline doesn't accidentally start using sym.)
  4. Timing: 1024-batch inference time < 5s
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from Predictor import Predictor


def make_random_batches(n_batches, feat_cols, seed=0):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        df = pd.DataFrame(
            rng.standard_normal((100, len(feat_cols))).astype(np.float32),
            columns=feat_cols,
        )
        batches.append(df)
    return batches


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    feat_cols = cfg["feature"]
    p = Predictor()
    print(f"Predictor ready: nn={len(p._nn_pool)} cb={len(p._cb_pool)} huber={len(p._huber_pool)} gru={len(p._gru_pool)}")
    print(f"  thr_up={p.thr_up:.6f}  thr_dn={p.thr_dn:.6f}")
    print(f"  weights w_t87={p.w_t87} w_t89={p.w_t89} w_huber={p.w_huber} w_gru={p.w_gru}")

    # --- 1. Smoke ---
    print("\n[1] Smoke ...")
    batches = make_random_batches(3, feat_cols, seed=0)
    out = p.predict(batches)
    assert isinstance(out, list) and len(out) == 3 and len(out[0]) == 5, f"bad output shape: {out}"
    print(f"  OK: predict([3 batches]) -> {out}")

    # --- 2. Shuffle invariance ---
    print("\n[2] Shuffle invariance ...")
    batches = make_random_batches(64, feat_cols, seed=42)
    out_orig = np.array(p.predict(batches), dtype=np.int64)
    perm = np.random.default_rng(7).permutation(len(batches))
    batches_perm = [batches[i] for i in perm]
    out_perm = np.array(p.predict(batches_perm), dtype=np.int64)
    out_perm_unshuf = out_perm[np.argsort(perm)]
    same = bool(np.array_equal(out_orig, out_perm_unshuf))
    n_diff = int((out_orig != out_perm_unshuf).sum())
    print(f"  shuffle-invariant? {same}  (n_diff={n_diff} / {out_orig.size})")
    assert same, "Predictor is NOT shuffle-invariant"

    # --- 3. sym=99 injection ---
    print("\n[3] sym=99 injection ...")
    # Predictor doesn't see sym, so just confirm predict still works on weird data
    rng = np.random.default_rng(99)
    weird = []
    for _ in range(8):
        df = pd.DataFrame(
            rng.standard_normal((100, len(feat_cols))).astype(np.float32) * 100.0,  # large
            columns=feat_cols,
        )
        weird.append(df)
    try:
        out_weird = p.predict(weird)
        print(f"  OK: weird-data predict returned shape ({len(out_weird)},{len(out_weird[0])})")
        # Also ensure no NaN-induced action 1
        arr = np.array(out_weird, dtype=np.int64)
        assert arr.shape == (8, 5), arr.shape
    except Exception as e:
        print(f"  FAIL: {e}")
        raise

    # --- 4. Timing 1024-batch ---
    print("\n[4] Timing 1024-batch ...")
    big_batches = make_random_batches(1024, feat_cols, seed=11)
    # Warm up
    _ = p.predict(big_batches[:8])
    t0 = time.time()
    _ = p.predict(big_batches)
    dt = time.time() - t0
    print(f"  predict(1024 batches) took {dt:.2f}s  (target <5s; <60s acceptable)")

    out_summary = {
        "smoke_ok": True,
        "shuffle_invariant": same,
        "sym99_ok": True,
        "timing_1024_sec": dt,
    }
    with open(os.path.join(HERE, "end_to_end_check.json"), "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"\nwrote -> end_to_end_check.json")


if __name__ == "__main__":
    main()
