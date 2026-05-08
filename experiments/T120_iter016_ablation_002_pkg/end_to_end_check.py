"""End-to-end smoke for the T119 ablation Predictor.

Checks:
  1. predict() runs and returns a (B, 5) list
  2. shuffle-invariance: permuting the input batch order yields permuted output
  3. statelessness: calling predict twice on same input yields identical output
  4. action distribution: h=5/10/20/40 must be all-flat (active=false in thresholds)
  5. timing on a 4096-sample call (proxy for one platform predict() invocation)
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "pkg")
sys.path.insert(0, PKG)

from Predictor import Predictor, RAW_COLS_TRAIN_ORDER  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T95_CACHE = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob", "cache")


def _load_real_windows(n=4096, seed=7):
    """Load real LOB windows + extend with synthetic non-LOB cols for the 154-col layout."""
    lob = np.load(os.path.join(T95_CACHE, "lob_test_top5.npz"))
    lob_arr = lob["lob_arr"]
    sess_id = lob["sess_id_for_sample"]
    t_for_sample = lob["t_for_sample"].astype(np.int32)

    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sess_id), size=n)
    sids = sess_id[pick]
    ts = t_for_sample[pick]
    idx_w = ts[:, None] + np.arange(-99, 1, dtype=np.int32)[None, :]
    lob_win = lob_arr[sids[:, None], idx_w, :]

    GRU_RAW_COLS = []
    for i in range(1, 6):
        GRU_RAW_COLS.extend([f"bid{i}", f"bsize{i}", f"ask{i}", f"asize{i}"])

    full_cols = list(RAW_COLS_TRAIN_ORDER)
    K = len(full_cols)
    col_to_idx = {c: i for i, c in enumerate(full_cols)}

    base = rng.standard_normal((n, 100, K)).astype(np.float32) * 0.01
    for c in full_cols:
        if c.startswith("bid") and not c.startswith("bid_") and not c.startswith("bsize") and "diff" not in c and "rate" not in c and "mean" not in c:
            base[:, :, col_to_idx[c]] = 1.0 + base[:, :, col_to_idx[c]]
        if c.startswith("ask") and not c.startswith("ask_") and not c.startswith("asize") and "diff" not in c and "rate" not in c and "mean" not in c:
            base[:, :, col_to_idx[c]] = 1.0 + base[:, :, col_to_idx[c]]
        if c.startswith("midprice"):
            base[:, :, col_to_idx[c]] = 1.0 + base[:, :, col_to_idx[c]]
        if c.startswith("close") or c.startswith("open") or c.startswith("high") or c.startswith("low"):
            base[:, :, col_to_idx[c]] = 1.0 + base[:, :, col_to_idx[c]]

    for j, c in enumerate(GRU_RAW_COLS):
        base[:, :, col_to_idx[c]] = lob_win[:, :, j]

    return base


def main():
    n = 4096
    print(f"=== T119 end_to_end_check (n={n}) ===", flush=True)
    X = _load_real_windows(n=n)
    print(f"  loaded windows shape={X.shape}", flush=True)

    batches = [pd.DataFrame(X[i], columns=list(RAW_COLS_TRAIN_ORDER)) for i in range(n)]

    p = Predictor()
    has_long = p._h_long is not None
    short_active = sorted(p._h_short.keys())
    print(f"  h_long active: {has_long}  h_short active: {short_active}", flush=True)

    t0 = time.time()
    out1 = p.predict(batches)
    t_first = time.time() - t0
    out1_arr = np.array(out1, dtype=np.int64)
    print(f"  first predict({n}) -> shape={out1_arr.shape} in {t_first:.2f}s", flush=True)

    for j, h in enumerate([5, 10, 20, 40, 60]):
        col = out1_arr[:, j]
        c0 = int((col == 0).sum()); c1 = int((col == 1).sum()); c2 = int((col == 2).sum())
        print(f"    h={h:>2}: short={c0:5d}  flat={c1:5d}  long={c2:5d}", flush=True)

    # Short horizons must be all-flat
    for j, h in enumerate([5, 10, 20, 40]):
        col = out1_arr[:, j]
        n_active = int((col != 1).sum())
        assert n_active == 0, f"h={h} expected all flat but got {n_active} active"

    # Statelessness
    out2 = p.predict(batches)
    out2_arr = np.array(out2, dtype=np.int64)
    eq = bool(np.array_equal(out1_arr, out2_arr))
    print(f"  stateless check (call#2 == call#1): {eq}", flush=True)
    assert eq, "Predictor is NOT stateless"

    # Shuffle-invariance
    perm = np.random.default_rng(123).permutation(n)
    inv = np.argsort(perm)
    batches_p = [batches[i] for i in perm]
    out3 = p.predict(batches_p)
    out3_arr = np.array(out3, dtype=np.int64)[inv]
    eq2 = bool(np.array_equal(out1_arr, out3_arr))
    print(f"  shuffle-invariance check: {eq2}", flush=True)
    assert eq2, "Predictor is NOT shuffle-invariant"

    extrapolated_442k = t_first / n * 442_080
    print(f"\n  timing: {t_first*1000/n:.2f} ms/sample  -> 442k = {extrapolated_442k:.1f}s", flush=True)

    rep = {
        "n": n,
        "predict_time_seconds": t_first,
        "extrapolated_442k_seconds": extrapolated_442k,
        "stateless_ok": eq,
        "shuffle_invariant_ok": eq2,
        "h_long_active": has_long,
        "h_short_active": short_active,
        "action_distribution": {
            f"h{h}": {"short": int((out1_arr[:, j] == 0).sum()),
                      "flat": int((out1_arr[:, j] == 1).sum()),
                      "long": int((out1_arr[:, j] == 2).sum())}
            for j, h in enumerate([5, 10, 20, 40, 60])
        },
    }
    out_path = os.path.join(HERE, "end_to_end_check.json")
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\n  wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
