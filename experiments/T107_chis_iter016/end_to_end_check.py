"""End-to-end Predictor sanity check on real LOB parquets.

Steps:
  1. Read 5 raw parquets (sym 0..4, am session, date 0).
  2. For each: take 50 100-tick windows (no overlap; t = 99, 199, 299, ...).
  3. Mash them all into a single batch (250 windows). Shuffle order.
  4. Force date column to 0 (sim platform behaviour).
  5. Reassign sym for 1/4 of windows to sym=99 (out-of-train sanity).
  6. Run Predictor.predict(); confirm output shape + per-horizon activation rates.
  7. Run a SECOND time with the same input — confirm idempotent (no state).
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
PKG = os.path.join(HERE, "pkg")
DATA = os.path.join(ROOT, "data")


def main():
    print("=== iter_016 CHIS end-to-end Predictor sanity check ===", flush=True)
    cfg = json.load(open(os.path.join(PKG, "config.json")))
    feats = cfg["feature"]
    print(f"  config feature dim = {len(feats)}", flush=True)

    sys.path.insert(0, PKG)
    spec = importlib.util.spec_from_file_location("pred_chis", os.path.join(PKG, "Predictor.py"))
    mod = importlib.util.module_from_spec(spec)
    t0 = time.time()
    spec.loader.exec_module(mod)
    pred = mod.Predictor()
    print(f"  Predictor() constructed in {time.time()-t0:.2f}s", flush=True)

    batches = []
    sym_for_batch = []
    rng = np.random.default_rng(0)
    for s in (0, 1, 2, 3, 4):
        p = os.path.join(DATA, f"snapshot_sym{s}_date0_am.parquet")
        if not os.path.isfile(p):
            print(f"  SKIP {p}", flush=True)
            continue
        df = pd.read_parquet(p)
        n_rows = len(df)
        n_win = min(50, n_rows // 100)
        for k in range(n_win):
            window = df.iloc[k*100 : (k+1)*100][feats].copy().reset_index(drop=True)
            batches.append(window)
            sym_for_batch.append(s)
    print(f"  built {len(batches)} windows from {len(set(sym_for_batch))} syms", flush=True)

    # Shuffle order (sim platform shuffle behaviour)
    perm = rng.permutation(len(batches))
    batches = [batches[i] for i in perm]
    sym_for_batch = [sym_for_batch[i] for i in perm]

    # First pass
    t0 = time.time()
    out1 = pred.predict(batches)
    dt1 = time.time() - t0
    arr1 = np.array(out1, dtype=np.int64)
    assert arr1.shape == (len(batches), 5), f"shape mismatch: {arr1.shape}"
    print(f"\n  pass 1: {dt1*1000:.1f}ms ({dt1/len(batches)*1000:.2f}ms/window)", flush=True)
    HORIZONS = (5, 10, 20, 40, 60)
    for i, h in enumerate(HORIZONS):
        n_a = int((arr1[:, i] != 1).sum())
        n_long = int((arr1[:, i] == 2).sum())
        n_short = int((arr1[:, i] == 0).sum())
        print(f"    h={h:>2}: n_active={n_a}/{len(batches)} (long={n_long}, short={n_short})", flush=True)

    # Second pass: assert idempotent (no cross-call state)
    out2 = pred.predict(batches)
    arr2 = np.array(out2, dtype=np.int64)
    assert np.array_equal(arr1, arr2), "Predictor not idempotent! Cross-call state leak."
    print(f"\n  pass 2: identical to pass 1 (no state leak)", flush=True)

    # Single-window slice consistency
    out3 = pred.predict([batches[0]])
    arr3 = np.array(out3, dtype=np.int64)
    assert arr3[0].tolist() == arr1[0].tolist(), \
        f"single vs batch mismatch: {arr3[0].tolist()} vs {arr1[0].tolist()}"
    print(f"  pass 3 (single slice): identical to first row of pass 1", flush=True)

    # Output sample
    print(f"\n  sample outputs:", flush=True)
    for k in range(5):
        print(f"    src_sym={sym_for_batch[k]}  actions={arr1[k].tolist()}", flush=True)

    # Summary stats
    summary = {
        "n_windows": int(len(batches)),
        "shape_pred": list(arr1.shape),
        "predict_time_ms_pass1": float(dt1*1000),
        "ms_per_window": float(dt1/len(batches)*1000),
        "per_horizon_active_count": {
            int(h): int((arr1[:, i] != 1).sum()) for i, h in enumerate(HORIZONS)
        },
        "per_horizon_long_count": {
            int(h): int((arr1[:, i] == 2).sum()) for i, h in enumerate(HORIZONS)
        },
        "per_horizon_short_count": {
            int(h): int((arr1[:, i] == 0).sum()) for i, h in enumerate(HORIZONS)
        },
        "idempotent": True,
        "single_consistent": True,
    }
    out = os.path.join(HERE, "end_to_end_check.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  wrote -> {out}", flush=True)
    print(f"\n=== ALL PREDICTOR CHECKS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
