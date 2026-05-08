"""Timing test for iter_015 Predictor (T87 SPO+ DFL)."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")

sys.path.insert(0, os.path.join(HERE, "iter015_pkg"))
import importlib
for k in list(sys.modules.keys()):
    if "Predictor" in k or "fast_features" in k:
        del sys.modules[k]
import Predictor as P_mod
importlib.reload(P_mod)


def build_batches(n: int = 1024):
    fn = os.path.join(DATA_DIR, "snapshot_sym0_date96_am.parquet")
    df = pd.read_parquet(fn)
    rng = np.random.default_rng(42)
    t_max = len(df) - 60 - 1
    ts = rng.integers(99, t_max, size=n)
    return [df.iloc[t-99:t+1].reset_index(drop=True) for t in ts]


def main():
    print("=== iter_015 timing test ===", flush=True)
    pred = P_mod.Predictor()
    print(f"  LGB: {dict((h, len(v)) for h, v in pred._lgb_lists.items())}", flush=True)
    print(f"  NN:  {dict((h, len(v)) for h, v in pred._nn_lists.items())}", flush=True)

    warm = build_batches(64)
    pred.predict(warm)

    batches = build_batches(1024)
    print(f"  built 1024 batches; running 3 timing rounds...", flush=True)
    times = []
    for r in range(3):
        t0 = time.time()
        out = pred.predict(batches)
        dt = time.time() - t0
        times.append(dt)
        print(f"    round {r+1}: {dt:.3f}s ({len(out)} predictions)", flush=True)

    mean_t = float(np.mean(times))
    print(f"\n  mean: {mean_t:.3f}s for 1024-batch inference", flush=True)
    print(f"  iter_014 reference: 0.66s (5 LGB + 5 NN)", flush=True)
    print(f"  estimated full-test (442 080 rows): {mean_t * 442080 / 1024:.0f}s = {mean_t * 442080 / 1024 / 60:.1f} min",
          flush=True)
    print(f"  platform budget: 10 800s (3h) per submission", flush=True)


if __name__ == "__main__":
    main()
