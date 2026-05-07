"""T74: extend T68 schemeP cache with 6 time-derived features (schemeQ).

Time features derived from cache fields `t` (tick index 0..2000, 3-second cadence)
and `sess_idx` (0=am, 1=pm):

  1. sec_norm        : t / 2000 in [0, 1] (session-relative position)
  2. sin_sess_pos    : sin(2*pi * t/2000)
  3. cos_sess_pos    : cos(2*pi * t/2000)
  4. is_open_30min   : 1.0 if t < 600 (first 30 min of session) else 0.0
  5. is_close_30min  : 1.0 if t > 1400 (last 30 min of session) else 0.0
  6. is_pm           : float(sess_idx == 1)

Output:
  cache/schemeQ_train.npz, cache/schemeQ_val.npz, cache/schemeQ_test.npz
  cache/schemeQ_feat_names.txt    (376 names; schemeP 370 + 6 time)
  cache/schemeQ_extra_feat_names.txt (222 = 216 schemeP extras + 6 time)

Constraint compliance:
  * sym is NOT used (time features are sym-invariant)
  * date is NOT used (the t/sess_idx values come purely from the 100-tick window position)
  * All features are stateless: only need t/sess_idx of the *current* tick
  * For inference, t/sess_idx must be derived from the `time` column of the input
    DataFrame (added to config.feature) — see Predictor.

This script reuses the existing T68 schemeP caches; no parquet re-processing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

T68_DIR = os.path.join(ROOT, "experiments", "T68_stage5_features")
SCHEMEP_CACHE = os.path.join(T68_DIR, "cache")
OUT_DIR = os.path.join(HERE, "cache")
os.makedirs(OUT_DIR, exist_ok=True)

TIME_FEAT_NAMES = [
    "sec_norm",
    "sin_sess_pos",
    "cos_sess_pos",
    "is_open_30min",
    "is_close_30min",
    "is_pm",
]

T_MAX = 2000.0  # session has T=2001 ticks (0..2000)


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def compute_time_features(t: np.ndarray, sess_idx: np.ndarray) -> np.ndarray:
    """Returns (N, 6) float32 array of time features."""
    t = t.astype(np.float32)
    s = sess_idx.astype(np.float32)
    sec_norm = t / T_MAX
    angle = 2.0 * np.pi * sec_norm
    sin_p = np.sin(angle).astype(np.float32)
    cos_p = np.cos(angle).astype(np.float32)
    is_open = (t < 600).astype(np.float32)
    is_close = (t > 1400).astype(np.float32)
    is_pm = (s > 0.5).astype(np.float32)
    return np.stack([sec_norm.astype(np.float32), sin_p, cos_p, is_open, is_close, is_pm], axis=1)


def extend_split(name: str) -> None:
    in_path = os.path.join(SCHEMEP_CACHE, f"schemeP_{name}.npz")
    out_path = os.path.join(OUT_DIR, f"schemeQ_{name}.npz")
    print(f"=== extending {in_path} -> {out_path} ===", flush=True)
    progress("loading", split=name)
    t0 = time.time()
    d = np.load(in_path)
    payload = {k: d[k] for k in d.files}
    n_rows = payload["X"].shape[0]
    print(f"  loaded {n_rows:,} rows, {payload['X'].shape[1]}-d, {time.time() - t0:.1f}s",
          flush=True)

    progress("computing_time", split=name)
    tfeat = compute_time_features(payload["t"], payload["sess_idx"])
    print(f"  time features shape={tfeat.shape}, ranges:", flush=True)
    for i, n in enumerate(TIME_FEAT_NAMES):
        v = tfeat[:, i]
        print(f"    {n:18s} min={v.min():+.4f} max={v.max():+.4f} mean={v.mean():+.4f}",
              flush=True)

    X_new = np.concatenate([payload["X"], tfeat], axis=1).astype(np.float32, copy=False)
    print(f"  X_new shape={X_new.shape}", flush=True)
    payload["X"] = X_new

    progress("saving", split=name)
    np.savez(out_path, **payload)
    print(f"  saved {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)", flush=True)


def main():
    progress("starting")
    # Read schemeP feature names
    schemep_full_path = os.path.join(SCHEMEP_CACHE, "schemeP_feat_names.txt")
    schemep_extra_path = os.path.join(SCHEMEP_CACHE, "schemeP_extra_feat_names.txt")
    with open(schemep_full_path) as f:
        full_names = [line.strip() for line in f if line.strip()]
    with open(schemep_extra_path) as f:
        extra_names = [line.strip() for line in f if line.strip()]
    print(f"schemeP: {len(full_names)} full / {len(extra_names)} extras", flush=True)

    # New names
    new_full_names = full_names + TIME_FEAT_NAMES
    new_extra_names = extra_names + TIME_FEAT_NAMES
    print(f"schemeQ: {len(new_full_names)} full / {len(new_extra_names)} extras", flush=True)
    assert len(new_full_names) == 376
    assert len(new_extra_names) == 222

    full_out = os.path.join(OUT_DIR, "schemeQ_feat_names.txt")
    extra_out = os.path.join(OUT_DIR, "schemeQ_extra_feat_names.txt")
    with open(full_out, "w") as f:
        f.write("\n".join(new_full_names) + "\n")
    with open(extra_out, "w") as f:
        f.write("\n".join(new_extra_names) + "\n")
    print(f"wrote {full_out} & {extra_out}", flush=True)

    for split in ("train", "val", "test"):
        extend_split(split)

    progress("done")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
