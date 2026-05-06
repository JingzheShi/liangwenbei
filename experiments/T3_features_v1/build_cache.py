"""Build feature_v1 cache for all 1200 (sym, date, session) parquets.

Reads each `data/snapshot_sym{X}_date{Y}_{am,pm}.parquet`, computes the 72 new
features via `compute.compute_all`, concatenates with the original 163 cols,
and writes to `data/features_v1/snapshot_sym{X}_date{Y}_{am,pm}.parquet`.

Idempotent: skips files that already exist with full 235-col schema. Run with
`--force` to overwrite.

Usage:
    python experiments/T3_features_v1/build_cache.py [--workers N] [--force] [--limit K]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import Tuple

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO_ROOT)

from compute import compute_all, feature_v1_columns  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
OUT_DIR = os.path.join(REPO_ROOT, "data", "features_v1")

N_SYMS = 5
N_DATES = 120
SESSIONS = ("am", "pm")
EXPECTED_BASE_COLS = 163
EXPECTED_NEW_COLS = 72
EXPECTED_TOTAL_COLS = EXPECTED_BASE_COLS + EXPECTED_NEW_COLS
EXPECTED_ROWS = 2001


def _path(sym: int, date: int, session: str, dir_: str) -> str:
    return os.path.join(dir_, f"snapshot_sym{sym}_date{date}_{session}.parquet")


def _process_one(args: Tuple[int, int, str, bool]) -> Tuple[str, str, float]:
    """Process a single (sym, date, session). Returns (path, status, elapsed_s)."""
    sym, date, session, force = args
    in_path = _path(sym, date, session, DATA_DIR)
    out_path = _path(sym, date, session, OUT_DIR)

    if os.path.exists(out_path) and not force:
        # Quick metadata check: must have right col count
        try:
            import pyarrow.parquet as pq
            schema = pq.ParquetFile(out_path).schema_arrow
            if len(schema.names) == EXPECTED_TOTAL_COLS:
                return (out_path, "skip", 0.0)
        except Exception:
            pass  # treat as needing rebuild

    t0 = time.time()
    try:
        df = pd.read_parquet(in_path)
        new = compute_all(df, session=session)
        merged = pd.concat([df.reset_index(drop=True), new.reset_index(drop=True)], axis=1)
        # Sanity: 235 cols, 2001 rows
        assert merged.shape[1] == EXPECTED_TOTAL_COLS, (
            f"{in_path}: expected {EXPECTED_TOTAL_COLS} cols, got {merged.shape[1]}"
        )
        merged.to_parquet(out_path, compression="snappy", index=False)
        return (out_path, "ok", time.time() - t0)
    except Exception as e:
        return (out_path, f"FAIL: {type(e).__name__}: {e}", time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N files (smoke test)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = []
    for sym in range(N_SYMS):
        for date in range(N_DATES):
            for sess in SESSIONS:
                jobs.append((sym, date, sess, args.force))
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"[T3-build_cache] {len(jobs)} files; workers={args.workers}; force={args.force}")
    print(f"[T3-build_cache] out_dir={OUT_DIR}")
    t0 = time.time()

    n_ok = n_skip = n_fail = 0
    fail_msgs = []

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, (path, status, elapsed) in enumerate(pool.imap_unordered(_process_one, jobs, chunksize=4)):
                if status == "ok":
                    n_ok += 1
                elif status == "skip":
                    n_skip += 1
                else:
                    n_fail += 1
                    fail_msgs.append(f"{path}: {status}")
                if (i + 1) % 100 == 0 or (i + 1) == len(jobs):
                    print(f"  [{i+1}/{len(jobs)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                          f"elapsed={time.time()-t0:.1f}s")
    else:
        for i, job in enumerate(jobs):
            path, status, elapsed = _process_one(job)
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_fail += 1
                fail_msgs.append(f"{path}: {status}")
            if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                print(f"  [{i+1}/{len(jobs)}] ok={n_ok} skip={n_skip} fail={n_fail} "
                      f"elapsed={time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    # Disk size
    total_size = 0
    for f in os.listdir(OUT_DIR):
        if f.endswith(".parquet"):
            total_size += os.path.getsize(os.path.join(OUT_DIR, f))
    print(f"\n[T3-build_cache] done in {elapsed:.1f}s")
    print(f"  ok={n_ok}, skip={n_skip}, fail={n_fail}")
    print(f"  out_dir size: {total_size / (1024**2):.1f} MB ({total_size} bytes)")
    if fail_msgs:
        print(f"\nFailures ({len(fail_msgs)}):")
        for m in fail_msgs[:20]:
            print(f"  {m}")
        if len(fail_msgs) > 20:
            print(f"  ... and {len(fail_msgs)-20} more")
        sys.exit(1)


if __name__ == "__main__":
    main()
