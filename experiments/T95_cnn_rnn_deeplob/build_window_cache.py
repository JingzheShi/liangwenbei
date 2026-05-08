"""Build per-session raw LOB arrays from snapshot parquets, aligned with schemeP cache index.

Output: experiments/T95_cnn_rnn_deeplob/cache/lob_{split}.npz
  - lob_arr: (n_sessions, T_max=2001, F) float32 — per-session per-tick LOB features
  - sess_keys: (n_sessions, 3) int — (sym, date, sess) keys
  - sess_id_for_sample: (n_samples,) int — index into sess_keys for each schemeP row
  - t_for_sample: (n_samples,) int — t (tick index 99..1940) for each schemeP row
  - mp_t / mp_t60 / y60 / sym / date / sess_idx / t — passthrough from schemeP for convenience

Feature set (default = "top5_compact"):
  bid1..5, bsize1..5, ask1..5, asize1..5  → 20 cols
"""
from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
SCHEMEP_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
CACHE_DIR = os.path.join(HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_lob_cols(scheme="top5"):
    """Return list of LOB col names + the index assignment."""
    if scheme == "top5":
        cols = []
        for i in range(1, 6):
            cols.append(f"bid{i}")
            cols.append(f"bsize{i}")
            cols.append(f"ask{i}")
            cols.append(f"asize{i}")
        return cols  # 20 cols
    elif scheme == "top10":
        cols = []
        for i in range(1, 11):
            cols.append(f"bid{i}")
            cols.append(f"bsize{i}")
            cols.append(f"ask{i}")
            cols.append(f"asize{i}")
        return cols  # 40 cols
    else:
        raise ValueError(scheme)


def build_split(split: str, scheme: str = "top5"):
    """split: 'train' or 'test'"""
    print(f"\n=== build_split({split}, scheme={scheme}) ===", flush=True)
    p = os.path.join(SCHEMEP_CACHE, f"schemeP_{split}.npz")
    d = np.load(p)
    sym = d["sym"]
    date = d["date"]
    sess = d["sess_idx"]
    t = d["t"]
    n = len(sym)
    print(f"  n_samples={n}", flush=True)

    # Identify unique sessions
    sess_keys_arr = np.stack([sym, date, sess], axis=1)  # (n, 3)
    # use a structured tuple for unique
    sess_keys_pd = pd.DataFrame(sess_keys_arr, columns=["sym","date","sess"])
    uniq = sess_keys_pd.drop_duplicates().sort_values(["sym","date","sess"]).reset_index(drop=True)
    print(f"  n_sessions={len(uniq)}", flush=True)

    # Map session key to id
    key_to_id = {(int(r["sym"]), int(r["date"]), int(r["sess"])): i for i, r in uniq.iterrows()}
    sess_id_for_sample = np.array(
        [key_to_id[(int(s), int(dt), int(se))] for s, dt, se in zip(sym, date, sess)],
        dtype=np.int32,
    )
    sess_keys = uniq.values.astype(np.int32)  # (n_sessions, 3)

    # Load LOB cols
    lob_cols = get_lob_cols(scheme)
    F = len(lob_cols)
    T_max = 2001  # known constant
    print(f"  F={F} T_max={T_max}", flush=True)

    lob_arr = np.zeros((len(uniq), T_max, F), dtype=np.float32)
    sess_str = {0: "am", 1: "pm"}
    t0 = time.time()
    n_loaded = 0
    nan_total = 0
    for sess_id, (s, dt, se) in enumerate(sess_keys):
        s, dt, se = int(s), int(dt), int(se)
        fpath = os.path.join(DATA_DIR, f"snapshot_sym{s}_date{dt}_{sess_str[se]}.parquet")
        df = pd.read_parquet(fpath, columns=lob_cols)
        arr = df.values.astype(np.float32)  # (T, F)
        if arr.shape[0] != T_max:
            print(f"  WARN: {fpath} has {arr.shape[0]} ticks (expected {T_max}); padding/truncating", flush=True)
            if arr.shape[0] < T_max:
                # pad with last row
                pad = np.repeat(arr[-1:], T_max - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            else:
                arr = arr[:T_max]
        nan_total += int(np.isnan(arr).sum())
        lob_arr[sess_id] = arr
        n_loaded += 1
        if (sess_id + 1) % 100 == 0:
            print(f"  {sess_id+1}/{len(uniq)} sessions loaded ({time.time()-t0:.1f}s, nan={nan_total})", flush=True)

    print(f"  done: {n_loaded} sessions, total nan cells={nan_total}, elapsed={time.time()-t0:.1f}s", flush=True)

    # NaN imputation: forward-fill within session, then 0
    # Vectorized: where NaN, replace with previous non-NaN; for 1st row NaN, use 0
    if nan_total > 0:
        print(f"  forward-filling NaN in lob_arr...", flush=True)
        # per session, per-feature ffill
        # Simple approach: scan time axis
        # vectorized: use pandas fillna(method='ffill') over reshape
        # Reshape to (n_sess * T, F), but ffill needs to respect session boundaries
        # Easier: loop sessions
        for sess_id in range(lob_arr.shape[0]):
            ar = lob_arr[sess_id]  # (T, F)
            if np.isnan(ar).any():
                df = pd.DataFrame(ar)
                df = df.ffill().fillna(0.0)
                lob_arr[sess_id] = df.values.astype(np.float32)
        print(f"  ffill done", flush=True)

    # Save
    out = {
        "lob_arr": lob_arr,
        "sess_keys": sess_keys,
        "sess_id_for_sample": sess_id_for_sample,
        "t_for_sample": t.astype(np.int16),
        "sym": d["sym"],
        "date": d["date"],
        "sess_idx": d["sess_idx"],
        "t": d["t"],
        "y60": d["y60"],
        "mp_t": d["mp_t"],
        "mp_t60": d["mp_t60"],
    }
    out_path = os.path.join(CACHE_DIR, f"lob_{split}_{scheme}.npz")
    np.savez(out_path, **out)
    print(f"  saved {out_path}: lob_arr={lob_arr.shape} ({lob_arr.nbytes/1e6:.1f} MB)", flush=True)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="top5", choices=["top5", "top10"])
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    args = ap.parse_args()
    for sp in args.splits:
        build_split(sp, scheme=args.scheme)
