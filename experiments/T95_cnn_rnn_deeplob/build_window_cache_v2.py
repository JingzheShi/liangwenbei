"""Build per-session LOB cache with engineered per-tick features.

Adds to top5 (20 cols) the following derived per-tick features:
  - mid = (bid1 + ask1) / 2
  - mid_ret = mid_t - mid_{t-1} (per-tick return)
  - spread = ask1 - bid1
  - imb1, imb2, imb3, imb4, imb5  (size imbalance per level)
  - tot_bsize = log1p(sum bsize)
  - tot_asize = log1p(sum asize)
  - bsize_ratio = bsize1 / (bsize1 + asize1 + 1e-9)
  - tick_idx = t / 2001 (positional info, BUT this is per-session positional, not date)
                Actually skip — could leak date-of-day info. Use OHLC-derived stuff instead.

Output: cache/lob_{split}_v2.npz with same structure as v1, plus extra cols.
F dim = 20 (raw) + 11 (engineered) = 31 cols.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
SCHEMEP_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")
CACHE_DIR = os.path.join(HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def get_raw_cols():
    cols = []
    for i in range(1, 6):
        cols += [f"bid{i}", f"bsize{i}", f"ask{i}", f"asize{i}"]
    return cols


def build_per_tick_features(df: pd.DataFrame) -> np.ndarray:
    """df has the raw LOB cols (top5). Returns (T, 31) np.float32 array.
    Cols order:
      0..19 = raw (bid1, bsize1, ask1, asize1, bid2, bsize2, ..., bsize5, ask5, asize5)
      20    = mid
      21    = mid_ret
      22    = spread
      23..27 = imb1..5
      28    = log1p(sum bsize)
      29    = log1p(sum asize)
      30    = bsize1 / (bsize1 + asize1)
    """
    raw_cols = get_raw_cols()
    raw = df[raw_cols].values.astype(np.float32)  # (T, 20)
    bid1 = raw[:, 0]; bsize1 = raw[:, 1]; ask1 = raw[:, 2]; asize1 = raw[:, 3]
    bid2 = raw[:, 4]; bsize2 = raw[:, 5]; ask2 = raw[:, 6]; asize2 = raw[:, 7]
    bid3 = raw[:, 8]; bsize3 = raw[:, 9]; ask3 = raw[:, 10]; asize3 = raw[:, 11]
    bid4 = raw[:, 12]; bsize4 = raw[:, 13]; ask4 = raw[:, 14]; asize4 = raw[:, 15]
    bid5 = raw[:, 16]; bsize5 = raw[:, 17]; ask5 = raw[:, 18]; asize5 = raw[:, 19]

    mid = (bid1 + ask1) / 2.0
    mid_ret = np.empty_like(mid); mid_ret[0] = 0.0; mid_ret[1:] = mid[1:] - mid[:-1]
    spread = ask1 - bid1

    eps = 1e-9
    imb1 = (bsize1 - asize1) / (bsize1 + asize1 + eps)
    imb2 = (bsize2 - asize2) / (bsize2 + asize2 + eps)
    imb3 = (bsize3 - asize3) / (bsize3 + asize3 + eps)
    imb4 = (bsize4 - asize4) / (bsize4 + asize4 + eps)
    imb5 = (bsize5 - asize5) / (bsize5 + asize5 + eps)

    tot_b = bsize1 + bsize2 + bsize3 + bsize4 + bsize5
    tot_a = asize1 + asize2 + asize3 + asize4 + asize5
    log_tot_b = np.log1p(np.maximum(tot_b, 0.0))
    log_tot_a = np.log1p(np.maximum(tot_a, 0.0))
    bsize_ratio = bsize1 / (bsize1 + asize1 + eps)

    extra = np.stack([mid, mid_ret, spread, imb1, imb2, imb3, imb4, imb5,
                       log_tot_b, log_tot_a, bsize_ratio], axis=1).astype(np.float32)
    out = np.concatenate([raw, extra], axis=1)  # (T, 31)
    return out


def build_split(split: str):
    print(f"\n=== build_split_v2({split}) ===", flush=True)
    p = os.path.join(SCHEMEP_CACHE, f"schemeP_{split}.npz")
    d = np.load(p)
    sym = d["sym"]; date = d["date"]; sess = d["sess_idx"]; t = d["t"]
    n = len(sym)
    print(f"  n_samples={n}", flush=True)

    sess_keys_arr = np.stack([sym, date, sess], axis=1)
    uniq = pd.DataFrame(sess_keys_arr, columns=["sym","date","sess"]).drop_duplicates() \
                .sort_values(["sym","date","sess"]).reset_index(drop=True)
    n_sess = len(uniq)
    print(f"  n_sessions={n_sess}", flush=True)

    key_to_id = {(int(r["sym"]), int(r["date"]), int(r["sess"])): i
                 for i, r in uniq.iterrows()}
    sess_id_for_sample = np.array(
        [key_to_id[(int(s), int(dt), int(se))] for s, dt, se in zip(sym, date, sess)],
        dtype=np.int32,
    )
    sess_keys = uniq.values.astype(np.int32)

    raw_cols = get_raw_cols()
    T_max = 2001
    F = 31
    lob_arr = np.zeros((n_sess, T_max, F), dtype=np.float32)
    sess_str = {0: "am", 1: "pm"}
    t0 = time.time()
    nan_total = 0
    for sess_id, (s, dt, se) in enumerate(sess_keys):
        s, dt, se = int(s), int(dt), int(se)
        fpath = os.path.join(DATA_DIR, f"snapshot_sym{s}_date{dt}_{sess_str[se]}.parquet")
        df = pd.read_parquet(fpath, columns=raw_cols)
        if len(df) != T_max:
            if len(df) < T_max:
                pad = pd.concat([df, pd.DataFrame(np.repeat(df.iloc[-1:].values,
                                                              T_max - len(df), axis=0),
                                                    columns=raw_cols)], ignore_index=True)
                df = pad
            else:
                df = df.iloc[:T_max].reset_index(drop=True)
        # ffill NaN
        if df[raw_cols].isna().any().any():
            df[raw_cols] = df[raw_cols].ffill().fillna(0.0)
        feats = build_per_tick_features(df)
        nan_total += int(np.isnan(feats).sum())
        if np.isnan(feats).any():
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        lob_arr[sess_id] = feats
        if (sess_id + 1) % 200 == 0:
            print(f"  {sess_id+1}/{n_sess} sessions ({time.time()-t0:.1f}s, nan={nan_total})", flush=True)
    print(f"  done {n_sess} sessions, {time.time()-t0:.1f}s, total nan={nan_total}", flush=True)

    out = {
        "lob_arr": lob_arr,
        "sess_keys": sess_keys,
        "sess_id_for_sample": sess_id_for_sample,
        "t_for_sample": t.astype(np.int16),
        "sym": d["sym"], "date": d["date"], "sess_idx": d["sess_idx"], "t": d["t"],
        "y60": d["y60"], "mp_t": d["mp_t"], "mp_t60": d["mp_t60"],
    }
    out_path = os.path.join(CACHE_DIR, f"lob_{split}_v2.npz")
    np.savez(out_path, **out)
    print(f"  saved {out_path} ({lob_arr.nbytes/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "train"])
    args = ap.parse_args()
    for sp in args.splits:
        build_split(sp)
