"""Build 12-dim multi-scale fixed-lag log-return features for schemeP train/test.

For each NPZ row (sym, date, sess_idx, t), look up snapshot[(sym,date,sess)] and
compute, for k in (1, 2, 5, 10, 20, 50):
    log_ret_mid_k  = log((mid1[t]+1) / (mid1[t-k]+1))
    log_ret_wmp_k  = log((wmp1[t]+1) / (wmp1[t-k]+1))
where wmp1 = (bid1*asize1 + ask1*bsize1) / (bsize1 + asize1).
Both are sym-agnostic (ratios are scale-invariant) and stateless.

Outputs:
  /root/projects/liangwenbei_workdir/experiments/R4_logret_lag/cache/
    lagret_train.npz  — keys: 'lag' (N_train, 12) float32, 'feat_names' list
    lagret_test.npz   — same for test split
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import os, sys, time

DATA_DIR = "/root/projects/liangwenbei_workdir/data"
CACHE_IN = "/tmp/lwb_pkg_extract/lwb_remote_pkg/cache"
OUT_DIR = "/root/projects/liangwenbei_workdir/experiments/R4_logret_lag/cache"
os.makedirs(OUT_DIR, exist_ok=True)

LAGS = (1, 2, 5, 10, 20, 50)
SESS_NAMES = {0: "am", 1: "pm"}
CLIP = 0.05


def feat_names():
    names = []
    for k in LAGS:
        names.append(f"logret_mid1_lag{k}")
    for k in LAGS:
        names.append(f"logret_wmp1_lag{k}")
    return names


def build_for_split(split):
    print(f"\n=== building {split} ===", flush=True)
    npz_path = os.path.join(CACHE_IN, f"schemeP_{split}.npz")
    d = np.load(npz_path)
    sym = d["sym"]      # int8 (0..4)
    date = d["date"]    # int16
    sess = d["sess_idx"]  # int8 (0=am, 1=pm)
    t = d["t"]          # int16, in [99, 1940]
    N = len(sym)
    print(f"  N={N:,}", flush=True)

    # Group by (sym, date, sess)
    keys = (sym.astype(np.int64) * 256 * 2 +
            date.astype(np.int64) * 2 +
            sess.astype(np.int64))
    order = np.argsort(keys, kind="stable")
    keys_s = keys[order]
    sym_s = sym[order]
    date_s = date[order]
    sess_s = sess[order]
    t_s = t[order]

    out = np.zeros((N, 2 * len(LAGS)), dtype=np.float32)

    # iter over unique (sym,date,sess) groups
    uniq, group_starts = np.unique(keys_s, return_index=True)
    group_ends = np.append(group_starts[1:], N)
    n_groups = len(uniq)
    print(f"  n_groups (sym,date,sess)={n_groups}", flush=True)

    t0 = time.time()
    for gi in range(n_groups):
        s, e = group_starts[gi], group_ends[gi]
        s_sym = int(sym_s[s])
        s_date = int(date_s[s])
        s_sess = int(sess_s[s])
        ts_g = t_s[s:e]  # (n_g,)

        snap_path = os.path.join(DATA_DIR,
                                 f"snapshot_sym{s_sym}_date{s_date}_{SESS_NAMES[s_sess]}.parquet")
        df = pd.read_parquet(snap_path,
                             columns=["midprice1", "bid1", "ask1", "bsize1", "asize1"])
        mid = df["midprice1"].to_numpy(np.float64)
        bid = df["bid1"].to_numpy(np.float64)
        ask = df["ask1"].to_numpy(np.float64)
        bsz = df["bsize1"].to_numpy(np.float64)
        asz = df["asize1"].to_numpy(np.float64)
        # NaN can occur when one side of the book is empty (snapshots have
        # missing ask1/asize1 at some t).  Fall back to midprice1 in that case.
        bad = (~np.isfinite(bid)) | (~np.isfinite(ask)) | \
              (~np.isfinite(bsz)) | (~np.isfinite(asz))
        bid = np.where(bad, mid, bid)
        ask = np.where(bad, mid, ask)
        bsz = np.where(bad, 1.0, bsz)
        asz = np.where(bad, 1.0, asz)
        denom = bsz + asz
        # wmp1 = (bid * asize + ask * bsize) / (bsize + asize)
        # if denom == 0 fall back to (bid+ask)/2
        safe_denom = np.where(denom > 0, denom, 1.0)
        wmp = np.where(denom > 0,
                       (bid * asz + ask * bsz) / safe_denom,
                       0.5 * (bid + ask))

        mid_t = mid[ts_g]
        wmp_t = wmp[ts_g]
        for ki, k in enumerate(LAGS):
            mid_lag = mid[ts_g - k]
            wmp_lag = wmp[ts_g - k]
            lr_m = np.log((mid_t + 1.0) / (mid_lag + 1.0))
            lr_w = np.log((wmp_t + 1.0) / (wmp_lag + 1.0))
            out[order[s:e], ki]               = np.clip(lr_m, -CLIP, CLIP).astype(np.float32)
            out[order[s:e], ki + len(LAGS)]   = np.clip(lr_w, -CLIP, CLIP).astype(np.float32)

        if (gi + 1) % 100 == 0 or gi == n_groups - 1:
            elapsed = time.time() - t0
            rate = (gi + 1) / elapsed
            eta_sec = (n_groups - gi - 1) / max(rate, 1e-6)
            print(f"  group {gi+1}/{n_groups}  elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta_sec:.0f}s",
                  flush=True)

    fn = feat_names()
    out_path = os.path.join(OUT_DIR, f"lagret_{split}.npz")
    np.savez_compressed(out_path,
                        lag=out, feat_names=np.array(fn, dtype=object))
    print(f"  saved {out_path}: {out.shape} dtype={out.dtype}", flush=True)
    print(f"  stats: mean={out.mean():.4e} std={out.std():.4e} "
          f"clip_hi_frac={(out >= CLIP).mean():.4e} "
          f"clip_lo_frac={(out <= -CLIP).mean():.4e}", flush=True)
    return out_path


def main():
    t0 = time.time()
    p1 = build_for_split("train")
    p2 = build_for_split("test")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Outputs:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
