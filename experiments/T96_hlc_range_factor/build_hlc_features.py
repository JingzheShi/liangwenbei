"""T96: Build HLC-range factor features (adapted from feature.md).

Original idea (daily freq):
    adj_high = high + 3 * high_std_window
    adj_low  = low - 3 * low_std_window
    hlc_typ  = (adj_high + adj_low + 2*close) / 4
    hlc_range = (hlc_typ - adj_low) / (adj_high - adj_low)   # in [0,1]
    factor_raw = - hlc_range * daily_smooth_return
    factor_dema = DEMA(factor_raw)

Adaptation to HFT tick data:
- Each parquet = one (sym, date, session) of 2001 ticks
- Use the per-tick OHLC columns (already in raw data)
- Rolling window sizes W in {60, 300, 1500} (short / mid / session-scale)
- Smoothed return = EMA(close.diff(), span=3) → rolling sum over W
- Output features per (sym, date, sess, t) for t∈[99, 1940] to match schemeP cache

Outputs:
    cache/T96_train.npz  (X_extra: shape (1473600, n_extra))
    cache/T96_test.npz   (X_extra: shape (442080, n_extra))
    cache/T96_feat_names.txt
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")
T68_CACHE = os.path.join(ROOT, "experiments", "T68_stage5_features", "cache")

CACHE_DIR = os.path.join(HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

WINDOWS = (60, 300, 1500)
DEMA_SPAN = 20  # for smoothing factor_raw
EMA_RET_SPAN = 3  # for smoothing per-tick return
T_MIN = 99
T_MAX = 1940
EPS = 1e-9


def progress(step, **extra):
    p = os.path.join(HERE, "worker-progress.json")
    payload = {"status": "running", "step": step,
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **extra}
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def dema(x: np.ndarray, span: int) -> np.ndarray:
    """Double exponential moving average. Vectorized via pandas ewm."""
    s = pd.Series(x)
    e1 = s.ewm(span=span, adjust=False).mean().to_numpy()
    e2 = pd.Series(e1).ewm(span=span, adjust=False).mean().to_numpy()
    return 2.0 * e1 - e2


def compute_session_features(df: pd.DataFrame, windows=WINDOWS, dema_span=DEMA_SPAN,
                             ema_ret_span=EMA_RET_SPAN) -> pd.DataFrame:
    """Given one parquet (2001 rows for one (sym,date,sess)),
    return a DataFrame with new features for ALL rows.
    """
    n = len(df)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)

    # Per-tick return (simple diff to avoid log issues with negatives)
    ret = np.zeros(n, dtype=np.float64)
    ret[1:] = close[1:] - close[:-1]
    ret_smooth = pd.Series(ret).ewm(span=ema_ret_span, adjust=False).mean().to_numpy()

    feats = {}
    for W in windows:
        # Rolling std of high/low; min_periods=max(2, W//4) to allow warmup
        mp = max(2, W // 4)
        high_std = (pd.Series(high).rolling(window=W, min_periods=mp).std()
                    .fillna(0.0).to_numpy())
        low_std = (pd.Series(low).rolling(window=W, min_periods=mp).std()
                   .fillna(0.0).to_numpy())

        adj_high = high + 3.0 * high_std
        adj_low = low - 3.0 * low_std
        rng = adj_high - adj_low  # >= 0 since high >= low

        hlc_typ = 0.25 * (adj_high + adj_low + 2.0 * close)
        # Safe divide; when rng == 0 (constant prices), set to 0.5 (middle)
        hlc_range = np.where(rng > EPS, (hlc_typ - adj_low) / np.maximum(rng, EPS), 0.5)
        hlc_range = np.clip(hlc_range, -0.5, 1.5).astype(np.float64)  # mostly in [0,1]

        # daily_smooth_return = rolling sum of smoothed return
        daily_sret = (pd.Series(ret_smooth).rolling(window=W, min_periods=mp).sum()
                      .fillna(0.0).to_numpy())

        factor_raw = -hlc_range * daily_sret
        factor_dema = dema(factor_raw, span=dema_span)

        # Without risk-adjust (using raw high/low for comparison)
        rng_simple = high - low
        hlc_typ_simple = 0.25 * (high + low + 2.0 * close)
        hlc_range_simple = np.where(rng_simple > EPS,
                                    (hlc_typ_simple - low) / np.maximum(rng_simple, EPS),
                                    0.5)
        hlc_range_simple = np.clip(hlc_range_simple, -0.5, 1.5)
        factor_raw_simple = -hlc_range_simple * daily_sret

        feats[f"hlc_range_W{W}"] = hlc_range.astype(np.float32)
        feats[f"hlc_range_simple_W{W}"] = hlc_range_simple.astype(np.float32)
        feats[f"factor_raw_W{W}"] = factor_raw.astype(np.float32)
        feats[f"factor_dema_W{W}"] = factor_dema.astype(np.float32)
        feats[f"factor_raw_simple_W{W}"] = factor_raw_simple.astype(np.float32)
        feats[f"daily_sret_W{W}"] = daily_sret.astype(np.float32)
        feats[f"adj_range_W{W}"] = (adj_high - adj_low).astype(np.float32)
        feats[f"hlc_typ_W{W}"] = hlc_typ.astype(np.float32)

    return pd.DataFrame(feats)


def feature_names(windows=WINDOWS):
    names = []
    for W in windows:
        names += [
            f"hlc_range_W{W}",
            f"hlc_range_simple_W{W}",
            f"factor_raw_W{W}",
            f"factor_dema_W{W}",
            f"factor_raw_simple_W{W}",
            f"daily_sret_W{W}",
            f"adj_range_W{W}",
            f"hlc_typ_W{W}",
        ]
    return names


def build_split(split: str):
    """Build features for 'train' or 'test' to match schemeP cache row order."""
    print(f"=== build {split} ===", flush=True)
    progress(f"loading_{split}_keys")
    cache = np.load(os.path.join(T68_CACHE, f"schemeP_{split}.npz"))
    sym = cache["sym"].astype(np.int32)
    date = cache["date"].astype(np.int32)
    sess = cache["sess_idx"].astype(np.int32)
    t = cache["t"].astype(np.int32)
    n = len(sym)
    print(f"  schemeP {split}: n={n:,}", flush=True)

    feat_cols = feature_names()
    n_feat = len(feat_cols)
    out = np.zeros((n, n_feat), dtype=np.float32)

    # Group by (sym, date, sess) — process each parquet once
    keys = pd.DataFrame({"sym": sym, "date": date, "sess": sess, "t": t,
                         "row_idx": np.arange(n, dtype=np.int64)})
    grouped = keys.groupby(["sym", "date", "sess"], sort=True)
    n_groups = grouped.ngroups
    print(f"  n_sessions={n_groups}", flush=True)

    sess_map = {0: "am", 1: "pm"}
    t_load = 0.0
    t_compute = 0.0
    t_assign = 0.0

    n_done = 0
    t0 = time.time()
    for (s_, d_, ss_), grp in grouped:
        path = os.path.join(DATA_DIR, f"snapshot_sym{s_}_date{d_}_{sess_map[int(ss_)]}.parquet")
        ta = time.time()
        df_par = pd.read_parquet(path, columns=["high", "low", "close"])
        t_load += time.time() - ta
        ta = time.time()
        f_df = compute_session_features(df_par)
        t_compute += time.time() - ta
        ta = time.time()
        # Take rows in grp at corresponding t indices
        t_idx = grp["t"].to_numpy(np.int64)
        row_idx = grp["row_idx"].to_numpy(np.int64)
        # f_df rows are 0..2000, take t_idx
        out[row_idx] = f_df.iloc[t_idx][feat_cols].to_numpy(np.float32)
        t_assign += time.time() - ta
        n_done += 1
        if n_done % 50 == 0 or n_done == n_groups:
            elapsed = time.time() - t0
            eta = elapsed / n_done * (n_groups - n_done)
            print(f"    {n_done}/{n_groups}  elapsed={elapsed:.1f}s "
                  f"eta={eta:.1f}s  load={t_load:.1f} comp={t_compute:.1f} assign={t_assign:.1f}",
                  flush=True)

    print(f"  out shape: {out.shape}  range [{out.min():+.4e}, {out.max():+.4e}]", flush=True)
    np.savez_compressed(
        os.path.join(CACHE_DIR, f"T96_{split}.npz"),
        X_extra=out,
        sym=sym.astype(np.int8),
        date=date.astype(np.int16),
        sess=sess.astype(np.int8),
        t=t.astype(np.int16),
    )
    return out, feat_cols


def main():
    t0 = time.time()
    progress("starting")
    out_train, names = build_split("train")
    out_test, _ = build_split("test")

    with open(os.path.join(CACHE_DIR, "T96_feat_names.txt"), "w") as f:
        for nm in names:
            f.write(nm + "\n")

    print(f"\n=== quick stats per feature on train ===", flush=True)
    print(f"  {'feature':<30s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s}",
          flush=True)
    for i, nm in enumerate(names):
        col = out_train[:, i]
        print(f"  {nm:<30s} {col.mean():>+12.4e} {col.std():>12.4e} "
              f"{col.min():>+12.4e} {col.max():>+12.4e}", flush=True)

    print(f"\nDONE in {time.time()-t0:.1f}s.", flush=True)
    progress("done", n_features=len(names))


if __name__ == "__main__":
    main()
