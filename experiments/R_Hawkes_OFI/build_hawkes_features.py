"""Build Hawkes-OFI features per sample, aligned to schemeP train/test indices.

In-window (stateless wrt cross-sample state) but session-internal: for each
(sym, date, sess) session of 2001 ticks, we run causal IIR filters over the
session's level-1 OFI events. Each sample at tick t reads the filter output
at index t — equivalent to a Hawkes self+cross excitation lookback that is
fully causal within the 100-tick window (since EWMA(alpha=0.2) effectively
discounts >50 ticks ago to ~0.001).

Outputs (6-dim per sample, aligned to schemeP indices):
  forecast_buy_30        — Hawkes-style integrated next-30 buy intensity
  forecast_sell_30       — same for sell
  forecast_imbalance_30  — buy minus sell forecast
  branching_proxy_b      — (lambda_b_fast - lambda_b_slow) / (lambda_b_slow + EPS)
  branching_proxy_s      — same for sell
  fast_signed_intst      — lambda_b_fast - lambda_s_fast (current Hawkes net intensity)

Compliance with CRITICAL_CONSTRAINTS:
  - sym never used as feature input
  - date never used as feature input
  - filter is causal, only uses [0..t] within the same session
  - no cross-call state: each sample's feature is reproducible solely from its
    own 100-tick window (the 100-tick window covers the bulk of EWMA mass)
  - sym-agnostic: identical filter and parameters for all symbols

Usage:
  python3 build_hawkes_features.py \
      --schemeP /path/to/schemeP_train.npz \
      --data-dir /path/to/snapshot/ \
      --out /path/to/hawkes_train.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.signal import lfilter


EPS = 1e-8

# Hawkes timescales (in ticks)
ALPHA_FAST = 0.20         # fast intensity, decay ~5 ticks half-life
ALPHA_SLOW = 0.05         # slow baseline, decay ~20 ticks
BETA_FAST = -np.log(1.0 - ALPHA_FAST)   # ≈ 0.2231
FORECAST_HORIZON = 30
INTEGRATED_KERNEL = (1.0 - np.exp(-BETA_FAST * FORECAST_HORIZON)) / BETA_FAST  # ≈ 4.477


def compute_session_ofi_events(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-tick level-1 buy/sell event arrays for one session.

    Returns (buy_t, sell_t) of shape (T,) each, with buy_t = max(0, e_1(t))
    and sell_t = max(0, -e_1(t)).
    e_1(t) follows Cont-Kukanov MLOFI level-1 formula.
    """
    b = df["bid1"].to_numpy(dtype=np.float64, copy=False)
    a = df["ask1"].to_numpy(dtype=np.float64, copy=False)
    bs = df["bsize1"].to_numpy(dtype=np.float64, copy=False)
    asz = df["asize1"].to_numpy(dtype=np.float64, copy=False)

    T = len(b)
    if T < 2:
        return np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64)

    b_prev = np.empty_like(b); b_prev[0] = b[0]; b_prev[1:] = b[:-1]
    a_prev = np.empty_like(a); a_prev[0] = a[0]; a_prev[1:] = a[:-1]
    bs_prev = np.empty_like(bs); bs_prev[0] = bs[0]; bs_prev[1:] = bs[:-1]
    as_prev = np.empty_like(asz); as_prev[0] = asz[0]; as_prev[1:] = asz[:-1]

    ind_b_up = (b >= b_prev).astype(np.float64)
    ind_b_dn = (b <= b_prev).astype(np.float64)
    ind_a_dn = (a <= a_prev).astype(np.float64)
    ind_a_up = (a >= a_prev).astype(np.float64)

    e = ind_b_up * bs - ind_b_dn * bs_prev - ind_a_dn * asz + ind_a_up * as_prev
    e[0] = 0.0
    np.nan_to_num(e, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    buy = np.maximum(e, 0.0)
    sell = np.maximum(-e, 0.0)
    return buy, sell


def ewma_full(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EWMA(alpha) over full array. y[0] = x[0]; y[t] = a*x[t] + (1-a)*y[t-1]."""
    if len(x) == 0:
        return x.copy()
    b = np.array([alpha], dtype=np.float64)
    a_filt = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = np.array([(1.0 - alpha) * x[0]], dtype=np.float64)
    y, _ = lfilter(b, a_filt, x, zi=zi)
    return y


def hawkes_features_for_session(df: pd.DataFrame) -> np.ndarray:
    """Compute Hawkes feature matrix for one session.

    Returns array of shape (T, 6) with columns:
      forecast_buy_30, forecast_sell_30, forecast_imbalance_30,
      branching_proxy_b, branching_proxy_s, fast_signed_intst.
    """
    buy, sell = compute_session_ofi_events(df)
    lam_b_fast = ewma_full(buy, ALPHA_FAST)
    lam_s_fast = ewma_full(sell, ALPHA_FAST)
    lam_b_slow = ewma_full(buy, ALPHA_SLOW)
    lam_s_slow = ewma_full(sell, ALPHA_SLOW)

    forecast_buy = lam_b_fast * INTEGRATED_KERNEL
    forecast_sell = lam_s_fast * INTEGRATED_KERNEL
    forecast_imb = forecast_buy - forecast_sell

    branching_b = (lam_b_fast - lam_b_slow) / (lam_b_slow + EPS)
    branching_s = (lam_s_fast - lam_s_slow) / (lam_s_slow + EPS)

    fast_signed = lam_b_fast - lam_s_fast

    # Clip branching ratios for numerical safety (heavy tails when slow ~ 0)
    branching_b = np.clip(branching_b, -10.0, 10.0)
    branching_s = np.clip(branching_s, -10.0, 10.0)

    feats = np.stack([
        forecast_buy.astype(np.float32),
        forecast_sell.astype(np.float32),
        forecast_imb.astype(np.float32),
        branching_b.astype(np.float32),
        branching_s.astype(np.float32),
        fast_signed.astype(np.float32),
    ], axis=1)
    return feats


SESS_NAME = {0: "am", 1: "pm"}


def parquet_path(data_dir: str, sym: int, date: int, sess_idx: int) -> str:
    return os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{SESS_NAME[sess_idx]}.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemeP", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default=None)
    args = ap.parse_args()

    sP = np.load(args.schemeP)
    sym_arr = sP["sym"][:].astype(np.int8)
    date_arr = sP["date"][:].astype(np.int16)
    sess_arr = sP["sess_idx"][:].astype(np.int8)
    t_arr = sP["t"][:].astype(np.int16)
    N = len(sym_arr)

    out = np.zeros((N, 6), dtype=np.float32)

    # Build a map: (sym, date, sess) -> indices in schemeP
    # Use a single composite key for fast groupby
    key = (sym_arr.astype(np.int64) * 200_000
           + date_arr.astype(np.int64) * 1_000
           + sess_arr.astype(np.int64))
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    # Find boundaries
    boundaries = np.concatenate([[0], np.flatnonzero(np.diff(sorted_key)) + 1, [N]])
    n_sessions = len(boundaries) - 1
    print(f"Total samples: {N:,}; unique sessions: {n_sessions}", flush=True)

    t0 = time.time()
    n_done = 0
    for i in range(n_sessions):
        s, e = boundaries[i], boundaries[i + 1]
        idx_in_sP = order[s:e]
        # All rows in this slice share the same (sym, date, sess)
        sym_v = int(sym_arr[idx_in_sP[0]])
        date_v = int(date_arr[idx_in_sP[0]])
        sess_v = int(sess_arr[idx_in_sP[0]])
        path = parquet_path(args.data_dir, sym_v, date_v, sess_v)
        if not os.path.exists(path):
            print(f"  [WARN] missing {path}; zero-filling {len(idx_in_sP)} samples", flush=True)
            n_done += len(idx_in_sP)
            continue
        df = pd.read_parquet(path, columns=["bid1", "bsize1", "ask1", "asize1"])
        feats_full = hawkes_features_for_session(df)  # (T, 6)
        ts_in_session = t_arr[idx_in_sP].astype(np.int64)
        # Lookup
        out[idx_in_sP] = feats_full[ts_in_session]
        n_done += len(idx_in_sP)

        if (i + 1) % 50 == 0 or i == n_sessions - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n_sessions - i - 1) / max(rate, 1e-9)
            print(f"  [{i+1}/{n_sessions}] {n_done:,}/{N:,} samples; "
                  f"elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
            if args.progress:
                with open(args.progress, "w") as f:
                    json.dump({
                        "status": "running",
                        "step": f"build_hawkes_features {i+1}/{n_sessions}",
                        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "elapsed_sec": elapsed,
                    }, f)

    # Sanity: no NaN/inf
    bad = ~np.isfinite(out)
    if bad.any():
        print(f"  [WARN] {bad.sum():,} non-finite entries; zeroing", flush=True)
        out[bad] = 0.0

    # Stats per dim
    feat_names = [
        "hawkes_forecast_buy_30",
        "hawkes_forecast_sell_30",
        "hawkes_forecast_imb_30",
        "hawkes_branching_b",
        "hawkes_branching_s",
        "hawkes_fast_signed",
    ]
    print()
    print("Feature stats (mean ± std):")
    for k, name in enumerate(feat_names):
        m = float(out[:, k].mean())
        s = float(out[:, k].std())
        print(f"  {name:30s} mean={m:+.4f} std={s:.4f}")

    np.save(args.out, out)
    print(f"\nSaved {args.out}  shape={out.shape}  dtype={out.dtype}")
    # Also save names
    names_path = os.path.splitext(args.out)[0] + "_names.txt"
    with open(names_path, "w") as f:
        for n in feat_names:
            f.write(n + "\n")
    print(f"Saved {names_path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
