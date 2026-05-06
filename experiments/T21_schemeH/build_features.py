"""Build Scheme H last-tick features.

Scheme H = Scheme C 226d - 3 (drop time encoding) + R20 ~68 = 291d total.
Layout (in order):
  [0:154]   raw 154 cols
  [154:223] T3 69 cols (drop trailing 3 time-encoding cols)
  [223:291] R20 68 cols

We also keep the trailing 3 time-encoding cols at the end for completeness, so
that the np.savez X has shape (N, 294) and feat_names.txt has 294 lines. The
loso_train.py for T21 simply slices `X[:, :-N_DROP_TAIL]` like T11.

Per-session valid t range = [WINDOW-1, T-1-MAX_HORIZON].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "experiments", "T3_features_v1"))

from src.data.split import get_split  # noqa: E402
from src.data.dataset import get_default_feature_cols  # noqa: E402
from compute import feature_v1_columns  # noqa: E402
from r20_features import compute_r20, r20_feature_columns  # noqa: E402

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}


def get_scheme_h_columns():
    """Return:
        - all_cols: [raw 154, T3 69 (no time enc), R20 68, T3 time-enc 3]
        - 154 raw cols
        - 69 T3 cols (excluding time encoding)
        - R20 cols
    """
    raw = get_default_feature_cols()  # 154
    t3 = feature_v1_columns()  # 72; trailing 3 are time encoding
    t3_no_time = [c for c in t3 if not c.startswith("time_")]
    t3_time = [c for c in t3 if c.startswith("time_")]
    r20 = r20_feature_columns()  # 68

    all_cols = raw + t3_no_time + r20 + t3_time
    return all_cols, raw, t3_no_time, r20, t3_time


def build_split(sym_dates, all_cols, raw_cols, t3_no_time_cols, r20_cols, t3_time_cols, cache_dir):
    Xs = []
    ys = {h: [] for h in HORIZONS}
    mps = {f"mp_t{h}": [] for h in HORIZONS}
    mp_t = []
    syms, dates, sess_idxs, ts = [], [], [], []

    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    n_total_cols = len(all_cols)
    amount_idx = raw_cols.index("amount_delta") if "amount_delta" in raw_cols else -1

    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(cache_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path)
        T = len(df)
        valid_lo = WINDOW - 1
        valid_hi = T - 1 - MAX_HORIZON
        if valid_hi < valid_lo:
            continue

        # Compute R20 fresh per session
        r20_df = compute_r20(df)

        # Build feature matrix in canonical order
        # The T3 cols (mlofi, wmp, rv, ewma, time_*) are already in df from data/features_v1.
        # Slice rows after computing.
        sl = slice(valid_lo, valid_hi + 1)
        n = valid_hi - valid_lo + 1

        X_sl = np.empty((n, n_total_cols), dtype=np.float32)

        # raw 154
        X_sl[:, :len(raw_cols)] = df[raw_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        # T3 no time (69)
        c_off = len(raw_cols)
        X_sl[:, c_off:c_off + len(t3_no_time_cols)] = (
            df[t3_no_time_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        )
        c_off += len(t3_no_time_cols)
        # R20 (68)
        X_sl[:, c_off:c_off + len(r20_cols)] = (
            r20_df[r20_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        )
        c_off += len(r20_cols)
        # T3 time-encoding (3)
        X_sl[:, c_off:c_off + len(t3_time_cols)] = (
            df[t3_time_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        )

        # Apply log1p to amount_delta (preserve sign) — same as T5b
        if amount_idx >= 0:
            col = X_sl[:, amount_idx]
            X_sl[:, amount_idx] = np.sign(col) * np.log1p(np.abs(col))

        # Replace any inf with 0 and clamp NaNs (defensive — should be rare)
        bad = ~np.isfinite(X_sl)
        if bad.any():
            X_sl = np.where(bad, 0.0, X_sl).astype(np.float32)

        Xs.append(X_sl)

        midprice = df["midprice"].to_numpy(dtype=np.float32, copy=False)
        mp_t.append(midprice[sl])
        for h in HORIZONS:
            y = df[f"label_{h}"].iloc[sl].to_numpy(dtype=np.int8, copy=False)
            ys[h].append(y)
            mp_h = midprice[valid_lo + h: valid_hi + 1 + h]
            mps[f"mp_t{h}"].append(mp_h)

        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(np.arange(valid_lo, valid_hi + 1, dtype=np.int16))

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            print(f"  [{i + 1}/{n_sessions}] sessions; elapsed {elapsed:.1f}s "
                  f"({(i + 1) / elapsed:.1f}/s)", flush=True)

    out = {
        "X": np.concatenate(Xs, axis=0),
        "mp_t": np.concatenate(mp_t, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    for h in HORIZONS:
        out[f"y{h}"] = np.concatenate(ys[h], axis=0)
        out[f"mp_t{h}"] = np.concatenate(mps[f"mp_t{h}"], axis=0)
    print(f"  -> X shape {out['X'].shape} mem {out['X'].nbytes / 1e9:.2f}GB", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--cache-dir", default="data/features_v1")
    ap.add_argument("--out-dir", default="experiments/T21_schemeH/cache")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_cols, raw, t3_no_time, r20, t3_time = get_scheme_h_columns()
    print(f"Scheme H total: {len(all_cols)} cols (154 raw + {len(t3_no_time)} T3 + "
          f"{len(r20)} R20 + {len(t3_time)} time-enc)", flush=True)

    # Sanity check on first session
    df0 = pd.read_parquet(os.path.join(args.cache_dir, "snapshot_sym0_date0_am.parquet"))
    r20_0 = compute_r20(df0)
    nan99 = r20_0.iloc[WINDOW - 1:].isna().any().any()
    inf99 = np.isinf(r20_0.iloc[WINDOW - 1:].to_numpy()).any()
    print(f"  sanity: row 99+ NaN={nan99} Inf={inf99}", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True)
            continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(
            split[key], all_cols, raw, t3_no_time, r20, t3_time, args.cache_dir,
        )
        out_path = os.path.join(args.out_dir, f"schemeH_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    names_path = os.path.join(args.out_dir, "schemeH_feat_names.txt")
    with open(names_path, "w") as f:
        for n in all_cols:
            f.write(n + "\n")
    print(f"feature names -> {names_path}  ({len(all_cols)} cols)", flush=True)

    names_no_time_path = os.path.join(args.out_dir, "schemeH_no_time_feat_names.txt")
    with open(names_no_time_path, "w") as f:
        for n in all_cols[:-3]:
            f.write(n + "\n")
    print(f"feature names (no time) -> {names_no_time_path}  ({len(all_cols) - 3} cols)", flush=True)


if __name__ == "__main__":
    main()
