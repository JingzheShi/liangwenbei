"""Build Scheme I caches: SchemeC 226d + Alpha101 + Alpha191.

Cache layout (one big cache; we slice columns later for the 3 sub-schemes):
  [0:154]            raw 154 cols
  [154:226]          T3 69 cols (no time encoding) + 3 time encoding (last 3)
                     ⇒ identical to schemeC layout: 154 + 69 + 3 = 226
  [226:259]          Alpha101 (33 cols)
  [259:383]          Alpha191 (124 cols)

Total = 226 + 33 + 124 = 383 cols.

Three schemes are derived from this single cache:
  Scheme I_a (alpha-only):    cols 226..383 → 157 cols
  Scheme I_b (alpha+raw):     all 383 cols, BUT we drop the trailing 3 time enc
                              so 380 cols
  Scheme I_c (curated):       chosen later from feature importance.

We store the time-encoding cols as last-3 of the schemeC block (positions
223:226 in the canonical 226d order) so we can drop them if needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
from alpha101 import compute_alpha101, alpha101_columns  # noqa: E402
from alpha191 import compute_alpha191, alpha191_columns  # noqa: E402

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}


def get_scheme_i_columns():
    """Return:
        all_cols: raw 154 + T3 69 (no time) + alpha101 + alpha191 + T3 time(3)
        ⇒ alpha and raw blocks are contiguous; trailing 3 are time encoding.
    """
    raw = get_default_feature_cols()  # 154
    t3 = feature_v1_columns()  # 72
    t3_no_time = [c for c in t3 if not c.startswith("time_")]
    t3_time = [c for c in t3 if c.startswith("time_")]
    a101_cols = alpha101_columns()
    a191_cols = alpha191_columns()
    all_cols = raw + t3_no_time + a101_cols + a191_cols + t3_time
    return all_cols, raw, t3_no_time, t3_time, a101_cols, a191_cols


def build_split(sym_dates, all_cols, raw_cols, t3_no_time_cols, t3_time_cols,
                a101_cols, a191_cols, cache_dir):
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

        a101_df = compute_alpha101(df)
        a191_df = compute_alpha191(df)

        sl = slice(valid_lo, valid_hi + 1)
        n = valid_hi - valid_lo + 1

        X_sl = np.empty((n, n_total_cols), dtype=np.float32)

        c_off = 0
        # raw 154
        X_sl[:, c_off:c_off + len(raw_cols)] = (
            df[raw_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        )
        c_off += len(raw_cols)
        # T3 no time (69)
        X_sl[:, c_off:c_off + len(t3_no_time_cols)] = (
            df[t3_no_time_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        )
        c_off += len(t3_no_time_cols)
        # alpha101
        X_sl[:, c_off:c_off + len(a101_cols)] = (
            a101_df[a101_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        )
        c_off += len(a101_cols)
        # alpha191
        X_sl[:, c_off:c_off + len(a191_cols)] = (
            a191_df[a191_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        )
        c_off += len(a191_cols)
        # T3 time-encoding (3 — at the very end)
        X_sl[:, c_off:c_off + len(t3_time_cols)] = (
            df[t3_time_cols].iloc[sl].to_numpy(dtype=np.float32, copy=False)
        )

        # Apply log1p to amount_delta (preserve sign) — same as T5b
        if amount_idx >= 0:
            col = X_sl[:, amount_idx]
            X_sl[:, amount_idx] = np.sign(col) * np.log1p(np.abs(col))

        # Replace any inf/NaN with 0 (defensive — alpha cols can produce these)
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
    ap.add_argument("--out-dir", default="experiments/T22_alpha101_alpha191/cache")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_cols, raw, t3_no_time, t3_time, a101_cols, a191_cols = get_scheme_i_columns()
    print(f"Scheme I total: {len(all_cols)} cols (154 raw + {len(t3_no_time)} T3-no-time "
          f"+ {len(a101_cols)} Alpha101 + {len(a191_cols)} Alpha191 + {len(t3_time)} time-enc)",
          flush=True)

    df0 = pd.read_parquet(os.path.join(args.cache_dir, "snapshot_sym0_date0_am.parquet"))
    a101_0 = compute_alpha101(df0)
    a191_0 = compute_alpha191(df0)
    print(f"  sanity: alpha101 shape {a101_0.shape}, alpha191 shape {a191_0.shape}", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True)
            continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], all_cols, raw, t3_no_time, t3_time,
                          a101_cols, a191_cols, args.cache_dir)
        out_path = os.path.join(args.out_dir, f"schemeI_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    names_path = os.path.join(args.out_dir, "schemeI_feat_names.txt")
    with open(names_path, "w") as f:
        for n in all_cols:
            f.write(n + "\n")
    print(f"feature names -> {names_path}  ({len(all_cols)} cols)", flush=True)

    names_no_time_path = os.path.join(args.out_dir, "schemeI_no_time_feat_names.txt")
    with open(names_no_time_path, "w") as f:
        for n in all_cols[:-len(t3_time)]:
            f.write(n + "\n")
    print(f"feature names (no time) -> {names_no_time_path}  "
          f"({len(all_cols) - len(t3_time)} cols)", flush=True)

    # Also save the index ranges for downstream slicing
    layout = {
        "raw_154": [0, len(raw)],
        "t3_no_time_69": [len(raw), len(raw) + len(t3_no_time)],
        "alpha101_33": [len(raw) + len(t3_no_time),
                        len(raw) + len(t3_no_time) + len(a101_cols)],
        "alpha191_124": [len(raw) + len(t3_no_time) + len(a101_cols),
                         len(raw) + len(t3_no_time) + len(a101_cols) + len(a191_cols)],
        "t3_time_3": [len(raw) + len(t3_no_time) + len(a101_cols) + len(a191_cols),
                      len(all_cols)],
        "n_a101": len(a101_cols),
        "n_a191": len(a191_cols),
    }
    with open(os.path.join(args.out_dir, "schemeI_layout.json"), "w") as f:
        json.dump(layout, f, indent=2)
    print(f"layout -> {os.path.join(args.out_dir, 'schemeI_layout.json')}", flush=True)


if __name__ == "__main__":
    main()
