"""T9 Scheme E: T2 Scheme A 154 last-tick + F1-F4 sym-agnostic features.

Implements the T8 diagnostic recommendations:
  F1: amount_delta z-score within the past-100-tick window (last-tick + q95)
  F2: spread-normalized Δmidprice (last-tick + max abs over window)
  F3: realized-vol-normalized current move (current |Δmid| / std(Δmid) over window)
  F4: order-flow imbalance ratios (5 ratios at last tick)

All features are sym-agnostic (within-window normalization or scale-free ratios).
No `sym` / `date` / `time` enters X.
Window size W=100 matches Predictor's input.

Output (per split): .npz with keys
  X        (N, D) float32        D = 154 + 10 = 164
  y60      (N,)   int8
  mp_t     (N,)   float32
  mp_t60   (N,)   float32
  sym      (N,)   int8           (only used for LOSO masking — never feature)
  date     (N,)   int16          (only metadata — never feature)
  sess_idx (N,)   int8
  t        (N,)   int16

Strict no-leakage: window at row t uses ticks [t-99..t] inclusive (right-closed).
This matches inference where Predictor sees the past 100 ticks ending at t.

Usage:
    python build_features.py --which train,val,test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.data.split import get_split, get_file_path  # noqa: E402
from src.data.dataset import get_default_feature_cols  # noqa: E402

WINDOW = 100
HORIZON = 60
EPS = 1e-8
SESS_TO_IDX = {"am": 0, "pm": 1}

# F-feature group names (used to build feat_names + ablation masks)
F1_NAMES = ["F1_amount_z_last", "F1_amount_z_q95"]
F2_NAMES = ["F2_spread_norm_move_last", "F2_spread_norm_move_max_abs"]
F3_NAMES = ["F3_rel_vol_quantile_last"]
F4_NAMES = [
    "F4_mb_over_lb",
    "F4_ma_over_la",
    "F4_cancel_ratio_buy",
    "F4_cancel_ratio_sell",
    "F4_book_pressure",
]


def _windowed_view(arr1d: np.ndarray, W: int) -> np.ndarray:
    """sliding_window_view on 1D array → (T-W+1, W); pad with NaN front."""
    sw = np.lib.stride_tricks.sliding_window_view(arr1d, window_shape=W)
    return sw  # (T-W+1, W)


def compute_F1(amount_delta: np.ndarray, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """F1: log1p(|amount|) z-score within window; sign-preserved.

    Returns (z_last, z_q95) each shape (T,); first W-1 rows = NaN.
    """
    log_amt = np.sign(amount_delta) * np.log1p(np.abs(amount_delta)).astype(np.float32)
    sw = _windowed_view(log_amt, W)  # (T-W+1, W)
    mean = sw.mean(axis=-1, dtype=np.float32)
    std = sw.std(axis=-1, dtype=np.float32) + EPS
    # z-score of every tick within its trailing window — but we only need the last tick.
    # last tick = sw[:, -1]; z_last = (sw[:, -1] - mean) / std
    z_full = (sw - mean[:, None]) / std[:, None]  # (T-W+1, W)
    z_last = z_full[:, -1].astype(np.float32)
    z_q95 = np.quantile(z_full, 0.95, axis=-1).astype(np.float32)
    pad = np.full((W - 1,), np.nan, dtype=np.float32)
    return np.concatenate([pad, z_last]), np.concatenate([pad, z_q95])


def compute_F2(midprice: np.ndarray, spread_actual: np.ndarray, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """F2: Δmidprice / spread_actual.

    spread_actual = ask1 - bid1 = spread1 + 1 (in normalized price units).
    Δmid at t uses mid[t] - mid[t-1]; mid[0]'s diff defined as 0.
    Returns (last, max_abs) each (T,); first W-1 rows = NaN (mid[0] uses prepend).
    """
    delta_mid = np.diff(midprice, prepend=midprice[:1]).astype(np.float32)
    norm_move = delta_mid / (np.abs(spread_actual) + EPS)
    sw = _windowed_view(norm_move, W)  # (T-W+1, W)
    last = sw[:, -1].astype(np.float32)
    max_abs = np.abs(sw).max(axis=-1).astype(np.float32)
    pad = np.full((W - 1,), np.nan, dtype=np.float32)
    return np.concatenate([pad, last]), np.concatenate([pad, max_abs])


def compute_F3(midprice: np.ndarray, W: int) -> np.ndarray:
    """F3: |current Δmid| / realized_vol(Δmid over window).

    Returns (T,); first W-1 = NaN.
    """
    delta_mid = np.diff(midprice, prepend=midprice[:1]).astype(np.float32)
    sw = _windowed_view(delta_mid, W)  # (T-W+1, W)
    realized_vol = sw.std(axis=-1, dtype=np.float32) + EPS
    cur_abs = np.abs(sw[:, -1]).astype(np.float32)
    rel = cur_abs / realized_vol
    pad = np.full((W - 1,), np.nan, dtype=np.float32)
    return np.concatenate([pad, rel])


def compute_F4(intst_cols: dict) -> np.ndarray:
    """F4: 5 order-flow imbalance ratios at last tick (per row).

    Each input is a (T,) array. Returns (T, 5) float32 — defined for all t (no padding).
    """
    lb = intst_cols["lb"]
    la = intst_cols["la"]
    mb = intst_cols["mb"]
    ma = intst_cols["ma"]
    cb = intst_cols["cb"]
    ca = intst_cols["ca"]

    f4_mb_over_lb = mb / (lb + mb + EPS)
    f4_ma_over_la = ma / (la + ma + EPS)
    f4_cancel_buy = cb / (lb + mb + cb + EPS)
    f4_cancel_sell = ca / (la + ma + ca + EPS)
    f4_book_pressure = (mb + lb) / (ma + la + mb + lb + EPS)

    out = np.stack(
        [f4_mb_over_lb, f4_ma_over_la, f4_cancel_buy, f4_cancel_sell, f4_book_pressure],
        axis=1,
    ).astype(np.float32)
    return out


def build_features_one_session(
    df: pd.DataFrame, feat_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X_valid (N, 154+12), y60, mp_t, mp_t60, t_valid)."""
    T = len(df)
    valid_lo = WINDOW - 1
    valid_hi = T - 1 - HORIZON
    if valid_hi < valid_lo:
        F = 154 + 10
        return (
            np.zeros((0, F), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int16),
        )

    X = df[feat_cols].to_numpy(dtype=np.float32, copy=True)  # (T, 154)
    # T2 Scheme A: log1p amount_delta in baseline (sign-preserved)
    if "amount_delta" in feat_cols:
        idx = feat_cols.index("amount_delta")
        col = X[:, idx]
        X[:, idx] = np.sign(col) * np.log1p(np.abs(col))

    midprice = df["midprice"].to_numpy(dtype=np.float32, copy=False)
    y60 = df["label_60"].to_numpy(dtype=np.int8, copy=False)

    # ----- F1 from raw amount_delta (BEFORE log1p) -----
    raw_amount = df["amount_delta"].to_numpy(dtype=np.float32, copy=False)
    f1_last, f1_q95 = compute_F1(raw_amount, WINDOW)

    # ----- F2 from raw bid1 / ask1 (BEFORE any log) -----
    raw_bid1 = df["bid1"].to_numpy(dtype=np.float32, copy=False)
    raw_ask1 = df["ask1"].to_numpy(dtype=np.float32, copy=False)
    raw_spread1 = df["spread1"].to_numpy(dtype=np.float32, copy=False)
    spread_actual = raw_spread1 + 1.0  # ask1 - bid1
    mid_actual = (raw_bid1 + raw_ask1) / 2.0
    f2_last, f2_max = compute_F2(mid_actual, spread_actual, WINDOW)

    # ----- F3 from same midprice -----
    f3_last = compute_F3(mid_actual, WINDOW)

    # ----- F4 from raw intst columns (no padding needed; defined per-tick) -----
    intst = {
        "lb": df["lb_intst"].to_numpy(dtype=np.float32, copy=False),
        "la": df["la_intst"].to_numpy(dtype=np.float32, copy=False),
        "mb": df["mb_intst"].to_numpy(dtype=np.float32, copy=False),
        "ma": df["ma_intst"].to_numpy(dtype=np.float32, copy=False),
        "cb": df["cb_intst"].to_numpy(dtype=np.float32, copy=False),
        "ca": df["ca_intst"].to_numpy(dtype=np.float32, copy=False),
    }
    f4_all = compute_F4(intst)  # (T, 5)

    # ----- Concatenate: 154 + 2 + 2 + 1 + 5 = 164 -----
    f1_block = np.stack([f1_last, f1_q95], axis=1)             # (T, 2)
    f2_block = np.stack([f2_last, f2_max], axis=1)             # (T, 2)
    f3_block = f3_last[:, None]                                 # (T, 1)
    feats_full = np.concatenate(
        [X, f1_block, f2_block, f3_block, f4_all], axis=1
    )  # (T, 164)

    sl = slice(valid_lo, valid_hi + 1)
    X_valid = feats_full[sl]
    y_valid = y60[sl]
    mp_t = midprice[sl]
    mp_t60 = midprice[valid_lo + HORIZON : valid_hi + 1 + HORIZON]
    t_valid = np.arange(valid_lo, valid_hi + 1, dtype=np.int16)

    if not np.isfinite(X_valid).all():
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

    return X_valid, y_valid, mp_t, mp_t60, t_valid


def feature_names(feat_cols: Sequence[str]) -> List[str]:
    return list(feat_cols) + F1_NAMES + F2_NAMES + F3_NAMES + F4_NAMES


def get_feature_group_indices(feat_names: List[str]) -> dict:
    """Returns {'A': [...], 'F1': [...], 'F2': [...], 'F3': [...], 'F4': [...]}.

    Indices are positions in the X column dimension (0..D-1).
    """
    groups = {"A": [], "F1": [], "F2": [], "F3": [], "F4": []}
    for i, n in enumerate(feat_names):
        if n.startswith("F1_"):
            groups["F1"].append(i)
        elif n.startswith("F2_"):
            groups["F2"].append(i)
        elif n.startswith("F3_"):
            groups["F3"].append(i)
        elif n.startswith("F4_"):
            groups["F4"].append(i)
        else:
            groups["A"].append(i)
    return groups


def build_split(sym_dates, feat_cols: List[str], data_dir: str) -> dict:
    Xs, ys, mts, mt60s = [], [], [], []
    syms, dates, sess_idxs, ts = [], [], [], []
    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    for i, (sym, date, sess) in enumerate(sym_dates):
        path = get_file_path(sym, date, sess, data_dir=data_dir)
        df = pd.read_parquet(path)
        X, y60, mp_t, mp_t60, t_valid = build_features_one_session(df, feat_cols)
        n = len(X)
        Xs.append(X)
        ys.append(y60)
        mts.append(mp_t)
        mt60s.append(mp_t60)
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(t_valid)
        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_sessions}] sessions, elapsed {elapsed:.1f}s", flush=True)

    out = {
        "X": np.concatenate(Xs, axis=0),
        "y60": np.concatenate(ys, axis=0),
        "mp_t": np.concatenate(mts, axis=0),
        "mp_t60": np.concatenate(mt60s, axis=0),
        "sym": np.concatenate(syms, axis=0),
        "date": np.concatenate(dates, axis=0),
        "sess_idx": np.concatenate(sess_idxs, axis=0),
        "t": np.concatenate(ts, axis=0),
    }
    print(
        f"  -> X shape {out['X'].shape} dtype {out['X'].dtype} "
        f"mem {out['X'].nbytes/1e9:.2f}GB",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="experiments/T9_schemeE/cache")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feat_cols = get_default_feature_cols()
    feat_names = feature_names(feat_cols)
    print(f"Scheme E: {len(feat_names)} features per sample (154 A + 2 F1 + 2 F2 + 1 F3 + 5 F4)",
          flush=True)
    groups = get_feature_group_indices(feat_names)
    for k, idxs in groups.items():
        print(f"  group {k}: {len(idxs)} cols (e.g. {idxs[:3]}{'...' if len(idxs)>3 else ''})",
              flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True)
            continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], feat_cols, args.data_dir)
        out_path = os.path.join(args.out_dir, f"schemeE_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    names_path = os.path.join(args.out_dir, "schemeE_feat_names.txt")
    with open(names_path, "w") as f:
        for n in feat_names:
            f.write(n + "\n")
    print(f"feature names -> {names_path}", flush=True)


if __name__ == "__main__":
    main()
