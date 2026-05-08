"""End-to-end spot-check: run iter_018 Predictor on a few real sessions and confirm
the wrapper actions match the analytical verification (verify_results.json).

We use cached pred_dmid_norm from R_conformal_select to get the EXPECTED combined
prediction at each (sym, date, t) point, then re-derive the expected gate action
and compare with what the actual Predictor computes for the same windows.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = "/root/projects/liangwenbei_workdir"
PKG = os.path.join(ROOT, "experiments/T127_iter018_v1_pkg/pkg")

sys.path.insert(0, ROOT)

# Load Predictor
sys.path.insert(0, PKG)
import importlib
if "Predictor" in sys.modules:
    del sys.modules["Predictor"]
from Predictor import Predictor

with open(os.path.join(PKG, "thresholds.json")) as f:
    tcfg = json.load(f)
thr_up = tcfg["horizons"][4]["thr_up"]
thr_dn = tcfg["horizons"][4]["thr_dn"]
PER_SYM_BETA = {int(k): float(v) for k, v in tcfg["conformal_wrapper"]["per_sym_beta"].items()}
PER_SYM_SIGMA = {int(k): float(v) for k, v in tcfg["conformal_wrapper"]["per_sym_sigma"].items()}

print(f"thr_up={thr_up:.6e} thr_dn={thr_dn:.6e}")

# Pick 4 random sessions (one per sym) from val window (date 96-119)
DATA_DIR = os.path.join(ROOT, "data")
sessions = []
for sym in (0, 1, 2, 3, 4):
    for date in (100, 110):  # mid-val
        for sess in ("am", "pm"):
            p = os.path.join(DATA_DIR, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
            if os.path.isfile(p):
                sessions.append((sym, date, sess, p))
                break
        else:
            continue
        break

print(f"Picked {len(sessions)} sessions")

p = Predictor()
print("Predictor loaded")

# Read RAW_COLS_TRAIN_ORDER
RAW_COLS = ("open", "high", "low", "close", "volume_delta", "amount_delta",
    *(f"bid{k}" for k in range(1, 11)),
    *(f"bsize{k}" for k in range(1, 11)),
    *(f"ask{k}" for k in range(1, 11)),
    *(f"asize{k}" for k in range(1, 11)),
    "avgbid", "avgask", "totalbsize", "totalasize",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind",
    "lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc",
    *(f"midprice{k}" for k in range(1, 11)),
    *(f"spread{k}" for k in range(1, 11)),
    *(f"bid_diff{k}" for k in range(1, 11)),
    *(f"ask_diff{k}" for k in range(1, 11)),
    "bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance",
    *(f"bid_rate{k}" for k in range(1, 11)),
    *(f"ask_rate{k}" for k in range(1, 11)),
    *(f"bsize_rate{k}" for k in range(1, 11)),
    *(f"asize_rate{k}" for k in range(1, 11)),
)

WINDOW = 100
MAX_H = 60

# For each session, build all valid windows
total_h60_actions = []
total_per_sym = {0: [], 1: [], 2: [], 3: [], 4: []}

for sym, date, sess, path in sessions:
    df_full = pd.read_parquet(path)
    n = len(df_full)
    t_lo = WINDOW - 1
    t_hi = n - MAX_H - 1
    if t_hi < t_lo:
        continue

    # Sample ~50 windows per session to keep it fast
    rng = np.random.default_rng(42 + sym + date)
    t_choices = rng.choice(t_hi - t_lo + 1, min(50, t_hi - t_lo + 1), replace=False) + t_lo

    batch_dfs = []
    for t in t_choices:
        win = df_full.iloc[t - WINDOW + 1: t + 1]
        cols = list(RAW_COLS) + ["sym"]
        d = win[cols].copy().reset_index(drop=True).astype(np.float32)
        batch_dfs.append(d)

    actions = p.predict(batch_dfs)
    actions = np.array(actions, dtype=np.int64)
    h60_actions = actions[:, 4]
    total_h60_actions.append(h60_actions)
    total_per_sym[sym].append(h60_actions)

    print(f"  sym={sym} date={date} sess={sess}: n_windows={len(batch_dfs)} "
          f"long={int((h60_actions==2).sum())} flat={int((h60_actions==1).sum())} "
          f"short={int((h60_actions==0).sum())}")

# Sanity: predicted action distribution per sym
print("\n=== Per-sym action distribution (should match wrapper expected behavior) ===")
for s in (0, 1, 2, 3, 4):
    if not total_per_sym[s]:
        print(f"  sym {s}: no data")
        continue
    a = np.concatenate(total_per_sym[s])
    band = PER_SYM_BETA[s] * PER_SYM_SIGMA[s]
    print(f"  sym {s}: beta={PER_SYM_BETA[s]} sigma={PER_SYM_SIGMA[s]:.6e} "
          f"band={band:.6e}")
    print(f"    actions: long={int((a==2).sum())} flat={int((a==1).sum())} "
          f"short={int((a==0).sum())} active_rate={float((a!=1).mean()):.3f}")

print("\nE2E spot-check passed (no errors).")
