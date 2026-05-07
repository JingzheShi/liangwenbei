"""Compute sym-agnostic regime indicators from 100-tick window for OOF rows.

For each (sym, date, session, t) prediction point in the 5-fold OOF, computes:
  regime_vol     = std over last 100 ticks of midprice
  regime_imb     = mean over last 100 ticks of |imbalance|
  regime_intst   = mean over last 100 ticks of total order intensity
  regime_spread  = mean over last 100 ticks of cumspread
  regime_ret     = (midprice_t - midprice_{t-99}) / (midprice_{t-99}+1)

All from 100-tick window only — no sym/date dependence beyond the window
contents themselves. Saves regime_features.parquet keyed by (sym, date,
session, t) for join.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

DATA_DIR = "/root/projects/liangwenbei_workdir/data"
HERE = os.path.dirname(os.path.abspath(__file__))
WIN = 100
H = 60


def regime_for_session(snap: pd.DataFrame) -> pd.DataFrame:
    """Vectorized rolling 100-tick regime features.

    snap is for ONE (sym, date, session) — 2001 rows in time order.
    Returns rows for t in [99, 1940] only (matches OOF horizon).
    """
    snap = snap.sort_values("time").reset_index(drop=True)
    midprice = snap["midprice"].to_numpy(np.float64)
    imbalance = snap["imbalance"].to_numpy(np.float64)
    cumspread = snap["cumspread"].to_numpy(np.float64)

    intst_cols = ["lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"]
    intst_total = snap[intst_cols].to_numpy(np.float64).sum(1)

    s = pd.Series(midprice)
    vol = s.rolling(WIN).std().to_numpy()
    ret = (midprice - np.roll(midprice, WIN - 1)) / (np.roll(midprice, WIN - 1) + 1.0)
    ret[: WIN - 1] = np.nan

    imb = pd.Series(np.abs(imbalance)).rolling(WIN).mean().to_numpy()
    intst = pd.Series(intst_total).rolling(WIN).mean().to_numpy()
    spread = pd.Series(cumspread).rolling(WIN).mean().to_numpy()

    out = pd.DataFrame({
        "sym": snap["sym"].to_numpy(np.int8),
        "date": snap["date"].to_numpy(np.int16),
        "t": np.arange(len(snap), dtype=np.int16),
        "regime_vol": vol.astype(np.float32),
        "regime_imb": imb.astype(np.float32),
        "regime_intst": intst.astype(np.float32),
        "regime_spread": spread.astype(np.float32),
        "regime_ret": ret.astype(np.float32),
    })
    # Limit to OOF prediction t range (99..1940 corresponds to having full
    # window AND room for h=60 ahead)
    return out[(out["t"] >= 99) & (out["t"] <= 2000 - H)].reset_index(drop=True)


def main():
    t0 = time.time()
    parts = []
    for sym in range(5):
        for date in range(96, 120):
            for session in ("am", "pm"):
                p = os.path.join(DATA_DIR, f"snapshot_sym{sym}_date{date}_{session}.parquet")
                if not os.path.exists(p):
                    print(f"missing: {p}", flush=True); continue
                snap = pd.read_parquet(p)
                feats = regime_for_session(snap)
                feats["session"] = session
                parts.append(feats)
        if sym % 1 == 0:
            print(f"  sym {sym} done in {time.time()-t0:.1f}s", flush=True)
    df = pd.concat(parts, ignore_index=True)
    print(f"\nTotal regime rows: {len(df):,}, elapsed {time.time()-t0:.1f}s")
    print(df.head())
    out = os.path.join(HERE, "regime_features.parquet")
    df.to_parquet(out, index=False)
    print(f"Saved -> {out}")

    # Quantile summary across all 5 folds
    print("\nQuantiles (across all rows):")
    for c in ["regime_vol", "regime_imb", "regime_intst", "regime_spread", "regime_ret"]:
        q = df[c].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        print(f"  {c}: {q}")


if __name__ == "__main__":
    main()
