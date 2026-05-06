"""T20 Scheme G: feature explosion (226d -> ~400d).

Adds these on top of Scheme C 226d:

Group A (20):  Multi-window MLOFI, W in {10, 30}, levels 1..10  (W=100 dropped: NaN at t=99)
Group B (20):  Bid-only OFI (e_b_k), W in {20, 60}, levels 1..10
Group C (20):  Ask-only OFI (e_a_k), W in {20, 60}, levels 1..10
Group D (8):   Rolling skew/kurt (W in {20,60}) of [mid_diff, imbalance]
Group E (8):   Rolling stats max/min/q10/q90 of (midprice1[s] - midprice1[t]) at W in {20, 60}
Group F (8):   Rolling stats max/min/q10/q90 of spread1 at W in {20, 60}
Group G (8):   Rolling stats max/min/q10/q90 of imbalance at W in {20, 60}
Group H (40):  Lag deltas of 10 important features x 4 lags (5, 10, 20, 60)
Group I (4):   Multi-W mid momentum: midprice1[t] - midprice1[t-W], W in {5,10,20,60}
Group J (4):   Multi-W wmp_lvl1 momentum, W in {5,10,20,60}
Group K (16):  Interactions (mostly products of important pairs)
Group L (3):   EWMA of mid_diff, alpha in {0.05, 0.1, 0.3}
Group M (3):   EWMA of imbalance, alpha in {0.05, 0.1, 0.3}
Group N (3):   Net order flow components: lb_intst-cb_intst, la_intst-ca_intst, mb_intst-ma_intst
Group O (4):   RV at W in {30, 80} on log-return of WMP1 + RV at W in {30, 80} on midprice1 diff

Total new: ~169 dims.  226 + ~169 = ~395 dims.
All vectorized; rolling ops only use ticks within the 100-row inference window.
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
sys.path.insert(0, os.path.join(ROOT, "experiments", "T3_features_v1"))

from src.data.split import get_split  # noqa: E402
from src.data.dataset import get_default_feature_cols  # noqa: E402
from compute import feature_v1_columns  # noqa: E402

WINDOW = 100
HORIZONS = (5, 10, 20, 40, 60)
MAX_HORIZON = max(HORIZONS)
SESS_TO_IDX = {"am": 0, "pm": 1}

# === New feature config ===
G_MLOFI_WINDOWS = (10, 30)         # W=100 omitted (NaN at t=99 due to e[0]=NaN)
G_BIDASK_OFI_WINDOWS = (20, 60)
G_ROLLING_HIGHER_WINDOWS = (20, 60)
G_LAG_LAGS = (5, 10, 20, 60)
G_MOMENTUM_WINDOWS = (5, 10, 20, 60)
G_EWMA_ALPHAS = (0.05, 0.1, 0.3)
G_RV_WINDOWS = (30, 80)
LOB_LEVELS = tuple(range(1, 11))

# Lag-delta features (representative state vars). These exist as base cols.
LAG_FEATS = (
    "midprice1", "wmp_lvl1", "spread1", "imbalance", "cumspread",
    "avgbid", "avgask", "bsize1", "asize1", "close",
)


# ---------- new feature computation ----------

def _compute_e_per_lvl(df: pd.DataFrame) -> dict:
    """Per-tick e_k(t) for combined OFI (same as T3.compute_mlofi inner)."""
    e_per_lvl = {}
    for k in LOB_LEVELS:
        b = df[f"bid{k}"]
        a = df[f"ask{k}"]
        bs = df[f"bsize{k}"]
        as_ = df[f"asize{k}"]
        b_prev = b.shift(1)
        a_prev = a.shift(1)
        bs_prev = bs.shift(1)
        as_prev = as_.shift(1)
        ind_b_up = (b >= b_prev).astype(np.float64)
        ind_b_dn = (b <= b_prev).astype(np.float64)
        ind_a_dn = (a <= a_prev).astype(np.float64)
        ind_a_up = (a >= a_prev).astype(np.float64)
        e = ind_b_up * bs - ind_b_dn * bs_prev - ind_a_dn * as_ + ind_a_up * as_prev
        e_per_lvl[k] = e
    return e_per_lvl


def _compute_e_bid_only(df: pd.DataFrame) -> dict:
    """Bid-side only contribution to OFI:
        e_b_k(t) = I(b_k_t >= b_k_{t-1}) * bs_k_t - I(b_k_t <= b_k_{t-1}) * bs_k_{t-1}
    """
    eb = {}
    for k in LOB_LEVELS:
        b = df[f"bid{k}"]
        bs = df[f"bsize{k}"]
        b_prev = b.shift(1)
        bs_prev = bs.shift(1)
        ind_b_up = (b >= b_prev).astype(np.float64)
        ind_b_dn = (b <= b_prev).astype(np.float64)
        eb[k] = ind_b_up * bs - ind_b_dn * bs_prev
    return eb


def _compute_e_ask_only(df: pd.DataFrame) -> dict:
    """Ask-side only contribution to OFI:
        e_a_k(t) = -I(a_k_t <= a_k_{t-1}) * as_k_t + I(a_k_t >= a_k_{t-1}) * as_k_{t-1}
    """
    ea = {}
    for k in LOB_LEVELS:
        a = df[f"ask{k}"]
        as_ = df[f"asize{k}"]
        a_prev = a.shift(1)
        as_prev = as_.shift(1)
        ind_a_dn = (a <= a_prev).astype(np.float64)
        ind_a_up = (a >= a_prev).astype(np.float64)
        ea[k] = -ind_a_dn * as_ + ind_a_up * as_prev
    return ea


def _rolling_q(s: pd.Series, W: int, q: float) -> pd.Series:
    return s.rolling(window=W, min_periods=W).quantile(q)


def compute_g_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all Scheme G new features for one session DataFrame.

    Input: parquet with 235 cols (3 meta + 154 raw + 72 T3 + 6 labels/midprice).
    Output: DataFrame with new cols (~169 dims).
    """
    out = {}
    n = len(df)

    # ------- Group A: multi-window MLOFI W in {10, 30} -------
    e_per_lvl = _compute_e_per_lvl(df)
    for W in G_MLOFI_WINDOWS:
        for k in LOB_LEVELS:
            out[f"g_mlofi_W{W}_lvl{k}"] = (
                e_per_lvl[k].rolling(window=W, min_periods=W).sum()
            )

    # ------- Group B: bid-only OFI W in {20, 60} -------
    eb = _compute_e_bid_only(df)
    for W in G_BIDASK_OFI_WINDOWS:
        for k in LOB_LEVELS:
            out[f"g_bidofi_W{W}_lvl{k}"] = (
                eb[k].rolling(window=W, min_periods=W).sum()
            )

    # ------- Group C: ask-only OFI W in {20, 60} -------
    ea = _compute_e_ask_only(df)
    for W in G_BIDASK_OFI_WINDOWS:
        for k in LOB_LEVELS:
            out[f"g_askofi_W{W}_lvl{k}"] = (
                ea[k].rolling(window=W, min_periods=W).sum()
            )

    # ------- Group D: skew/kurt of mid_diff and imbalance -------
    midprice1 = df["midprice1"]
    mid_diff = midprice1.diff()
    imbalance = df["imbalance"]
    for W in G_ROLLING_HIGHER_WINDOWS:
        out[f"g_mid_diff_skew_W{W}"] = mid_diff.rolling(W, min_periods=W).skew()
        out[f"g_mid_diff_kurt_W{W}"] = mid_diff.rolling(W, min_periods=W).kurt()
        out[f"g_imbalance_skew_W{W}"] = imbalance.rolling(W, min_periods=W).skew()
        out[f"g_imbalance_kurt_W{W}"] = imbalance.rolling(W, min_periods=W).kurt()

    # ------- Group E: rolling stats of (midprice1[s] - midprice1[t]) -------
    # max/min of (midprice1[s] - midprice1[t]) over s in [t-W+1..t] = (max midprice in window) - midprice1[t]
    # quantiles likewise.
    for W in G_ROLLING_HIGHER_WINDOWS:
        roll_max = midprice1.rolling(W, min_periods=W).max()
        roll_min = midprice1.rolling(W, min_periods=W).min()
        roll_q10 = _rolling_q(midprice1, W, 0.1)
        roll_q90 = _rolling_q(midprice1, W, 0.9)
        out[f"g_mid_max_rel_W{W}"] = roll_max - midprice1
        out[f"g_mid_min_rel_W{W}"] = roll_min - midprice1
        out[f"g_mid_q10_rel_W{W}"] = roll_q10 - midprice1
        out[f"g_mid_q90_rel_W{W}"] = roll_q90 - midprice1

    # ------- Group F: rolling stats of spread1 -------
    spread1 = df["spread1"]
    for W in G_ROLLING_HIGHER_WINDOWS:
        out[f"g_spread1_max_W{W}"] = spread1.rolling(W, min_periods=W).max()
        out[f"g_spread1_min_W{W}"] = spread1.rolling(W, min_periods=W).min()
        out[f"g_spread1_q10_W{W}"] = _rolling_q(spread1, W, 0.1)
        out[f"g_spread1_q90_W{W}"] = _rolling_q(spread1, W, 0.9)

    # ------- Group G: rolling stats of imbalance -------
    for W in G_ROLLING_HIGHER_WINDOWS:
        out[f"g_imbalance_max_W{W}"] = imbalance.rolling(W, min_periods=W).max()
        out[f"g_imbalance_min_W{W}"] = imbalance.rolling(W, min_periods=W).min()
        out[f"g_imbalance_q10_W{W}"] = _rolling_q(imbalance, W, 0.1)
        out[f"g_imbalance_q90_W{W}"] = _rolling_q(imbalance, W, 0.9)

    # ------- Group H: lag deltas -------
    for feat in LAG_FEATS:
        s = df[feat]
        for L in G_LAG_LAGS:
            out[f"g_lagdelta_{feat}_L{L}"] = s - s.shift(L)

    # ------- Group I: midprice1 momentum -------
    for L in G_MOMENTUM_WINDOWS:
        out[f"g_mid_mom_W{L}"] = midprice1 - midprice1.shift(L)

    # ------- Group J: wmp_lvl1 momentum -------
    wmp1 = df["wmp_lvl1"]
    for L in G_MOMENTUM_WINDOWS:
        out[f"g_wmp_mom_W{L}"] = wmp1 - wmp1.shift(L)

    # ------- Group K: interactions (16 dims) -------
    bid1 = df["bid1"]; ask1 = df["ask1"]
    bsize1 = df["bsize1"]; asize1 = df["asize1"]
    rv_w20 = df["rv_w20"]
    rv_w50 = df["rv_w50"]
    mlofi60_lvl1 = df["mlofi_W60_lvl1"]
    mlofi5_lvl1 = df["mlofi_W5_lvl1"]
    cumspread = df["cumspread"]

    out["g_int_spread1_x_imbalance"] = spread1 * imbalance
    out["g_int_imbalance_x_rv_w20"] = imbalance * rv_w20
    out["g_int_imbalance_x_rv_w50"] = imbalance * rv_w50
    out["g_int_mlofi60_x_imbalance"] = mlofi60_lvl1 * imbalance
    out["g_int_mlofi5_x_imbalance"] = mlofi5_lvl1 * imbalance
    out["g_int_mlofi60_x_spread1"] = mlofi60_lvl1 * spread1
    out["g_int_spread1_x_rv_w20"] = spread1 * rv_w20
    out["g_int_bsize_minus_asize_lvl1"] = bsize1 - asize1
    out["g_int_bid_diff_x_ask_diff"] = df["bid_diff1"] * df["ask_diff1"]
    out["g_int_close_x_volume_delta"] = df["close"] * df["volume_delta"]
    out["g_int_cumspread_x_imbalance"] = cumspread * imbalance
    out["g_int_wmp_balance_x_spread1"] = df["wmp_balance_12"] * spread1
    out["g_int_avgbid_minus_avgask"] = df["avgbid"] - df["avgask"]
    out["g_int_totalbsize_minus_totalasize"] = df["totalbsize"] - df["totalasize"]
    out["g_int_bsize_rate_x_asize_rate"] = df["bsize_rate1"] * df["asize_rate1"]
    out["g_int_mlofi_short_long_diff"] = mlofi5_lvl1 - mlofi60_lvl1

    # ------- Group L: EWMA of mid_diff -------
    for alpha in G_EWMA_ALPHAS:
        out[f"g_ewma_a{alpha}_mid_diff"] = (
            mid_diff.fillna(0.0).ewm(alpha=alpha, adjust=False).mean()
        )

    # ------- Group M: EWMA of imbalance -------
    for alpha in G_EWMA_ALPHAS:
        out[f"g_ewma_a{alpha}_imbalance"] = (
            imbalance.ewm(alpha=alpha, adjust=False).mean()
        )

    # ------- Group N: net order flow components -------
    out["g_net_lb_minus_cb"] = df["lb_intst"] - df["cb_intst"]
    out["g_net_la_minus_ca"] = df["la_intst"] - df["ca_intst"]
    out["g_net_mb_minus_ma"] = df["mb_intst"] - df["ma_intst"]

    # ------- Group O: extra RV windows -------
    safe = wmp1 + 1.0
    safe = safe.where(safe > 0, np.nan)
    log_ret = np.log(safe / safe.shift(1))
    sq = log_ret * log_ret
    for W in G_RV_WINDOWS:
        rolled = sq.rolling(window=W, min_periods=W).sum().clip(lower=0.0)
        out[f"g_rv_w{W}"] = np.sqrt(rolled)
    # Midprice diff RV
    md_sq = mid_diff * mid_diff
    for W in G_RV_WINDOWS:
        out[f"g_mid_diff_rv_w{W}"] = np.sqrt(
            md_sq.rolling(window=W, min_periods=W).sum().clip(lower=0.0)
        )

    return pd.DataFrame(out, index=df.index)


def g_feature_columns() -> List[str]:
    """Return the new G feature column names in order."""
    cols = []
    for W in G_MLOFI_WINDOWS:
        for k in LOB_LEVELS:
            cols.append(f"g_mlofi_W{W}_lvl{k}")
    for W in G_BIDASK_OFI_WINDOWS:
        for k in LOB_LEVELS:
            cols.append(f"g_bidofi_W{W}_lvl{k}")
    for W in G_BIDASK_OFI_WINDOWS:
        for k in LOB_LEVELS:
            cols.append(f"g_askofi_W{W}_lvl{k}")
    for W in G_ROLLING_HIGHER_WINDOWS:
        cols.append(f"g_mid_diff_skew_W{W}")
        cols.append(f"g_mid_diff_kurt_W{W}")
        cols.append(f"g_imbalance_skew_W{W}")
        cols.append(f"g_imbalance_kurt_W{W}")
    for W in G_ROLLING_HIGHER_WINDOWS:
        cols.append(f"g_mid_max_rel_W{W}")
        cols.append(f"g_mid_min_rel_W{W}")
        cols.append(f"g_mid_q10_rel_W{W}")
        cols.append(f"g_mid_q90_rel_W{W}")
    for W in G_ROLLING_HIGHER_WINDOWS:
        cols.append(f"g_spread1_max_W{W}")
        cols.append(f"g_spread1_min_W{W}")
        cols.append(f"g_spread1_q10_W{W}")
        cols.append(f"g_spread1_q90_W{W}")
    for W in G_ROLLING_HIGHER_WINDOWS:
        cols.append(f"g_imbalance_max_W{W}")
        cols.append(f"g_imbalance_min_W{W}")
        cols.append(f"g_imbalance_q10_W{W}")
        cols.append(f"g_imbalance_q90_W{W}")
    for feat in LAG_FEATS:
        for L in G_LAG_LAGS:
            cols.append(f"g_lagdelta_{feat}_L{L}")
    for L in G_MOMENTUM_WINDOWS:
        cols.append(f"g_mid_mom_W{L}")
    for L in G_MOMENTUM_WINDOWS:
        cols.append(f"g_wmp_mom_W{L}")
    cols += [
        "g_int_spread1_x_imbalance",
        "g_int_imbalance_x_rv_w20",
        "g_int_imbalance_x_rv_w50",
        "g_int_mlofi60_x_imbalance",
        "g_int_mlofi5_x_imbalance",
        "g_int_mlofi60_x_spread1",
        "g_int_spread1_x_rv_w20",
        "g_int_bsize_minus_asize_lvl1",
        "g_int_bid_diff_x_ask_diff",
        "g_int_close_x_volume_delta",
        "g_int_cumspread_x_imbalance",
        "g_int_wmp_balance_x_spread1",
        "g_int_avgbid_minus_avgask",
        "g_int_totalbsize_minus_totalasize",
        "g_int_bsize_rate_x_asize_rate",
        "g_int_mlofi_short_long_diff",
    ]
    for alpha in G_EWMA_ALPHAS:
        cols.append(f"g_ewma_a{alpha}_mid_diff")
    for alpha in G_EWMA_ALPHAS:
        cols.append(f"g_ewma_a{alpha}_imbalance")
    cols += ["g_net_lb_minus_cb", "g_net_la_minus_ca", "g_net_mb_minus_ma"]
    for W in G_RV_WINDOWS:
        cols.append(f"g_rv_w{W}")
    for W in G_RV_WINDOWS:
        cols.append(f"g_mid_diff_rv_w{W}")
    return cols


def get_scheme_g_columns() -> List[str]:
    """226 base + ~169 new G."""
    raw = get_default_feature_cols()
    t3 = feature_v1_columns()
    g = g_feature_columns()
    return raw + t3 + g


# ---------- per-session build (mirrors T5b) ----------

def build_split(sym_dates, base_cols: List[str], g_cols: List[str], cache_dir: str) -> dict:
    Xs = []
    ys = {h: [] for h in HORIZONS}
    mps = {f"mp_t{h}": [] for h in HORIZONS}
    mp_t = []
    syms, dates, sess_idxs, ts = [], [], [], []

    n_sessions = len(sym_dates)
    log_every = max(1, n_sessions // 20)
    t0 = time.time()
    n_g = len(g_cols)

    for i, (sym, date, sess) in enumerate(sym_dates):
        path = os.path.join(cache_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        df = pd.read_parquet(path)
        T = len(df)
        valid_lo = WINDOW - 1
        valid_hi = T - 1 - MAX_HORIZON
        if valid_hi < valid_lo:
            continue

        # Compute new G feats on full session (vectorized)
        g_df = compute_g_features(df)
        # Concat base 226 cols + new G cols horizontally then slice
        full = pd.concat([df[base_cols], g_df], axis=1)

        sl = slice(valid_lo, valid_hi + 1)
        X_sl = full.iloc[sl].to_numpy(dtype=np.float32, copy=False)

        # log1p amount_delta (preserve sign)
        if "amount_delta" in base_cols:
            j = base_cols.index("amount_delta")
            col = X_sl[:, j].copy()
            X_sl = X_sl.copy()
            X_sl[:, j] = np.sign(col) * np.log1p(np.abs(col))

        # Replace any inf/nan in G feats (skew/kurt can be NaN if no variance)
        # We need to keep training valid even when std=0 over a window → fillna 0 for new feats
        n_base = len(base_cols)
        new_block = X_sl[:, n_base:]
        if np.isnan(new_block).any() or np.isinf(new_block).any():
            new_block = np.where(np.isfinite(new_block), new_block, 0.0).astype(np.float32)
            X_sl[:, n_base:] = new_block

        Xs.append(X_sl)
        midprice = df["midprice"].to_numpy(dtype=np.float32, copy=False)
        mp_t.append(midprice[sl])

        for h in HORIZONS:
            y = df[f"label_{h}"].iloc[sl].to_numpy(dtype=np.int8, copy=False)
            ys[h].append(y)
            mp_h = midprice[valid_lo + h : valid_hi + 1 + h]
            mps[f"mp_t{h}"].append(mp_h)

        n = len(X_sl)
        syms.append(np.full(n, sym, dtype=np.int8))
        dates.append(np.full(n, date, dtype=np.int16))
        sess_idxs.append(np.full(n, SESS_TO_IDX[sess], dtype=np.int8))
        ts.append(np.arange(valid_lo, valid_hi + 1, dtype=np.int16))

        if (i + 1) % log_every == 0 or i + 1 == n_sessions:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{n_sessions}] sessions; elapsed {elapsed:.1f}s "
                  f"(g_dim={n_g})", flush=True)

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
    print(f"  -> X shape {out['X'].shape} mem {out['X'].nbytes/1e9:.2f}GB", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="local_debug")
    ap.add_argument("--cache-dir", default="data/features_v1")
    ap.add_argument("--out-dir", default="experiments/T20_schemeG/cache")
    ap.add_argument("--which", default="train,val,test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw = get_default_feature_cols()
    t3 = feature_v1_columns()
    base_cols = raw + t3
    g_cols = g_feature_columns()
    feat_cols = base_cols + g_cols
    print(f"Scheme G: 154 raw + 72 T3 + {len(g_cols)} G = {len(feat_cols)} features", flush=True)

    # NaN check on first session (should still NaN-clean after slicing 99..end)
    df0 = pd.read_parquet(os.path.join(args.cache_dir, "snapshot_sym0_date0_am.parquet"))
    g_df0 = compute_g_features(df0)
    full0 = pd.concat([df0[base_cols], g_df0], axis=1)
    has_nan = full0[feat_cols].iloc[WINDOW - 1 :].isna().any()
    nan_cols = has_nan[has_nan].index.tolist()
    print(f"  first session NaN cols after WINDOW-1 (will be filled to 0): {nan_cols[:10]}"
          f" (total {len(nan_cols)})", flush=True)

    split = get_split(args.split)
    wanted = [w.strip() for w in args.which.split(",")]
    for key in wanted:
        if key not in split:
            print(f"  skip unknown key {key}", flush=True); continue
        print(f"=== Building {key} ({len(split[key])} sessions) ===", flush=True)
        out = build_split(split[key], base_cols, g_cols, args.cache_dir)
        out_path = os.path.join(args.out_dir, f"schemeG_{key}.npz")
        np.savez(out_path, **out)
        print(f"  saved -> {out_path}", flush=True)

    names_path = os.path.join(args.out_dir, "schemeG_feat_names.txt")
    with open(names_path, "w") as f:
        for n in feat_cols:
            f.write(n + "\n")
    print(f"feature names ({len(feat_cols)}) -> {names_path}", flush=True)

    # 223-d names file (drop last 3 time-encoding) for compatibility with iter pipeline
    # Scheme G feats come AFTER time-encoding (idx 224..225..226 are time, then G).
    # So we drop time-encoding from the *base* and keep G appended.
    base_223 = base_cols[:-3]  # drop time_minutes_since_session_start, time_session_progress, time_is_pm
    feat_223 = base_223 + g_cols
    names223_path = os.path.join(args.out_dir, "schemeG_223base_feat_names.txt")
    with open(names223_path, "w") as f:
        for n in feat_223:
            f.write(n + "\n")
    print(f"feature names (223base+G = {len(feat_223)}) -> {names223_path}", flush=True)


if __name__ == "__main__":
    main()
