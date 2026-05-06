"""
T8: sym=2 diagnostic — per-sym aggregate statistics across all 1200 files.

For each sym in {0..4} compute:
- midprice/return distribution stats
- spread, volume, amount, imbalance, order intensities
- label_60 distribution

Output: results.json + sym_stats_table.csv
"""
import os
import json
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

DATA_DIR = "/root/projects/liangwenbei_workdir/data"
OUT_DIR = "/root/projects/liangwenbei_workdir/experiments/T8_sym2_diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

# Columns we care about (saves IO time)
KEEP_COLS = [
    "date", "sym", "time",
    "midprice", "spread1", "volume_delta", "amount_delta",
    "imbalance", "cumspread",
    "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
    "label_5", "label_10", "label_20", "label_40", "label_60",
    "totalbsize", "totalasize",
    "bid_mean", "ask_mean",
    "bsize_mean", "asize_mean",
    "midprice1",
]

print("[load] loading per-sym aggregates ...")
all_dfs = []
files = sorted(glob(f"{DATA_DIR}/snapshot_sym*.parquet"))
print(f"[load] {len(files)} parquet files found")

# To compute log returns we need within-session continuity, so load per (sym,date,session)
# But for global stats we only need the columns. Let's load all 1200 files but only kept cols.

for f in files:
    df = pd.read_parquet(f, columns=KEEP_COLS)
    # session info from filename
    base = Path(f).stem  # snapshot_sym2_date100_am
    parts = base.split("_")
    df["session"] = parts[-1]  # am or pm
    all_dfs.append(df)

big = pd.concat(all_dfs, ignore_index=True)
# sym is stored as string in parquet; cast to int
big["sym"] = big["sym"].astype(int)
print(f"[load] big df: {big.shape}, sym uniques: {sorted(big['sym'].unique())}")
del all_dfs

# ---------- Per-sym stats ----------
def quantile_stats(s):
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "q05": float(s.quantile(0.05)),
        "q50": float(s.quantile(0.5)),
        "q95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }

stats = {}
for sym in [0, 1, 2, 3, 4]:
    sub = big[big["sym"] == sym]
    print(f"[stats] sym={sym}: {len(sub)} rows")
    s = {}

    # midprice
    s["midprice"] = quantile_stats(sub["midprice"])

    # log return per tick (within session)
    sub_sorted = sub.sort_values(["date", "session", "time"])
    grp = sub_sorted.groupby(["date", "session"])["midprice"]
    # midprice is a "rate" already; compute (mid_t+1 - mid_t)
    mid_diff = grp.diff()
    abs_diff = mid_diff.abs()
    s["mid_diff"] = {
        "mean": float(mid_diff.mean()),
        "std": float(mid_diff.std()),
        "abs_mean": float(abs_diff.mean()),
        "abs_q95": float(abs_diff.quantile(0.95)),
        "skew": float(mid_diff.skew()),
        "kurt": float(mid_diff.kurtosis()),
    }

    # spread1
    s["spread1"] = quantile_stats(sub["spread1"])

    # volume_delta
    s["volume_delta"] = quantile_stats(sub["volume_delta"])
    # amount_delta (huge magnitude — log1p)
    log_amt = np.log1p(sub["amount_delta"].clip(lower=0))
    s["log_amount_delta"] = quantile_stats(log_amt)
    s["amount_delta_raw_mean"] = float(sub["amount_delta"].mean())

    # imbalance
    s["imbalance"] = quantile_stats(sub["imbalance"])
    s["cumspread"] = quantile_stats(sub["cumspread"])

    # label distributions
    for lab in ["label_5", "label_10", "label_20", "label_40", "label_60"]:
        cnt = sub[lab].value_counts(normalize=True).to_dict()
        s[f"{lab}_dist"] = {f"p{int(k)}": float(v) for k, v in cnt.items()}

    # 6 order intensities
    intst = {}
    for col in ["lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"]:
        intst[col] = {"mean": float(sub[col].mean()), "std": float(sub[col].std())}
    s["order_intensities"] = intst

    # Big-move rate: |midprice_t+60 - midprice_t| > 0.001 (=0.1%)
    grp2 = sub_sorted.groupby(["date", "session"])["midprice"]
    mid_60 = grp2.shift(-60)
    big_move = (mid_60 - sub_sorted["midprice"]).abs()
    s["abs_60tick_move_mean"] = float(big_move.mean())
    s["abs_60tick_move_q95"] = float(big_move.quantile(0.95))
    s["pct_60tick_move_gt_0.1pct"] = float((big_move > 0.001).mean())

    # totalbsize / totalasize
    s["totalbsize"] = quantile_stats(sub["totalbsize"])
    s["totalasize"] = quantile_stats(sub["totalasize"])

    stats[f"sym_{sym}"] = s

# Save raw stats
with open(f"{OUT_DIR}/sym_stats_raw.json", "w") as f:
    json.dump(stats, f, indent=2, default=float)
print("[save] sym_stats_raw.json")

# ---------- Build long table for cross-sym z-score comparison ----------
# Flatten to (metric, sym0, sym1, sym2, sym3, sym4)
flat = {}
def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten(v, prefix + k + ".")
        else:
            flat[prefix + k] = v

flat_per_sym = {}
for sym in [0, 1, 2, 3, 4]:
    flat = {}
    flatten(stats[f"sym_{sym}"])
    flat_per_sym[sym] = flat

table_rows = []
all_metrics = sorted(set().union(*[d.keys() for d in flat_per_sym.values()]))
for m in all_metrics:
    row = {"metric": m}
    vals = []
    for sym in [0, 1, 2, 3, 4]:
        v = flat_per_sym[sym].get(m, np.nan)
        row[f"sym{sym}"] = v
        vals.append(v)
    vals = np.array(vals, dtype=float)
    # z-score of sym=2 vs other 4
    others = np.concatenate([vals[:2], vals[3:]])  # idx 0,1,3,4
    if np.all(np.isfinite(others)) and others.std() > 0 and np.isfinite(vals[2]):
        z2 = (vals[2] - others.mean()) / others.std(ddof=0)
    else:
        z2 = np.nan
    row["z_sym2_vs_others"] = float(z2) if np.isfinite(z2) else None
    row["abs_z_sym2"] = abs(row["z_sym2_vs_others"]) if row["z_sym2_vs_others"] is not None else None
    table_rows.append(row)

df_table = pd.DataFrame(table_rows)
df_table.to_csv(f"{OUT_DIR}/sym_stats_table.csv", index=False)
print("[save] sym_stats_table.csv")

# Top-K abs z-score deviations for sym=2
df_sorted = df_table.dropna(subset=["abs_z_sym2"]).sort_values("abs_z_sym2", ascending=False)
print("\n=== Top 20 sym=2 anomaly metrics (by |z| vs other 4 syms) ===")
for _, row in df_sorted.head(20).iterrows():
    print(f"  {row['metric']:50s}  z={row['z_sym2_vs_others']:+.2f}  "
          f"sym2={row['sym2']:.4g}  others_avg={np.mean([row['sym0'],row['sym1'],row['sym3'],row['sym4']]):.4g}")

with open(f"{OUT_DIR}/zscore_top20.json", "w") as f:
    json.dump(df_sorted.head(20).to_dict(orient="records"), f, indent=2, default=float)
print("[save] zscore_top20.json")
print("[done] compute_stats.py")
