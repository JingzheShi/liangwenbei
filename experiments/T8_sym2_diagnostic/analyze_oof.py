"""
T8: T2 Scheme B sym=2 LOSO OOF prediction error analysis.

- Distribution of max_prob on sym=2
- Where (date/session) is the model worst on sym=2?
- "False positive direction" vs "correct direction" — feature differences
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

WORKDIR = "/root/projects/liangwenbei_workdir"
OUT = f"{WORKDIR}/experiments/T8_sym2_diagnostic"
os.makedirs(OUT, exist_ok=True)

# ---------- Load OOF predictions ----------
print("[oof] loading T4 LOSO held=2 OOF predictions ...")
pred = pd.read_parquet(f"{WORKDIR}/experiments/T4_loso_validate/loso_pred_schemeB_held2.parquet")
print(f"[oof] pred shape: {pred.shape}")
print(f"[oof] dates: {pred['date'].min()}..{pred['date'].max()}, sessions: {pred['session'].unique()}")
print(f"[oof] pred_label_60 dist:\n{pred['pred_label_60'].value_counts(normalize=True).to_dict()}")
print(f"[oof] true_label_60 dist:\n{pred['true_label_60'].value_counts(normalize=True).to_dict()}")

pred["max_prob"] = pred[["prob_0", "prob_1", "prob_2"]].max(axis=1)
pred["correct"] = (pred["pred_label_60"] == pred["true_label_60"]).astype(int)

# Aggregate accuracy per (date, session)
acc_by_date = pred.groupby(["date", "session"])["correct"].agg(["mean", "count"]).reset_index()
acc_by_date.columns = ["date", "session", "acc", "n"]
print("\n[oof] accuracy by date/session (sorted by acc):")
print(acc_by_date.sort_values("acc").head(10).to_string(index=False))
print(acc_by_date.sort_values("acc", ascending=False).head(10).to_string(index=False))

# Max-prob distribution
print(f"\n[oof] max_prob: mean={pred.max_prob.mean():.3f}, median={pred.max_prob.median():.3f}, q05={pred.max_prob.quantile(0.05):.3f}, q95={pred.max_prob.quantile(0.95):.3f}")

# Confusion analysis (pred direction vs truth)
print("\n[oof] confusion (rows=true, cols=pred):")
conf = pd.crosstab(pred["true_label_60"], pred["pred_label_60"], normalize="index")
print(conf.round(3))

# ---------- Compare features: TP-direction vs FP-direction ----------
# Define "directional pred" = argmax in {0, 2} (skip 1 = flat)
# True positive: pred == true (and both directional 0 or 2)
# False positive: pred is direction (0 or 2) but true is opposite direction
# Need to load matching features for sym=2 to compare

# Load sym=2 raw features (only test dates 96..119)
print("\n[features] loading sym=2 dates 96..119 raw data ...")
keep = ["date", "sym", "time", "midprice", "amount_delta", "volume_delta",
        "imbalance", "spread1", "lb_intst", "la_intst", "mb_intst", "ma_intst",
        "cb_intst", "ca_intst", "totalbsize", "totalasize"]
dfs = []
for d in range(96, 120):
    for sess in ["am", "pm"]:
        f = f"{WORKDIR}/data/snapshot_sym2_date{d}_{sess}.parquet"
        if os.path.exists(f):
            df = pd.read_parquet(f, columns=keep)
            df["session"] = sess
            df["t"] = np.arange(len(df))
            dfs.append(df)

raw = pd.concat(dfs, ignore_index=True)
raw["sym"] = raw["sym"].astype(int)
raw["date"] = raw["date"].astype(int)
pred["date"] = pred["date"].astype(int)
pred["sym"] = pred["sym"].astype(int)
print(f"[features] raw sym=2 test rows: {raw.shape}")

# Merge OOF predictions with features on (sym, date, session, t)
merged = pred.merge(raw, on=["sym", "date", "session", "t"], how="inner",
                    suffixes=("_pred", ""))
print(f"[features] merged: {merged.shape}")

# Categorize predictions
merged["pred_dir"] = merged["pred_label_60"].map({0: "down", 1: "flat", 2: "up"})
merged["true_dir"] = merged["true_label_60"].map({0: "down", 1: "flat", 2: "up"})

# True positive direction = correct direction prediction
merged["category"] = "other"
merged.loc[(merged.pred_label_60 == 0) & (merged.true_label_60 == 0), "category"] = "TP_down"
merged.loc[(merged.pred_label_60 == 2) & (merged.true_label_60 == 2), "category"] = "TP_up"
merged.loc[(merged.pred_label_60 == 0) & (merged.true_label_60 == 2), "category"] = "FP_pred_down_was_up"
merged.loc[(merged.pred_label_60 == 2) & (merged.true_label_60 == 0), "category"] = "FP_pred_up_was_down"
merged.loc[(merged.pred_label_60 == 1) & (merged.true_label_60 == 1), "category"] = "TP_flat"
merged.loc[(merged.pred_label_60 == 0) & (merged.true_label_60 == 1), "category"] = "FP_pred_down_was_flat"
merged.loc[(merged.pred_label_60 == 2) & (merged.true_label_60 == 1), "category"] = "FP_pred_up_was_flat"

cat_counts = merged["category"].value_counts()
print(f"\n[features] category counts:\n{cat_counts}")

# Compare features across categories
comp_cols = ["midprice", "amount_delta", "volume_delta", "imbalance", "spread1",
             "lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst",
             "totalbsize", "totalasize"]
group_stats = merged.groupby("category")[comp_cols].agg(["mean", "std"]).round(6)
group_stats.to_csv(f"{OUT}/oof_feature_by_category.csv")
print(f"[save] oof_feature_by_category.csv")

# Compute "FP vs TP" deltas (separately for up and down)
def compare(cat_a, cat_b):
    mu_a = merged[merged.category == cat_a][comp_cols].mean()
    mu_b = merged[merged.category == cat_b][comp_cols].mean()
    sd_b = merged[merged.category == cat_b][comp_cols].std()
    delta = (mu_a - mu_b) / sd_b
    return delta

# FP up (pred up but was down) vs TP up
delta_fp_up = compare("FP_pred_up_was_down", "TP_up")
delta_fp_down = compare("FP_pred_down_was_up", "TP_down")
print("\n[features] FP_up vs TP_up (z-units of TP_up sd):")
print(delta_fp_up.round(2))
print("\n[features] FP_down vs TP_down (z-units of TP_down sd):")
print(delta_fp_down.round(2))

# Save the merged-summary
summary = {
    "n_oof": len(pred),
    "max_prob_stats": {
        "mean": float(pred.max_prob.mean()),
        "median": float(pred.max_prob.median()),
        "q05": float(pred.max_prob.quantile(0.05)),
        "q95": float(pred.max_prob.quantile(0.95)),
    },
    "overall_acc": float(pred.correct.mean()),
    "true_dist": pred.true_label_60.value_counts(normalize=True).to_dict(),
    "pred_dist": pred.pred_label_60.value_counts(normalize=True).to_dict(),
    "category_counts": cat_counts.to_dict(),
    "worst_5_dates": acc_by_date.sort_values("acc").head(5).to_dict(orient="records"),
    "best_5_dates": acc_by_date.sort_values("acc", ascending=False).head(5).to_dict(orient="records"),
    "FP_up_vs_TP_up_z": {k: float(v) for k, v in delta_fp_up.to_dict().items()},
    "FP_down_vs_TP_down_z": {k: float(v) for k, v in delta_fp_down.to_dict().items()},
}
with open(f"{OUT}/oof_analysis.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"[save] oof_analysis.json")
print("[done]")
