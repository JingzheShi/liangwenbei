"""Aggregate feature importance over all 5 saved fold boosters (seed=42).

Loads loso_model_h10_seed42_held{0..4}.txt and computes mean gain/split,
identifies (a) top-30 by gain, (b) R20 new factors that made top 30,
(c) zero-importance dead features (gain==0 or split==0), useful for next-round pruning.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb  # noqa: E402

H = 10
SEED = 42
SYMS = (0, 1, 2, 3, 4)


def main():
    feat_names_path = os.path.join(HERE, "cache", "schemeH_no_time_feat_names.txt")
    with open(feat_names_path) as f:
        feat_names = [l.strip() for l in f]
    print(f"feat dim = {len(feat_names)}", flush=True)

    gains = []
    splits = []
    for k in SYMS:
        mp = os.path.join(HERE, f"loso_model_h{H}_seed{SEED}_held{k}.txt")
        bst = lgb.Booster(model_file=mp)
        gains.append(np.array(bst.feature_importance(importance_type="gain"), dtype=np.float64))
        splits.append(np.array(bst.feature_importance(importance_type="split"), dtype=np.float64))
    gain_arr = np.stack(gains, axis=0)
    split_arr = np.stack(splits, axis=0)

    # Try to read original 226-d feature list (Scheme C) so we can label which are NEW R20
    schemeC_feat_path = os.path.join(ROOT, "experiments", "T5b_features_multihorizon", "cache",
                                     "schemeC_223d_feat_names.txt")
    schemeC = set()
    if os.path.exists(schemeC_feat_path):
        with open(schemeC_feat_path) as f:
            schemeC = {l.strip() for l in f}
        print(f"loaded {len(schemeC)} Scheme C feat names from {schemeC_feat_path}")
    else:
        print(f"WARN: no scheme C feat name file found, can't label R20-NEW", flush=True)

    df = pd.DataFrame({
        "feat_name": feat_names,
        "gain_mean": gain_arr.mean(axis=0),
        "gain_std": gain_arr.std(axis=0),
        "split_mean": split_arr.mean(axis=0),
        "is_r20_new": [n not in schemeC for n in feat_names] if schemeC else [False] * len(feat_names),
    }).sort_values("gain_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    out_csv = os.path.join(HERE, "feat_importance_5fold.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    print("\n=== TOP 30 BY GAIN (mean over 5 folds) ===")
    print(f"{'rank':>4} {'feat':40s} {'gain':>10s} {'split':>8s} {'r20_new':>8s}")
    for _, r in df.head(30).iterrows():
        flag = "**NEW**" if r["is_r20_new"] else ""
        print(f"{int(r['rank']):>4} {r['feat_name']:40s} {r['gain_mean']:>10.0f} "
              f"{r['split_mean']:>8.0f} {flag:>8s}")

    n_r20 = int(df["is_r20_new"].sum())
    n_top30 = int(df.head(30)["is_r20_new"].sum())
    n_top50 = int(df.head(50)["is_r20_new"].sum())
    n_dead = int((df["gain_mean"] < 1).sum())
    n_dead_r20 = int(((df["gain_mean"] < 1) & df["is_r20_new"]).sum())

    print(f"\n=== R20 NEW factor ranks ===")
    print(f"total R20 new          : {n_r20}")
    print(f"R20 in top 30          : {n_top30}/30")
    print(f"R20 in top 50          : {n_top50}/50")
    print(f"feats with gain<1 (dead): {n_dead}")
    print(f"  of which R20 new     : {n_dead_r20}/{n_r20}")

    print(f"\n=== R20 NEW factors with rank > 100 (candidates to prune) ===")
    weak_r20 = df[(df["is_r20_new"]) & (df["rank"] > 100)].sort_values("rank")
    for _, r in weak_r20.head(40).iterrows():
        print(f"  rank{int(r['rank']):>3} {r['feat_name']:40s} gain={r['gain_mean']:.0f}")
    print(f"\n  total R20 NEW with rank>100: {len(weak_r20)}")

    summary = {
        "n_features": len(feat_names),
        "n_r20_new": n_r20,
        "r20_in_top30": n_top30,
        "r20_in_top50": n_top50,
        "n_dead_features_gain_lt_1": n_dead,
        "n_dead_r20": n_dead_r20,
        "top30": df.head(30)[["rank", "feat_name", "gain_mean", "split_mean", "is_r20_new"]].to_dict(orient="records"),
        "weak_r20_rank_gt100": weak_r20[["rank", "feat_name", "gain_mean"]].to_dict(orient="records"),
    }
    out_json = os.path.join(HERE, "feat_importance_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
