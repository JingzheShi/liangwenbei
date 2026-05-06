"""T35: Feature importance analysis on Scheme K LOSO models.

Loads the 5 LOSO booster files for a given (variant, seed, horizon),
computes mean gain importance over folds, prints top-30, and counts how
many of them are ReVol / SG features.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--syms", default="0,1,2,3,4")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    syms = [int(x) for x in args.syms.split(",")]

    fi_total = None
    feat_names = None
    for k in syms:
        p = os.path.join(
            HERE,
            f"loso_model_h{args.horizon}_{args.variant}_seed{args.seed}_held{k}.txt",
        )
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
        b = lgb.Booster(model_file=p)
        if feat_names is None:
            feat_names = b.feature_name()
        gain = b.feature_importance(importance_type="gain")
        fi_total = gain.astype(np.float64) if fi_total is None else fi_total + gain.astype(np.float64)

    fi_mean = fi_total / float(len(syms))
    df = pd.DataFrame({"feature": feat_names, "mean_gain": fi_mean})
    df = df.sort_values("mean_gain", ascending=False).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)
    df["is_revol"] = df["feature"].str.startswith("revol_")
    df["is_sg"] = df["feature"].str.startswith("sg_")
    df["is_extra"] = df["is_revol"] | df["is_sg"]

    top = df.head(args.top)
    print(f"\n=== Top {args.top} feature importance (mean gain over {len(syms)} LOSO folds) ===")
    print(f"variant={args.variant} seed={args.seed} h={args.horizon}\n")
    for _, r in top.iterrows():
        tag = "REVOL" if r["is_revol"] else ("SG" if r["is_sg"] else "BASE")
        print(f"  {r['rank']:>3}  [{tag:>5}]  {r['feature']:<40}  gain={r['mean_gain']:.2f}")

    n_rev_top = int(top["is_revol"].sum())
    n_sg_top = int(top["is_sg"].sum())
    print(f"\n=== In top-{args.top}: {n_rev_top} ReVol, {n_sg_top} SG, {args.top - n_rev_top - n_sg_top} base ===")

    # Aggregate gain share
    total_gain = float(df["mean_gain"].sum())
    g_rev = float(df.loc[df["is_revol"], "mean_gain"].sum())
    g_sg = float(df.loc[df["is_sg"], "mean_gain"].sum())
    g_base = float(df.loc[~df["is_extra"], "mean_gain"].sum())
    print(f"Total gain shares (over all {len(df)} feats): "
          f"ReVol {g_rev/total_gain*100:.1f}%, SG {g_sg/total_gain*100:.1f}%, "
          f"base {g_base/total_gain*100:.1f}%")

    out_path = os.path.join(
        HERE, f"feat_importance_h{args.horizon}_{args.variant}_seed{args.seed}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"saved -> {out_path}")

    json_out = os.path.join(
        HERE, f"feat_importance_h{args.horizon}_{args.variant}_seed{args.seed}.json"
    )
    with open(json_out, "w") as f:
        json.dump({
            "variant": args.variant, "seed": args.seed, "horizon": args.horizon,
            "n_features": len(df),
            "top": top[["rank", "feature", "mean_gain", "is_revol", "is_sg"]].to_dict(orient="records"),
            "n_revol_in_top": n_rev_top,
            "n_sg_in_top": n_sg_top,
            "gain_share": {"revol": g_rev/total_gain, "sg": g_sg/total_gain, "base": g_base/total_gain},
        }, f, indent=2)


if __name__ == "__main__":
    main()
