"""Analyze T33 feature importance — compare to T26 baseline (h=60).

Reports:
- Top 30 features by mean fi_gain across LOSO folds (per variant/seed)
- How many of top 30 are NEW (long-window) vs base 223-d
- Feature category importance breakdown (A/B/C/D/E/F)
- Bottom 30 (importance ~0) — candidates for pruning
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def category_of(name: str) -> str:
    if name.startswith(("A_", "B_", "C_", "D_", "E_", "F_")):
        return name[0]
    return "base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    fi_path = os.path.join(HERE, "feat_importance_T33.json")
    with open(fi_path) as f:
        fi = json.load(f)

    keys = [k for k in fi if k.startswith(f"{args.variant}_seed{args.seed}_held")]
    if not keys:
        print(f"No keys found for variant={args.variant} seed={args.seed}")
        print("Available:", sorted(set(k.rsplit("_held", 1)[0] for k in fi)))
        return

    feat_names: List[str] = fi[keys[0]]["feat_names"]
    gains = np.array([fi[k]["fi_gain"] for k in keys])
    splits = np.array([fi[k]["fi_split"] for k in keys])
    print(f"Variant={args.variant} seed={args.seed}: {len(keys)} folds, {len(feat_names)} features",
          flush=True)
    mean_gain = gains.mean(axis=0)
    mean_split = splits.mean(axis=0)
    order = np.argsort(-mean_gain)

    print(f"\nTop {args.top} features by mean fi_gain:")
    for i, idx in enumerate(order[:args.top]):
        cat = category_of(feat_names[idx])
        print(f"  {i+1:2d}. [{idx:3d}] ({cat}) {feat_names[idx]:50s}  gain={mean_gain[idx]:>10.0f}  split={mean_split[idx]:>5.0f}",
              flush=True)

    # Category breakdown
    print(f"\nCategory breakdown of mean fi_gain (sum / count / mean / fraction):")
    cats = {}
    for n, g in zip(feat_names, mean_gain):
        c = category_of(n)
        cats.setdefault(c, []).append(g)
    total_gain = mean_gain.sum()
    for c in ("base", "A", "B", "C", "D", "E", "F"):
        if c in cats:
            arr = np.array(cats[c])
            print(f"  {c:5s}: count={len(arr):3d}  sum={arr.sum():>10.0f} ({100*arr.sum()/total_gain:5.1f}%)  "
                  f"mean={arr.mean():>9.0f}  max={arr.max():>9.0f}",
                  flush=True)

    # How many of top N are long-window (non-base)?
    top_idx = order[:args.top]
    n_long = sum(1 for i in top_idx if category_of(feat_names[i]) != "base")
    print(f"\n{n_long}/{args.top} top features are long-window (A/B/C/D/E/F)")

    # Long-window features that made top 50
    print(f"\nLong-window features in top 50:")
    for i, idx in enumerate(order[:50]):
        cat = category_of(feat_names[idx])
        if cat != "base":
            print(f"  rank {i+1:2d}: ({cat}) {feat_names[idx]:50s}  gain={mean_gain[idx]:>10.0f}",
                  flush=True)

    # Bottom 30 (low importance — candidates for prune)
    print(f"\nBottom 30 features (lowest gain):")
    for idx in order[-30:]:
        cat = category_of(feat_names[idx])
        print(f"  [{idx:3d}] ({cat}) {feat_names[idx]:50s}  gain={mean_gain[idx]:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
