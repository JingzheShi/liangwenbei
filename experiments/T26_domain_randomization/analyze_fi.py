"""T26: feature importance comparison across DR variants.

Reads feat_importance_T26.json (written by train_loso.py) and computes per
variant: mean fi_gain across 5 folds, normalized; then ranks; reports top-20
per variant and per-feature rank shift vs baseline.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fi-path", default=os.path.join(HERE, "feat_importance_T26.json"))
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    with open(args.fi_path) as f:
        data = json.load(f)

    # data: dict[ "{variant}_held{k}" -> {fi_gain, fi_split, feat_names} ]
    feat_names = None
    by_variant: dict[str, list[np.ndarray]] = {}
    for key, payload in data.items():
        variant, held = key.rsplit("_held", 1)
        gain = np.array(payload["fi_gain"], dtype=np.float64)
        if feat_names is None:
            feat_names = payload["feat_names"]
        by_variant.setdefault(variant, []).append(gain)

    if feat_names is None:
        raise SystemExit("no feature importances loaded")

    summary = {}
    for v, gains in by_variant.items():
        stack = np.stack(gains, axis=0)  # (5, F)
        # normalize each fold to sum 1, then average
        stack_n = stack / np.maximum(stack.sum(axis=1, keepdims=True), 1e-9)
        mean = stack_n.mean(axis=0)
        summary[v] = mean

    # rank per variant (descending importance)
    ranks = {}
    for v, mean in summary.items():
        order = np.argsort(-mean)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        ranks[v] = rank

    # Compare each non-baseline vs baseline
    if "baseline" not in ranks:
        print("WARNING: baseline not present; skipping rank-shift")
    else:
        base_rank = ranks["baseline"]
        for v in [x for x in ranks if x != "baseline"]:
            r = ranks[v]
            shift = np.abs(r - base_rank)
            n_changed = int((shift > 10).sum())
            print(
                f"  {v:10s}: features with |rank shift| > 10 vs baseline = {n_changed}/{len(shift)} "
                f"(median={int(np.median(shift))}, p95={int(np.percentile(shift, 95))})",
                flush=True,
            )

    # Top-20 per variant
    rows = []
    for v, mean in summary.items():
        order = np.argsort(-mean)
        for rk, idx in enumerate(order[:args.top]):
            rows.append({
                "variant": v,
                "rank": rk,
                "feat": feat_names[idx],
                "mean_norm_gain": float(mean[idx]),
            })
    df = pd.DataFrame(rows)
    out_csv = os.path.join(HERE, "feat_importance_top20.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nsaved top-{args.top} per variant -> {out_csv}", flush=True)

    # Compact pivot
    pivot = df.pivot(index="rank", columns="variant", values="feat")
    print("\nTop features by rank (variant columns):")
    print(pivot.to_string(max_colwidth=30))


if __name__ == "__main__":
    main()
