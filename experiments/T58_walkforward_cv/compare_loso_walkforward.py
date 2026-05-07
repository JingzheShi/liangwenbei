"""Compare LOSO vs walk-forward CV results.

LOSO baselines (R34 stage3 schemeN, aug_a, h60, GPU LightGBM):
  - 1-seed (seed42) raw argmax sum cum_pnl  = +2.504  (5 folds × 24d × 1sym = 120 sym-days)
  - 5-seed prob-ensemble raw argmax         = +5.769  (120 sym-days, ensembled)
  - 5-seed prob-ensemble + DE thresh         = +16.384 (120 sym-days, DE-thresholded)
                                               (T_up=0.45, T_dn=0.40)

For walk-forward, results are summed cum_pnl across folds, normalized to 120 sym-days
for direct comparison: cum_pnl_per_120sd = cum_pnl_sum * (120 / total_sym_days).

Usage: python3 compare_loso_walkforward.py results_pilot_seed42.json
"""
from __future__ import annotations

import json
import sys


LOSO_BASELINES = {
    "1-seed_raw_argmax_seed42":      2.504,
    "5-seed_raw_argmax_prob_ensemble": 5.769,
    "5-seed_DE_thresh":              16.384,
}


def main():
    if len(sys.argv) < 2:
        print("usage: compare_loso_walkforward.py results.json [...]")
        sys.exit(1)

    print("=" * 78)
    print("LOSO baselines (cum_pnl on 120 sym-days = 5 syms × 24 days):")
    for name, val in LOSO_BASELINES.items():
        print(f"  {name:40s}: {val:+.4f}")
    print("=" * 78)

    for path in sys.argv[1:]:
        print(f"\n## Loaded: {path}")
        with open(path) as f:
            d = json.load(f)
        for scheme, blob in d["schemes"].items():
            print(f"\n### Scheme {scheme}  (n_folds={len(blob['per_seed'][list(blob['per_seed'].keys())[0]])})")
            for seed_str, agg in blob["aggregate_by_seed"].items():
                print(f"  seed={seed_str}: cum_pnl_sum={agg['cum_pnl_sum']:+.4f}  "
                      f"normalized_120sd={agg['cum_pnl_per_120_sym_days']:+.4f}  "
                      f"n_pos_folds={agg['n_pos_folds']}/{agg['n_folds']}")
            ms = blob["multi_seed_summary"]
            print(f"  multi-seed mean normalized: {ms['mean_normalized']:+.4f} "
                  f"(std={ms['std_normalized']:.4f}, n_seeds={ms['n_seeds']})")
        # comparison table
        print("\n## Comparison (normalized to 120 sym-days = LOSO scale)")
        print(f"  {'Scheme':<25} {'mean_norm':<12} {'vs LOSO 1-seed (+2.504)':<25} {'vs LOSO 5-seed raw (+5.77)':<25}")
        for scheme, blob in d["schemes"].items():
            mean_norm = blob["multi_seed_summary"]["mean_normalized"]
            d1 = mean_norm - LOSO_BASELINES["1-seed_raw_argmax_seed42"]
            d5 = mean_norm - LOSO_BASELINES["5-seed_raw_argmax_prob_ensemble"]
            print(f"  {scheme:<25} {mean_norm:+.4f}      delta={d1:+.4f}              delta={d5:+.4f}")


if __name__ == "__main__":
    main()
