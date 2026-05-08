"""T118 follow-up: test all rules under SHUFFLED row order (production scenario).

In production the platform shuffles rows, so any batch-dependent rule must be tested
on shuffled data to confirm transmission. Rules that aggregate (TBT global top-K, STK
fold-OOF) are order-invariant by construction; NSF is per-row so also invariant.
BAT applies within-batch quantile, so its result depends on row order.

We re-run all rules on a permutation of the test set and compare LOSO/per_sym to
the sorted-order results. If BAT collapses, then the +47.13 was a sym-sorted artifact.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from eval_4in1 import (
    build_4way_pred, eval_actions, baseline_de, rule_nsf, rule_tbt, rule_stk, rule_bat,
    THR_UP_BASE, THR_DN_BASE, ROOT, FEE,
)

HERE = Path("/root/projects/liangwenbei_workdir/experiments/T118_decision_4in1")


def main():
    p_comb, meta, sym = build_4way_pred()
    n = len(p_comb)

    # load spread for NSF
    schemeP = np.load(ROOT / "experiments/T68_stage5_features/cache/schemeP_test.npz")
    feat_names_path = ROOT / "experiments/T68_stage5_features/cache/schemeP_feat_names.txt"
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f]
    i_ask1 = all_names.index("ask1")
    i_bid1 = all_names.index("bid1")
    spread_t = (schemeP["X"][:, i_ask1] - schemeP["X"][:, i_bid1]).astype(np.float64)

    mp_t = meta["midprice_t"].to_numpy(dtype=np.float64)
    mp_th = meta["midprice_th"].to_numpy(dtype=np.float64)

    # baseline rate (from sorted run)
    a_base = baseline_de(p_comb)
    rate_base = float((a_base != 1).sum() / len(a_base))
    print(f"baseline_active_rate = {rate_base:.4f}", flush=True)

    rng = np.random.default_rng(2026)
    perm = rng.permutation(n)
    inv = np.argsort(perm)

    p_perm = p_comb[perm]
    sp_perm = spread_t[perm]

    results = {}

    # baseline (order-invariant)
    print("\n--- BASELINE under shuffle ---", flush=True)
    a = baseline_de(p_perm)[inv]
    r = eval_actions(a, meta, sym, "baseline_de_shuffled")
    print(f"  loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    results["baseline_de_shuffled"] = r

    # NSF best (fe=2.88e-4, g=0)  -- per-row, order-invariant
    print("\n--- NSF (fe=2.88e-4, g=0) under shuffle ---", flush=True)
    a = rule_nsf(p_perm, sp_perm, THR_UP_BASE, 0.0)[inv]
    r = eval_actions(a, meta, sym, "nsf_best_shuffled")
    print(f"  loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    results["nsf_best_shuffled"] = r

    # TBT (global top-K, order-invariant)
    print("\n--- TBT under shuffle ---", flush=True)
    a = rule_tbt(p_perm, rate_base)[inv]
    r = eval_actions(a, meta, sym, "tbt_shuffled")
    print(f"  loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    results["tbt_shuffled"] = r

    # BAT — order DEPENDENT, real test
    print("\n--- BAT under shuffle (KEY TEST) ---", flush=True)
    bat_shuf = []
    for batch in [1024, 4096, 16384, 100000]:
        a = rule_bat(p_perm, rate_base, batch=batch)[inv]
        r = eval_actions(a, meta, sym, f"bat_b{batch}_shuffled")
        r["batch"] = batch
        bat_shuf.append(r)
        print(f"  batch={batch}: loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)
    results["bat_shuffled"] = bat_shuf

    # also test small batches to assess feasible production sizes
    print("\n--- BAT (small batches) under shuffle ---", flush=True)
    for batch in [128, 256, 512]:
        a = rule_bat(p_perm, rate_base, batch=batch)[inv]
        r = eval_actions(a, meta, sym, f"bat_b{batch}_shuffled")
        r["batch"] = batch
        bat_shuf.append(r)
        print(f"  batch={batch}: loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  rate={r['active_rate']:.4f}", flush=True)

    # Multi-seed BAT robustness: test across permutations
    print("\n--- BAT batch=1024 across multiple shuffles ---", flush=True)
    bat_multi = []
    for seed in [42, 123, 999, 2026, 31337]:
        rng2 = np.random.default_rng(seed)
        p2 = rng2.permutation(n)
        i2 = np.argsort(p2)
        a = rule_bat(p_comb[p2], rate_base, batch=1024)[i2]
        r = eval_actions(a, meta, sym, f"bat_b1024_perm{seed}")
        r["perm_seed"] = seed
        bat_multi.append(r)
        print(f"  perm_seed={seed}: loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}", flush=True)
    results["bat_b1024_across_perms"] = bat_multi

    out = HERE / "results_shuffle.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote -> {out}", flush=True)


if __name__ == "__main__":
    main()
