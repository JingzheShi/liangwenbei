"""T57 DE 4D pilot — runs DE on a given pred-prefix (single seed) to compare
schemes A_linear / B_sqrt / C_capped vs T53 baseline DE on same single-seed."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
sys.path.insert(0, T53_DIR)

from de_thresh import (  # noqa: E402
    N_FOLDS, PROB_COLS,
    load_fold_avg, fold_arrays_from_dfs, gate_asymmetric, vectorized_pnl,
    de_search,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True,
                    help="prediction file prefix (relative to HERE)")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== T57 DE pilot prefix={args.prefix} seeds={seeds} ===", flush=True)

    # Patch load_fold_avg to use HERE
    import de_thresh as dt
    dt.HERE = HERE  # type: ignore[attr-defined]

    t0 = time.time()
    folds = {k: load_fold_avg(args.prefix, k, seeds) for k in range(N_FOLDS)}
    fas = fold_arrays_from_dfs(folds)
    print(f"  loaded {sum(len(f) for f in folds.values()):,} OOF rows in "
          f"{time.time()-t0:.1f}s", flush=True)

    raw_pf = []
    for fa in fas:
        pred = fa.probs.argmax(axis=1).astype(np.int8)
        raw_pf.append(float(vectorized_pnl(pred, fa.label, fa.mp_t, fa.mp_th).sum()))
    print(f"  raw argmax sum={sum(raw_pf):+.4f}  per_fold={[round(x,3) for x in raw_pf]}",
          flush=True)

    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    de_runs = de_search(fas, bounds, seeds=(0, 1, 2))
    best = de_runs[0]
    print(f"\n  Best DE: sum={best['sum_cum_pnl']:+.4f} at "
          f"(T_up={best['T_up']:.4f}, T_dn={best['T_dn']:.4f}, "
          f"d_up={best['d_up']:.4f}, d_dn={best['d_dn']:.4f})", flush=True)
    print(f"  per_fold: {[round(x,3) for x in best['per_fold_pnl']]}", flush=True)

    out = {
        "prefix": args.prefix,
        "seeds": seeds,
        "raw_argmax_sum": float(sum(raw_pf)),
        "raw_argmax_per_fold": raw_pf,
        "de_best": best,
        "de_runs": de_runs,
    }
    with open(os.path.join(HERE, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
