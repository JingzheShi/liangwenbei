"""T57 DE 4D thresh search on 5-seed-averaged C_capped OOF (h=60)."""
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

import de_thresh as dt
from de_thresh import (  # noqa: E402
    N_FOLDS, PROB_COLS,
    fold_arrays_from_dfs, gate_asymmetric, vectorized_pnl,
    coarse_sweep, de_search, neighborhood_check,
)


def load_fold_avg_local(prefix: str, k: int, seeds):
    base = pd.read_parquet(os.path.join(HERE, f"{prefix}_seed{seeds[0]}_held{k}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(HERE, f"{prefix}_seed{s}_held{k}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch fold={k} seed={s}: {len(df_s)} vs {n}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    out = base.drop(columns=PROB_COLS).copy()
    out["prob_0"] = p_avg[:, 0].astype(np.float32)
    out["prob_1"] = p_avg[:, 1].astype(np.float32)
    out["prob_2"] = p_avg[:, 2].astype(np.float32)
    out["fold"] = np.int8(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="loso_pred_h60_C")
    ap.add_argument("--seeds", default="1,7,13,42,100")
    ap.add_argument("--out", default="de_results_5seed.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"=== T57 DE 5-seed prefix={args.prefix} seeds={seeds} ===", flush=True)

    t0 = time.time()
    folds = {k: load_fold_avg_local(args.prefix, k, seeds) for k in range(N_FOLDS)}
    fas = fold_arrays_from_dfs(folds)
    print(f"  loaded {sum(len(f) for f in folds.values()):,} OOF rows in "
          f"{time.time()-t0:.1f}s", flush=True)

    raw_pf = []
    for fa in fas:
        pred = fa.probs.argmax(axis=1).astype(np.int8)
        raw_pf.append(float(vectorized_pnl(pred, fa.label, fa.mp_t, fa.mp_th).sum()))
    print(f"  raw argmax sum={sum(raw_pf):+.4f}  per_fold={[round(x,3) for x in raw_pf]}",
          flush=True)

    print(f"\n[1/3] Coarse asymmetric sweep ...", flush=True)
    t1 = time.time()
    coarse_df = coarse_sweep(fas)
    print(f"  coarse done in {time.time()-t1:.1f}s. Top 5:", flush=True)
    print(coarse_df.head(5)[["T_up", "T_dn", "d_up", "d_dn", "sum_cum_pnl",
                              "n_pos_folds"]].to_string(index=False))

    print(f"\n[2/3] DE 4D differential evolution ...", flush=True)
    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    de_runs = de_search(fas, bounds, seeds=(0, 1, 2, 7, 42))
    best = de_runs[0]
    print(f"\n  Best DE: sum={best['sum_cum_pnl']:+.4f} at "
          f"(T_up={best['T_up']:.4f}, T_dn={best['T_dn']:.4f}, "
          f"d_up={best['d_up']:.4f}, d_dn={best['d_dn']:.4f})", flush=True)
    print(f"  per_fold: {[round(x,3) for x in best['per_fold_pnl']]} "
          f"std={best['std_cum_pnl']:.4f} pos={best['n_pos_folds']}/5", flush=True)

    print(f"\n[3/3] Neighborhood ...", flush=True)
    nb = neighborhood_check(fas, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
    print(nb.head(5).to_string(index=False))
    n_robust = int((nb["sum_cum_pnl"] >= best["sum_cum_pnl"] - 0.5).sum())
    print(f"  configs within 0.5 of optimum: {n_robust} of {len(nb)}", flush=True)

    out = {
        "prefix": args.prefix,
        "seeds": seeds,
        "raw_argmax_sum": float(sum(raw_pf)),
        "raw_argmax_per_fold": raw_pf,
        "coarse_top10": coarse_df.head(10).to_dict(orient="records"),
        "coarse_best_sum": float(coarse_df.iloc[0]["sum_cum_pnl"]),
        "de_runs": de_runs,
        "de_best": best,
        "neighborhood_top10": nb.head(10).to_dict(orient="records"),
        "neighborhood_robust_count": n_robust,
    }
    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
