"""T96: cross-correlation analysis vs T75 (LGB), T81 (NN), T87 (SPO+/DFL).

Computes Pearson correlation of per-row averaged predictions across
the 5 seeds, on the test set.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SEEDS = [1, 7, 13, 42, 100]

FEE = 0.0001
SYMS = (0, 1, 2, 3, 4)


def load_avg(dirname, prefix, suffix=""):
    base = pd.read_parquet(os.path.join(dirname, f"{prefix}_seed{SEEDS[0]}{suffix}.parquet"))
    n = len(base)
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for s in SEEDS[1:]:
        df_s = pd.read_parquet(os.path.join(dirname, f"{prefix}_seed{s}{suffix}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"len mismatch {prefix}_seed{s}: {len(df_s)} vs {n}")
        p_sum += df_s["pred_dmid_norm"].to_numpy(np.float64)
    return base, p_sum / float(len(SEEDS))


def make_de_search(pred, sym_arr, mp_t, mp_th):
    # Pre-compute per-sym constants
    diff = mp_th - mp_t
    denom = mp_t + 1.0
    fee_amount = FEE * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    # Pre-split per-sym
    masks = [sym_arr == k for k in SYMS]

    def f(x):
        tu, td = x
        # Vectorized action
        action_side = np.zeros(len(pred), dtype=np.float64)
        action_side[pred > tu] = 1.0
        action_side[pred < -td] = -1.0
        # PnL = (side * diff - fee_amount * |side|) / denom
        pnl = (action_side * diff - fee_amount * np.abs(action_side)) / denom
        s = 0.0
        for m in masks:
            s += pnl[m].sum()
        return -float(s)

    return f


def search_de(pred, sym_arr, mp_t, mp_th):
    f = make_de_search(pred, sym_arr, mp_t, mp_th)
    bounds = [(0.0, 0.004), (0.0, 0.004)]
    runs = []
    for sd in [0, 1, 2, 7, 42]:
        r = differential_evolution(
            f, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, init="sobol",
        )
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def main():
    sources = []
    base_t75, avg_t75 = load_avg(os.path.join(ROOT, "experiments", "T75_regression_dmid"), "pred_T75")
    sources.append(("T75_lgb", avg_t75))

    t81_dir = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")
    base_t81, avg_t81 = load_avg(t81_dir, "pred_T81")
    sources.append(("T81_nn", avg_t81))

    t87_dir = os.path.join(ROOT, "experiments", "T87_spo_dfl")
    if os.path.isdir(t87_dir):
        try:
            base_t87, avg_t87 = load_avg(t87_dir, "pred_T87", "_main")
            sources.append(("T87_spo", avg_t87))
        except Exception as e:
            print(f"  T87 load failed: {e}")

    base_t96, avg_t96 = load_avg(HERE, "pred_T96")
    sources.append(("T96_lgb_schemeQ", avg_t96))

    n = len(base_t96)
    print(f"loaded {len(sources)} prediction sources, n={n:,}", flush=True)
    names = [s[0] for s in sources]
    preds = [s[1] for s in sources]
    K = len(sources)
    corr_mat = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            corr_mat[i, j] = float(np.corrcoef(preds[i], preds[j])[0, 1])
    print("\n=== Cross-correlation matrix ===", flush=True)
    print("           " + "  ".join(f"{n_:>16s}" for n_ in names), flush=True)
    for i, ni in enumerate(names):
        print(f"{ni:>16s}  " + "  ".join(f"{corr_mat[i,j]:+16.4f}" for j in range(K)),
              flush=True)

    mp_t = base_t96["midprice_t"].to_numpy(np.float64)
    mp_th = base_t96["midprice_th"].to_numpy(np.float64)
    sym_arr = base_t96["sym"].to_numpy(np.int8)

    print("\n=== Ensemble combinations (DE asym thresh) ===", flush=True)
    name_to_pred = {n_: p for n_, p in zip(names, preds)}
    combos = [
        ("T75 only", [("T75_lgb", 1.0)]),
        ("T81 only", [("T81_nn", 1.0)]),
        ("T96 only", [("T96_lgb_schemeQ", 1.0)]),
        ("T75+T81 (50/50)", [("T75_lgb", 0.5), ("T81_nn", 0.5)]),
        ("T75+T96 (50/50)", [("T75_lgb", 0.5), ("T96_lgb_schemeQ", 0.5)]),
        ("T81+T96 (50/50)", [("T81_nn", 0.5), ("T96_lgb_schemeQ", 0.5)]),
        ("T75+T81+T96 (1/3 each)", [("T75_lgb", 1/3), ("T81_nn", 1/3), ("T96_lgb_schemeQ", 1/3)]),
        ("T75+T81+T96 (40/40/20)", [("T75_lgb", 0.4), ("T81_nn", 0.4), ("T96_lgb_schemeQ", 0.2)]),
        ("T75+T81+T96 (45/45/10)", [("T75_lgb", 0.45), ("T81_nn", 0.45), ("T96_lgb_schemeQ", 0.10)]),
        ("T75+T81+T87+T96 (1/4 each)", [("T75_lgb", 0.25), ("T81_nn", 0.25), ("T87_spo", 0.25), ("T96_lgb_schemeQ", 0.25)]),
    ]
    if "T87_spo" not in name_to_pred:
        combos = [c for c in combos if "T87" not in c[0]]

    results = []
    for cname, weights in combos:
        ens = np.zeros(n, dtype=np.float64)
        wsum = 0.0
        for n_, w in weights:
            if n_ in name_to_pred:
                ens += w * name_to_pred[n_]
                wsum += w
        ens /= wsum
        v, tu, td = search_de(ens, sym_arr, mp_t, mp_th)
        print(f"  {cname:<32s}  pnl={v:+.4f}  thr_up={tu:.6f}  thr_dn={td:.6f}", flush=True)
        results.append({"combo": cname, "weights": weights, "loso_pnl": v,
                        "thr_up": tu, "thr_dn": td})

    out = {
        "names": names,
        "corr_matrix": corr_mat.tolist(),
        "ensembles": results,
    }
    with open(os.path.join(HERE, "cross_corr_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote cross_corr_results.json", flush=True)


if __name__ == "__main__":
    main()
