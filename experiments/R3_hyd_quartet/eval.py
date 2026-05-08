"""R3 LOSO-equiv evaluation:
- mean preds across 3 seeds per arm
- DE 2D thresh (T_up, T_dn) shared across all syms (per-arm)
- compute per-sym PnL, sum = LOSO-equivalent
- delta_loso = trick - baseline ; delta_per_sym_min = min(per-sym deltas)
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def gate_asym(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def avg_preds(parquet_paths):
    base = pd.read_parquet(parquet_paths[0])
    p = np.zeros(len(base), dtype=np.float64)
    for pp in parquet_paths:
        df = pd.read_parquet(pp)
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch in {pp}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(parquet_paths)


def fit_de_2d(p_arr, sym_arr, mp_t, mp_th, de_seeds=(0, 42, 1)):
    folds = [(p_arr[sym_arr == k], mp_t[sym_arr == k], mp_th[sym_arr == k])
             for k in SYMS]
    def obj(x):
        s = 0.0
        for fp, fmt, fmth in folds:
            a = gate_asym(fp, x[0], x[1])
            s += vectorized_pnl(a, fmt, fmth).sum()
        return -float(s)
    best_x, best_y = None, np.inf
    for ds in de_seeds:
        r = differential_evolution(obj,
                                   bounds=[(0, 0.005), (0, 0.005)],
                                   seed=ds, maxiter=80, popsize=24,
                                   tol=1e-7, polish=True, workers=1)
        if r.fun < best_y:
            best_y, best_x = float(r.fun), tuple(float(v) for v in r.x)
    return best_x, -best_y


def per_sym_pnl(p_arr, sym_arr, mp_t, mp_th, tu, td):
    out = {}
    for k in SYMS:
        m = sym_arr == k
        a = gate_asym(p_arr[m], tu, td)
        pnl = vectorized_pnl(a, mp_t[m], mp_th[m]).sum()
        out[int(k)] = float(pnl)
    return out


def evaluate_arm(name, paths):
    print(f"\n=== {name} ({len(paths)} seeds) ===", flush=True)
    base, p_avg = avg_preds(paths)
    sym_arr = base["sym"].to_numpy()
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)
    t0 = time.time()
    (tu, td), tot_pnl = fit_de_2d(p_avg, sym_arr, mp_t, mp_th)
    print(f"  DE 2D: tu={tu:.4e} td={td:.4e} fit={time.time()-t0:.1f}s",
          flush=True)
    per = per_sym_pnl(p_avg, sym_arr, mp_t, mp_th, tu, td)
    loso = sum(per.values())
    print(f"  per-sym: {per}  loso={loso:.4f}", flush=True)
    return {
        "arm": name,
        "n_seeds": len(paths),
        "thresh": {"T_up": tu, "T_dn": td},
        "per_sym": per,
        "loso_equiv": loso,
    }


def main():
    baseline_paths = [
        os.path.join(HERE, f"pred_R3_baseline_seed{s}.parquet")
        for s in (1, 7, 42)
    ]
    trick_paths = [
        os.path.join(HERE, f"pred_R3_trick_seed{s}.parquet")
        for s in (1, 7, 42)
    ]

    res_b = evaluate_arm("baseline (schemeP 359-d, 3-seed)", baseline_paths)
    res_t = evaluate_arm("trick (schemeP+HYD 375-d, 3-seed)", trick_paths)

    delta_per_sym = {k: res_t["per_sym"][int(k)] - res_b["per_sym"][int(k)]
                     for k in SYMS}
    delta_loso = res_t["loso_equiv"] - res_b["loso_equiv"]
    delta_per_sym_min = min(delta_per_sym.values())
    delta_per_sym_max = max(delta_per_sym.values())

    out = {
        "baseline": res_b,
        "trick": res_t,
        "delta": {
            "delta_loso": delta_loso,
            "delta_per_sym": delta_per_sym,
            "delta_per_sym_min": delta_per_sym_min,
            "delta_per_sym_max": delta_per_sym_max,
        },
    }
    out_path = os.path.join(HERE, "loso_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}", flush=True)
    print(f"\nDELTA: loso={delta_loso:+.4f}  per_sym={delta_per_sym}  "
          f"min={delta_per_sym_min:+.4f}  max={delta_per_sym_max:+.4f}", flush=True)


if __name__ == "__main__":
    main()
