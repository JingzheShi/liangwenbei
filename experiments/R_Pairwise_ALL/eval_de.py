"""R_Pairwise_ALL DE 2D LOSO-equiv evaluation.

Mirrors R3 eval.py: 3-seed mean prediction per arm, DE 2D (T_up, T_dn)
optimized over full 442k local test, per-sym PnL summed = LOSO-equiv.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = (0, 1, 2, 3, 4)
SEEDS = (1, 7, 42)
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


def avg_preds(prefix):
    base = pd.read_parquet(os.path.join(HERE, f"pred_RP_{prefix}_seed{SEEDS[0]}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(HERE, f"pred_RP_{prefix}_seed{s}.parquet"))
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch {prefix} seed{s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(SEEDS)


def fit_de_2d(p, sym, mp_t, mp_th, de_seeds=(0, 1, 2, 7, 42)):
    folds = [(p[sym == k], mp_t[sym == k], mp_th[sym == k]) for k in SYMS]
    def obj(x):
        s = 0.0
        for fp, fmt, fmth in folds:
            a = gate_asym(fp, x[0], x[1])
            s += vectorized_pnl(a, fmt, fmth).sum()
        return -float(s)
    best_x, best_y = None, np.inf
    for ds in de_seeds:
        r = differential_evolution(obj, bounds=[(0, 0.005), (0, 0.005)],
                                   seed=ds, maxiter=80, popsize=24,
                                   tol=1e-7, polish=True, init="sobol")
        if r.fun < best_y:
            best_y, best_x = float(r.fun), tuple(float(v) for v in r.x)
    return best_x, -best_y


def per_sym_pnl(p, sym, mp_t, mp_th, tu, td):
    out = {}
    for k in SYMS:
        m = sym == k
        a = gate_asym(p[m], tu, td)
        out[int(k)] = float(vectorized_pnl(a, mp_t[m], mp_th[m]).sum())
    return out


def evaluate(label, prefix):
    print(f"\n=== {label} ({prefix}) ===", flush=True)
    base, pavg = avg_preds(prefix)
    sym = base["sym"].to_numpy()
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)
    t0 = time.time()
    (tu, td), tot = fit_de_2d(pavg, sym, mp_t, mp_th)
    print(f"  DE 2D: tu={tu:.4e} td={td:.4e} fit={time.time()-t0:.1f}s", flush=True)
    per = per_sym_pnl(pavg, sym, mp_t, mp_th, tu, td)
    loso = sum(per.values())
    print(f"  per-sym: {per}  loso={loso:.4f}", flush=True)
    return {"arm": label, "thresh": {"T_up": tu, "T_dn": td},
            "per_sym": per, "loso_equiv": loso}


def per_seed_eval(prefix):
    """Independent per-seed DE eval (no ensemble) — for delta consistency."""
    out = []
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(HERE, f"pred_RP_{prefix}_seed{s}.parquet"))
        sym = df["sym"].to_numpy()
        mp_t = df["midprice_t"].to_numpy(np.float64)
        mp_th = df["midprice_th"].to_numpy(np.float64)
        p = df["pred_dmid_norm"].to_numpy(np.float64)
        (tu, td), tot = fit_de_2d(p, sym, mp_t, mp_th, de_seeds=(0, 42))
        per = per_sym_pnl(p, sym, mp_t, mp_th, tu, td)
        out.append({"seed": s, "thresh": {"T_up": tu, "T_dn": td},
                    "per_sym": per, "loso_equiv": sum(per.values())})
    return out


def main():
    res_base = evaluate("baseline", "baseline")
    res_trick = evaluate("trick", "trick")

    delta_loso = res_trick["loso_equiv"] - res_base["loso_equiv"]
    delta_per_sym = {k: res_trick["per_sym"][k] - res_base["per_sym"][k] for k in res_base["per_sym"]}
    delta_psmin = min(delta_per_sym.values())

    print(f"\n=== ENSEMBLE DELTA ===", flush=True)
    print(f"  loso: trick={res_trick['loso_equiv']:.4f}  baseline={res_base['loso_equiv']:.4f}  delta={delta_loso:+.4f}", flush=True)
    print(f"  per-sym delta: {delta_per_sym}  min={delta_psmin:+.4f}", flush=True)

    # per-seed deltas
    print(f"\n=== PER-SEED ===", flush=True)
    base_seeds = per_seed_eval("baseline")
    trick_seeds = per_seed_eval("trick")
    per_seed_deltas = []
    for bs, ts in zip(base_seeds, trick_seeds):
        d = ts["loso_equiv"] - bs["loso_equiv"]
        per_seed_deltas.append({"seed": bs["seed"],
                                "baseline": bs["loso_equiv"],
                                "trick": ts["loso_equiv"],
                                "delta": d})
        print(f"  seed={bs['seed']}  baseline={bs['loso_equiv']:.4f}  trick={ts['loso_equiv']:.4f}  delta={d:+.4f}", flush=True)
    seed_deltas = [pd["delta"] for pd in per_seed_deltas]
    print(f"  mean={np.mean(seed_deltas):+.4f}  std={np.std(seed_deltas, ddof=1):.4f}  min={min(seed_deltas):+.4f}  max={max(seed_deltas):+.4f}", flush=True)

    out = {"baseline": res_base, "trick": res_trick,
           "delta_loso": delta_loso,
           "delta_per_sym": delta_per_sym,
           "delta_per_sym_min": delta_psmin,
           "per_seed": per_seed_deltas,
           "per_seed_delta_mean": float(np.mean(seed_deltas)),
           "per_seed_delta_std": float(np.std(seed_deltas, ddof=1)),
           "success_gate_loso_geq_0p5": delta_loso > 0.5}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nresults.json written")


if __name__ == "__main__":
    main()
