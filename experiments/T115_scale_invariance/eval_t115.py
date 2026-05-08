"""T115 evaluation: DE thresh search + 3-seed average per variant.

Compute:
  - Per-variant 3-seed averaged predictions
  - DE thresh search (asymmetric tu, td)
  - Per-sym pnl breakdown
  - delta_loso = variant - baseline
  - delta_per_sym_min = min variant per_sym - min baseline per_sym

Output:
  eval_t115.json
  eval_t115.log (printed table)
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 42)
VARIANTS = ("baseline", "variant_A", "variant_B", "variant_C")


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


def split_sym(df):
    return [{"sym": int(k),
             "pred": df[df["sym"] == k]["pred_dmid_norm"].to_numpy(np.float64),
             "mp_t": df[df["sym"] == k]["midprice_t"].to_numpy(np.float64),
             "mp_th": df[df["sym"] == k]["midprice_th"].to_numpy(np.float64)}
            for k in SYMS]


def make_obj(folds):
    pred = [f["pred"] for f in folds]
    mp_t = [f["mp_t"] for f in folds]
    mp_th = [f["mp_th"] for f in folds]
    def f(x):
        s = 0.0
        for i in range(len(folds)):
            a = gate_asym(pred[i], x[0], x[1])
            s += vectorized_pnl(a, mp_t[i], mp_th[i]).sum()
        return -float(s)
    return f


def de(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def avg_preds(variant, alpha="0.001"):
    base = pd.read_parquet(os.path.join(HERE,
        f"pred_T115_{variant}_huber{alpha}_seed{SEEDS[0]}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    per_seed_preds = {}
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(HERE,
            f"pred_T115_{variant}_huber{alpha}_seed{s}.parquet"))
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch {variant} seed{s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
        per_seed_preds[s] = df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(SEEDS), per_seed_preds


def evaluate(p, base, label):
    df = base.copy()
    df["pred_dmid_norm"] = p.astype(np.float32)
    folds = split_sym(df)
    obj = make_obj(folds)
    s, tu, td = de(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    per = []
    for f in folds:
        a = gate_asym(f["pred"], tu, td)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    print(f"  [{label:<55s}] DE={s:+.4f} thr_up={tu:.4e} thr_dn={td:.4e}", flush=True)
    print(f"    per_sym = {[f'{x:+.2f}' for x in per]} (min={min(per):+.2f})", flush=True)
    return {"label": label, "thr_up": tu, "thr_dn": td, "de_sum": s,
            "per_sym": per, "per_sym_min": min(per)}


def main():
    print("=== T115 evaluation ===", flush=True)
    t0 = time.time()
    results = {}
    avg_preds_by_variant = {}
    for variant in VARIANTS:
        print(f"\n--- {variant} ---", flush=True)
        base, p_avg, per_seed = avg_preds(variant)
        avg_preds_by_variant[variant] = p_avg
        # 3-seed average
        r = evaluate(p_avg, base, f"{variant}-3seed-avg")
        # Per-seed eval
        per_seed_results = {}
        for s in SEEDS:
            r_s = evaluate(per_seed[s], base, f"{variant}-seed{s}")
            per_seed_results[s] = r_s
        results[variant] = {
            "3seed_avg": r,
            "per_seed": per_seed_results,
        }

    # Compute deltas vs baseline
    print(f"\n{'='*100}\n=== DELTA SUMMARY (vs baseline 3-seed avg) ===\n{'='*100}", flush=True)
    base_de = results["baseline"]["3seed_avg"]["de_sum"]
    base_min = results["baseline"]["3seed_avg"]["per_sym_min"]
    base_per_sym = results["baseline"]["3seed_avg"]["per_sym"]
    print(f"baseline 3seed: DE={base_de:+.4f}, per_sym_min={base_min:+.4f}, per_sym={[f'{x:+.2f}' for x in base_per_sym]}", flush=True)
    for variant in VARIANTS:
        if variant == "baseline":
            continue
        v_de = results[variant]["3seed_avg"]["de_sum"]
        v_min = results[variant]["3seed_avg"]["per_sym_min"]
        v_per_sym = results[variant]["3seed_avg"]["per_sym"]
        delta_de = v_de - base_de
        delta_min = v_min - base_min
        print(f"{variant:<12s}: DE={v_de:+.4f} (Δ{delta_de:+.4f}) per_sym_min={v_min:+.4f} (Δ{delta_min:+.4f})", flush=True)
        print(f"             per_sym diff={[f'{v_per_sym[i]-base_per_sym[i]:+.2f}' for i in range(5)]}", flush=True)

    out = os.path.join(HERE, "eval_t115.json")
    with open(out, "w") as f:
        # Convert results to JSON-safe (remove numpy)
        json.dump({"results": results,
                   "delta_summary": {
                        "vs_baseline": {
                            v: {
                                "delta_de": results[v]["3seed_avg"]["de_sum"] - base_de,
                                "delta_per_sym_min": results[v]["3seed_avg"]["per_sym_min"] - base_min,
                            } for v in VARIANTS if v != "baseline"
                        }
                   }}, f, indent=2)
    print(f"\nwrote -> {out}", flush=True)
    print(f"\nTotal time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
