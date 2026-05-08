"""R_stack3_compound DE 4D thresh + LOSO eval.

For each variant V0 / V1 / V2 / V3 / Vlogret:
  1. Average pred_dmid_norm across 3 seeds (1, 7, 42)
  2. DE search over (T_up, T_dn) shared across syms (2D)
     ALSO DE search 4D: (T_up_lo, T_up_hi, T_dn_lo, T_dn_hi) — but 2D shared is the
     standard format; we run 2D for direct comparison with R3/R4/T117.
  3. Per-sym PnL @ DE thresh, sum = LOSO-equivalent

Then compute compound_lift, additive_predicted, efficiency.
"""
from __future__ import annotations
import json, os, time, argparse
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 42)
VARIANTS = ("V0", "V1", "V2", "V3", "Vlogret")


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
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


def de_search(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd,
                                   maxiter=80, popsize=24,
                                   polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def avg_preds(prefix, pred_dir):
    base = pd.read_parquet(os.path.join(pred_dir, f"pred_{prefix}_seed{SEEDS[0]}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(pred_dir, f"pred_{prefix}_seed{s}.parquet"))
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch {prefix} seed{s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(SEEDS)


def evaluate(label, prefix, pred_dir):
    print(f"\n--- {label} ({prefix}) ---", flush=True)
    base, pavg = avg_preds(prefix, pred_dir)
    df = base.copy()
    df["pred_dmid_norm"] = pavg.astype(np.float32)
    folds = split_sym(df)
    obj = make_obj(folds)
    s, tu, td = de_search(obj, [(0.0, 0.0040), (0.0, 0.0040)])
    per = []
    for f in folds:
        a = gate_asym(f["pred"], tu, td)
        per.append(float(vectorized_pnl(a, f["mp_t"], f["mp_th"]).sum()))
    print(f"  DE: sum={s:+.4f} tu={tu:.4e} td={td:.4e}", flush=True)
    print(f"  per_sym = {[round(x,3) for x in per]}", flush=True)
    return {"label": label, "prefix": prefix,
            "loso": s, "thr_up": tu, "thr_dn": td,
            "per_sym": per, "per_sym_min": min(per), "per_sym_max": max(per),
            "per_sym_std": float(np.std(per))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=HERE)
    ap.add_argument("--out-json", default=os.path.join(HERE, "eval_results.json"))
    args = ap.parse_args()

    print("=== R_stack3 compound LGB Huber 3-seed eval ===", flush=True)
    t0 = time.time()
    results = {}
    for v in VARIANTS:
        # may not all exist; skip if missing
        first = os.path.join(args.pred_dir, f"pred_{v}_seed{SEEDS[0]}.parquet")
        if not os.path.exists(first):
            print(f"  SKIP {v} (no preds)", flush=True)
            continue
        results[v] = evaluate(v, v, args.pred_dir)

    # Compound analytics
    metrics = {}
    if "V0" in results:
        v0 = results["V0"]["loso"]
        for v in ("V1", "V2", "V3", "Vlogret"):
            if v in results:
                metrics[f"delta_{v}"] = results[v]["loso"] - v0
        if "V3" in results:
            metrics["compound_lift"] = results["V3"]["loso"] - v0
        # additive predicted: sum of single-trick lifts
        # lift_HYD ≈ V1 - V0
        # lift_lagret ≈ Vlogret - V0  (or task-stated +1.60)
        # lift_monotone ≈ V3 - V2  (approx; T117-on-this-baseline)
        # additive = lift_HYD + lift_lagret + lift_monotone
        lift_hyd = (results["V1"]["loso"] - v0) if "V1" in results else None
        lift_logret = (results["Vlogret"]["loso"] - v0) if "Vlogret" in results else None
        lift_mono = (results["V3"]["loso"] - results["V2"]["loso"]) if ("V3" in results and "V2" in results) else None
        metrics["lift_hyd_only"] = lift_hyd
        metrics["lift_logret_only"] = lift_logret
        metrics["lift_monotone_only"] = lift_mono
        if lift_hyd is not None and lift_logret is not None and lift_mono is not None:
            additive_predicted = lift_hyd + lift_logret + lift_mono
            metrics["additive_predicted"] = additive_predicted
            if "compound_lift" in metrics:
                metrics["compound_efficiency"] = (
                    metrics["compound_lift"] / additive_predicted
                    if additive_predicted > 1e-9 else None
                )

    print("\n" + "=" * 80, flush=True)
    print("Summary", flush=True)
    print("=" * 80, flush=True)
    for v, r in results.items():
        print(f"  {v}: LOSO={r['loso']:+.4f} per_sym_min={r['per_sym_min']:.3f}"
              f" per_sym={[round(x,2) for x in r['per_sym']]}", flush=True)
    print("\nCompound:", flush=True)
    for k, v in metrics.items():
        if v is None:
            print(f"  {k} = N/A", flush=True)
        else:
            print(f"  {k} = {v:+.4f}", flush=True)
    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)

    out = {
        "task": "R_stack3 compound stacking experiment",
        "seeds": list(SEEDS),
        "variants": list(results.keys()),
        "results": results,
        "metrics": metrics,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
