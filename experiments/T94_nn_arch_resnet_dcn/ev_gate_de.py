"""T94: DE 4D thresh evaluator for ensemble candidates.

Given a list of pred parquet files (each with pred_dmid_norm), average them with optional weights,
then run the standard symmetric k sweep + DE asymmetric (single + LOSO-equiv).

Usage:
    python3 ev_gate_de.py --preds tag1:1.0 tag2:1.5 ... --label "ensemble_X"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER014_REF = 38.28


def vec_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    diff = mp_th - mp_t
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def split_by_sym(df, pred_col):
    out = []
    for k in SYMS:
        m = df["sym"] == k
        out.append({
            "sym": int(k),
            "pred": df.loc[m, pred_col].to_numpy(np.float64),
            "mp_t": df.loc[m, "midprice_t"].to_numpy(np.float64),
            "mp_th": df.loc[m, "midprice_th"].to_numpy(np.float64),
            "n": int(m.sum()),
        })
    return out


def make_obj_loso(folds):
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for fk in folds:
            a = ev_gate_asym(fk["pred"], thr_up, thr_dn)
            s += vec_pnl(a, fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def make_obj_single(pred, mp_t, mp_th):
    def f(x):
        thr_up, thr_dn = x
        a = ev_gate_asym(pred, thr_up, thr_dn)
        return -float(vec_pnl(a, mp_t, mp_th).sum())
    return f


def de(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", init="sobol",
        )
        runs.append({"seed": int(sd), "thr_up": float(r.x[0]),
                     "thr_dn": float(r.x[1]), "obj_val": float(-r.fun)})
    runs.sort(key=lambda x: x["obj_val"], reverse=True)
    return runs


def resolve_pred(spec: str):
    """spec = path or tag. Try local pred_<tag>_seed*.parquet (5-seed avg) or
    <T81|T75|T94>:tag for sister dirs."""
    if ":" in spec:
        prefix, val = spec.split(":", 1)
    else:
        prefix, val = "T94", spec
    if prefix == "T81":
        d = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")
        prefix_p = "pred_T81"
        seeds = [1, 7, 13, 42, 100]
    elif prefix == "T75":
        d = os.path.join(ROOT, "experiments", "T75_regression_dmid")
        prefix_p = "pred_T75"
        seeds = [1, 7, 13, 42, 100]
    elif prefix == "T94":
        d = HERE
        prefix_p = f"pred_{val}"
        # detect seeds available
        import glob
        files = sorted(glob.glob(os.path.join(d, f"{prefix_p}_seed*.parquet")))
        seeds = []
        for f in files:
            base = os.path.basename(f)
            try:
                s = int(base.replace(f"{prefix_p}_seed", "").replace(".parquet", ""))
                seeds.append(s)
            except Exception:
                pass
        if not seeds:
            raise FileNotFoundError(f"no {prefix_p}_seed*.parquet in {d}")
    else:
        raise ValueError(prefix)
    base = pd.read_parquet(os.path.join(d, f"{prefix_p}_seed{seeds[0]}.parquet"))
    p = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    for s in seeds[1:]:
        df = pd.read_parquet(os.path.join(d, f"{prefix_p}_seed{s}.parquet"))
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(seeds)
    base = base.copy()
    base["pred_dmid_norm"] = p.astype(np.float32)
    return base, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True,
                    help="format: spec:weight, e.g. T81:1.0 mlp_wide:0.5")
    ap.add_argument("--label", required=True)
    ap.add_argument("--save-json", action="store_true")
    args = ap.parse_args()

    # Parse and load
    components = []
    for spec_w in args.preds:
        spec, _, w = spec_w.rpartition(":")
        if spec == "" or w == "":
            spec = spec_w
            w = "1.0"
        try:
            float(w)
            ok_w = True
        except ValueError:
            spec = spec_w
            w = "1.0"
            ok_w = True
        df, seeds = resolve_pred(spec)
        components.append({"spec": spec, "weight": float(w), "seeds": seeds, "df": df})
        print(f"  loaded {spec} (5-seed avg, n={len(df):,}, weight={float(w):.2f})", flush=True)

    # Standardize each prediction series so weights are comparable, then weighted-average
    # Actually for ensemble of regression we just weighted-sum directly.
    base = components[0]["df"].copy()
    p_sum = np.zeros(len(base), dtype=np.float64)
    w_sum = 0.0
    for c in components:
        # rescale each component's pred to unit std (so weights are comparable across heterogenous models)
        p = c["df"]["pred_dmid_norm"].to_numpy(np.float64)
        std = p.std()
        if std > 0:
            p = p / std
        p_sum += c["weight"] * p
        w_sum += c["weight"]
    base["pred_dmid_norm"] = (p_sum / w_sum).astype(np.float32)

    # Now eval. EV-gate threshold is in the *standardized* prediction space.
    # We DE-search in [0, 0.5] (since std==1, k * 2*FEE no longer applies; use absolute).
    folds = split_by_sym(base, "pred_dmid_norm")
    pred_all = base["pred_dmid_norm"].to_numpy(np.float64)
    mp_t_all = base["midprice_t"].to_numpy(np.float64)
    mp_th_all = base["midprice_th"].to_numpy(np.float64)

    # DE in standardized space (predictions are zero-mean unit-std-ish per component, summed)
    # bounds wide enough to allow extreme thresh
    bounds = [(0.0, 1.5), (0.0, 1.5)]

    print(f"\n=== EV-gate DE on {args.label} ===", flush=True)
    print(f"  pred mean={pred_all.mean():.4f} std={pred_all.std():.4f}", flush=True)

    print("\n[DE single]", flush=True)
    de_single = de(make_obj_single(pred_all, mp_t_all, mp_th_all), bounds)
    bs = de_single[0]
    a_s = ev_gate_asym(pred_all, bs["thr_up"], bs["thr_dn"])
    sing_total = float(vec_pnl(a_s, mp_t_all, mp_th_all).sum())
    print(f"  best: thr_up={bs['thr_up']:.4f} thr_dn={bs['thr_dn']:.4f}  total={sing_total:+.4f}",
          flush=True)

    print("\n[DE LOSO-equiv]", flush=True)
    de_loso = de(make_obj_loso(folds), bounds)
    bl = de_loso[0]
    per_pnls = []
    per_na = []
    for fk in folds:
        a = ev_gate_asym(fk["pred"], bl["thr_up"], bl["thr_dn"])
        per_pnls.append(float(vec_pnl(a, fk["mp_t"], fk["mp_th"]).sum()))
        per_na.append(int((a != 1).sum()))
    loso_sum = float(sum(per_pnls))
    a_l = ev_gate_asym(pred_all, bl["thr_up"], bl["thr_dn"])
    loso_total = float(vec_pnl(a_l, mp_t_all, mp_th_all).sum())
    print(f"  best: thr_up={bl['thr_up']:.4f} thr_dn={bl['thr_dn']:.4f}", flush=True)
    print(f"  sum_per_sym={loso_sum:+.4f}  per_sym=[{', '.join(f'{x:+.3f}' for x in per_pnls)}]",
          flush=True)
    print(f"  single_total={loso_total:+.4f}", flush=True)
    print(f"\n  vs iter_014 ({ITER014_REF:+.2f}): {loso_sum-ITER014_REF:+.4f}", flush=True)

    if args.save_json:
        out = {"label": args.label, "components": [
            {"spec": c["spec"], "weight": c["weight"], "seeds": c["seeds"]} for c in components
        ], "de_single": {"thr_up": bs["thr_up"], "thr_dn": bs["thr_dn"], "total": sing_total},
            "de_loso": {"thr_up": bl["thr_up"], "thr_dn": bl["thr_dn"],
                        "sum_per_sym": loso_sum, "single_total": loso_total,
                        "per_sym": per_pnls, "per_sym_n_active": per_na},
            "vs_iter014": loso_sum - ITER014_REF}
        out_path = os.path.join(HERE, f"ensemble_{args.label}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
