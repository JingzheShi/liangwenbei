"""R_multinn_spo: ensemble eval — cross-corr matrix + DE thresh on combos.

Loads:
  - 4 archs × 3 seeds (R_multinn_spo NN preds)
  - T75 LGB 5-seed avg from preds dir

Computes:
  1. Per-arch 3-seed avg
  2. Cross-corr matrix (4×4) on per-arch avg preds
  3. Cross-corr each NN avg vs T75 LGB avg
  4. DE LOSO eval on:
     - Each arch alone (3-seed avg)
     - 3-way NN-only ensemble: t87_baseline + wide_mlp + deep_resnet
     - 4-way NN+LGB: above + T75
     - 4-way all-NN: t87 + wide + resnet + glu
     - 5-way all-NN+LGB
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
PREDS_DIR = os.environ.get("LWB_PREDS_DIR", "/root/lwb_remote_pkg/preds")

ARCHS = ["t87_baseline", "wide_mlp", "deep_resnet", "glu_mlp"]
SEEDS_NN = [1, 7, 42]
SEEDS_LGB = [1, 7, 13, 42, 100]
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# References (LOSO-equiv on local 442k test set)
ITER014_LOSO = 38.28
ITER015_LOSO = 40.09
T87_NN_ALONE_REF = "from local data"


def vectorized_pnl(action, mp_t, mp_th):
    side = action.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate_sym(pred, k):
    thr = k * 2.0 * FEE
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr] = 2
    a[pred < -thr] = 0
    return a


def ev_gate_asym(pred, thr_up, thr_dn):
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def split_by_sym(df, pred_arr):
    out = []
    sym = df["sym"].to_numpy()
    mp_t = df["midprice_t"].to_numpy(np.float64)
    mp_th = df["midprice_th"].to_numpy(np.float64)
    for k in SYMS:
        m = (sym == k)
        out.append({
            "sym": int(k),
            "pred": pred_arr[m].astype(np.float64),
            "mp_t": mp_t[m],
            "mp_th": mp_th[m],
            "n": int(m.sum()),
        })
    return out


def make_obj_loso(folds):
    def f(x):
        thr_up, thr_dn = x
        s = 0.0
        for fk in folds:
            a = ev_gate_asym(fk["pred"], thr_up, thr_dn)
            s += vectorized_pnl(a, fk["mp_t"], fk["mp_th"]).sum()
        return -float(s)
    return f


def de_search(obj, bounds, seeds=(0, 1, 2, 7, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "thr_up": float(r.x[0]),
            "thr_dn": float(r.x[1]),
            "obj": float(-r.fun),
        })
    runs.sort(key=lambda x: x["obj"], reverse=True)
    return runs


def eval_pred(pred_arr, base_df, label, do_de=True):
    folds = split_by_sym(base_df, pred_arr)

    # symmetric sweep
    sym_sweep = []
    for k in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
        per = [float(vectorized_pnl(ev_gate_sym(fk["pred"], k),
                                    fk["mp_t"], fk["mp_th"]).sum()) for fk in folds]
        sym_sweep.append({"k": k, "sum_per_sym": sum(per), "per_sym": per})
    best_sym = max(sym_sweep, key=lambda r: r["sum_per_sym"])

    de_loso = None
    if do_de:
        runs = de_search(make_obj_loso(folds), [(0.0, 0.0040), (0.0, 0.0040)])
        best = runs[0]
        thr_up, thr_dn = best["thr_up"], best["thr_dn"]
        per = [float(vectorized_pnl(ev_gate_asym(fk["pred"], thr_up, thr_dn),
                                    fk["mp_t"], fk["mp_th"]).sum()) for fk in folds]
        de_loso = {"thr_up": thr_up, "thr_dn": thr_dn,
                   "sum_per_sym": float(sum(per)), "per_sym": per}

    out = {
        "label": label,
        "sym_sweep": sym_sweep,
        "best_sym": best_sym,
        "de_loso": de_loso,
    }
    print(f"\n=== {label} ===", flush=True)
    print(f"  best sym k={best_sym['k']}: sum_per_sym={best_sym['sum_per_sym']:+.4f}", flush=True)
    if de_loso:
        print(f"  DE LOSO:  thr_up={de_loso['thr_up']:.5f} thr_dn={de_loso['thr_dn']:.5f}",
              flush=True)
        print(f"  DE LOSO sum_per_sym={de_loso['sum_per_sym']:+.4f} per_sym={de_loso['per_sym']}",
              flush=True)
        print(f"  vs iter_014 (+{ITER014_LOSO}): {de_loso['sum_per_sym']-ITER014_LOSO:+.4f}",
              flush=True)
        print(f"  vs iter_015 (+{ITER015_LOSO}): {de_loso['sum_per_sym']-ITER015_LOSO:+.4f}",
              flush=True)
    return out


def load_arch_avg(arch, seeds=SEEDS_NN):
    """Load per-seed preds for an arch, return (avg_pred, base_df, per_seed)."""
    base = None
    p_seeds = {}
    for s in seeds:
        path = os.path.join(HERE, f"pred_{arch}_seed{s}.parquet")
        df = pd.read_parquet(path)
        if base is None:
            base = df[["sym", "date", "t", "midprice_t", "midprice_th"]].copy()
        else:
            assert df["sym"].equals(base["sym"]) and df["t"].equals(base["t"]), \
                f"row order mismatch {path}"
        p_seeds[s] = df["pred_dmid_norm"].to_numpy(np.float64)
    p_avg = np.mean(np.stack([p_seeds[s] for s in seeds], axis=0), axis=0)
    return p_avg, base, p_seeds


def load_t75_avg(seeds=SEEDS_LGB):
    """T75 LGB 5-seed avg."""
    base = None
    p_avg = None
    for s in seeds:
        path = os.path.join(PREDS_DIR, f"pred_T75_seed{s}.parquet")
        df = pd.read_parquet(path)
        if base is None:
            base = df[["sym", "date", "t"]].copy()
            p_avg = np.zeros(len(df), dtype=np.float64)
        else:
            assert df["sym"].equals(base["sym"]) and df["t"].equals(base["t"]), \
                f"row order mismatch {path}"
        p_avg += df["pred_dmid_norm"].to_numpy(np.float64)
    p_avg /= len(seeds)
    return p_avg, base


def main():
    print("=== R_multinn_spo ensemble eval ===", flush=True)

    # Load all archs
    arch_preds = {}
    arch_base = None
    for arch in ARCHS:
        try:
            p, b, _ = load_arch_avg(arch)
            arch_preds[arch] = p
            if arch_base is None:
                arch_base = b
            else:
                assert b["sym"].equals(arch_base["sym"]), f"base mismatch for {arch}"
            print(f"  loaded {arch}: pred mean={p.mean():+.6f} std={p.std():.6f}", flush=True)
        except Exception as e:
            print(f"  FAIL load {arch}: {e}", flush=True)

    # Load T75
    p_t75, t75_base = load_t75_avg()
    print(f"  loaded T75 LGB: pred mean={p_t75.mean():+.6f} std={p_t75.std():.6f}", flush=True)
    # Sanity: t75 base should match arch base
    assert t75_base["sym"].equals(arch_base["sym"]) and t75_base["t"].equals(arch_base["t"]), \
        "T75 row order mismatch with NN preds"

    # === Cross-corr matrix ===
    print("\n=== Cross-corr matrix (NN per-arch avg) ===", flush=True)
    archs_loaded = list(arch_preds.keys())
    cc = np.zeros((len(archs_loaded), len(archs_loaded)))
    for i, a in enumerate(archs_loaded):
        for j, b in enumerate(archs_loaded):
            cc[i, j] = float(np.corrcoef(arch_preds[a], arch_preds[b])[0, 1])
    print("       " + " ".join(f"{a[:11]:>11s}" for a in archs_loaded), flush=True)
    for i, a in enumerate(archs_loaded):
        print(f"  {a[:11]:>11s}: " +
              " ".join(f"{cc[i,j]:>11.4f}" for j in range(len(archs_loaded))),
              flush=True)
    # NN-vs-T75
    print("\n=== NN-vs-T75 LGB cross-corr ===", flush=True)
    nn_t75_cc = {}
    for a in archs_loaded:
        c = float(np.corrcoef(arch_preds[a], p_t75)[0, 1])
        nn_t75_cc[a] = c
        print(f"  {a:>15s}: {c:.4f}", flush=True)

    # off-diag stats
    off = []
    for i in range(len(archs_loaded)):
        for j in range(i + 1, len(archs_loaded)):
            off.append(cc[i, j])
    off = np.asarray(off)
    print(f"\n  NN-NN off-diag: min={off.min():.4f}  max={off.max():.4f}  "
          f"mean={off.mean():.4f}", flush=True)

    # === Eval each arch alone ===
    arch_results = {}
    for a in archs_loaded:
        r = eval_pred(arch_preds[a], arch_base, f"{a} (3-seed avg)", do_de=True)
        arch_results[a] = r

    # === Eval T75 alone (just for sanity) ===
    r_t75 = eval_pred(p_t75, arch_base, "T75 LGB (5-seed avg)", do_de=True)

    # === Ensembles ===
    ensemble_results = []

    def avg_preds(*arrs):
        return np.mean(np.stack(arrs, axis=0), axis=0)

    # 3-way NN-only (canonical)
    if all(a in arch_preds for a in ["t87_baseline", "wide_mlp", "deep_resnet"]):
        p_3way = avg_preds(arch_preds["t87_baseline"], arch_preds["wide_mlp"],
                           arch_preds["deep_resnet"])
        r = eval_pred(p_3way, arch_base, "3-way NN (t87+wide+resnet)", do_de=True)
        ensemble_results.append(r)

    # 4-way NN-only
    if len(archs_loaded) == 4:
        p_4nn = avg_preds(*[arch_preds[a] for a in ARCHS])
        r = eval_pred(p_4nn, arch_base, "4-way NN (all archs)", do_de=True)
        ensemble_results.append(r)

    # 4-way NN+LGB
    if all(a in arch_preds for a in ["t87_baseline", "wide_mlp", "deep_resnet"]):
        p_3way = avg_preds(arch_preds["t87_baseline"], arch_preds["wide_mlp"],
                           arch_preds["deep_resnet"])
        for w_nn, w_lgb in [(1.0, 1.0), (1.5, 1.0), (1.0, 1.5), (2.0, 1.0), (1.0, 2.0)]:
            p_combo = (w_nn * p_3way + w_lgb * p_t75) / (w_nn + w_lgb)
            r = eval_pred(p_combo, arch_base,
                          f"3-way NN + T75 (w_nn={w_nn} w_lgb={w_lgb})", do_de=True)
            r["w_nn"] = w_nn
            r["w_lgb"] = w_lgb
            ensemble_results.append(r)

    # 5-way NN+LGB
    if len(archs_loaded) == 4:
        p_4nn = avg_preds(*[arch_preds[a] for a in ARCHS])
        for w_nn, w_lgb in [(1.0, 1.0), (1.5, 1.0), (1.0, 1.5)]:
            p_combo = (w_nn * p_4nn + w_lgb * p_t75) / (w_nn + w_lgb)
            r = eval_pred(p_combo, arch_base,
                          f"4-way NN + T75 (w_nn={w_nn} w_lgb={w_lgb})", do_de=True)
            r["w_nn"] = w_nn
            r["w_lgb"] = w_lgb
            ensemble_results.append(r)

    # === Best summary ===
    all_with_de = list(arch_results.values()) + [r_t75] + ensemble_results
    all_with_de = [r for r in all_with_de if r.get("de_loso")]
    best = max(all_with_de, key=lambda r: r["de_loso"]["sum_per_sym"])
    print(f"\n{'='*78}", flush=True)
    print(f"BEST: {best['label']}  → DE LOSO {best['de_loso']['sum_per_sym']:+.4f}  "
          f"(vs iter_015 {best['de_loso']['sum_per_sym'] - ITER015_LOSO:+.4f})",
          flush=True)
    print('=' * 78, flush=True)

    # Save full results
    out = {
        "ARCHS": ARCHS,
        "SEEDS_NN": SEEDS_NN,
        "SEEDS_LGB": SEEDS_LGB,
        "cross_corr_matrix": {a: {b: float(cc[i, j])
                                  for j, b in enumerate(archs_loaded)}
                              for i, a in enumerate(archs_loaded)},
        "nn_vs_t75_cc": nn_t75_cc,
        "cc_off_diag": {"min": float(off.min()), "max": float(off.max()),
                        "mean": float(off.mean())},
        "arch_results": arch_results,
        "t75_alone": r_t75,
        "ensemble_results": ensemble_results,
        "best": best,
        "iter014_ref": ITER014_LOSO,
        "iter015_ref": ITER015_LOSO,
    }
    out_path = os.path.join(HERE, "ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
