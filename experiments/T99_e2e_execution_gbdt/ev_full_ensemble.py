"""T99 full ensemble eval: LGB Huber + CB Huber + T87 + T89 + T75 + T95.

Tests:
  - All standalone
  - 2-way pairings (LGB_Huber + T87, LGB_Huber + T89, etc.)
  - 3-way: T87 + T89 + LGB_Huber (best for replacing T75)
  - 4-way: T87 + T89 + LGB_Huber + T95 (with T95 as low-corr diversifier)
  - 5-way: all
  - LGB_Huber + CB_Huber (within-T99) ensemble
  - Adding CB_Huber on top of T87+T89+LGB_Huber

Output: ev_full_ensemble.json
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T75 = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T87 = os.path.join(ROOT, "experiments", "T87_spo_dfl")
T89 = os.path.join(ROOT, "experiments", "T89_catboost_regression")
T95 = os.path.join(ROOT, "experiments", "T95_cnn_rnn_deeplob")

SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
SEEDS = (1, 7, 13, 42, 100)


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


def make_obj(folds_pred, folds_mpt, folds_mpth):
    def f(x):
        s = 0.0
        for i in range(len(folds_pred)):
            a = gate_asym(folds_pred[i], x[0], x[1])
            s += vectorized_pnl(a, folds_mpt[i], folds_mpth[i]).sum()
        return -float(s)
    return f


def de(obj, bounds=((0, 0.0040), (0, 0.0040)), seeds=(0, 42)):
    runs = []
    for sd in seeds:
        r = differential_evolution(obj, bounds=bounds, seed=sd, maxiter=60,
                                   popsize=20, polish=True, tol=1e-7, init="sobol")
        runs.append((float(-r.fun), float(r.x[0]), float(r.x[1])))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0]


def avg_preds(dir_, prefix, suffix=""):
    base = pd.read_parquet(os.path.join(dir_, f"{prefix}{SEEDS[0]}{suffix}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(dir_, f"{prefix}{s}{suffix}.parquet"))
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch {prefix}{s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(SEEDS)


def evaluate(p, base, label):
    sym_arr = base["sym"].to_numpy()
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)
    folds_pred = [p[sym_arr == k] for k in SYMS]
    folds_mpt = [mp_t[sym_arr == k] for k in SYMS]
    folds_mpth = [mp_th[sym_arr == k] for k in SYMS]
    obj = make_obj(folds_pred, folds_mpt, folds_mpth)
    s, tu, td = de(obj)
    per = []
    for i in range(len(SYMS)):
        a = gate_asym(folds_pred[i], tu, td)
        per.append(float(vectorized_pnl(a, folds_mpt[i], folds_mpth[i]).sum()))
    print(f"  [{label:<60s}] DE={s:+.4f} thr=({tu:.5f},{td:.5f}) "
          f"per_sym={[f'{x:+.2f}' for x in per]}", flush=True)
    return {"label": label, "thr_up": tu, "thr_dn": td, "de_sum": s, "per_sym": per}


def main():
    print("=== T99 full ensemble eval ===", flush=True)
    t0 = time.time()
    base_h, p_lgbh = avg_preds(HERE, "pred_T99_huber_a0.001_seed")
    base_t75, p_t75 = avg_preds(T75, "pred_T75_seed")
    base_t87, p_t87 = avg_preds(T87, "pred_T87_seed", suffix="_main")
    base_t89, p_t89 = avg_preds(T89, "pred_T89_seed")
    base_t95, p_t95 = avg_preds(T95, "pred_T95_gru_w100_C_seed")
    base_cbh, p_cbh = avg_preds(HERE, "pred_T99_cb_huber_a0.001_seed")

    for nm, b in [("T75", base_t75), ("T87", base_t87), ("T89", base_t89),
                  ("T95", base_t95), ("CB_Huber", base_cbh)]:
        if not (base_h["sym"].equals(b["sym"]) and base_h["t"].equals(b["t"])):
            raise RuntimeError(f"row mismatch {nm}")

    names = ["LGB_Huber", "CB_Huber", "T75_RMSE", "T87_SPO+", "T89_CB_RMSE", "T95_GRU"]
    preds = [p_lgbh, p_cbh, p_t75, p_t87, p_t89, p_t95]

    print("\n[Cross-correlation matrix]", flush=True)
    n = len(names)
    cm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cm[i, j] = float(np.corrcoef(preds[i], preds[j])[0, 1])
    print(f"  {'':14s}" + " ".join(f"{nm[:10]:>10s}" for nm in names), flush=True)
    for i in range(n):
        print(f"  {names[i][:14]:14s}" + " ".join(f"{cm[i,j]:>+10.4f}" for j in range(n)),
              flush=True)

    print(f"\n[Standalone DE asym LOSO]", flush=True)
    results = []
    for nm, p in zip(names, preds):
        results.append(evaluate(p, base_h, f"{nm}-only"))

    print(f"\n[Within-T99: LGB_Huber + CB_Huber]", flush=True)
    for w in [0.5, 0.7, 1.0, 1.5, 2.0]:
        p = (p_lgbh + w * p_cbh) / (1.0 + w)
        results.append(evaluate(p, base_h, f"LGB_Huber + CB_Huber w=1:{w}"))

    print(f"\n[2-way: T87 + LGB_Huber]", flush=True)
    for w in [0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0]:
        p = (p_t87 + w * p_lgbh) / (1.0 + w)
        results.append(evaluate(p, base_h, f"T87 + LGB_Huber w=1:{w}"))

    print(f"\n[2-way: T89 + LGB_Huber]", flush=True)
    for w in [0.5, 0.7, 1.0, 1.5]:
        p = (p_t89 + w * p_lgbh) / (1.0 + w)
        results.append(evaluate(p, base_h, f"T89 + LGB_Huber w=1:{w}"))

    print(f"\n[3-way: T87+T89+LGB_Huber (replace T75)]", flush=True)
    # Original iter_015 was T87+T89 (1:1.5) + T75 (?). Now Huber replaces T75
    for wlgbh in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        p = (p_t87 + 1.5 * p_t89 + wlgbh * p_lgbh) / (1.0 + 1.5 + wlgbh)
        results.append(evaluate(p, base_h, f"T87 + 1.5*T89 + {wlgbh}*LGB_Huber"))

    print(f"\n[3-way: T87+LGB_Huber+CB_Huber]", flush=True)
    for w_cbh in [0.3, 0.5, 0.7, 1.0]:
        p = (p_t87 + 1.0 * p_lgbh + w_cbh * p_cbh) / (1.0 + 1.0 + w_cbh)
        results.append(evaluate(p, base_h, f"T87 + LGB_Huber + {w_cbh}*CB_Huber"))

    print(f"\n[4-way: T87+T89+LGB_Huber+T95]", flush=True)
    for wlgbh in [0.5, 0.7, 1.0]:
        for w95 in [0.3, 0.5, 0.7, 1.0]:
            p = (p_t87 + 1.5 * p_t89 + wlgbh * p_lgbh + w95 * p_t95) / (
                1.0 + 1.5 + wlgbh + w95)
            results.append(evaluate(p, base_h,
                f"T87 + 1.5*T89 + {wlgbh}*LGB_Huber + {w95}*T95"))

    print(f"\n[4-way: T87+T89+T95+CB_Huber (no LGB_Huber)]", flush=True)
    for w_cbh in [0.3, 0.5, 0.7, 1.0]:
        for w95 in [0.5, 0.7, 1.0]:
            p = (p_t87 + 1.5 * p_t89 + w_cbh * p_cbh + w95 * p_t95) / (
                1.0 + 1.5 + w_cbh + w95)
            results.append(evaluate(p, base_h,
                f"T87 + 1.5*T89 + {w_cbh}*CB_Huber + {w95}*T95"))

    print(f"\n[5-way: T87+T89+LGB_Huber+CB_Huber+T95]", flush=True)
    for wlgbh in [0.5, 0.7, 1.0]:
        for w_cbh in [0.3, 0.5]:
            for w95 in [0.5, 0.7]:
                p = (p_t87 + 1.5 * p_t89 + wlgbh * p_lgbh + w_cbh * p_cbh
                     + w95 * p_t95) / (1.0 + 1.5 + wlgbh + w_cbh + w95)
                results.append(evaluate(p, base_h,
                    f"T87 + 1.5*T89 + {wlgbh}*LGB_Huber + {w_cbh}*CB_Huber + {w95}*T95"))

    # Sort
    results.sort(key=lambda r: -r["de_sum"])
    print(f"\n=== TOP 15 ===", flush=True)
    for r in results[:15]:
        print(f"  {r['label']:65s} de_sum={r['de_sum']:+.4f}  thr=({r['thr_up']:.5f},{r['thr_dn']:.5f})", flush=True)

    print(f"\n=== Reference ===", flush=True)
    print(f"  T87+T89 w=1:1.5 = +41.01 (current iter_015 = T87+T75 = +40.09)", flush=True)
    print(f"  T87+T89+T95 (4-way ensemble eval) = +42.34", flush=True)
    print(f"  Best result so far: {results[0]['label']}  DE={results[0]['de_sum']:+.4f}", flush=True)

    out = {
        "task": "T99 full ensemble eval",
        "cross_corr_matrix": {names[i]: {names[j]: float(cm[i,j]) for j in range(n)} for i in range(n)},
        "all_results": results,
        "top15": results[:15],
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "ev_full_ensemble.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote ev_full_ensemble.json (took {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
