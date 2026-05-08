"""T99: refinement around the winner T87+1.5*T89+1.0*LGB_Huber+1.0*T95 = +43.71.

Tests fine-grained weight perturbations + alternatives without T89 (since T89 vs LGB_Huber
correlation is 0.86 — adding both may be redundant).
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
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


def gate(p, tu, td):
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def avg_preds(dir_, prefix, suffix=""):
    base = pd.read_parquet(os.path.join(dir_, f"{prefix}{SEEDS[0]}{suffix}.parquet"))
    p = np.zeros(len(base), dtype=np.float64)
    for s in SEEDS:
        df = pd.read_parquet(os.path.join(dir_, f"{prefix}{s}{suffix}.parquet"))
        if not (df["sym"].equals(base["sym"]) and df["t"].equals(base["t"])):
            raise RuntimeError(f"row mismatch {prefix}{s}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return base, p / len(SEEDS)


def evaluate(p, base, label, de_seeds=(0, 42, 1)):
    sym_arr = base["sym"].to_numpy()
    mp_t = base["midprice_t"].to_numpy(np.float64)
    mp_th = base["midprice_th"].to_numpy(np.float64)
    folds = [(p[sym_arr == k], mp_t[sym_arr == k], mp_th[sym_arr == k]) for k in SYMS]

    def obj(x):
        s = 0.0
        for fp, fmt, fmth in folds:
            a = gate(fp, x[0], x[1])
            s += vectorized_pnl(a, fmt, fmth).sum()
        return -float(s)

    best = (-1e9, 0.0, 0.0)
    for sd in de_seeds:
        r = differential_evolution(obj, bounds=[(0, 0.0040), (0, 0.0040)], seed=sd,
                                   maxiter=80, popsize=24, polish=True, tol=1e-7,
                                   init="sobol")
        v = float(-r.fun)
        if v > best[0]:
            best = (v, float(r.x[0]), float(r.x[1]))
    s, tu, td = best
    per = []
    for fp, fmt, fmth in folds:
        a = gate(fp, tu, td)
        per.append(float(vectorized_pnl(a, fmt, fmth).sum()))
    print(f"  [{label:<70s}] DE={s:+.4f} thr=({tu:.5f},{td:.5f}) per_sym={[f'{x:+.2f}' for x in per]}",
          flush=True)
    return {"label": label, "thr_up": tu, "thr_dn": td, "de_sum": s, "per_sym": per}


def main():
    print("=== T99 winner refinement ===", flush=True)
    t0 = time.time()
    base_h, p_lgbh = avg_preds(HERE, "pred_T99_huber_a0.001_seed")
    base_t87, p_t87 = avg_preds(T87, "pred_T87_seed", suffix="_main")
    base_t89, p_t89 = avg_preds(T89, "pred_T89_seed")
    base_t95, p_t95 = avg_preds(T95, "pred_T95_gru_w100_C_seed")
    base_cbh, p_cbh = avg_preds(HERE, "pred_T99_cb_huber_a0.001_seed")

    results = []

    # Confirm baselines
    print("\n[Baselines]", flush=True)
    results.append(evaluate((p_t87 + 1.5*p_t89) / 2.5, base_h, "T87 + 1.5*T89 (iter_015 baseline alt)"))
    results.append(evaluate((p_t87 + 1.5*p_t89 + 1.0*p_t95) / 3.5, base_h, "T87 + 1.5*T89 + 1.0*T95 (T87+T89+T95)"))

    # 3-way without T89
    print(f"\n[3-way: T87 + LGB_Huber + T95 (drop T89)]", flush=True)
    for w_h in [0.7, 1.0, 1.2, 1.5]:
        for w_95 in [0.5, 0.7, 1.0, 1.2]:
            p = (p_t87 + w_h * p_lgbh + w_95 * p_t95) / (1 + w_h + w_95)
            results.append(evaluate(p, base_h, f"T87 + {w_h}*LGB_Huber + {w_95}*T95"))

    # 4-way confirmed winner + neighbors
    print(f"\n[4-way: T87+T89+LGB_Huber+T95 fine-grained]", flush=True)
    for w_h in [0.8, 1.0, 1.2, 1.5]:
        for w_95 in [0.7, 1.0, 1.2, 1.5]:
            p = (p_t87 + 1.5 * p_t89 + w_h * p_lgbh + w_95 * p_t95) / (1 + 1.5 + w_h + w_95)
            results.append(evaluate(p, base_h, f"T87 + 1.5*T89 + {w_h}*LGB_Huber + {w_95}*T95"))

    # 4-way with T89 weight tuning too
    print(f"\n[4-way: with T89 weight tuning]", flush=True)
    for w_t89 in [1.0, 1.2, 1.5, 1.8]:
        for w_h in [1.0, 1.2]:
            p = (p_t87 + w_t89 * p_t89 + w_h * p_lgbh + 1.0 * p_t95) / (1 + w_t89 + w_h + 1.0)
            results.append(evaluate(p, base_h, f"T87 + {w_t89}*T89 + {w_h}*LGB_Huber + 1.0*T95"))

    # Replace T89 with CB_Huber (since CB_Huber is more directly aligned)
    print(f"\n[4-way: T87+CB_Huber(replaces T89)+LGB_Huber+T95]", flush=True)
    for w_cbh in [0.5, 0.7, 1.0]:
        for w_h in [0.7, 1.0]:
            p = (p_t87 + w_cbh * p_cbh + w_h * p_lgbh + 1.0 * p_t95) / (1 + w_cbh + w_h + 1.0)
            results.append(evaluate(p, base_h, f"T87 + {w_cbh}*CB_Huber + {w_h}*LGB_Huber + 1.0*T95"))

    # Sort
    results.sort(key=lambda r: -r["de_sum"])
    print(f"\n=== TOP 10 ===", flush=True)
    for r in results[:10]:
        print(f"  {r['label']:75s} de_sum={r['de_sum']:+.4f}  thr=({r['thr_up']:.5f},{r['thr_dn']:.5f})",
              flush=True)

    out = {"all_results": results, "top10": results[:10],
           "elapsed_sec": time.time() - t0}
    with open(os.path.join(HERE, "ev_winner_refine.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote ev_winner_refine.json (took {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
