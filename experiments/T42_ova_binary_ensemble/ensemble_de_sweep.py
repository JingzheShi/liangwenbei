"""T42: ensemble 5-seed multiclass OOF (iter_005b) with single-seed B_up/B_dn
binary heads, then run differential-evolution 4-D asymmetric threshold search
on each ensemble strategy.

5 ensemble strategies (only p0 / p2 are touched; p1 = 1 - p0 - p2):

    E1 replace : new_p2 = B_up_p, new_p0 = B_dn_p
    E2 average : new_p2 = (multi_p2 + B_up_p) / 2  (same for p0)
    E3 weighted: new_p2 = 0.6 * multi_p2 + 0.4 * B_up_p  (same for p0)
    E4 geomean : new_p2 = sqrt(multi_p2 * B_up_p)  (same for p0)
    E5 max     : new_p2 = max(multi_p2, B_up_p)  (same for p0)

After computing the new (p0, p2) we recover p1 = max(0, 1 - p0 - p2). If
p0 + p2 > 1 we renormalise the triple back to a simplex (preserving p1
non-negativity).  Then DE 4D (T_up, T_dn, d_up, d_dn) maximises sum PnL
across the 5 LOSO folds.
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
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

# Reuse the gating + PnL utilities from T30 (same dataset).
T30_DIR = os.path.join(ROOT, "experiments", "T30_threshold_optim")
sys.path.insert(0, T30_DIR)
from _common import (  # noqa: E402
    gate_asymmetric,
    load_all_folds,
    vectorized_pnl,
)

SEED_BIN = 42
N_FOLDS = 5
H = 60


def _bin_path(head: str, held: int) -> str:
    return os.path.join(HERE, f"{head}_seed{SEED_BIN}_held{held}.parquet")


def load_binary_fold(head: str, held: int) -> np.ndarray:
    """Return P(positive) array aligned to the multiclass OOF row order."""
    df = pd.read_parquet(_bin_path(head, held))
    return df["prob_pos"].to_numpy(np.float32), df


def _renormalise_simplex(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """Map each (p0,p1,p2) row to a valid probability simplex.

    1. clip to [0, 1]
    2. if p0 + p2 > 1, scale (p0, p2) down so that p0 + p2 = 1, p1 = 0
    3. else p1 = 1 - p0 - p2, with p1 clipped to >= 0
    """
    p0c = np.clip(p0, 0.0, 1.0).astype(np.float64)
    p2c = np.clip(p2, 0.0, 1.0).astype(np.float64)
    s = p0c + p2c
    over = s > 1.0
    if over.any():
        scale = np.where(over, 1.0 / s, 1.0)
        p0c = p0c * scale
        p2c = p2c * scale
    p1n = 1.0 - p0c - p2c
    p1n = np.clip(p1n, 0.0, 1.0)
    return p0c.astype(np.float32), p1n.astype(np.float32), p2c.astype(np.float32)


ENSEMBLE_NAMES = ["E1_replace", "E2_average", "E3_w0p6_0p4",
                  "E4_geomean", "E5_max"]


def make_ensemble(name: str, multi_p: np.ndarray,
                  b_up: np.ndarray, b_dn: np.ndarray) -> np.ndarray:
    """Return ensembled (n,3) probability array.

    multi_p: (n, 3) softmax from 5-seed iter_005b ensemble
    b_up:    (n,) P(label==2) from single-seed B_up
    b_dn:    (n,) P(label==0) from single-seed B_dn
    """
    p0_m = multi_p[:, 0].astype(np.float64)
    p2_m = multi_p[:, 2].astype(np.float64)
    bu = b_up.astype(np.float64)
    bd = b_dn.astype(np.float64)

    if name == "E1_replace":
        new_p2 = bu
        new_p0 = bd
    elif name == "E2_average":
        new_p2 = 0.5 * (p2_m + bu)
        new_p0 = 0.5 * (p0_m + bd)
    elif name == "E3_w0p6_0p4":
        new_p2 = 0.6 * p2_m + 0.4 * bu
        new_p0 = 0.6 * p0_m + 0.4 * bd
    elif name == "E4_geomean":
        new_p2 = np.sqrt(np.clip(p2_m * bu, 0.0, 1.0))
        new_p0 = np.sqrt(np.clip(p0_m * bd, 0.0, 1.0))
    elif name == "E5_max":
        new_p2 = np.maximum(p2_m, bu)
        new_p0 = np.maximum(p0_m, bd)
    else:
        raise ValueError(name)

    # placeholder p1 (renormalised below)
    p1_dummy = np.zeros_like(new_p0)
    p0r, p1r, p2r = _renormalise_simplex(new_p0, p1_dummy, new_p2)
    return np.stack([p0r, p1r, p2r], axis=1).astype(np.float32)


def evaluate_4d_de(ens_per_fold, label_per_fold, mp_t_per_fold, mp_th_per_fold,
                   tag: str, n_runs: int = 5):
    """Run scipy.differential_evolution on (T_up, T_dn, d_up, d_dn). Return best."""
    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]

    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for k in range(N_FOLDS):
            pred = gate_asymmetric(ens_per_fold[k], Tu, Td, du, dd)
            s += vectorized_pnl(pred, label_per_fold[k],
                                mp_t_per_fold[k], mp_th_per_fold[k]).sum()
        return -float(s)

    runs = []
    for seed in (0, 1, 2, 7, 42):
        t0 = time.time()
        result = differential_evolution(
            f, bounds=bounds, seed=seed, maxiter=80, popsize=24,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        sum_pnl = -result.fun
        Tu, Td, du, dd = result.x
        per = []
        n_active = []
        for k in range(N_FOLDS):
            pred = gate_asymmetric(ens_per_fold[k], Tu, Td, du, dd)
            per.append(float(vectorized_pnl(
                pred, label_per_fold[k],
                mp_t_per_fold[k], mp_th_per_fold[k]).sum()))
            n_active.append(int((pred != 1).sum()))
        runs.append({
            "seed": seed,
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "sum_cum_pnl": float(sum_pnl),
            "per_fold_pnl": per,
            "n_active_per_fold": n_active,
            "n_pos_folds": int(sum(1 for x in per if x > 0)),
            "std_cum_pnl": float(np.std(per, ddof=0)),
            "nfev": int(result.nfev),
            "elapsed_sec": time.time() - t0,
        })
        print(
            f"  [{tag}] seed={seed}: T_up={Tu:.4f} T_dn={Td:.4f} "
            f"d_up={du:.4f} d_dn={dd:.4f} -> sum={sum_pnl:+.4f} "
            f"per_fold={[round(x,3) for x in per]} "
            f"n_active={n_active} ({result.nfev} evals, {time.time()-t0:.1f}s)",
            flush=True,
        )

    runs.sort(key=lambda r: r["sum_cum_pnl"], reverse=True)
    return runs


def main():
    t_main = time.time()
    print(f"=== T42 ensemble + DE sweep (h={H}, bin_seed={SEED_BIN}) ===",
          flush=True)

    # 1) Load 5-seed iter_005b OOF (probabilities already mean-averaged)
    folds = load_all_folds()
    multi_p_per_fold = []
    label_per_fold = []
    mp_t_per_fold = []
    mp_th_per_fold = []
    sym_per_fold = []
    for k in range(N_FOLDS):
        df = folds[k]
        multi_p_per_fold.append(df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32))
        label_per_fold.append(df["true_label"].to_numpy(np.int64))
        mp_t_per_fold.append(df["midprice_t"].to_numpy(np.float64))
        mp_th_per_fold.append(df["midprice_th"].to_numpy(np.float64))
        sym_per_fold.append(df["sym"].to_numpy(np.int8))

    # 2) Load binary OOF, align row order to multiclass via (sym, date, session, t)
    b_up_per_fold = []
    b_dn_per_fold = []
    for k in range(N_FOLDS):
        bu, bu_df = load_binary_fold("B_up", k)
        bd, bd_df = load_binary_fold("B_dn", k)
        # Validate alignment to multiclass OOF for fold k
        df_mc = folds[k]
        assert len(bu_df) == len(df_mc), (
            f"row mismatch fold {k}: B_up {len(bu_df)} vs multi {len(df_mc)}")
        assert len(bd_df) == len(df_mc), (
            f"row mismatch fold {k}: B_dn {len(bd_df)} vs multi {len(df_mc)}")
        # Multi OOF and binary OOF were both produced in the same row order
        # (mask m_te=test_full['sym']==held, identical for every trainer call).
        # As a defensive sanity check, verify (sym,date,t) tuples match elementwise.
        for col in ("sym", "date", "t"):
            mc_v = df_mc[col].to_numpy()
            bu_v = bu_df[col].to_numpy()
            bd_v = bd_df[col].to_numpy()
            if not (np.array_equal(mc_v, bu_v) and np.array_equal(mc_v, bd_v)):
                raise RuntimeError(
                    f"col '{col}' order mismatch in fold {k}: "
                    f"multi[:5]={mc_v[:5]} bup[:5]={bu_v[:5]} bdn[:5]={bd_v[:5]}")
        b_up_per_fold.append(bu)
        b_dn_per_fold.append(bd)
        print(
            f"  fold {k}: n={len(bu)}  multi p0/p2_mean={multi_p_per_fold[k][:,0].mean():.4f}/"
            f"{multi_p_per_fold[k][:,2].mean():.4f}  "
            f"B_up_mean={bu.mean():.4f}  B_dn_mean={bd.mean():.4f}",
            flush=True,
        )

    # Diagnostic: per-class accuracy / recall on raw probabilities
    print("\n--- Raw recall diagnostics (argmax over multi vs argmax with binary) ---",
          flush=True)
    for k in range(N_FOLDS):
        y = label_per_fold[k]
        mp = multi_p_per_fold[k]
        am = mp.argmax(axis=1)
        n0 = int((y == 0).sum()); n2 = int((y == 2).sum())
        rec0_m = (am[y == 0] == 0).mean() if n0 else 0.0
        rec2_m = (am[y == 2] == 2).mean() if n2 else 0.0
        # Binary head as standalone classifier (threshold 0.5)
        bup = b_up_per_fold[k]
        bdn = b_dn_per_fold[k]
        rec2_b = (bup[y == 2] >= 0.5).mean() if n2 else 0.0
        rec0_b = (bdn[y == 0] >= 0.5).mean() if n0 else 0.0
        print(
            f"  fold {k}: recall(2) multi={rec2_m:.4f}  "
            f"binary(thr0.5)={rec2_b:.4f} | "
            f"recall(0) multi={rec0_m:.4f}  "
            f"binary(thr0.5)={rec0_b:.4f}",
            flush=True,
        )

    # 3) For each ensemble strategy, build per-fold ens probs and run DE
    summary = {}
    for name in ENSEMBLE_NAMES:
        print(f"\n--- Ensemble {name} ---", flush=True)
        ens_per_fold = []
        for k in range(N_FOLDS):
            ens = make_ensemble(name,
                                multi_p_per_fold[k],
                                b_up_per_fold[k],
                                b_dn_per_fold[k])
            ens_per_fold.append(ens)
            row_sum = ens.sum(axis=1)
            assert np.allclose(row_sum, 1.0, atol=1e-3), (
                f"prob simplex broken {name} fold {k}: "
                f"row_sum [{row_sum.min():.4f}, {row_sum.max():.4f}]")
        de_runs = evaluate_4d_de(ens_per_fold, label_per_fold,
                                 mp_t_per_fold, mp_th_per_fold,
                                 tag=name, n_runs=5)
        best = de_runs[0]
        print(
            f"  [{name}] BEST sum={best['sum_cum_pnl']:+.4f} std={best['std_cum_pnl']:.4f} "
            f"pos={best['n_pos_folds']}/5  per_fold={[round(x,3) for x in best['per_fold_pnl']]}",
            flush=True,
        )
        summary[name] = {"de_runs": de_runs, "best": best}

    # 4) Reference baseline: iter_006 DE optimum on plain 5-seed multi (no binary)
    print("\n--- Reference: plain 5-seed multi (DE re-run for fairness) ---", flush=True)
    ref_runs = evaluate_4d_de(multi_p_per_fold, label_per_fold,
                              mp_t_per_fold, mp_th_per_fold,
                              tag="REF_multi_only", n_runs=5)
    summary["REF_multi_only"] = {"de_runs": ref_runs, "best": ref_runs[0]}

    # 5) Pick winner
    print("\n=== T42 SUMMARY ===", flush=True)
    table = []
    for name, blob in summary.items():
        b = blob["best"]
        table.append({
            "ensemble": name,
            "best_sum": b["sum_cum_pnl"],
            "best_std": b["std_cum_pnl"],
            "n_pos_folds": b["n_pos_folds"],
            "T_up": b["T_up"], "T_dn": b["T_dn"],
            "d_up": b["d_up"], "d_dn": b["d_dn"],
            "per_fold": b["per_fold_pnl"],
        })
    df_sum = pd.DataFrame(table).sort_values("best_sum", ascending=False)
    print(df_sum.to_string(index=False), flush=True)

    iter_006_sum = 13.6062
    threshold_pkg = 14.0  # task spec
    winning = df_sum.iloc[0]
    out = {
        "task": "T42 OvA binary ensemble + DE 4D thresh sweep",
        "horizon": H,
        "bin_seed": SEED_BIN,
        "iter_006_sum_reference": iter_006_sum,
        "package_threshold": threshold_pkg,
        "summary_table": df_sum.to_dict(orient="records"),
        "by_ensemble": {
            name: {
                "best": blob["best"],
                "all_de_runs": blob["de_runs"],
            } for name, blob in summary.items()
        },
        "winner": {
            "ensemble": winning["ensemble"],
            "best_sum": float(winning["best_sum"]),
            "uplift_vs_iter_006": float(winning["best_sum"] - iter_006_sum),
            "exceeds_threshold": bool(winning["best_sum"] > threshold_pkg),
        },
        "elapsed_sec": time.time() - t_main,
    }
    out_path = os.path.join(HERE, "ensemble_de_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}", flush=True)
    print(
        f"\nWINNER: {winning['ensemble']}  sum={winning['best_sum']:+.4f}  "
        f"uplift_vs_iter_006={winning['best_sum'] - iter_006_sum:+.4f}  "
        f"exceeds_pkg_threshold(>{threshold_pkg}) = "
        f"{bool(winning['best_sum'] > threshold_pkg)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
