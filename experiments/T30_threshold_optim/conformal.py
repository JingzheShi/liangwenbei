"""Split-conformal threshold on 5-seed aug_a OOF h_60.

Procedure (per fold k):
  - "calibration set" = OOF rows from the 4 NON-held folds (i.e. k' != k)
  - "test set" = OOF rows from fold k (held-out sym)
  - For each calibration row, compute side_max = max(prob_0, prob_2)
  - Threshold T_alpha = quantile_{1-alpha}(side_max)  (per-fold)
  - Apply T_alpha as gating threshold on test set; pred=2 if p2>=T_alpha & p2>p0,
    pred=0 if p0>=T_alpha & p0>p2, else 1.

This is a pseudo-conformal coverage construction — guarantees roughly alpha
fraction of "active" predictions on the calibration set. The OOF estimate of
sum_cum_pnl on the test fold is the deliverable; sweep alpha and pick best.

Note: this still respects sym-agnostic + stateless contract — at inference,
T_alpha is a single scalar baked into the predictor (no state, no sym
lookup, no date).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from _common import N_FOLDS, fold_arrays, gate_asymmetric, load_all_folds, vectorized_pnl

HERE = os.path.dirname(os.path.abspath(__file__))


def conformal_threshold(probs_calib: np.ndarray, alpha: float) -> float:
    side_max = np.maximum(probs_calib[:, 0], probs_calib[:, 2])
    return float(np.quantile(side_max, 1.0 - alpha))


def evaluate_alpha(fas, alpha: float, asym: bool = False) -> dict:
    """For each fold use other 4 folds as calibration."""
    per = []
    chosen_T = []
    for k in range(N_FOLDS):
        calib = np.concatenate([fas[j].probs for j in range(N_FOLDS) if j != k])
        if not asym:
            T = conformal_threshold(calib, alpha)
            chosen_T.append((float(T), float(T)))
            pred = gate_asymmetric(fas[k].probs, T, T, 0.0, 0.0)
        else:
            T_up = float(np.quantile(calib[:, 2], 1.0 - alpha))
            T_dn = float(np.quantile(calib[:, 0], 1.0 - alpha))
            chosen_T.append((T_up, T_dn))
            pred = gate_asymmetric(fas[k].probs, T_up, T_dn, 0.0, 0.0)
        s = float(vectorized_pnl(pred, fas[k].label, fas[k].mp_t, fas[k].mp_th).sum())
        per.append(s)
    return {
        "alpha": alpha,
        "asym": asym,
        "sum_cum_pnl": float(sum(per)),
        "per_fold_pnl": per,
        "per_fold_T": chosen_T,
        "n_pos_folds": int(sum(1 for x in per if x > 0)),
        "std_cum_pnl": float(np.std(per, ddof=0)),
    }


def main():
    t0 = time.time()
    folds = load_all_folds()
    fas = fold_arrays(folds)

    alphas = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    results = []
    print("symmetric (single-T) conformal:")
    for a in alphas:
        r = evaluate_alpha(fas, a, asym=False)
        results.append(r)
        print(f"  alpha={a:.2f}: sum={r['sum_cum_pnl']:+.4f}, T_per_fold="
              f"{[(round(t[0],3)) for t in r['per_fold_T']]}, "
              f"per_fold={[round(x,3) for x in r['per_fold_pnl']]}, "
              f"pos={r['n_pos_folds']}/5")

    print("\nasymmetric (T_up != T_dn) conformal:")
    for a in alphas:
        r = evaluate_alpha(fas, a, asym=True)
        results.append(r)
        print(f"  alpha={a:.2f}: sum={r['sum_cum_pnl']:+.4f}, "
              f"per_fold={[round(x,3) for x in r['per_fold_pnl']]}, "
              f"pos={r['n_pos_folds']}/5")

    best = max(results, key=lambda r: r["sum_cum_pnl"])
    print(f"\nBest conformal: alpha={best['alpha']}, asym={best['asym']}, "
          f"sum={best['sum_cum_pnl']:+.4f}")

    out = {
        "task": "T30 split-conformal threshold sweep",
        "alphas": alphas,
        "results": results,
        "best": best,
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "conformal_results.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
