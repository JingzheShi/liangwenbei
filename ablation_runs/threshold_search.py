"""Rerun threshold search on saved preds with proper grid scaling.

For both regression and classification: auto-scale grid to predictions percentile range.
Outputs sym/asym best PnL for h=60.
"""
import argparse, json, os
import numpy as np


def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=1e-4):
    a = np.zeros_like(y_pred)
    a[y_pred > thr_up] = 1.0
    a[y_pred < -thr_dn] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl = (raw - cost) / (mp_t + 1.0)
    n_act = int((a != 0).sum())
    return float(pnl.sum()), n_act


def search_2d(y, mp_t, mp_th, grid):
    best = (-1e18, None, None, 0)
    for up in grid:
        for dn in grid:
            s, na = compute_pnl(y, mp_t, mp_th, up, dn)
            if s > best[0]:
                best = (s, float(up), float(dn), na)
    return best


def search_sym(y, mp_t, mp_th, scale=None):
    if scale is None:
        scale = float(np.mean(np.abs(y)))
    if scale <= 0:
        scale = 1e-5
    best = (-1e18, None, None, 0)
    for k in np.linspace(0.05, 3.0, 60):
        thr = k * scale
        s, na = compute_pnl(y, mp_t, mp_th, thr, thr)
        if s > best[0]:
            best = (s, float(thr), float(k), na)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    d = np.load(args.preds)
    yp_val = d["yp_val"].astype(np.float64)
    mp_t_v = d["mp_t_val"].astype(np.float64)
    mp_th_v = d["mp_th_val"].astype(np.float64)
    yp_test = d["yp_test"].astype(np.float64)
    mp_t_t = d["mp_t_test"].astype(np.float64)
    mp_th_t = d["mp_th_test"].astype(np.float64)

    pred_abs_q = np.percentile(np.abs(yp_val), [50, 70, 85, 95, 99])
    print(f"abs(yp_val) percentiles [50,70,85,95,99] = {pred_abs_q}")
    # asym grid: from 0.3*median to 1.5*p95
    lo = max(1e-7, 0.3 * pred_abs_q[0])
    hi = max(lo * 2, 1.5 * pred_abs_q[3])
    grid = np.linspace(lo, hi, 31)
    print(f"grid lo={lo:.4e} hi={hi:.4e} n={len(grid)}")

    sym_s, sym_thr, sym_k, sym_na = search_sym(yp_val, mp_t_v, mp_th_v)
    asym_s, up, dn, asym_na = search_2d(yp_val, mp_t_v, mp_th_v, grid)

    sym_test_s, sym_test_na = compute_pnl(yp_test, mp_t_t, mp_th_t, sym_thr, sym_thr)
    asym_test_s, asym_test_na = compute_pnl(yp_test, mp_t_t, mp_th_t, up, dn)

    res = {
        "name": args.name,
        "abs_pred_percentiles": pred_abs_q.tolist(),
        "val": {
            "sym_pnl": sym_s, "sym_thr": sym_thr, "sym_k": sym_k, "sym_n_active": sym_na,
            "asym_pnl": asym_s, "asym_thr_up": up, "asym_thr_dn": dn, "asym_n_active": asym_na,
        },
        "test_at_val_thr": {
            "sym_pnl": sym_test_s, "sym_n_active": sym_test_na,
            "asym_pnl": asym_test_s, "asym_n_active": asym_test_na,
        },
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
