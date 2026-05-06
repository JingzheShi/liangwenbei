"""
T34.4 — Per-horizon calibration regression: predict platform_score from
LOSO-derived metrics.

Known data points (LOSO sum, platform sum, n_active, per_trade_LOSO, horizon, brittleness):

    (mmpc_demo, h_5)  loso=+4.09  platform=-9.08
    (mmpc_demo, h_10) loso=+6.16  platform=-13.32
    (mmpc_demo, h_20) loso=+6.33  platform=-6.65
    (mmpc_demo, h_40) loso=+6.27  platform=-16.07
    (mmpc_demo, h_60) loso=+3.57  platform=-23.13
    (iter_002,  h_5)  loso=+17.49 platform=-7.68
    (iter_002,  h_10) loso=+21.86 platform=-8.64
    (iter_002,  h_20) loso=+19.70 platform=-5.09
    (iter_002,  h_40) loso=+11.71 platform=+2.02
    (iter_002,  h_60) loso=+6.30  platform=+4.07

Models tried (in order of complexity):

    M1 (per-horizon shift):    platform = loso + shift_h
    M2 (per-horizon linear):   platform = a_h * loso + b_h     (univariate)
    M3 (multivariate global):  platform = α*loso + β*per_trade + γ*log(n_active) + δ*h
                               (single global fit, no per-horizon coefs)
    M4 (M3 + brittleness):     M3 + ε * brittleness@0.2

Validation: leave-one-out cross validation, R² + RMSE on held-out platform predictions.

NOTE on n_active: mmpc_demo n_active is from FULL local test (442k rows), but
iter_002 LOSO n_active is from 5×88416 = 442k OOF rows — so they're directly
comparable in scale. Good. We pass log(n_active) to compress dynamic range.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oof_io import HORIZONS, load_oof, cum_pnl_metrics  # noqa: E402

PLATFORM_DATA = {
    # (model, horizon): (loso_cum, loso_n_active, loso_per_trade, platform_cum)
    ("mmpc_demo", 5):  (4.0922, 25097, 1.6306e-4, -9.079),
    ("mmpc_demo", 10): (6.1549, 65019, 9.466e-5, -13.319),
    ("mmpc_demo", 20): (6.3339, 19725, 3.211e-4, -6.649),
    ("mmpc_demo", 40): (6.2723, 52124, 1.2034e-4, -16.065),
    ("mmpc_demo", 60): (3.5725, 112629, 3.172e-5, -23.132),
    ("iter_002", 5):   (17.4917, 80454, 2.174e-4, -7.68),
    ("iter_002", 10):  (21.8595, 97135, 2.250e-4, -8.64),
    ("iter_002", 20):  (19.6999, 88106, 2.236e-4, -5.09),
    ("iter_002", 40):  (11.7051, 74883, 1.563e-4, 2.02),
    ("iter_002", 60):  (6.3002, 55288, 1.140e-4, 4.07),
}


def fit_per_horizon_shift(rows: List[Dict]) -> Dict[int, float]:
    """M1: per-horizon shift. shift_h = mean(platform - loso) over models for that h."""
    by_h: Dict[int, List[float]] = {}
    for r in rows:
        by_h.setdefault(r["horizon"], []).append(r["platform"] - r["loso"])
    return {h: float(np.mean(s)) for h, s in by_h.items()}


def fit_per_horizon_linear(rows: List[Dict]) -> Dict[int, Tuple[float, float]]:
    """M2: per-horizon least-squares fit platform = a*loso + b."""
    out = {}
    for h in sorted({r["horizon"] for r in rows}):
        sub = [r for r in rows if r["horizon"] == h]
        if len(sub) < 2:
            # Only 1 data point — fall back to shift
            r0 = sub[0]
            out[h] = (1.0, r0["platform"] - r0["loso"])
            continue
        x = np.array([r["loso"] for r in sub])
        y = np.array([r["platform"] for r in sub])
        a, b = np.polyfit(x, y, 1)
        out[h] = (float(a), float(b))
    return out


def fit_multivariate(rows: List[Dict], features: List[str]) -> Dict:
    """M3/M4: global linear regression with intercept."""
    X = np.array([[r[f] for f in features] for r in rows])
    y = np.array([r["platform"] for r in rows])
    # Add intercept
    X1 = np.hstack([X, np.ones((X.shape[0], 1))])
    # Least squares (with regularization for stability — small ridge)
    XtX = X1.T @ X1 + 1e-6 * np.eye(X1.shape[1])
    coef = np.linalg.solve(XtX, X1.T @ y)
    return {
        "features": features,
        "coef": [float(c) for c in coef[:-1]],
        "intercept": float(coef[-1]),
    }


def predict_multivariate(model: Dict, row: Dict) -> float:
    val = model["intercept"]
    for f, c in zip(model["features"], model["coef"]):
        val += c * row[f]
    return val


def predict_per_horizon_shift(model: Dict[int, float], row: Dict) -> float:
    return row["loso"] + model[row["horizon"]]


def predict_per_horizon_linear(model: Dict[int, Tuple[float, float]], row: Dict) -> float:
    a, b = model[row["horizon"]]
    return a * row["loso"] + b


def loo_cv(rows: List[Dict], fit_fn, predict_fn) -> Dict:
    """Leave-one-out CV. Returns predictions, R², RMSE, mean abs err."""
    preds = []
    for i in range(len(rows)):
        train = [r for j, r in enumerate(rows) if j != i]
        test = rows[i]
        m = fit_fn(train)
        try:
            pred = predict_fn(m, test)
        except KeyError:
            # If horizon not seen in train (per-horizon model with single point)
            pred = test["loso"]
        preds.append({"loso": test["loso"], "platform_true": test["platform"],
                      "platform_pred": float(pred), "model": test["model"],
                      "horizon": test["horizon"]})

    y_true = np.array([p["platform_true"] for p in preds])
    y_pred = np.array([p["platform_pred"] for p in preds])
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mae = float(np.abs(y_true - y_pred).mean())
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"loo_predictions": preds, "rmse": rmse, "mae": mae, "r2": r2}


def get_brittleness(model: str, h: int) -> float:
    """Read brittleness@0.2 from perturbation_stress_results.json (shrinkage mode).

    Falls back to NaN if not available (e.g. mmpc_demo).
    """
    p = os.path.join(HERE, "perturbation_stress_results.json")
    if not os.path.exists(p):
        return float("nan")
    with open(p) as f:
        d = json.load(f)
    key = f"{model}_h{h}_shrinkage"
    if key in d:
        return float(d[key]["brittleness_at_0.2"])
    return float("nan")


def main():
    rows = []
    for (model, h), (loso, n_active, per_trade, platform) in PLATFORM_DATA.items():
        rows.append({
            "model": model,
            "horizon": h,
            "loso": loso,
            "n_active": n_active,
            "log_n_active": math.log(n_active),
            "per_trade": per_trade,
            "brittleness": get_brittleness(model, h),
            "platform": platform,
        })

    # M1: per-horizon shift
    print("\n=== M1: Per-horizon shift (platform = loso + shift_h) ===")
    m1 = fit_per_horizon_shift(rows)
    for h in HORIZONS:
        # Show breakdown: mmpc_demo's shift vs iter_002's shift for this h
        mmpc_r = next(r for r in rows if r["model"] == "mmpc_demo" and r["horizon"] == h)
        i002_r = next(r for r in rows if r["model"] == "iter_002" and r["horizon"] == h)
        mmpc_shift = mmpc_r["platform"] - mmpc_r["loso"]
        i002_shift = i002_r["platform"] - i002_r["loso"]
        print(f"  h={h}: mean shift = {m1[h]:+.4f}  (mmpc {mmpc_shift:+.2f}, iter_002 {i002_shift:+.2f})")
    cv1 = loo_cv(rows, fit_per_horizon_shift, predict_per_horizon_shift)
    print(f"  LOO: R²={cv1['r2']:+.3f}  RMSE={cv1['rmse']:.3f}  MAE={cv1['mae']:.3f}")

    # M2: per-horizon linear
    print("\n=== M2: Per-horizon linear (platform = a_h * loso + b_h, only 2 pts/h → exact fit) ===")
    m2 = fit_per_horizon_linear(rows)
    for h in HORIZONS:
        a, b = m2[h]
        print(f"  h={h}: a={a:+.4f}  b={b:+.4f}")
    cv2 = loo_cv(rows, fit_per_horizon_linear, predict_per_horizon_linear)
    print(f"  LOO: R²={cv2['r2']:+.3f}  RMSE={cv2['rmse']:.3f}  MAE={cv2['mae']:.3f}")
    print("  (LOO fails for per-horizon linear since each h has only 2 obs)")

    # M3: multivariate (no brittleness — works for mmpc + iter_002 even without OOF brittleness)
    feats3 = ["loso", "per_trade", "log_n_active", "horizon"]
    print(f"\n=== M3: Multivariate ({feats3}) ===")
    m3 = fit_multivariate(rows, feats3)
    for f, c in zip(m3["features"], m3["coef"]):
        print(f"  {f}: {c:+.6f}")
    print(f"  intercept: {m3['intercept']:+.4f}")
    cv3 = loo_cv(rows, lambda rs: fit_multivariate(rs, feats3), predict_multivariate)
    print(f"  LOO: R²={cv3['r2']:+.3f}  RMSE={cv3['rmse']:.3f}  MAE={cv3['mae']:.3f}")

    # M4: + brittleness — only works if all rows have brittleness. For mmpc we don't.
    # Skip M4 for the cross-model fit (mmpc has no OOF). But fit only on iter_002.
    iter_only = [r for r in rows if r["model"] == "iter_002"]
    feats4 = ["loso", "per_trade", "log_n_active", "horizon", "brittleness"]
    print(f"\n=== M4: + brittleness (iter_002 only, n=5, severely underconstrained) ===")
    print("  (skipped — n=5 with 5 features = exact fit, no LOO meaningful)")

    # Predict iter_005b platform score using best model (M2 = per-horizon linear)
    print("\n=== Predicted iter_005b platform_score ===")
    iter_005b_rows = []
    for h in (40, 60):
        df = load_oof("iter_005b", h)
        m = cum_pnl_metrics(df)
        iter_005b_rows.append({
            "model": "iter_005b",
            "horizon": h,
            "loso": m["cum_pnl"],
            "n_active": m["n_active"],
            "log_n_active": math.log(max(m["n_active"], 1)),
            "per_trade": m["per_trade_pnl"],
            "brittleness": get_brittleness("iter_005b", h),
        })
    # Inactive horizons: pred=1 for h_5/10/20, so cum_pnl=0 directly (no calibration needed)
    for h in (5, 10, 20):
        iter_005b_rows.append({
            "model": "iter_005b",
            "horizon": h,
            "loso": 0.0,
            "n_active": 0,
            "log_n_active": 0.0,
            "per_trade": 0.0,
            "brittleness": float("nan"),
            "_inactive": True,
        })

    # Use M2 per-horizon linear (gives 2 points → 1 line / interpolation) for h=40, 60
    print(f"  {'horizon':<8} {'loso':>8} {'M1_pred':>9} {'M2_pred':>9} {'M3_pred':>9}")
    pred_table = []
    for r in iter_005b_rows:
        if r.get("_inactive"):
            pred_table.append({"horizon": r["horizon"], "loso": 0.0,
                               "M1_pred": 0.0, "M2_pred": 0.0, "M3_pred": 0.0,
                               "_inactive": True})
            print(f"  h_{r['horizon']:<6} {0.0:>+8.2f} {'INACT':>9} {'INACT':>9} {'INACT':>9}")
            continue
        p1 = predict_per_horizon_shift(m1, r)
        p2 = predict_per_horizon_linear(m2, r)
        p3 = predict_multivariate(m3, r)
        pred_table.append({"horizon": r["horizon"], "loso": r["loso"],
                           "M1_pred": p1, "M2_pred": p2, "M3_pred": p3})
        print(f"  h_{r['horizon']:<6} {r['loso']:>+8.2f} {p1:>+9.2f} {p2:>+9.2f} {p3:>+9.2f}")

    # M5 = anchor-based prediction. iter_005b is architecturally close to iter_002
    # (both LightGBM Scheme C 226-d, threshold rule, single model class).
    # mmpc_demo is a fundamentally different zero-alpha DeepLOB baseline.
    # When predicting iter_005b's platform score, the iter_002 shift is a
    # MUCH better anchor than the mmpc_demo shift.
    #
    # M5 prediction: platform = loso + iter_002_shift_for_horizon
    iter_002_shifts_by_h = {
        h: next(r["platform"] - r["loso"] for r in rows
                if r["model"] == "iter_002" and r["horizon"] == h)
        for h in HORIZONS
    }
    print("\n  M5 (anchor on iter_002 shift, since iter_005b shares iter_002's arch):")
    for h, s in iter_002_shifts_by_h.items():
        print(f"    iter_002 shift h={h}: {s:+.4f}")
    for r in pred_table:
        if r.get("_inactive"):
            r["M5_pred"] = 0.0
            continue
        r["M5_pred"] = float(r["loso"] + iter_002_shifts_by_h[r["horizon"]])

    # M5 = robust ensemble: median of M1/M3 (drop M2 since extrapolates badly)
    # We also clip M2 predictions to keep them sane.
    print("\n  Robust ensemble (median of M1, M3) and clipped M2:")
    for r in pred_table:
        if r.get("_inactive"):
            r["M2_clipped"] = 0.0
            r["M_robust"] = 0.0
            continue
        # Clip M2 to ±3× max abs platform observed (~70 cap is way too loose)
        # Better: clip to range of observed platform values [-23.13, +4.07]
        r["M2_clipped"] = float(np.clip(r["M2_pred"], -25.0, 10.0))
        r["M_robust"] = float(np.median([r["M1_pred"], r["M3_pred"]]))
        print(f"    h_{r['horizon']}: M2_clipped={r['M2_clipped']:+.2f}  "
              f"M_robust=median(M1,M3)={r['M_robust']:+.2f}")

    # iter_005b best score (max over horizons under each model)
    print(f"\n  iter_005b best horizon (predicted by various models):")
    for model_key in ("M1_pred", "M2_pred", "M2_clipped", "M3_pred", "M_robust", "M5_pred"):
        best = max(pred_table, key=lambda r: r[model_key])
        print(f"    {model_key:>11}: best h={best['horizon']} → platform_pred={best[model_key]:+.2f}")
    print("\n  M5_pred (anchor on iter_002) is the recommended estimate "
          "for iter_005b given architectural similarity.")

    # Save
    out = {
        "calibration_data": rows,
        "M1_per_horizon_shift": {"shifts": m1, "loo_cv": cv1},
        "M2_per_horizon_linear": {"coefs": {h: list(v) for h, v in m2.items()}, "loo_cv": cv2},
        "M3_multivariate": {"model": m3, "loo_cv": cv3},
        "iter_005b_predictions": pred_table,
    }
    out_path = os.path.join(HERE, "calibration_regression_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
    print(f"\n[calibration] saved {out_path}")


if __name__ == "__main__":
    main()
