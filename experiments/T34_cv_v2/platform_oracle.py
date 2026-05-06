"""
T34.6 — Platform Oracle: integrate LO2SO + per-trade gate + perturbation +
calibration + bootstrap into a single dataframe.

For any (model, horizon) with OOF available, output:

    iter | horizon | LOSO_sum | LOSO_per_trade | LO2SO_mean | LO2SO_min |
    brittleness | platform_pred_M5 | bootstrap_ci_2.5 | bootstrap_ci_97.5

Recommendation engine:
    flag (model, horizon) as "platform-likely-positive" iff:
      - LOSO sum > 0
      - bootstrap p_positive >= 0.95
      - LO2SO min pair > -3.0 (worst case is bounded)
      - platform_pred_M5 > 0

This is the canonical scorecard. Each iter, run this to decide what to submit.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oof_io import HORIZONS, load_oof, cum_pnl_metrics  # noqa: E402

# Reuse computations from previous scripts via JSON
LO2SO_RESULTS = os.path.join(HERE, "lo2so_results.json")
BOOT_RESULTS = os.path.join(HERE, "bootstrap_ci_results.json")
PERT_RESULTS = os.path.join(HERE, "perturbation_stress_results.json")
CALIB_RESULTS = os.path.join(HERE, "calibration_regression_results.json")


def _safe_load(p):
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def make_oracle_row(model: str, horizon: int) -> Dict:
    """For a (model, horizon), pull together all metrics."""
    df = load_oof(model, horizon)
    base = cum_pnl_metrics(df)
    lo2so = _safe_load(LO2SO_RESULTS).get(f"{model}_h{horizon}", {})
    boot = _safe_load(BOOT_RESULTS).get(f"{model}_h{horizon}", {})
    pert = _safe_load(PERT_RESULTS).get(f"{model}_h{horizon}_shrinkage", {})
    calib = _safe_load(CALIB_RESULTS)
    cal_preds = {p["horizon"]: p for p in calib.get("iter_005b_predictions", [])}

    row = {
        "model": model,
        "horizon": horizon,
        "loso_sum": round(base["cum_pnl"], 4),
        "loso_per_trade": round(base["per_trade_pnl"], 6),
        "loso_n_active": int(base["n_active"]),
        "lo2so_mean": round(lo2so.get("lo2so_mean", float("nan")), 4) if lo2so else None,
        "lo2so_min_pair": round(lo2so.get("lo2so_min_pair", float("nan")), 4) if lo2so else None,
        "lo2so_max_pair": round(lo2so.get("lo2so_max_pair", float("nan")), 4) if lo2so else None,
        "brittleness@0.2": round(pert.get("brittleness_at_0.2", float("nan")), 3) if pert else None,
        "boot_ci_2.5": round(boot.get("ci_2.5", float("nan")), 3) if boot else None,
        "boot_ci_97.5": round(boot.get("ci_97.5", float("nan")), 3) if boot else None,
        "boot_p_pos": round(boot.get("p_positive", float("nan")), 3) if boot else None,
    }

    # Platform prediction: M5 (anchor on iter_002 shift). This is for iter_005b only;
    # for other models, use iter_002 shift as anchor for new LightGBM-class iters.
    if model == "iter_005b" and horizon in cal_preds:
        row["platform_pred_M5"] = round(cal_preds[horizon].get("M5_pred", float("nan")), 3)
        row["platform_pred_M_robust"] = round(cal_preds[horizon].get("M_robust", float("nan")), 3)
    else:
        row["platform_pred_M5"] = None
        row["platform_pred_M_robust"] = None

    return row


def recommend(row: Dict) -> str:
    """Trade/no-trade recommendation."""
    if row["loso_sum"] is None or row["loso_sum"] <= 0:
        return "DO_NOT_USE"
    if row.get("boot_p_pos") is not None and row["boot_p_pos"] < 0.95:
        return "RISKY (bootstrap p_pos < 0.95)"
    if row.get("lo2so_min_pair") is not None and row["lo2so_min_pair"] < -3.0:
        return f"BRITTLE (LO2SO min pair {row['lo2so_min_pair']} < -3)"
    if row.get("platform_pred_M5") is not None and row["platform_pred_M5"] <= 0:
        return f"PLATFORM-NEGATIVE (M5 pred {row['platform_pred_M5']})"
    return "GREEN"


def main():
    rows = []
    targets = (
        [("iter_002", h) for h in HORIZONS]
        + [("iter_005b", h) for h in HORIZONS]
        + [("t26_aug_a", 60), ("t26_baseline", 60)]
    )
    for model, h in targets:
        try:
            row = make_oracle_row(model, h)
            row["recommend"] = recommend(row)
            rows.append(row)
        except (FileNotFoundError, ValueError) as e:
            print(f"[oracle] skip {model} h={h}: {e}")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # Save
    out_csv = os.path.join(HERE, "platform_oracle.csv")
    out_json = os.path.join(HERE, "platform_oracle.json")
    df.to_csv(out_csv, index=False)
    df.to_json(out_json, orient="records", indent=2)
    print(f"\n[oracle] saved {out_csv}")
    print(f"[oracle] saved {out_json}")

    # Print iter_005b summary specifically
    print("\n=== iter_005b platform-score prediction summary ===")
    sub = df[df["model"] == "iter_005b"]
    print(sub.to_string(index=False))
    if not sub.empty:
        # Best horizon by M5 prediction
        sub_active = sub[sub["loso_sum"] > 0].copy()
        if not sub_active.empty:
            best = sub_active.loc[sub_active["platform_pred_M5"].idxmax()]
            ci_lo = best["boot_ci_2.5"]
            ci_hi = best["boot_ci_97.5"]
            print(f"\n  → BEST horizon = h_{best['horizon']}")
            print(f"     LOSO point estimate = {best['loso_sum']:+.3f}  (95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}])")
            print(f"     M5 platform prediction (anchor on iter_002) = {best['platform_pred_M5']:+.2f}")
            print(f"     Recommendation: {best['recommend']}")


if __name__ == "__main__":
    main()
