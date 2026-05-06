"""T24 iter_004b (aggressive) builder.

Strategy: copy iter_002, then tighten thresholds on h_5/h_10/h_20 to maximise
LOSO per-trade PnL while keeping all 5 folds positive. Long horizons
(h_40, h_60) keep iter_002 settings.

Operating points come from `sweep_aggressive.py`:
  h_5:  T=0.85 d=0.20 -> sum +7.55, per_trade +0.000643, n_act 11743
  h_10: T=0.80 d=0.20 -> sum +6.46, per_trade +0.000702, n_act  9198
  h_20: T=0.70 d=0.20 -> sum +6.71, per_trade +0.000647, n_act 10376

Usage:
    python experiments/T24_iter004/sweep_aggressive.py   # generate sweep
    python experiments/T24_iter004/build_iter004b.py     # build package
"""
from __future__ import annotations

import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "submission", "iter_002_lgbm_schemeC")
DST = os.path.join(ROOT, "submission", "iter_004b_lgbm_schemeC_aggressive")

THRESHOLDS = {
    5:  {"T": 0.85, "delta": 0.20, "active": True,
         "loso_best_sum": 7.554, "loso_per_trade_pnl": 0.000643,
         "loso_n_pos_folds": 5, "loso_n_active": 11743},
    10: {"T": 0.80, "delta": 0.20, "active": True,
         "loso_best_sum": 6.455, "loso_per_trade_pnl": 0.000702,
         "loso_n_pos_folds": 5, "loso_n_active": 9198},
    20: {"T": 0.70, "delta": 0.20, "active": True,
         "loso_best_sum": 6.712, "loso_per_trade_pnl": 0.000647,
         "loso_n_pos_folds": 5, "loso_n_active": 10376},
    40: {"T": 0.50, "delta": 0.00, "active": True},
    60: {"T": 0.50, "delta": 0.20, "active": True},
}


def main():
    os.makedirs(DST, exist_ok=True)
    for fn in os.listdir(SRC):
        s = os.path.join(SRC, fn)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(DST, fn))
    with open(os.path.join(SRC, "thresholds.json")) as f:
        src_t = json.load(f)
    by_h = {int(r["h"]): r for r in src_t["horizons"]}

    horizons_out = []
    for h in (5, 10, 20, 40, 60):
        cfg = THRESHOLDS[h]
        src_row = by_h[h]
        row = {"h": h, "T": cfg["T"], "delta": cfg["delta"], "active": cfg["active"]}
        if h in (5, 10, 20):
            row.update({
                "loso_best_sum": cfg["loso_best_sum"],
                "loso_per_trade_pnl": cfg["loso_per_trade_pnl"],
                "loso_n_pos_folds": cfg["loso_n_pos_folds"],
                "loso_n_active": cfg["loso_n_active"],
            })
        else:
            row["loso_best_sum"] = src_row.get("loso_best_sum")
            row["loso_n_pos_folds"] = src_row.get("loso_n_pos_folds")
        horizons_out.append(row)

    out_path = os.path.join(DST, "thresholds.json")
    with open(out_path, "w") as f:
        json.dump({
            "_doc": "iter_004b (aggressive): tight thresholds on h_5/h_10/h_20 selected for highest LOSO per-trade PnL while keeping all 5 folds positive.",
            "_source": "T24_iter004 sweep_aggressive_summary.json (T_grid 0.50-0.85, delta_grid 0.0-0.40).",
            "horizons": horizons_out,
        }, f, indent=2)
    print(f"thresholds -> {out_path}")
    for h in horizons_out:
        print(f"  h={h['h']:2d}: T={h['T']:.2f} d={h['delta']:.2f}")


if __name__ == "__main__":
    main()
