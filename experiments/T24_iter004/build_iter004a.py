"""T24 iter_004a (safe) builder.

Strategy: copy iter_002 verbatim, but mark h_5/h_10/h_20 as inactive in
thresholds.json. Predictor.predict already handles `active=false` by
leaving the column at the default value of 1 (no trade). This guarantees
short horizons contribute 0 to the platform "best of 5" score, so the
final score is floored at the iter_002 h_60 result (+4.07).

Usage:
    python experiments/T24_iter004/build_iter004a.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "submission", "iter_002_lgbm_schemeC")
DST = os.path.join(ROOT, "submission", "iter_004a_lgbm_schemeC_safe")

THRESHOLDS = {
    5:  {"T": 0.99, "delta": 0.99, "active": False},
    10: {"T": 0.99, "delta": 0.99, "active": False},
    20: {"T": 0.99, "delta": 0.99, "active": False},
    40: {"T": 0.50, "delta": 0.00, "active": True},
    60: {"T": 0.50, "delta": 0.20, "active": True},
}

REASONS = {
    5:  "iter_002 platform = -7.68, per-trade ~ -fee. Disabling avoids drag.",
    10: "iter_002 platform = -8.64, per-trade ~ -fee. Disabling avoids drag.",
    20: "iter_002 platform = -5.09, per-trade slightly negative. Disable.",
    40: "iter_002 platform = +2.02, per-trade > 0. Keep unchanged.",
    60: "iter_002 platform = +4.07 (BEST), per-trade = +0.000117. Keep.",
}


def main():
    os.makedirs(DST, exist_ok=True)
    for fn in os.listdir(SRC):
        s = os.path.join(SRC, fn)
        if os.path.isfile(s):
            shutil.copy(s, os.path.join(DST, fn))
    # Read iter_002 thresholds for LOSO fields
    with open(os.path.join(SRC, "thresholds.json")) as f:
        src_t = json.load(f)
    by_h = {int(r["h"]): r for r in src_t["horizons"]}

    horizons_out = []
    for h in (5, 10, 20, 40, 60):
        cfg = THRESHOLDS[h]
        src_row = by_h[h]
        horizons_out.append({
            "h": h,
            "T": cfg["T"], "delta": cfg["delta"], "active": cfg["active"],
            "_reason": REASONS[h],
            "loso_best_sum": src_row.get("loso_best_sum"),
            "loso_n_pos_folds": src_row.get("loso_n_pos_folds"),
        })

    out_path = os.path.join(DST, "thresholds.json")
    with open(out_path, "w") as f:
        json.dump({
            "_doc": "iter_004a (safe): short horizons inactive (output 1). Floors short-horizon platform PnL at 0; long horizons (h_40, h_60) preserve iter_002 +2.02 / +4.07.",
            "_source": "iter_002 with h_5/10/20 active=false.",
            "horizons": horizons_out,
        }, f, indent=2)
    print(f"thresholds -> {out_path}")
    for h in horizons_out:
        flag = "ACTIVE" if h["active"] else "  flat"
        print(f"  h={h['h']:2d}: {flag} (T={h['T']:.2f} d={h['delta']:.2f})")


if __name__ == "__main__":
    main()
