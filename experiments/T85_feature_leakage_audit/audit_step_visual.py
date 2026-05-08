"""Visualize step-signal response for representative features.

For a few prominent feature classes (dual_z, qrank, signed_rv, kyle_lam,
ewma_ofi, signed_bv, spread_reg, vol_burst, mlofi), print the feature
value as a function of step position s where:
  rows < s : LOW
  rows >= s: HIGH

A causal feature with sub-window W must satisfy:
  - feature(s) is constant for s in [0, 100-W]   (step entirely in past part
    that's outside the last-W slice → last W is all HIGH; transitions for
    small s only matter for full-window features)
  - feature(s) varies in [100-W, 100]            (step inside the last-W slice)
  - feature(s=100) (no step, all LOW) = feature applied to constant LOW
"""
from __future__ import annotations
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

with open(os.path.join(HERE, "audit_results.json")) as f:
    r = json.load(f)

fn = r["feat_names"]
traces = np.array(r["test2"]["feat_traces"])  # (101, NF)

# Representative features to visualize
SAMPLES = [
    "rv_w5", "rv_w20", "rv_w50",
    "dualz_midprice1", "dualz_spread1",
    "qrank_W100_midprice", "qrank_W100_amount_delta",
    "signed_rv_W20", "signed_rv_W100",
    "kyle_lam_W50", "kyle_lam_W100",
    "rskew_W20", "rskew_W50",
    "ewma_ofi_a0.1_lvl1", "ewma_a0.1_lb_intst",
    "signed_bv_W20", "signed_bv_W100",
    "spread_reg_W20", "spread_reg_W50",
    "vol_burst_W20", "vol_burst_W50",
    "trade_pers_W20", "trade_pers_W50",
    "adapt_mom_W20", "adapt_mom_W100",
    "mlofi_W5_lvl1", "mlofi_W20_lvl1", "mlofi_W60_lvl1",
    "gofi_W5_lvl1", "gofi_W60_lvl1",
    "jshare_W20", "jshare_W100",
    "cancel_imb_W20", "cancel_imb_W100",
    "roll_eff_spr_ratio_W30", "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5", "liq_asym_top5_W100",
]
W_MAP = {  # declared sub-window for each feature class (used to mark expected plateau)
    "rv_w5": 5, "rv_w20": 20, "rv_w50": 50,
    "dualz_midprice1": 100, "dualz_spread1": 100,  # uses both 20 & 100
    "qrank_W100_midprice": 100, "qrank_W100_amount_delta": 100,
    "signed_rv_W20": 20, "signed_rv_W100": 100,
    "kyle_lam_W50": 50, "kyle_lam_W100": 100,
    "rskew_W20": 20, "rskew_W50": 50,
    "ewma_ofi_a0.1_lvl1": 999, "ewma_a0.1_lb_intst": 999,  # EWMA = full window
    "signed_bv_W20": 20, "signed_bv_W100": 100,
    "spread_reg_W20": 20, "spread_reg_W50": 50,
    "vol_burst_W20": 20, "vol_burst_W50": 50,
    "trade_pers_W20": 20, "trade_pers_W50": 50,
    "adapt_mom_W20": 20, "adapt_mom_W100": 100,
    "mlofi_W5_lvl1": 5, "mlofi_W20_lvl1": 20, "mlofi_W60_lvl1": 60,
    "gofi_W5_lvl1": 5, "gofi_W60_lvl1": 60,
    "jshare_W20": 20, "jshare_W100": 100,
    "cancel_imb_W20": 20, "cancel_imb_W100": 100,
    "roll_eff_spr_ratio_W30": 30, "roll_eff_spr_ratio_W100": 100,
    "liq_asym_top5_W5": 5, "liq_asym_top5_W100": 100,
}
name_to_idx = {n: i for i, n in enumerate(fn)}


def causal_check(name: str) -> str:
    """Verify that the feature is invariant to step position s when s is small
    (step entirely outside last-W). Returns a one-char verdict.

    Convention: s = step position. rows[<s]=LOW, rows[>=s]=HIGH. For W,
    last W ticks are rows [100-W, 99]. If s <= 100-W, all last-W rows are
    HIGH so feature value should be constant for s in [0, 100-W].
    """
    W = W_MAP[name]
    if W >= 100:
        return "n/a"  # full-window or EWMA — sensitive to all rows
    idx = name_to_idx[name]
    plateau_region = traces[:max(0, 100 - W) + 1, idx]
    var = float(np.var(plateau_region))
    rng = float(plateau_region.max() - plateau_region.min())
    return f"plateau range[s∈0..{100-W}]={rng:.3e}"


print(f"{'feature':<35} {'W':>4} {'@s=0':>10} {'@s=80':>10} {'@s=99':>10} {'@s=100':>10} {'plateau_check':<35}")
print("-" * 120)
for name in SAMPLES:
    if name not in name_to_idx:
        print(f"  [WARN] {name} not in feature names")
        continue
    idx = name_to_idx[name]
    W = W_MAP[name]
    v0 = traces[0, idx]    # all HIGH
    v80 = traces[80, idx]  # last 20 are HIGH
    v99 = traces[99, idx]  # only last tick HIGH
    v100 = traces[100, idx]  # all LOW
    chk = causal_check(name)
    print(f"{name:<35} {W:>4} {v0:>10.4f} {v80:>10.4f} {v99:>10.4f} {v100:>10.4f} {chk}")

# Now: compute the "first-s-where-feature-changes-from-baseline (s=100, all-LOW)"
# This locates the EFFECTIVE leftmost row of dependence.
# Causal feature with sub-window W → first changing s_change = 99 - W + 1 (or earlier
# if there's lag-1 dependence: s_change = 99 - W).
print("\nLEFTMOST step position s* such that feature(s*) != feature(s=99) (i.e. last W edge):")
print(f"{'feature':<35} {'declared W':>10} {'derived s*':>10} {'expected (lag1)':>15}")
print("-" * 80)
for name in SAMPLES:
    if name not in name_to_idx:
        continue
    idx = name_to_idx[name]
    W = W_MAP[name]
    base = traces[100, idx]  # all LOW
    # walk s from 100 downward; first s where trace differs from base by > tol
    s_first = None
    tol = 1e-9
    for s in range(100, -1, -1):
        if abs(traces[s, idx] - base) > tol:
            s_first = s
            break
    expected = 99 - W if W < 100 else 0
    print(f"{name:<35} {W:>10} {str(s_first):>10} {expected:>15}")
