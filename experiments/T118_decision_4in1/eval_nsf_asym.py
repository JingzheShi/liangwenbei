"""T118 follow-up: asymmetric NSF using DE thresholds as base.

NSF symmetric form lost +0.5 LOSO because it forced thr_dn = thr_up = fee_eff.
Asym form: pred > thr_up + gamma*spread → long; pred < -(thr_dn + gamma*spread) → short.
This is the natural sym-extension that preserves the DE-tuned asym structure.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_4in1 import (
    build_4way_pred, eval_actions, baseline_de, compute_pnl,
    THR_UP_BASE, THR_DN_BASE, ROOT,
)

HERE = Path("/root/projects/liangwenbei_workdir/experiments/T118_decision_4in1")


def rule_nsf_asym(p_comb, spread_t, thr_up_base, thr_dn_base, gamma):
    actions = np.full(len(p_comb), 1, dtype=np.int8)
    thr_up = thr_up_base + gamma * spread_t
    thr_dn = thr_dn_base + gamma * spread_t
    actions[p_comb > thr_up] = 2
    actions[p_comb < -thr_dn] = 0
    return actions


def main():
    p_comb, meta, sym = build_4way_pred()
    n = len(p_comb)
    schemeP = np.load(ROOT / "experiments/T68_stage5_features/cache/schemeP_test.npz")
    feat_names_path = ROOT / "experiments/T68_stage5_features/cache/schemeP_feat_names.txt"
    with open(feat_names_path) as f:
        all_names = [l.strip() for l in f]
    spread_t = (schemeP["X"][:, all_names.index("ask1")]
                - schemeP["X"][:, all_names.index("bid1")]).astype(np.float64)

    print(f"baseline thr_up={THR_UP_BASE:.6e} thr_dn={THR_DN_BASE:.6e}", flush=True)

    print("\n=== NSF asymmetric (DE base + gamma*spread) ===", flush=True)
    rows = []
    for g in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        a = rule_nsf_asym(p_comb, spread_t, THR_UP_BASE, THR_DN_BASE, g)
        r = eval_actions(a, meta, sym, f"nsf_asym_g{g:.2f}")
        r["gamma"] = g
        rows.append(r)
        print(f"  g={g:.2f}: loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  "
              f"rate={r['active_rate']:.4f}  per_sym={[f'{x:+.3f}' for x in r['per_sym']]}",
              flush=True)

    # also: scale only thr_up by spread (asymmetric only on upside)
    print("\n=== NSF asymmetric (gamma applied only to long side) ===", flush=True)
    upside_rows = []
    for g in [0.0, 0.10, 0.20, 0.30]:
        actions = np.full(n, 1, dtype=np.int8)
        thr_up = THR_UP_BASE + g * spread_t
        actions[p_comb > thr_up] = 2
        actions[p_comb < -THR_DN_BASE] = 0
        r = eval_actions(actions, meta, sym, f"nsf_upside_g{g:.2f}")
        r["gamma"] = g
        upside_rows.append(r)
        print(f"  g={g:.2f}: loso={r['loso']:+.4f}  per_sym_min={r['per_sym_min']:+.4f}  "
              f"rate={r['active_rate']:.4f}", flush=True)

    out = {
        "baseline_thresholds": {"thr_up": THR_UP_BASE, "thr_dn": THR_DN_BASE},
        "nsf_asym_results": rows,
        "nsf_upside_only": upside_rows,
    }
    out_path = HERE / "results_nsf_asym.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
