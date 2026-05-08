"""DE asymmetric thresh eval: T75 LGB L2 baseline vs MILD-trick (advval cap [0.5,2.0]).

Loads:
  baseline:    /root/lwb_remote_pkg/preds/pred_T75_seed{42,7,13}.parquet (avg)
  mild trick:  /root/lwb_work_t75_advval/pred_T75L2_mild_seed{42,7,13}.parquet (avg)

Output: results_mild.json  with both label-tagged.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/root/lwb_work_t75_advval")
from de_eval_advval import (
    SEEDS, SYMS, FEE,
    vectorized_pnl, ev_gate_asymmetric, split_by_sym,
    make_obj_loso_asym, de_search, per_sym_pnl_at, load_avg, evaluate,
)
HERE = "/root/lwb_work_t75_advval"
BASELINE_DIR = "/root/lwb_remote_pkg/preds"


def main():
    base_paths = [os.path.join(BASELINE_DIR, f"pred_T75_seed{s}.parquet")
                  for s in SEEDS]
    mild_paths = [os.path.join(HERE, f"pred_T75L2_mild_seed{s}.parquet")
                  for s in SEEDS]
    df_base = load_avg(base_paths)
    df_mild = load_avg(mild_paths)
    print(f"  baseline pred mean={df_base['pred_dmid_norm'].mean():.6e} std={df_base['pred_dmid_norm'].std():.6e}", flush=True)
    print(f"  mild     pred mean={df_mild['pred_dmid_norm'].mean():.6e} std={df_mild['pred_dmid_norm'].std():.6e}", flush=True)
    res_base = evaluate(df_base, f"BASELINE (T75 LGB L2 3-seed avg {SEEDS})")
    res_mild = evaluate(df_mild, f"MILD trick (advval cap [0.5,2.0])")

    delta = res_mild["total_loso_equiv"] - res_base["total_loso_equiv"]
    delta_per_sym = [t - b for t, b in zip(res_mild["per_sym_pnl"], res_base["per_sym_pnl"])]
    print(f"\n=== DELTA (mild - baseline) ===", flush=True)
    print(f"  per-sym: {[f'{v:+.4f}' for v in delta_per_sym]}", flush=True)
    print(f"  total = {delta:+.4f}  min_per_sym = {min(delta_per_sym):+.4f}",
          flush=True)
    summary = {
        "task": "T75 LGB L2 advval-MILD reweight (cap [0.5,2.0]) (3 seeds 42/7/13)",
        "seeds": list(SEEDS),
        "baseline": res_base,
        "trick_mild": res_mild,
        "delta_total_loso": delta,
        "delta_per_sym": delta_per_sym,
        "delta_min_per_sym": float(min(delta_per_sym)),
    }
    with open(os.path.join(HERE, "results_mild.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {os.path.join(HERE, 'results_mild.json')}", flush=True)


if __name__ == "__main__":
    main()
