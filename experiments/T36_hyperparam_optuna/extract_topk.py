"""Extract top-K Optuna trials by cum_pnl into a JSON config for stage 2 validation."""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", default=os.path.join(HERE, "optuna_summary.json"))
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "top_configs.json"))
    args = ap.parse_args()

    with open(args.summary_json) as f:
        s = json.load(f)
    trials = s["trial_log"]
    trials = [t for t in trials if t["cum_pnl"] > -50]  # drop failures
    trials.sort(key=lambda t: -t["cum_pnl"])
    top = trials[: args.top_k]

    out = []
    for r in top:
        params = dict(r["params"])
        aug_range = float(params.pop("aug_a_range"))
        out.append({
            "name": f"trial_{r['trial_number']}_pnl{r['cum_pnl']:+.3f}",
            "params": params,
            "aug_a_range": aug_range,
            "trial_number": r["trial_number"],
            "held2_cum_pnl": r["cum_pnl"],
            "best_iter_held2": r["best_iter"],
            "best_T_held2": r["best_T"],
            "best_d_held2": r["best_d"],
        })
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"=== top {args.top_k} Optuna trials (held=2 cum_pnl) ===")
    for r in out:
        print(f"  {r['name']}: held2={r['held2_cum_pnl']:+.4f} best_iter={r['best_iter_held2']}")
        print(f"    aug_a_range={r['aug_a_range']:.4f}")
        for k, v in r["params"].items():
            if isinstance(v, float):
                print(f"    {k}={v:.6g}")
            else:
                print(f"    {k}={v}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
