"""Compare T64 strategies + write results.json + REPORT.md."""
from __future__ import annotations

import json
import os
import sys
from glob import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T59_DIR = os.path.join(ROOT, "experiments", "T59_fullsym_train")

PILOT_SEED = 42
SEEDS_5 = [1, 7, 13, 42, 100]


def load_summary(strategy, seed):
    p = os.path.join(HERE, f"summary_{strategy}_seed{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def load_t59_v1():
    """Use T59 full-sym summary for V1 baseline."""
    p = os.path.join(T59_DIR, "full_summary_h60_r34_stage3_fullsym.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def t59_v1_seed42():
    """Parse seed=42 from T59 train_seed42.log (not in summary file)."""
    return {
        "best_iter": 151,
        "test_full_cum_pnl": 16.5731,
        "loso_equiv_sum_argmax": 16.5731,
        "per_sym": {0: 2.22, 1: -1.077, 2: -0.257, 3: 3.409, 4: 12.278},
    }


def t59_v1_5seed():
    """T59 4-seed (no 42) data + manual seed42 → 5-seed."""
    sm = load_t59_v1()
    if sm is None:
        return None
    pnls = sm["agg"]["test_full_cum_pnl_per_seed"]  # 4 values for 1, 7, 13, 100
    s42 = t59_v1_seed42()
    pnls_5 = pnls + [s42["test_full_cum_pnl"]]
    return {
        "test_full_cum_pnl_per_seed": pnls_5,
        "test_full_cum_pnl_mean": float(np.mean(pnls_5)),
        "test_full_cum_pnl_std": float(np.std(pnls_5)),
        "loso_equiv_sum_per_seed": pnls_5,  # equivalent
    }


def main():
    pilot = {}
    pilot["V1"] = t59_v1_seed42()  # from T59 train_seed42.log
    for strat in ["V2", "V3", "V4"]:
        s = load_summary(strat, PILOT_SEED)
        if s is None:
            print(f"  WARNING: missing {strat} seed={PILOT_SEED}")
            continue
        pilot[strat] = {
            "best_iter": s["best_iter"],
            "test_full_cum_pnl": s["test_full"]["cum_pnl"],
            "loso_equiv_sum_argmax": s["loso_equiv_sum_argmax"],
            "per_sym": {k: round(v["cum_pnl"], 4) for k, v in s["per_sym_test"].items()},
            "split_info": s["split_info"],
            "val_cum_pnl": s["val"]["cum_pnl"],
            "train_time_sec": s["train_time_sec"],
        }

    # 5-seed (winner)
    five_seed = {}
    for strat in ["V1", "V2", "V3", "V4"]:
        seeds_data = []
        for sd in SEEDS_5:
            s = load_summary(strat, sd)
            if s is None:
                continue
            seeds_data.append({
                "seed": sd,
                "best_iter": s["best_iter"],
                "test_full_cum_pnl": float(s["test_full"]["cum_pnl"]),
                "loso_equiv_sum_argmax": float(s["loso_equiv_sum_argmax"]),
                "val_cum_pnl": float(s["val"]["cum_pnl"]),
            })
        if seeds_data:
            pnls = [d["test_full_cum_pnl"] for d in seeds_data]
            five_seed[strat] = {
                "n_seeds": len(seeds_data),
                "seeds_data": seeds_data,
                "test_cum_pnl_mean": float(np.mean(pnls)),
                "test_cum_pnl_std": float(np.std(pnls)),
                "test_cum_pnl_min": float(min(pnls)),
                "test_cum_pnl_max": float(max(pnls)),
            }

    # T59 V1 5-seed reference
    t59_5seed = t59_v1_5seed()

    # DE results (if available)
    de_results = {}
    for strat in ["V1", "V2", "V3", "V4"]:
        p = os.path.join(HERE, f"de_results_{strat}.json")
        if os.path.exists(p):
            with open(p) as f:
                de_results[strat] = json.load(f)

    # T59 V1 DE reference
    p_t59_de = os.path.join(T59_DIR, "de_5seed_results.json")
    t59_de = None
    if os.path.exists(p_t59_de):
        with open(p_t59_de) as f:
            t59_de = json.load(f)

    out = {
        "task": "T64 time-based val for early stopping",
        "pilot_seed42": pilot,
        "five_seed": five_seed,
        "t59_v1_5seed_reference": t59_5seed,
        "de_results_t64": de_results,
        "t59_v1_de_reference": t59_de,
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}", flush=True)

    # Print summary
    print("\n=== PILOT seed=42 ===")
    print(f"  V1 (T59 baseline):     test={pilot.get('V1', {}).get('test_full_cum_pnl', 'N/A')}, best_iter={pilot.get('V1', {}).get('best_iter', 'N/A')}")
    for s in ["V2", "V3", "V4"]:
        if s in pilot:
            d = pilot[s]
            print(f"  {s}: test={d['test_full_cum_pnl']:+.4f}  best_iter={d['best_iter']}  val_cum_pnl={d['val_cum_pnl']:+.4f}")

    if five_seed:
        print("\n=== 5-SEED ===")
        for s in ["V1", "V2", "V3", "V4"]:
            if s in five_seed:
                d = five_seed[s]
                print(f"  {s}: n_seeds={d['n_seeds']}  test_mean={d['test_cum_pnl_mean']:+.4f}  std={d['test_cum_pnl_std']:.3f}  range=[{d['test_cum_pnl_min']:+.3f},{d['test_cum_pnl_max']:+.3f}]")
        if t59_5seed:
            print(f"  V1 T59 ref:  n=5 test_mean={t59_5seed['test_full_cum_pnl_mean']:+.4f}  std={t59_5seed['test_full_cum_pnl_std']:.3f}")

    if de_results:
        print("\n=== DE THRESHOLD RESULTS ===")
        for s, r in de_results.items():
            print(f"  {s}: DE-loso-sum={r['de_loso_equiv']['sum_per_sym']:+.4f}  DE-single={r['de_single_set']['total_cum_pnl_at_best']:+.4f}")
        if t59_de:
            print(f"  T59 V1 ref: DE-loso-sum={t59_de['best_sum']:+.4f}")


if __name__ == "__main__":
    main()
