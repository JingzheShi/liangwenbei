"""Ensemble experiment for v4 sweep runs.

Loads preds.npz from multiple runs, averages predictions, re-searches
threshold on val, evaluates on test. Saves results to ensemble_results.json.

Run: python ensemble_v4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_v2 import eval_predictions, eval_test
from data_loader import load_grouped_data


def load_preds(paths, key_val="preds_val", key_test="preds_test"):
    """Stack predictions from multiple runs. Returns (val_stack, test_stack)."""
    val_list, test_list = [], []
    for p in paths:
        d = np.load(p)
        val_list.append(d[key_val])
        test_list.append(d[key_test])
    return np.stack(val_list, axis=0), np.stack(test_list, axis=0)  # (N_models, n_groups, n_sym)


def ensemble_eval(val_stack, test_stack, val_data, test_data, target_scale, name):
    """Average preds, search threshold on val, evaluate on test."""
    preds_val_ens  = val_stack.mean(axis=0)   # (n_groups, 5)
    preds_test_ens = test_stack.mean(axis=0)  # (n_groups, 5)

    val_results, val_thrs = eval_predictions(preds_val_ens, val_data, target_scale)
    test_results = eval_test(preds_test_ens, test_data, val_thrs, target_scale)

    total_val  = val_results['total_pnl']
    total_test = test_results['total_pnl']
    sym_pnls = {f"sym{s}": test_results[f'sym{s}_pnl'] for s in range(5)}
    print(f"  [{name}] val_pnl={total_val:+.4f}  test_pnl={total_test:+.4f}")
    for s in range(5):
        print(f"    sym{s}: {sym_pnls[f'sym{s}']:+.4f}")
    return {
        "name": name,
        "n_models": int(val_stack.shape[0]),
        "val_pnl": float(total_val),
        "test_pnl": float(total_test),
        "test_sym_pnl": {k: float(v) for k, v in sym_pnls.items()},
        "val_thresholds": {str(s): float(v) for s, v in val_thrs.items()},
    }


def main():
    print("Loading data ...", flush=True)
    _, val_data, test_data, stats = load_grouped_data(level="L8", verbose=False)
    target_scale = stats['target_scale']
    print(f"  target_scale={target_scale}  val_groups={val_data['X_grp'].shape[0]}  test_groups={test_data['X_grp'].shape[0]}")

    # ── Define run paths ───────────────────────────────────────────────────
    v4_dir    = HERE / "runs_v4"
    v2p_dir   = HERE / "runs_v2_pruned"
    SEEDS     = [0, 1, 2, 3, 4]

    def v4_paths(arch):
        return [v4_dir / f"{arch}_pruned259_s{s}" / "preds.npz" for s in SEEDS]

    def v2p_paths(arch):
        return [v2p_dir / f"{arch}_pruned259_s{s}" / "preds.npz" for s in SEEDS]

    configs = {
        # homogeneous ensembles (single arch × 5 seeds)
        "sae_cross_sym_x5":      v4_paths("sae_cross_sym"),
        "sae_mlp_x5":            v2p_paths("sae_mlp"),
        "sae_wider_x5":          v4_paths("sae_wider"),
        "vae_mlp_x5":            v4_paths("vae_mlp"),
        "multi_h_sae_x5":        v4_paths("multi_h_sae"),
        # heterogeneous: SOTA cross_sym + sae_mlp (most diverse pair)
        "sae_cross_sym_x5+sae_mlp_x5": v4_paths("sae_cross_sym") + v2p_paths("sae_mlp"),
        # full v4+baseline 25 models
        "all_v4_x20_plus_base_x5": (v4_paths("sae_cross_sym") + v4_paths("sae_mlp")
                                     if False else  # sae_mlp is in v2p not v4
                                     v4_paths("sae_cross_sym") + v4_paths("sae_wider")
                                     + v4_paths("vae_mlp") + v4_paths("multi_h_sae")
                                     + v2p_paths("sae_mlp")),
        # top-2 diverse: cross_sym + vae_mlp
        "sae_cross_sym_x5+vae_mlp_x5": v4_paths("sae_cross_sym") + v4_paths("vae_mlp"),
    }

    results = []

    for name, paths in configs.items():
        # Verify all paths exist
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"  SKIP [{name}]: missing {missing}")
            continue
        print(f"\nEnsemble: {name}  (n={len(paths)} models)")
        val_stack, test_stack = load_preds(paths)
        r = ensemble_eval(val_stack, test_stack, val_data, test_data, target_scale, name)
        results.append(r)

    # Sort by test_pnl
    results.sort(key=lambda x: x["test_pnl"], reverse=True)

    print("\n=== ENSEMBLE SUMMARY ===")
    print(f"{'name':<40}  {'n_models':>8}  {'test_pnl':>10}  {'val_pnl':>10}")
    print("-" * 72)
    for r in results:
        print(f"  {r['name']:<38}  {r['n_models']:>8}  {r['test_pnl']:>+10.4f}  {r['val_pnl']:>+10.4f}")

    out_path = HERE / "ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump({"configs": results}, f, indent=2)
    print(f"\nSaved → {out_path}")

    # best result
    if results:
        best = results[0]
        print(f"\nBEST: {best['name']}  test_pnl={best['test_pnl']:+.4f}")


if __name__ == "__main__":
    main()
