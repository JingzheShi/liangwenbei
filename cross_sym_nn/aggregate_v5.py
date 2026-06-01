"""Aggregate v5 attn-deepdive results: per-arch 5-seed mean/std for test_pnl & val_pnl."""
from __future__ import annotations
import json, glob, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs_v5"
RUNS_V4 = HERE / "runs_v4"


def gather(root: Path, arch: str):
    files = sorted(root.glob(f"{arch}_pruned*_s*/results.json"))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
            if d.get("status") != "ok":
                continue
            rows.append({
                "arch": arch,
                "seed": d["seed"],
                "test_pnl": d["test_pnl"],
                "val_pnl": d["val_pnl"],
                "val_ic": d["val_ic"],
                "test_ic": d["test_ic"],
                "best_step": d["best_step"],
                "t_sec": d.get("t_train_sec", 0),
                "n_params": d["n_params"],
            })
        except Exception as e:
            print(f"  failed to load {f}: {e}")
    return rows


def summarize(rows, name):
    if not rows:
        return None
    ts = np.array([r["test_pnl"] for r in rows])
    vs = np.array([r["val_pnl"] for r in rows])
    return {
        "name": name,
        "n_seed": len(rows),
        "test_mean": float(ts.mean()),
        "test_std":  float(ts.std(ddof=1)) if len(ts) > 1 else 0.0,
        "test_min":  float(ts.min()),
        "test_max":  float(ts.max()),
        "val_mean":  float(vs.mean()),
        "val_std":   float(vs.std(ddof=1)) if len(vs) > 1 else 0.0,
        "n_params":  rows[0]["n_params"],
        "best_step_mean": float(np.mean([r["best_step"] for r in rows])),
        "t_sec_mean":    float(np.mean([r["t_sec"] for r in rows])),
        "seeds": [r["seed"] for r in rows],
        "test_pnls": [r["test_pnl"] for r in rows],
        "val_pnls":  [r["val_pnl"]  for r in rows],
    }


def main():
    archs_v5 = ["v1_deeper2", "v1_deeper3", "v2_gated",
                "v3_wider", "v3_wider_d96", "v5_xfmr3"]
    sota = "sae_cross_sym"
    mlp_baseline = 31.44  # from prior reports (MLP baseline)

    print("=" * 78)
    print("V5 ATTN DEEPDIVE AGGREGATION")
    print("=" * 78)

    # v4 sae_cross_sym SOTA reference
    sota_rows = gather(RUNS_V4, sota)
    sota_summary = summarize(sota_rows, sota) if sota_rows else None

    results = []
    for arch in archs_v5:
        rows = gather(RUNS, arch)
        if rows:
            summ = summarize(rows, arch)
            results.append(summ)

    # Header line
    print(f"{'arch':<18s} {'n_seed':>6} {'test_mean':>10} {'test_std':>9} "
          f"{'val_mean':>9} {'n_params':>9} {'vs_sota':>8} {'vs_mlp':>8}")
    print("-" * 78)
    sota_mean = sota_summary["test_mean"] if sota_summary else None
    if sota_summary:
        s = sota_summary
        print(f"{s['name']:<18s} {s['n_seed']:>6d} {s['test_mean']:>+10.4f} "
              f"{s['test_std']:>9.4f} {s['val_mean']:>+9.4f} {s['n_params']:>9,d}  "
              f"{'(SOTA)':>8s} {s['test_mean']-mlp_baseline:>+8.2f}")
        print("-" * 78)
    for r in results:
        delta_sota = r["test_mean"] - (sota_mean if sota_mean else r["test_mean"])
        delta_mlp = r["test_mean"] - mlp_baseline
        marker = " *NEW SOTA*" if (sota_mean and r["test_mean"] > sota_mean) else ""
        print(f"{r['name']:<18s} {r['n_seed']:>6d} {r['test_mean']:>+10.4f} "
              f"{r['test_std']:>9.4f} {r['val_mean']:>+9.4f} {r['n_params']:>9,d} "
              f"{delta_sota:>+8.4f} {delta_mlp:>+8.2f}{marker}")

    # Best
    if results:
        best = max(results, key=lambda r: r["test_mean"])
        print("\nBest v5 variant: {} (test_mean={:+.4f}, val_mean={:+.4f})".format(
            best["name"], best["test_mean"], best["val_mean"]))
        if sota_mean:
            print(f"vs sae_cross_sym SOTA: {best['test_mean']-sota_mean:+.4f}")
        print(f"vs MLP baseline:       {best['test_mean']-mlp_baseline:+.2f}")

    out = {
        "sota_baseline": sota_summary,
        "mlp_baseline": mlp_baseline,
        "variants": results,
        "best": best if results else None,
    }
    with open(HERE / "v5_aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {HERE/'v5_aggregate.json'}")


if __name__ == "__main__":
    main()
