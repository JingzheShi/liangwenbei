"""Aggregate V4 runs vs sae_mlp 259d baseline.

Reads:
  runs_v4/<arch>_pruned259_s{0..4}/results.json
  runs_v2_pruned/sae_mlp_pruned259_s{0..4}/results.json   (baseline)

Outputs a markdown table-fragment to stdout & saves runs_v4_summary.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
V4_ARCHS = ["multi_h_sae", "sae_cross_sym", "sae_wider", "vae_mlp"]
SEEDS = [0, 1, 2, 3, 4]
BASE_DIR = HERE / "runs_v2_pruned"
V4_DIR   = HERE / "runs_v4"


def load_seeds(arch_dir: Path, name_fmt: str) -> list[dict]:
    out = []
    for s in SEEDS:
        p = arch_dir / name_fmt.format(s=s) / "results.json"
        if not p.exists():
            continue
        with open(p) as f:
            out.append(json.load(f))
    return out


def fmt_mean_std(vals, fmt="{:+.4f}"):
    if not vals:
        return "N/A"
    arr = np.array(vals)
    return f"{fmt.format(arr.mean())} ± {fmt.format(arr.std(ddof=0)).lstrip('+')}"


def main():
    baseline = load_seeds(BASE_DIR, "sae_mlp_pruned259_s{s}")
    print(f"[base] sae_mlp 259d: {len(baseline)} seeds loaded from {BASE_DIR}")

    rows = []
    rows.append(("sae_mlp (baseline)", baseline))
    for arch in V4_ARCHS:
        runs = load_seeds(V4_DIR, f"{arch}_pruned259_s{{s}}")
        rows.append((arch, runs))
        print(f"[v4] {arch}: {len(runs)} seeds loaded")

    summary = {}

    print()
    header = f"| {'arch':<20} | {'n_seed':>6} | {'n_params':>10} | {'val_pnl (mean ± std)':>26} | {'test_pnl (mean ± std)':>26} | {'mean_best_step':>14} |"
    sep    = f"|{'-' * 22}|{'-' * 8}|{'-' * 12}|{'-' * 28}|{'-' * 28}|{'-' * 16}|"
    print(header)
    print(sep)

    for name, runs in rows:
        if not runs:
            print(f"| {name:<20} | {'0':>6} | {'-':>10} | {'-':>26} | {'-':>26} | {'-':>14} |")
            summary[name] = {"n_seed": 0}
            continue
        vps = [r['val_pnl'] for r in runs if r.get('status') != 'failed']
        tps = [r['test_pnl'] for r in runs if r.get('status') != 'failed']
        steps = [r.get('best_step', 0) for r in runs if r.get('status') != 'failed']
        n_params = runs[0].get('n_params', 0)
        val_str  = fmt_mean_std(vps)
        test_str = fmt_mean_std(tps)
        step_avg = int(np.mean(steps)) if steps else 0
        print(f"| {name:<20} | {len(runs):>6} | {n_params:>10,} | {val_str:>26} | {test_str:>26} | {step_avg:>14} |")
        summary[name] = {
            "n_seed": len(runs),
            "n_params": int(n_params),
            "val_pnl_mean": float(np.mean(vps)) if vps else None,
            "val_pnl_std":  float(np.std(vps, ddof=0)) if vps else None,
            "test_pnl_mean": float(np.mean(tps)) if tps else None,
            "test_pnl_std":  float(np.std(tps, ddof=0)) if tps else None,
            "test_pnl_per_seed": tps,
            "val_pnl_per_seed": vps,
            "best_step_per_seed": steps,
            "best_step_mean": step_avg,
        }

    # vs-baseline delta
    base_mean = summary["sae_mlp (baseline)"].get("test_pnl_mean")
    if base_mean is not None:
        print()
        print(f"Baseline test_pnl mean = {base_mean:+.4f}\n")
        print(f"| {'arch':<20} | {'Δ test_pnl':>12} | {'beats baseline?':>16} |")
        print(f"|{'-' * 22}|{'-' * 14}|{'-' * 18}|")
        for arch in V4_ARCHS:
            s = summary.get(arch, {})
            tm = s.get("test_pnl_mean")
            if tm is None:
                print(f"| {arch:<20} | {'N/A':>12} | {'N/A':>16} |")
                continue
            delta = tm - base_mean
            beats = "YES" if delta > 0 else "no"
            print(f"| {arch:<20} | {delta:+.4f} | {beats:>16} |")

    with open(HERE / "runs_v4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved → {HERE / 'runs_v4_summary.json'}")


if __name__ == "__main__":
    main()
