"""Analyze all v2 runs and write v2_summary.json."""
import json
from pathlib import Path
from collections import defaultdict

MLP_BASELINE = 31.44
DEEPSETS_BASELINE = None  # new arch, no prior baseline

runs_dir = Path(__file__).resolve().parent / "runs_v2"
results = []

for f in runs_dir.glob("*/results.json"):
    try:
        r = json.load(open(f))
        r["_dir"] = f.parent.name
        results.append(r)
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not results:
    print("No results found!")
    exit(1)

by_arch = defaultdict(list)
for r in results:
    by_arch[r["arch"]].append(r)

print("=== v2 Results by Arch ===")
arch_summary = {}
for arch in sorted(by_arch.keys()):
    rs = by_arch[arch]
    test_pnls = [r["test_pnl"] for r in rs]
    val_pnls = [r["val_pnl"] for r in rs]
    best_steps = [r["best_step"] for r in rs]
    n = len(rs)
    mean_test = sum(test_pnls) / n
    max_test = max(test_pnls)
    mean_val = sum(val_pnls) / n
    print(f"\n{arch} (n={n}):")
    print(f"  test_pnl: mean={mean_test:.2f}, max={max_test:.2f}, min={min(test_pnls):.2f}")
    print(f"  val_pnl:  mean={mean_val:.2f}")
    print(f"  best_step: {best_steps}")
    vs_baseline = mean_test - MLP_BASELINE
    print(f"  vs MLP baseline ({MLP_BASELINE}): {vs_baseline:+.2f}")
    arch_summary[arch] = {
        "n": n,
        "seeds": [r["seed"] for r in rs],
        "test_pnl_mean": round(mean_test, 4),
        "test_pnl_max": round(max_test, 4),
        "test_pnl_min": round(min(test_pnls), 4),
        "test_pnl_per_seed": {r["seed"]: round(r["test_pnl"], 4) for r in rs},
        "val_pnl_mean": round(mean_val, 4),
        "best_step_mean": round(sum(best_steps) / n, 0),
        "vs_mlp_baseline": round(vs_baseline, 4),
        "test_sym_pnl_mean": {
            f"sym{s}": round(sum(r["test_sym_pnl"][f"sym{s}"] for r in rs if f"sym{s}" in r.get("test_sym_pnl", {})) / n, 4)
            for s in range(5)
        } if rs[0].get("test_sym_pnl") else {},
    }

print(f"\n=== Summary ===")
for arch, s in sorted(arch_summary.items(), key=lambda x: -x[1]["test_pnl_mean"]):
    print(f"{arch}: mean={s['test_pnl_mean']:.2f}, max={s['test_pnl_max']:.2f}, vs_baseline={s['vs_mlp_baseline']:+.2f}")

summary = {
    "mlp_baseline": MLP_BASELINE,
    "n_total_runs": len(results),
    "arch_summary": arch_summary,
    "all_runs": [
        {
            "arch": r["arch"], "seed": r["seed"],
            "test_pnl": round(r["test_pnl"], 4),
            "val_pnl": round(r["val_pnl"], 4),
            "best_step": r["best_step"],
            "test_ic": round(r.get("test_ic", 0), 4),
            "n_params": r.get("n_params", 0),
        }
        for r in sorted(results, key=lambda x: (x["arch"], x["seed"]))
    ],
}

out = Path(__file__).resolve().parent / "v2_summary.json"
json.dump(summary, open(out, "w"), indent=2)
print(f"\nWrote {out}")
