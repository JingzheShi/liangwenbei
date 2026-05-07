"""T38 quick result summary — read results.json + print ranking + per_fold table."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)

baseline = R.get("iter_006_baseline_sum", 13.6062)
print(f"=== T38 RANKING vs iter_006 baseline ({baseline:+.4f}) ===")
print(f"{'combo':28s}  {'sum':>10s}  {'Δvs06':>8s}  {'pos':>4s}  {'std':>6s}  per_fold")

ranked = R["ranking"]
for r in ranked:
    name = r["combo"]
    s = r["sum"]
    pos = r["n_pos_folds"]
    std = r["std"]
    delta = s - baseline
    pf = R["results_by_combo"][name]["de_best"]["per_fold_pnl"]
    pf_str = "[" + ", ".join(f"{x:+5.2f}" for x in pf) + "]"
    flag = " ★" if s > baseline else ""
    print(f"{name:28s}  {s:+10.4f}  {delta:+8.4f}  {pos}/5   {std:6.3f}  {pf_str}{flag}")

# best
best_name = ranked[0]["combo"]
best = R["results_by_combo"][best_name]["de_best"]
print(f"\n=== BEST: {best_name} ===")
print(f"  T_up={best['T_up']:.4f}  T_dn={best['T_dn']:.4f}  d_up={best['d_up']:.4f}  d_dn={best['d_dn']:.4f}")
print(f"  sum={best['sum_cum_pnl']:+.4f}  per_fold={[round(x,3) for x in best['per_fold_pnl']]}")
print(f"  Δvs iter_006_baseline = {best['sum_cum_pnl']-baseline:+.4f}")
