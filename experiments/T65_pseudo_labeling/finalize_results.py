"""T65 finalize results.

Reads pseudo_summary_h60.json and de_results.json, writes results.json and
fills in REPORT.md.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    sum_path = os.path.join(HERE, "pseudo_summary_h60.json")
    de_path = os.path.join(HERE, "de_results.json")
    pseudo_summary = json.load(open(sum_path))
    de = json.load(open(de_path))

    seeds = sorted(int(k) for k in pseudo_summary["seed_results"])
    per_seed_lines = []
    for s in seeds:
        r = pseudo_summary["seed_results"][str(s)]
        per_seed_lines.append({
            "seed": s,
            "best_iter": r["best_iter"],
            "test_full_cum_pnl": r["test_full"]["cum_pnl"],
            "test_per_sym_sum": r["loso_equiv_sum_argmax"],
            "test_acc": r["test_full"]["accuracy"],
        })

    iter010 = 24.52
    de_full_sum = de["de_loso_full"]["sum"]
    de_held_sum = de["de_loso_held"]["sum"]
    raw_full_sum = de["raw_argmax_full"]["loso_equiv_sum"]
    raw_held_sum = de["raw_argmax_held"]["loso_equiv_sum"]
    held_at_full_thr = de["held_at_full_thresh"]["sum"]

    results = {
        "task": "T65 pseudo-labeling 5-seed (R34 stage 3 + 14,251 pseudo @ w=0.5)",
        "n_pseudo": pseudo_summary["seed_results"][str(seeds[0])]["n_pseudo"],
        "metrics": {
            "raw_argmax_full_sum": raw_full_sum,
            "raw_argmax_held_sum": raw_held_sum,
            "de_loso_full_sum": de_full_sum,
            "de_loso_held_sum": de_held_sum,
            "held_at_full_thresh_sum": held_at_full_thr,
        },
        "per_seed_lines": per_seed_lines,
        "vs_iter010": {
            "iter010": iter010,
            "de_full_diff": de_full_sum - iter010,
            "de_held_diff": de_held_sum - iter010,
            "held_at_full_thresh_diff": held_at_full_thr - iter010,
        },
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote -> {out_path}", flush=True)

    # Build report
    rep_lines = []
    rep_lines.append("# T65 Pseudo-Labeling — Self-Train Augmentation")
    rep_lines.append("")
    decision = "iter_011 candidate" if held_at_full_thr > iter010 + 0.50 else (
        "marginal — not promoting" if abs(held_at_full_thr - iter010) < 0.50 else "regression"
    )
    rep_lines.append(f"**Status:** {decision}")
    rep_lines.append(f"**vs iter_010 (+{iter010}):** held-out @ full-thresh = {held_at_full_thr:+.4f}  ({held_at_full_thr - iter010:+.4f})")
    rep_lines.append(f"**Full-set DE LOSO:** {de_full_sum:+.4f}  ({de_full_sum - iter010:+.4f}) — but biased upward by training on test X.")
    rep_lines.append("")
    rep_lines.append("## Approach")
    rep_lines.append("")
    rep_lines.append("Self-train: predict test → high-conf directional rows become pseudo train data.")
    rep_lines.append("- 5-seed-avg model from T59 generates probs on full 442k test (date 96-119).")
    rep_lines.append("- Gate: `max(prob_0, prob_2) > 0.55 AND |prob_0 - prob_2| > 0.10` → 14,251 pseudo rows (3.22%).")
    rep_lines.append("- Pseudo label = `argmax(prob_0, prob_2)` (binary 0/2; flat=1 not used).")
    rep_lines.append("- Pseudo data appended to original 1.47M train (post-aug_a) with `sample_weight=0.5 × class_balance`.")
    rep_lines.append("- Retrain T59-style 5-seed (hyperparam-diverse, GPU LightGBM, R34 stage 3 340-d).")
    rep_lines.append("")
    rep_lines.append("## Pseudo Quality")
    rep_lines.append("")
    rep_lines.append(f"- 14,251 high-confidence directional rows (~3.22% of test).")
    rep_lines.append(f"- Per-class: 6,767 up + 7,484 down.")
    rep_lines.append(f"- Pseudo accuracy vs true labels: ~48% (30% are actually flat=1).")
    rep_lines.append(f"- The model's directional belief is correct in many cases, but the move size doesn't reach α=0.1%.")
    rep_lines.append(f"- This is why we cap `sample_weight=0.5` to limit pseudo's influence.")
    rep_lines.append("")
    rep_lines.append("## Per-seed results")
    rep_lines.append("")
    rep_lines.append("| seed | best_iter | test cum_pnl (full) | per-sym sum | acc |")
    rep_lines.append("|---|---|---|---|---|")
    for r in per_seed_lines:
        rep_lines.append(f"| {r['seed']} | {r['best_iter']} | {r['test_full_cum_pnl']:+.4f} | {r['test_per_sym_sum']:+.4f} | {r['test_acc']:.4f} |")
    rep_lines.append("")
    rep_lines.append("## 5-seed-avg DE 4D thresh")
    rep_lines.append("")
    rep_lines.append("| Eval scope | Raw argmax sum | DE LOSO sum | Δ vs iter_010 (+24.52) |")
    rep_lines.append("|---|---|---|---|")
    rep_lines.append(f"| Full 442k (TRAINING contains pseudo subset) | {raw_full_sum:+.4f} | {de_full_sum:+.4f} | {de_full_sum - iter010:+.4f} |")
    rep_lines.append(f"| Held-out (non-pseudo, ~{de['n_test_held']:,}) | {raw_held_sum:+.4f} | {de_held_sum:+.4f} | {de_held_sum - iter010:+.4f} |")
    rep_lines.append(f"| Held-out @ full-thresh (deployment-realistic) | — | {held_at_full_thr:+.4f} | {held_at_full_thr - iter010:+.4f} |")
    rep_lines.append("")
    rep_lines.append("## Caveats")
    rep_lines.append("")
    rep_lines.append("⚠️ **Test-X memorization**: pseudo training rows are pulled from test X. Even with synthetic")
    rep_lines.append("(self-generated) labels rather than ground truth, the model sees the test feature distribution,")
    rep_lines.append("which can bias held-in evaluation upward. The fairer view is the held-out (non-pseudo) cum_pnl.")
    rep_lines.append("")
    rep_lines.append("⚠️ **Production realism**: this technique is only valid when we have access to the test X. At")
    rep_lines.append("submission time, we'd need to retrain with pseudo from the *platform's* hidden test, which is")
    rep_lines.append("not feasible. So a T65 winning model would have memorized our local test — its platform")
    rep_lines.append("performance is the open question. Best-case expectation: improvement matches the held-out delta.")

    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("\n".join(rep_lines) + "\n")
    print(f"wrote -> REPORT.md", flush=True)


if __name__ == "__main__":
    main()
