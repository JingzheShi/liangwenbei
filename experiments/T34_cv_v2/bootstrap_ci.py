"""
T34.5 — Session-level bootstrap CI on OOF cum_pnl.

Why:
    cum_pnl is a SUM, dominated by a few high-magnitude active sessions.
    The point estimate hides large variance. By resampling sessions
    (with replacement), we get a non-parametric 95% CI.

Methodology:
    1. Compute per-session cum_pnl (240 sessions).
    2. Bootstrap B=2000 times: resample n_sessions=240 with replacement → sum.
    3. Report 2.5%/50%/97.5% percentiles + std.

Important: we resample SESSIONS (sym, date, am/pm), NOT rows. This respects
intra-session correlation (within a session, predictions of t and t+1 are
not independent) — sessions are the natural exchangeable unit.

Output: 95% CI of cum_pnl per (model, horizon).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oof_io import HORIZONS, load_oof, cum_pnl_per_session  # noqa: E402

B = 2000
RNG_SEED = 7777


def bootstrap_session_pnl(per_sess_pnl: np.ndarray, B: int, rng: np.random.Generator) -> Dict:
    """Resample per-session pnls with replacement; return CI dict."""
    n = len(per_sess_pnl)
    idx = rng.integers(0, n, size=(B, n))
    sums = per_sess_pnl[idx].sum(axis=1)
    return {
        "n_sessions": int(n),
        "B": int(B),
        "point_estimate": float(per_sess_pnl.sum()),
        "boot_mean": float(sums.mean()),
        "boot_median": float(np.median(sums)),
        "boot_std": float(sums.std(ddof=1)),
        "ci_2.5": float(np.percentile(sums, 2.5)),
        "ci_97.5": float(np.percentile(sums, 97.5)),
        "p_positive": float((sums > 0).mean()),
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    targets = (
        [("iter_002", h) for h in HORIZONS]
        + [("iter_005b", 40), ("iter_005b", 60), ("t26_aug_a", 60), ("t26_baseline", 60)]
    )

    results = {}
    print(f"\n{'model_h':<22} {'point':>9} {'boot_mean':>10} {'std':>7} "
          f"{'ci_2.5':>9} {'ci_97.5':>9} {'p_pos':>6} {'n_sess':>6}")
    for model, h in targets:
        df = load_oof(model, h)
        per_sess = cum_pnl_per_session(df)
        ci = bootstrap_session_pnl(per_sess["cum_pnl"].to_numpy(), B=B, rng=rng)
        key = f"{model}_h{h}"
        results[key] = ci
        print(f"{key:<22} {ci['point_estimate']:>+9.3f} {ci['boot_mean']:>+10.3f} "
              f"{ci['boot_std']:>7.3f} {ci['ci_2.5']:>+9.3f} {ci['ci_97.5']:>+9.3f} "
              f"{ci['p_positive']:>6.3f} {ci['n_sessions']:>6d}")

    out_path = os.path.join(HERE, "bootstrap_ci_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[bootstrap_ci] saved {out_path}")


if __name__ == "__main__":
    main()
