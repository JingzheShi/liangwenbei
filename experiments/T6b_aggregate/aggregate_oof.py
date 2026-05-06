"""T6b: Aggregate held-out predictions across 5 seeds into ensemble OOF.

Inputs:
    seed=42 (T4):
        experiments/T4_loso_validate/loso_pred_schemeA_held{0..4}.parquet
    seed in {1, 7, 13, 100} (T6):
        experiments/T6_ensemble_multiseed/loso_pred_schemeA_seed{S}_held{0..4}.parquet

For each fold K in [0..4]:
    - Load 5 single-seed prob files (same OOF rows under sym=K).
    - Verify (sym, date, session, t, true_label_60) row keys match across seeds.
    - Mean prob_0/prob_1/prob_2 across the 5 seeds -> ensemble probabilities.

Outputs (5 files):
    experiments/T6_ensemble_multiseed/loso_pred_ensemble_held{0..4}.parquet
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T4_DIR = os.path.join(ROOT, "experiments", "T4_loso_validate")
T6_DIR = os.path.join(ROOT, "experiments", "T6_ensemble_multiseed")

SEEDS = [42, 1, 7, 13, 100]


def progress(step: str, **extra) -> None:
    p = os.path.join(HERE, "worker-progress.json")
    payload = {
        "status": "running",
        "step": step,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)


def find_pred_path(seed: int, held: int) -> str:
    if seed == 42:
        cand = os.path.join(T4_DIR, f"loso_pred_schemeA_held{held}.parquet")
    else:
        cand = os.path.join(T6_DIR, f"loso_pred_schemeA_seed{seed}_held{held}.parquet")
    if not os.path.exists(cand):
        raise FileNotFoundError(cand)
    return cand


def main():
    progress("starting", seeds=SEEDS)
    out_dir = T6_DIR
    os.makedirs(out_dir, exist_ok=True)

    summary = {"seeds": SEEDS, "folds": []}
    for held in range(5):
        progress(f"aggregating_fold_{held}", seeds=SEEDS)
        dfs = []
        for s in SEEDS:
            p = find_pred_path(s, held)
            df = pd.read_parquet(p).sort_values(
                ["sym", "date", "session", "t"]
            ).reset_index(drop=True)
            dfs.append(df)
            print(f"  fold {held} seed {s:>3}: {len(df):,} rows  "
                  f"({os.path.relpath(p, ROOT)})", flush=True)

        n0 = len(dfs[0])
        for s, df in zip(SEEDS, dfs):
            assert len(df) == n0, f"row count mismatch held={held} seed={s}: {len(df)} vs {n0}"

        ref_keys = dfs[0][["sym", "date", "session", "t", "true_label_60"]].to_numpy()
        for s, df in zip(SEEDS[1:], dfs[1:]):
            cur_keys = df[["sym", "date", "session", "t", "true_label_60"]].to_numpy()
            assert np.array_equal(ref_keys, cur_keys), (
                f"key mismatch held={held} seed={s} after sort"
            )

        prob_stack = np.stack(
            [df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32) for df in dfs],
            axis=0,
        )  # (5_seeds, N, 3)
        prob_avg = prob_stack.mean(axis=0).astype(np.float32)
        ens_argmax = prob_avg.argmax(axis=1).astype(np.int8)

        # Sanity: probs sum ~ 1
        sums = prob_avg.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-3), f"prob row-sum drift: {sums.min()}..{sums.max()}"

        out = pd.DataFrame({
            "sym": dfs[0]["sym"].astype(np.int8),
            "date": dfs[0]["date"].astype(np.int16),
            "session": dfs[0]["session"],
            "t": dfs[0]["t"].astype(np.int16),
            "true_label_60": dfs[0]["true_label_60"].astype(np.int8),
            "pred_label_60": ens_argmax,
            "prob_0": prob_avg[:, 0],
            "prob_1": prob_avg[:, 1],
            "prob_2": prob_avg[:, 2],
            "midprice_t": dfs[0]["midprice_t"].astype(np.float32),
            "midprice_t60": dfs[0]["midprice_t60"].astype(np.float32),
        })
        out_path = os.path.join(out_dir, f"loso_pred_ensemble_held{held}.parquet")
        out.to_parquet(out_path, index=False)
        print(f"  -> {out_path}", flush=True)

        summary["folds"].append({
            "held": held,
            "n_rows": int(n0),
            "out": os.path.relpath(out_path, ROOT),
        })

    summary["status"] = "ok"
    with open(os.path.join(HERE, "aggregate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    progress("done", n_folds=5)
    print("\naggregation complete", flush=True)


if __name__ == "__main__":
    main()
