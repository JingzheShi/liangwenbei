"""T6: Aggregate held-out predictions across seeds into ensemble OOF.

Inputs:
    Single-seed (seed=42) from T4:
        experiments/T4_loso_validate/loso_pred_scheme{A,B}_held{K}.parquet
    Additional seeds from T6:
        experiments/T6_ensemble_multiseed/loso_pred_scheme{A,B}_seed{S}_held{K}.parquet

For each fold K, average prob_0/1/2 across all seeds (including seed=42 from T4).

Outputs (per fold):
    ensemble_pred_scheme{A,B}_held{K}.parquet  -- columns: sym, date, session, t,
                                                  true_label_60, prob_0/1/2,
                                                  midprice_t, midprice_t60

Usage:
    python aggregate_predictions.py --scheme A --seeds 42,1,7,13,100
    python aggregate_predictions.py --scheme B --seeds 42,1,7
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T4_DIR = os.path.join(ROOT, "experiments", "T4_loso_validate")


def find_pred_path(scheme: str, seed: int, held: int) -> str:
    """T4 produced 'loso_pred_scheme{X}_held{K}.parquet' with seed=42 default.
    T6 produces 'loso_pred_scheme{X}_seed{S}_held{K}.parquet'.
    """
    if seed == 42:
        candidate = os.path.join(T4_DIR, f"loso_pred_scheme{scheme}_held{held}.parquet")
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(HERE, f"loso_pred_scheme{scheme}_seed{seed}_held{held}.parquet")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"No prediction file for scheme={scheme} seed={seed} held={held}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=("A", "B"), required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated seeds, e.g. 42,1,7,13,100")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Aggregating scheme={args.scheme} seeds={seeds}", flush=True)

    for held in range(5):
        # Load all seeds' fold-K predictions
        dfs = []
        for s in seeds:
            p = find_pred_path(args.scheme, s, held)
            df = pd.read_parquet(p)
            dfs.append(df)
            print(f"  fold {held} seed {s}: {len(df):,} rows from {os.path.relpath(p, ROOT)}", flush=True)

        # Sanity: same row count + same key columns (sym, date, session, t)
        n0 = len(dfs[0])
        for s, df in zip(seeds, dfs):
            assert len(df) == n0, f"row count mismatch held={held} seed={s}: {len(df)} vs {n0}"
        # Also sanity: match (sym, date, session, t, true_label_60) tuples
        # We assume same ordering since each comes from the same val/test mask
        for s, df in zip(seeds[1:], dfs[1:]):
            ref = dfs[0][["sym", "date", "session", "t", "true_label_60"]].to_numpy()
            cur = df[["sym", "date", "session", "t", "true_label_60"]].to_numpy()
            if not np.array_equal(ref, cur):
                # If ordering differs, sort both then compare
                print(f"  WARN: held={held} seed={s} key mismatch, attempting sort & merge", flush=True)
                dfs[0] = dfs[0].sort_values(["sym", "date", "session", "t"]).reset_index(drop=True)
                dfs[seeds.index(s)] = df.sort_values(["sym", "date", "session", "t"]).reset_index(drop=True)
                ref2 = dfs[0][["sym", "date", "session", "t"]].to_numpy()
                cur2 = dfs[seeds.index(s)][["sym", "date", "session", "t"]].to_numpy()
                assert np.array_equal(ref2, cur2), \
                    f"unrecoverable key mismatch held={held} seed={s}"

        # Average probs
        prob_stack = np.stack([df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
                               for df in dfs], axis=0)
        prob_avg = prob_stack.mean(axis=0).astype(np.float32)
        ensemble_argmax = prob_avg.argmax(axis=1).astype(np.int8)

        out = pd.DataFrame({
            "sym": dfs[0]["sym"].astype(np.int8),
            "date": dfs[0]["date"].astype(np.int16),
            "session": dfs[0]["session"],
            "t": dfs[0]["t"].astype(np.int16),
            "true_label_60": dfs[0]["true_label_60"].astype(np.int8),
            "pred_label_60": ensemble_argmax,
            "prob_0": prob_avg[:, 0],
            "prob_1": prob_avg[:, 1],
            "prob_2": prob_avg[:, 2],
            "midprice_t": dfs[0]["midprice_t"].astype(np.float32),
            "midprice_t60": dfs[0]["midprice_t60"].astype(np.float32),
        })
        out_path = os.path.join(HERE, f"ensemble_pred_scheme{args.scheme}_held{held}.parquet")
        out.to_parquet(out_path, index=False)
        print(f"  -> {out_path}", flush=True)

    print("\nensemble OOF aggregation done", flush=True)


if __name__ == "__main__":
    main()
