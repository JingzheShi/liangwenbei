"""T11: Aggregate multi-seed LOSO probabilities into ensemble OOF.

For each (horizon, fold), average prob_0/1/2 across all seeds.
Outputs:
    loso_pred_ensemble_h{H}_held{K}.parquet  (averaged probs + meta)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="10")
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--syms", default="0,1,2,3,4")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    target_syms = [int(x) for x in args.syms.split(",")]

    print(f"Aggregating horizons={horizons} seeds={seeds}", flush=True)

    for H in horizons:
        print(f"\n--- h={H} ---", flush=True)
        for held in target_syms:
            dfs = []
            for s in seeds:
                p = os.path.join(HERE, f"loso_pred_h{H}_seed{s}_held{held}.parquet")
                if not os.path.exists(p):
                    sys.exit(f"missing {p}")
                df = pd.read_parquet(p)
                dfs.append(df)
                print(f"  fold{held} seed{s}: {len(df):,} rows", flush=True)

            n0 = len(dfs[0])
            for s, df in zip(seeds, dfs):
                assert len(df) == n0, f"row count mismatch held={held} seed={s}"

            # Sanity: same key ordering across seeds
            ref_keys = dfs[0][["sym", "date", "session", "t"]].to_numpy()
            for s, df in list(zip(seeds, dfs))[1:]:
                cur_keys = df[["sym", "date", "session", "t"]].to_numpy()
                if not np.array_equal(ref_keys, cur_keys):
                    sys.exit(f"key mismatch held={held} seed={s}")

            prob_stack = np.stack(
                [df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32) for df in dfs], axis=0
            )
            prob_avg = prob_stack.mean(axis=0).astype(np.float32)
            ensemble_argmax = prob_avg.argmax(axis=1).astype(np.int8)

            out = pd.DataFrame({
                "sym": dfs[0]["sym"].astype(np.int8),
                "date": dfs[0]["date"].astype(np.int16),
                "session": dfs[0]["session"],
                "t": dfs[0]["t"].astype(np.int16),
                "true_label": dfs[0]["true_label"].astype(np.int8),
                "pred_label": ensemble_argmax,
                "prob_0": prob_avg[:, 0],
                "prob_1": prob_avg[:, 1],
                "prob_2": prob_avg[:, 2],
                "midprice_t": dfs[0]["midprice_t"].astype(np.float32),
                "midprice_th": dfs[0]["midprice_th"].astype(np.float32),
            })
            out_path = os.path.join(HERE, f"loso_pred_ensemble_h{H}_held{held}.parquet")
            out.to_parquet(out_path, index=False)
            print(f"  -> {out_path}", flush=True)

    print("\nensemble aggregation done", flush=True)


if __name__ == "__main__":
    main()
