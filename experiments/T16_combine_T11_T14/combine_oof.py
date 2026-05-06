"""T16: Combine T11 (5-seed Scheme C) + T14 (mixup α=0.2) OOF predictions.

Builds 6-model ensemble probability per fold = mean(prob across 6 models),
runs threshold sweep, compares vs T11-only ensemble (+22.22).

Inputs (all already saved by T11/T14 workers):
    experiments/T11_schemeC_multiseed/loso_pred_h10_seed{S}_held{K}.parquet  (5 seeds)
    experiments/T14_cross_sym_mixup/loso_pred_h10_alpha0p20_held{K}.parquet  (1 mixup)

Outputs:
    loso_pred_combined_h10_held{K}.parquet  (6-model averaged probs)
    sweep_combined_h10.csv
    results.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

T11_DIR = os.path.join(ROOT, "experiments", "T11_schemeC_multiseed")
T14_DIR = os.path.join(ROOT, "experiments", "T14_cross_sym_mixup")
SEEDS = [42, 1, 7, 13, 100]
SYMS = [0, 1, 2, 3, 4]
H = 10

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def thresholded(probs, T, d):
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + d)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def fold_metric(df, T, d):
    probs = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
    pred = thresholded(probs, T, d)
    m = _per_horizon_metrics(
        pred,
        df["true_label"].to_numpy(np.int64),
        df["midprice_t"].to_numpy(np.float32),
        df["midprice_th"].to_numpy(np.float32),
        fee_rate=0.0001,
    )
    return float(m["cum_pnl"]), int(m["n_predictions_active"]), float(m["accuracy"])


def main():
    print(f"=== T16 combine T11(5 seeds) + T14(mixup) OOF h={H} ===", flush=True)
    t0 = time.time()

    # 1. Load and combine
    combined = {}
    for k in SYMS:
        dfs = []
        # T11 5 seeds
        for s in SEEDS:
            p = os.path.join(T11_DIR, f"loso_pred_h{H}_seed{s}_held{k}.parquet")
            dfs.append(pd.read_parquet(p))
        # T14 mixup
        p = os.path.join(T14_DIR, f"loso_pred_h{H}_alpha0p20_held{k}.parquet")
        dfs.append(pd.read_parquet(p))

        # Sanity: same row keys
        ref = dfs[0][["sym", "date", "session", "t"]].to_numpy()
        for j, df in enumerate(dfs[1:], start=1):
            cur = df[["sym", "date", "session", "t"]].to_numpy()
            if not np.array_equal(ref, cur):
                sys.exit(f"key mismatch held={k} model={j}")

        prob_stack = np.stack(
            [df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32) for df in dfs], axis=0
        )
        prob_avg = prob_stack.mean(axis=0).astype(np.float32)
        out = pd.DataFrame({
            "sym": dfs[0]["sym"].astype(np.int8),
            "date": dfs[0]["date"].astype(np.int16),
            "session": dfs[0]["session"],
            "t": dfs[0]["t"].astype(np.int16),
            "true_label": dfs[0]["true_label"].astype(np.int8),
            "pred_label": prob_avg.argmax(axis=1).astype(np.int8),
            "prob_0": prob_avg[:, 0],
            "prob_1": prob_avg[:, 1],
            "prob_2": prob_avg[:, 2],
            "midprice_t": dfs[0]["midprice_t"].astype(np.float32),
            "midprice_th": dfs[0]["midprice_th"].astype(np.float32),
        })
        combined[k] = out
        out_path = os.path.join(HERE, f"loso_pred_combined_h{H}_held{k}.parquet")
        out.to_parquet(out_path, index=False)
        print(f"  fold{k}: {len(out):,} rows -> {os.path.basename(out_path)}", flush=True)

    # 2. Baseline: argmax
    base_pf = []
    for k in SYMS:
        df = combined[k]
        pred = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32).argmax(1).astype(np.int8)
        m = _per_horizon_metrics(
            pred,
            df["true_label"].to_numpy(np.int64),
            df["midprice_t"].to_numpy(np.float32),
            df["midprice_th"].to_numpy(np.float32),
            fee_rate=0.0001,
        )
        base_pf.append(float(m["cum_pnl"]))
    print(f"\n6-model argmax sum = {sum(base_pf):+.4f} per_fold={[round(x,3) for x in base_pf]}",
          flush=True)

    # 3. Sweep
    rows = []
    for T in T_GRID:
        for d in D_GRID:
            per_fold = []
            for k in SYMS:
                cum, n_act, acc = fold_metric(combined[k], T, d)
                per_fold.append((cum, n_act, acc))
            sum_cum = sum(r[0] for r in per_fold)
            n_pos = sum(1 for r in per_fold if r[0] > 0)
            mean_acc = float(np.mean([r[2] for r in per_fold]))
            rows.append({
                "T": T, "delta": d, "sum_cum_pnl": sum_cum,
                "mean_cum_pnl": sum_cum / len(SYMS),
                "n_pos_folds": n_pos, "mean_accuracy": mean_acc,
                "sum_n_active": int(sum(r[1] for r in per_fold)),
                **{f"fold{k}_pnl": per_fold[i][0] for i, k in enumerate(SYMS)},
            })
    sweep_df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False)
    sweep_df.to_csv(os.path.join(HERE, f"sweep_combined_h{H}.csv"), index=False)
    best = sweep_df.iloc[0].to_dict()
    print(f"\n[combined 6-model] best (T={best['T']}, d={best['delta']}) "
          f"sum={best['sum_cum_pnl']:+.4f} pos={int(best['n_pos_folds'])}/5", flush=True)
    print(f"  per-fold: {[round(best[f'fold{k}_pnl'],3) for k in SYMS]}", flush=True)
    print(f"\nTop 5 grid points:", flush=True)
    print(sweep_df.head(5).to_string(index=False), flush=True)

    # 4. Compare T11-only baseline (already in threshold_results.json)
    t11_path = os.path.join(T11_DIR, "threshold_results.json")
    t11_cmp = None
    if os.path.isfile(t11_path):
        with open(t11_path) as f:
            t11_data = json.load(f)
        t11_cmp = t11_data["per_horizon"][f"h{H}"]["ensemble"]["best"]
        print(f"\n[T11-only ensemble] best (T={t11_cmp['T']}, d={t11_cmp['delta']}) "
              f"sum={t11_cmp['sum_cum_pnl']:+.4f}", flush=True)
        delta = best["sum_cum_pnl"] - t11_cmp["sum_cum_pnl"]
        print(f"=> 6-model uplift over T11-only = {delta:+.4f}", flush=True)
    print(f"\nelapsed: {time.time()-t0:.1f}s", flush=True)

    out = {
        "task": "T16 combine T11(5 seeds) + T14(mixup α=0.2) OOF — 6-model ensemble",
        "horizon": H,
        "n_models": 6,
        "models": [f"T11_seed{s}" for s in SEEDS] + ["T14_mixup_alpha0p20"],
        "syms": SYMS,
        "argmax_baseline": {"sum_cum_pnl": float(sum(base_pf)), "per_fold": base_pf},
        "best": {
            "T": float(best["T"]), "delta": float(best["delta"]),
            "sum_cum_pnl": float(best["sum_cum_pnl"]),
            "n_pos_folds": int(best["n_pos_folds"]),
            "mean_accuracy": float(best["mean_accuracy"]),
            "sum_n_active": int(best["sum_n_active"]),
            "per_fold_pnl": [float(best[f"fold{k}_pnl"]) for k in SYMS],
        },
        "top10_grid": sweep_df.head(10).to_dict(orient="records"),
        "t11_only_best": t11_cmp,
        "uplift_vs_t11_only": (
            float(best["sum_cum_pnl"] - t11_cmp["sum_cum_pnl"])
            if t11_cmp is not None else None
        ),
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("results.json written", flush=True)


if __name__ == "__main__":
    main()
