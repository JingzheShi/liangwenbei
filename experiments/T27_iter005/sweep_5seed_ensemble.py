"""T27: Sweep thresholds on the 5-seed aug_a ensemble OOF predictions for h=60.

Loads loso_pred_h60_aug_a_seed{S}_held{K}.parquet for S in {42,1,7,13,100} and
K in 0..4. For each fold K, ensemble = mean of probs over the 5 seeds. Then
sweep (T, delta) on the concatenated 5-fold OOF.

NOTE: seed=42 predictions live in T26 (loso_pred_h60_aug_a_held{K}.parquet).
"""
from __future__ import annotations

import argparse
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

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")

T_GRID = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
D_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def load_seed_fold(seed: int, held: int, H: int = 60) -> pd.DataFrame:
    if seed == 42:
        # seed 42 is the T26 baseline (no _seed suffix in T26 naming)
        p = os.path.join(T26_DIR, f"loso_pred_h{H}_aug_a_held{held}.parquet")
    else:
        p = os.path.join(HERE, f"loso_pred_h{H}_aug_a_seed{seed}_held{held}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return pd.read_parquet(p)


def thresholded_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    pred_side = np.where(p2 > p0, 2, 0)
    return np.where(take, pred_side, 1).astype(np.int8)


def per_fold_metrics(probs: np.ndarray, true_lab: np.ndarray,
                     mp_t: np.ndarray, mp_th: np.ndarray,
                     T: float, delta: float) -> dict:
    pred = thresholded_pred(probs, T, delta)
    m = _per_horizon_metrics(
        pred, true_lab.astype(np.int64),
        mp_t.astype(np.float32), mp_th.astype(np.float32),
        fee_rate=0.0001,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "single_pnl": float(m["single_pnl"]),
        "accuracy": float(m["accuracy"]),
        "n_active": int(m["n_predictions_active"]),
        "f0_5_macro": float(m["f0_5_macro"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--syms", default="0,1,2,3,4")
    args = ap.parse_args()

    H = args.horizon
    seeds = [int(s) for s in args.seeds.split(",")]
    target_syms = [int(s) for s in args.syms.split(",")]

    print(f"=== T27 5-seed ensemble OOF sweep h={H} seeds={seeds} ===", flush=True)
    t0 = time.time()

    # Build per-fold ensemble probs
    fold_blobs = {}
    for k in target_syms:
        per_seed = []
        meta_df = None
        for s in seeds:
            d = load_seed_fold(s, k, H)
            per_seed.append(d[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32))
            if meta_df is None:
                meta_df = d[["sym", "date", "session", "t",
                             "true_label", "midprice_t", "midprice_th"]].copy()
        # Mean ensemble
        ens = np.mean(np.stack(per_seed, axis=0), axis=0)
        fold_blobs[k] = {"probs": ens, "meta": meta_df}
        print(
            f"  fold {k}: {len(meta_df):,} rows, ensemble of {len(seeds)} seeds",
            flush=True,
        )

    # Per-seed best (single-seed reference)
    print("\n--- Per-seed (T=0.45,d=0.10) reference ---", flush=True)
    for s in seeds:
        per_fold_pnl = []
        for k in target_syms:
            d = load_seed_fold(s, k, H)
            probs = d[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
            r = per_fold_metrics(
                probs, d["true_label"].to_numpy(),
                d["midprice_t"].to_numpy(), d["midprice_th"].to_numpy(),
                T=0.45, delta=0.10,
            )
            per_fold_pnl.append(r["cum_pnl"])
        print(
            f"  seed{s:>3d}: T=0.45 d=0.10 sum={sum(per_fold_pnl):+.4f} "
            f"per_fold={[f'{x:+.3f}' for x in per_fold_pnl]}",
            flush=True,
        )

    # Ensemble sweep
    print(f"\n--- Ensemble ({len(seeds)} seeds) threshold sweep ---", flush=True)
    rows = []
    for T in T_GRID:
        for d_delta in D_GRID:
            per_fold = []
            for k in target_syms:
                blob = fold_blobs[k]
                meta = blob["meta"]
                r = per_fold_metrics(
                    blob["probs"],
                    meta["true_label"].to_numpy(),
                    meta["midprice_t"].to_numpy(),
                    meta["midprice_th"].to_numpy(),
                    T=T, delta=d_delta,
                )
                per_fold.append(r)
            sum_cum = sum(r["cum_pnl"] for r in per_fold)
            std_pnl = float(np.std([r["cum_pnl"] for r in per_fold], ddof=0))
            n_pos = int(sum(1 for r in per_fold if r["cum_pnl"] > 0))
            sum_active = int(sum(r["n_active"] for r in per_fold))
            rows.append({
                "T": T, "delta": d_delta, "sum_cum_pnl": sum_cum,
                "mean_cum_pnl": sum_cum / len(per_fold),
                "std_cum_pnl": std_pnl,
                "n_pos_folds": n_pos,
                "sum_n_active": sum_active,
                **{f"fold{k}_pnl": per_fold[i]["cum_pnl"]
                   for i, k in enumerate(target_syms)},
            })

    sweep_df = pd.DataFrame(rows).sort_values("sum_cum_pnl", ascending=False)
    out_csv = os.path.join(HERE, f"sweep_5seed_ensemble_h{H}.csv")
    sweep_df.to_csv(out_csv, index=False)
    print(f"  saved sweep -> {out_csv}", flush=True)

    print("\n--- Top 10 (T,d) ensemble ---", flush=True)
    print(sweep_df.head(10).to_string(index=False), flush=True)

    best = sweep_df.iloc[0].to_dict()
    print(
        f"\n  BEST: T={best['T']:.2f} d={best['delta']:.2f} "
        f"sum={best['sum_cum_pnl']:+.4f} std={best['std_cum_pnl']:.4f} "
        f"pos={int(best['n_pos_folds'])}/5 active={int(best['sum_n_active'])}",
        flush=True,
    )

    out = {
        "task": "T27 5-seed ensemble OOF threshold sweep h=60",
        "horizon": H, "seeds": seeds,
        "T_grid": T_GRID, "delta_grid": D_GRID,
        "best": {
            "T": float(best["T"]), "delta": float(best["delta"]),
            "sum_cum_pnl": float(best["sum_cum_pnl"]),
            "std_cum_pnl": float(best["std_cum_pnl"]),
            "n_pos_folds": int(best["n_pos_folds"]),
            "sum_n_active": int(best["sum_n_active"]),
            "per_fold_pnl": [float(best[f"fold{k}_pnl"]) for k in target_syms],
        },
        "top10_grid": sweep_df.head(10).to_dict(orient="records"),
        "elapsed_sec": time.time() - t0,
    }
    out_path = os.path.join(HERE, "sweep_5seed_ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
