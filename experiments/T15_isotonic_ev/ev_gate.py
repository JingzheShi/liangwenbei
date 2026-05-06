"""T15 Step 2 & 3: EV-gate sweep on calibrated LOSO OOF.

Pipeline:
  1. Load loso_pred_h{H}_held{K}_cal.parquet (from calibrate.py)
  2. For each held-fold K: estimate E_dp_up / E_dp_down on the OTHER 4 folds (avoid leakage).
     Save deploy E_dp_up_h{H} / E_dp_down_h{H} from ALL 5 folds.
  3. Apply EV-gate decision rule:
        EV(pred=2) = p2_cal * E_up - p0_cal * E_down - 2*fee
        EV(pred=0) = -p2_cal * E_up + p0_cal * E_down - 2*fee
        EV(pred=1) = 0
        argmax → predict
        Add conservatism C: require argmax > C to take the side, else flat.
  4. Sweep C ∈ {0, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3} for each horizon.
  5. Compare:
        baseline: iter_002 raw + (T,δ) gate (from thresholds.json)
        + isotonic + same (T,δ) gate (post-cal threshold)
        + isotonic + EV gate (C ∈ sweep)

Compliance: sym-agnostic (E_dp constants are global).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from src.eval.pnl import _per_horizon_metrics  # noqa: E402

HORIZONS = (5, 10, 20, 40, 60)
FEE = 0.0001
C_GRID = [0.0, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]

# iter_002 thresholds (from submission/iter_002_lgbm_schemeC/thresholds.json)
ITER002_TD = {
    5: (0.60, 0.10),
    10: (0.55, 0.10),
    20: (0.50, 0.05),
    40: (0.50, 0.00),
    60: (0.50, 0.20),
}


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


def threshold_pred(probs: np.ndarray, T: float, delta: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T) & (side_max > p1 + delta)
    side = np.where(p2 > p0, 2, 0)
    return np.where(take, side, 1).astype(np.int64)


def ev_gate_pred(probs: np.ndarray, e_up: float, e_dn: float, fee: float, c: float) -> np.ndarray:
    """probs: (N, 3) calibrated. Returns (N,) int predictions."""
    p0 = probs[:, 0]; p2 = probs[:, 2]
    ev_up = p2 * e_up - p0 * e_dn - 2.0 * fee
    ev_dn = -p2 * e_up + p0 * e_dn - 2.0 * fee
    # ev_flat = 0
    pred = np.full(probs.shape[0], 1, dtype=np.int64)
    take_up = (ev_up > c) & (ev_up > ev_dn)
    take_dn = (ev_dn > c) & (ev_dn > ev_up)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def fold_pnl(df: pd.DataFrame, pred: np.ndarray) -> dict:
    m = _per_horizon_metrics(
        pred,
        df["true_label"].to_numpy(np.int64),
        df["midprice_t"].to_numpy(np.float64),
        df["midprice_th"].to_numpy(np.float64),
        fee_rate=FEE,
    )
    return {
        "cum_pnl": float(m["cum_pnl"]),
        "n_active": int(m["n_predictions_active"]),
        "accuracy": float(m["accuracy"]),
    }


def main():
    t0 = time.time()
    progress("start_ev_gate")

    # Try wandb (best-effort)
    WANDB_OK = False
    try:
        import wandb
        wandb.init(
            project="liangwenbei",
            entity="cjxh21-Tsinghua University",
            name="T15-isotonic-ev",
            config={"horizons": HORIZONS, "C_grid": C_GRID, "fee": FEE},
            reinit=True,
        )
        WANDB_OK = True
    except Exception as e:
        print(f"[wandb] disabled: {e}", flush=True)

    summary: dict = {"horizons": {}, "config": {"C_grid": C_GRID, "fee": FEE}}
    deploy_e_dp: dict = {}

    # Load calibrated OOF for each horizon
    for H in HORIZONS:
        progress(f"sweep_h{H}")
        print(f"\n=== Horizon h={H} ===", flush=True)
        fold_dfs = []
        for k in range(5):
            p = os.path.join(HERE, f"loso_pred_h{H}_held{k}_cal.parquet")
            fold_dfs.append(pd.read_parquet(p))

        # Compute deploy E_dp from ALL folds (sym-agnostic, global)
        all_df = pd.concat(fold_dfs, ignore_index=True)
        diff_all = (all_df["midprice_th"] - all_df["midprice_t"]).to_numpy(np.float64)
        true_all = all_df["true_label"].to_numpy(np.int64)
        e_up_global = float(diff_all[true_all == 2].mean()) if (true_all == 2).any() else 0.0
        # E_down is the magnitude of negative move (positive value)
        e_dn_global = float(-diff_all[true_all == 0].mean()) if (true_all == 0).any() else 0.0
        deploy_e_dp[H] = (e_up_global, e_dn_global)
        print(f"  deploy global: E_up={e_up_global:.6f} E_dn={e_dn_global:.6f}", flush=True)

        # Per-fold leak-safe estimates
        e_up_fold = []
        e_dn_fold = []
        for k in range(5):
            others = pd.concat([fold_dfs[j] for j in range(5) if j != k], ignore_index=True)
            d = (others["midprice_th"] - others["midprice_t"]).to_numpy(np.float64)
            y = others["true_label"].to_numpy(np.int64)
            eu = float(d[y == 2].mean()) if (y == 2).any() else 0.0
            ed = float(-d[y == 0].mean()) if (y == 0).any() else 0.0
            e_up_fold.append(eu)
            e_dn_fold.append(ed)

        # ----- Variant A: iter_002 baseline (pre-cal) + (T, δ) gate -----
        T_iter, d_iter = ITER002_TD[H]
        baseline_pf = []
        for k in range(5):
            df = fold_dfs[k]
            probs_orig = df[["prob_0", "prob_1", "prob_2"]].to_numpy(np.float32)
            pred = threshold_pred(probs_orig, T_iter, d_iter)
            baseline_pf.append(fold_pnl(df, pred))
        baseline_sum = sum(r["cum_pnl"] for r in baseline_pf)
        baseline_pos = sum(1 for r in baseline_pf if r["cum_pnl"] > 0)
        print(f"  iter_002 (T={T_iter}, δ={d_iter}): sum={baseline_sum:+.4f} pos={baseline_pos}/5", flush=True)

        # ----- Variant B: cal + same (T, δ) gate -----
        cal_thresh_pf = []
        for k in range(5):
            df = fold_dfs[k]
            probs_cal = df[["prob_0_cal", "prob_1_cal", "prob_2_cal"]].to_numpy(np.float32)
            pred = threshold_pred(probs_cal, T_iter, d_iter)
            cal_thresh_pf.append(fold_pnl(df, pred))
        cal_thresh_sum = sum(r["cum_pnl"] for r in cal_thresh_pf)
        cal_thresh_pos = sum(1 for r in cal_thresh_pf if r["cum_pnl"] > 0)
        print(f"  cal + (T={T_iter}, δ={d_iter}): sum={cal_thresh_sum:+.4f} pos={cal_thresh_pos}/5", flush=True)

        # ----- Variant C: cal + EV gate sweep -----
        ev_results = []
        for c in C_GRID:
            pf = []
            for k in range(5):
                df = fold_dfs[k]
                probs_cal = df[["prob_0_cal", "prob_1_cal", "prob_2_cal"]].to_numpy(np.float32)
                pred = ev_gate_pred(probs_cal, e_up_fold[k], e_dn_fold[k], FEE, c)
                pf.append(fold_pnl(df, pred))
            s = sum(r["cum_pnl"] for r in pf)
            n_pos = sum(1 for r in pf if r["cum_pnl"] > 0)
            n_active = sum(r["n_active"] for r in pf)
            ev_results.append({
                "C": c,
                "sum_cum_pnl": s,
                "n_pos": n_pos,
                "n_active": n_active,
                "per_fold": pf,
            })
            print(f"  cal + EV(C={c}): sum={s:+.4f} pos={n_pos}/5 active={n_active}", flush=True)
            if WANDB_OK:
                wandb.log({
                    f"h{H}/ev_C": c,
                    f"h{H}/ev_sum_pnl": s,
                    f"h{H}/ev_n_pos": n_pos,
                    f"h{H}/ev_n_active": n_active,
                })

        # Best EV
        best_ev = max(ev_results, key=lambda r: r["sum_cum_pnl"])
        print(f"  BEST EV: C={best_ev['C']} sum={best_ev['sum_cum_pnl']:+.4f} pos={best_ev['n_pos']}/5", flush=True)

        summary["horizons"][f"h_{H}"] = {
            "iter_002_baseline": {
                "T": T_iter,
                "delta": d_iter,
                "sum_cum_pnl": baseline_sum,
                "n_pos": baseline_pos,
                "per_fold": baseline_pf,
            },
            "cal_thresh": {
                "T": T_iter,
                "delta": d_iter,
                "sum_cum_pnl": cal_thresh_sum,
                "n_pos": cal_thresh_pos,
                "per_fold": cal_thresh_pf,
            },
            "ev_sweep": ev_results,
            "ev_best": best_ev,
            "e_up_global": e_up_global,
            "e_dn_global": e_dn_global,
            "e_up_fold": e_up_fold,
            "e_dn_fold": e_dn_fold,
        }
        if WANDB_OK:
            wandb.log({
                f"h{H}/baseline_sum_pnl": baseline_sum,
                f"h{H}/cal_thresh_sum_pnl": cal_thresh_sum,
                f"h{H}/best_ev_sum_pnl": best_ev["sum_cum_pnl"],
                f"h{H}/best_ev_C": best_ev["C"],
                f"h{H}/delta_vs_iter002": best_ev["sum_cum_pnl"] - baseline_sum,
            })

    # Save deploy E_dp
    deploy_p = os.path.join(HERE, "deploy_e_dp.json")
    with open(deploy_p, "w") as f:
        json.dump({str(H): {"E_up": float(deploy_e_dp[H][0]), "E_dn": float(deploy_e_dp[H][1])}
                   for H in HORIZONS}, f, indent=2)
    print(f"\ndeploy E_dp saved -> {deploy_p}", flush=True)

    summary["elapsed_sec"] = time.time() - t0

    # Best per horizon: pick variant with highest sum
    decisions = {}
    print("\n=== SUMMARY: best variant per horizon ===", flush=True)
    for H in HORIZONS:
        h = summary["horizons"][f"h_{H}"]
        cands = [
            ("iter_002", h["iter_002_baseline"]["sum_cum_pnl"]),
            ("cal_thresh", h["cal_thresh"]["sum_cum_pnl"]),
            ("cal_ev_best", h["ev_best"]["sum_cum_pnl"]),
        ]
        cands.sort(key=lambda x: x[1], reverse=True)
        delta_vs_iter002 = cands[0][1] - h["iter_002_baseline"]["sum_cum_pnl"]
        decisions[f"h_{H}"] = {
            "best_variant": cands[0][0],
            "best_sum": cands[0][1],
            "iter_002_sum": h["iter_002_baseline"]["sum_cum_pnl"],
            "delta_vs_iter002": delta_vs_iter002,
        }
        print(f"  h_{H}: iter_002={cands[2][1]:+.4f}  cal_thresh={[c for c in cands if c[0]=='cal_thresh'][0][1]:+.4f}  cal_ev={[c for c in cands if c[0]=='cal_ev_best'][0][1]:+.4f}  → BEST={cands[0][0]} ({cands[0][1]:+.4f})  Δ={delta_vs_iter002:+.4f}", flush=True)
    summary["decisions"] = decisions

    h10_dec = decisions["h_10"]
    if h10_dec["best_sum"] > 22.5 and h10_dec["best_variant"] != "iter_002":
        summary["build_iter008"] = True
        summary["iter008_h10_best_sum"] = h10_dec["best_sum"]
        print(f"\n✓ h_10 best ({h10_dec['best_variant']}) = {h10_dec['best_sum']:+.4f} > +22.5 → BUILD iter_008", flush=True)
    else:
        summary["build_iter008"] = False
        print(f"\n✗ h_10 best = {h10_dec['best_sum']:+.4f} ≤ +22.5 OR iter_002 → NO iter_008", flush=True)

    with open(os.path.join(HERE, "ev_sweep_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nev_sweep_results.json saved (elapsed {summary['elapsed_sec']:.1f}s)", flush=True)

    if WANDB_OK:
        for H in HORIZONS:
            d = decisions[f"h_{H}"]
            wandb.summary[f"h{H}_best_variant"] = d["best_variant"]
            wandb.summary[f"h{H}_best_sum"] = d["best_sum"]
            wandb.summary[f"h{H}_delta_vs_iter002"] = d["delta_vs_iter002"]
        wandb.summary["build_iter008"] = bool(summary["build_iter008"])
        wandb.finish()

    progress("ev_gate_done", build_iter008=bool(summary["build_iter008"]))


if __name__ == "__main__":
    main()
