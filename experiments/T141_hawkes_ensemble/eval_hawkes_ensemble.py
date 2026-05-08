"""T141: Ensemble R_hawkes (3-seed) with iter_018 v1 stack (T87 NN + T75 LGB).

Tests 4 configs (A/B/C/D), each with and without iter_018 conformal wrapper.
Uses DE 2D (T_up, T_dn) asymmetric threshold search, LOSO PnL evaluation.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")
T87_DIR = os.path.join(ROOT, "experiments", "T87_spo_dfl")
HAWKES_DIR = os.path.join(ROOT, "experiments", "R_hawkes")

SEEDS_T75 = (1, 7, 13, 42, 100)
SEEDS_T87 = (1, 7, 13, 42, 100)
SEEDS_HAWKES = (1, 7, 42)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001

# iter_018 v1 conformal per-sym bands (beta * sigma per sym)
CONFORMAL_BAND = {
    0: 2.403901e-05,
    1: 1.884959e-04,
    2: 1.356665e-04,
    3: 0.0,
    4: 0.0,
}


def load_avg_preds(dir_path: str, pattern: str, seeds: tuple) -> np.ndarray:
    first = pd.read_parquet(os.path.join(dir_path, pattern.format(s=seeds[0])))
    p = first["pred_dmid_norm"].to_numpy(np.float64)
    for s in seeds[1:]:
        df = pd.read_parquet(os.path.join(dir_path, pattern.format(s=s)))
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    return p / len(seeds)


def load_meta(dir_path: str, pattern: str, seed: int):
    df = pd.read_parquet(os.path.join(dir_path, pattern.format(s=seed)))
    return (
        df["sym"].to_numpy(np.int8),
        df["midprice_t"].to_numpy(np.float64),
        df["midprice_th"].to_numpy(np.float64),
    )


def vectorized_pnl(action: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    side = action.astype(np.float64) - 1.0
    diff = mp_th - mp_t
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    return (side * diff - fee) / denom


def gate_2d(p: np.ndarray, tu: float, td: float) -> np.ndarray:
    a = np.full(len(p), 1, dtype=np.int8)
    a[p > tu] = 2
    a[p < -td] = 0
    return a


def gate_2d_conformal(p: np.ndarray, sym: np.ndarray, tu: float, td: float) -> np.ndarray:
    a = np.full(len(p), 1, dtype=np.int8)
    for s, band in CONFORMAL_BAND.items():
        mask = (sym == s)
        a[mask & (p > (tu + band))] = 2
        a[mask & (p < -(td + band))] = 0
    return a


def loso_pnl(p: np.ndarray, sym: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray,
             tu: float, td: float, conformal: bool = False) -> float:
    total = 0.0
    for s in SYMS:
        m = (sym == s)
        if conformal:
            a = gate_2d_conformal(p[m], sym[m], tu, td)
        else:
            a = gate_2d(p[m], tu, td)
        total += vectorized_pnl(a, mp_t[m], mp_th[m]).sum()
    return float(total)


def per_sym_breakdown(p: np.ndarray, sym: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray,
                      tu: float, td: float, conformal: bool = False) -> dict:
    out = {}
    for s in SYMS:
        m = (sym == s)
        if conformal:
            a = gate_2d_conformal(p[m], sym[m], tu, td)
        else:
            a = gate_2d(p[m], tu, td)
        pnl = vectorized_pnl(a, mp_t[m], mp_th[m]).sum()
        out[f"sym{s}"] = float(pnl)
        out[f"sym{s}_n_active"] = int((a != 1).sum())
    return out


def fit_de(p: np.ndarray, sym: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray,
           conformal: bool = False, de_seeds=(0, 42, 1)) -> tuple:
    def obj(x):
        tu, td = x
        return -loso_pnl(p, sym, mp_t, mp_th, tu, td, conformal=conformal)

    best_x, best_y = None, np.inf
    for ds in de_seeds:
        r = differential_evolution(
            obj,
            bounds=[(0.0, 0.008), (0.0, 0.008)],
            seed=ds, maxiter=120, popsize=32,
            tol=1e-7, polish=True, workers=1, init="sobol",
        )
        if r.fun < best_y:
            best_y = float(r.fun)
            best_x = (float(r.x[0]), float(r.x[1]))
    return best_x, -best_y


def main():
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] T141: Hawkes ensemble evaluation")

    # WandB init
    run = None
    if WANDB_AVAILABLE:
        try:
            run = wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name="T141-hawkes-ensemble-iter018",
                config={
                    "task": "T141",
                    "seeds_t75": list(SEEDS_T75),
                    "seeds_t87": list(SEEDS_T87),
                    "seeds_hawkes": list(SEEDS_HAWKES),
                },
            )
        except Exception as e:
            print(f"  WandB init failed: {e}")

    # Update progress
    def update_progress(step, metrics=None):
        prog = {
            "status": "running",
            "step": step,
            "metrics": metrics or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(os.path.join(HERE, "worker-progress.json"), "w") as f:
            json.dump(prog, f, indent=2)
        print(f"  >> {step}")

    update_progress("loading predictions")

    # Load averaged predictions
    print("\n=== Loading predictions ===")
    p_t87 = load_avg_preds(T87_DIR, "pred_T87_seed{s}_main.parquet", SEEDS_T87)
    print(f"  T87 avg (5-seed): min={p_t87.min():.5f} max={p_t87.max():.5f}")

    p_t75 = load_avg_preds(T75_DIR, "pred_T75_seed{s}.parquet", SEEDS_T75)
    print(f"  T75 avg (5-seed): min={p_t75.min():.5f} max={p_t75.max():.5f}")

    p_hawkes = load_avg_preds(HAWKES_DIR, "pred_hawkes_seed{s}.parquet", SEEDS_HAWKES)
    print(f"  Hawkes avg (3-seed): min={p_hawkes.min():.5f} max={p_hawkes.max():.5f}")

    # Load meta from T75 seed1 (ground truth / market data)
    sym, mp_t, mp_th = load_meta(T75_DIR, "pred_T75_seed{s}.parquet", 1)
    print(f"  Loaded meta: {len(sym):,} rows, syms={sorted(set(sym.tolist()))}")

    # Verify alignment
    sym_h, _, _ = load_meta(HAWKES_DIR, "pred_hawkes_seed{s}.parquet", 1)
    assert np.array_equal(sym, sym_h), "sym mismatch T75 vs hawkes"

    # Define configs
    configs = {
        "A_iter018_baseline": (1.0 * p_t87 + 1.5 * p_t75) / 2.5,
        "B_replace_t75_with_hawkes": (1.0 * p_t87 + 1.5 * p_hawkes) / 2.5,
        "C_3way_hawkes_0.7": (1.0 * p_t87 + 1.5 * p_t75 + 0.7 * p_hawkes) / 3.2,
        "D_3way_equal": (1.0 * p_t87 + 1.0 * p_t75 + 1.0 * p_hawkes) / 3.0,
    }

    results = {
        "task": "T141: Hawkes ensemble evaluation",
        "configs": {},
        "baseline_iter018_v1": 41.4939,
        "baseline_iter018_v1_conformal": 41.1,
    }

    print("\n=== Evaluating configs ===")
    for cfg_name, p_blend in configs.items():
        print(f"\n--- Config: {cfg_name} ---")
        update_progress(f"evaluating {cfg_name}")

        cfg_result = {}

        # Without conformal
        print("  DE without conformal...")
        (tu, td), pnl_raw = fit_de(p_blend, sym, mp_t, mp_th, conformal=False)
        per_sym_raw = per_sym_breakdown(p_blend, sym, mp_t, mp_th, tu, td, conformal=False)
        n_active_raw = sum(per_sym_raw[f"sym{s}_n_active"] for s in SYMS)
        print(f"  T_up={tu:.6f}, T_dn={td:.6f}")
        print(f"  LOSO PnL (no conformal): {pnl_raw:+.4f}")
        print(f"  Per-sym: {[round(per_sym_raw[f'sym{s}'], 2) for s in SYMS]}")
        print(f"  N active: {n_active_raw:,}")

        cfg_result["no_conformal"] = {
            "loso_pnl": pnl_raw,
            "thresh_up": tu,
            "thresh_dn": td,
            "per_sym": per_sym_raw,
            "n_active": n_active_raw,
        }

        # With conformal (re-optimize thresholds with conformal constraint)
        print("  DE with conformal...")
        (tu_c, td_c), pnl_conf = fit_de(p_blend, sym, mp_t, mp_th, conformal=True)
        per_sym_conf = per_sym_breakdown(p_blend, sym, mp_t, mp_th, tu_c, td_c, conformal=True)
        n_active_conf = sum(per_sym_conf[f"sym{s}_n_active"] for s in SYMS)
        print(f"  T_up={tu_c:.6f}, T_dn={td_c:.6f}")
        print(f"  LOSO PnL (conformal): {pnl_conf:+.4f}")
        print(f"  Per-sym: {[round(per_sym_conf[f'sym{s}'], 2) for s in SYMS]}")
        print(f"  N active: {n_active_conf:,}")

        cfg_result["conformal"] = {
            "loso_pnl": pnl_conf,
            "thresh_up": tu_c,
            "thresh_dn": td_c,
            "per_sym": per_sym_conf,
            "n_active": n_active_conf,
        }

        delta_no_conf = pnl_raw - results["baseline_iter018_v1"]
        delta_conf = pnl_conf - results["baseline_iter018_v1"]
        cfg_result["delta_vs_iter018_no_conformal"] = delta_no_conf
        cfg_result["delta_vs_iter018_conformal_applied"] = delta_conf

        results["configs"][cfg_name] = cfg_result

        if run:
            wandb.log({
                f"{cfg_name}/pnl_no_conformal": pnl_raw,
                f"{cfg_name}/pnl_conformal": pnl_conf,
                f"{cfg_name}/delta_no_conformal": delta_no_conf,
                f"{cfg_name}/delta_conformal": delta_conf,
            })

    # Find best config
    best_name = None
    best_pnl = -np.inf
    for cfg_name, cfg_r in results["configs"].items():
        max_pnl = max(cfg_r["no_conformal"]["loso_pnl"], cfg_r["conformal"]["loso_pnl"])
        if max_pnl > best_pnl:
            best_pnl = max_pnl
            best_name = cfg_name

    best_cfg = results["configs"][best_name]
    best_mode = "conformal" if best_cfg["conformal"]["loso_pnl"] >= best_cfg["no_conformal"]["loso_pnl"] else "no_conformal"
    best_pnl_val = best_cfg[best_mode]["loso_pnl"]
    delta_over_baseline = best_pnl_val - results["baseline_iter018_v1"]

    print(f"\n=== BEST CONFIG: {best_name} ({best_mode}) ===")
    print(f"  LOSO PnL: {best_pnl_val:+.4f}")
    print(f"  Delta vs iter_018 v1 (+41.4939): {delta_over_baseline:+.4f}")

    results["best"] = {
        "config": best_name,
        "mode": best_mode,
        "loso_pnl": best_pnl_val,
        "delta_vs_iter018_v1": delta_over_baseline,
        "per_sym": best_cfg[best_mode]["per_sym"],
    }

    # iter_019 v3 recommendation
    iter019_v3 = None
    THRESHOLD_FOR_V3 = 0.5
    if delta_over_baseline > THRESHOLD_FOR_V3:
        best_r = best_cfg[best_mode]
        iter019_v3 = {
            "candidate": True,
            "based_on_config": best_name,
            "mode": best_mode,
            "ensemble_weights": _get_weights(best_name),
            "thresh_up": best_r["thresh_up"],
            "thresh_dn": best_r["thresh_dn"],
            "conformal_bands": CONFORMAL_BAND if best_mode == "conformal" else None,
            "expected_loso_pnl": best_pnl_val,
            "delta_vs_iter018_v1": delta_over_baseline,
            "note": (
                f"R_hawkes 3-seed avg blended with T87 NN 5-seed + T75 LGB 5-seed "
                f"per config {best_name}. Best DE thresholds re-optimized for this blend."
            ),
        }
        print(f"\n  *** iter_019 v3 CANDIDATE: delta={delta_over_baseline:+.4f} > {THRESHOLD_FOR_V3} ***")
    else:
        iter019_v3 = {
            "candidate": False,
            "reason": f"Best delta {delta_over_baseline:+.4f} <= threshold {THRESHOLD_FOR_V3}",
        }
        print(f"\n  No iter_019 v3 candidate: delta={delta_over_baseline:+.4f}")

    results["iter019_v3_recommendation"] = iter019_v3

    # Transmission estimate
    results["transmission_note"] = (
        "iter_018 v1 local LOSO = +41.49, platform SOTA = +28.93 "
        "(transmission ~0.70x). If hawkes blending gives +X local, "
        "estimate platform = +28.93 + X * 0.70."
    )
    if delta_over_baseline > 0:
        est_platform_delta = delta_over_baseline * 0.70
        results["estimated_platform_delta"] = est_platform_delta
        results["estimated_platform_pnl"] = 28.93 + est_platform_delta

    # Save results
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Final progress
    with open(os.path.join(HERE, "worker-progress.json"), "w") as f:
        json.dump({
            "status": "completed",
            "step": "done",
            "metrics": {
                "best_config": best_name,
                "best_pnl": best_pnl_val,
                "delta_vs_iter018": delta_over_baseline,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }, f, indent=2)

    if run:
        wandb.log({
            "best/config": best_name,
            "best/loso_pnl": best_pnl_val,
            "best/delta_vs_iter018": delta_over_baseline,
        })
        wandb.finish()

    return results


def _get_weights(cfg_name: str) -> dict:
    if "A_iter018" in cfg_name:
        return {"t87": 1.0, "t75": 1.5, "hawkes": 0.0, "total": 2.5}
    elif "B_replace" in cfg_name:
        return {"t87": 1.0, "t75": 0.0, "hawkes": 1.5, "total": 2.5}
    elif "C_3way_hawkes_0.7" in cfg_name:
        return {"t87": 1.0, "t75": 1.5, "hawkes": 0.7, "total": 3.2}
    elif "D_3way_equal" in cfg_name:
        return {"t87": 1.0, "t75": 1.0, "hawkes": 1.0, "total": 3.0}
    return {}


if __name__ == "__main__":
    results = main()
    best = results.get("best", {})
    cfg = best.get("config", "unknown")
    mode = best.get("mode", "unknown")
    pnl = best.get("loso_pnl", 0.0)
    delta = best.get("delta_vs_iter018_v1", 0.0)
    is_cand = results.get("iter019_v3_recommendation", {}).get("candidate", False)
    print(f"\nRESULT: task=[T141 hawkes ensemble] metrics={{best_config={cfg},best_pnl={pnl:.4f},delta_vs_iter018={delta:+.4f},iter019_v3_candidate={is_cand}}} notes=[best={cfg}({mode}) delta={delta:+.4f}]")
