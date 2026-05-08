"""T141b: Hawkes ensemble Configs B/C/D only (A already done in T141).

A: +40.12 / +41.92 (no/with conformal) — skip.
Evaluate B/C/D with tighter DE budget to avoid OOM:
  maxiter=50, popsize=16, 3 DE seeds.
For best of B/C/D: also eval WITH conformal.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from datetime import datetime, timezone

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

# iter_018 v1 per-sym conformal bands (beta * sigma) — same as T141
CONFORMAL_BAND = {
    0: 2.403901e-05,
    1: 1.884959e-04,
    2: 1.356665e-04,
    3: 0.0,
    4: 0.0,
}

# A results (already done)
A_PNL_NO_CONF = 40.12
A_PNL_CONF = 41.92


def now():
    return datetime.now(timezone.utc).isoformat()


def upd(step, metrics=None):
    prog = {"status": "running", "step": step, "metrics": metrics or {}, "timestamp": now()}
    with open(os.path.join(HERE, "worker-progress.json"), "w") as f:
        json.dump(prog, f, indent=2)
    print(f"  [{now()[:19]}] {step}")


def load_avg_preds(dir_path: str, pattern: str, seeds: tuple) -> np.ndarray:
    first = pd.read_parquet(os.path.join(dir_path, pattern.format(s=seeds[0])))
    p = first["pred_dmid_norm"].to_numpy(np.float32)
    del first
    for s in seeds[1:]:
        df = pd.read_parquet(os.path.join(dir_path, pattern.format(s=s)))
        p += df["pred_dmid_norm"].to_numpy(np.float32)
        del df
    return (p / len(seeds)).astype(np.float64)


def load_meta(dir_path: str, pattern: str, seed: int):
    df = pd.read_parquet(os.path.join(dir_path, pattern.format(s=seed)))
    sym = df["sym"].to_numpy(np.int8)
    mp_t = df["midprice_t"].to_numpy(np.float64)
    mp_th = df["midprice_th"].to_numpy(np.float64)
    del df
    return sym, mp_t, mp_th


def vectorized_pnl(action: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    side = action.astype(np.float64) - 1.0
    diff = mp_th - mp_t
    fee = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    return (side * diff - fee) / (mp_t + 1.0)


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


def loso_pnl_val(p: np.ndarray, sym: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray,
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


def per_sym_breakdown(p, sym, mp_t, mp_th, tu, td, conformal=False):
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


def fit_de(p, sym, mp_t, mp_th, conformal=False, de_seeds=(0, 42, 1)):
    """Tight DE: maxiter=50, popsize=16, 3 seeds."""
    def obj(x):
        tu, td = x
        return -loso_pnl_val(p, sym, mp_t, mp_th, tu, td, conformal=conformal)

    best_x, best_y = None, np.inf
    for ds in de_seeds:
        r = differential_evolution(
            obj,
            bounds=[(0.0, 0.008), (0.0, 0.008)],
            seed=ds, maxiter=50, popsize=16,
            tol=1e-7, polish=True, workers=1, init="sobol",
        )
        if r.fun < best_y:
            best_y = float(r.fun)
            best_x = (float(r.x[0]), float(r.x[1]))
    return best_x, -best_y


def main():
    print(f"[{now()[:19]}] T141b: Hawkes ensemble B/C/D evaluation (A already done)")
    upd("starting")

    run = None
    if WANDB_AVAILABLE:
        try:
            run = wandb.init(
                project="liangwenbei",
                entity="cjxh21-Tsinghua University",
                name="T141b-hawkes-ensemble-BCD",
                config={
                    "task": "T141b",
                    "skip_config_A": True,
                    "de_maxiter": 50,
                    "de_popsize": 16,
                    "de_seeds": [0, 42, 1],
                },
            )
        except Exception as e:
            print(f"  WandB init failed: {e}")

    upd("loading predictions")

    print("\n=== Loading predictions ===")
    p_t87 = load_avg_preds(T87_DIR, "pred_T87_seed{s}_main.parquet", SEEDS_T87)
    print(f"  T87 avg (5-seed): n={len(p_t87):,}")
    p_t75 = load_avg_preds(T75_DIR, "pred_T75_seed{s}.parquet", SEEDS_T75)
    print(f"  T75 avg (5-seed): n={len(p_t75):,}")
    p_hawkes = load_avg_preds(HAWKES_DIR, "pred_hawkes_seed{s}.parquet", SEEDS_HAWKES)
    print(f"  Hawkes avg (3-seed): n={len(p_hawkes):,}")

    sym, mp_t, mp_th = load_meta(T75_DIR, "pred_T75_seed{s}.parquet", 1)
    print(f"  Meta: {len(sym):,} rows")

    # Verify alignment
    sym_h, _, _ = load_meta(HAWKES_DIR, "pred_hawkes_seed{s}.parquet", 1)
    assert np.array_equal(sym, sym_h), "sym mismatch T75 vs hawkes"
    del sym_h
    gc.collect()

    # Configs B/C/D
    configs = {
        "B_replace_t75_with_hawkes": (1.0 * p_t87 + 1.5 * p_hawkes) / 2.5,
        "C_3way_hawkes_0.7": (1.0 * p_t87 + 1.5 * p_t75 + 0.7 * p_hawkes) / 3.2,
        "D_3way_equal": (1.0 * p_t87 + 1.0 * p_t75 + 1.0 * p_hawkes) / 3.0,
    }
    # Free raw preds now that blends are made
    del p_t87, p_t75, p_hawkes
    gc.collect()

    results = {
        "task": "T141b: Hawkes ensemble B/C/D eval",
        "config_A_already_done": {"no_conformal": A_PNL_NO_CONF, "conformal": A_PNL_CONF},
        "configs": {},
    }

    print("\n=== Evaluating Configs B/C/D (no conformal) ===")
    for cfg_name, p_blend in configs.items():
        print(f"\n--- Config: {cfg_name} ---")
        upd(f"DE no-conformal {cfg_name}")

        (tu, td), pnl_raw = fit_de(p_blend, sym, mp_t, mp_th, conformal=False)
        per_sym_raw = per_sym_breakdown(p_blend, sym, mp_t, mp_th, tu, td, conformal=False)
        n_active = sum(per_sym_raw[f"sym{s}_n_active"] for s in SYMS)
        print(f"  T_up={tu:.6f}, T_dn={td:.6f}")
        print(f"  LOSO PnL (no conformal): {pnl_raw:+.4f}")
        print(f"  Per-sym: {[round(per_sym_raw[f'sym{s}'], 2) for s in SYMS]}")
        print(f"  N active: {n_active:,}")

        results["configs"][cfg_name] = {
            "no_conformal": {
                "loso_pnl": pnl_raw,
                "thresh_up": tu,
                "thresh_dn": td,
                "per_sym": per_sym_raw,
                "n_active": n_active,
            },
            "delta_vs_A_no_conf": pnl_raw - A_PNL_NO_CONF,
        }
        if run:
            wandb.log({f"{cfg_name}/pnl_no_conformal": pnl_raw,
                       f"{cfg_name}/delta_vs_A": pnl_raw - A_PNL_NO_CONF})

        gc.collect()

    # Identify best (no-conformal) of B/C/D
    best_name = max(results["configs"], key=lambda k: results["configs"][k]["no_conformal"]["loso_pnl"])
    best_pnl_raw = results["configs"][best_name]["no_conformal"]["loso_pnl"]
    print(f"\n=== Best of B/C/D (no conformal): {best_name} = {best_pnl_raw:+.4f} ===")

    # Eval conformal for best config
    print(f"\n=== Evaluating conformal for best config: {best_name} ===")
    upd(f"DE conformal {best_name}")

    p_best = configs[best_name]
    (tu_c, td_c), pnl_conf = fit_de(p_best, sym, mp_t, mp_th, conformal=True)
    per_sym_conf = per_sym_breakdown(p_best, sym, mp_t, mp_th, tu_c, td_c, conformal=True)
    n_active_conf = sum(per_sym_conf[f"sym{s}_n_active"] for s in SYMS)
    print(f"  T_up={tu_c:.6f}, T_dn={td_c:.6f}")
    print(f"  LOSO PnL (conformal): {pnl_conf:+.4f}")
    print(f"  Per-sym: {[round(per_sym_conf[f'sym{s}'], 2) for s in SYMS]}")
    print(f"  N active: {n_active_conf:,}")

    results["configs"][best_name]["conformal"] = {
        "loso_pnl": pnl_conf,
        "thresh_up": tu_c,
        "thresh_dn": td_c,
        "per_sym": per_sym_conf,
        "n_active": n_active_conf,
    }
    results["configs"][best_name]["delta_vs_A_conf"] = pnl_conf - A_PNL_CONF

    if run:
        wandb.log({f"{best_name}/pnl_conformal": pnl_conf})

    del configs
    gc.collect()

    # Summary
    best_mode_pnl = max(best_pnl_raw, pnl_conf)
    best_mode = "conformal" if pnl_conf >= best_pnl_raw else "no_conformal"
    # Compare to A no-conformal (40.12) and A conformal (41.92)
    delta_vs_A_no_conf = best_pnl_raw - A_PNL_NO_CONF
    delta_vs_A_conf = pnl_conf - A_PNL_CONF

    print(f"\n=== SUMMARY ===")
    print(f"  Config A (baseline): {A_PNL_NO_CONF:+.2f} / {A_PNL_CONF:+.2f} (no/with conformal)")
    for cfg, v in results["configs"].items():
        nc = v["no_conformal"]["loso_pnl"]
        cf = v.get("conformal", {}).get("loso_pnl", "—")
        cf_s = f"{cf:+.4f}" if isinstance(cf, float) else cf
        print(f"  {cfg}: {nc:+.4f} (no conf) / {cf_s} (conf)")
    print(f"  Best of BCD: {best_name} [{best_mode}] = {best_mode_pnl:+.4f}")
    print(f"  Delta vs A (no-conf): {delta_vs_A_no_conf:+.4f}")
    print(f"  Delta vs A (conf):    {delta_vs_A_conf:+.4f}")

    results["best"] = {
        "config": best_name,
        "mode": best_mode,
        "loso_pnl": best_mode_pnl,
        "delta_vs_A_no_conformal": delta_vs_A_no_conf,
        "delta_vs_A_conformal": delta_vs_A_conf,
    }

    # iter_019 v3 recommendation
    THRESHOLD = 0.5
    beat_A_conf = pnl_conf - A_PNL_CONF
    best_delta = max(delta_vs_A_no_conf, beat_A_conf)
    if best_delta > THRESHOLD:
        best_r = results["configs"][best_name][best_mode]
        results["iter019_v3_recommended"] = {
            "candidate": True,
            "config": best_name,
            "mode": best_mode,
            "weights": _get_weights(best_name),
            "thresh_up": best_r["thresh_up"],
            "thresh_dn": best_r["thresh_dn"],
            "conformal_bands": CONFORMAL_BAND if best_mode == "conformal" else None,
            "expected_loso_pnl": best_mode_pnl,
            "delta_vs_A": best_delta,
            "transmission_estimate": {
                "local_baseline_A_conf": A_PNL_CONF,
                "platform_SOTA": 28.93,
                "expected_platform_delta": round(beat_A_conf * 0.70, 3),
                "expected_platform_pnl": round(28.93 + beat_A_conf * 0.70, 3),
            },
            "recommendation": "SHIP" if beat_A_conf > THRESHOLD else "MARGINAL",
        }
        print(f"\n  *** iter_019 v3 CANDIDATE: best_delta={best_delta:+.4f} > {THRESHOLD} ***")
    else:
        results["iter019_v3_recommended"] = {
            "candidate": False,
            "reason": f"best_delta {best_delta:+.4f} <= threshold {THRESHOLD} — no improvement over A",
        }
        print(f"\n  No iter_019 v3 candidate (best_delta={best_delta:+.4f})")

    # Save results
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    with open(os.path.join(HERE, "worker-progress.json"), "w") as f:
        json.dump({
            "status": "completed", "step": "done",
            "metrics": {"best_config": best_name, "best_pnl": best_mode_pnl, "delta_vs_A": best_delta},
            "timestamp": now(),
        }, f, indent=2)

    if run:
        wandb.log({"best/config_name": best_name, "best/pnl": best_mode_pnl, "best/delta_vs_A": best_delta})
        wandb.finish()

    return results, best_name, best_mode, best_mode_pnl, best_delta


def _get_weights(cfg_name):
    if "B_replace" in cfg_name:
        return {"t87": 1.0, "t75": 0.0, "hawkes": 1.5, "total": 2.5}
    elif "C_3way" in cfg_name:
        return {"t87": 1.0, "t75": 1.5, "hawkes": 0.7, "total": 3.2}
    elif "D_3way" in cfg_name:
        return {"t87": 1.0, "t75": 1.0, "hawkes": 1.0, "total": 3.0}
    return {}


if __name__ == "__main__":
    results, best_name, best_mode, best_pnl, best_delta = main()
    is_cand = results.get("iter019_v3_recommended", {}).get("candidate", False)
    print(f"\nRESULT: task=[T141b hawkes ensemble BCD] metrics={{best_config={best_name},best_pnl={best_pnl:.4f},delta_vs_A={best_delta:+.4f},iter019_v3_candidate={is_cand}}} notes=[A_baseline={A_PNL_NO_CONF}/{A_PNL_CONF} best={best_name}({best_mode})]")
