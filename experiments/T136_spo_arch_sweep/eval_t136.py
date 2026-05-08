"""T136 evaluation:
  1. Per-variant 3-seed ensemble PnL (default thr + iter_018 thr)
  2. Stack each variant with T75 LGB 5-seed → eval against iter_018 v1 baseline
  3. Compare to T87 5-seed baseline
  4. Report transmission estimate (LOSO-equivalent local 442k -> platform)

Outputs: results.json + console table.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = "/root/projects/liangwenbei_workdir"

T87_DIR = os.path.join(WORKDIR, "experiments", "T87_spo_dfl")
T75_DIR = os.path.join(WORKDIR, "experiments", "T75_regression_dmid")

SEEDS = [1, 7, 42]
VARIANTS = ["A", "B", "C", "D"]
FEE = 0.0001

# Use iter_018 v1 thresholds (real production thresh) — task said
# 4.21e-4/1.86e-4 (iter_015) but iter_018 v1 uses these.
THR_ITER18 = (0.00029995433796130955, 0.0002158528296175958)
THR_ITER15 = (4.21e-4, 1.86e-4)
THR_DEFAULT = (2 * FEE, 2 * FEE)
W_NN = 1.0
W_LGB = 1.5


def evaluate_pnl(pred, mp_t, mp_th, thr_up, thr_dn):
    pred_action = np.full(len(pred), 1, dtype=np.int8)
    pred_action[pred > thr_up] = 2
    pred_action[pred < -thr_dn] = 0
    side = pred_action.astype(np.float64) - 1.0
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * np.abs(side) * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    pnl = (side * diff - fee_pnl) / denom
    n_act = int((pred_action != 1).sum())
    return float(pnl.sum()), n_act


def per_sym_pnl(pred, mp_t, mp_th, sym, thr_up, thr_dn):
    out = {}
    for s in sorted(set(sym.tolist())):
        m = sym == s
        p, n = evaluate_pnl(pred[m], mp_t[m], mp_th[m], thr_up, thr_dn)
        out[int(s)] = {"pnl": p, "n_active": n, "n": int(m.sum())}
    return out


def load_seed_avg(paths):
    """Load N pred parquets and return mean pred + key columns from first."""
    dfs = [pd.read_parquet(p) for p in paths]
    base = dfs[0][["sym", "date", "session", "t", "true_label", "true_dmid_norm",
                   "midprice_t", "midprice_th"]].copy()
    preds = np.stack([d["pred_dmid_norm"].to_numpy() for d in dfs], axis=0)
    base["pred_dmid_norm"] = preds.mean(axis=0)
    base["pred_dmid_norm_std"] = preds.std(axis=0)
    return base


def pred_match_index(target_df, source_df):
    """Reorder source_df rows to match target_df by (sym, date, session, t).
    Returns a numpy array of source preds in target order.
    """
    # build index
    key_t = (target_df["sym"].astype(int).astype(str) + "|" +
             target_df["date"].astype(int).astype(str) + "|" +
             target_df["session"].astype(str) + "|" +
             target_df["t"].astype(int).astype(str))
    key_s = (source_df["sym"].astype(int).astype(str) + "|" +
             source_df["date"].astype(int).astype(str) + "|" +
             source_df["session"].astype(str) + "|" +
             source_df["t"].astype(int).astype(str))
    src = pd.DataFrame({"key": key_s, "pred": source_df["pred_dmid_norm"].to_numpy()})
    src = src.set_index("key")
    return src.loc[key_t.to_list(), "pred"].to_numpy()


def main():
    print("=== T136 evaluation ===", flush=True)
    results = {"task": "T136 SPO+ NN architecture sweep",
               "iter_018_v1_baseline_no_abstain": 39.7467,
               "iter_018_v1_baseline_with_abstain": 41.4939,
               "T87_5seed_local_442k_baseline": None,
               "variants": {},
               "stacks": {}}

    # T87 5-seed baseline on full 442k test set
    t87_paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet")
                 for s in [1, 7, 13, 42, 100]]
    t87_paths = [p for p in t87_paths if os.path.exists(p)]
    print(f"  T87 paths: {len(t87_paths)} found")
    t87 = load_seed_avg(t87_paths)
    mp_t = t87["midprice_t"].to_numpy()
    mp_th = t87["midprice_th"].to_numpy()
    sym = t87["sym"].to_numpy()
    pred_t87 = t87["pred_dmid_norm"].to_numpy()
    pnl_t87_def, n_t87_def = evaluate_pnl(pred_t87, mp_t, mp_th, *THR_DEFAULT)
    pnl_t87_15, n_t87_15 = evaluate_pnl(pred_t87, mp_t, mp_th, *THR_ITER15)
    pnl_t87_18, n_t87_18 = evaluate_pnl(pred_t87, mp_t, mp_th, *THR_ITER18)
    print(f"  T87 (5 seeds avg, NN-only):")
    print(f"    thr=(2e-4,2e-4)   pnl={pnl_t87_def:+.3f} n_act={n_t87_def:,}")
    print(f"    thr=iter15        pnl={pnl_t87_15:+.3f} n_act={n_t87_15:,}")
    print(f"    thr=iter18        pnl={pnl_t87_18:+.3f} n_act={n_t87_18:,}")
    results["T87_5seed_local_442k_baseline"] = {
        "thr_default": {"pnl": pnl_t87_def, "n_active": n_t87_def, "thr": THR_DEFAULT},
        "thr_iter15":  {"pnl": pnl_t87_15,  "n_active": n_t87_15,  "thr": THR_ITER15},
        "thr_iter18":  {"pnl": pnl_t87_18,  "n_active": n_t87_18,  "thr": THR_ITER18},
    }

    # T87 3-seed (just {1,7,42}) for direct comparison with T136 (also 3 seeds)
    t87_3paths = [os.path.join(T87_DIR, f"pred_T87_seed{s}_main.parquet") for s in SEEDS]
    if all(os.path.exists(p) for p in t87_3paths):
        t87_3 = load_seed_avg(t87_3paths)
        pred_t87_3 = t87_3["pred_dmid_norm"].to_numpy()
        pnl_t87_3def, n_t87_3def = evaluate_pnl(pred_t87_3, mp_t, mp_th, *THR_DEFAULT)
        pnl_t87_3_15, n_t87_3_15 = evaluate_pnl(pred_t87_3, mp_t, mp_th, *THR_ITER15)
        pnl_t87_3_18, n_t87_3_18 = evaluate_pnl(pred_t87_3, mp_t, mp_th, *THR_ITER18)
        print(f"  T87 (3 seeds {SEEDS} avg, for fair compare):")
        print(f"    thr=(2e-4,2e-4)   pnl={pnl_t87_3def:+.3f}")
        print(f"    thr=iter15        pnl={pnl_t87_3_15:+.3f}")
        print(f"    thr=iter18        pnl={pnl_t87_3_18:+.3f}")
        results["T87_3seed_baseline"] = {
            "thr_default": {"pnl": pnl_t87_3def, "n_active": n_t87_3def},
            "thr_iter15":  {"pnl": pnl_t87_3_15,  "n_active": n_t87_3_15},
            "thr_iter18":  {"pnl": pnl_t87_3_18,  "n_active": n_t87_3_18},
        }

    # Load T75 LGB 5-seed avg (need to align with t87 row order via keys)
    t75_paths = [os.path.join(T75_DIR, f"pred_T75_seed{s}.parquet")
                 for s in [1, 7, 13, 42, 100]]
    t75_paths = [p for p in t75_paths if os.path.exists(p)]
    pred_t75_aligned = None
    if t75_paths:
        print(f"  T75 paths: {len(t75_paths)} found")
        t75 = load_seed_avg(t75_paths)
        try:
            pred_t75_aligned = pred_match_index(t87, t75)
        except Exception as e:
            print(f"  WARN T75 alignment failed: {e}")
    if pred_t75_aligned is not None:
        # iter18 baseline replication: w_nn*pred_t87 + w_lgb*pred_t75 with thr_iter18
        stk = W_NN * pred_t87 + W_LGB * pred_t75_aligned
        # iter_018 v1 uses normalized weights; since thr is calibrated to that
        # specific combiner, eval with /sum_w to match scale
        stk_norm = stk / (W_NN + W_LGB)
        pnl_stk_def, n_stk_def = evaluate_pnl(stk_norm, mp_t, mp_th, *THR_DEFAULT)
        pnl_stk_15, n_stk_15 = evaluate_pnl(stk_norm, mp_t, mp_th, *THR_ITER15)
        pnl_stk_18, n_stk_18 = evaluate_pnl(stk_norm, mp_t, mp_th, *THR_ITER18)
        print(f"  T87+T75 stack (NN+LGB, iter_018 weights, normalized):")
        print(f"    thr=iter18  pnl={pnl_stk_18:+.3f}  n_act={n_stk_18:,}  "
              f"(target=39.75 unweighted-or-41.49 with abstain)")
        results["T87_T75_iter18_replica"] = {
            "thr_iter18": {"pnl": pnl_stk_18, "n_active": n_stk_18},
            "thr_default": {"pnl": pnl_stk_def, "n_active": n_stk_def},
            "thr_iter15": {"pnl": pnl_stk_15, "n_active": n_stk_15},
        }

    # Per-variant T136 ensemble
    for V in VARIANTS:
        paths = [os.path.join(HERE, f"pred_T136_{V}_seed{s}.parquet") for s in SEEDS]
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            print(f"  variant {V}: NO predictions found, skip")
            continue
        if len(existing) < len(paths):
            print(f"  variant {V}: only {len(existing)}/{len(paths)} seeds available")
        try:
            v = load_seed_avg(existing)
        except Exception as e:
            print(f"  variant {V}: load failed: {e}")
            continue
        # Align to t87 grid (might be same grid though)
        try:
            pred_v_aligned = pred_match_index(t87, v)
        except Exception as e:
            print(f"  variant {V}: alignment failed: {e}; using as-is")
            pred_v_aligned = v["pred_dmid_norm"].to_numpy()

        # NN-only eval
        pnl_v_def, n_v_def = evaluate_pnl(pred_v_aligned, mp_t, mp_th, *THR_DEFAULT)
        pnl_v_15, n_v_15 = evaluate_pnl(pred_v_aligned, mp_t, mp_th, *THR_ITER15)
        pnl_v_18, n_v_18 = evaluate_pnl(pred_v_aligned, mp_t, mp_th, *THR_ITER18)
        per_sym_v = per_sym_pnl(pred_v_aligned, mp_t, mp_th, sym, *THR_ITER18)
        print(f"\n  variant {V} (NN-only, {len(existing)} seeds avg):")
        print(f"    thr=(2e-4,2e-4)   pnl={pnl_v_def:+.3f}  n_act={n_v_def:,}")
        print(f"    thr=iter15        pnl={pnl_v_15:+.3f}   n_act={n_v_15:,}")
        print(f"    thr=iter18        pnl={pnl_v_18:+.3f}   n_act={n_v_18:,}")
        print(f"    per-sym iter18: {dict((k, round(d['pnl'], 2)) for k, d in per_sym_v.items())}")

        results["variants"][V] = {
            "n_seeds": len(existing),
            "thr_default": {"pnl": pnl_v_def, "n_active": n_v_def},
            "thr_iter15":  {"pnl": pnl_v_15,  "n_active": n_v_15},
            "thr_iter18":  {"pnl": pnl_v_18,  "n_active": n_v_18},
            "per_sym_iter18": {str(k): d for k, d in per_sym_v.items()},
            "delta_vs_T87_3seed_iter18": (pnl_v_18 - pnl_t87_3_18) if 'pnl_t87_3_18' in dir() else None,
            "delta_vs_T87_5seed_iter18": pnl_v_18 - pnl_t87_18,
        }

        # Stack with T75 LGB
        if pred_t75_aligned is not None:
            stk_v = W_NN * pred_v_aligned + W_LGB * pred_t75_aligned
            stk_v_norm = stk_v / (W_NN + W_LGB)
            pnl_sv_18, n_sv_18 = evaluate_pnl(stk_v_norm, mp_t, mp_th, *THR_ITER18)
            per_sym_sv = per_sym_pnl(stk_v_norm, mp_t, mp_th, sym, *THR_ITER18)
            print(f"  variant {V} + T75 LGB stack (replace T87 in iter_018 v1):")
            print(f"    thr=iter18  pnl={pnl_sv_18:+.3f}  n_act={n_sv_18:,}  "
                  f"(vs T87+T75=39.75 baseline)")
            results["stacks"][V] = {
                "thr_iter18": {"pnl": pnl_sv_18, "n_active": n_sv_18},
                "per_sym_iter18": {str(k): d for k, d in per_sym_sv.items()},
                "delta_vs_iter018_baseline_no_abstain": pnl_sv_18 - 39.7467,
                "delta_vs_iter018_with_abstain": pnl_sv_18 - 41.4939,
            }

    # Verdict
    verdicts = []
    for V in VARIANTS:
        sv = results["stacks"].get(V)
        nv = results["variants"].get(V)
        if not sv:
            verdicts.append(f"{V}: SKIP (no preds or no T75 stack)")
            continue
        delta_no_abst = sv["delta_vs_iter018_baseline_no_abstain"]
        delta_nn = nv["delta_vs_T87_5seed_iter18"]
        if delta_no_abst > 0.5:
            v = "WORTH iter_019 candidate"
        elif delta_no_abst > -0.3:
            v = "MAYBE — within noise of T87"
        else:
            v = "NO — worse than T87"
        verdicts.append(
            f"{V}: stack_pnl={sv['thr_iter18']['pnl']:+.3f}  "
            f"Δ_vs_T87_NN={delta_nn:+.3f}  Δ_vs_iter018_no_abst={delta_no_abst:+.3f}  → {v}"
        )
    print("\n=== VERDICT ===")
    for v in verdicts:
        print("  " + v)
    results["verdict"] = verdicts

    # Transmission estimate: based on iter_018 v1's pattern
    # iter_018 v1 local 442k = 41.49 (with abstain) or 39.75 (no abstain)
    # Platform = +28.93. So ratio ≈ 28.93 / 41.49 ≈ 0.697.
    transmission_ratio = 28.93 / 41.4939
    results["transmission_ratio_iter18_basis"] = transmission_ratio
    for V, sv in results["stacks"].items():
        local = sv["thr_iter18"]["pnl"]
        results["stacks"][V]["est_platform_pnl"] = local * transmission_ratio
    if results["stacks"]:
        print(f"\n  transmission ratio (iter_018 basis) = {transmission_ratio:.3f}")
        for V, sv in results["stacks"].items():
            print(f"  variant {V} stack: est platform PnL = {sv['est_platform_pnl']:+.3f}")

    out = os.path.join(HERE, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
