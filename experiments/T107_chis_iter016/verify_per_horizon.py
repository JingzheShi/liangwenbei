"""Per-horizon LOSO-equiv verify for iter_016 CHIS.

For h ∈ {5, 10, 20, 40}: reuse T98 pred_lgb / pred_cb parquets, apply our
2-way ensemble weights + asymmetric thresholds (from thresholds.json).

For h = 60: reuse iter_015 v2 full_test_verify.json (already validated).

Reports per-horizon cum_pnl, per-sym breakdown, and MAX over horizons.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T98_DIR = os.path.join(ROOT, "experiments", "T98_multihorizon_iter016")
T106_DIR = os.path.join(ROOT, "experiments", "T106_iter015v2_pkg")

PKG_DIR = os.path.join(HERE, "pkg")
SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER014 = 38.281
ITER015 = 40.13
ITER015V2 = 43.227


def vectorized_pnl(act, mp_t, mp_th):
    side = act.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee_pnl) / denom


def ev_gate(p, thr_up, thr_dn):
    a = np.full(p.shape[0], 1, dtype=np.int8)
    a[p > thr_up] = 2
    a[p < -thr_dn] = 0
    return a


def load_horizon_preds_short(h: int):
    """Load and average 5-seed LGB and CB preds. Returns DataFrame."""
    base = None
    p_lgb_acc = None
    n_lgb = 0
    for s in SEEDS:
        f = os.path.join(T98_DIR, f"pred_lgb_h{h}_seed{s}.parquet")
        if not os.path.isfile(f):
            continue
        df = pd.read_parquet(f)
        if base is None:
            base = df[["sym", "date", "session", "t", "midprice_t", "midprice_th"]].copy()
        p = df["pred_dmid_norm"].to_numpy(np.float64)
        p_lgb_acc = p if p_lgb_acc is None else p_lgb_acc + p
        n_lgb += 1
    p_cb_acc = None
    n_cb = 0
    for s in SEEDS:
        f = os.path.join(T98_DIR, f"pred_cb_h{h}_seed{s}.parquet")
        if not os.path.isfile(f):
            continue
        df = pd.read_parquet(f)
        if base is None:
            base = df[["sym", "date", "session", "t", "midprice_t", "midprice_th"]].copy()
        p = df["pred_dmid_norm"].to_numpy(np.float64)
        p_cb_acc = p if p_cb_acc is None else p_cb_acc + p
        n_cb += 1
    out = base
    out["pred_lgb"] = p_lgb_acc / n_lgb if p_lgb_acc is not None else 0.0
    out["pred_cb"] = p_cb_acc / n_cb if p_cb_acc is not None else 0.0
    out["_n_lgb"] = n_lgb
    out["_n_cb"] = n_cb
    return out


def evaluate_short(h: int, w_lgb: float, w_cb: float, thr_up: float, thr_dn: float):
    df = load_horizon_preds_short(h)
    wsum = w_lgb + w_cb
    pred = (w_lgb * df["pred_lgb"].to_numpy(np.float64)
            + w_cb * df["pred_cb"].to_numpy(np.float64)) / wsum
    actions = ev_gate(pred.astype(np.float32), thr_up, thr_dn)
    mp_t = df["midprice_t"].to_numpy(np.float64)
    mp_th = df["midprice_th"].to_numpy(np.float64)
    pnl = vectorized_pnl(actions, mp_t, mp_th)
    sym = df["sym"].to_numpy(np.int64)
    per_sym = []
    n_act = []
    for s in SYMS:
        m = sym == s
        per_sym.append(float(pnl[m].sum()))
        n_act.append(int((actions[m] != 1).sum()))
    return {
        "h": h,
        "kind": "lgb_cb_2way",
        "w_lgb": w_lgb, "w_cb": w_cb,
        "thr_up": thr_up, "thr_dn": thr_dn,
        "cum_pnl_total": float(pnl.sum()),
        "sum_per_sym": float(sum(per_sym)),
        "per_sym": per_sym,
        "n_active_per_sym": n_act,
        "n_active_total": int((actions != 1).sum()),
        "n_rows": int(len(actions)),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
    }


def main():
    print("=== iter_016 CHIS per-horizon LOSO-equiv verify ===", flush=True)
    with open(os.path.join(PKG_DIR, "thresholds.json")) as f:
        tcfg = json.load(f)

    results = {}
    h60_loso = None
    for hcfg in tcfg["horizons"]:
        H = int(hcfg["h"])
        if hcfg["kind"] == "lgb_cb_2way":
            r = evaluate_short(
                H,
                w_lgb=float(hcfg["w_lgb"]), w_cb=float(hcfg["w_cb"]),
                thr_up=float(hcfg["thr_up"]), thr_dn=float(hcfg["thr_dn"]),
            )
            print(f"\nh={H} (lgb_cb_2way):  cum_pnl={r['cum_pnl_total']:+.4f}  "
                  f"n_active={r['n_active_total']:,}/{r['n_rows']:,}",
                  flush=True)
            for s, ps, na in zip(SYMS, r["per_sym"], r["n_active_per_sym"]):
                print(f"    sym={s}: cum_pnl={ps:+.4f}  n_active={na:,}", flush=True)
            results[str(H)] = r
        elif hcfg["kind"] == "fourway_iter015v2":
            # Reuse iter_015 v2 verify result directly (already in repo).
            verify_path = os.path.join(T106_DIR, "full_test_verify.json")
            with open(verify_path) as f:
                v = json.load(f)
            r = {
                "h": H,
                "kind": "fourway_iter015v2",
                "w_t87": float(hcfg["w_t87"]), "w_t89": float(hcfg["w_t89"]),
                "w_huber": float(hcfg["w_huber"]), "w_gru": float(hcfg["w_gru"]),
                "thr_up": float(hcfg["thr_up"]), "thr_dn": float(hcfg["thr_dn"]),
                "cum_pnl_total": float(v["cum_pnl_total"]),
                "sum_per_sym": float(v["sum_per_sym"]),
                "per_sym": v["per_sym"],
                "n_active_total": int(v["n_active"]),
                "n_rows": 442080,
            }
            h60_loso = r["sum_per_sym"]
            print(f"\nh={H} (fourway_iter015v2):  cum_pnl={r['cum_pnl_total']:+.4f}  "
                  f"sum_per_sym={r['sum_per_sym']:+.4f}  "
                  f"n_active={r['n_active_total']:,}/{r['n_rows']:,}", flush=True)
            for s, ps in zip(SYMS, r["per_sym"]):
                print(f"    sym={s}: cum_pnl={ps:+.4f}", flush=True)
            results[str(H)] = r

    print(f"\n{'='*60}\n=== SUMMARY (CHIS, MAX scoring) ===\n{'='*60}", flush=True)
    horizons = [5, 10, 20, 40, 60]
    losos = []
    for H in horizons:
        r = results[str(H)]
        loso = r["sum_per_sym"] if "sum_per_sym" in r else r["cum_pnl_total"]
        losos.append(loso)
        print(f"  h={H:>2}: cum_pnl={loso:+.4f}", flush=True)

    max_loso = max(losos)
    max_h = horizons[int(np.argmax(losos))]
    print(f"\n  MAX over 5 horizons: h={max_h}  cum_pnl={max_loso:+.4f}", flush=True)
    print(f"  vs iter_014 (+{ITER014:.2f}):     {max_loso - ITER014:+.4f}", flush=True)
    print(f"  vs iter_015 (+{ITER015:.2f}):     {max_loso - ITER015:+.4f}", flush=True)
    print(f"  vs iter_015 v2 (+{ITER015V2:.2f}):{max_loso - ITER015V2:+.4f}", flush=True)

    out = {
        "horizons": results,
        "horizon_loso_list": losos,
        "max_loso": float(max_loso),
        "max_horizon": int(max_h),
        "vs_iter014": float(max_loso - ITER014),
        "vs_iter015": float(max_loso - ITER015),
        "vs_iter015v2": float(max_loso - ITER015V2),
    }
    out_path = os.path.join(HERE, "verify_per_horizon.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
