"""T80 — Recompute iter_002 LOSO-equiv on 442k schemeC test set.

Use the cached schemeC_test.npz (X has 226 cols = 154 raw + 72 T3, with
amount_delta already log1p-signed). iter_002 expects 223 cols (drops the 3
time-encoding T3 features). So slice X[:, :223].

Then run iter_002's 5 horizon LightGBM models with the bundled thresholds
(T, delta) — NO new tuning. Compute per-horizon cum_pnl and per-sym
loso_equiv_sum (which equals total cum_pnl since pnl is row-additive and
splitting by sym is just a partition).

Calibration target:
  - iter_002 platform actual = +4.07 (h_5 active, the platform-submitted horizon)
  - iter_013 platform actual = +19.23 (h_60)
  - iter_013 LOSO-equiv (h_60 DE asym) = +36.23
  - Want: iter_002 LOSO-equiv (matching submitted horizon)
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import lightgbm as lgb


HERE = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.dirname(os.path.dirname(HERE))
TEST_NPZ = os.path.join(
    WORKDIR, "experiments/T5b_features_multihorizon/cache/schemeC_test.npz"
)
ITER_002_DIR = os.path.join(WORKDIR, "submission/iter_002_lgbm_schemeC")
ITER_002_223D_FEAT = os.path.join(
    WORKDIR, "experiments/T5b_features_multihorizon/cache/schemeC_223d_feat_names.txt"
)
ITER_002_226D_FEAT = os.path.join(
    WORKDIR, "experiments/T5b_features_multihorizon/cache/schemeC_feat_names.txt"
)

FEE = 0.0001  # consistent with all DE/eval scripts in this workdir
HORIZONS = (5, 10, 20, 40, 60)


def vectorized_pnl(pred: np.ndarray, label: np.ndarray,
                   mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    side = pred.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


def threshold_predict(probs: np.ndarray, T_thr: float, delta: float) -> np.ndarray:
    """iter_002 SYMMETRIC threshold rule (from Predictor.py)."""
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    side_max = np.maximum(p0, p2)
    take = (side_max >= T_thr) & (side_max > p1 + delta)
    side_pred = np.where(p2 > p0, 2, 0)
    return np.where(take, side_pred, 1).astype(np.int8)


def per_sym_pnl(pred, label, mp_t, mp_th, sym):
    parts = []
    by_sym = {}
    for s in np.unique(sym):
        m = sym == s
        v = float(vectorized_pnl(pred[m], label[m], mp_t[m], mp_th[m]).sum())
        parts.append(v)
        by_sym[int(s)] = {
            "n": int(m.sum()),
            "n_active": int((pred[m] != 1).sum()),
            "cum_pnl": v,
        }
    return float(sum(parts)), by_sym


def main():
    t0 = time.time()
    print("=== T80 iter_002 LOSO-equiv recompute ===", flush=True)

    # 1. Load test cache
    print(f"\nLoading {TEST_NPZ}", flush=True)
    d = np.load(TEST_NPZ)
    X226 = d["X"]
    print(f"  X shape: {X226.shape}, dtype: {X226.dtype}")
    sym = d["sym"]
    n_total = X226.shape[0]

    # 2. Verify the first 223 cols match iter_002's expected feature ordering
    with open(ITER_002_223D_FEAT) as f:
        names_223 = [l.strip() for l in f if l.strip()]
    with open(ITER_002_226D_FEAT) as f:
        names_226 = [l.strip() for l in f if l.strip()]
    assert names_226[:223] == names_223, "first 223 cols of 226-d cache must match 223-d order"
    assert names_226[223:] == [
        "time_minutes_since_session_start",
        "time_session_progress",
        "time_is_pm",
    ], "tail 3 cols are time encoding"
    print("  Feature alignment OK: X[:, :223] = iter_002 input (drop tail 3 time cols)")

    X223 = X226[:, :223]
    print(f"  X223 shape: {X223.shape}")

    # 3. Verify amount_delta has been log1p-transformed in cache (sanity)
    j_ad = names_223.index("amount_delta")
    ad = X223[:, j_ad]
    print(f"  amount_delta stats: min={ad.min():.3f}, max={ad.max():.3f}, "
          f"abs_p99={np.percentile(np.abs(ad), 99):.3f}")
    # raw amount_delta would be in millions; log1p-signed gets it ~0-30 range
    assert np.percentile(np.abs(ad), 99) < 100, "amount_delta should be log1p-signed (small magnitude)"
    print("  amount_delta passes sanity check (already log1p-signed in cache)")

    # 4. Load thresholds + 5 horizon models
    with open(os.path.join(ITER_002_DIR, "thresholds.json")) as f:
        tcfg = json.load(f)
    horizons_cfg = {int(h["h"]): h for h in tcfg["horizons"]}
    print(f"\nThresholds:")
    for h in HORIZONS:
        c = horizons_cfg[h]
        print(f"  h={h}: T={c['T']}, delta={c['delta']}, active={c['active']}")

    boosters = {}
    for h in HORIZONS:
        mp = os.path.join(ITER_002_DIR, f"model_h{h}.txt")
        boosters[h] = lgb.Booster(model_file=mp)
        print(f"  loaded model_h{h}.txt (n_features={boosters[h].num_feature()})")

    # 5. Run inference per horizon, compute per-sym cum_pnl
    results = {"task": "T80 iter_002 LOSO-equiv recompute", "n_total": n_total,
               "horizons": {}, "elapsed_sec": None}

    for h in HORIZONS:
        c = horizons_cfg[h]
        T_thr = float(c["T"]); delta = float(c["delta"])
        booster = boosters[h]
        print(f"\n--- horizon {h} (T={T_thr}, delta={delta}) ---", flush=True)
        t1 = time.time()
        probs = booster.predict(X223)
        print(f"  predict: {time.time() - t1:.1f}s, probs shape {probs.shape}")

        # Some boosters output binary (?). Should be 3-class soft.
        assert probs.ndim == 2 and probs.shape[1] == 3, f"expected 3-class probs, got {probs.shape}"

        pred = threshold_predict(probs, T_thr, delta)
        n_active = int((pred != 1).sum())
        n_up = int((pred == 2).sum())
        n_dn = int((pred == 0).sum())
        print(f"  preds: n_active={n_active} (n_up={n_up}, n_dn={n_dn})")

        label = d[f"y{h}"]
        mp_t = d["mp_t"]
        mp_th = d[f"mp_t{h}"]

        total = float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
        loso_sum, per_sym = per_sym_pnl(pred, label, mp_t, mp_th, sym)
        # also compute mean (per-row PnL)
        mean_pnl = total / n_total

        print(f"  total cum_pnl = {total:+.4f}")
        print(f"  loso_equiv_sum (per-sym sum) = {loso_sum:+.4f}")
        print(f"  per-sym breakdown:")
        for s, info in sorted(per_sym.items()):
            print(f"    sym={s}: n={info['n']}, n_active={info['n_active']}, cum_pnl={info['cum_pnl']:+.4f}")

        results["horizons"][str(h)] = {
            "T": T_thr,
            "delta": delta,
            "active_in_thresholds": bool(c["active"]),
            "n_active": n_active,
            "n_up": n_up,
            "n_dn": n_dn,
            "total_cum_pnl": total,
            "loso_equiv_sum": loso_sum,
            "mean_pnl_per_row": mean_pnl,
            "per_sym": per_sym,
        }

    # 6. Best horizon
    best_h = max(HORIZONS, key=lambda h: results["horizons"][str(h)]["loso_equiv_sum"])
    results["best_horizon"] = best_h
    results["best_loso_equiv"] = results["horizons"][str(best_h)]["loso_equiv_sum"]

    elapsed = time.time() - t0
    results["elapsed_sec"] = elapsed
    print(f"\nDone in {elapsed:.1f}s")
    print(f"BEST: h={best_h}, loso_equiv = {results['best_loso_equiv']:+.4f}")

    out_json = os.path.join(HERE, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_json}")


if __name__ == "__main__":
    main()
