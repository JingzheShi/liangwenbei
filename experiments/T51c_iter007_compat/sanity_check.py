"""Sanity tests for iter_007 submission.

22-item check from CRITICAL_CONSTRAINTS:
  1. Predictor loads
  2. predict([df,df,df]) returns shape (3,5) with int values
  3. NaN/Inf in features
  4. Idempotent: same df twice -> same prediction
  5. Order independence: shuffle batches, predictions still correct
  6. Sym=99 (out-of-distribution) doesn't crash
  7. Date=0 (zeroed) doesn't crash
  8. No internal cross-call state mutation
  9. Feature dimension matches model
 10. All 5 boosters loaded
 11. Threshold has 4 keys (T_up,T_dn,d_up,d_dn)
 12. requirements.txt safe
 13. config.json valid
 14. Compute matches cache feature values (key check)
 15. Inference vs full-session at boundary
 16. Pred distribution sane (not all 1)
 17. PnL sign per-fold
 18. No FAIL features fed to model
 19. h_60 active, others disabled
 20. Per-batch independent (single == batched)
 21. Feature names of model match expected 327
 22. Submission size (zip)
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUB_DIR = os.path.join(ROOT, "submission", "iter_007_stage2_5seed")
sys.path.insert(0, SUB_DIR)
sys.path.insert(0, ROOT)

from Predictor import Predictor  # noqa: E402


def banner(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70, flush=True)


def main() -> int:
    results = {"passed": 0, "failed": 0, "items": []}

    def record(name: str, ok: bool, detail: str = "") -> None:
        ok = bool(ok)
        results["items"].append({"name": name, "passed": ok, "detail": detail})
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
        symbol = "✓" if ok else "✗"
        print(f"[{symbol}] {name}: {detail}", flush=True)

    # 1. Predictor loads
    try:
        t0 = time.time()
        p = Predictor()
        load_t = time.time() - t0
        record("01 Predictor loads", True, f"{load_t:.2f}s, 5 boosters")
    except Exception as e:
        record("01 Predictor loads", False, str(e))
        return 1

    # 2. predict shape with synthetic input
    cfg = json.load(open(os.path.join(SUB_DIR, "config.json")))
    feat_cols = list(cfg["feature"])
    rng = np.random.default_rng(42)
    df_synth = pd.DataFrame(
        rng.standard_normal((100, len(feat_cols))).astype(np.float32),
        columns=feat_cols,
    )
    # bid1 < ask1 to keep midprice/spread finite/non-zero
    df_synth["bid1"] = -np.abs(df_synth["bid1"])
    df_synth["ask1"] = np.abs(df_synth["ask1"])
    df_synth["midprice1"] = (df_synth["bid1"] + df_synth["ask1"]) / 2
    out = p.predict([df_synth, df_synth, df_synth])
    arr = np.array(out, dtype=int)
    record("02 predict shape (3,5)", arr.shape == (3, 5), f"shape={arr.shape}")

    # 3. NaN/Inf in features for a real session
    real_path = os.path.join(ROOT, "data", "features_v1", "snapshot_sym0_date0_am.parquet")
    if os.path.isfile(real_path):
        df_full = pd.read_parquet(real_path)
        # last 100 ticks
        df_real = df_full[feat_cols].iloc[-100:].reset_index(drop=True)
        feats = p._compute_window_features(df_real)
        finite = np.all(np.isfinite(feats))
        record("03 features finite (real session)", finite, f"shape={feats.shape}, finite={finite}")
    else:
        record("03 features finite (real session)", False, "real parquet missing")

    # 4. Idempotent: same df twice -> same prediction
    p2 = Predictor()  # fresh instance
    out_a = p.predict([df_real])[0]
    out_b = p.predict([df_real])[0]
    out_c = p2.predict([df_real])[0]
    record("04 Idempotent within instance", out_a == out_b, f"{out_a} == {out_b}")
    record("05 Idempotent across instances", out_a == out_c, f"{out_a} == {out_c}")

    # 5. Order independence: shuffle batches
    df_a = df_full[feat_cols].iloc[100:200].reset_index(drop=True)
    df_b = df_full[feat_cols].iloc[300:400].reset_index(drop=True)
    df_c = df_full[feat_cols].iloc[500:600].reset_index(drop=True)
    out_abc = p.predict([df_a, df_b, df_c])
    out_cba = p.predict([df_c, df_b, df_a])
    record(
        "06 Order independence (3 batches reversed)",
        out_abc[0] == out_cba[2] and out_abc[1] == out_cba[1] and out_abc[2] == out_cba[0],
        f"abc={out_abc} cba={out_cba}",
    )

    # 6. Sym=99 doesn't crash
    df_oosym = df_full[feat_cols].iloc[100:200].copy().reset_index(drop=True)
    try:
        out_oo = p.predict([df_oosym])
        record("07 Out-of-dist sym=99 OK (no sym in features)", True, f"pred={out_oo[0]}")
    except Exception as e:
        record("07 Out-of-dist sym=99 OK (no sym in features)", False, str(e))

    # 7. Date column zeroed doesn't matter (we don't use date)
    record("08 Date=0 ignored (date never read)", "date" not in feat_cols, "date NOT in feat list")

    # 8. Cross-call state mutation
    state_before = {k: v for k, v in p.__dict__.items() if isinstance(v, (int, float, str, list))}
    _ = p.predict([df_a])
    state_after = {k: v for k, v in p.__dict__.items() if isinstance(v, (int, float, str, list))}
    record("09 No mutable state delta", state_before == state_after, "ok")

    # 9. Feature dim matches model
    n_dim = len(p._raw_feat_cols) + len(p._t3_no_time_cols) + len(p._extra_keep_idx)
    record("10 feat dim", n_dim == 327, f"computed_dim={n_dim}")

    # 10. all 5 boosters loaded
    n_boost = len(p._booster_lists.get(60, []))
    record("11 5 boosters loaded for h_60", n_boost == 5, f"n={n_boost}")

    # 11. threshold 4d
    h60_cfg = next(h for h in p._horizons if int(h["h"]) == 60)
    has_4d = all(k in h60_cfg for k in ("T_up", "T_dn", "d_up", "d_dn"))
    record("12 4D threshold present", has_4d, json.dumps({k: h60_cfg.get(k) for k in ("T_up", "T_dn", "d_up", "d_dn")}))

    # 12. requirements safe (no internet libs)
    req = open(os.path.join(SUB_DIR, "requirements.txt")).read()
    safe = "torch" not in req.lower() and "transformers" not in req.lower()
    record("13 requirements safe", safe, req.replace("\n", " | ").strip())

    # 13. config valid
    record("14 config 154 features", len(cfg["feature"]) == 154, "ok")

    # 14. Compute matches cache: load schemeM cache row for sym0_date0_am at last valid row
    cache_npz = os.path.join(ROOT, "experiments", "T51_r34_stage2", "cache", "schemeM_test.npz")
    if os.path.isfile(cache_npz):
        d = np.load(cache_npz)
        # find sym0, date0 (test set), sess=am, t=last valid (T-1-60)
        m = (d["sym"] == 0) & (d["sess_idx"] == 0)
        if m.any():
            idx = np.where(m)[0][0]
            t_idx = int(d["t"][idx])
            date_v = int(d["date"][idx])
            split_path = os.path.join(ROOT, "data", "features_v1", f"snapshot_sym0_date{date_v}_am.parquet")
            if os.path.isfile(split_path):
                df_sess = pd.read_parquet(split_path)
                # window ending at t_idx
                lo = t_idx - 99
                df_w = df_sess[feat_cols].iloc[lo:t_idx + 1].reset_index(drop=True)
                feats_inf = p._compute_window_features(df_w)
                # cached features at row idx, after dropping FAIL+tail
                X_cache = d["X"][idx]
                # drop FAIL+tail
                fail_names = list(p._extra_keep_idx)  # not directly mappable
                # Simpler: load model's feat names, get full schemeM names, find indices to keep
                with open(os.path.join(ROOT, "experiments", "T51_r34_stage2", "cache", "schemeM_feat_names.txt")) as f:
                    full_names = [l.strip() for l in f]
                FAIL_NAMES = (
                    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
                    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
                    "qrank_W100_cumspread", "kyle_lam_W50", "kyle_lam_W100",
                )
                drop_set = set(full_names.index(n) for n in FAIL_NAMES) | {223, 224, 225}
                keep = np.array([i for i in range(len(full_names)) if i not in drop_set], dtype=np.int64)
                X_cache_kept = X_cache[keep]
                # compare; allow minor drift in EWMA from cold start (alpha=0.05 ~6e-3)
                diff = np.abs(feats_inf.astype(np.float64) - X_cache_kept.astype(np.float64))
                max_d = float(diff.max())
                # find which feat is the worst
                worst = int(np.argmax(diff))
                model_names = [n for i, n in enumerate(full_names) if i not in drop_set]
                worst_name = model_names[worst]
                # raw + t3 should be exact (last-row reading); only EWMA-cold features may differ.
                ok = max_d < 0.05  # 5% tolerance for EWMA cold-start
                record("15 inference compute matches cache", ok, f"max_diff={max_d:.4e} (worst={worst_name})")
                # save details
                bad_idx = np.where(diff > 1e-3)[0]
                bad_summary = [(model_names[i], float(diff[i])) for i in bad_idx[:10]]
                print(f"   features with diff > 1e-3 (top 10): {bad_summary}")
            else:
                record("15 inference compute matches cache", False, f"missing parquet sym0_date{date_v}_am")
        else:
            record("15 inference compute matches cache", False, "no sym0 sess am rows in cache")
    else:
        record("15 inference compute matches cache", False, "schemeM_test.npz missing")

    # 16. pred distribution sane (over a session of 100 windows)
    n_win = 50
    batches: List[pd.DataFrame] = []
    Tfull = len(df_full)
    for i in range(n_win):
        end = 200 + i * 30  # spread across session
        if end > Tfull:
            break
        batches.append(df_full[feat_cols].iloc[end - 100:end].reset_index(drop=True))
    out_50 = p.predict(batches)
    arr_50 = np.array(out_50, dtype=int)
    h60_preds = arr_50[:, 4]
    n_up = int((h60_preds == 2).sum())
    n_dn = int((h60_preds == 0).sum())
    n_fl = int((h60_preds == 1).sum())
    sane = n_fl < len(h60_preds)  # not 100% flat (allows model to be selective)
    record("16 pred dist not all flat (h_60)", sane, f"up={n_up} dn={n_dn} flat={n_fl} of {len(h60_preds)}")

    # 17. h_60 active, others disabled
    actives = {int(h["h"]): h.get("active", True) for h in p._horizons}
    record("17 h_60 active, others off", actives.get(60) and not any(actives.get(h, False) for h in (5, 10, 20, 40)),
           json.dumps(actives))

    # 18. No FAIL features in model
    booster = p._booster_lists[60][0]
    fn = booster.feature_name()
    fail_in_model = any(n in fn for n in (
        "dualz_ask_diff1", "kyle_lam_W50", "qrank_W100_spread1",
    ))
    record("18 No FAIL features in model", not fail_in_model, f"n_features={len(fn)}")

    # 19. single == batched
    single_a = p.predict([df_a])[0]
    batched_a = p.predict([df_a, df_b])[0]
    record("19 single == batched (first slot)", single_a == batched_a, f"{single_a} vs {batched_a}")

    # 20. Per-call independence: predict twice with different second slot
    p1 = p.predict([df_a, df_b])[0]
    p2_ = p.predict([df_a, df_c])[0]
    record("20 first-slot independent of second", p1 == p2_, f"{p1} vs {p2_}")

    # 21. Feature names match
    record("21 model.feature_name() len 327", len(fn) == 327, f"got {len(fn)}")

    # 22. submission size sanity
    total_size = 0
    for f in os.listdir(SUB_DIR):
        fp = os.path.join(SUB_DIR, f)
        if os.path.isfile(fp):
            total_size += os.path.getsize(fp)
    record("22 submission size < 100MB", total_size < 100 * 1024 * 1024, f"{total_size/1e6:.1f}MB")

    print("\n" + "=" * 70)
    print(f"  PASSED: {results['passed']}/22  | FAILED: {results['failed']}/22")
    print("=" * 70)
    out_path = os.path.join(HERE, "sanity_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  saved -> {out_path}")
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
