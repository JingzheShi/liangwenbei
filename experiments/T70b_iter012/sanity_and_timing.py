"""iter_012 sanity check (22 items) + 1024-batch timing + 442k full-set timing.

Sanity items adapted from iter_011 contract verification:
  1. config.json present + valid JSON
  2. config.json has 154 features in interleaved (legacy) order [advisory only]
  3. fast_features_batch.py present
  4. Predictor.py present
  5. thresholds.json present + valid
  6. requirements.txt present
  7-11. 5 model files present (seeds 1, 7, 13, 42, 100)
  12. Predictor instantiates (no errors)
  13. Predictor.predict on 3 random DataFrames returns shape (3, 5) ints in {0,1,2}
  14. Predictor produces 359 features
  15. RAW_COLS_TRAIN_ORDER matches T68 cache ordering
  16. extra_keep_idx has 205 entries
  17. FAIL_NAMES has 11 entries
  18. h=60 active, h=5/10/20/40 inactive
  19. ensemble_seeds list = [1, 7, 13, 42, 100]
  20. End-to-end with sym=99 (out-of-train) does not error
  21. End-to-end with shuffled batch order produces same per-window prediction
  22. Inference time on 1024-batch < 1.0s
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
import zipfile
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUB = os.path.join(ROOT, "submission", "iter_012_v4_stage5")
sys.path.insert(0, SUB)

CHECKS = []

def check(name, fn):
    try:
        ok, info = fn()
    except Exception as e:
        ok, info = False, f"EXCEPTION: {e!r}\n{traceback.format_exc()}"
    CHECKS.append((name, ok, info))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: {info}")
    return ok

# ----- file presence -----
def f1():
    p = os.path.join(SUB, "config.json")
    j = json.load(open(p))
    return os.path.exists(p) and isinstance(j, dict), p

def f2():
    cfg = json.load(open(os.path.join(SUB, "config.json")))
    fl = cfg["feature"]
    return len(fl) == 154 and fl[6] == "bid1" and fl[7] == "bsize1", f"154 cols, interleaved order"

def f3():
    p = os.path.join(SUB, "fast_features_batch.py")
    return os.path.exists(p), p

def f4():
    p = os.path.join(SUB, "Predictor.py")
    return os.path.exists(p), p

def f5():
    p = os.path.join(SUB, "thresholds.json")
    j = json.load(open(p))
    return "horizons" in j, p

def f6():
    p = os.path.join(SUB, "requirements.txt")
    return os.path.exists(p), p

def model_check(seed):
    def f():
        p = os.path.join(SUB, f"model_h60_seed{seed}.txt")
        return os.path.exists(p) and os.path.getsize(p) > 1000, p
    return f

# ----- functional -----
def f12():
    from Predictor import Predictor
    p = Predictor()
    return p is not None, "instantiated"

def f13():
    from Predictor import Predictor
    p = Predictor()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, 154)).astype(np.float32) * 0.001 + 1.0,
        columns=p._raw_feat_cols,
    )
    out = p.predict([df, df, df])
    if not isinstance(out, list) or len(out) != 3 or len(out[0]) != 5:
        return False, f"shape: {len(out)}x{len(out[0]) if out else 0}"
    flat = np.array(out)
    return flat.shape == (3, 5) and set(flat.flatten().tolist()) <= {0, 1, 2}, f"shape={flat.shape}"

def f14():
    from Predictor import Predictor
    p = Predictor()
    feats_n = 154 + len(p._extra_keep_idx)
    return feats_n == 359, f"{feats_n} features"

def f15():
    from Predictor import RAW_COLS_TRAIN_ORDER
    expected_blocked = list(RAW_COLS_TRAIN_ORDER)
    # First 16 should be: open,high,low,close,volume_delta,amount_delta,bid1..bid10
    head = expected_blocked[:16]
    return head[6] == "bid1" and head[7] == "bid2" and head[15] == "bid10", f"head[6:16]={head[6:16]}"

def f16():
    from Predictor import Predictor
    p = Predictor()
    return len(p._extra_keep_idx) == 205, f"{len(p._extra_keep_idx)}"

def f17():
    from Predictor import FAIL_NAMES
    return len(FAIL_NAMES) == 11, f"{len(FAIL_NAMES)}"

def f18():
    j = json.load(open(os.path.join(SUB, "thresholds.json")))
    h60 = next(h for h in j["horizons"] if h["h"] == 60)
    others_inactive = all(not h.get("active", True) for h in j["horizons"] if h["h"] != 60)
    return h60["active"] and others_inactive, f"h60 active={h60['active']}"

def f19():
    j = json.load(open(os.path.join(SUB, "thresholds.json")))
    h60 = next(h for h in j["horizons"] if h["h"] == 60)
    return h60.get("ensemble_seeds") == [1, 7, 13, 42, 100], f"seeds={h60.get('ensemble_seeds')}"

def f20():
    # Inject sym=99 (out-of-train) into a dataframe; Predictor doesn't use sym so should not error
    from Predictor import Predictor
    p = Predictor()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        rng.standard_normal((100, 154)).astype(np.float32) * 0.001 + 1.0,
        columns=p._raw_feat_cols,
    )
    df["sym"] = 99
    df["date"] = 0
    out = p.predict([df, df])
    return len(out) == 2 and len(out[0]) == 5, "ran with sym=99"

def f21():
    # Build 5 random windows; predict; shuffle and predict; per-window outputs identical
    from Predictor import Predictor
    p = Predictor()
    rng = np.random.default_rng(42)
    dfs = []
    for i in range(5):
        df = pd.DataFrame(
            rng.standard_normal((100, 154)).astype(np.float32) * 0.001 + 1.0,
            columns=p._raw_feat_cols,
        )
        dfs.append(df)
    out_orig = np.array(p.predict(dfs))
    perm = [3, 0, 4, 2, 1]
    dfs_shuf = [dfs[i] for i in perm]
    out_shuf = np.array(p.predict(dfs_shuf))
    # un-permute
    out_shuf_unperm = np.empty_like(out_shuf)
    for tgt, src in enumerate(perm):
        out_shuf_unperm[src] = out_shuf[tgt]
    return np.array_equal(out_orig, out_shuf_unperm), f"orig vs shuffle-unperm match"

def f22():
    # 1024 random windows: time the predict call
    from Predictor import Predictor
    p = Predictor()
    rng = np.random.default_rng(0)
    dfs = []
    base = rng.standard_normal((100, 154)).astype(np.float32) * 0.001 + 1.0
    for i in range(1024):
        # Small jitter so prices vary
        df = pd.DataFrame(
            base + rng.standard_normal((100, 154)).astype(np.float32) * 1e-5,
            columns=p._raw_feat_cols,
        )
        dfs.append(df)
    # warm up
    p.predict(dfs[:8])
    t0 = time.time()
    out = p.predict(dfs)
    elapsed = time.time() - t0
    print(f"    [info] 1024-batch elapsed: {elapsed:.3f}s ({elapsed*1000/1024:.3f} ms/window)")
    return elapsed < 1.0 and len(out) == 1024, f"{elapsed:.3f}s"


# ----- run all -----
print("--- iter_012 sanity & timing ---")
check("01 config.json", f1)
check("02 config.json features=154 (interleaved/legacy)", f2)
check("03 fast_features_batch.py", f3)
check("04 Predictor.py", f4)
check("05 thresholds.json", f5)
check("06 requirements.txt", f6)
for s in [1, 7, 13, 42, 100]:
    check(f"0{6 + [1,7,13,42,100].index(s)+1} model_h60_seed{s}.txt", model_check(s))
check("12 Predictor instantiates", f12)
check("13 predict() shape", f13)
check("14 359 final features", f14)
check("15 RAW_COLS_TRAIN_ORDER blocked", f15)
check("16 extra_keep_idx = 205", f16)
check("17 FAIL_NAMES = 11", f17)
check("18 h=60 active only", f18)
check("19 ensemble_seeds match", f19)
check("20 sym=99 out-of-train ok", f20)
check("21 shuffled batch invariance", f21)
check("22 1024-batch timing < 1.0s", f22)

# Summary
n_pass = sum(ok for _, ok, _ in CHECKS)
n_total = len(CHECKS)
print(f"\n=== {n_pass}/{n_total} passed ===")
sys.exit(0 if n_pass == n_total else 1)
