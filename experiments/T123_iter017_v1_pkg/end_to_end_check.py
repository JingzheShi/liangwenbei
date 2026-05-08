"""End-to-end check for iter_017 v1 Predictor (T123 LGB + T87 NN with HYD)."""
from __future__ import annotations

import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

for k in list(sys.modules.keys()):
    if "Predictor" in k or "fast_features" in k:
        del sys.modules[k]
import Predictor as P_mod  # noqa
importlib.reload(P_mod)

DATA_DIR = os.path.join(ROOT, "data")
WINDOW = 100
FEE = 0.0001


def load_test_batches(n_files=10, n_test_per_file=8):
    test_files = []
    for sym in (0, 1, 2, 3, 4):
        for date in (96, 97):
            for sess in ("am", "pm"):
                fn = os.path.join(DATA_DIR, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
                if os.path.isfile(fn):
                    test_files.append(fn)
                    if len(test_files) >= n_files:
                        break
            if len(test_files) >= n_files:
                break
        if len(test_files) >= n_files:
            break

    print(f"  loading {len(test_files)} files", flush=True)
    rng = np.random.default_rng(42)
    batches = []
    truth = []
    for fn in test_files:
        df = pd.read_parquet(fn)
        t_max = len(df) - 60 - 1
        ts = rng.integers(99, t_max, size=n_test_per_file)
        for t in ts:
            window = df.iloc[t-99:t+1].reset_index(drop=True)
            assert len(window) == 100
            batches.append(window)
            mp_t = float(window["midprice"].iloc[-1])
            mp_th = float(df["midprice"].iloc[t+60])
            label_60 = int(df["label_60"].iloc[t])
            sym = int(df["sym"].iloc[t]) if "sym" in df.columns else -1
            date = int(df["date"].iloc[t]) if "date" in df.columns else -1
            truth.append({"sym": sym, "date": date, "t": int(t),
                          "mp_t": mp_t, "mp_th": mp_th, "label_60": label_60})
    return batches, truth


def main():
    print("=== iter_017 v1 end-to-end check ===", flush=True)
    pred_obj = P_mod.Predictor()
    print(f"  Predictor loaded:", flush=True)
    print(f"    LGB ensembles: {dict((h, len(v)) for h, v in pred_obj._lgb_lists.items())}",
          flush=True)
    print(f"    NN ensembles:  {dict((h, len(v)) for h, v in pred_obj._nn_lists.items())}",
          flush=True)
    print(f"    weights:       {pred_obj._weights}", flush=True)

    batches, truth = load_test_batches(n_files=10, n_test_per_file=8)
    print(f"  built {len(batches)} batches", flush=True)

    t0 = time.time()
    out = pred_obj.predict(batches)
    print(f"  predict time: {time.time()-t0:.3f}s for {len(batches)} batches", flush=True)
    out_arr = np.array(out, dtype=np.int64)
    assert out_arr.shape == (len(batches), 5), out_arr.shape
    print(f"  output shape: {out_arr.shape}", flush=True)

    actions_60 = out_arr[:, 4]
    pnl_total = 0.0
    n_act = 0
    for i, t in enumerate(truth):
        side = float(actions_60[i]) - 1.0
        if abs(side) > 0:
            n_act += 1
        diff = t["mp_th"] - t["mp_t"]
        fee_pnl = FEE * abs(side) * abs((t["mp_th"] + 1.0) + (t["mp_t"] + 1.0))
        denom = t["mp_t"] + 1.0
        pnl_total += (side * diff - fee_pnl) / denom

    print(f"  cum_pnl (small sample h60): {pnl_total:+.5f}  n_active={n_act}/{len(batches)}",
          flush=True)

    # Shuffle invariance
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(batches))
    batches_shuf = [batches[i] for i in perm]
    out_shuf = pred_obj.predict(batches_shuf)
    out_shuf_arr = np.array(out_shuf, dtype=np.int64)
    out_unshuf = out_shuf_arr[np.argsort(perm)]
    if not np.array_equal(out_unshuf, out_arr):
        diff_n = int((out_unshuf != out_arr).sum())
        raise AssertionError(f"shuffle invariance broken: {diff_n} disagreements")
    print(f"  shuffle invariance: OK", flush=True)

    # sym=99 injection
    if "sym" in batches[0].columns:
        b99 = [b.copy() for b in batches[:8]]
        for b in b99:
            b["sym"] = 99
        out99 = pred_obj.predict(b99)
        out99_arr = np.array(out99, dtype=np.int64)
        # Expected: no crash; with iter_015 thresholds and sym not used,
        # outputs may agree with original. We just check no crash.
        diff = int((out99_arr != out_arr[:8]).sum())
        print(f"  sym=99 injection: no crash, predictions differ in {diff}/{out99_arr.size} cells",
              flush=True)

    # Spot-check first 5
    feats_359, feats_375 = pred_obj._compute_batch_features(batches[:8])
    lgbs = pred_obj._lgb_lists[60]
    nns = pred_obj._nn_lists[60]
    p_lgb = pred_obj._ensemble_predict_lgb(lgbs, feats_375)
    p_nn = pred_obj._ensemble_predict_nn(nns, feats_359)
    w_nn, w_lgb = pred_obj._weights[60]
    p_combined = (w_nn * p_nn + w_lgb * p_lgb) / (w_nn + w_lgb)
    print(f"  spot-check first 5 batches:", flush=True)
    for i in range(min(5, len(batches))):
        print(f"    [{i}] sym={truth[i]['sym']} t={truth[i]['t']}  "
              f"pred_lgb={p_lgb[i]:+.6f} pred_nn={p_nn[i]:+.6f} combined={p_combined[i]:+.6f} "
              f"action={out_arr[i, 4]}  Δmid={truth[i]['mp_th']-truth[i]['mp_t']:+.4f}",
              flush=True)

    print("\n=== all checks passed ===", flush=True)


if __name__ == "__main__":
    main()
