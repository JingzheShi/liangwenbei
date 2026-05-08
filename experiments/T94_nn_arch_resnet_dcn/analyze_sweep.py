"""T94: After single-seed sweep, summarize results and compute cross-corr to T81 NN.

For each tag with summary_<tag>_seed42.json + pred_<tag>_seed42.parquet:
  - print/save: arch, reg, n_params, train_time, val_corr, test_corr,
                ev_gate_k1_pnl (single), loso_equiv_pnl,
                cross-corr to T81 NN preds (averaged over seeds 1,7,13,42,100),
                cross-corr to T75 LGB preds.
  - rank by ev_gate_k1_loso_equiv

Saves sweep_summary.csv + REPORT preliminary lines.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
T81_DIR = os.path.join(ROOT, "experiments", "T81_nn_regression_pnl")
T75_DIR = os.path.join(ROOT, "experiments", "T75_regression_dmid")


def load_avg_pred(dir_, prefix, seeds):
    paths = [os.path.join(dir_, f"{prefix}_seed{s}.parquet") for s in seeds]
    base = pd.read_parquet(paths[0])
    p = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    n = len(base)
    for path in paths[1:]:
        df = pd.read_parquet(path)
        if len(df) != n:
            raise RuntimeError(f"len mismatch: {path}")
        p += df["pred_dmid_norm"].to_numpy(np.float64)
    p /= len(paths)
    out = base.copy()
    out["pred_dmid_norm"] = p.astype(np.float32)
    return out


def main():
    # Load reference predictions
    t81_avg = load_avg_pred(T81_DIR, "pred_T81", [1, 7, 13, 42, 100])
    t75_avg = load_avg_pred(T75_DIR, "pred_T75", [1, 7, 13, 42, 100])
    t81_pred = t81_avg["pred_dmid_norm"].to_numpy(np.float64)
    t75_pred = t75_avg["pred_dmid_norm"].to_numpy(np.float64)
    print(f"Loaded T81 NN avg ({len(t81_pred):,}) and T75 LGB avg ({len(t75_pred):,})", flush=True)

    summaries = []
    for path in sorted(glob.glob(os.path.join(HERE, "summary_*_seed42.json"))):
        tag = os.path.basename(path).replace("summary_", "").replace("_seed42.json", "")
        with open(path) as f:
            s = json.load(f)
        # corresponding pred
        pred_path = os.path.join(HERE, f"pred_{tag}_seed42.parquet")
        if not os.path.exists(pred_path):
            print(f"  skip (no pred): {tag}", flush=True)
            continue
        df = pd.read_parquet(pred_path)
        if len(df) != len(t81_avg):
            print(f"  skip (len mismatch): {tag}", flush=True)
            continue
        my_pred = df["pred_dmid_norm"].to_numpy(np.float64)

        # cross-corr (per row, Pearson)
        cc_t81 = float(np.corrcoef(my_pred, t81_pred)[0, 1])
        cc_t75 = float(np.corrcoef(my_pred, t75_pred)[0, 1])

        summaries.append({
            "tag": tag,
            "arch": s["arch"],
            "reg": s["reg"],
            "n_params": s["n_params"],
            "train_time_sec": s.get("train_time_sec"),
            "val_corr": s["val"]["corr"],
            "test_corr": s["test"]["corr"],
            "ev_gate_k1_pnl_single": s["test"]["ev_gate_k1_cum_pnl"],
            "ev_gate_k1_loso_equiv": s["test"]["ev_gate_k1_loso_equiv"],
            "n_active": s["test"]["ev_gate_k1_n_active"],
            "cross_corr_T81NN": cc_t81,
            "cross_corr_T75LGB": cc_t75,
            "best_epoch": s["best_epoch"],
            "best_val_mse": s["best_val_mse"],
        })

    df = pd.DataFrame(summaries)
    df = df.sort_values("ev_gate_k1_loso_equiv", ascending=False).reset_index(drop=True)
    print("\n=== T94 single-seed sweep results (sorted by EV-gate k=1 LOSO-equiv) ===\n", flush=True)
    cols = ["tag", "arch", "reg", "n_params", "test_corr", "ev_gate_k1_loso_equiv",
            "cross_corr_T81NN", "cross_corr_T75LGB"]
    print(df[cols].to_string(index=False), flush=True)
    csv_path = os.path.join(HERE, "sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}", flush=True)

    # Also print baseline T81 single-seed for reference
    t81_seed42 = pd.read_parquet(os.path.join(T81_DIR, "pred_T81_seed42.parquet"))
    t81_pred42 = t81_seed42["pred_dmid_norm"].to_numpy(np.float64)
    cc42_to_avg = float(np.corrcoef(t81_pred42, t81_pred)[0, 1])
    print(f"\nReference: cross-corr(T81 seed42, T81 5-seed avg) = {cc42_to_avg:.4f}", flush=True)


if __name__ == "__main__":
    main()
