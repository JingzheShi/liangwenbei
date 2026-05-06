"""Evaluate best_model.pt on test split (240 sessions) using src/eval/dataset_eval.

Outputs:
  - test_predictions.parquet: per-sample (sym, date, session, t, true_label_*, pred_label_*, midprice_t, midprice_t+h)
  - test_metrics.json: aggregated PnL + classification metrics (5 horizons + best_score)
  - per_session_pnl.parquet: per-session cum_pnl breakdown for visualization
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXAMPLE_DIR = os.path.join(ROOT, "examples", "mmpc_demo")
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from src.data.split import get_file_path, get_split  # noqa: E402
from src.eval.pnl import HORIZONS, HORIZON_NAMES, compute_pnl, sanitize_for_json  # noqa: E402
from model import DeepLOB  # noqa: E402


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model = DeepLOB(num_classes_per_head=list(meta["num_classes_per_head"])).to(device)
    # Materialize LazyLinear by running dummy
    n_features = len(meta["feature_columns"])
    with torch.no_grad():
        dummy = torch.zeros(2, 1, 100, n_features, device=device)
        model(dummy)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, meta


@torch.no_grad()
def predict_session(
    model,
    device,
    feat_arr: np.ndarray,
    valid_t: np.ndarray,
    window: int,
    mean: np.ndarray,
    std: np.ndarray,
    log1p_idx: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Build windows from feat_arr (T, F) → run model in batches → return (N, 5) preds."""
    n = len(valid_t)
    n_h = 5
    preds = np.zeros((n, n_h), dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_t = valid_t[start:end]
        # Stack windows: shape (B, window, F)
        # Use slice arithmetic; window is fixed
        B = end - start
        x = np.empty((B, window, feat_arr.shape[1]), dtype=np.float32)
        for j, t in enumerate(chunk_t):
            x[j] = feat_arr[t - window + 1 : t + 1]
        # log1p + z-score + nan->0
        x[:, :, log1p_idx] = np.log1p(x[:, :, log1p_idx])
        x = (x - mean) / std
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
        xt = torch.from_numpy(x).unsqueeze(1).to(device)
        logits = model(xt)
        for h in range(n_h):
            preds[start:end, h] = logits[h].argmax(1).cpu().numpy()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(HERE, "best_model.pt"))
    ap.add_argument("--out_dir", type=str, default=HERE)
    ap.add_argument("--data_dir", type=str, default=os.path.join(ROOT, "data"))
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--fee_rate", type=float, default=0.0001)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    model, meta = load_model(args.ckpt, device)
    print(f"[eval] loaded model from {args.ckpt}, best_epoch={meta.get('best_epoch')}")

    feature_cols = list(meta["feature_columns"])
    log1p_cols = list(meta["log1p_cols"])
    mean = np.asarray(meta["mean"], dtype=np.float32)  # (F,)
    std = np.asarray(meta["std"], dtype=np.float32)
    log1p_idx = np.array([feature_cols.index(c) for c in log1p_cols], dtype=np.int64)

    splits = get_split(mode="local_debug")
    test_keys = splits["test"]
    print(f"[eval] {len(test_keys)} test sessions")

    window = 100
    horizons = list(HORIZONS)
    n_h = len(horizons)
    label_cols = [f"label_{h}" for h in horizons]

    all_pred = []
    all_label = []
    all_mp_t = []
    all_mp_tn = []
    per_session_rows = []
    big_rows = []  # for parquet

    t0 = time.time()
    for i, (sym, date, sess) in enumerate(test_keys):
        path = get_file_path(sym, date, sess, data_dir=args.data_dir)
        df = pd.read_parquet(path)
        if len(df) < window + max(horizons):
            print(f"  skip too-short session: {sym}/{date}/{sess}")
            continue
        feat_arr = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        midprice = df["midprice"].to_numpy(dtype=np.float64)
        label_full = df[label_cols].to_numpy(dtype=np.int64)
        n = len(df)
        t_lo = window - 1
        t_hi = n - max(horizons) - 1
        valid_t = np.arange(t_lo, t_hi + 1, dtype=np.int64)

        preds = predict_session(
            model, device, feat_arr, valid_t, window, mean, std, log1p_idx, args.batch_size
        )

        mp_t = midprice[valid_t]
        mp_tn = np.stack([midprice[valid_t + h] for h in horizons], axis=1)
        labels_t = label_full[valid_t]

        all_pred.append(preds)
        all_label.append(labels_t)
        all_mp_t.append(mp_t)
        all_mp_tn.append(mp_tn)

        # Per-session PnL
        sess_pnl = compute_pnl(preds, labels_t, mp_t.astype(np.float64), mp_tn.astype(np.float64), fee_rate=args.fee_rate)
        per_session_rows.append({
            "sym": int(sym), "date": int(date), "session": str(sess),
            "n_samples": int(preds.shape[0]),
            **{f"cum_pnl_{hn}": sess_pnl[hn]["cum_pnl"] for hn in HORIZON_NAMES},
            **{f"single_pnl_{hn}": sess_pnl[hn]["single_pnl"] for hn in HORIZON_NAMES},
            **{f"acc_{hn}": sess_pnl[hn]["accuracy"] for hn in HORIZON_NAMES},
        })

        # Per-sample rows
        n_samples = preds.shape[0]
        row_dict = {
            "sym": np.full(n_samples, int(sym), dtype=np.int32),
            "date": np.full(n_samples, int(date), dtype=np.int32),
            "session": np.full(n_samples, str(sess), dtype=object),
            "t": valid_t.astype(np.int32),
            "midprice_t": mp_t.astype(np.float32),
        }
        for h_idx, h in enumerate(horizons):
            row_dict[f"true_label_{h}"] = labels_t[:, h_idx].astype(np.int8)
            row_dict[f"pred_label_{h}"] = preds[:, h_idx].astype(np.int8)
            row_dict[f"midprice_t{h}"] = mp_tn[:, h_idx].astype(np.float32)
        big_rows.append(pd.DataFrame(row_dict))

        if (i + 1) % 30 == 0:
            print(f"  [{i+1}/{len(test_keys)}] elapsed={time.time()-t0:.1f}s")

    pred_cat = np.concatenate(all_pred, axis=0)
    label_cat = np.concatenate(all_label, axis=0)
    mp_t_cat = np.concatenate(all_mp_t, axis=0).astype(np.float64)
    mp_tn_cat = np.concatenate(all_mp_tn, axis=0).astype(np.float64)

    out = compute_pnl(pred_cat, label_cat, mp_t_cat, mp_tn_cat, fee_rate=args.fee_rate)
    out["meta"] = {
        "n_sessions": len(per_session_rows),
        "n_samples_total": int(pred_cat.shape[0]),
        "fee_rate": args.fee_rate,
        "best_epoch": meta.get("best_epoch"),
    }

    print(f"[eval] aggregated test metrics:")
    for hn in HORIZON_NAMES:
        m = out[hn]
        print(f"  {hn}: acc={m['accuracy']:.3f} prec_macro={m['precision_macro']!r} cum_pnl={m['cum_pnl']:.4f} single_pnl={m['single_pnl']:.4f} pred_dist={m['pred_distribution']}")
    print(f"  best_score={out['best_score']:.4f} ({out['best_horizon']})")

    # Save outputs
    with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
        json.dump(sanitize_for_json(out), f, indent=2)

    big_df = pd.concat(big_rows, axis=0, ignore_index=True)
    big_df.to_parquet(os.path.join(args.out_dir, "test_predictions.parquet"), index=False)
    print(f"[eval] saved {len(big_df)} rows -> test_predictions.parquet")

    per_sess_df = pd.DataFrame(per_session_rows)
    per_sess_df.to_parquet(os.path.join(args.out_dir, "per_session_pnl.parquet"), index=False)
    print(f"[eval] saved per-session PnL -> per_session_pnl.parquet")
    print(f"[eval] elapsed total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
