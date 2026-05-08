"""T76 decoupled thresh per intraday bucket.

Hypothesis: HFT prob distribution differs by intraday position
(open volatile, mid stable, close-out). One global thresh underfits.

Procedure:
  1. Load 5-seed avg OOF (T70) -> 442k rows.
  2. Assign each row to bucket 1..4 by t (within-session quartile).
  3. DE 4D thresh per bucket -> sum 4 buckets cum_pnl.
  4. Compare to iter_012 single thresh = +26.44.
  5. Unbiased eval: dates 96-107 (in-sample, fit thresh) vs 108-119 (out-sample).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

T53_DIR = os.path.join(ROOT, "experiments", "T53_r34_stage3")
sys.path.insert(0, T53_DIR)
from de_thresh import PROB_COLS, gate_asymmetric, vectorized_pnl  # noqa: E402

T70_DIR = os.path.join(ROOT, "experiments", "T70_v4_stage5")
SEEDS = [1, 7, 13, 42, 100]
ITER012_BASELINE = 26.4396

# -- t boundary candidates -----------------------------------------------------
# Session t range: 99 .. 1940 (1842 ticks).
# Bucket policy "30/60/60/30 minutes" -> 1:2:2:1 ratio of session length.
T_MIN, T_MAX = 99, 1940  # inclusive
def bucket_quartile(t: np.ndarray) -> np.ndarray:
    # Q1/Q2/Q3/Q4 (each 25% of session).
    edges = np.quantile(np.arange(T_MIN, T_MAX + 1), [0.25, 0.5, 0.75])
    return np.digitize(t, edges)  # 0..3

def bucket_30_60_60_30(t: np.ndarray) -> np.ndarray:
    # 1/6, 2/6, 2/6, 1/6.
    span = T_MAX - T_MIN + 1
    e1 = T_MIN + span / 6.0
    e2 = T_MIN + span * 3.0 / 6.0
    e3 = T_MIN + span * 5.0 / 6.0
    return np.digitize(t, [e1, e2, e3])  # 0..3


# -- load 5-seed avg pred ------------------------------------------------------
def load_avg(seeds=SEEDS) -> pd.DataFrame:
    base = pd.read_parquet(os.path.join(T70_DIR, f"pred_T70_seed{seeds[0]}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in seeds[1:]:
        df_s = pd.read_parquet(os.path.join(T70_DIR, f"pred_T70_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch seed={s}")
        if not (df_s["sym"].equals(base["sym"]) and df_s["date"].equals(base["date"]) and df_s["t"].equals(base["t"])):
            raise RuntimeError(f"row order mismatch seed={s}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / float(len(seeds))
    out = base.drop(columns=PROB_COLS).copy()
    out["prob_0"] = p_avg[:, 0].astype(np.float32)
    out["prob_1"] = p_avg[:, 1].astype(np.float32)
    out["prob_2"] = p_avg[:, 2].astype(np.float32)
    return out


# -- DE 4D ---------------------------------------------------------------------
BOUNDS = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]


def de_search(probs, label, mp_t, mp_th, n_seeds=3, maxiter=50, popsize=20):
    def obj(x):
        Tu, Td, du, dd = x
        pred = gate_asymmetric(probs, Tu, Td, du, dd)
        return -float(vectorized_pnl(pred, label, mp_t, mp_th).sum())
    runs = []
    for sd in [0, 1, 7, 42, 100][:n_seeds]:
        r = differential_evolution(
            obj, bounds=BOUNDS, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        runs.append({
            "seed": int(sd),
            "T_up": float(r.x[0]), "T_dn": float(r.x[1]),
            "d_up": float(r.x[2]), "d_dn": float(r.x[3]),
            "obj": float(-r.fun),
        })
    runs.sort(key=lambda x: x["obj"], reverse=True)
    return runs


def cum_pnl(probs, label, mp_t, mp_th, Tu, Td, du, dd):
    pred = gate_asymmetric(probs, Tu, Td, du, dd)
    pnl = vectorized_pnl(pred, label, mp_t, mp_th)
    return float(pnl.sum()), int((pred != 1).sum())


def per_sym_pnl(df_block, Tu, Td, du, dd):
    out = []
    for k in (0, 1, 2, 3, 4):
        sub = df_block[df_block["sym"] == k]
        if len(sub) == 0:
            out.append(0.0); continue
        pred = gate_asymmetric(sub[PROB_COLS].to_numpy(np.float32), Tu, Td, du, dd)
        pnl = vectorized_pnl(pred, sub["true_label"].to_numpy(np.int64),
                             sub["midprice_t"].to_numpy(np.float64),
                             sub["midprice_th"].to_numpy(np.float64))
        out.append(float(pnl.sum()))
    return out


def fit_buckets(df, bucket_col, n_buckets=4, label_prefix="bucket"):
    """For each bucket, DE fit 4D thresh; return per-bucket best + total sum."""
    results = {}
    total = 0.0
    total_per_sym = np.zeros(5)
    for b in range(n_buckets):
        sub = df[df[bucket_col] == b]
        probs = sub[PROB_COLS].to_numpy(np.float32)
        label = sub["true_label"].to_numpy(np.int64)
        mp_t = sub["midprice_t"].to_numpy(np.float64)
        mp_th = sub["midprice_th"].to_numpy(np.float64)
        runs = de_search(probs, label, mp_t, mp_th)
        best = runs[0]
        c, na = cum_pnl(probs, label, mp_t, mp_th, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
        ps = per_sym_pnl(sub, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
        results[f"{label_prefix}{b}"] = {
            "n": int(len(sub)),
            "best": best,
            "cum_pnl": c,
            "n_active": na,
            "per_sym_pnl": ps,
        }
        total += c
        total_per_sym += np.array(ps)
        print(f"  {label_prefix}{b}: n={len(sub):,} thresh=({best['T_up']:.3f},{best['T_dn']:.3f},{best['d_up']:.4f},{best['d_dn']:.4f}) pnl={c:+.4f} n_active={na}", flush=True)
    print(f"  TOTAL sum: {total:+.4f}  per_sym sum: {total_per_sym.tolist()}", flush=True)
    return results, total, total_per_sym.tolist()


def fit_global(df):
    probs = df[PROB_COLS].to_numpy(np.float32)
    label = df["true_label"].to_numpy(np.int64)
    mp_t = df["midprice_t"].to_numpy(np.float64)
    mp_th = df["midprice_th"].to_numpy(np.float64)
    runs = de_search(probs, label, mp_t, mp_th)
    best = runs[0]
    c, na = cum_pnl(probs, label, mp_t, mp_th, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
    ps = per_sym_pnl(df, best["T_up"], best["T_dn"], best["d_up"], best["d_dn"])
    print(f"  global: n={len(df):,} thresh=({best['T_up']:.3f},{best['T_dn']:.3f},{best['d_up']:.4f},{best['d_dn']:.4f}) pnl={c:+.4f} n_active={na}", flush=True)
    return {"best": best, "cum_pnl": c, "n_active": na, "per_sym_pnl": ps}


def apply_thresh_buckets(df, bucket_col, bucket_thresh):
    """Apply pre-fit per-bucket thresholds to df; return total cum_pnl + per_sym_sum."""
    total = 0.0
    total_per_sym = np.zeros(5)
    per_bucket = []
    for b, th in enumerate(bucket_thresh):
        sub = df[df[bucket_col] == b]
        probs = sub[PROB_COLS].to_numpy(np.float32)
        label = sub["true_label"].to_numpy(np.int64)
        mp_t = sub["midprice_t"].to_numpy(np.float64)
        mp_th = sub["midprice_th"].to_numpy(np.float64)
        c, na = cum_pnl(probs, label, mp_t, mp_th, th["T_up"], th["T_dn"], th["d_up"], th["d_dn"])
        ps = per_sym_pnl(sub, th["T_up"], th["T_dn"], th["d_up"], th["d_dn"])
        total += c
        total_per_sym += np.array(ps)
        per_bucket.append({"bucket": b, "n": int(len(sub)), "cum_pnl": c, "n_active": na, "per_sym_pnl": ps})
    return total, total_per_sym.tolist(), per_bucket


def apply_thresh_global(df, th):
    probs = df[PROB_COLS].to_numpy(np.float32)
    label = df["true_label"].to_numpy(np.int64)
    mp_t = df["midprice_t"].to_numpy(np.float64)
    mp_th = df["midprice_th"].to_numpy(np.float64)
    c, na = cum_pnl(probs, label, mp_t, mp_th, th["T_up"], th["T_dn"], th["d_up"], th["d_dn"])
    ps = per_sym_pnl(df, th["T_up"], th["T_dn"], th["d_up"], th["d_dn"])
    return c, ps, na


# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print(f"=== T76 decoupled thresh per intraday bucket ===", flush=True)
    df = load_avg()
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s", flush=True)

    df["bucket_q4"] = bucket_quartile(df["t"].to_numpy())
    df["bucket_136"] = bucket_30_60_60_30(df["t"].to_numpy())
    print(f"  quartile bucket counts: {df['bucket_q4'].value_counts().sort_index().to_dict()}")
    print(f"  1:2:2:1   bucket counts: {df['bucket_136'].value_counts().sort_index().to_dict()}")

    out = {
        "task": "T76 decoupled thresh per intraday bucket",
        "n_test": int(len(df)),
        "iter012_baseline": ITER012_BASELINE,
        "seeds": SEEDS,
    }

    # ---- Step A: full 442k DE per bucket (in-sample) -------------------------
    print(f"\n[A1] Full 442k - quartile (Q1/Q2/Q3/Q4):", flush=True)
    res_q4, sum_q4, ps_q4 = fit_buckets(df, "bucket_q4", n_buckets=4, label_prefix="q")
    out["full_quartile"] = {"per_bucket": res_q4, "total_cum_pnl": sum_q4, "per_sym_sum": ps_q4,
                            "vs_iter012": sum_q4 - ITER012_BASELINE}

    print(f"\n[A2] Full 442k - 1:2:2:1 (open30/mid60/mid60/close30):", flush=True)
    res_136, sum_136, ps_136 = fit_buckets(df, "bucket_136", n_buckets=4, label_prefix="b136_")
    out["full_136"] = {"per_bucket": res_136, "total_cum_pnl": sum_136, "per_sym_sum": ps_136,
                       "vs_iter012": sum_136 - ITER012_BASELINE}

    print(f"\n[A3] Full 442k - global single thresh (sanity, should match iter_012):", flush=True)
    res_g = fit_global(df)
    out["full_global"] = res_g

    # ---- Step B: unbiased eval - date split 96-107 vs 108-119 ----------------
    print(f"\n[B] UNBIASED EVAL: dates 96-107 fit, 108-119 eval", flush=True)
    fit_df = df[df["date"] <= 107].copy()
    eval_df = df[df["date"] >= 108].copy()
    print(f"  fit_df:  n={len(fit_df):,} dates [{fit_df.date.min()},{fit_df.date.max()}]")
    print(f"  eval_df: n={len(eval_df):,} dates [{eval_df.date.min()},{eval_df.date.max()}]")

    out["unbiased_eval"] = {}

    # B1: global thresh
    print(f"\n[B1] global thresh - fit on 96-107:", flush=True)
    fit_g = fit_global(fit_df)
    eval_g_pnl, eval_g_ps, eval_g_na = apply_thresh_global(eval_df, fit_g["best"])
    print(f"  in-sample (96-107):  pnl={fit_g['cum_pnl']:+.4f}")
    print(f"  out-sample (108-119): pnl={eval_g_pnl:+.4f} per_sym_sum={sum(eval_g_ps):+.4f}")
    out["unbiased_eval"]["global"] = {
        "fit_thresh": fit_g["best"],
        "in_sample_pnl": fit_g["cum_pnl"],
        "out_sample_pnl": eval_g_pnl,
        "out_sample_per_sym_sum": float(sum(eval_g_ps)),
        "out_sample_n_active": eval_g_na,
    }

    # B2: quartile per-bucket
    print(f"\n[B2] quartile per-bucket - fit on 96-107:", flush=True)
    res_q4_fit, sum_q4_fit, _ = fit_buckets(fit_df, "bucket_q4", n_buckets=4, label_prefix="q")
    bucket_thresh_q4 = [res_q4_fit[f"q{b}"]["best"] for b in range(4)]
    print(f"\n  apply to 108-119:", flush=True)
    eval_q4_total, eval_q4_per_sym, eval_q4_per_bucket = apply_thresh_buckets(eval_df, "bucket_q4", bucket_thresh_q4)
    print(f"  in-sample (96-107):  sum={sum_q4_fit:+.4f}")
    print(f"  out-sample (108-119): sum={eval_q4_total:+.4f} per_sym_sum={sum(eval_q4_per_sym):+.4f}")
    out["unbiased_eval"]["quartile"] = {
        "fit_per_bucket": res_q4_fit,
        "fit_thresh_per_bucket": bucket_thresh_q4,
        "in_sample_total": sum_q4_fit,
        "out_sample_total": eval_q4_total,
        "out_sample_per_sym_sum": float(sum(eval_q4_per_sym)),
        "out_sample_per_bucket": eval_q4_per_bucket,
    }

    # B3: 1:2:2:1
    print(f"\n[B3] 1:2:2:1 per-bucket - fit on 96-107:", flush=True)
    res_136_fit, sum_136_fit, _ = fit_buckets(fit_df, "bucket_136", n_buckets=4, label_prefix="b136_")
    bucket_thresh_136 = [res_136_fit[f"b136_{b}"]["best"] for b in range(4)]
    print(f"\n  apply to 108-119:", flush=True)
    eval_136_total, eval_136_per_sym, eval_136_per_bucket = apply_thresh_buckets(eval_df, "bucket_136", bucket_thresh_136)
    print(f"  in-sample (96-107):  sum={sum_136_fit:+.4f}")
    print(f"  out-sample (108-119): sum={eval_136_total:+.4f} per_sym_sum={sum(eval_136_per_sym):+.4f}")
    out["unbiased_eval"]["b136"] = {
        "fit_per_bucket": res_136_fit,
        "fit_thresh_per_bucket": bucket_thresh_136,
        "in_sample_total": sum_136_fit,
        "out_sample_total": eval_136_total,
        "out_sample_per_sym_sum": float(sum(eval_136_per_sym)),
        "out_sample_per_bucket": eval_136_per_bucket,
    }

    # ---- Step C: decision summary -------------------------------------------
    out["summary"] = {
        "iter012_baseline_full": ITER012_BASELINE,
        "full_quartile_total": sum_q4,
        "full_136_total": sum_136,
        "full_quartile_gain": sum_q4 - ITER012_BASELINE,
        "full_136_gain": sum_136 - ITER012_BASELINE,
        # Unbiased gain = out_sample(per-bucket) - out_sample(global)
        "unbiased_quartile_gain": eval_q4_total - eval_g_pnl,
        "unbiased_136_gain": eval_136_total - eval_g_pnl,
        # In-sample gain on fit half (just for reference)
        "in_sample_quartile_gain": sum_q4_fit - fit_g["cum_pnl"],
        "in_sample_136_gain": sum_136_fit - fit_g["cum_pnl"],
    }

    print(f"\n{'='*78}\n=== T76 SUMMARY ===\n{'='*78}", flush=True)
    print(f"  iter_012 baseline (full 442k, single thresh): +{ITER012_BASELINE:.4f}", flush=True)
    print(f"  Full quartile sum:        {sum_q4:+.4f}  (vs iter_012: {sum_q4 - ITER012_BASELINE:+.4f})", flush=True)
    print(f"  Full 1:2:2:1 sum:         {sum_136:+.4f}  (vs iter_012: {sum_136 - ITER012_BASELINE:+.4f})", flush=True)
    print(f"  Unbiased eval (fit on 96-107, eval on 108-119):", flush=True)
    print(f"    global thresh:  in={fit_g['cum_pnl']:+.4f}  out={eval_g_pnl:+.4f}", flush=True)
    print(f"    quartile:       in={sum_q4_fit:+.4f}  out={eval_q4_total:+.4f}  gain_vs_global={eval_q4_total-eval_g_pnl:+.4f}", flush=True)
    print(f"    1:2:2:1:        in={sum_136_fit:+.4f}  out={eval_136_total:+.4f}  gain_vs_global={eval_136_total-eval_g_pnl:+.4f}", flush=True)

    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nwrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
