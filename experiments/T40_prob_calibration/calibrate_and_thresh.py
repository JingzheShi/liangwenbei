"""T40 — Probability calibration + DE 4D thresh.

Hypothesis: iter_006 (5-seed iter_005b h_60 ensemble) has accuracy near top of
public leaderboard but recall is low (active rate ~24%). The 5-seed average may
have squashed probs toward 1/3, which when run through (T_up, T_dn, d_up, d_dn)
gates pushes too many predictions to label=1.

Try 3 calibration methods (each fit cross-fold to avoid leakage on the eval fold):
  A) Platt scaling   — per-class one-vs-rest LogisticRegression
  B) Isotonic        — per-class one-vs-rest IsotonicRegression
  C) Temperature     — single scalar T applied to log-probs (softmax re-norm)

Calibrators are fit on folds {0..4}\{k} OOF preds, applied to fold k. This is
clean: the calibrator never sees fold-k labels.

Reuses DE 4D thresh logic from T38_3way_ensemble/de_thresh_3way.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

T26_DIR = os.path.join(ROOT, "experiments", "T26_domain_randomization")
T27_DIR = os.path.join(ROOT, "experiments", "T27_iter005")

SEEDS = [42, 1, 7, 13, 100]
N_FOLDS = 5
FEE = 0.0001
PROB_COLS = ["prob_0", "prob_1", "prob_2"]
EPS = 1e-12


# ---------- IO ----------
def load_iter005b_seed_avg(k: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """5-seed avg probs + meta for fold k. seed42 from T26, others from T27."""
    base = pd.read_parquet(os.path.join(T26_DIR, f"loso_pred_h60_aug_a_held{k}.parquet"))
    p_sum = base[PROB_COLS].to_numpy(np.float64).copy()
    n = len(base)
    for s in [1, 7, 13, 100]:
        df_s = pd.read_parquet(
            os.path.join(T27_DIR, f"loso_pred_h60_aug_a_seed{s}_held{k}.parquet")
        )
        if len(df_s) != n:
            raise RuntimeError(f"row count mismatch fold={k} seed={s}: {len(df_s)} vs {n}")
        if not (df_s["true_label"].values == base["true_label"].values).all():
            raise RuntimeError(f"label mismatch fold={k} seed={s}")
        p_sum += df_s[PROB_COLS].to_numpy(np.float64)
    p_avg = p_sum / 5.0
    return p_avg.astype(np.float64), base[["true_label", "midprice_t", "midprice_th"]].copy()


# ---------- Thresh + PnL ----------
def gate_asymmetric(probs: np.ndarray, T_up: float, T_dn: float,
                    d_up: float, d_dn: float) -> np.ndarray:
    p0 = probs[:, 0]; p1 = probs[:, 1]; p2 = probs[:, 2]
    take_up = (p2 >= T_up) & (p2 > p1 + d_up) & (p2 > p0)
    take_dn = (p0 >= T_dn) & (p0 > p1 + d_dn) & (p0 > p2)
    pred = np.full(probs.shape[0], 1, dtype=np.int8)
    pred[take_up] = 2
    pred[take_dn] = 0
    return pred


def vectorized_pnl(pred: np.ndarray, label: np.ndarray,
                   mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    side = pred.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


@dataclass
class FoldArrays:
    fold: int
    probs: np.ndarray
    label: np.ndarray
    mp_t: np.ndarray
    mp_th: np.ndarray


def make_objective(fas):
    label = [fa.label for fa in fas]
    mp_t = [fa.mp_t for fa in fas]
    mp_th = [fa.mp_th for fa in fas]
    probs = [fa.probs for fa in fas]
    K = len(fas)

    def f(x):
        Tu, Td, du, dd = x
        s = 0.0
        for k in range(K):
            pred = gate_asymmetric(probs[k], Tu, Td, du, dd)
            s += vectorized_pnl(pred, label[k], mp_t[k], mp_th[k]).sum()
        return -float(s)
    return f


def f0_5_macro(pred: np.ndarray, label: np.ndarray) -> float:
    """Macro F0.5 over 3 classes."""
    f_vals = []
    beta2 = 0.5 * 0.5
    for c in [0, 1, 2]:
        tp = int(((pred == c) & (label == c)).sum())
        fp = int(((pred == c) & (label != c)).sum())
        fn = int(((pred != c) & (label == c)).sum())
        if tp + fp == 0:
            prec = 0.0
        else:
            prec = tp / (tp + fp)
        if tp + fn == 0:
            rec = 0.0
        else:
            rec = tp / (tp + fn)
        if prec == 0.0 and rec == 0.0:
            f_vals.append(0.0)
        else:
            f_vals.append(((1 + beta2) * prec * rec) / (beta2 * prec + rec))
    return float(np.mean(f_vals))


def de_search(fas, bounds, seeds=(0, 1, 7), maxiter=60, popsize=20):
    f = make_objective(fas)
    runs = []
    for seed in seeds:
        t_run = time.time()
        result = differential_evolution(
            f, bounds=bounds, seed=seed, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        sum_pnl = -result.fun
        Tu, Td, du, dd = result.x
        per = []
        n_active = []
        all_pred = []
        all_label = []
        for k in range(len(fas)):
            pred = gate_asymmetric(fas[k].probs, Tu, Td, du, dd)
            per.append(float(vectorized_pnl(pred, fas[k].label, fas[k].mp_t, fas[k].mp_th).sum()))
            n_active.append(int((pred != 1).sum()))
            all_pred.append(pred)
            all_label.append(fas[k].label)
        all_pred_cat = np.concatenate(all_pred)
        all_label_cat = np.concatenate(all_label)
        f05 = f0_5_macro(all_pred_cat, all_label_cat)
        acc = float((all_pred_cat == all_label_cat).mean())
        runs.append({
            "seed": int(seed),
            "T_up": float(Tu), "T_dn": float(Td),
            "d_up": float(du), "d_dn": float(dd),
            "sum_cum_pnl": float(sum_pnl),
            "per_fold_pnl": per,
            "n_active_per_fold": n_active,
            "n_active_total": int(sum(n_active)),
            "active_rate": float(sum(n_active) / sum(len(fa.label) for fa in fas)),
            "n_pos_folds": int(sum(1 for x in per if x > 0)),
            "std_cum_pnl": float(np.std(per, ddof=0)),
            "f0_5_macro": float(f05),
            "accuracy": float(acc),
            "nfev": int(result.nfev),
            "elapsed_sec": float(time.time() - t_run),
        })
        print(f"    DE seed={seed}: T_up={Tu:.4f} T_dn={Td:.4f} d_up={du:.4f} d_dn={dd:.4f} "
              f"-> sum={sum_pnl:+.4f}  active={sum(n_active):,} "
              f"per_fold={[round(x,3) for x in per]} "
              f"({result.nfev} evals, {time.time()-t_run:.1f}s)", flush=True)
    runs.sort(key=lambda r: r["sum_cum_pnl"], reverse=True)
    return runs


def build_fas(probs_per_fold: List[np.ndarray], meta_per_fold: List[pd.DataFrame]) -> List[FoldArrays]:
    out = []
    for k in range(N_FOLDS):
        df = meta_per_fold[k]
        out.append(FoldArrays(
            fold=k,
            probs=probs_per_fold[k].astype(np.float64),
            label=df["true_label"].to_numpy(np.int64),
            mp_t=df["midprice_t"].to_numpy(np.float64),
            mp_th=df["midprice_th"].to_numpy(np.float64),
        ))
    return out


# ---------- Calibration ----------
def calibrate_platt_per_fold(probs_per_fold: List[np.ndarray],
                             labels_per_fold: List[np.ndarray]
                             ) -> List[np.ndarray]:
    """Per-class one-vs-rest LR using probs as a 1D feature.
    Calibrator for fold k is fit on folds {0..4}\\{k}.
    """
    out = [None] * N_FOLDS
    for k in range(N_FOLDS):
        train_p = np.concatenate([probs_per_fold[j] for j in range(N_FOLDS) if j != k])
        train_y = np.concatenate([labels_per_fold[j] for j in range(N_FOLDS) if j != k])
        out_p = np.zeros_like(probs_per_fold[k], dtype=np.float64)
        for c in range(3):
            # logit(p_c) input -> safer than raw p for LR
            X_train = np.log(np.clip(train_p[:, c:c+1], EPS, 1 - EPS) /
                             np.clip(1 - train_p[:, c:c+1], EPS, 1.0))
            y_train = (train_y == c).astype(int)
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(X_train, y_train)
            X_test = np.log(np.clip(probs_per_fold[k][:, c:c+1], EPS, 1 - EPS) /
                            np.clip(1 - probs_per_fold[k][:, c:c+1], EPS, 1.0))
            out_p[:, c] = lr.predict_proba(X_test)[:, 1]
        # Renormalize so probs sum to 1
        out_p /= out_p.sum(axis=1, keepdims=True).clip(min=EPS)
        out[k] = out_p
    return out


def calibrate_isotonic_per_fold(probs_per_fold: List[np.ndarray],
                                labels_per_fold: List[np.ndarray]
                                ) -> List[np.ndarray]:
    """Per-class one-vs-rest IsotonicRegression."""
    out = [None] * N_FOLDS
    for k in range(N_FOLDS):
        train_p = np.concatenate([probs_per_fold[j] for j in range(N_FOLDS) if j != k])
        train_y = np.concatenate([labels_per_fold[j] for j in range(N_FOLDS) if j != k])
        out_p = np.zeros_like(probs_per_fold[k], dtype=np.float64)
        for c in range(3):
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(train_p[:, c], (train_y == c).astype(float))
            out_p[:, c] = iso.predict(probs_per_fold[k][:, c])
        out_p /= out_p.sum(axis=1, keepdims=True).clip(min=EPS)
        out[k] = out_p
    return out


def calibrate_temperature_per_fold(probs_per_fold: List[np.ndarray],
                                   labels_per_fold: List[np.ndarray]
                                   ) -> Tuple[List[np.ndarray], List[float]]:
    """Single scalar T per fold; logit = log(p), z = logit/T, p = softmax(z).
    T fit by minimizing NLL on training (other) folds.
    """
    out = [None] * N_FOLDS
    Ts = []
    for k in range(N_FOLDS):
        train_p = np.concatenate([probs_per_fold[j] for j in range(N_FOLDS) if j != k])
        train_y = np.concatenate([labels_per_fold[j] for j in range(N_FOLDS) if j != k])
        log_p_train = np.log(np.clip(train_p, EPS, 1.0))

        def nll(T):
            z = log_p_train / T
            z = z - z.max(axis=1, keepdims=True)
            ez = np.exp(z)
            p = ez / ez.sum(axis=1, keepdims=True)
            return -np.mean(np.log(np.clip(p[np.arange(len(train_y)), train_y], EPS, 1.0)))

        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded",
                              options={"xatol": 1e-4})
        T_opt = float(res.x)
        Ts.append(T_opt)
        # apply
        log_p_test = np.log(np.clip(probs_per_fold[k], EPS, 1.0))
        z = log_p_test / T_opt
        z = z - z.max(axis=1, keepdims=True)
        ez = np.exp(z)
        out[k] = ez / ez.sum(axis=1, keepdims=True)
    return out, Ts


# ---------- Reliability diagram ----------
def reliability_per_class(probs: np.ndarray, labels: np.ndarray,
                          n_bins: int = 15) -> Dict:
    """Per-class one-vs-rest reliability data."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = {}
    for c in range(3):
        p_c = probs[:, c]
        y_c = (labels == c).astype(float)
        bin_idx = np.clip(np.digitize(p_c, edges) - 1, 0, n_bins - 1)
        bin_mid = []
        bin_acc = []
        bin_n = []
        for b in range(n_bins):
            m = bin_idx == b
            if m.sum() < 50:
                continue
            bin_mid.append(float(p_c[m].mean()))
            bin_acc.append(float(y_c[m].mean()))
            bin_n.append(int(m.sum()))
        ece = 0.0
        if len(bin_mid) > 0:
            total_n = sum(bin_n)
            ece = sum(abs(a - m) * n / total_n for a, m, n in zip(bin_acc, bin_mid, bin_n))
        out[f"class_{c}"] = {
            "bin_mid": bin_mid, "bin_acc": bin_acc, "bin_n": bin_n,
            "ece": float(ece),
        }
    return out


def plot_reliability(rel_baseline: Dict, rel_calibrated: Dict, fold: int,
                     calib_name: str, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    cls_names = ["class 0 (down)", "class 1 (flat)", "class 2 (up)"]
    for c, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect")
        b = rel_baseline[f"class_{c}"]
        if b["bin_mid"]:
            ax.plot(b["bin_mid"], b["bin_acc"], "o-",
                    label=f"raw (ECE={b['ece']:.3f})", color="C0")
        cal = rel_calibrated[f"class_{c}"]
        if cal["bin_mid"]:
            ax.plot(cal["bin_mid"], cal["bin_acc"], "s-",
                    label=f"{calib_name} (ECE={cal['ece']:.3f})", color="C1")
        ax.set_xlabel("predicted prob")
        ax.set_ylabel("empirical freq")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"fold {fold} — {cls_names[c]}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Reliability diagram — fold {fold} — {calib_name} vs raw",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------- Main ----------
def evaluate_variant(name: str, probs_per_fold: List[np.ndarray],
                     meta_per_fold: List[pd.DataFrame]) -> dict:
    print(f"\n=== {name} ===", flush=True)
    fas = build_fas(probs_per_fold, meta_per_fold)
    raw_pf = []
    for fa in fas:
        pred = fa.probs.argmax(axis=1).astype(np.int8)
        raw_pf.append(float(vectorized_pnl(pred, fa.label, fa.mp_t, fa.mp_th).sum()))
    print(f"  raw argmax sum={sum(raw_pf):+.4f} per_fold={[round(x,3) for x in raw_pf]}",
          flush=True)

    bounds = [(0.34, 0.75), (0.34, 0.75), (0.0, 0.30), (0.0, 0.30)]
    de_runs = de_search(fas, bounds, seeds=(0, 1, 7))
    best = de_runs[0]
    print(f"  DE best: sum={best['sum_cum_pnl']:+.4f} active={best['n_active_total']:,} "
          f"({best['active_rate']*100:.2f}%) F0.5={best['f0_5_macro']:.4f} "
          f"acc={best['accuracy']:.4f}",
          flush=True)
    return {
        "variant": name,
        "raw_argmax_sum": float(sum(raw_pf)),
        "raw_argmax_per_fold": raw_pf,
        "de_best": best,
        "de_runs": de_runs,
    }


def main():
    t0 = time.time()
    print(f"=== T40 prob calibration + DE thresh ===", flush=True)
    print(f"  loading iter_005b 5-seed avg ...", flush=True)
    probs_raw = []
    meta = []
    for k in range(N_FOLDS):
        p, df = load_iter005b_seed_avg(k)
        probs_raw.append(p)
        meta.append(df)
        print(f"    fold {k}: {len(df):,} rows", flush=True)
    print(f"  loaded ({time.time()-t0:.1f}s)", flush=True)

    labels_per_fold = [df["true_label"].to_numpy(np.int64) for df in meta]

    # Build calibrated variants
    print(f"\n--- Fitting calibrators ---", flush=True)
    print(f"  Platt ...", flush=True)
    probs_platt = calibrate_platt_per_fold(probs_raw, labels_per_fold)
    print(f"  Isotonic ...", flush=True)
    probs_isotonic = calibrate_isotonic_per_fold(probs_raw, labels_per_fold)
    print(f"  Temperature ...", flush=True)
    probs_temp, T_per_fold = calibrate_temperature_per_fold(probs_raw, labels_per_fold)
    print(f"    T per fold: {[round(t, 4) for t in T_per_fold]}", flush=True)

    # Generate reliability diagrams (per fold)
    print(f"\n--- Reliability diagrams ---", flush=True)
    rel_summary = {"raw": [], "platt": [], "isotonic": [], "temperature": []}
    for k in range(N_FOLDS):
        rel_raw = reliability_per_class(probs_raw[k], labels_per_fold[k])
        rel_platt = reliability_per_class(probs_platt[k], labels_per_fold[k])
        rel_iso = reliability_per_class(probs_isotonic[k], labels_per_fold[k])
        rel_temp = reliability_per_class(probs_temp[k], labels_per_fold[k])
        rel_summary["raw"].append(rel_raw)
        rel_summary["platt"].append(rel_platt)
        rel_summary["isotonic"].append(rel_iso)
        rel_summary["temperature"].append(rel_temp)
        # plot raw vs each calibrated
        plot_reliability(rel_raw, rel_platt, k, "Platt",
                         os.path.join(HERE, f"reliability_diag_fold{k}_platt.png"))
        plot_reliability(rel_raw, rel_iso, k, "Isotonic",
                         os.path.join(HERE, f"reliability_diag_fold{k}_isotonic.png"))
        plot_reliability(rel_raw, rel_temp, k, "Temperature",
                         os.path.join(HERE, f"reliability_diag_fold{k}_temperature.png"))
        # Combined plot
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for c, ax in enumerate(axes):
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="perfect")
            for src, lbl, color in [
                (rel_raw, "raw", "C0"),
                (rel_platt, "Platt", "C1"),
                (rel_iso, "Iso", "C2"),
                (rel_temp, "Temp", "C3"),
            ]:
                d = src[f"class_{c}"]
                if d["bin_mid"]:
                    ax.plot(d["bin_mid"], d["bin_acc"], "o-",
                            label=f"{lbl} (ECE={d['ece']:.3f})", color=color)
            ax.set_xlabel("predicted prob")
            ax.set_ylabel("empirical freq")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"fold {k} — class {c}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Reliability diagram — fold {k} — all variants", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"reliability_diag_fold{k}.png"), dpi=110)
        plt.close(fig)
        print(f"  fold {k}: ECEs raw={[round(rel_raw[f'class_{c}']['ece'],4) for c in range(3)]} "
              f"platt={[round(rel_platt[f'class_{c}']['ece'],4) for c in range(3)]} "
              f"iso={[round(rel_iso[f'class_{c}']['ece'],4) for c in range(3)]} "
              f"temp={[round(rel_temp[f'class_{c}']['ece'],4) for c in range(3)]}", flush=True)

    # ---------- DE thresh per variant ----------
    variants = {
        "iter_006_baseline_raw": probs_raw,
        "Platt": probs_platt,
        "Isotonic": probs_isotonic,
        "Temperature": probs_temp,
    }
    all_results = {}
    for name, ppf in variants.items():
        all_results[name] = evaluate_variant(name, ppf, meta)

    # Build summary table
    print(f"\n=== SUMMARY ===", flush=True)
    summary_rows = []
    base_sum = all_results["iter_006_baseline_raw"]["de_best"]["sum_cum_pnl"]
    for name, r in all_results.items():
        b = r["de_best"]
        summary_rows.append({
            "variant": name,
            "sum_cum_pnl": b["sum_cum_pnl"],
            "n_pos_folds": b["n_pos_folds"],
            "active_rate_pct": b["active_rate"] * 100,
            "n_active": b["n_active_total"],
            "f0_5_macro": b["f0_5_macro"],
            "accuracy": b["accuracy"],
            "vs_baseline": b["sum_cum_pnl"] - base_sum,
            "T_up": b["T_up"], "T_dn": b["T_dn"],
            "d_up": b["d_up"], "d_dn": b["d_dn"],
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False), flush=True)

    # Save outputs
    rel_summary_serializable = {
        method: [
            {f"class_{c}": {k: v for k, v in fold_data[f"class_{c}"].items()}
             for c in range(3)}
            for fold_data in folds
        ]
        for method, folds in rel_summary.items()
    }
    out = {
        "task": "T40 probability calibration + DE 4D thresh",
        "iter_006_baseline_historical_sum": 13.61,
        "iter_006_baseline_reproduced_sum": float(base_sum),
        "iter_006_baseline_active_rate_pct": float(
            all_results["iter_006_baseline_raw"]["de_best"]["active_rate"] * 100
        ),
        "T_per_fold_temperature": T_per_fold,
        "summary_table": summary_rows,
        "results_by_variant": all_results,
        "reliability_eces": {
            method: [
                {f"class_{c}": folds[k][f"class_{c}"]["ece"]
                 for c in range(3)} for k in range(N_FOLDS)
            ]
            for method, folds in rel_summary.items()
        },
        "elapsed_sec": float(time.time() - t0),
    }
    out_path = os.path.join(HERE, "results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nDone in {time.time()-t0:.1f}s -> {out_path}", flush=True)

    # Decide
    best_name, best_sum = max(
        ((n, r["de_best"]["sum_cum_pnl"]) for n, r in all_results.items()),
        key=lambda x: x[1]
    )
    uplift = best_sum - base_sum
    print(f"\nBEST variant: {best_name} sum={best_sum:+.4f} uplift vs baseline={uplift:+.4f}",
          flush=True)
    print(f"\nRESULT: task=calibration metrics={{best_variant={best_name}, "
          f"best_sum={best_sum:+.4f}, uplift={uplift:+.4f}, "
          f"baseline_sum={base_sum:+.4f}}}",
          flush=True)


if __name__ == "__main__":
    main()
