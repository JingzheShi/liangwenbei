"""H-Horizon study: analyses A, B, C, D, E, F.

A — signal-scaling law (std(y_h) ∝ √h)
B — bid-ask-bounce noise decomposition (Roll ρ + BV/RV + bounce share by h)
C — IC by horizon (Spearman, Pearson, R², train-vs-test gap)
D — cost-vs-signal ratio at best threshold
E — decile calibration (predicted ŷ vs realized return per quantile)
F — h=60-model with √(h/60) threshold scaling vs horizon-specific models

Reads cache from /root/projects/liangwenbei_workdir/ablation_runs/cache_log1p
Reads model predictions from ./preds/pred_h{H}_seed{S}.npz
Writes figures to ./ and metrics_by_horizon.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr, pearsonr

HERE = Path(__file__).parent
CACHE_DIR = Path("/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p")
PRED_DIR = HERE / "preds"
HORIZONS = [5, 10, 20, 40, 60]
SEEDS = [1, 2, 3, 4, 5]
SYMS = [0, 1, 2, 3, 4]
FEE = 0.0001

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

CMAP = plt.cm.viridis
HCOLOR = {h: CMAP(i / (len(HORIZONS) - 1)) for i, h in enumerate(HORIZONS)}

# ============================================================
# Helpers
# ============================================================

def y_regression(mp_t, mp_th):
    return ((mp_th - mp_t) / (mp_t + 1.0)).astype(np.float32)


def pnl_regression(pred_dir, mp_t, mp_th):
    """Vectorized PnL identical to existing pipeline (FEE = 1bp per side).

    pred_dir ∈ {-1, 0, +1}.
    """
    side = pred_dir.astype(np.float64)
    abs_side = np.abs(side)
    diff = mp_th.astype(np.float64) - mp_t.astype(np.float64)
    fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t.astype(np.float64) + 1.0
    return (side * diff - fee) / denom


def best_threshold_de(yhat, mp_t, mp_th, bounds=None, seeds=(0, 1, 2)):
    if bounds is None:
        s = float(np.std(yhat))
        # search up to ~3σ of yhat
        hi = max(3.0 * s, 1e-6)
        bounds = [(0.0, hi), (0.0, hi)]

    def obj(x):
        th_up, th_dn = x
        pred = np.zeros_like(yhat, dtype=np.int8)
        pred[yhat > th_up] = 1
        pred[yhat < -th_dn] = -1
        return -float(pnl_regression(pred, mp_t, mp_th).sum())

    best = None
    for sd in seeds:
        r = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=50, popsize=16,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        if best is None or r.fun < best.fun:
            best = r
    th_up, th_dn = float(best.x[0]), float(best.x[1])
    pnl = -float(best.fun)
    return th_up, th_dn, pnl


def apply_threshold(yhat, th_up, th_dn):
    pred = np.zeros_like(yhat, dtype=np.int8)
    pred[yhat > th_up] = 1
    pred[yhat < -th_dn] = -1
    return pred


# ============================================================
# A — Signal scaling law
# ============================================================

def analysis_A():
    print("\n[A] signal scaling law")
    d = np.load(CACHE_DIR / "schemeP_train.npz")
    sym_all = d["sym"][:]
    mp_t = d["mp_t"].astype(np.float64)
    rows = []
    for H in HORIZONS:
        mp_h = d[f"mp_t{H}"].astype(np.float64)
        y = y_regression(mp_t, mp_h)
        # global
        rows.append(dict(h=H, sym=-1, std=float(y.std()), mean_abs=float(np.mean(np.abs(y))),
                          n=int(len(y))))
        for s in SYMS:
            m = (sym_all == s)
            rows.append(dict(h=H, sym=int(s), std=float(y[m].std()),
                              mean_abs=float(np.mean(np.abs(y[m]))),
                              n=int(m.sum())))
    d.close()

    # Fit log-log slope on global rows
    hs = np.array(HORIZONS, dtype=float)
    g = {r["h"]: r for r in rows if r["sym"] == -1}
    stds = np.array([g[h]["std"] for h in HORIZONS])
    mas = np.array([g[h]["mean_abs"] for h in HORIZONS])
    slope_std, intercept_std = np.polyfit(np.log(hs), np.log(stds), 1)
    slope_ma, intercept_ma = np.polyfit(np.log(hs), np.log(mas), 1)
    print(f"  global slope log std(y_h) ~ {slope_std:.4f} * log(h)  (expect 0.5)")
    print(f"  global slope log mean|y_h| ~ {slope_ma:.4f} * log(h)")

    per_sym_slopes = {}
    for s in SYMS:
        std_s = np.array([next(r["std"] for r in rows if r["h"] == h and r["sym"] == s) for h in HORIZONS])
        sl, _ = np.polyfit(np.log(hs), np.log(std_s), 1)
        per_sym_slopes[int(s)] = float(sl)
        print(f"  sym={s}: slope={sl:.4f}  std_h60={std_s[-1]:.4e}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(hs, stds, "o-", color="black", label="global", lw=2, ms=7)
    ax[0].plot(hs, np.exp(intercept_std) * hs ** slope_std, "--", color="red", alpha=0.6,
               label=f"fit: slope={slope_std:.3f}")
    ax[0].plot(hs, stds[0] * np.sqrt(hs / hs[0]), ":", color="blue", alpha=0.6, label="√h reference")
    for s in SYMS:
        std_s = np.array([next(r["std"] for r in rows if r["h"] == h and r["sym"] == s) for h in HORIZONS])
        ax[0].plot(hs, std_s, "x-", color=plt.cm.tab10(s), alpha=0.55, ms=6, label=f"sym={s}")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("horizon h (ticks)")
    ax[0].set_ylabel("std( y_h )")
    ax[0].set_title("A1. Signal scaling: σ(y_h) ∝ h^p")
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].grid(True, which="both", alpha=0.3)

    ratios = stds / (stds[0] * np.sqrt(hs / hs[0]))
    ax[1].plot(hs, ratios, "o-", color="black", label="global", lw=2)
    for s in SYMS:
        std_s = np.array([next(r["std"] for r in rows if r["h"] == h and r["sym"] == s) for h in HORIZONS])
        ax[1].plot(hs, std_s / (std_s[0] * np.sqrt(hs / hs[0])), "x-",
                   color=plt.cm.tab10(s), alpha=0.55, label=f"sym={s}")
    ax[1].axhline(1.0, color="blue", ls="--", alpha=0.5, label="perfect √h")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("horizon h")
    ax[1].set_ylabel("σ(y_h) / [σ(y_5) · √(h/5)]")
    ax[1].set_title("A2. Deviation from √h scaling (1.0 = perfect)")
    ax[1].legend(ncol=2, fontsize=8)
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(HERE / "fig_A_signal_scaling.pdf")
    plt.close()
    print(f"  -> fig_A_signal_scaling.pdf")

    return dict(
        slope_std_global=float(slope_std),
        slope_meanabs_global=float(slope_ma),
        per_sym_slopes=per_sym_slopes,
        rows=rows,
    )


# ============================================================
# B — Bid-ask bounce noise decomposition
# ============================================================

def analysis_B():
    print("\n[B] bid-ask bounce / RV-BV decomposition")
    d = np.load(CACHE_DIR / "schemeP_train.npz")
    sym_all = d["sym"][:]
    date_all = d["date"][:]
    sess_all = d["sess_idx"][:]
    t_all = d["t"][:]
    mp_t = d["mp_t"].astype(np.float64)
    d.close()

    # 1-tick log return; mp_t here is already log1p, so the differencing within a session
    # is itself a log-return-like quantity. Use diff within (sym, date, sess).
    # Order: sort by sym/date/sess/t to ensure consecutive diff is in time.
    order = np.lexsort((t_all, sess_all, date_all, sym_all))
    mp_o = mp_t[order]
    sym_o = sym_all[order]
    date_o = date_all[order]
    sess_o = sess_all[order]

    # Compute first-difference (≈ tick return for log1p mp_t)
    r_o = np.empty_like(mp_o)
    r_o[1:] = mp_o[1:] - mp_o[:-1]
    r_o[0] = 0.0
    # Mark cross-group boundaries as NaN (first tick in each session has no predecessor)
    group_key = sym_o.astype(np.int64) * 1_000_000 + date_o.astype(np.int64) * 100 + sess_o.astype(np.int64)
    is_boundary = np.empty(len(group_key), dtype=bool)
    is_boundary[0] = True
    is_boundary[1:] = group_key[1:] != group_key[:-1]
    r_o[is_boundary] = np.nan

    valid = ~np.isnan(r_o)
    print(f"  total ticks: {len(r_o):,}, valid 1-step returns: {valid.sum():,}")

    # Global 1-lag autocorrelation
    r = r_o[valid]
    # pair with previous within same group
    r_lag = np.empty_like(r_o); r_lag[:] = np.nan
    r_lag[1:] = r_o[:-1]
    r_lag[is_boundary] = np.nan
    m_pair = ~np.isnan(r_o) & ~np.isnan(r_lag)
    rho_global = float(np.corrcoef(r_o[m_pair], r_lag[m_pair])[0, 1])
    print(f"  global ρ(r_t, r_{{t-1}}) = {rho_global:.5f}  (negative ⇒ bid-ask bounce)")

    # Per sym × date 1-lag autocorrelation
    per_sym_rho = {}
    for s in SYMS:
        per_dates = []
        ds = np.unique(date_all[sym_all == s])
        for dd in ds:
            mask = (sym_all == s) & (date_all == dd)
            order_d = np.lexsort((t_all[mask], sess_all[mask]))
            r_d = mp_t[mask][order_d]
            ses_d = sess_all[mask][order_d]
            sess_change = np.r_[True, ses_d[1:] != ses_d[:-1]]
            dr = np.r_[np.nan, np.diff(r_d)]
            dr[sess_change] = np.nan
            dr_lag = np.r_[np.nan, dr[:-1]]
            m = ~np.isnan(dr) & ~np.isnan(dr_lag)
            if m.sum() > 100:
                rr = np.corrcoef(dr[m], dr_lag[m])[0, 1]
                if np.isfinite(rr):
                    per_dates.append(float(rr))
        per_sym_rho[int(s)] = dict(
            mean=float(np.mean(per_dates)),
            median=float(np.median(per_dates)),
            std=float(np.std(per_dates)),
            n=len(per_dates),
        )
        print(f"  sym={s}: ρ mean={per_sym_rho[int(s)]['mean']:+.5f}  median={per_sym_rho[int(s)]['median']:+.5f}  n_dates={len(per_dates)}")

    # Bounce-noise share by horizon: compare var(sum_h r_t) vs h * var(r_t)
    # If ρ < 0, var(sum) < h * var(r): the "missing variance" is the bounce reduction.
    var_r = float(np.nanvar(r_o))
    bounce_table = []
    for H in HORIZONS:
        # need sum over consecutive H returns within same session
        # use cumulative sum + Hth diff
        cs = np.cumsum(np.where(np.isnan(r_o), 0.0, r_o))
        # rolling H-tick sum
        sum_h = np.full_like(cs, np.nan)
        sum_h[H:] = cs[H:] - cs[:-H]
        # validity: all H steps within same group
        same = np.ones(len(r_o), dtype=bool)
        for k in range(1, H + 1):
            same[k:] &= (group_key[k:] == group_key[:-k])
            same[:k] = False
        valid_h = same & ~np.isnan(sum_h)
        sum_h_v = sum_h[valid_h]
        var_sum = float(np.var(sum_h_v))
        bounce_share = 1.0 - var_sum / (H * var_r)
        # Decomposition: var(sum_h r) = h*var(r) + 2 sum_{lag=1..h-1} (h-lag) cov(r, r_lag)
        # bounce_share > 0 means the bounce reduced the variance below i.i.d.
        bounce_table.append(dict(
            h=int(H),
            var_sum=var_sum,
            h_times_var=float(H * var_r),
            bounce_share=float(bounce_share),
            n=int(valid_h.sum()),
        ))
        print(f"  h={H}: var(sum_h r)={var_sum:.3e}  h*var(r)={H*var_r:.3e}  bounce_share={bounce_share:+.4f}")

    # BV vs RV by horizon: jshare = (RV - BV) / RV.
    # For each (sym, date, sess) window we compute realized vs bipower variation
    # over h consecutive ticks (rolling), then average jshare per horizon.
    rv_bv_rows = []
    for H in HORIZONS:
        sq = np.where(np.isnan(r_o), 0.0, r_o ** 2)
        ab = np.where(np.isnan(r_o), 0.0, np.abs(r_o))
        # Need |r_t| * |r_{t-1}|
        ab_prev = np.r_[0.0, ab[:-1]]
        ab_prev[is_boundary] = 0.0  # invalid at session start
        bp_term = ab * ab_prev  # bipower per tick
        cs_rv = np.cumsum(sq)
        cs_bv = np.cumsum(bp_term)
        rv_h = np.full_like(cs_rv, np.nan)
        bv_h = np.full_like(cs_bv, np.nan)
        rv_h[H:] = cs_rv[H:] - cs_rv[:-H]
        bv_h[H:] = cs_bv[H:] - cs_bv[:-H]
        bv_h *= (np.pi / 2.0)
        same = np.ones(len(r_o), dtype=bool)
        for k in range(1, H + 1):
            same[k:] &= (group_key[k:] == group_key[:-k])
            same[:k] = False
        m = same & ~np.isnan(rv_h) & (rv_h > 1e-12)
        rv_m = rv_h[m]; bv_m = bv_h[m]
        jshare = 1.0 - bv_m / rv_m
        rv_bv_rows.append(dict(
            h=int(H),
            mean_rv=float(rv_m.mean()),
            mean_bv=float(bv_m.mean()),
            mean_jshare=float(np.mean(jshare)),
            median_jshare=float(np.median(jshare)),
            n=int(len(jshare)),
        ))
        print(f"  h={H}: mean_RV={rv_m.mean():.3e}  mean_BV={bv_m.mean():.3e}  mean_jshare={np.mean(jshare):+.4f}")

    # Figure B
    hs = np.array(HORIZONS, dtype=float)
    bs = np.array([r["bounce_share"] for r in bounce_table])
    js = np.array([r["mean_jshare"] for r in rv_bv_rows])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    # Panel B1: bounce share by h
    ax[0].plot(hs, bs * 100, "o-", color="C0", lw=2, ms=8)
    ax[0].axhline(0, color="k", ls=":", alpha=0.5)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("horizon h"); ax[0].set_ylabel("bounce noise share  1 − var(Σr_h)/(h·var r)  [%]")
    ax[0].set_title(f"B1. Bid-ask bounce share by h  (global ρ = {rho_global:+.4f})")
    ax[0].grid(True, alpha=0.3)
    for hi, v in zip(hs, bs * 100):
        ax[0].annotate(f"{v:+.1f}%", xy=(hi, v), xytext=(0, 8), textcoords="offset points",
                       ha="center", fontsize=9)
    # Panel B2: jshare by h
    ax[1].plot(hs, js * 100, "o-", color="C3", lw=2, ms=8, label="mean")
    ax[1].plot(hs, [r["median_jshare"] * 100 for r in rv_bv_rows], "s--", color="C4", lw=1.5, label="median")
    ax[1].axhline(0, color="k", ls=":", alpha=0.5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("horizon h"); ax[1].set_ylabel("jump share  (RV − BV) / RV  [%]")
    ax[1].set_title("B2. Jump-share decomposition (jumps vs continuous noise)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    # Panel B3: per-sym ρ
    syms = sorted(per_sym_rho.keys())
    means = [per_sym_rho[s]["mean"] for s in syms]
    medians = [per_sym_rho[s]["median"] for s in syms]
    x = np.arange(len(syms))
    ax[2].bar(x - 0.18, means, 0.36, color="C0", label="mean ρ across dates")
    ax[2].bar(x + 0.18, medians, 0.36, color="C1", label="median ρ across dates")
    ax[2].axhline(0, color="k", ls="-", alpha=0.5)
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"sym {s}" for s in syms])
    ax[2].set_ylabel("ρ(r_t, r_{t-1})")
    ax[2].set_title("B3. Per-sym 1-lag autocorr (negative ⇒ bid-ask bounce)")
    ax[2].legend()
    ax[2].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(HERE / "fig_B_bounce_noise.pdf")
    plt.close()
    print(f"  -> fig_B_bounce_noise.pdf")

    return dict(
        rho_global=rho_global,
        per_sym_rho=per_sym_rho,
        bounce_share_by_h=bounce_table,
        rv_bv_by_h=rv_bv_rows,
    )


# ============================================================
# C — IC by horizon (using saved predictions, avg over seeds)
# ============================================================

def _avg_preds(H, key_yhat, key_y, key_extra=None):
    """Average yhat over seeds; y is identical across seeds so take seed1."""
    preds = []
    y0 = None; extras = {}
    for S in SEEDS:
        p = PRED_DIR / f"pred_h{H}_seed{S}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        preds.append(d[key_yhat])
        if y0 is None:
            y0 = d[key_y]
            if key_extra:
                for k in key_extra:
                    extras[k] = d[k]
        d.close()
    if not preds:
        return None
    yhat_avg = np.mean(preds, axis=0)
    return yhat_avg, y0, extras, len(preds)


def analysis_C():
    print("\n[C] IC by horizon")
    rows = []
    per_seed_rows = []
    for H in HORIZONS:
        # per-seed (for error bars)
        for S in SEEDS:
            p = PRED_DIR / f"pred_h{H}_seed{S}.npz"
            if not p.exists():
                continue
            d = np.load(p)
            for split in ["train", "val", "test"]:
                yhat = d[f"yhat_{split}"]
                y = d[f"y_{split}"]
                sym = d[f"sym_{split}"]
                ic = spearmanr(yhat, y)[0]
                r = pearsonr(yhat, y)[0]
                r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
                row = dict(h=H, seed=S, split=split, sym=-1,
                           ic=float(ic), r=float(r), r2=float(r2),
                           std_yhat=float(yhat.std()), mean_abs_y=float(np.mean(np.abs(y))))
                per_seed_rows.append(row)
                for ss in SYMS:
                    m = (sym == ss)
                    if m.sum() < 50:
                        continue
                    ic_s = spearmanr(yhat[m], y[m])[0]
                    per_seed_rows.append(dict(h=H, seed=S, split=split, sym=int(ss),
                                              ic=float(ic_s), r=float(np.nan), r2=float(np.nan),
                                              std_yhat=float(yhat[m].std()), mean_abs_y=float(np.mean(np.abs(y[m])))))
            d.close()

        # avg-pred
        for split in ["train", "val", "test"]:
            r = _avg_preds(H, f"yhat_{split}", f"y_{split}", key_extra=[f"sym_{split}"])
            if r is None: continue
            yhat_avg, y, extras, n_seed = r
            sym = extras[f"sym_{split}"]
            ic = spearmanr(yhat_avg, y)[0]
            pr = pearsonr(yhat_avg, y)[0]
            r2 = 1.0 - np.sum((y - yhat_avg) ** 2) / np.sum((y - y.mean()) ** 2)
            rows.append(dict(h=H, agg="seedavg", split=split, sym=-1,
                              ic=float(ic), r=float(pr), r2=float(r2),
                              n_seed=int(n_seed),
                              std_yhat=float(yhat_avg.std()),
                              mean_abs_y=float(np.mean(np.abs(y)))))
            for ss in SYMS:
                m = (sym == ss)
                if m.sum() < 50: continue
                ic_s = spearmanr(yhat_avg[m], y[m])[0]
                rows.append(dict(h=H, agg="seedavg", split=split, sym=int(ss),
                                  ic=float(ic_s), n=int(m.sum())))
    print(f"  seed-avg IC rows: {len(rows)}, per-seed rows: {len(per_seed_rows)}")

    # ---- Figure C1: IC by h × split (with seed error bars) ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for split, col in zip(["train", "val", "test"], ["C0", "C1", "C3"]):
        ic_means = []; ic_stds = []
        for H in HORIZONS:
            vs = [r["ic"] for r in per_seed_rows if r["h"] == H and r["split"] == split and r["sym"] == -1]
            ic_means.append(np.mean(vs)); ic_stds.append(np.std(vs))
        ax[0].errorbar(HORIZONS, ic_means, yerr=ic_stds, fmt="o-", capsize=4, lw=2,
                       label=f"{split}", color=col, ms=7)
    ax[0].set_xscale("log"); ax[0].set_xlabel("horizon h"); ax[0].set_ylabel("Spearman IC")
    ax[0].set_title("C1. Spearman IC by horizon  (5 seeds, mean ± std)")
    ax[0].grid(True, alpha=0.3); ax[0].legend()
    ax[0].axhline(0, color="k", ls=":", alpha=0.5)

    # ---- Figure C2: train→test gap ----
    gap = []
    for H in HORIZONS:
        ic_tr = [r["ic"] for r in per_seed_rows if r["h"] == H and r["split"] == "train" and r["sym"] == -1]
        ic_te = [r["ic"] for r in per_seed_rows if r["h"] == H and r["split"] == "test" and r["sym"] == -1]
        gap.append((np.mean(ic_tr) - np.mean(ic_te), np.std(ic_tr) + np.std(ic_te)))
    ax[1].errorbar(HORIZONS, [g[0] for g in gap], yerr=[g[1] for g in gap],
                   fmt="s-", capsize=4, color="purple", lw=2, ms=8, label="train − test IC")
    ax[1].set_xscale("log"); ax[1].set_xlabel("horizon h"); ax[1].set_ylabel("IC gap (train − test)")
    ax[1].set_title("C2. Generalization gap by horizon  (larger ⇒ worse OOD)")
    ax[1].grid(True, alpha=0.3); ax[1].legend()
    ax[1].axhline(0, color="k", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(HERE / "fig_C_IC_by_horizon.pdf")
    plt.close()
    print("  -> fig_C_IC_by_horizon.pdf")

    # Per-sym IC test heatmap (seed-avg)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    M = np.zeros((len(HORIZONS), len(SYMS)))
    for i, H in enumerate(HORIZONS):
        for j, s in enumerate(SYMS):
            ics = [r["ic"] for r in rows if r["h"] == H and r["split"] == "test" and r["sym"] == s]
            M[i, j] = ics[0] if ics else np.nan
    im = ax.imshow(M, cmap="RdYlGn", vmin=-max(0.1, abs(np.nanmin(M))), vmax=max(0.1, abs(np.nanmax(M))), aspect="auto")
    for i in range(len(HORIZONS)):
        for j in range(len(SYMS)):
            ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center",
                    color="black", fontsize=10)
    ax.set_yticks(range(len(HORIZONS))); ax.set_yticklabels([f"h={h}" for h in HORIZONS])
    ax.set_xticks(range(len(SYMS))); ax.set_xticklabels([f"sym {s}" for s in SYMS])
    ax.set_title("C3. Test IC per (horizon, sym) — seed-averaged")
    plt.colorbar(im, ax=ax, label="Spearman IC")
    plt.tight_layout()
    plt.savefig(HERE / "fig_C2_IC_train_vs_test_gap.pdf")
    plt.close()
    print("  -> fig_C2_IC_train_vs_test_gap.pdf")

    return dict(per_seed=per_seed_rows, seedavg=rows)


# ============================================================
# D — cost vs signal ratio at best threshold
# ============================================================

def analysis_D():
    print("\n[D] cost-signal ratio at best threshold")
    rows = []
    for H in HORIZONS:
        rv = _avg_preds(H, "yhat_val", "y_val", key_extra=["mp_t_val", "mp_th_val"])
        rt = _avg_preds(H, "yhat_test", "y_test", key_extra=["mp_t_test", "mp_th_test"])
        if rv is None or rt is None:
            continue
        yhat_val, y_val, extv, _ = rv
        yhat_test, y_test, extt, _ = rt
        mp_t_v = extv["mp_t_val"]; mp_th_v = extv["mp_th_val"]
        mp_t_t = extt["mp_t_test"]; mp_th_t = extt["mp_th_test"]

        th_up, th_dn, pnl_val = best_threshold_de(yhat_val, mp_t_v, mp_th_v)
        pred_test = apply_threshold(yhat_test, th_up, th_dn)
        n_trades = int(np.sum(pred_test != 0))
        pnl_test_arr = pnl_regression(pred_test, mp_t_t, mp_th_t)
        pnl_test = float(pnl_test_arr.sum())
        # cost: FEE * abs_side * abs((mp_th+1)+(mp_t+1)) / (mp_t+1)
        side = pred_test.astype(np.float64)
        abs_side = np.abs(side)
        diff = mp_th_t - mp_t_t
        denom = mp_t_t + 1.0
        gross = (side * diff) / denom
        cost_term = (FEE * abs_side * np.abs((mp_th_t + 1.0) + (mp_t_t + 1.0))) / denom
        # restrict to actively traded rows
        mask = pred_test != 0
        gross_trades = gross[mask]
        cost_trades = cost_term[mask]
        signal_abs_mean = float(np.mean(np.abs(diff[mask] / denom[mask])))
        cost_mean = float(cost_trades.mean()) if len(cost_trades) else 0.0
        cost_signal_ratio = cost_mean / signal_abs_mean if signal_abs_mean > 0 else np.nan

        # net per trade
        net_per_trade = float((gross_trades - cost_trades).mean()) if len(gross_trades) else 0.0
        rows.append(dict(
            h=H, th_up=th_up, th_dn=th_dn,
            pnl_val=pnl_val, pnl_test=pnl_test,
            n_trades=n_trades, n_total=int(len(pred_test)),
            trade_rate=float(n_trades / len(pred_test)),
            signal_abs_mean=signal_abs_mean,
            cost_mean=cost_mean,
            cost_signal_ratio=cost_signal_ratio,
            net_per_trade=net_per_trade,
        ))
        print(f"  h={H}: th_up={th_up:.5f}  th_dn={th_dn:.5f}  pnl_test={pnl_test:+.3f}  "
              f"trades={n_trades:,}/{len(pred_test):,}  cost/signal={cost_signal_ratio:.3f}  "
              f"net_pnl_per_trade={net_per_trade:+.2e}")

    # Figure D
    hs = np.array(HORIZONS, dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    csr = np.array([r["cost_signal_ratio"] for r in rows])
    ax[0].plot(hs, csr, "o-", color="C3", lw=2, ms=8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("horizon h")
    ax[0].set_ylabel("cost / |signal|  (at best threshold)")
    ax[0].set_title("D1. Cost / signal ratio decays as h grows")
    ax[0].grid(True, alpha=0.3)
    for hi, v in zip(hs, csr):
        ax[0].annotate(f"{v:.2f}", xy=(hi, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)

    npt = np.array([r["net_per_trade"] for r in rows])
    ax[1].plot(hs, npt * 1e4, "o-", color="C2", lw=2, ms=8)
    ax[1].axhline(0, color="k", ls=":", alpha=0.5)
    ax[1].set_xscale("log"); ax[1].set_xlabel("horizon h")
    ax[1].set_ylabel("net PnL / trade (bp)")
    ax[1].set_title("D2. Net PnL per trade by horizon")
    ax[1].grid(True, alpha=0.3)

    pnl_t = np.array([r["pnl_test"] for r in rows])
    nt = np.array([r["n_trades"] for r in rows])
    ax[2].bar(np.arange(len(hs)), pnl_t, color=["C0","C1","C2","C3","C4"])
    ax[2].set_xticks(range(len(hs))); ax[2].set_xticklabels([f"h={int(h)}" for h in hs])
    ax[2].set_ylabel("test cum PnL")
    ax[2].set_title("D3. Test cum PnL by horizon (DE-on-val threshold)")
    for i, (p, n) in enumerate(zip(pnl_t, nt)):
        ax[2].text(i, p, f"{p:+.2f}\nn={n:,}", ha="center",
                   va="bottom" if p > 0 else "top", fontsize=9)
    ax[2].axhline(0, color="k", ls=":", alpha=0.5)
    ax[2].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(HERE / "fig_D_cost_signal.pdf")
    plt.close()
    print("  -> fig_D_cost_signal.pdf")

    return rows


# ============================================================
# E — decile calibration
# ============================================================

def analysis_E():
    print("\n[E] decile calibration")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for H in HORIZONS:
        r = _avg_preds(H, "yhat_test", "y_test", key_extra=["mp_t_test", "mp_th_test"])
        if r is None: continue
        yhat, y, ext, _ = r
        # quantile bins
        q = np.quantile(yhat, np.linspace(0, 1, 11))
        # ensure strictly increasing
        q[0] -= 1e-12; q[-1] += 1e-12
        bin_idx = np.digitize(yhat, q[1:-1])
        mean_y = np.array([y[bin_idx == k].mean() if (bin_idx == k).sum() > 0 else np.nan for k in range(10)])
        axes[0].plot(np.arange(10) + 1, mean_y * 1e4, "o-", color=HCOLOR[H], lw=2, ms=6, label=f"h={H}")
    axes[0].axhline(0, color="k", ls=":", alpha=0.5)
    axes[0].set_xlabel("yhat decile (1=lowest, 10=highest)")
    axes[0].set_ylabel("mean realized y_h (bp)")
    axes[0].set_title("E1. Decile calibration: mean realized y_h by predicted decile")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # PnL by decile  (long top 2 vs short bottom 2)
    pnl_by_decile = {}
    for H in HORIZONS:
        r = _avg_preds(H, "yhat_test", "y_test", key_extra=["mp_t_test", "mp_th_test"])
        if r is None: continue
        yhat, y, ext, _ = r
        mp_t = ext["mp_t_test"]; mp_th = ext["mp_th_test"]
        q = np.quantile(yhat, np.linspace(0, 1, 11))
        q[0] -= 1e-12; q[-1] += 1e-12
        bin_idx = np.digitize(yhat, q[1:-1])
        # for each decile, dummy long-only on it
        per_dec_pnl = []
        for k in range(10):
            mask = bin_idx == k
            if mask.sum() == 0:
                per_dec_pnl.append(0.0)
                continue
            pred = np.zeros_like(yhat, dtype=np.int8)
            if k >= 8:
                pred[mask] = 1
            elif k <= 1:
                pred[mask] = -1
            pnl = float(pnl_regression(pred, mp_t, mp_th).sum())
            per_dec_pnl.append(pnl)
        pnl_by_decile[H] = per_dec_pnl
        axes[1].plot(np.arange(10) + 1, per_dec_pnl, "o-", color=HCOLOR[H], lw=2, ms=6, label=f"h={H}")
    axes[1].axhline(0, color="k", ls=":", alpha=0.5)
    axes[1].set_xlabel("yhat decile (deciles 1-2 traded SHORT, 9-10 LONG, others neutral)")
    axes[1].set_ylabel("cum PnL (test set)")
    axes[1].set_title("E2. Per-decile PnL contribution by horizon")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(HERE / "fig_E_calibration.pdf")
    plt.close()
    print("  -> fig_E_calibration.pdf")
    return dict(pnl_by_decile=pnl_by_decile)


# ============================================================
# F — h=60-model with √(h/60) threshold scaling
# ============================================================

def analysis_F():
    print("\n[F] h=60 model + √(h/60) threshold scaling")
    # Use h=60 seed-avg model: get yhat_val/test, find best (th_up, th_dn) on val for H=60
    r60_v = _avg_preds(60, "yhat_val", "y_val", key_extra=["mp_t_val", "mp_th_val"])
    r60_t = _avg_preds(60, "yhat_test", "y_test", key_extra=["mp_t_test", "mp_th_test"])
    if r60_v is None or r60_t is None:
        print("  missing h=60 predictions — skipping F")
        return None
    yhat60_v, y60_v, extv60, _ = r60_v
    th_up_60, th_dn_60, pnl_val_60 = best_threshold_de(yhat60_v, extv60["mp_t_val"], extv60["mp_th_val"])
    yhat60_t = r60_t[0]
    pred60_test = apply_threshold(yhat60_t, th_up_60, th_dn_60)
    pnl60_test = float(pnl_regression(pred60_test, r60_t[2]["mp_t_test"], r60_t[2]["mp_th_test"]).sum())

    rows = []
    for H in HORIZONS:
        scale = np.sqrt(H / 60.0)
        th_up_h = th_up_60 * scale
        th_dn_h = th_dn_60 * scale
        # need mp_t / mp_th for THIS horizon (test set)
        # the per-horizon predictions npz already has them
        d = np.load(PRED_DIR / f"pred_h{H}_seed1.npz")  # mp_t/mp_th same across seeds
        mp_t_h = d["mp_t_test"]; mp_th_h = d["mp_th_test"]
        d.close()
        pred_h_scaled = apply_threshold(yhat60_t, th_up_h, th_dn_h)
        pnl_scaled = float(pnl_regression(pred_h_scaled, mp_t_h, mp_th_h).sum())
        n_trades_scaled = int(np.sum(pred_h_scaled != 0))

        # compare with dedicated model
        rh = _avg_preds(H, "yhat_test", "y_test", key_extra=["mp_t_test", "mp_th_test"])
        if rh is None: continue
        yhat_h_t = rh[0]
        # find best (th_up_h_native, th_dn_h_native) on val for THIS horizon
        rv_h = _avg_preds(H, "yhat_val", "y_val", key_extra=["mp_t_val", "mp_th_val"])
        if rv_h is not None:
            th_up_h_native, th_dn_h_native, pnl_val_h = best_threshold_de(
                rv_h[0], rv_h[2]["mp_t_val"], rv_h[2]["mp_th_val"]
            )
            pred_h_native = apply_threshold(yhat_h_t, th_up_h_native, th_dn_h_native)
            pnl_native = float(pnl_regression(pred_h_native, mp_t_h, mp_th_h).sum())
            n_trades_native = int(np.sum(pred_h_native != 0))
        else:
            pnl_native = np.nan; n_trades_native = -1
            th_up_h_native = th_dn_h_native = np.nan

        rows.append(dict(
            h=H, scale_sqrt=float(scale),
            th_up_scaled=float(th_up_h), th_dn_scaled=float(th_dn_h),
            pnl_h60_sqrt_scaled=pnl_scaled, n_trades_sqrt_scaled=n_trades_scaled,
            th_up_native=float(th_up_h_native), th_dn_native=float(th_dn_h_native),
            pnl_dedicated=pnl_native, n_trades_dedicated=n_trades_native,
            advantage_dedicated_minus_h60sqrt=float(pnl_native - pnl_scaled),
        ))
        print(f"  h={H}: h60-√scale={pnl_scaled:+.3f} (n_tr={n_trades_scaled:,}) "
              f"vs dedicated={pnl_native:+.3f} (n_tr={n_trades_native:,}) "
              f"Δ={pnl_native-pnl_scaled:+.3f}")

    # Figure F
    hs = np.array(HORIZONS, dtype=float)
    pnl_s = np.array([r["pnl_h60_sqrt_scaled"] for r in rows])
    pnl_d = np.array([r["pnl_dedicated"] for r in rows])
    width = 0.4
    x = np.arange(len(hs))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].bar(x - width / 2, pnl_s, width, color="C0", label="h=60 model + √(h/60) thresh")
    ax[0].bar(x + width / 2, pnl_d, width, color="C3", label="dedicated h-model (DE on val)")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"h={int(h)}" for h in hs])
    ax[0].set_ylabel("test cum PnL")
    ax[0].set_title("F1. Single h=60 model + √-scaled threshold vs per-h dedicated training")
    ax[0].axhline(0, color="k", ls=":", alpha=0.5)
    ax[0].legend(); ax[0].grid(True, alpha=0.3, axis="y")
    for i, (a, b) in enumerate(zip(pnl_s, pnl_d)):
        ax[0].text(i - width / 2, a, f"{a:+.2f}", ha="center", va="bottom" if a > 0 else "top", fontsize=9)
        ax[0].text(i + width / 2, b, f"{b:+.2f}", ha="center", va="bottom" if b > 0 else "top", fontsize=9)

    diff = pnl_d - pnl_s
    ax[1].bar(x, diff, color=["C2" if v >= 0 else "C3" for v in diff])
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"h={int(h)}" for h in hs])
    ax[1].set_ylabel("PnL advantage of dedicated model")
    ax[1].set_title("F2. Dedicated − h60-√scaled  (positive ⇒ dedicated wins)")
    ax[1].axhline(0, color="k", ls=":", alpha=0.5)
    ax[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(diff):
        ax[1].text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v > 0 else "top", fontsize=10)
    plt.tight_layout()
    plt.savefig(HERE / "fig_F_threshold_scaling.pdf")
    plt.close()
    print("  -> fig_F_threshold_scaling.pdf")

    return dict(rows=rows, th_up_60=th_up_60, th_dn_60=th_dn_60, pnl_h60_test=pnl60_test)


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    out = {}
    out["A"] = analysis_A()
    out["B"] = analysis_B()
    out["C"] = analysis_C()
    out["D"] = analysis_D()
    out["E"] = analysis_E()
    out["F"] = analysis_F()
    print(f"\n[done] {time.time()-t0:.1f}s")
    with open(HERE / "metrics_by_horizon.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("  -> metrics_by_horizon.json")


if __name__ == "__main__":
    main()
