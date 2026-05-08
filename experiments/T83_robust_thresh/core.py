"""T83 — core utilities for robust threshold strategies.

Loads the 5-seed-averaged regression Δmid predictions from T75 and provides:
  - PnL evaluation (vectorized, asymmetric / symmetric EV gate)
  - Train/eval splits (date-based, random, sym leave-one-out, K-fold)
  - Strategy fitters (symmetric k, DE asym, bootstrap median, K-fold CV-DE,
    conformal quantile, regularized DE)
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

HERE = os.path.dirname(os.path.abspath(__file__))
T75 = os.path.abspath(os.path.join(HERE, "..", "T75_regression_dmid"))
SEEDS = (1, 7, 13, 42, 100)
SYMS = (0, 1, 2, 3, 4)
FEE = 0.0001
ITER013_FULL_PNL = 36.2281    # DE asym on full 442k → LOSO-equiv per_sym sum


def load_pred_avg() -> pd.DataFrame:
    base = pd.read_parquet(os.path.join(T75, f"pred_T75_seed{SEEDS[0]}.parquet"))
    p_sum = base["pred_dmid_norm"].to_numpy(np.float64).copy()
    n = len(base)
    for s in SEEDS[1:]:
        df_s = pd.read_parquet(os.path.join(T75, f"pred_T75_seed{s}.parquet"))
        if len(df_s) != n:
            raise RuntimeError(f"row mismatch seed={s}")
        p_sum += df_s["pred_dmid_norm"].to_numpy(np.float64)
    p_avg = p_sum / float(len(SEEDS))
    base = base.copy()
    base["pred"] = p_avg.astype(np.float64)
    base["dmid_th_minus_t"] = base["midprice_th"].astype(np.float64) - base["midprice_t"].astype(np.float64)
    base["mp_t"] = base["midprice_t"].astype(np.float64)
    base["mp_th"] = base["midprice_th"].astype(np.float64)
    return base[["sym", "date", "session", "t", "true_label", "true_dmid_norm", "pred", "mp_t", "mp_th"]]


def vectorized_pnl(actions: np.ndarray, mp_t: np.ndarray, mp_th: np.ndarray) -> np.ndarray:
    """Per-row PnL contribution under the platform formula."""
    side = actions.astype(np.float64) - 1.0
    abs_side = np.abs(side)
    diff = mp_th - mp_t
    fee_pnl = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0
    return (side * diff - fee_pnl) / denom


def sym_sum(df: pd.DataFrame, actions: np.ndarray) -> float:
    """LOSO-equiv: per-sym cum_pnl summed (matches platform's per-sym normalization).

    Each sym contributes the sum of its row PnLs; we then sum across syms.
    Equal to total PnL if no per-sym weighting; we keep separated for diagnostics.
    """
    pnl = vectorized_pnl(actions, df["mp_t"].to_numpy(), df["mp_th"].to_numpy())
    df_pnl = pd.DataFrame({"sym": df["sym"].to_numpy(), "pnl": pnl})
    per = df_pnl.groupby("sym")["pnl"].sum()
    return float(per.sum())


def per_sym_dict(df: pd.DataFrame, actions: np.ndarray) -> dict:
    pnl = vectorized_pnl(actions, df["mp_t"].to_numpy(), df["mp_th"].to_numpy())
    df_pnl = pd.DataFrame({"sym": df["sym"].to_numpy(), "pnl": pnl})
    per = df_pnl.groupby("sym")["pnl"].sum().to_dict()
    return {int(k): float(v) for k, v in per.items()}


def gate_asym(pred: np.ndarray, thr_up: float, thr_dn: float) -> np.ndarray:
    """thr_up, thr_dn POSITIVE; down cutoff is -thr_dn."""
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr_up] = 2
    a[pred < -thr_dn] = 0
    return a


def gate_sym(pred: np.ndarray, thr: float) -> np.ndarray:
    a = np.full(len(pred), 1, dtype=np.int8)
    a[pred > thr] = 2
    a[pred < -thr] = 0
    return a


# -----------------------------------------------------------------------------
# Strategy fitters: each returns a dict {"thr_up": float, "thr_dn": float, ...}
# -----------------------------------------------------------------------------

def fit_symmetric_grid(df_train: pd.DataFrame, k_grid=None) -> dict:
    """Grid search symmetric k ∈ k_grid (default 0.25..3.5 in 0.05 steps)."""
    if k_grid is None:
        k_grid = np.arange(0.25, 3.55, 0.05)
    pred = df_train["pred"].to_numpy()
    best_k = None
    best_pnl = -1e18
    for k in k_grid:
        thr = k * 2 * FEE
        a = gate_sym(pred, thr)
        pnl = sym_sum(df_train, a)
        if pnl > best_pnl:
            best_pnl = pnl
            best_k = float(k)
    thr = best_k * 2 * FEE
    return {"strategy": "sym_grid", "k": best_k, "thr_up": thr, "thr_dn": thr, "train_pnl": best_pnl}


def fit_de_asym(df_train: pd.DataFrame, bounds=((1e-5, 8e-4), (1e-5, 8e-4)),
                de_seeds=(0, 1, 2, 7, 42), maxiter=80, popsize=24) -> dict:
    """DE asym: tune (thr_up, thr_dn) to maximize sym_sum on train."""
    pred = df_train["pred"].to_numpy()
    mp_t = df_train["mp_t"].to_numpy()
    mp_th = df_train["mp_th"].to_numpy()
    sym_arr = df_train["sym"].to_numpy()

    def obj(x):
        thr_up, thr_dn = x
        a = gate_asym(pred, thr_up, thr_dn)
        side = a.astype(np.float64) - 1.0
        abs_side = np.abs(side)
        fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
        pnl_row = (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)
        # sum per sym then sum
        s = 0.0
        for k in SYMS:
            mask = sym_arr == k
            if mask.any():
                s += pnl_row[mask].sum()
        return -float(s)

    best = None
    runs = []
    for sd in de_seeds:
        result = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        rec = {"de_seed": int(sd), "thr_up": float(result.x[0]), "thr_dn": float(result.x[1]),
               "train_pnl": float(-result.fun)}
        runs.append(rec)
        if best is None or rec["train_pnl"] > best["train_pnl"]:
            best = rec
    return {"strategy": "de_asym", "thr_up": best["thr_up"], "thr_dn": best["thr_dn"],
            "train_pnl": best["train_pnl"], "de_runs": runs}


def fit_de_asym_kfold(df_train: pd.DataFrame, n_folds: int = 5, fold_axis: str = "date",
                      bounds=((1e-5, 8e-4), (1e-5, 8e-4)), de_seed=0,
                      maxiter=60, popsize=20) -> dict:
    """K-fold CV-DE: split df_train into K folds; on each fold fit DE on the OTHER
    K-1 folds, store the threshold; final thr = MEDIAN across folds.

    fold_axis = "date" → split by date; "row" → random row split.
    """
    if fold_axis == "date":
        dates = sorted(df_train["date"].unique())
        # equal-sized contiguous date folds
        chunks = np.array_split(dates, n_folds)
        fold_assignments = []
        for i, ch in enumerate(chunks):
            mask = df_train["date"].isin(ch.tolist()).to_numpy()
            fold_assignments.append(mask)
    else:
        rng = np.random.default_rng(0)
        n = len(df_train)
        idx = rng.permutation(n)
        chunks = np.array_split(idx, n_folds)
        fold_assignments = []
        for ch in chunks:
            mask = np.zeros(n, dtype=bool)
            mask[ch] = True
            fold_assignments.append(mask)

    fold_thrs = []
    for i, mask in enumerate(fold_assignments):
        # fit DE on rows NOT in fold i
        df_fit = df_train[~mask]
        rec = fit_de_asym(df_fit, bounds=bounds, de_seeds=(de_seed,), maxiter=maxiter, popsize=popsize)
        fold_thrs.append((rec["thr_up"], rec["thr_dn"]))

    arr = np.array(fold_thrs)
    thr_up_med = float(np.median(arr[:, 0]))
    thr_dn_med = float(np.median(arr[:, 1]))
    a = gate_asym(df_train["pred"].to_numpy(), thr_up_med, thr_dn_med)
    train_pnl = sym_sum(df_train, a)
    return {"strategy": f"kfold_de_{n_folds}_{fold_axis}",
            "thr_up": thr_up_med, "thr_dn": thr_dn_med,
            "train_pnl": train_pnl, "fold_thrs": fold_thrs}


def fit_bootstrap_de(df_train: pd.DataFrame, n_boot: int = 50, frac: float = 0.7,
                     bounds=((1e-5, 8e-4), (1e-5, 8e-4)), seed: int = 42,
                     maxiter=50, popsize=18) -> dict:
    """Bootstrap median DE: B=n_boot bootstrap samples (frac of rows w/ replacement),
    DE on each, return median thr."""
    rng = np.random.default_rng(seed)
    n = len(df_train)
    boot_thrs = []
    for b in range(n_boot):
        idx = rng.integers(0, n, int(frac * n))
        df_b = df_train.iloc[idx].reset_index(drop=True)
        rec = fit_de_asym(df_b, bounds=bounds, de_seeds=(b,), maxiter=maxiter, popsize=popsize)
        boot_thrs.append((rec["thr_up"], rec["thr_dn"]))
    arr = np.array(boot_thrs)
    thr_up_med = float(np.median(arr[:, 0]))
    thr_dn_med = float(np.median(arr[:, 1]))
    a = gate_asym(df_train["pred"].to_numpy(), thr_up_med, thr_dn_med)
    train_pnl = sym_sum(df_train, a)
    return {"strategy": f"bootstrap_de_n{n_boot}_f{frac}",
            "thr_up": thr_up_med, "thr_dn": thr_dn_med,
            "train_pnl": train_pnl,
            "thr_up_q25": float(np.quantile(arr[:, 0], 0.25)),
            "thr_up_q75": float(np.quantile(arr[:, 0], 0.75)),
            "thr_dn_q25": float(np.quantile(arr[:, 1], 0.25)),
            "thr_dn_q75": float(np.quantile(arr[:, 1], 0.75))}


def fit_de_regularized(df_train: pd.DataFrame, lam: float = 1.0,
                       bounds=((1e-5, 8e-4), (1e-5, 8e-4)),
                       de_seeds=(0, 1, 2, 7, 42), maxiter=60, popsize=20) -> dict:
    """DE asym with a symmetry penalty: maximize PnL - lam * (thr_up - thr_dn)^2 / FEE^2.

    Discourages the DE from drifting toward extreme asymmetry that overfits a
    sample-specific bias.
    """
    pred = df_train["pred"].to_numpy()
    mp_t = df_train["mp_t"].to_numpy()
    mp_th = df_train["mp_th"].to_numpy()
    sym_arr = df_train["sym"].to_numpy()

    def obj(x):
        thr_up, thr_dn = x
        a = gate_asym(pred, thr_up, thr_dn)
        side = a.astype(np.float64) - 1.0
        abs_side = np.abs(side)
        fee = FEE * abs_side * np.abs((mp_th + 1.0) + (mp_t + 1.0))
        pnl_row = (side * (mp_th - mp_t) - fee) / (mp_t + 1.0)
        s = 0.0
        for k in SYMS:
            mask = sym_arr == k
            if mask.any():
                s += pnl_row[mask].sum()
        # symmetry penalty in units of FEE
        pen = lam * ((thr_up - thr_dn) / FEE) ** 2
        return -float(s) + pen

    best = None
    for sd in de_seeds:
        result = differential_evolution(
            obj, bounds=bounds, seed=sd, maxiter=maxiter, popsize=popsize,
            polish=True, tol=1e-7, mutation=(0.5, 1.0), recombination=0.7,
            updating="deferred", workers=1, init="sobol",
        )
        rec = {"de_seed": int(sd), "thr_up": float(result.x[0]), "thr_dn": float(result.x[1]),
               "train_obj": float(-result.fun)}
        # actual PnL (no penalty)
        a = gate_asym(pred, rec["thr_up"], rec["thr_dn"])
        rec["train_pnl"] = sym_sum(df_train, a)
        if best is None or rec["train_pnl"] > best["train_pnl"]:
            best = rec
    return {"strategy": f"de_reg_lam{lam}", "thr_up": best["thr_up"], "thr_dn": best["thr_dn"],
            "train_pnl": best["train_pnl"]}


def eval_strategy(df_eval: pd.DataFrame, thr_up: float, thr_dn: float) -> dict:
    a = gate_asym(df_eval["pred"].to_numpy(), thr_up, thr_dn)
    pnl = sym_sum(df_eval, a)
    per = per_sym_dict(df_eval, a)
    return {"eval_pnl": pnl, "per_sym": per, "n_active": int((a != 1).sum())}


# -----------------------------------------------------------------------------
# Splits
# -----------------------------------------------------------------------------

def split_by_date(df: pd.DataFrame, train_dates, eval_dates):
    train = df[df["date"].isin(train_dates)].reset_index(drop=True)
    eval_ = df[df["date"].isin(eval_dates)].reset_index(drop=True)
    return train, eval_


def split_random(df: pd.DataFrame, frac_train: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.permutation(n)
    cut = int(frac_train * n)
    train = df.iloc[idx[:cut]].reset_index(drop=True)
    eval_ = df.iloc[idx[cut:]].reset_index(drop=True)
    return train, eval_
