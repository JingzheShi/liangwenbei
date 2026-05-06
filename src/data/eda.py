"""EDA：扫描全量 1200 个 parquet，生成完整性检查、分布统计和"瞎猜"基线 PnL 报告。

输出：
  analysis/eda_report.md             — 主报告（中文）
  analysis/figures/*.png             — 配图
  analysis/eda_stats.json            — 机器可读的关键数字（供后续 worker 引用）

运行：
  cd /root/projects/liangwenbei_workdir && python src/data/eda.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, time as dtime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.split import N_DATES, N_SYMS, SESSIONS, get_file_path

# matplotlib 在无显示环境
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt

# 配中文字体：手动 addfont 系统字体（matplotlib 默认不会扫 .ttc）
for _p in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
):
    if os.path.exists(_p):
        try:
            _fm.fontManager.addfont(_p)
        except Exception:
            pass
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "AR PL UMing CN", "DejaVu Sans"]
matplotlib.rcParams["font.serif"] = ["Noto Serif CJK JP", "AR PL UMing CN", "DejaVu Serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ------------------------------ 配置 ------------------------------------------

DATA_DIR = "data"
OUT_DIR = "analysis"
FIG_DIR = os.path.join(OUT_DIR, "figures")
EXPECTED_ROWS = 2001
HORIZONS = (5, 10, 20, 40, 60)
LABEL_COLS = tuple(f"label_{h}" for h in HORIZONS)
ALPHA_BY_HORIZON = {5: 5e-4, 10: 5e-4, 20: 1e-3, 40: 1e-3, 60: 1e-3}
FEE_RATE = 1e-4  # 0.01% 双边


def _expected_times(session: str) -> np.ndarray:
    """返回 session 内的 2001 个期望时间戳 (HH:MM:SS) 的秒数。"""
    if session == "am":
        start = 9 * 3600 + 40 * 60
    elif session == "pm":
        start = 13 * 3600 + 10 * 60
    else:
        raise ValueError(session)
    return np.arange(EXPECTED_ROWS, dtype=np.int64) * 3 + start


def _time_to_seconds(t) -> int:
    if isinstance(t, dtime):
        return t.hour * 3600 + t.minute * 60 + t.second
    if isinstance(t, str):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    if isinstance(t, pd.Timestamp):
        return t.hour * 3600 + t.minute * 60 + t.second
    raise TypeError(f"unknown time type: {type(t)}")


# ------------------------------ 单文件扫描 -------------------------------------

class FileStat:
    __slots__ = (
        "key", "rows", "n_nan_total", "nan_by_col", "n_inf_total",
        "midprice", "labels", "amount_delta",
        "time_violations", "midprice_zero_count", "midprice_nan_count",
        "elapsed",
    )

    def __init__(self, key, rows):
        self.key = key  # (sym, date, session)
        self.rows = rows
        self.n_nan_total = 0
        self.nan_by_col: Dict[str, int] = {}
        self.n_inf_total = 0
        self.midprice = None  # np.ndarray
        self.labels = None    # np.ndarray (T, 5) int
        self.amount_delta = None  # np.ndarray
        self.time_violations = 0
        self.midprice_zero_count = 0
        self.midprice_nan_count = 0
        self.elapsed = 0.0


def scan_one(sym: int, date: int, session: str) -> FileStat:
    t0 = time.time()
    path = get_file_path(sym, date, session, data_dir=DATA_DIR)
    fs = FileStat((sym, date, session), 0)
    if not os.path.exists(path):
        fs.elapsed = time.time() - t0
        return fs
    df = pd.read_parquet(path)
    fs.rows = len(df)

    # NaN by column (numeric only — 也含 bool)
    num = df.select_dtypes(include=[np.number])
    nan_per = num.isna().sum()
    fs.n_nan_total = int(nan_per.sum())
    if fs.n_nan_total > 0:
        fs.nan_by_col = {c: int(v) for c, v in nan_per.items() if v > 0}

    # Inf
    inf_mask = np.isinf(num.to_numpy())
    fs.n_inf_total = int(inf_mask.sum())

    # midprice 异常
    mp = df["midprice"].to_numpy(dtype=np.float64)
    fs.midprice_nan_count = int(np.isnan(mp).sum())
    fs.midprice_zero_count = int(((mp + 1.0) == 0.0).sum())  # 价格归零意味着 mid=-1

    # time 完整性
    expected = _expected_times(session)
    actual_secs = np.array([_time_to_seconds(t) for t in df["time"].iloc[:len(df)]], dtype=np.int64)
    if len(actual_secs) == len(expected):
        fs.time_violations = int((actual_secs != expected).sum())
    else:
        fs.time_violations = -1  # 长度不对

    # 收集标签 / midprice / amount_delta（用于全局统计）
    fs.midprice = mp.astype(np.float32)
    fs.labels = df[list(LABEL_COLS)].to_numpy(dtype=np.int8)
    fs.amount_delta = df["amount_delta"].to_numpy(dtype=np.float64)

    fs.elapsed = time.time() - t0
    return fs


# ------------------------------ 主流程 ----------------------------------------

def collect_all() -> List[FileStat]:
    keys = [(sym, d, s) for sym in range(N_SYMS) for d in range(N_DATES) for s in SESSIONS]
    stats = []
    t0 = time.time()
    for i, (sym, d, s) in enumerate(keys):
        fs = scan_one(sym, d, s)
        stats.append(fs)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  scanned {i+1}/{len(keys)} ({elapsed:.1f}s)")
    print(f"  done. total {len(keys)} files, elapsed {time.time()-t0:.1f}s")
    return stats


# ------------------------------ 完整性 ----------------------------------------

def integrity_summary(stats: List[FileStat]) -> dict:
    keys_seen = {fs.key for fs in stats if fs.rows > 0}
    expected = {(sym, d, s) for sym in range(N_SYMS) for d in range(N_DATES) for s in SESSIONS}
    missing = sorted(expected - keys_seen)
    rows_violations = [fs.key for fs in stats if fs.rows not in (0, EXPECTED_ROWS)]
    nan_files = [(fs.key, fs.n_nan_total, fs.nan_by_col) for fs in stats if fs.n_nan_total > 0]
    inf_files = [(fs.key, fs.n_inf_total) for fs in stats if fs.n_inf_total > 0]
    time_violations = [(fs.key, fs.time_violations) for fs in stats if fs.time_violations not in (0, None)]
    mid_zero = [(fs.key, fs.midprice_zero_count) for fs in stats if fs.midprice_zero_count > 0]
    mid_nan = [(fs.key, fs.midprice_nan_count) for fs in stats if fs.midprice_nan_count > 0]

    summary = {
        "n_files_seen": len(keys_seen),
        "n_files_expected": 1200,
        "missing_files": [list(k) for k in missing],
        "row_count_violations": [(list(fs.key), int(fs.rows)) for fs in stats if fs.rows not in (0, EXPECTED_ROWS)][:50],
        "n_files_with_nan": len(nan_files),
        "n_files_with_inf": len(inf_files),
        "n_files_with_time_violation": len(time_violations),
        "n_files_with_midprice_zero": len(mid_zero),
        "n_files_with_midprice_nan": len(mid_nan),
        "total_rows": int(sum(fs.rows for fs in stats)),
    }
    if nan_files:
        # 保留所有，但只汇总每个文件的 (key, total_nan, n_cols_affected, only_ask_side)
        summary["nan_file_summary"] = []
        for k, n, by in nan_files:
            cols = list(by.keys())
            ask_only = all(
                c.startswith("ask") or c.startswith("spread") or c.startswith("asize")
                for c in cols
            )
            summary["nan_file_summary"].append({
                "key": list(k),
                "total_nan": int(n),
                "n_cols_affected": len(cols),
                "ask_side_only": bool(ask_only),
                "max_nan_in_one_col": int(max(by.values())) if by else 0,
            })
    if time_violations:
        summary["sample_time_violations"] = [
            (list(k), n) for k, n in time_violations[:5]
        ]
    if mid_zero:
        summary["sample_midprice_zero"] = [
            (list(k), n) for k, n in mid_zero[:5]
        ]
    return summary


# ------------------------------ 分布统计 --------------------------------------

def distribution_stats(stats: List[FileStat]) -> dict:
    by_sym_mp = defaultdict(list)  # sym -> list of (midprice values)
    by_sym_amt = defaultdict(list)
    label_count = np.zeros((N_SYMS, len(LABEL_COLS), 3), dtype=np.int64)  # (sym, label_idx, class)
    label_count_total = np.zeros((len(LABEL_COLS), 3), dtype=np.int64)

    for fs in stats:
        if fs.midprice is None:
            continue
        sym = fs.key[0]
        by_sym_mp[sym].append(fs.midprice)
        by_sym_amt[sym].append(fs.amount_delta)
        # label distribution: fs.labels shape (T, 5)
        for li in range(len(LABEL_COLS)):
            for cls in range(3):
                c = int((fs.labels[:, li] == cls).sum())
                label_count[sym, li, cls] += c
                label_count_total[li, cls] += c

    midprice_stats = {}
    amount_stats = {}
    for sym in range(N_SYMS):
        mp = np.concatenate(by_sym_mp[sym]) if by_sym_mp[sym] else np.array([])
        amt = np.concatenate(by_sym_amt[sym]) if by_sym_amt[sym] else np.array([])
        midprice_stats[sym] = {
            "min": float(mp.min()), "max": float(mp.max()),
            "mean": float(mp.mean()), "std": float(mp.std()),
            "q01": float(np.quantile(mp, 0.01)),
            "q50": float(np.quantile(mp, 0.5)),
            "q99": float(np.quantile(mp, 0.99)),
        }
        amount_stats[sym] = {
            "min": float(amt.min()), "max": float(amt.max()),
            "mean": float(amt.mean()), "std": float(amt.std()),
            "q01": float(np.quantile(amt, 0.01)),
            "q50": float(np.quantile(amt, 0.5)),
            "q99": float(np.quantile(amt, 0.99)),
        }

    label_dist = {
        "by_sym_horizon": {},
        "by_horizon": {},
    }
    for sym in range(N_SYMS):
        for li, lc in enumerate(LABEL_COLS):
            tot = int(label_count[sym, li].sum())
            label_dist["by_sym_horizon"][f"sym{sym}_{lc}"] = {
                "0_down": int(label_count[sym, li, 0]),
                "1_flat": int(label_count[sym, li, 1]),
                "2_up": int(label_count[sym, li, 2]),
                "p_down": float(label_count[sym, li, 0]) / tot if tot else None,
                "p_flat": float(label_count[sym, li, 1]) / tot if tot else None,
                "p_up": float(label_count[sym, li, 2]) / tot if tot else None,
            }
    for li, lc in enumerate(LABEL_COLS):
        tot = int(label_count_total[li].sum())
        label_dist["by_horizon"][lc] = {
            "0_down": int(label_count_total[li, 0]),
            "1_flat": int(label_count_total[li, 1]),
            "2_up": int(label_count_total[li, 2]),
            "p_down": float(label_count_total[li, 0]) / tot if tot else None,
            "p_flat": float(label_count_total[li, 1]) / tot if tot else None,
            "p_up": float(label_count_total[li, 2]) / tot if tot else None,
        }
    return {"midprice_by_sym": midprice_stats, "amount_delta_by_sym": amount_stats, "label_dist": label_dist, "_label_count_by_sym": label_count, "_label_count_total": label_count_total}


# ------------------------------ 单 tick 跳变 -----------------------------------

def midprice_diff_distribution(stats: List[FileStat]) -> dict:
    """每 session 内 midprice 的 1-tick diff 直方分布。"""
    diffs = []
    pct_unchanged_per_session = []
    for fs in stats:
        if fs.midprice is None:
            continue
        d = np.diff(fs.midprice)
        diffs.append(d)
        pct_unchanged_per_session.append(float((d == 0).mean()))
    diffs = np.concatenate(diffs)
    return {
        "n_diffs": int(diffs.size),
        "pct_zero": float((diffs == 0).mean()),
        "pct_zero_sess_mean": float(np.mean(pct_unchanged_per_session)),
        "pct_zero_sess_min": float(np.min(pct_unchanged_per_session)),
        "pct_zero_sess_max": float(np.max(pct_unchanged_per_session)),
        "abs_quantiles": {
            "q50": float(np.quantile(np.abs(diffs), 0.5)),
            "q90": float(np.quantile(np.abs(diffs), 0.9)),
            "q99": float(np.quantile(np.abs(diffs), 0.99)),
            "max": float(np.abs(diffs).max()),
        },
        "_diffs_for_plot": diffs,
    }


# ------------------------------ 随机基线 PnL -----------------------------------

def random_baseline_pnl(stats: List[FileStat]) -> dict:
    """计算"瞎猜涨跌"和"瞎猜涨跌平"两种策略在每个 horizon 的期望 per-trade PnL 和累计 PnL.

    PnL 公式（评分用）：
      pnl = [ (a-1)·(mid_tn - mid_t) − fee·|a-1|·((mid_tn+1)+(mid_t+1)) ] / (mid_t + 1)

    其中 a 是预测动作 (0/1/2)。我们对每个 (t, horizon) 处理：
      策略 R2 = uniform {0, 2}：1 笔交易，期望 pnl = E_a[pnl(a)]
      策略 R3 = uniform {0, 1, 2}：1/3 概率 skip
      策略 always_up: a=2
      策略 always_down: a=0
    """
    out = {}
    # 先把所有 session 的 midprice 拼成一个数组，再按每个 horizon 取 (mid_t, mid_tn) 对
    per_horizon_results = {}
    for h in HORIZONS:
        per_horizon_results[h] = {
            "n_pairs": 0,
            "sum_dp": 0.0,
            "sum_avg_price_plus": 0.0,  # ((mid_t+1) + (mid_tn+1))
            "sum_pnl_up": 0.0,
            "sum_pnl_down": 0.0,
            "sum_inv_p1": 0.0,
        }
    for fs in stats:
        if fs.midprice is None:
            continue
        mp = fs.midprice.astype(np.float64)
        T = mp.size
        for h in HORIZONS:
            if T <= h:
                continue
            mid_t = mp[: T - h]
            mid_tn = mp[h:]
            # 假设 mid_t + 1 不为 0（已检查过）
            denom = mid_t + 1.0
            dp = mid_tn - mid_t
            fee = FEE_RATE * ((mid_t + 1.0) + (mid_tn + 1.0))
            pnl_up = (dp - fee) / denom        # 预测 2（涨）
            pnl_down = (-dp - fee) / denom     # 预测 0（跌）
            agg = per_horizon_results[h]
            agg["n_pairs"] += int(mid_t.size)
            agg["sum_dp"] += float(dp.sum())
            agg["sum_avg_price_plus"] += float(((mid_t + 1.0) + (mid_tn + 1.0)).sum())
            agg["sum_pnl_up"] += float(pnl_up.sum())
            agg["sum_pnl_down"] += float(pnl_down.sum())
            agg["sum_inv_p1"] += float((1.0 / denom).sum())

    for h in HORIZONS:
        agg = per_horizon_results[h]
        n = agg["n_pairs"]
        if n == 0:
            continue
        # 策略 R2: uniform {0, 2} → 平均 (pnl_up + pnl_down) / 2，每个 t 都交易
        sum_pnl_R2 = 0.5 * (agg["sum_pnl_up"] + agg["sum_pnl_down"])
        # 策略 R3: 1/3 概率 skip → 每 3 个 t 期望交易 2 次，平均 pnl
        sum_pnl_R3 = (1.0 / 3.0) * (agg["sum_pnl_up"] + agg["sum_pnl_down"])  # 等价
        out[h] = {
            "n_pairs": n,
            "always_up_total": agg["sum_pnl_up"],
            "always_up_per_trade": agg["sum_pnl_up"] / n,
            "always_down_total": agg["sum_pnl_down"],
            "always_down_per_trade": agg["sum_pnl_down"] / n,
            "rand_2cls_total": sum_pnl_R2,
            "rand_2cls_per_trade": sum_pnl_R2 / n,  # 每个 t 都交易
            "rand_3cls_total": sum_pnl_R3,
            "rand_3cls_per_trade": sum_pnl_R3 / (2 * n / 3),  # 仅 2/3 t 真正交易
            "fee_avg_per_trade": FEE_RATE * agg["sum_avg_price_plus"] / n,
            "abs_dp_mean": abs(agg["sum_dp"]) / n,
        }
    return out


# ------------------------------ 绘图 ------------------------------------------

def plot_label_distribution(label_dist: dict, fig_path: str) -> None:
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(15, 3), sharey=True)
    for ax, lc in zip(axes, LABEL_COLS):
        d = label_dist["by_horizon"][lc]
        bars = [d["p_down"], d["p_flat"], d["p_up"]]
        ax.bar(["跌", "平", "涨"], bars, color=["#d62728", "#7f7f7f", "#2ca02c"])
        ax.set_title(lc)
        ax.set_ylim(0, 1)
        for i, v in enumerate(bars):
            ax.text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=8)
    fig.suptitle("各 horizon 标签分布（全数据）")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def plot_label_by_sym(label_count_by_sym: np.ndarray, fig_path: str) -> None:
    """label_count_by_sym shape (5_sym, 5_horizon, 3_class)."""
    fig, axes = plt.subplots(N_SYMS, 1, figsize=(10, 8), sharex=True)
    width = 0.25
    x = np.arange(len(HORIZONS))
    for sym in range(N_SYMS):
        ax = axes[sym]
        counts = label_count_by_sym[sym]  # (5_horizon, 3)
        totals = counts.sum(axis=1, keepdims=True)
        probs = counts / totals
        ax.bar(x - width, probs[:, 0], width, color="#d62728", label="跌")
        ax.bar(x, probs[:, 1], width, color="#7f7f7f", label="平")
        ax.bar(x + width, probs[:, 2], width, color="#2ca02c", label="涨")
        ax.set_ylabel(f"sym{sym}")
        ax.set_ylim(0, 1)
        if sym == 0:
            ax.legend(loc="upper right", ncol=3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(LABEL_COLS)
    axes[-1].set_xlabel("horizon")
    fig.suptitle("各股票 × horizon 的标签分布")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def plot_midprice_diff_hist(diffs: np.ndarray, fig_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 左：包含零，对数 y 轴
    bins = 81
    rng = np.quantile(np.abs(diffs), 0.99)
    axes[0].hist(diffs, bins=bins, range=(-rng, rng), color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_title(f"midprice 单 tick 跳变直方图 (中央 99% 区间, n={diffs.size:,})")
    axes[0].set_xlabel("Δmidprice (相对昨收)")
    axes[0].set_ylabel("count (log)")
    # 右：仅非零跳变（绝对值）
    nz = np.abs(diffs[diffs != 0])
    if nz.size:
        axes[1].hist(np.log10(nz + 1e-12), bins=60, color="#ff7f0e")
        axes[1].set_title(f"非零跳变绝对值 log10 分布 (n={nz.size:,})")
        axes[1].set_xlabel("log10|Δmidprice|")
        axes[1].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def plot_baseline_pnl(rand_pnl: dict, fig_path: str) -> None:
    horizons = list(rand_pnl.keys())
    R2 = [rand_pnl[h]["rand_2cls_per_trade"] for h in horizons]
    AU = [rand_pnl[h]["always_up_per_trade"] for h in horizons]
    AD = [rand_pnl[h]["always_down_per_trade"] for h in horizons]
    fee = [rand_pnl[h]["fee_avg_per_trade"] for h in horizons]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(horizons))
    ax.bar(x - 0.3, R2, 0.2, label="随机 {跌,涨} 每笔 PnL")
    ax.bar(x - 0.1, AU, 0.2, label="总是预测涨 每笔 PnL")
    ax.bar(x + 0.1, AD, 0.2, label="总是预测跌 每笔 PnL")
    ax.bar(x + 0.3, [-f for f in fee], 0.2, label="单笔手续费 (负)", color="#aaaaaa")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"label_{h}" for h in horizons])
    ax.set_ylabel("per-trade PnL（相对昨收的比例）")
    ax.set_title("随机/朴素 baseline 每笔 PnL（含 0.01% 双边手续费）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def plot_amount_delta_by_sym(amount_stats: dict, fig_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    syms = list(amount_stats.keys())
    means = [amount_stats[s]["mean"] for s in syms]
    q99 = [amount_stats[s]["q99"] for s in syms]
    q01 = [amount_stats[s]["q01"] for s in syms]
    x = np.arange(len(syms))
    ax.bar(x, means, 0.5, label="均值")
    ax.scatter(x, q99, color="red", label="q99", zorder=3)
    ax.scatter(x, q01, color="blue", label="q01", zorder=3)
    ax.set_yscale("symlog")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sym{s}" for s in syms])
    ax.set_ylabel("amount_delta (元，symlog)")
    ax.set_title("各股票 amount_delta 量级")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


def plot_midprice_by_sym(midprice_stats: dict, fig_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    syms = list(midprice_stats.keys())
    x = np.arange(len(syms))
    means = [midprice_stats[s]["mean"] for s in syms]
    q01 = [midprice_stats[s]["q01"] for s in syms]
    q50 = [midprice_stats[s]["q50"] for s in syms]
    q99 = [midprice_stats[s]["q99"] for s in syms]
    ax.errorbar(x, q50, yerr=[
        np.array(q50) - np.array(q01),
        np.array(q99) - np.array(q50),
    ], fmt="o", capsize=5, label="q01–q50–q99")
    ax.scatter(x, means, marker="x", color="red", label="均值")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"sym{s}" for s in syms])
    ax.set_ylabel("midprice (相对昨收涨跌幅)")
    ax.set_title("各股票 midprice 分布概览")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)


# ------------------------------ 主入口 ----------------------------------------

def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    print("[eda] 1/5 扫描全量 parquet ...")
    stats = collect_all()
    print("[eda] 2/5 完整性汇总 ...")
    integ = integrity_summary(stats)
    print("[eda] 3/5 分布统计 ...")
    dist = distribution_stats(stats)
    print("[eda] 4/5 单 tick 跳变 + 随机基线 PnL ...")
    diff_info = midprice_diff_distribution(stats)
    rand_pnl = random_baseline_pnl(stats)

    print("[eda] 5/5 画图 + 写报告 ...")
    plot_label_distribution(dist["label_dist"], os.path.join(FIG_DIR, "label_distribution_by_horizon.png"))
    plot_label_by_sym(dist["_label_count_by_sym"], os.path.join(FIG_DIR, "label_distribution_by_sym.png"))
    plot_midprice_diff_hist(diff_info["_diffs_for_plot"], os.path.join(FIG_DIR, "midprice_tick_diff.png"))
    plot_baseline_pnl(rand_pnl, os.path.join(FIG_DIR, "random_baseline_pnl.png"))
    plot_amount_delta_by_sym(dist["amount_delta_by_sym"], os.path.join(FIG_DIR, "amount_delta_by_sym.png"))
    plot_midprice_by_sym(dist["midprice_by_sym"], os.path.join(FIG_DIR, "midprice_by_sym.png"))

    # 落 JSON
    out_json = {
        "integrity": integ,
        "distribution": {
            "midprice_by_sym": dist["midprice_by_sym"],
            "amount_delta_by_sym": dist["amount_delta_by_sym"],
            "label_dist": dist["label_dist"],
        },
        "midprice_diff": {k: v for k, v in diff_info.items() if k != "_diffs_for_plot"},
        "random_baseline_pnl": {str(k): v for k, v in rand_pnl.items()},
        "fee_rate": FEE_RATE,
    }
    with open(os.path.join(OUT_DIR, "eda_stats.json"), "w") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    # 写报告
    write_report(out_json)
    print("[eda] done.")
    return 0


def write_report(out: dict) -> None:
    integ = out["integrity"]
    dist = out["distribution"]
    diff = out["midprice_diff"]
    rb = out["random_baseline_pnl"]

    lines: List[str] = []
    lines.append(f"# 良文杯数据 EDA 报告\n")
    lines.append(f"_生成时间: {datetime.now().isoformat(timespec='seconds')}_\n")

    lines.append("## 1. 数据完整性\n")
    lines.append(f"- 文件数：发现 **{integ['n_files_seen']}** / 期望 **{integ['n_files_expected']}**")
    if integ["missing_files"]:
        lines.append(f"  - 缺失：{integ['missing_files'][:10]}{' ...' if len(integ['missing_files']) > 10 else ''}")
    else:
        lines.append("  - ✅ 无缺失")
    lines.append(f"- 总行数：{integ['total_rows']:,}（期望 {1200 * EXPECTED_ROWS:,}）")
    if integ["row_count_violations"]:
        lines.append(f"  - ⚠️ 行数异常的文件：{integ['row_count_violations'][:5]}")
    else:
        lines.append("  - ✅ 所有文件均为 2001 行")
    lines.append(f"- NaN 异常文件：**{integ['n_files_with_nan']}** 个 / 1200")
    if integ["n_files_with_nan"]:
        lines.append("")
        lines.append("  | (sym, date, session) | total NaN | 受影响列数 | 单列最多 NaN | 仅 ask 侧? |")
        lines.append("  |---|---|---|---|---|")
        for s in integ.get("nan_file_summary", []):
            k = s["key"]
            lines.append(
                f"  | ({k[0]}, {k[1]}, {k[2]}) | {s['total_nan']:,} | "
                f"{s['n_cols_affected']} | {s['max_nan_in_one_col']} | "
                f"{'✅' if s['ask_side_only'] else '❌'} |"
            )
        lines.append("")
        lines.append(
            "  > 所有 NaN 都集中在 **ask 侧 LOB 深度**（ask{N}/asize{N}/spread{N}/ask_diff{N}/ask_rate{N}/asize_rate{N}）。"
            "bid 侧、midprice、label 全部干净。\n"
            "  > 出现 NaN 是 A 股**涨停**机制的副作用：限价 = +9.97% 时卖盘几乎被买光，深档位无报价。"
            "  最严重的 (1, 91, pm) 有 1308/2001 ≈ 65% tick 处于涨停。\n"
            "  > **训练对策**：可选 (a) 直接 drop 这 8 个 session（仅 0.67% 数据），或 (b) NaN → 0 + 增加 ask_missing 指示位。"
        )
    lines.append(f"- Inf 异常文件：{integ['n_files_with_inf']} 个")
    lines.append(f"- 时间戳异常文件：{integ['n_files_with_time_violation']} 个")
    if integ["n_files_with_time_violation"]:
        lines.append(f"  - 样本：{integ.get('sample_time_violations', [])}")
    lines.append(f"- midprice = -1 (会让 PnL 分母为 0) 的文件：{integ['n_files_with_midprice_zero']} 个")
    lines.append(f"- midprice = NaN 的文件：{integ['n_files_with_midprice_nan']} 个")
    if integ["n_files_with_midprice_zero"] == 0 and integ["n_files_with_midprice_nan"] == 0:
        lines.append("  - ✅ midprice 列健康，PnL 分母安全")
    lines.append("")

    lines.append("## 2. midprice 分布（按 sym 聚合）\n")
    lines.append("| sym | min | q01 | q50 | mean | q99 | max | std |")
    lines.append("|-----|-----|-----|-----|------|-----|-----|-----|")
    for sym, st in dist["midprice_by_sym"].items():
        lines.append(
            f"| {sym} | {st['min']:.4%} | {st['q01']:.4%} | {st['q50']:.4%} | {st['mean']:.4%} | "
            f"{st['q99']:.4%} | {st['max']:.4%} | {st['std']:.4%} |"
        )
    lines.append("")
    lines.append("![midprice 分布](figures/midprice_by_sym.png)\n")

    lines.append("## 3. amount_delta 量级（唯一未归一化字段）\n")
    lines.append("| sym | min | q01 | q50 | mean | q99 | max | std |")
    lines.append("|-----|-----|-----|-----|------|-----|-----|-----|")
    for sym, st in dist["amount_delta_by_sym"].items():
        lines.append(
            f"| {sym} | {st['min']:.3g} | {st['q01']:.3g} | {st['q50']:.3g} | {st['mean']:.3g} | "
            f"{st['q99']:.3g} | {st['max']:.3g} | {st['std']:.3g} |"
        )
    lines.append("")
    lines.append("> ⚠️ amount_delta 跨 sym 量级差异巨大，模型用前必须归一化（log1p 或按 sym z-score）。\n")
    lines.append("![amount_delta 量级](figures/amount_delta_by_sym.png)\n")

    lines.append("## 4. 标签分布\n")
    lines.append("### 4.1 总体（不分 sym）\n")
    lines.append("| label | p_down | p_flat | p_up | n |")
    lines.append("|-------|--------|--------|------|---|")
    for lc, d in dist["label_dist"]["by_horizon"].items():
        n = d["0_down"] + d["1_flat"] + d["2_up"]
        lines.append(
            f"| {lc} | {d['p_down']:.2%} | {d['p_flat']:.2%} | {d['p_up']:.2%} | {n:,} |"
        )
    lines.append("")
    lines.append("![标签分布 by horizon](figures/label_distribution_by_horizon.png)\n")

    lines.append("### 4.2 按 sym × horizon\n")
    lines.append("![标签分布 by sym](figures/label_distribution_by_sym.png)\n")

    lines.append("## 5. midprice 单 tick 跳变\n")
    lines.append(
        f"- 全部 1-tick diff 数：{diff['n_diffs']:,}\n"
        f"- 单 tick midprice 不动比例：**{diff['pct_zero']:.2%}**（session 平均 {diff['pct_zero_sess_mean']:.2%}, "
        f"min {diff['pct_zero_sess_min']:.2%}, max {diff['pct_zero_sess_max']:.2%}）\n"
        f"- |Δmidprice| 分位数：q50={diff['abs_quantiles']['q50']:.2e}, q90={diff['abs_quantiles']['q90']:.2e}, "
        f"q99={diff['abs_quantiles']['q99']:.2e}, max={diff['abs_quantiles']['max']:.2e}\n"
    )
    lines.append("![midprice 单 tick 跳变](figures/midprice_tick_diff.png)\n")

    lines.append("## 6. \"瞎猜\" 基线 PnL（关键）\n")
    lines.append("- 评分公式：`pnl = [(a-1)·Δmid - 0.0001·|a-1|·((mid_t+1)+(mid_tn+1))] / (mid_t+1)`")
    lines.append("- 手续费 0.01% 双边 → 单笔手续费 ≈ 0.0002 / (mid_t+1) ≈ **0.02%**\n")
    lines.append("| horizon | 随机{跌,涨}/每笔 | 总是涨/每笔 | 总是跌/每笔 | 单笔手续费 | n_pairs |")
    lines.append("|---------|------------------|--------------|--------------|------------|---------|")
    for h, st in rb.items():
        lines.append(
            f"| {h} | {st['rand_2cls_per_trade']:+.4%} | {st['always_up_per_trade']:+.4%} | "
            f"{st['always_down_per_trade']:+.4%} | {st['fee_avg_per_trade']:.4%} | {st['n_pairs']:,} |"
        )
    lines.append("")
    lines.append(
        "**结论**：随机预测{跌,涨} 在所有 horizon 上每笔期望 PnL ≈ -0.02%（手续费），与 horizon 无关。\n"
        "总是预测涨/跌的 PnL 取决于数据期内 midprice 的整体漂移；如果训练-评测期分布漂移，'always-up' 也可能正——"
        "这正是为什么需要按时间划分 train/val/test 而不是随机切。\n"
    )
    lines.append("**模型必须 beat 的下限**：单笔 PnL 至少 > 0（即净收益 > 手续费 0.02%），整体 PnL 才为正。\n")
    lines.append("![随机基线 PnL](figures/random_baseline_pnl.png)\n")

    lines.append("## 7. horizon 选择建议\n")
    # 根据 label flat 比例越低，方向信号越强
    horizon_signal = []
    for lc, d in dist["label_dist"]["by_horizon"].items():
        h = int(lc.split("_")[1])
        flat = d["p_flat"]
        non_flat = 1 - flat
        # 用类不平衡程度衡量"信息量"：值越接近 1/2(均衡的非平类) 信号越强
        horizon_signal.append((h, flat, non_flat))
    horizon_signal.sort(key=lambda x: -x[2])
    lines.append("| horizon | p_flat | p_非平 (出手机会比例) |")
    lines.append("|---------|--------|------------------------|")
    for h, flat, nonflat in horizon_signal:
        lines.append(f"| label_{h} | {flat:.2%} | {nonflat:.2%} |")
    lines.append("")
    lines.append(
        "**注意**：α 阈值不同造成的非单调——label_10 的 p_非平 (34.9%) > label_20 (26.2%)，"
        "因为 5/10 用 α=0.05%、20/40/60 用 α=0.1%；阈值翻倍时出手机会突然变少，到 40/60 因 horizon 长再回升。\n"
        "短 horizon（5/10）每笔潜在收益小，手续费占比相对更高 → 需要 precision 更高才划算。\n"
        "长 horizon（40/60）每笔潜在收益大但回看跨度长，预测难度也高。\n"
        "**建议**：5 个 horizon 都训一个 head，最后按提交规则取最好的 1 个 horizon 上榜。"
        "label_10 和 label_60 是出手机会最多的两档，重点优化。\n"
    )

    lines.append("## 8. 关键提醒（写给后续 worker）\n")
    lines.append(
        "1. midprice 是**相对昨收的涨跌幅**（中心在 0），不是真实价格。计算 PnL 必须用 `midprice + 1`。\n"
        "2. amount_delta 跨 sym 差好几个量级，**模型前必须归一化**（建议 log1p 后按 sym z-score）。\n"
        "3. time 字段在每个 session 内连续 3s，无缺失 → 不需要插值。\n"
        "4. label 已经标注好（无 NaN），无需重新计算；想改 α 阈值的话用 midprice + horizon 自行重标。\n"
        "5. 5 个 head 同时训练时损失要平衡：因为 p_flat 不同，类权重 / focal loss 应按 horizon 单独调。\n"
        "6. PnL 评测时空仓的 t（label=1，即平类）应被忽略，但 acc/f1 算指标时要全算 → 训练损失和评测口径不一致。\n"
        "7. 8 个 NaN 文件全部在 sym=1 / sym=3 的涨停日，仅影响 ask 侧深档；bid 侧和 midprice 干净。"
        "训练时建议把 NaN 替换为 0 并增加 `ask_missing` 指示位，避免直接喂 NaN 给模型。\n"
        "8. 单 tick midprice **70.9% 概率不动**——这是 LOB 数据的常见性质。模型若只学'下一 tick 是否动'会被噪声主导，"
        "应直接学 horizon=5/10/20/40/60 ticks 后的方向。\n"
    )
    with open(os.path.join(OUT_DIR, "eda_report.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
