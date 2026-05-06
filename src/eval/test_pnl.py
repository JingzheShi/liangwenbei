"""
Sanity-check 测试 src/eval/pnl.py 的实现。

5 个 case:
1. 完美预测 (pred = label)：PnL 远高于其他 trivial 策略
2. 全预测 1 (不变)：cum_pnl 严格 == 0 (no trade no fee)
3. 全预测 2 (涨)：cum_pnl 通常为负 (手续费 + 反向移动)
4. 全预测 0 (跌)：cum_pnl 通常为负
5. 随机预测：sanity 检查不出错且接近全预测 0/2 加权平均

加载真实数据：data/snapshot_sym0_date0_am.parquet。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `from pnl import ...` regardless of how this script is invoked.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from pnl import (  # noqa: E402
    HORIZON_NAMES,
    HORIZONS,
    compute_pnl,
)

DATA_PATH = ROOT / "data" / "snapshot_sym0_date0_am.parquet"
WINDOW = 100  # 输入 window，决定 valid t 的下界


def build_eval_arrays(df: pd.DataFrame, horizons=HORIZONS, window: int = WINDOW):
    """
    从单个 session DataFrame 构造合法评测点：
        valid t 范围 [window-1, len-max(horizons)-1]
    返回:
        label  (N, 5) int
        mp_t   (N,) float
        mp_tn  (N, 5) float
    """
    n = len(df)
    max_h = max(horizons)
    t_lo = window - 1
    t_hi = n - max_h - 1
    valid_t = np.arange(t_lo, t_hi + 1)  # 闭区间
    assert len(valid_t) > 0, f"no valid t in session of len {n}"

    midprice = df["midprice"].to_numpy(dtype=np.float64)
    label_cols = [f"label_{h}" for h in horizons]
    label = df[label_cols].to_numpy(dtype=np.int64)

    mp_t = midprice[valid_t]
    mp_tn = np.stack([midprice[valid_t + h] for h in horizons], axis=1)
    label_t = label[valid_t]
    return label_t, mp_t, mp_tn, valid_t


def _summarize(name: str, result: dict, *, indent: int = 2) -> str:
    """格式化打印每个 horizon 的关键指标。"""
    pad = " " * indent
    lines = [f"{name}:"]
    for h in HORIZON_NAMES:
        m = result[h]
        lines.append(
            f"{pad}{h}: cum={m['cum_pnl']:+.6f}  single={m['single_pnl']:+.6e}  "
            f"acc={m['accuracy']:.4f}  P_up={_fmt(m['precision_up'])}  "
            f"P_dn={_fmt(m['precision_down'])}  active={m['n_predictions_active']}/{m['n_total']}  "
            f"pred_dist={m['pred_distribution']}"
        )
    lines.append(f"{pad}best_score={result['best_score']:+.6f} @ {result['best_horizon']}")
    return "\n".join(lines)


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  nan "
    return f"{x:.4f}"


def main() -> int:
    print(f"[test_pnl] loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    label, mp_t, mp_tn, valid_t = build_eval_arrays(df)
    n = label.shape[0]
    print(f"[test_pnl] valid samples: {n} (t in [{valid_t[0]}, {valid_t[-1]}])")
    print(f"[test_pnl] true label distribution per horizon:")
    for i, h in enumerate(HORIZON_NAMES):
        u, c = np.unique(label[:, i], return_counts=True)
        d = {int(k): int(v) for k, v in zip(u, c)}
        print(f"           {h}: {d}")

    # --------------- Case 1: perfect prediction ---------------
    print("\n=== Case 1: pred = label (perfect) ===")
    res_perfect = compute_pnl(label.copy(), label, mp_t, mp_tn)
    print(_summarize("perfect", res_perfect))

    # --------------- Case 2: all predict 1 (flat) ---------------
    print("\n=== Case 2: pred = 1 everywhere (no trade) ===")
    pred_all_1 = np.ones_like(label)
    res_flat = compute_pnl(pred_all_1, label, mp_t, mp_tn)
    print(_summarize("all_1", res_flat))

    # --------------- Case 3: all predict 2 (long) ---------------
    print("\n=== Case 3: pred = 2 everywhere (always long) ===")
    pred_all_2 = np.full_like(label, 2)
    res_long = compute_pnl(pred_all_2, label, mp_t, mp_tn)
    print(_summarize("all_2", res_long))

    # --------------- Case 4: all predict 0 (short) ---------------
    print("\n=== Case 4: pred = 0 everywhere (always short) ===")
    pred_all_0 = np.zeros_like(label)
    res_short = compute_pnl(pred_all_0, label, mp_t, mp_tn)
    print(_summarize("all_0", res_short))

    # --------------- Case 5: random prediction ---------------
    print("\n=== Case 5: random pred (uniform 0/1/2) ===")
    rng = np.random.default_rng(42)
    pred_rand = rng.integers(0, 3, size=label.shape)
    res_rand = compute_pnl(pred_rand, label, mp_t, mp_tn)
    print(_summarize("random", res_rand))

    # ------------------ Assertions ------------------
    print("\n=== Assertions ===")
    failures: list[str] = []

    # (a) all-flat strict zero on every horizon
    for h in HORIZON_NAMES:
        if abs(res_flat[h]["cum_pnl"]) > 1e-12:
            failures.append(f"all-1 cum_pnl on {h} = {res_flat[h]['cum_pnl']} (expected 0)")
        if res_flat[h]["n_predictions_active"] != 0:
            failures.append(f"all-1 active count on {h} = {res_flat[h]['n_predictions_active']}")
    print(f"  (a) all-1 strict zero PnL across all 5 horizons: "
          f"{'PASS' if not failures else 'FAIL'}")

    # (b) perfect dominates trivial baselines
    for h in HORIZON_NAMES:
        p = res_perfect[h]["cum_pnl"]
        for name, other in [("all_1", res_flat), ("all_2", res_long),
                             ("all_0", res_short), ("random", res_rand)]:
            o = other[h]["cum_pnl"]
            if not (p >= o):
                failures.append(f"perfect cum_pnl on {h} ({p:.6e}) < {name} ({o:.6e})")
    print("  (b) perfect >= every trivial baseline on every horizon: "
          f"{'PASS' if not [f for f in failures if 'perfect' in f] else 'FAIL'}")

    # (c) perfect accuracy = 1.0
    for h in HORIZON_NAMES:
        if abs(res_perfect[h]["accuracy"] - 1.0) > 1e-9:
            failures.append(f"perfect acc on {h} = {res_perfect[h]['accuracy']} (expected 1.0)")
    print("  (c) perfect accuracy == 1.0: "
          f"{'PASS' if not [f for f in failures if 'perfect acc' in f] else 'FAIL'}")

    # (d) all-2 cum_pnl == sum(diff - fee)/denom (manual recompute)
    for i, h in enumerate(HORIZON_NAMES):
        diff = mp_tn[:, i] - mp_t
        fee = 0.0001 * ((mp_tn[:, i] + 1.0) + (mp_t + 1.0))
        manual = float(((1.0 * diff - fee) / (mp_t + 1.0)).sum())
        got = res_long[h]["cum_pnl"]
        if abs(manual - got) > 1e-9:
            failures.append(f"all-2 cum_pnl on {h}: manual {manual} vs got {got}")
    print("  (d) all-2 PnL matches manual vectorized recompute: "
          f"{'PASS' if not [f for f in failures if 'all-2 cum_pnl' in f] else 'FAIL'}")

    # (e) sign symmetry: all-2 and all-0 differ by 2*sum(diff)/denom (no fee diff)
    for i, h in enumerate(HORIZON_NAMES):
        diff = mp_tn[:, i] - mp_t
        denom = mp_t + 1.0
        expected_delta = float((2.0 * diff / denom).sum())
        got_delta = res_long[h]["cum_pnl"] - res_short[h]["cum_pnl"]
        if abs(expected_delta - got_delta) > 1e-9:
            failures.append(f"all-2 vs all-0 delta mismatch on {h}: {got_delta} vs {expected_delta}")
    print("  (e) (all-2 cum) - (all-0 cum) == 2 * sum(diff/denom): "
          f"{'PASS' if not [f for f in failures if 'all-2 vs all-0' in f] else 'FAIL'}")

    # (f) random PnL within (min(all-0, all-2), perfect) ballpark per horizon
    #     This isn't a strict bound (random can occasionally beat all-X by chance),
    #     but it should be much worse than perfect.
    for h in HORIZON_NAMES:
        if not (res_rand[h]["cum_pnl"] < res_perfect[h]["cum_pnl"]):
            failures.append(f"random {h} ({res_rand[h]['cum_pnl']:.6e}) "
                            f">= perfect ({res_perfect[h]['cum_pnl']:.6e})")
    print("  (f) random < perfect: "
          f"{'PASS' if not [f for f in failures if 'random' in f] else 'FAIL'}")

    # ------------------ Final ------------------
    print()
    if failures:
        print(f"[test_pnl] FAILED ({len(failures)} issues):")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("[test_pnl] ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
