"""
本地 PnL 评测器 — 严格按"良文杯"官方评分公式实现。

PDF 1.5:
    pnl_single = [ (label_pred - 1) * (midprice_{t+n} - midprice_t)
                   - fee_rate * |label_pred - 1| * ((midprice_{t+n}+1) + (midprice_t+1)) ]
                 / (midprice_t + 1)

含义：
- midprice 字段 = 相对昨收的涨跌幅（无量纲）。midprice + 1 是相对昨收的价格倍数。
- label_pred = 2(涨)：side=+1，买入再卖出；label_pred = 0(跌)：side=-1，卖空再买回；
  label_pred = 1(平)：side=0，不交易，pnl = 0。
- 手续费 fee_rate = 0.0001 (万 1 = 0.01%)，双边均价上各扣一次。
- 累计收益率 = sum(pnl_single) — 这是模型评分。
- 单次收益率 = 累计 / (涨预测数 + 跌预测数)。
- 5 个 horizon 各算一份分；最高的那个用于排名。
"""
from __future__ import annotations

import numpy as np

HORIZONS: tuple[int, ...] = (5, 10, 20, 40, 60)
HORIZON_NAMES: list[str] = [f"label_{h}" for h in HORIZONS]


def _scalar_safe_div(num: float, den: float, default: float = 0.0) -> float:
    return float(num) / float(den) if den != 0 else float(default)


def _fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if np.isnan(precision) or np.isnan(recall):
        return float("nan")
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall)


def _macro_avg(*vals: float) -> float:
    finite = [v for v in vals if not (v is None or np.isnan(v))]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _per_horizon_metrics(
    pred_h: np.ndarray,
    label_h: np.ndarray,
    midprice_t: np.ndarray,
    midprice_tn_h: np.ndarray,
    fee_rate: float,
) -> dict:
    """
    对单个 horizon 计算所有指标。完全向量化。

    pred_h:        (N,) int — 0/1/2
    label_h:       (N,) int — 0/1/2
    midprice_t:    (N,) float — 当前 midprice (相对涨跌幅)
    midprice_tn_h: (N,) float — t+n 的 midprice
    """
    pred_h = pred_h.astype(np.int64)
    label_h = label_h.astype(np.int64)
    mp_t = midprice_t.astype(np.float64)
    mp_tn = midprice_tn_h.astype(np.float64)
    n_total = int(pred_h.shape[0])

    # ---------- PnL (vectorized) ----------
    side = pred_h.astype(np.float64) - 1.0          # -1, 0, +1
    abs_side = np.abs(side)                          # 0 或 1
    diff = mp_tn - mp_t                              # (N,)
    fee = fee_rate * abs_side * ((mp_tn + 1.0) + (mp_t + 1.0))
    denom = mp_t + 1.0                               # > 0 (除非股价归零)
    # 安全：实际 midprice 是相对涨跌幅，>-1，所以 denom > 0
    pnl = (side * diff - fee) / denom                # (N,)

    cum_pnl = float(pnl.sum())
    n_active = int((pred_h != 1).sum())
    single_pnl = _scalar_safe_div(cum_pnl, n_active, default=0.0)

    # ---------- Classification metrics ----------
    correct = pred_h == label_h
    accuracy = float(correct.mean()) if n_total > 0 else 0.0

    pred_up = pred_h == 2
    pred_dn = pred_h == 0
    label_up = label_h == 2
    label_dn = label_h == 0

    tp_up = int((pred_up & label_up).sum())
    tp_dn = int((pred_dn & label_dn).sum())
    n_pred_up = int(pred_up.sum())
    n_pred_dn = int(pred_dn.sum())
    n_label_up = int(label_up.sum())
    n_label_dn = int(label_dn.sum())

    precision_up = _scalar_safe_div(tp_up, n_pred_up, default=float("nan"))
    precision_dn = _scalar_safe_div(tp_dn, n_pred_dn, default=float("nan"))
    recall_up = _scalar_safe_div(tp_up, n_label_up, default=float("nan"))
    recall_dn = _scalar_safe_div(tp_dn, n_label_dn, default=float("nan"))

    precision_macro = _macro_avg(precision_up, precision_dn)
    recall_macro = _macro_avg(recall_up, recall_dn)

    f0_5_up = _fbeta(precision_up, recall_up, beta=0.5)
    f0_5_dn = _fbeta(precision_dn, recall_dn, beta=0.5)
    f0_5_macro = _macro_avg(f0_5_up, f0_5_dn)

    pred_dist = {0: int(pred_dn.sum()), 1: int((pred_h == 1).sum()), 2: int(pred_up.sum())}
    label_dist = {0: int(label_dn.sum()), 1: int((label_h == 1).sum()), 2: int(label_up.sum())}

    return {
        "accuracy": accuracy,
        "precision_up": precision_up,
        "precision_down": precision_dn,
        "precision_macro": precision_macro,
        "recall_up": recall_up,
        "recall_down": recall_dn,
        "recall_macro": recall_macro,
        "f0_5_up": f0_5_up,
        "f0_5_down": f0_5_dn,
        "f0_5_macro": f0_5_macro,
        "cum_pnl": cum_pnl,
        "single_pnl": single_pnl,
        "n_predictions_active": n_active,
        "n_total": n_total,
        "pred_distribution": pred_dist,
        "label_distribution": label_dist,
    }


def compute_pnl(
    pred: np.ndarray,
    label: np.ndarray,
    midprice_t: np.ndarray,
    midprice_tn: np.ndarray,
    fee_rate: float = 0.0001,
) -> dict:
    """
    计算 5 个 horizon 的 PnL + 分类指标。

    pred         : (N, 5) int — 每个样本对 5 个 horizon 的预测 (label_5/10/20/40/60)
    label        : (N, 5) int — 真实标签
    midprice_t   : (N,)    float — 当前 midprice
    midprice_tn  : (N, 5)  float — 未来 t+5/10/20/40/60 的 midprice
    fee_rate     : 单边手续费率 (默认 0.0001 = 0.01%)

    Returns: dict — 见模块 docstring 顶部，每个 horizon 一份指标 + best_score + best_horizon。
    """
    pred = np.asarray(pred)
    label = np.asarray(label)
    midprice_t = np.asarray(midprice_t, dtype=np.float64)
    midprice_tn = np.asarray(midprice_tn, dtype=np.float64)

    if pred.ndim != 2 or pred.shape[1] != len(HORIZONS):
        raise ValueError(f"pred shape must be (N, {len(HORIZONS)}); got {pred.shape}")
    if label.shape != pred.shape:
        raise ValueError(f"label shape {label.shape} must match pred {pred.shape}")
    if midprice_tn.shape != pred.shape:
        raise ValueError(f"midprice_tn shape {midprice_tn.shape} must match pred {pred.shape}")
    if midprice_t.shape != (pred.shape[0],):
        raise ValueError(f"midprice_t shape {midprice_t.shape} must be ({pred.shape[0]},)")

    out: dict = {}
    cum_pnls: list[float] = []
    for i, name in enumerate(HORIZON_NAMES):
        m = _per_horizon_metrics(
            pred[:, i], label[:, i], midprice_t, midprice_tn[:, i], fee_rate
        )
        out[name] = m
        cum_pnls.append(m["cum_pnl"])

    best_idx = int(np.argmax(cum_pnls))
    out["best_score"] = float(cum_pnls[best_idx])
    out["best_horizon"] = HORIZON_NAMES[best_idx]
    return out


def sanitize_for_json(obj):
    """递归把 NaN/Inf 替换为 None，方便 json.dump。"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return None if (np.isnan(x) or np.isinf(x)) else x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj
