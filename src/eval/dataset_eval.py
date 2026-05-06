"""
基于 Predictor 接口的本地评测 helper。

合法评测点定义（每个 session）：
    valid t 范围 [window-1, len-max(horizons)-1]
    输入 = 过去 window ticks 的特征 [t-window+1, t]
    目标 = 该行的 5 个 label + 该行 midprice + t+5/10/20/40/60 的 midprice

注意（PDF 2.4）：
- 测试点输入顺序会被打乱，模型不能跨样本看时序。本 helper 不打乱顺序，但同样不要在
  predict_fn 内做跨样本依赖。要 mimic 提交环境，可以在 batch 拼装前 shuffle。
- 不能向后看：构造 window 时只用 [t-window+1, t]，绝不含 t+1..t+max_horizon。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from .pnl import HORIZONS, compute_pnl
except ImportError:  # script-style invocation with HERE on sys.path
    from pnl import HORIZONS, compute_pnl  # type: ignore

PredictFn = Callable[[list[pd.DataFrame]], list[list[int]]]


def _windows_to_inputs(
    feat_df: pd.DataFrame,
    valid_t: np.ndarray,
    window: int,
) -> list[pd.DataFrame]:
    """
    把 valid_t 上的所有滑窗切出来作为 List[DataFrame]。
    """
    return [feat_df.iloc[t - window + 1 : t + 1] for t in valid_t]


def _gather_session(
    session_df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int,
    horizons: Sequence[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (feat_df_reset, valid_t, label, mp_t, mp_tn)。
    feat_df 已 reset_index(drop=True) 与 valid_t 对齐。
    """
    n = len(session_df)
    max_h = max(horizons)
    if n < window + max_h:
        return None  # type: ignore[return-value]

    feat_df = session_df[list(feature_cols)].reset_index(drop=True)
    midprice = session_df["midprice"].to_numpy(dtype=np.float64)
    label_cols = [f"label_{h}" for h in horizons]
    label_full = session_df[label_cols].to_numpy(dtype=np.int64)

    t_lo = window - 1
    t_hi = n - max_h - 1
    valid_t = np.arange(t_lo, t_hi + 1, dtype=np.int64)

    mp_t = midprice[valid_t]
    mp_tn = np.stack([midprice[valid_t + h] for h in horizons], axis=1)
    label_t = label_full[valid_t]

    return feat_df, valid_t, label_t, mp_t, mp_tn


def evaluate_on_session(
    predict_fn: PredictFn,
    session_df: pd.DataFrame,
    feature_cols: Sequence[str],
    window: int = 100,
    horizons: Sequence[int] = HORIZONS,
    batch_size: int = 1024,
    fee_rate: float = 0.0001,
) -> dict | None:
    """
    在单个 session 上调 predict_fn 并计算 PnL + 分类指标。

    predict_fn:    callable: List[pd.DataFrame] -> List[List[int]]
                   每个 DataFrame 是 (window 行 × len(feature_cols) 列)
                   输出每行长度 = len(horizons)
    session_df:    单个 (sym, date, am/pm) session（必须按时间升序）
    feature_cols:  传入 predict_fn 的特征列名列表
    window:        历史窗口长度（默认 100）
    horizons:      预测 horizon 列表（默认 (5,10,20,40,60)）
    batch_size:    调用 predict_fn 的 batch 大小（避免 OOM）
    fee_rate:      单边手续费率

    Returns: compute_pnl 的输出 dict；若 session 太短返回 None。
    """
    gathered = _gather_session(session_df, feature_cols, window, horizons)
    if gathered is None:
        return None
    feat_df, valid_t, label_t, mp_t, mp_tn = gathered
    n_h = len(horizons)
    n_samples = len(valid_t)

    preds = np.zeros((n_samples, n_h), dtype=np.int64)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        chunk_t = valid_t[start:end]
        windows = _windows_to_inputs(feat_df, chunk_t, window)
        out = predict_fn(windows)
        out_arr = np.asarray(out, dtype=np.int64)
        if out_arr.ndim == 1:
            # 兼容 single-label 模型输出 (batch,) — 广播到所有 horizon
            out_arr = np.repeat(out_arr[:, None], n_h, axis=1)
        if out_arr.shape != (end - start, n_h):
            raise ValueError(
                f"predict_fn output shape {out_arr.shape}, expected ({end-start}, {n_h})"
            )
        preds[start:end] = out_arr

    return compute_pnl(preds, label_t, mp_t, mp_tn, fee_rate=fee_rate)


def evaluate_on_dataset(
    predict_fn: PredictFn,
    dataset_dir: str | Path,
    sessions: Iterable[tuple[int | str, int | str, str]],
    feature_cols: Sequence[str],
    window: int = 100,
    horizons: Sequence[int] = HORIZONS,
    batch_size: int = 1024,
    fee_rate: float = 0.0001,
    file_template: str = "snapshot_sym{sym}_date{date}_{ampm}.parquet",
    verbose: bool = False,
) -> dict:
    """
    跨多个 session 聚合评测。所有样本拼起来一次性算 PnL。

    sessions: 可迭代 [(sym, date, 'am'|'pm'), ...]
    file_template: 文件名模板（默认匹配现有数据格式）
    """
    dataset_dir = Path(dataset_dir)
    all_pred: list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    all_mp_t: list[np.ndarray] = []
    all_mp_tn: list[np.ndarray] = []
    n_h = len(horizons)

    n_sessions = 0
    n_skipped = 0
    for sym, date, ampm in sessions:
        path = dataset_dir / file_template.format(sym=sym, date=date, ampm=ampm)
        if not path.exists():
            if verbose:
                print(f"[evaluate_on_dataset] skip missing file: {path}")
            n_skipped += 1
            continue
        session_df = pd.read_parquet(path)
        gathered = _gather_session(session_df, feature_cols, window, horizons)
        if gathered is None:
            n_skipped += 1
            continue
        feat_df, valid_t, label_t, mp_t, mp_tn = gathered
        n_samples = len(valid_t)

        preds = np.zeros((n_samples, n_h), dtype=np.int64)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            chunk_t = valid_t[start:end]
            windows = _windows_to_inputs(feat_df, chunk_t, window)
            out = predict_fn(windows)
            out_arr = np.asarray(out, dtype=np.int64)
            if out_arr.ndim == 1:
                out_arr = np.repeat(out_arr[:, None], n_h, axis=1)
            if out_arr.shape != (end - start, n_h):
                raise ValueError(
                    f"predict_fn output shape {out_arr.shape}, expected ({end-start}, {n_h})"
                )
            preds[start:end] = out_arr

        all_pred.append(preds)
        all_label.append(label_t)
        all_mp_t.append(mp_t)
        all_mp_tn.append(mp_tn)
        n_sessions += 1
        if verbose:
            print(f"[evaluate_on_dataset] session sym={sym} date={date} {ampm}: "
                  f"{n_samples} samples")

    if n_sessions == 0:
        raise RuntimeError(
            f"no valid sessions evaluated (skipped={n_skipped})"
        )

    pred_cat = np.concatenate(all_pred, axis=0)
    label_cat = np.concatenate(all_label, axis=0)
    mp_t_cat = np.concatenate(all_mp_t, axis=0)
    mp_tn_cat = np.concatenate(all_mp_tn, axis=0)

    result = compute_pnl(pred_cat, label_cat, mp_t_cat, mp_tn_cat, fee_rate=fee_rate)
    result["meta"] = {
        "n_sessions": n_sessions,
        "n_sessions_skipped": n_skipped,
        "n_samples_total": int(pred_cat.shape[0]),
    }
    return result
