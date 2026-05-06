"""LOBDataset：把 (sym, date, session) 三元组列表转成 PyTorch Dataset。

每个样本对应一个预测点 t，返回过去 window ticks 的特征以及 5 个 horizon 的 label。
跨 session 不滑窗（不同文件之间在物理上不相邻）。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .split import get_file_path

SymDate = Tuple[int, int, str]
DEFAULT_LABEL_COLS = ("label_5", "label_10", "label_20", "label_40", "label_60")
DEFAULT_HORIZONS = (5, 10, 20, 40, 60)


class LOBDataset(Dataset):
    def __init__(
        self,
        sym_dates: Sequence[SymDate],
        feature_cols: Sequence[str],
        label_cols: Sequence[str] = DEFAULT_LABEL_COLS,
        window: int = 100,
        data_dir: str = "data",
        cache_in_memory: bool = True,
        return_midprices: bool = False,
        return_meta: bool = False,
        max_horizon: int = 60,
        feature_dtype: np.dtype = np.float32,
    ) -> None:
        if window < 1:
            raise ValueError("window 必须 >= 1")
        if max_horizon < 1:
            raise ValueError("max_horizon 必须 >= 1")
        self.sym_dates: List[SymDate] = list(sym_dates)
        self.feature_cols: List[str] = list(feature_cols)
        self.label_cols: List[str] = list(label_cols)
        self.window = int(window)
        self.max_horizon = int(max_horizon)
        self.data_dir = data_dir
        self.cache_in_memory = cache_in_memory
        self.return_midprices = return_midprices
        self.return_meta = return_meta
        self.feature_dtype = feature_dtype

        # 每个 session 一个 entry：(features (T, F), labels (T, L), midprice (T,))
        # 使用 list 而不是 dict，配合 prefix-sum 索引
        self._sessions_X: List[Optional[np.ndarray]] = [None] * len(self.sym_dates)
        self._sessions_Y: List[Optional[np.ndarray]] = [None] * len(self.sym_dates)
        self._sessions_M: List[Optional[np.ndarray]] = [None] * len(self.sym_dates)
        self._session_lens: np.ndarray  # (n_sessions,) total tick count per session

        # Discover session lengths (一次扫描每个文件元数据 + 数值预读)
        self._session_lens = np.zeros(len(self.sym_dates), dtype=np.int64)
        if self.cache_in_memory:
            for i, key in enumerate(self.sym_dates):
                X, Y, M = self._load_one(key)
                self._sessions_X[i] = X
                self._sessions_Y[i] = Y
                self._sessions_M[i] = M
                self._session_lens[i] = len(X)
        else:
            # 仅读 row-count；__getitem__ 时再 lazy load
            for i, key in enumerate(self.sym_dates):
                path = get_file_path(*key, data_dir=self.data_dir)
                # 用 pyarrow metadata 拿 row count（不读数据）
                import pyarrow.parquet as pq

                self._session_lens[i] = pq.ParquetFile(path).metadata.num_rows

        # 每个 session 的有效样本数 = max(0, len - (window - 1) - max_horizon)
        valid_per_session = self._session_lens - (self.window - 1) - self.max_horizon
        valid_per_session = np.maximum(valid_per_session, 0)
        self._valid_per_session = valid_per_session.astype(np.int64)
        self._cum = np.cumsum(self._valid_per_session)
        self._total = int(self._cum[-1]) if len(self._cum) else 0

        # horizons：从 label_cols 名字反推（label_5 → 5）
        self.horizons = self._infer_horizons(self.label_cols)

    @staticmethod
    def _infer_horizons(label_cols: Sequence[str]) -> List[int]:
        out = []
        for c in label_cols:
            if not c.startswith("label_"):
                raise ValueError(f"label 列名必须形如 'label_<horizon>'，得到 {c!r}")
            out.append(int(c.split("_", 1)[1]))
        return out

    def _load_one(
        self, key: SymDate
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读一个 parquet → (X[T,F] float32, Y[T,L] int64, midprice[T,] float32)。"""
        path = get_file_path(*key, data_dir=self.data_dir)
        df = pd.read_parquet(path)
        # 安全检查：所有 feature/label/midprice 列必须存在
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            raise KeyError(f"{path} 缺少特征列: {missing}")
        missing_l = set(self.label_cols) - set(df.columns)
        if missing_l:
            raise KeyError(f"{path} 缺少标签列: {missing_l}")
        if "midprice" not in df.columns:
            raise KeyError(f"{path} 缺少 midprice 列")
        X = df[self.feature_cols].to_numpy(dtype=self.feature_dtype, copy=False)
        Y = df[self.label_cols].to_numpy(dtype=np.int64, copy=False)
        M = df["midprice"].to_numpy(dtype=np.float32, copy=False)
        return X, Y, M

    def _get_session(self, sidx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._sessions_X[sidx] is not None:
            return (
                self._sessions_X[sidx],
                self._sessions_Y[sidx],
                self._sessions_M[sidx],
            )
        # lazy load 路径
        return self._load_one(self.sym_dates[sidx])

    def __len__(self) -> int:
        return self._total

    def _idx_to_sess_t(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(f"idx {idx} out of range [0, {self._total})")
        # 找到第一个使得 _cum[s] > idx 的 s
        sidx = int(np.searchsorted(self._cum, idx, side="right"))
        prev = int(self._cum[sidx - 1]) if sidx > 0 else 0
        local = idx - prev  # 第 local 个有效 t（从 0 起）
        t = (self.window - 1) + local
        return sidx, t

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sidx, t = self._idx_to_sess_t(idx)
        X, Y, M = self._get_session(sidx)

        # 过去 window 个 tick：行索引 [t-window+1, t+1)
        x_slice = X[t - self.window + 1 : t + 1]  # (window, F)
        y = Y[t]  # (L,)

        sample: Dict[str, object] = {
            "x": torch.from_numpy(np.ascontiguousarray(x_slice)),
            "y": torch.from_numpy(y.astype(np.int64)),
        }

        if self.return_midprices:
            mid_t = float(M[t])
            future_idx = np.array(
                [t + h for h in self.horizons], dtype=np.int64
            )
            mid_tn = M[future_idx].astype(np.float32)
            sample["midprice_t"] = torch.tensor(mid_t, dtype=torch.float32)
            sample["midprice_tn"] = torch.from_numpy(mid_tn)

        if self.return_meta:
            sym, date, session = self.sym_dates[sidx]
            sample["meta"] = {
                "sym": int(sym),
                "date": int(date),
                "session": str(session),
                "t": int(t),
            }

        return sample


# ---- 列名分组工具：方便上层挑选 feature_cols ----------------------------------

def get_default_feature_cols() -> List[str]:
    """所有数值特征列（排除 date/sym/time/midprice/label_*）。共 154 列。"""
    return _build_feature_cols()


def _build_feature_cols() -> List[str]:
    excluded = {"date", "sym", "time", "midprice"} | {f"label_{h}" for h in (5, 10, 20, 40, 60)}
    # 通过读一个 parquet 拿全列名
    df = pd.read_parquet(_pick_any_parquet(), columns=None)
    return [c for c in df.columns if c not in excluded]


def _pick_any_parquet() -> str:
    candidates = [
        os.path.join("data", "snapshot_sym0_date0_am.parquet"),
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "snapshot_sym0_date0_am.parquet",
        ),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "找不到任何 data/snapshot_*.parquet，无法推断默认特征列。"
    )
