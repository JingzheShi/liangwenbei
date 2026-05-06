"""数据划分。

提供两套 split：
  - 'local_debug'：train/val/test 三段，本地评估用
  - 'submission'：train/val 两段，最大化训练数据
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

SymDate = Tuple[int, int, str]  # (sym, date, session)

N_SYMS = 5
N_DATES = 120
SESSIONS = ("am", "pm")


def _all_keys() -> List[SymDate]:
    """返回完整的 (sym, date, session) 列表，按 (sym, date, session) 排序。"""
    return [
        (sym, date, sess)
        for sym in range(N_SYMS)
        for date in range(N_DATES)
        for sess in SESSIONS
    ]


def _filter_dates(keys: List[SymDate], lo: int, hi: int) -> List[SymDate]:
    """date 范围 [lo, hi)。"""
    return [k for k in keys if lo <= k[1] < hi]


def get_split(mode: str = "local_debug") -> Dict[str, List[SymDate]]:
    """构造划分。

    mode='local_debug':
        train: date 0..79      (80 天)
        val:   date 80..95     (16 天)
        test:  date 96..119    (24 天)
    mode='submission':
        train: date 0..95      (96 天)
        val:   date 96..119    (24 天)   ← 等同 local_debug 的 test
    mode='rolling_val':
        TODO: 实现按时间滚动的多折验证（参见 README，目前花儿）
    """
    keys = _all_keys()
    if mode == "local_debug":
        return {
            "train": _filter_dates(keys, 0, 80),
            "val": _filter_dates(keys, 80, 96),
            "test": _filter_dates(keys, 96, 120),
        }
    if mode == "submission":
        return {
            "train": _filter_dates(keys, 0, 96),
            "val": _filter_dates(keys, 96, 120),
        }
    if mode == "rolling_val":
        raise NotImplementedError(
            "rolling_val 划分尚未实现。预期接口：返回 list[dict]，每个元素是一折 train/val。"
        )
    raise ValueError(f"未知 split mode: {mode!r}")


def get_file_path(sym: int, date: int, session: str, data_dir: str = "data") -> str:
    """根据三元组返回 parquet 文件路径。"""
    if session not in SESSIONS:
        raise ValueError(f"session 必须是 'am' 或 'pm'，得到 {session!r}")
    if not 0 <= sym < N_SYMS:
        raise ValueError(f"sym 必须在 [0,{N_SYMS}) 内，得到 {sym}")
    if not 0 <= date < N_DATES:
        raise ValueError(f"date 必须在 [0,{N_DATES}) 内，得到 {date}")
    return os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{session}.parquet")


if __name__ == "__main__":
    for mode in ("local_debug", "submission"):
        s = get_split(mode)
        sizes = {k: len(v) for k, v in s.items()}
        print(f"{mode}: {sizes}, total={sum(sizes.values())}")
    print("sample path:", get_file_path(0, 0, "am"))
