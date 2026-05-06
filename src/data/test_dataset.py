"""LOBDataset / split / get_file_path 的单元测试。

直接运行：python src/data/test_dataset.py
全部断言通过即认为数据基础设施正确。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import LOBDataset, get_default_feature_cols
from src.data.split import get_file_path, get_split


def test_split_shapes() -> None:
    s = get_split("local_debug")
    assert sorted(s) == ["test", "train", "val"]
    assert len(s["train"]) == 5 * 80 * 2  # 800
    assert len(s["val"]) == 5 * 16 * 2  # 160
    assert len(s["test"]) == 5 * 24 * 2  # 240
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 1200

    s2 = get_split("submission")
    assert sorted(s2) == ["train", "val"]
    assert len(s2["train"]) == 5 * 96 * 2  # 960
    assert len(s2["val"]) == 5 * 24 * 2  # 240

    # disjoint train/val/test in local_debug
    train_dates = {(k[0], k[1]) for k in s["train"]}
    val_dates = {(k[0], k[1]) for k in s["val"]}
    test_dates = {(k[0], k[1]) for k in s["test"]}
    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)

    # Each (sym, date) covered exactly once across the three sets
    assert len(train_dates) + len(val_dates) + len(test_dates) == 5 * 120
    print("[ok] test_split_shapes")


def test_get_file_path_exists() -> None:
    s = get_split("local_debug")
    # spot-check 4 files
    for k in [s["train"][0], s["val"][0], s["test"][0], s["train"][-1]]:
        path = get_file_path(*k)
        assert os.path.exists(path), f"missing: {path}"
    print("[ok] test_get_file_path_exists")


def _tiny_dataset(split_keys, **kw) -> LOBDataset:
    fcols = get_default_feature_cols()
    return LOBDataset(
        sym_dates=split_keys,
        feature_cols=fcols,
        window=100,
        max_horizon=60,
        cache_in_memory=True,
        **kw,
    )


def test_len_and_shape() -> None:
    s = get_split("local_debug")
    ds = _tiny_dataset(s["train"][:4])
    # 每个 session 2001 ticks, valid t ∈ [99, 1940] → 1842 个样本
    assert len(ds) == 1842 * 4
    sample = ds[0]
    assert sample["x"].shape == (100, 154)
    assert sample["x"].dtype == torch.float32
    assert sample["y"].shape == (5,)
    assert sample["y"].dtype == torch.int64
    print("[ok] test_len_and_shape")


def test_first_and_last_index() -> None:
    s = get_split("local_debug")
    ds = _tiny_dataset(s["train"][:3], return_meta=True)
    s0 = ds[0]
    s_last = ds[len(ds) - 1]
    s_neg1 = ds[-1]
    assert s0["x"].shape == (100, 154)
    assert s_last["x"].shape == (100, 154)
    # 负索引和正索引一致
    assert torch.equal(s_last["x"], s_neg1["x"])
    assert s_last["meta"] == s_neg1["meta"]
    # 第一个样本应在第 0 个 session 的 t=99
    assert s0["meta"]["t"] == 99
    # 最后一个样本应在最后一个 session 的 t=1940
    assert s_last["meta"]["t"] == 2001 - 1 - 60
    print("[ok] test_first_and_last_index")


def test_no_cross_session_leak() -> None:
    """边界检查：跨 session 不应复用数据。"""
    s = get_split("local_debug")
    ds = _tiny_dataset(s["train"][:3], return_meta=True)
    per_sess = 1842
    # idx = per_sess - 1 → session 0 的最后一个样本
    s_end = ds[per_sess - 1]
    # idx = per_sess → session 1 的第一个样本
    s_next = ds[per_sess]
    assert s_end["meta"]["session"] in ("am", "pm")
    assert (
        s_end["meta"]["sym"] != s_next["meta"]["sym"]
        or s_end["meta"]["date"] != s_next["meta"]["date"]
        or s_end["meta"]["session"] != s_next["meta"]["session"]
    )
    assert s_next["meta"]["t"] == 99  # 新 session 必须从 t=99 开始
    print("[ok] test_no_cross_session_leak")


def test_window_slice_consistency() -> None:
    """在同一 session 内连续两个样本，window 应 shift by 1。"""
    s = get_split("local_debug")
    ds = _tiny_dataset(s["train"][:1], return_meta=True)
    s0 = ds[0]
    s1 = ds[1]
    # s0 covers ticks [0..99]; s1 covers ticks [1..100]
    # → s0['x'][1:] == s1['x'][:-1]
    assert torch.equal(s0["x"][1:], s1["x"][:-1])
    print("[ok] test_window_slice_consistency")


def test_midprice_consistency() -> None:
    """midprice_t 必须等于该 t 行的真实 midprice (从 parquet 直接读对照)。"""
    import pandas as pd
    s = get_split("local_debug")
    ds = _tiny_dataset(s["train"][:1], return_midprices=True, return_meta=True)
    s10 = ds[10]
    sym, date, session = ds.sym_dates[0]
    df = pd.read_parquet(get_file_path(sym, date, session))
    t = s10["meta"]["t"]
    assert abs(s10["midprice_t"].item() - df["midprice"].iloc[t]) < 1e-7
    # midprice_tn[i] 应对应 midprice.iloc[t + horizons[i]]
    horizons = [5, 10, 20, 40, 60]
    for i, h in enumerate(horizons):
        assert abs(s10["midprice_tn"][i].item() - df["midprice"].iloc[t + h]) < 1e-7
    print("[ok] test_midprice_consistency")


def test_dataloader_smoke() -> None:
    s = get_split("local_debug")
    ds = _tiny_dataset(s["train"][:2])
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["x"].shape == (8, 100, 154)
    assert batch["y"].shape == (8, 5)
    print("[ok] test_dataloader_smoke")


def test_no_overlap_train_val_test_files() -> None:
    s = get_split("local_debug")
    train_paths = {get_file_path(*k) for k in s["train"]}
    val_paths = {get_file_path(*k) for k in s["val"]}
    test_paths = {get_file_path(*k) for k in s["test"]}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    print("[ok] test_no_overlap_train_val_test_files")


def test_invalid_inputs() -> None:
    try:
        get_file_path(0, 0, "evening")
    except ValueError:
        pass
    else:
        raise AssertionError("session 校验未触发")
    try:
        get_split("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("mode 校验未触发")
    try:
        get_split("rolling_val")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("rolling_val 应抛 NotImplementedError")
    print("[ok] test_invalid_inputs")


if __name__ == "__main__":
    test_split_shapes()
    test_get_file_path_exists()
    test_len_and_shape()
    test_first_and_last_index()
    test_no_cross_session_leak()
    test_window_slice_consistency()
    test_midprice_consistency()
    test_dataloader_smoke()
    test_no_overlap_train_val_test_files()
    test_invalid_inputs()
    print("\n=== ALL TESTS PASSED ===")
