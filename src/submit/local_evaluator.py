"""本地伪评测器：在受控环境模拟"良文杯"平台对提交包的调用方式。

用途：在本地把"提交包 → 平台调用 → 收集预测"整个链路跑通，
用于在真正提交前验证 Predictor 行为符合官方约定。

平台行为（PDF 2.4）：
  1. 读 <submit>/config.json 拿 python_version, batch, feature, label
  2. pip install -r requirements.txt
  3. from <submit>.Predictor import Predictor; p = Predictor()
  4. 多次调用 p.predict(x: List[pd.DataFrame]) -> List[List[int]]
     - 每次调用 x 长度 = batch
     - 每个 DataFrame：100 行 × len(feature) 列，列名严格按 config["feature"]
     - 返回长度 batch 的 list，每内层 list 长度 = len(label)，元素 ∈ {0,1,2}
  5. 评测点顺序被打乱；date 字段被置 0；sym ∈ {0..4}（可能含训练外股票）；time 保留
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements
    def tqdm(x, **kwargs):
        return x


SESSION_LEN = 2001
SLIDE_WINDOW = 100
DEFAULT_HORIZONS = (5, 10, 20, 40, 60)


@dataclass
class TestSession:
    """一段完整的测试 session（对应一个 parquet 文件）。

    full_df 包含原始全部列：date, sym, time, <features>, midprice, label_*
    sym/date/session 仅作为元信息记录，不进入 predictor 输入。
    """
    sym: int
    date: int
    session: str  # 'am' 或 'pm'
    full_df: pd.DataFrame


@dataclass
class EvalPoint:
    """单个评测点：在 session 中以 t 为窗口末端。"""
    session_idx: int  # TestSession 在 list 中的下标
    t: int            # 窗口末端 row index（窗口 = [t-99, t]）


@dataclass
class EvalResult:
    """LocalEvaluator.run() 的返回值。"""
    config: dict
    label_cols: List[str]
    feature_cols: List[str]
    horizons: List[int]
    # predictions[i, k] = 第 i 个评测点在第 k 个 head 上的预测类（0/1/2）
    predictions: np.ndarray  # shape (N, K), int
    true_labels: np.ndarray  # shape (N, K), int (NaN 标签处填 -1)
    # midprice_t / midprice_tn[k] = 评测点处的当前/未来 midprice
    midprice_t: np.ndarray   # shape (N,)
    midprice_tn: np.ndarray  # shape (N, K)
    # 元信息（每个评测点的来源）
    meta_sym: np.ndarray     # shape (N,)
    meta_date: np.ndarray    # shape (N,)
    meta_session: np.ndarray # shape (N,) dtype='<U2'
    meta_t: np.ndarray       # shape (N,)
    timing: dict = field(default_factory=dict)  # 'predict_total_s', 'num_batches', ...

    @property
    def num_points(self) -> int:
        return int(self.predictions.shape[0])


def _import_predictor(submit_dir: str):
    """把 submit_dir 当作一个 python 包导入，返回 Predictor 类。

    模拟官方调用：from <submit>.Predictor import Predictor
    submit_dir 的 basename 即包名。
    """
    submit_dir = os.path.abspath(submit_dir)
    parent = os.path.dirname(submit_dir)
    pkg_name = os.path.basename(submit_dir)

    if not os.path.isdir(submit_dir):
        raise FileNotFoundError(f"submit_dir 不存在或不是目录: {submit_dir}")
    init_py = os.path.join(submit_dir, "__init__.py")
    created_init = False
    if not os.path.isfile(init_py):
        # 平台默认按包加载；mmpc_demo 缺 __init__.py 时给临时补一个，
        # 这样 `from .model import ...` 这种相对导入也能跑。
        with open(init_py, "w", encoding="utf-8") as fh:
            fh.write("")
        created_init = True

    if parent not in sys.path:
        sys.path.insert(0, parent)

    # 清理可能存在的旧 import 缓存（例如重复 evaluate 不同 submit）。
    for mod_name in [pkg_name, f"{pkg_name}.Predictor", f"{pkg_name}.model"]:
        sys.modules.pop(mod_name, None)
    # 同时清掉裸名 'model'，避免 mmpc_demo 那种 try-from-top-level 落入旧模块。
    if "model" in sys.modules:
        # 只在它来自其他 submit_dir 时清掉，免得误伤本地 model 模块。
        mod = sys.modules["model"]
        mod_file = getattr(mod, "__file__", "") or ""
        if mod_file and submit_dir not in os.path.abspath(mod_file):
            sys.modules.pop("model", None)

    mod = importlib.import_module(f"{pkg_name}.Predictor")
    return mod.Predictor, created_init


class LocalEvaluator:
    """模拟官方评测平台的行为，确保提交包能在受控环境跑通。"""

    def __init__(
        self,
        submit_dir: str,
        test_data: List[TestSession],
        shuffle: bool = True,
        anonymize: bool = True,
        seed: int = 42,
        slide_window: int = SLIDE_WINDOW,
        horizons: Iterable[int] = DEFAULT_HORIZONS,
        progress: bool = True,
    ) -> None:
        self.submit_dir = os.path.abspath(submit_dir)
        self.test_data = list(test_data)
        self.shuffle = bool(shuffle)
        self.anonymize = bool(anonymize)
        self.seed = int(seed)
        self.slide_window = int(slide_window)
        self.horizons = list(horizons)
        self.progress = bool(progress)

        # 1) 读 config
        cfg_path = os.path.join(self.submit_dir, "config.json")
        with open(cfg_path, encoding="utf-8") as fh:
            self.config = json.load(fh)
        for required in ("python_version", "batch", "feature", "label"):
            if required not in self.config:
                raise ValueError(f"config.json 缺字段: {required}")
        self.batch_size = int(self.config["batch"])
        self.feature_cols: List[str] = list(self.config["feature"])
        self.label_cols: List[str] = list(self.config["label"])
        if self.batch_size <= 0:
            raise ValueError(f"batch 必须 > 0，得到 {self.batch_size}")

        # 2) import Predictor 类（按包方式，模拟平台）
        self.Predictor, self._created_init = _import_predictor(self.submit_dir)

        # 3) 实例化
        self.predictor = self.Predictor()

        # 4) 构造评测点（所有 session × 所有有效 t），可选 shuffle
        self.eval_points = self._build_eval_points()
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            order = rng.permutation(len(self.eval_points))
            self.eval_points = [self.eval_points[i] for i in order]

    # --------------------------------------------------------------- helpers

    def _build_eval_points(self) -> List[EvalPoint]:
        max_h = max(self.horizons) if self.horizons else 0
        points: List[EvalPoint] = []
        for si, sess in enumerate(self.test_data):
            n = len(sess.full_df)
            t_lo = self.slide_window - 1            # 窗口末端最小 = 99
            t_hi = n - max_h - 1                    # t + max_h 必须仍 < n
            for t in range(t_lo, t_hi + 1):
                points.append(EvalPoint(session_idx=si, t=t))
        return points

    def _make_window_df(self, point: EvalPoint) -> pd.DataFrame:
        """按 config["feature"] 列名构造 100×D 的输入 DataFrame。

        列名严格 = config["feature"]，顺序也与 config 一致（平台保证）。
        anonymize=True 时把 'date' 列置 0（仅当 date 在 feature 列表里——
        mmpc_demo / example_official 都不在；保留逻辑以备后用）。
        """
        sess = self.test_data[point.session_idx]
        t = point.t
        win = sess.full_df.iloc[t - self.slide_window + 1 : t + 1]
        out = win.loc[:, self.feature_cols].copy()
        # 类型统一成 float32：避免 object dtype 列被 to_numpy 时炸
        out = out.astype(np.float32, copy=False)
        if self.anonymize and "date" in out.columns:
            out["date"] = 0
        # index 重置，避免 100 个 row index 是原始位置导致下游困惑
        out.reset_index(drop=True, inplace=True)
        return out

    # ------------------------------------------------------------------ run

    def run(self) -> EvalResult:
        N = len(self.eval_points)
        K = len(self.label_cols)
        H = self.horizons
        if K == 0:
            raise ValueError("config.label 不能为空")

        predictions = np.full((N, K), -1, dtype=np.int64)
        true_labels = np.full((N, K), -1, dtype=np.int64)
        midprice_t = np.full((N,), np.nan, dtype=np.float64)
        midprice_tn = np.full((N, K), np.nan, dtype=np.float64)
        meta_sym = np.zeros((N,), dtype=np.int64)
        meta_date = np.zeros((N,), dtype=np.int64)
        meta_session = np.empty((N,), dtype="<U2")
        meta_t = np.zeros((N,), dtype=np.int64)

        # 预先把每个 session 的 midprice / labels 抽成 numpy，便于按 t/h 取
        per_session_mid: List[np.ndarray] = []
        per_session_labels: List[np.ndarray] = []
        for sess in self.test_data:
            per_session_mid.append(sess.full_df["midprice"].to_numpy(dtype=np.float64))
            mat = np.full((len(sess.full_df), K), -1, dtype=np.int64)
            for ki, lc in enumerate(self.label_cols):
                if lc in sess.full_df.columns:
                    col = sess.full_df[lc].to_numpy()
                    valid = ~pd.isna(col)
                    mat[valid, ki] = col[valid].astype(np.int64)
            per_session_labels.append(mat)

        # 把 label_cols 映射到 horizon 用于取 midprice_tn —— label_5 → 5 等。
        label_to_h: List[Optional[int]] = []
        for lc in self.label_cols:
            try:
                label_to_h.append(int(lc.split("_")[-1]))
            except Exception:
                label_to_h.append(None)

        n_batches = (N + self.batch_size - 1) // self.batch_size
        predict_total = 0.0
        first_batch_t: Optional[float] = None

        iterator = range(n_batches)
        if self.progress:
            iterator = tqdm(iterator, desc=f"eval batches (B={self.batch_size})", total=n_batches)

        for bi in iterator:
            lo = bi * self.batch_size
            hi = min(lo + self.batch_size, N)
            batch_points = self.eval_points[lo:hi]
            # 注意：最后一个 batch 可能不足 self.batch_size。平台在严格批拼接时
            # 一般会保证整除；这里我们把末批按实际长度送入，predictor 不应硬编码 B。
            batch_dfs = [self._make_window_df(p) for p in batch_points]

            t0 = time.perf_counter()
            preds = self.predictor.predict(batch_dfs)
            dt = time.perf_counter() - t0
            predict_total += dt
            if first_batch_t is None:
                first_batch_t = dt

            # 形状校验
            if not isinstance(preds, list) or len(preds) != len(batch_points):
                raise RuntimeError(
                    f"predict 返回长度不匹配：期望 {len(batch_points)}，"
                    f"实际 {len(preds) if hasattr(preds, '__len__') else type(preds)}"
                )
            for j, p in enumerate(preds):
                if not isinstance(p, (list, tuple)) or len(p) != K:
                    raise RuntimeError(
                        f"predict 返回第 {j} 个内层长度不匹配：期望 {K}，实际 {p}"
                    )
                for v in p:
                    iv = int(v)
                    if iv not in (0, 1, 2):
                        raise RuntimeError(f"predict 返回非法类标 {v}（必须 ∈ {{0,1,2}}）")
                predictions[lo + j] = [int(v) for v in p]

            # 收集 ground-truth、midprice、meta
            for j, point in enumerate(batch_points):
                idx = lo + j
                sess = self.test_data[point.session_idx]
                t = point.t
                meta_sym[idx] = sess.sym
                meta_date[idx] = sess.date
                meta_session[idx] = sess.session
                meta_t[idx] = t
                midprice_t[idx] = per_session_mid[point.session_idx][t]
                for ki, h in enumerate(label_to_h):
                    if h is not None and t + h < len(per_session_mid[point.session_idx]):
                        midprice_tn[idx, ki] = per_session_mid[point.session_idx][t + h]
                true_labels[idx] = per_session_labels[point.session_idx][t]

        timing = {
            "predict_total_s": predict_total,
            "num_batches": n_batches,
            "num_points": N,
            "batch_size": self.batch_size,
            "avg_batch_s": predict_total / max(n_batches, 1),
            "first_batch_s": first_batch_t,
            "throughput_points_per_s": N / predict_total if predict_total > 0 else float("nan"),
        }

        return EvalResult(
            config=self.config,
            label_cols=self.label_cols,
            feature_cols=self.feature_cols,
            horizons=H,
            predictions=predictions,
            true_labels=true_labels,
            midprice_t=midprice_t,
            midprice_tn=midprice_tn,
            meta_sym=meta_sym,
            meta_date=meta_date,
            meta_session=meta_session,
            meta_t=meta_t,
            timing=timing,
        )

    # ----------------------------------------------------------------- util

    def cleanup(self) -> None:
        """如果我们临时为 submit_dir 添加了 __init__.py，运行结束后清掉。"""
        if self._created_init:
            init_py = os.path.join(self.submit_dir, "__init__.py")
            try:
                os.remove(init_py)
            except OSError:
                pass


# ---------------------------------------------------------------- 数据加载

def load_test_sessions(
    spec: List[Tuple[int, int, str]],
    data_dir: str = "data",
) -> List[TestSession]:
    """根据 (sym, date, session) 列表加载 parquet 形成 TestSession 列表。"""
    out: List[TestSession] = []
    for sym, date, sess in spec:
        path = os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到测试数据: {path}")
        df = pd.read_parquet(path)
        out.append(TestSession(sym=int(sym), date=int(date), session=sess, full_df=df))
    return out
