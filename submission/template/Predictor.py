"""良文杯提交模板 — 在此基础上修改

关键约定（详见 ../RULES.md §2）：
- __init__ 无参数；加载模型 + 初始化设备
- predict(x: List[pd.DataFrame]) -> List[List[int]]
  - x 长度 = config.batch（最后一批可能不足，要动态处理）
  - 每个 DataFrame: 100 行 × len(config.feature) 列，列名严格 = config.feature
  - 返回外层 len(x)，内层 len(config.label)，元素 ∈ {0, 1, 2}
- 模型路径用 os.path.join(os.path.dirname(__file__), 'model.pth')，**不要用相对路径**
- GPU 模式: self.model.to(device); 输入 .to(device); 输出 .cpu().numpy().tolist()

CPU 评测优化（见 ../ENV_NOTES.md §4）：
若使用 CPU 评测且想限制线程数，把下面 3 行 uncomment 并放到 import torch 之前：
"""
# import os
# os.environ.setdefault("OMP_NUM_THREADS", "8")
# os.environ.setdefault("MKL_NUM_THREADS", "8")

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import torch

# 同包导入（平台按 from <pkg>.Predictor import Predictor 调用）。
# 直接 `python Predictor.py` 跑会因为 `from .model import` 报错——这是预期行为。
try:
    from .model import YourModel  # noqa: F401  TODO: 改成你的模型类名
except ImportError:
    from model import YourModel  # noqa: F401  fallback for direct exec


class Predictor:
    def __init__(self) -> None:
        # CPU 评测需限制线程数时（见 ../ENV_NOTES.md §4）：
        # torch.set_num_threads(8)
        # torch.set_num_interop_threads(2)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pth_path = os.path.join(os.path.dirname(__file__), "model.pth")
        # TODO: 实例化你的模型；构造参数从 ckpt 的 meta 里读
        # ckpt = torch.load(pth_path, map_location=self.device, weights_only=False)
        # self.model = YourModel(**ckpt["meta"])
        # self.model.load_state_dict(ckpt["model_state"])
        # self.model.to(self.device)
        # self.model.eval()
        raise NotImplementedError("TODO: 在 __init__ 里加载模型")

    def predict(self, x: List[pd.DataFrame]) -> List[List[int]]:
        """官方协议见 ../RULES.md §2.2。

        典型实现：
            arrs = [df.to_numpy(dtype=np.float32, copy=False) for df in x]
            batch = np.stack(arrs, axis=0)                      # (B, 100, D)
            t = torch.from_numpy(batch).to(self.device)         # (B, 100, D)
            with torch.no_grad():
                logits = self.model(t)  # tuple of K tensors, each (B, n_classes)
            preds = np.stack([h.argmax(1).cpu().numpy() for h in logits], axis=1)  # (B, K)
            return preds.astype(int).tolist()
        """
        raise NotImplementedError("TODO: 实现 predict")
