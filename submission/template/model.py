"""模型骨架 — 替换成你的实际模型。

要点：
- 输入 (B, 100, D) 或 (B, 1, 100, D)，看你模型的卷积 channel 习惯
- 输出 tuple of K tensors，每个 (B, n_classes_k)
- num_classes_per_head 一般是 [3, 3, 3, 3, 3]（5 个 horizon × 三分类）
- 必须是 FP32（不要 .half() / quantize）
"""
from __future__ import annotations

import torch
import torch.nn as nn


class YourModel(nn.Module):
    def __init__(
        self,
        num_classes_per_head: list[int],
        seq_len: int = 100,
        num_features: int = 12,
    ) -> None:
        super().__init__()
        self.num_classes_per_head = list(num_classes_per_head)
        self.seq_len = seq_len
        self.num_features = num_features

        # TODO: 替换成你的 backbone
        self.backbone = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.heads = nn.ModuleList([nn.Linear(32, c) for c in self.num_classes_per_head])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # x: (B, 100, D) → (B, D, 100) → conv1d
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.transpose(1, 2)
        h = self.backbone(x).flatten(1)
        return tuple(head(h) for head in self.heads)
