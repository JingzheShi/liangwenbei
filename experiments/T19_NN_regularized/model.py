"""DeepLOB with strong regularization for T19.

Changes vs. examples/mmpc_demo/model.py DeepLOB:
  - BatchNorm2d -> GroupNorm (more robust to small batch / OOD held-out sym)
  - Dropout2d(p=0.3) after each conv block
  - Dropout(p=0.5) after fc_hidden
  - Same multi-task head structure (5 heads, 3 classes each)

The intent is to keep the inductive bias of DeepLOB (LOB-aware conv stack)
while preventing the rapid epoch-1 overfit observed in T1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(num_channels: int) -> nn.GroupNorm:
    # 8 groups by default, fall back to 4/2/1 if num_channels not divisible.
    for g in (8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class _DeepLOBBackboneReg(nn.Module):
    def __init__(self, conv_dropout: float = 0.3) -> None:
        super().__init__()
        p = float(conv_dropout)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            _gn(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            _gn(32),
            nn.Conv2d(32, 32, kernel_size=(5, 1), stride=(2, 1)),
            nn.LeakyReLU(0.01),
            _gn(32),
            nn.Dropout2d(p=p),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            _gn(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            _gn(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(2, 1)),
            nn.Tanh(),
            _gn(32),
            nn.Dropout2d(p=p),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 8), padding="same"),
            nn.LeakyReLU(0.01),
            _gn(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            _gn(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), stride=(2, 1)),
            nn.LeakyReLU(0.01),
            _gn(32),
            nn.Dropout2d(p=p),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            _gn(64),
            nn.Conv2d(64, 16, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            _gn(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            _gn(64),
            nn.Conv2d(64, 16, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            _gn(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 16, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            _gn(16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        a = self.inp1(x)
        b = self.inp2(x)
        c = self.inp3(x)
        return torch.cat((a, b, c), dim=1)


class DeepLOBReg(nn.Module):
    """DeepLOB with Dropout + GroupNorm regularization."""

    def __init__(
        self,
        num_classes_per_head: list[int],
        seq_len: int = 100,
        num_features: int = 154,
        conv_dropout: float = 0.3,
        fc_dropout: float = 0.5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if not num_classes_per_head:
            raise ValueError("num_classes_per_head must be non-empty")
        self.num_classes_per_head = list(num_classes_per_head)
        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.conv_dropout = float(conv_dropout)
        self.fc_dropout = float(fc_dropout)
        self.hidden_dim = int(hidden_dim)

        self.backbone = _DeepLOBBackboneReg(conv_dropout=conv_dropout)
        self.fc_hidden = nn.LazyLinear(hidden_dim)
        self.fc_drop = nn.Dropout(p=fc_dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, c) for c in self.num_classes_per_head])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.backbone(x)
        h = torch.flatten(h, 1)
        h = self.fc_hidden(h)
        h = torch.nn.functional.leaky_relu(h, 0.01)
        h = self.fc_drop(h)
        return tuple(head(h) for head in self.heads)
