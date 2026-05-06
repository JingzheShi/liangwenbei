"""T32 DeepLOB single-head h_60 with strong regularization.

Differences vs. T19:
  - SINGLE head (h_60 only) — avoids multi-task overfit on h_10
  - conv_dropout 0.4 (was 0.3 in T19)
  - fc_dropout 0.5 (same)
  - GroupNorm (replaces BatchNorm, robust to OOD held-out sym; LayerNorm-like)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(num_channels: int) -> nn.GroupNorm:
    for g in (8, 4, 2, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=num_channels)
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class _DeepLOBBackbone(nn.Module):
    def __init__(self, conv_dropout: float = 0.4) -> None:
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


class DeepLOB_H60(nn.Module):
    """DeepLOB single-head for h_60 (3-class)."""

    def __init__(
        self,
        seq_len: int = 100,
        num_features: int = 154,
        conv_dropout: float = 0.4,
        fc_dropout: float = 0.5,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_features = int(num_features)
        self.conv_dropout = float(conv_dropout)
        self.fc_dropout = float(fc_dropout)
        self.hidden_dim = int(hidden_dim)

        self.backbone = _DeepLOBBackbone(conv_dropout=conv_dropout)
        self.fc_hidden = nn.LazyLinear(hidden_dim)
        self.fc_drop = nn.Dropout(p=fc_dropout)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = torch.flatten(h, 1)
        h = self.fc_hidden(h)
        h = torch.nn.functional.leaky_relu(h, 0.01)
        h = self.fc_drop(h)
        return self.head(h)
