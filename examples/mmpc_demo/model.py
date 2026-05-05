"""Model zoo for multi-task DeepLOB training."""

from __future__ import annotations

import torch
import torch.nn as nn


class _DeepLOBBackbone(nn.Module):
    """DeepLOB-style CNN stack."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1), stride=(2, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), stride=(2, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 8), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), stride=(2, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)
        return torch.cat((x_inp1, x_inp2, x_inp3), dim=1)


class DeepLOB(nn.Module):
    """Multi-task DeepLOB with one head per label column."""

    def __init__(
        self,
        num_classes_per_head: list[int],
        seq_len: int = 100,
        num_features: int = 8,
    ) -> None:
        super().__init__()
        if not num_classes_per_head:
            raise ValueError("num_classes_per_head must be non-empty")
        self.num_classes_per_head = list(num_classes_per_head)
        self.seq_len = int(seq_len)
        self.num_features = int(num_features)

        self.backbone = _DeepLOBBackbone()
        # Bind in_features from real tensor shape at first forward.
        self.fc_hidden = nn.LazyLinear(64)
        self.heads = nn.ModuleList([nn.Linear(64, c) for c in self.num_classes_per_head])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.backbone(x)
        h = torch.flatten(h, 1)
        h = self.fc_hidden(h)
        return tuple(head(h) for head in self.heads)


class DeepLOB_WideCNN(nn.Module):
    """Wider CNN variant."""

    def __init__(self, num_classes_per_head: list[int]):
        super().__init__()
        if not num_classes_per_head:
            raise ValueError("num_classes_per_head must be non-empty")
        self.num_classes_per_head = list(num_classes_per_head)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_hidden = nn.Linear(32, 64)
        self.heads = nn.ModuleList([nn.Linear(64, c) for c in self.num_classes_per_head])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.backbone(x)
        h = torch.flatten(h, 1)
        h = torch.relu(self.fc_hidden(h))
        return tuple(head(h) for head in self.heads)


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return torch.relu(x)


class DeepLOB_ResCNN(nn.Module):
    """Residual CNN variant."""

    def __init__(self, num_classes_per_head: list[int]):
        super().__init__()
        if not num_classes_per_head:
            raise ValueError("num_classes_per_head must be non-empty")
        self.num_classes_per_head = list(num_classes_per_head)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage = nn.Sequential(
            _ResBlock(16),
            _ResBlock(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_hidden = nn.Linear(24, 64)
        self.heads = nn.ModuleList([nn.Linear(64, c) for c in self.num_classes_per_head])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.stage(self.stem(x))
        h = torch.flatten(h, 1)
        h = torch.relu(self.fc_hidden(h))
        return tuple(head(h) for head in self.heads)


class DeepLOB_DropoutCNN(nn.Module):
    """CNN variant with dropout regularization."""

    def __init__(self, num_classes_per_head: list[int], dropout_p: float = 0.3):
        super().__init__()
        if not num_classes_per_head:
            raise ValueError("num_classes_per_head must be non-empty")
        self.num_classes_per_head = list(num_classes_per_head)
        p = float(dropout_p)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=p),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_hidden = nn.Sequential(nn.Linear(16, 64), nn.ReLU(inplace=True), nn.Dropout(p=p))
        self.heads = nn.ModuleList([nn.Linear(64, c) for c in self.num_classes_per_head])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.backbone(x)
        h = torch.flatten(h, 1)
        h = self.fc_hidden(h)
        return tuple(head(h) for head in self.heads)
