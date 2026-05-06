---
name: TLOB / MLPLOB — Dual Attention vs simple MLP for LOB (Berti 2025 v3)
description: TLOB Dual Attention transformer 达 SOTA on FI-2010 (+3.7 F1); 同 paper 的 MLPLOB 纯 MLP 也达 SOTA。简单架构胜出。
type: project
arxiv: 2502.15757v3
date: 2025-05
authors: Leonardo Berti et al.
priority: ★★★ — NN baseline 候选；**先试 MLPLOB**
---

# TLOB / MLPLOB

## TL;DR

**同一篇 paper 提出两个模型**：
1. **MLPLOB** — 纯 MLP（无 attention，无 conv）；**FI-2010 / Tesla / Intel / 2023-Bitcoin 全面达 SOTA**
2. **TLOB** — Dual-Attention Transformer（Temporal Self-Attention × Feature Self-Attention 因子化）；FI-2010 +3.7 F1 over BiNCTABL/DeepLOB

**关键意义**：MLPLOB 颠覆 "LOB 必须 conv+LSTM" 的传统认知。**简单架构在恰当训练下可达 SOTA**。

## 1. 模型架构

### 1.1 MLPLOB

```
Input: shape [T, F]  (T = sequence len, F = LOB features)
  → Flatten to [T·F]
  → MLP (3-5 layers, hidden 256-512, GELU)
  → Output: [3] for {down, flat, up}
```

参数：~500 K，**比 DeepLOB 小 5-10 倍**。

### 1.2 TLOB Dual Attention

不是普通 self-attn 在 [T·F] 上做（O((TF)²)），而是 **factorize**：

```
Block:
  x = x + TemporalSelfAttn(x)    # attention over T axis (per feature)
  x = x + FeatureSelfAttn(x)     # attention over F axis (per timestep)
  x = x + FFN(x)
```

复杂度 O(T²·F + F²·T) << O((T·F)²).

## 2. 关键实验

| Model | FI-2010 F1 | Tesla F1 | Intel F1 | 2023-BTC F1 |
|---|---|---|---|---|
| DeepLOB | baseline | baseline | baseline | baseline |
| BiNCTABL | +1.5 | +0.8 | +1.2 | +1.0 |
| **MLPLOB** | **SOTA** | **+1.3** | **+7.7** | SOTA |
| **TLOB** | **+3.7** | +1.3 | +7.7 | +leading |

**MLPLOB 与 TLOB 收敛速度都比 DeepLOB / BiNCTABL 快 ≥ 2×**。

## 3. 我们的项目相关性

### 3.1 如果 GBDT 路线触底，要上 NN

**先试 MLPLOB（不是 TLOB）**：

- 实现简单（~50 行 PyTorch）
- 训练快（~2 hours on V100）
- 参数少 → 不易过拟合 5-sym 数据

只有 MLPLOB **超过** LightGBM 5-seed ensemble，才有理由升级到 TLOB。

### 3.2 与硬约束兼容性

| 硬约束 | MLPLOB | TLOB |
|---|---|---|
| sym-agnostic（无 sym embedding）| ✅ 易实现 | ✅ 易实现 |
| stateless predict | ✅ pure forward | ✅ pure forward |
| date 置 0 | ✅ | ✅ |

## 4. 实施模板（MLPLOB minimal）

```python
class MLPLOB(nn.Module):
    def __init__(self, T, F, n_classes=3, hidden=512, n_layers=4):
        super().__init__()
        layers = []
        in_dim = T * F
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1)]
            in_dim = hidden
        layers += [nn.Linear(hidden, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, T, F]
        return self.net(x.flatten(1))
```

训练：AdamW(lr=1e-3, weight_decay=1e-4), CosineLRScheduler, 20-50 epochs。

**如果用 R10 S2 PnL custom loss 替代 CE，可叠加增益**。

## 5. 与 multi-horizon 的结合

- 对 h_10 / h_60 各训一个 MLPLOB
- 或共享 backbone + 2 个 head（multi-task）
- 或单 model 输出 [3 × n_horizon] = 6 logits

## 6. 风险

- 5-sym 数据量 vs FI-2010（5 stock × 10 day × ~5M ticks ≈ 250M）：我们数据可能少 5-10 ×；过拟合风险高 → 必须强 regularization (dropout 0.2+, weight decay 1e-3+)
- T·F flatten 后维度 = 100×40 = 4000，hidden 512 → 第一层 ≈ 2M params；占整模型大头；reduce hidden 到 256 或 T 到 50

## 7. 来源

- arxiv: https://arxiv.org/abs/2502.15757
- code: https://github.com/LeonardoBerti00/TLOB
- v3 html: https://arxiv.org/html/2502.15757v3

## 8. 备注

- 我们已有 `papers/model_lob_benchmark_briola_2023.md` 覆盖 Briola 2023 benchmark；本文是 Berti 2025 后续工作（同实验室）。
- TLOB 详细 dual-attention factorization 与 Spacetimeformer-LOB（已 R20 覆盖）思想类似。
