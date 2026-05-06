---
name: Better Inputs Matter More Than Stacking Another Hidden Layer (Wang 2025)
description: 在 crypto LOB 上证明 Savitzky-Golay 平滑 + XGBoost > DeepLOB；feature 质量 >> 网络深度
type: project
arxiv: 2506.05764v2
date: 2025-06
authors: Haochuan (Kevin) Wang (UChicago)
priority: ★★★★ — Savitzky-Golay 是 0.5 天小工具，可立刻部署
---

# Better Inputs Matter More Than Stacking Another Hidden Layer

## TL;DR

**核心论点**：在 short-horizon LOB prediction 中，**input quality / preprocessing > network depth**。
**实证**：Savitzky-Golay 平滑过的 LOB feature 上，XGBoost / Logistic 反而 **超过** DeepLOB 1-2 % 准确率。
**实操**：把所有 continuous LOB feature 用 SG filter 平滑一遍作为额外 feature。

## 1. 实验细节

### 1.1 模型集合

1. Logistic Regression（baseline）
2. XGBoost
3. CatBoost
4. CNN + CatBoost（hybrid）
5. CNN + XGBoost
6. CNN + LSTM (single layer)
7. **DeepLOB (3 Conv2D + LSTM)** — 经典深网

### 1.2 三种平滑

| 方法 | 结果 |
|---|---|
| **Savitzky-Golay** | ✅ 全模型受益 |
| Kalman filter | ❌ 参数过严，不灵活 |
| 无平滑 baseline | (低于上述两者) |

### 1.3 关键 ternary classification 准确率（含 SG）

| Setting | Accuracy 范围 |
|---|---|
| **with SG, all models** | **0.42–0.71** |
| no smoothing | 0.40–0.65 |
| with Kalman | 0.41–0.68 |

**Binary（up/down）**：SG 后 0.51–0.73；XGBoost / Logistic **超过** DeepLOB 1-2 %。

### 1.4 Sequence length

- T=10 vs T=1：accuracy +2 %，training cost ↑↑
- 推荐：先 T=1 baseline，验证 SG 增益再考虑 T 增加

## 2. Savitzky-Golay 简介

SG = 在 sliding window 内拟合多项式 + 取中心点的多项式值。

**优点**：
- 保留高阶矩（不像 moving average 把 peak 打平）
- 计算 O(window_size) per sample
- **Causal mode**（仅用过去 window）兼容我们 stateless 推理

**API（scipy）**：
```python
from scipy.signal import savgol_filter
smooth = savgol_filter(x, window_length=9, polyorder=3, mode='nearest')

# 因果版（避免用未来）
def causal_savgol(x, window=9, poly=3):
    """Causal Savitzky-Golay: only past values."""
    pad = window - 1
    x_pad = np.concatenate([x[:1].repeat(pad), x])  # left pad with first value
    out = savgol_filter(x_pad, window, poly, mode='nearest')
    return out[pad:]
```

## 3. 我们项目落地

### 3.1 Drop-in feature engineering

为以下 continuous features 生成 SG-smooth 副本：

| Original feature | SG smooth feature |
|---|---|
| `mid_price` | `mid_price_sg9` |
| `OFI` | `OFI_sg9` |
| `OBI` | `OBI_sg9` |
| `bid_ask_spread` | `bid_ask_spread_sg9` |
| `bid_vol_l1, ask_vol_l1` | `bid_vol_l1_sg9, ask_vol_l1_sg9` |
| `mid_return` | `mid_return_sg9` |

→ feature 总数 + ~ 10-15 个。

### 3.2 推荐参数

- window = 9（约 1-3 秒，视采样率）
- polyorder = 3
- mode = causal（自己实现或用 deque）

### 3.3 与 ReVol / Filter 的叠加序列

```
raw LOB
  → Order Filter (P16)         # 滤掉假报价
  → SG smooth (P10)            # 平滑剩余真信号
  → ReVol normalize (P17)      # μ̂/σ̂ 归一化
  → LightGBM input
```

每一步独立 ablation。

## 4. 期望增益

| 路径 | 预期 |
|---|---|
| 仅 SG（无 Filter / ReVol） | + 1 ~ + 3 |
| 与 Filter 叠加 | + 1.5 ~ + 4 |
| 与 ReVol 叠加 | + 2 ~ + 5 |

## 5. 实施 (≤ 0.5 天)

```python
# src/features/sg_smooth.py
import numpy as np
from scipy.signal import savgol_filter

CONTINUOUS_FEATS = ['mid_price', 'OFI', 'OBI', 'spread',
                    'bid_vol_l1', 'ask_vol_l1', 'mid_return']

def add_sg_features(df, window=9, polyorder=3):
    for feat in CONTINUOUS_FEATS:
        if feat in df.columns:
            x = df[feat].values
            df[f'{feat}_sg{window}'] = causal_savgol(x, window, polyorder)
    return df
```

## 6. 风险

- Window 太大（>15）→ 引入 lag → 在快变市场（large move）信号滞后
- Polyorder 太高（>5）→ 高阶多项式过拟合 noise → 反而加噪
- 与 ReVol 顺序：**先 SG 再 ReVol** 还是 **先 ReVol 再 SG**？
  - 推荐 **先 SG 再 ReVol**：SG 在 raw scale 上平滑（OK），ReVol 在 ε 域重 normalize
  - 验证：A/B 各跑一遍

## 7. 来源

- arxiv: https://arxiv.org/abs/2506.05764
- v2 html: https://arxiv.org/html/2506.05764v2
- 同思想（feature > model）也见 Briola 2023 微观结构 guide
