# Test-Time Adaptation for Non-stationary Time Series — Synthetic to Financial Markets

> arXiv:2602.00073 (2026)
> [Paper](https://arxiv.org/abs/2602.00073) | [HTML](https://arxiv.org/html/2602.00073)

## 为什么这篇关键

是少有的**直接在金融时序（SPY、QQQ、EUR/USD）上 benchmark TTA** 的论文，而且明确报告了 "什么有效、什么有害"。

## 方法

- **Lightweight TTA framework**：backbone 冻结，只更新 normalization affine parameters (γ, β of LayerNorm/BN)
- 输入：recent unlabeled windows（无 label）
- 优化目标：
  - **Classification**: minimize entropy + temporal consistency
  - **Regression**: minimize prediction variance across weak time-preserving augmentations + optional EMA teacher distillation
- **Stability mechanisms**:
  - Quadratic drift penalty（防 affine params 飘太远）
  - Uncertainty-triggered fallback（不确定时退回原模型）

## 实验

3 个金融数据集，3 个 regime
- SPY (US large-cap stock)
- QQQ (NASDAQ tech)
- EUR/USD (FX pair)
- Pandemic / High-inflation / Recovery regimes

## 结论（**对我们最重要**）

### ✅ 有效
- **Synthetic gradual drift** 上，norm-only TTA 改善 forecasting error
- **金融市场上，简单 BN stats update 是 robust default**

### ⚠️ 警告
> "More aggressive norm-only adaptation **can even hurt** [in financial markets]"

→ 在金融数据上，**激进的 TTA 反而损害**——简单更新 BN running stats 是上限。

## 对我们 setup 的启示

### 我们的硬约束 (Predictor 不能跨 call 维护 state)

- **原版 Tent 不可用**：跨 batch 累计 BN stats 违反 §2 评测约束
- **本论文方法也不能直接用**：他们用 "recent unlabeled windows" → 需要跨 call buffer，我们不允许

### 但有一个我们可用的等价物

**within-window normalization**：在当前 100×D 的 input 内部算 mean/std → z-score。
这等于把 100 ticks 当成"this batch"，在内部做 BN-stats 替换。

→ **T7 已经做了这件事**！T7 (window z-score) 验证了对 sym=2 brittleness 有效。

### 还可以做什么

NN 路线时，把 BatchNorm 的 running mean/var 替换成"input dependent"（每个 forward 用当前 batch 的 stats）→ test-time 自动 adapt。这是 **AdaIN 风格**，不需要 buffer。

伪代码：
```python
class WithinBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        # x: [B, T, F]; compute stats over T axis (within window)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_norm + self.beta
```

这个 module 的 inference 行为只依赖**当前 forward 的 input**，无 cross-call state，**完全合规**。

### 对 LightGBM 路线

T7 的 window z-score 已经吃了等价好处。无新增建议。

## 关键引用

> "On synthetic gradual drift, normalization-based TTA improves forecasting error, while in financial markets a simple batch-normalization statistics update is a robust default and more aggressive norm-only adaptation can even hurt."

## 与原版 Tent (Wang 2021) 的关系

| | 原版 Tent | 本文方法 |
|---|---|---|
| Update target | BN affine params (γ, β) | Norm affine + drift penalty + fallback |
| State | 需要跨 batch BN running stats | 同上（违反我们硬约束） |
| 数据 | image classification | TS forecasting + 金融 |
| 结论 | 显著改善 ImageNet-C | 在金融上 marginal，激进版有害 |

## 推荐

- **不直接做** TTA in inference
- **NN 路线时**用 within-batch norm 替换 BN，等价但合规
- T7 window z-score 已捕获大部分好处

## Sources
- [arXiv:2602.00073](https://arxiv.org/abs/2602.00073)
- [HTML](https://arxiv.org/html/2602.00073)
- 原版 Tent: [arXiv:2006.10726](https://arxiv.org/abs/2006.10726)
- [Awesome TTA list](https://github.com/tim-learn/awesome-test-time-adaptation/blob/main/TTA-OTTA.md)
