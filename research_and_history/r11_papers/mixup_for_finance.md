# Mixup — 对金融时序与跨股票泛化的实证

> 综合 Zhang et al. 2018 (mixup), Kaggle Jane Street, Amazon Science TS-mixup, ResearchGate Stock-Return augmentation
> [Original mixup arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
> [Embarrassingly Simple MixUp for TS arXiv:2304.04271](https://arxiv.org/abs/2304.04271)
> [Improving TS Forecasting with Mixup (Amazon Science)](https://www.amazon.science/publications/improving-time-series-forecasting-with-mixup-data-augmentation)
> [Data augmentation for stock return prediction](https://www.researchgate.net/publication/365908543_Data_augmentation_for_stock_return_prediction)

## 一句话

线性混合任意两个训练样本：`(x', y') = (λx_i + (1-λ)x_j, λy_i + (1-λ)y_j)`，λ ~ Beta(α, α)，把模型从"记忆训练点"推向"线性插值连续的 manifold"。

## 为什么对跨 sym 泛化有用

如果 mixup 时**强制 i, j 来自不同 sym**：
- 合成样本是"sym 之间的虚拟混合体"
- 模型学到的 decision boundary 必须对 sym-混合 invariant
- → 间接 enforce sym-agnostic representation，而又不需要在 inference 时用 sym

## 在金融数据上的 evidence

### 1. ResearchGate "Data augmentation for stock return prediction"
> "Mixup process creates new samples by averaging data from two original sources. After data augmentation, the LightGBM model is trained using the augmented data. Specifically, a synthetic price series is created from a random weighted sum of the N original price series... Prediction accuracy improves in **all cases** when using augmented data compared to non-augmented approaches."

→ **LightGBM + mixup 在 stock return prediction 全部 case 提升**。这是直接相关的 evidence。

### 2. Amazon Science "Improving TS Forecasting with Mixup Data Augmentation"
> "Different approaches for applying mixup to train neural forecasters have been evaluated on benchmark datasets, highlighting mixup's capability in **enhancing model performance across various hyper-parameter settings**."

### 3. Jane Street 1st place (Yirun)
> "Blending of data point pairs using the mixup technique to fill 'empty' spaces in training data and help soften overfitting."

### 4. arXiv:2304.04271 "Embarrassingly Simple MixUp for Time-series"
- MixUp++ 在 raw TS 上，LatentMixUp++ 在 latent space 上
- 1%~15% accuracy improvement on time series classification, both low and high label regimes

### 5. arXiv:2511.07930 IMA (2026)
- **Imputation-based Mixup with Self-Supervised Learning** for time series
- 推动 mixup 到 imputation 场景

## 对我们 LightGBM 路线的具体方案

### 方案 1: Pre-generated synthetic samples
```python
import numpy as np

def cross_sym_mixup(X, y, sym, n_aug=None, alpha=0.4, seed=42):
    """
    X: [N, F]  features
    y: [N, K]  labels (one-hot or regression)
    sym: [N]   sym ID
    n_aug: number of synthetic samples (default 0.5 * N)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if n_aug is None:
        n_aug = N // 2
    
    # Sample i, j ensuring different sym
    idx_i = rng.integers(0, N, size=n_aug)
    idx_j = np.empty(n_aug, dtype=np.int64)
    for k in range(n_aug):
        # Could be vectorized; this is illustrative
        candidates = np.where(sym != sym[idx_i[k]])[0]
        idx_j[k] = rng.choice(candidates)
    
    lam = rng.beta(alpha, alpha, size=n_aug)
    lam = np.maximum(lam, 1 - lam)  # symmetric mixup
    lam = lam[:, None]  # broadcast
    
    X_aug = lam * X[idx_i] + (1 - lam) * X[idx_j]
    y_aug = lam * y[idx_i] + (1 - lam) * y[idx_j]
    return X_aug, y_aug
```

### 方案 2: Latent-space mixup (NN only)
- 在 backbone 输出（256-d encoded feature）做 mixup
- 比 raw mixup 更稳定（latent space 更线性）

## 对我们 100×D 滑窗的特殊考虑

我们的"x"是 100×154 的窗口（时序 coherent）。直接 mixup 可能破坏时序结构。

**方案**：
- **方案 A** (推荐先试): 直接 elementwise mixup → 合成出"两个 sym 时序的叠加" → 学一个 robust feature extractor
- **方案 B**: **只 mix 静态 features**（per-tick 的当前状态），时序片段保持原样 → 更安全但增益弱
- **方案 C** (NN only): mix latent representation → mixup at bottleneck

建议先做方案 A，看 LOSO PnL 增益是否 > 0；如果有 noise overshooting，再切方案 B。

## α 选择

- α = 0.2: 弱混合，多数 sample 接近原样
- α = 0.4 (Jane Street 用)
- α = 1.0: 完全 uniform
- α 越大混合越激进；金融 data 推荐 α ∈ [0.2, 0.4]

## 与硬约束兼容性

- 训练时用 sym 选 i, j ✅ (training-only)
- inference 时不用任何 sym 信息 ✅
- 完全不破坏 §1 §2 §3 任何一条

## 预期增益

- Stock return prediction (LightGBM)：研究表明 **all cases improvement**
- 我们任务：LOSO sum +1 ~ +4（保守估计）

## 落地复杂度

- **0.5 天**：纯数据增强，pipeline 改一行
- 不需要重训 / 不需要新架构

## Sources
- [Original mixup paper Zhang 2018](https://arxiv.org/abs/1710.09412)
- [Embarrassingly Simple MixUp for TS](https://arxiv.org/abs/2304.04271)
- [Amazon Science TS mixup](https://www.amazon.science/publications/improving-time-series-forecasting-with-mixup-data-augmentation)
- [IMA Self-supervised mixup 2026](https://arxiv.org/html/2511.07930)
- [FrAug Frequency Domain Augmentation](https://openreview.net/forum?id=j83rZLZgYBv)
- [Jane Street Yirun writeup](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
- [Jane Street Discussion 224348](https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348)
- [Stock return augmentation ResearchGate](https://www.researchgate.net/publication/365908543_Data_augmentation_for_stock_return_prediction)
- [Robust Augmentation for MTS Classification](https://arxiv.org/pdf/2201.11739)
