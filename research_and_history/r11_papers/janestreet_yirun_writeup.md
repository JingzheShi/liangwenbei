# Jane Street Market Prediction — 1st Place (Yirun Zhang) Writeup Summary

> Kaggle Jane Street Market Prediction (2021)
> [Yirun's writeup](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
> [Numerai forum recap with full architecture](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338)

## 为什么这篇关键

Jane Street 比赛是**最贴近我们任务的金融 ML benchmark**：
- 高频 tabular features
- 跨多 instrument（虽然 instrument id 不公开，但相当于多个 sym）
- 评测时 distribution shift 大（live trading）
- **冠军方案直接面对了 cross-asset / OOD 难题**

## 架构（Supervised Autoencoder + Multi-task MLP）

**3 个输出 head**:
1. **Decoder**: 重建 input feature vector
2. **AE Targets**: 用 decoder output 预测 5 个 resp targets
3. **MLP Targets**: 用 (concat of original feature + encoder bottleneck) 预测 5 个 resp targets ← **inference 时用这个**

**Hidden units**: `[96, 96, 896, 448, 448, 256]`

**Dropout rates**: `[0.035, 0.035, 0.4, 0.1, 0.4, 0.3, 0.25, 0.4]`

**Activation**: Swish (避免 dead neuron)

**BatchNorm**: 全程使用

**Gaussian noise layer**: encoder 之前

**Loss**: 全部 MSE，三 head 等权相加

## 关键 trick

### 1. Mixup（很重要！）
> "Blending of data point pairs using the mixup technique to fill 'empty' spaces in training data and help soften overfitting."

- α 用 beta distribution（接近 uniform）
- 我们可以直接套：cross-sym mixup

### 2. Z-scored Context Features
> "Z-scores of original features relative to means of previous 100 opportunities were used to provide context to the model, helping it adjust risk based on market regime and divergence from that regime."

→ 这正是我们 T7 做的事（window z-score）。Yirun 也用 → **跨任务 validated trick**

### 3. Cross-Validation
> "Cross-validation with fold-based training to prevent label leakage"

- 4 folds，**day-level isolation**（不同 day 不进同一 fold）
- Validation days **spread throughout** the period（不是 sequential）→ 避免 intra-day leakage 同时 expose temporal variance

## Discussion 224348 中的细节

### Architecture: 3-layer FC
```
batch_norm → fc → leaky_relu → dropout
```
- Layer 1 dropout: 0.35
- Layer 2 dropout: 0.40
- Layer 3 dropout: 0.45

### Two specialized models
1. **Trend classification**: 用多 horizons 的 mean response 降噪
2. **Utility maximization**: 用 daily trend prediction + volatility prediction + z-scored 100-trade context

### Weighted loss
- 强调 high-weight opportunities（Jane Street 给每条样本一个 weight，weight 大 = 更重要）
- Cap weight at 0.4 → 防止 outlier 主导

## 跨我们任务的 takeaways

### 直接套用
1. **Cross-sym mixup**: ✅ 推荐做（r11 报告 Method B）
2. **Window z-score context**: ✅ T7 已经做了
3. **Day-level / Sym-level isolation CV**: ✅ 我们 LOSO 已经做了
4. **Multi-horizon mean ensemble**: ✅ T5b Scheme C 已经做了

### 需要改造
- **Supervised AE 架构**：我们 Predictor 入参是 100×D 滑窗（不是 1×D 行），所以 AE 要先 flatten 或用 1D-CNN encoder。
  - **建议**：flatten 100×154 → 15400-d → encoder → 256-d → decoder + 3-class head
- **Gaussian noise**：训练时 input 加 N(0, σ=0.02) 噪声，简单且有效

### 不能套
- **Per-instrument feature** (Jane Street 有 instrument id 但他们没公开用法)
- **Live retraining** (评测无接口)

## 代码位置

- 核心仓库 (作者发的)：[Yirun's Code](https://github.com/Yirun-Zhang)（部分公开）
- 复现：
  - [github.com/codefluence/jane_street](https://github.com/codefluence/jane_street)
  - [github.com/scaomath/kaggle-jane-street](https://github.com/scaomath/kaggle-jane-street)
  - [github.com/JLFDataScience/Jane-Street-Market-Prediction](https://github.com/JLFDataScience/Jane-Street-Market-Prediction)
- Crypto adaptation: [Kaggle: Jane Street 1st place solution → Crypto PyTorch](https://www.kaggle.com/code/neodory/jane-street-1st-place-solution-crypto-pytorch)

## 我们应该在 NN 路线上抄哪几样

1. **Mixup with α=0.4** ✅ 高优先级
2. **Multi-task heads**（任务头 + AE 重建头）作为正则化
3. **Gaussian input noise σ=0.02**
4. **Swish activation**
5. **三段式 dropout**：input low → middle high → late low
6. **Multi-horizon mean ensemble**（已做）
7. **Day-isolated K-fold + embargo**（已做）

## Sources
- [Yirun Kaggle Writeup](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
- [Numerai forum architecture recap](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338)
- [Discussion 224348 (mixup + 4-fold)](https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348)
- [github codefluence](https://github.com/codefluence/jane_street)
- [github scaomath kaggle-jane-street](https://github.com/scaomath/kaggle-jane-street)
- [Jane Street Real-Time Market Data Forecasting (2024 follow-up)](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
