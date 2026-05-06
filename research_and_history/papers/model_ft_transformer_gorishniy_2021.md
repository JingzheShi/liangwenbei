# Revisiting Deep Learning Models for Tabular Data (FT-Transformer)

- **Authors**: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko (Yandex Research)
- **arXiv**: 2106.11959 (Jun 2021, NeurIPS 2021)
- **Links**: https://arxiv.org/abs/2106.11959 ; code: https://github.com/yandex-research/rtdl-revisiting-models

## 核心贡献

1. 在 11 个 tabular 数据集上**严格比较** DL vs GBDT，结论：**没有统治性的 winner，但 GBDT 仍是默认选择**。
2. 提出 **FT-Transformer**：
   - 每个 numerical feature 经 **feature tokenizer**（线性投影 + bias）成 d 维 token；categorical feature 用 embedding 成 token。
   - 加 [CLS] token，过 N 层 vanilla Transformer encoder。
   - [CLS] 输出 → 分类/回归头。
3. 同时提出 **ResNet for Tabular**：简单的 ResMLP 块（Linear → BN → ReLU → Dropout → Linear → Dropout + skip）。这个简单 baseline 经常打败更复杂的 tabular DL（NODE / TabNet 等）。
4. **关键 finding**：
   - FT-Transformer > ResNet > 复杂 tabular DL (NODE / TabNet / DCN-V2 等)。
   - FT-Transformer **缩小**了与 GBDT 的差距，但**没全面超越**。GBDT 在 ~50% 数据集上仍领先。
   - 用 ensemble (FT-Transformer + XGBoost) 比单独 XGBoost 强。

## 主要架构（FT-Transformer）

```
[num_1, num_2, ..., cat_1, cat_2, ...]
  → Feature Tokenizer:
        num_i  → W_num × x_i + b_num_i  (一个 token)
        cat_j  → embedding[cat_j]       (一个 token)
  → prepend [CLS] token
  → N × {LayerNorm → MultiHeadAttn → LayerNorm → FFN(GeLU)} 
  → take [CLS] output → linear → output
```

## 实验设置

- 11 个公开 tabular 数据集（含 finance、healthcare、real-estate）。
- Hyperparameter：optuna 搜索。
- Metric: ROC AUC / RMSE 视任务而定。

## 对我们比赛的可借鉴点

**对我们这个场景非常相关**：154 维 LOB tabular feature × 100 tick 窗口，如果把窗口 flatten 成 vector，FT-Transformer 是个合理候选。

1. **可作为 GBDT 的 NN 队友 ensemble**：Optiver 2023 winner 用 CatBoost + GRU + Transformer 的 0.5/0.3/0.2 加权；类似 FT-Transformer 可作为 NN 那一份。
2. **如果不 flatten 时序，可以 per-tick 用 FT-Transformer + 跨 tick 用 LSTM**——两层结构。
3. **ResNet-Tab baseline 不要忘**：往往简单 ResMLP 块比 fancy 模型还强。我们的 NN baseline 不止 DeepLOB，也应该试 ResMLP。
4. **CLS token 提取全局表征**这个习惯应该用在我们的窗口表征上（替代 mean pooling）。

## 我的评分

- 实现成本：medium（rtdl 库现成）
- 优先级：**medium**（如果 GBDT 已经 top 但需要 NN ensemble 多样性，上 FT-Transformer）
