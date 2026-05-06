# TabNet: Attentive Interpretable Tabular Learning

- **Authors**: Sercan Ö. Arık, Tomas Pfister (Google Cloud AI)
- **arXiv**: 1908.07442 (Aug 2019, AAAI 2021)
- **Links**: https://arxiv.org/abs/1908.07442 ; PyTorch: https://github.com/dreamquark-ai/tabnet

## 核心贡献

1. 把 deep learning 引入 tabular 领域的 SOTA 之一，使用 **sequential attention + sparse feature selection**——每个 decision step 用 sparsemax 选少数 feature 来用。
2. 三个核心创新：
   - **Sparse attentive feature selection**：mask-based, 每步选 ~k 个 feature 处理。
   - **Multi-step decision blocks**：类似 GBDT 的迭代细化（每个 block 用一部分 feature 做局部决策）。
   - **Ghost batch normalization**：用小 batch 子集做 BN 稳定训练。
3. **自监督预训练**：在 unlabeled tabular data 上训"masked feature reconstruction"，再 fine-tune 监督任务。提升大数据 + 少 label 场景。
4. 在 11 个 tabular 数据集上 vs XGBoost：每个数据集都接近或超越（在原论文 cherry-picked 数据集上）。

## 主要架构

```
input features
  → Feature Transformer (FC + GLU) (shared across steps)
  → Attentive Transformer (mask via sparsemax) — 选 features
  → split: 一部分输出 logit, 一部分继续给下一步
  → repeat N steps, 累加 logit
  → softmax / regression head
```

## 后续 benchmark 的 reality check

- Shwartz-Ziv & Armon (2021) "Tabular Data: Deep Learning is Not All You Need"：在跨数据集严格测试下 **XGBoost 8/11 数据集赢 TabNet**；TabNet 在自家 paper 用的数据集上强，跨场景退化。
- Gorishniy et al. 2021 (FT-Transformer)：TabNet 整体落后于简单 ResNet-MLP 和 FT-Transformer。
- 2024 综述 (Ye et al.)："tree-mimic 网络如 NODE、TabNet 一般 underperform ensemble"。

## 对我们比赛的可借鉴点

1. **TabNet 不是首选**：在 154 维 LOB tabular 特征上，**LightGBM/CatBoost 几乎一定打过 TabNet**——参考社区共识 + 现实 benchmark。
2. **可以作为 ensemble 一员**：TabNet 提供"和 GBDT 不同的 inductive bias"，加入 stack/blend 时偶尔能提分（Optiver 2023 winner 就有 GRU+CatBoost+Transformer 的 mix）。
3. **Sparse feature selection 思路有用**：154 维 LOB 特征里很多是冗余的；TabNet 的 sparse mask 可以告诉我们哪些 feature 真的重要——用作 feature importance 工具。
4. **自监督预训练在我们场景受限**：我们 label 充足，自监督的优势小。

## 我的评分

- 实现成本：medium（pytorch-tabnet 库现成，但调参敏感）
- 优先级：**low**（GBDT 是主战场；TabNet 仅作为 ensemble diversity 候选）
