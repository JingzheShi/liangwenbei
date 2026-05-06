# Tabular Data: Deep Learning is Not All You Need

- **Authors**: Ravid Shwartz-Ziv, Amitai Armon (Intel Labs)
- **arXiv**: 2106.03253 (Jun 2021, Information Fusion 2022)
- **Links**: https://arxiv.org/abs/2106.03253 ; https://arxiv.org/pdf/2106.03253

## 核心贡献

1. **重要的 reality check**：作者拿 4 篇 tabular DL 论文里的模型 (TabNet, NODE, DNF-Net, 1D-CNN) + XGBoost，在 **11 个数据集**（其中 9 个是这些 DL 论文的"主场"）上跨数据集严格 benchmark。
2. **结论**：
   - **XGBoost 在 11/11 数据集中赢 11 次**或并列；DL 模型只在自己论文用的数据集上接近，跨数据集都退化。
   - 每个 DL 模型表现最好的数据集 = 其自己原论文的数据集（暗示 hyperparameter overfitting / cherry-picking）。
   - **XGBoost 的 hyperparameter search 比 DL 快**（少几个数量级）。
3. **但 ensemble 有用**：5 模型 ensemble (TabNet + NODE + DNF-Net + 1D-CNN + XGBoost) > XGBoost alone。证实**多样性 > 单一模型**。
4. **实用建议**：
   - 默认 baseline 用 XGBoost。
   - 想要 SOTA 时再上 DL，并加入 ensemble。
   - **不要用 DL 替代 XGBoost，要互补**。

## 对我们比赛的可借鉴点

**这是 tabular ML 领域最被引用的"DL 不一定赢"论文，对我们决策非常关键**：

1. **GBDT 必须是我们的 baseline**——所有"NN 是 SOTA"的直觉都需要 evidence。LOB tabular feature 很可能也是 GBDT 主场。
2. **想要冲击榜首一定要 ensemble**：单 GBDT + 单 NN > 任何单模型。Kaggle 高分方案几乎都是 ensemble。
3. **DL 论文的 cherry-picking 警告**：看到 "TLOB beats DeepLOB by 3.7 F1 on FI-2010"，要质疑 cross-dataset 是否还成立。Briola 2023 benchmark 已经证实"TransLOB 跨数据集崩盘"，类似剧情可能在 TLOB 上也会出现。
4. **优先把工程精力放在 feature engineering + GBDT 调参**（XGBoost/LightGBM/CatBoost），而不是 chase 最新 ts forecasting paper。
5. **Heterogeneous ensemble 比同质 ensemble 强**：CatBoost + GBDT + DeepLOB + FT-Transformer 这种组合优于 5 个 LightGBM seed avg。

## 注意 / 反例

- LOB 的"100 tick 窗口"不完全是纯 tabular——序列结构让 CNN/Transformer/LSTM 有合理位置。所以 NN 不是纯 dead 的。
- 但如果把窗口 flatten + 加 rolling-stats，就回到了 tabular 范式，GBDT 重新有竞争力。**这是我们要做的关键 ablation**。

## 我的评分

- 实现成本：N/A（论文不是模型）
- 优先级：**high**（决定我们 GBDT vs NN 的研发投入分配）
