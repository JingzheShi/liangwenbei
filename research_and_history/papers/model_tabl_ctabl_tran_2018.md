# Temporal Attention augmented Bilinear Network for Financial Time-Series Data Analysis (TABL / CTABL / BiN-CTABL)

- **Authors**: Dat Thanh Tran, Alexandros Iosifidis, Juho Kanniainen, Moncef Gabbouj (Tampere)
- **arXiv**: 1712.00975 (TABL, 2017); 后续 CTABL (2018) 和 BiN-CTABL (2020, arXiv 2003.00598)
- **Links**: https://arxiv.org/abs/1712.00975 ; BiN: https://arxiv.org/pdf/2003.00598

## 核心贡献

1. 提出 **Bilinear Layer**：把 LOB 输入当成 2D tensor (T × F)，bilinear projection 沿时间方向 + 沿特征方向各做一次 ——比把它 flatten 成 vector 后过 MLP 大幅减少参数量，又能保留 spatio-temporal 结构。
2. **TABL 加 attention**：在 bilinear projection 中插入 temporal attention（attention 矩阵学时间维的权重）。
3. **CTABL** = cascade TABL（多层堆叠）。
4. **BiN-CTABL**：加 **Bilinear Normalization** layer ——同时沿 temporal 维和 feature 维做 normalization，处理金融数据的非平稳和量级差异。
5. **结果**：在 LOB benchmark Briola 2023 中 BiN-CTABL **是最稳定的 SOTA**（FI-2010 82.6 F1，跨数据集鲁棒度第 1）。

## 主要架构（BiN-CTABL）

```
LOB tensor (T=100 × F=40)
  → Bilinear Normalization (joint temporal + feature norm)
  → TABL_1 (bilinear + temporal attention) → tensor → relu
  → TABL_2 → ...
  → final TABL → 分类 logit (3-class)
```

参数极少（k 量级），训练快，不像 DeepLOB 那样大 conv stack。

## 实验设置

- FI-2010 benchmark + 真实 NASDAQ 数据。
- Horizons: K=10, 50, 100。

## 对我们比赛的可借鉴点

1. **BiN-CTABL 是被实证检验的"性价比之王"**：参数远比 DeepLOB / TransLOB 少，泛化更好。**强烈建议作为第一个 NN baseline**（甚至先于 DeepLOB）。
2. **Bilinear Normalization** 这个 normalization 思路（沿时间 + 沿特征联合 normalize）值得移植到我们的 154 维 feature × 100 tick 窗口——比简单 z-score 应该更稳。
3. **Bilinear projection** 比 flatten + MLP 节省参数：100×154 = 15400 维输入，flatten 后 MLP 第一层就是 15400 × hidden = 4M 参数；bilinear 直接降到几 k 参数，计算资源友好。
4. **结合 attention 在 temporal 维度** 是一致认可的做法（DeepLOB-Attention、TLOB、BiN-CTABL 都这么做）。

## 我的评分

- 实现成本：medium（开源参考实现可得，但 bilinear layer 要小心）
- 优先级：**high**（被证实是 LOB 任务上最鲁棒的 NN，必跑）
