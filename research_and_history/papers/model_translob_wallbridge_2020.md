# Transformers for Limit Order Books (TransLOB)

- **Author**: James Wallbridge
- **arXiv**: 2003.00130 (2020)
- **Links**: https://arxiv.org/abs/2003.00130 ; code: https://github.com/jwallbridge/translob

## 核心贡献

1. 把 Transformer self-attention 引入 LOB mid-price 预测，证明在 FI-2010 上同时超过 DeepLOB 和当年的 LSTM baselines。
2. 架构 = 几层 1D dilated CNN（feature extractor，类 DeepLOB conv block）→ 2 个 Transformer blocks（3 头 self-attention，weights shared 跨 block）→ FC → softmax。
3. **样本效率高**：Transformer 比 LSTM 收敛快得多，适合在有限数据下学。
4. 把注意力图可视化能看到模型关注哪些 tick / 哪些 level，提供 interpretability。

## 提出的特征 / 输入表示

输入仍是 100×40 的 LOB raw tensor（同 DeepLOB）。CNN 是 dilated 的，让感受野覆盖更长时间。

## 对我们比赛的可借鉴点

- 如果 baseline 跑 DeepLOB 不够强，下一个迭代直接换 TransLOB，输入 schema 不变。
- 注意力可视化对**特征筛选**有意义——可以看哪些 level / 哪些 tick 距离对预测最重要，反过来指导 feature engineering。
- 后来 TLOB（2025）证明 dilated conv 可以替换为 spatial attention，效果更好。

## 我的评分

- 实现成本：medium
- 优先级：**medium**（更新但 TLOB 取代了它）
