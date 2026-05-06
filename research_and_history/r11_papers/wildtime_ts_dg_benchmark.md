# Wild-Time + Time Series Domain Generalization Benchmarks

> Yao et al. NeurIPS 2022 + Deng et al. 2024 (TKDD)
> [Wild-Time arXiv:2211.14238](https://arxiv.org/pdf/2211.14238)
> [Domain Generalization in Time Series Forecasting (Deng 2024)](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/deng-2024-domain.pdf)
> [LTG: Latent Temporal Generalization (2412.11171)](https://arxiv.org/html/2412.11171v1)

## 为什么这两篇关键

它们 systematically benchmark 了 IRM / V-REx / Group DRO / DANN / SWA / ERM 在**时序 distribution shift** 上的真实表现，能告诉我们"哪些 OOD 方法在 ts 上真的 work"。

## Wild-Time (Yao 2022, NeurIPS)

### 范围
5 个 real-world datasets，**temporal distribution shift**:
- 药物发现
- 患者预后
- 新闻分类
- arxiv 论文分类
- 金融数据 (FMoW + 别的)

### 关键发现
- **没有一个方法稳定胜出 ERM**——这是 sobering finding
- 对于 temporal shift，"adapt to the future" 比 "robust to all" 更重要
- IRM / V-REx 在某些 temporal shift 上 **比 ERM 还差**

### 对我们的 implication
我们的任务**主要不是 temporal shift**（虽然有），而是**cross-instrument shift**。Wild-Time 的悲观结论主要适用于 temporal，**对 group/source shift 的悲观度低**。

## Deng 2024 (TKDD) - Domain Generalization in TS Forecasting

### Setup
- M training source domains (e.g. M stores) → train
- K - M unseen test domains → test
- 对应我们：M=5 sym → train，可能 unseen sym → test

### Methods Compared
- **IDGM**: gradient matching for DG
- **Cedar**: cross-domain regularization with difficulty awareness
- **UniTime**: LLM-based cross-domain learning
- **DLinear / DeepAR / WaveNet / GPT4TS**: standard forecasters
- **DANN, wERM**: 列为"weaker baselines"（明确 underperform）

### 关键 takeaways
1. **DANN / wERM weak** in TS forecasting DG
2. **Gradient matching (IDGM)** outperforms most invariance-based methods
3. **Latent factor decomposition** (LTG, Conditional β-VAE) 是较新的方向

### LTG (Latent Temporal Generalization)
- Conditional β-VAE: encoder domain-agnostic, decoder conditioned on domain
- TS decomposition: trend + seasonal
- **Domain regularization**: latent vector partitioned into shared + domain-specific parts
  - "encouraging similarity of shared parts and dissimilarity of domain-specific parts"
- 比 IDGM 计算更便宜
- 在 RNN/CNN 上效果好，在 DLinear 上一般

## 对我们 setup 的 implication

### Negative findings to remember
1. **DANN 在 TS DG 上是 weak baseline** — 不要期待 DANN 给 +5 分以上的增益
2. **IRM 单独不够** — 必须配 invariance-aware data augmentation 或 latent decomposition
3. **没有 silver bullet** — Wild-Time 显示几乎所有方法都难稳定打败 ERM

### Positive directions
1. **Gradient matching (IDGM)**: 跟 V-REx 类似的正则化思路，但用梯度方向 alignment 而非 risk variance。
2. **Latent factor decomposition**: 比"所有 sym 共享 backbone"更细粒度，把 latent 拆成 shared + sym-specific。但**不能在 inference 用 sym → 必须只用 shared 部分**。
3. **Source domain weighting**: 跟 Group DRO 类似，但更鲁棒——dynamically weight sym 的训练 contribution。

## 对 LightGBM 路线的 mapping

| Method (NN-orig) | LightGBM 等价物 |
|---|---|
| IDGM (gradient matching) | 难映射；GBDT 没有 gradient direction 的概念 |
| Cedar (difficulty-aware) | sample weight 按 hard example 重 |
| LTG (latent decomp) | 难映射；GBDT 没有 latent space |
| **Group DRO** | ✅ sample reweighting per group |
| **Mixup** | ✅ 合成数据 |
| **Robust loss (focal/Huber)** | ✅ 直接 |
| DANN | ❌ 无 |

→ **结论**：LightGBM 路线的 OOD 工具箱比 NN 小，但 Group DRO + Mixup + Robust Loss 这三件套基本够用。

## 推荐阅读顺序

1. Wild-Time benchmark paper → 看 ERM 的强度
2. Deng 2024 TS DG → 看 DANN 弱的程度
3. LTG arXiv:2412.11171 → 看 latent decomposition 思路（NN 路线时参考）

## Sources
- [Wild-Time arXiv:2211.14238](https://arxiv.org/pdf/2211.14238)
- [Wild-Time OpenReview](https://openreview.net/forum?id=F9ENmZABB0)
- [Domain Generalization in Time Series (Deng 2024) PDF](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/deng-2024-domain.pdf)
- [TKDD published version](https://dl.acm.org/doi/10.1145/3643035)
- [LTG arXiv:2412.11171](https://arxiv.org/html/2412.11171v1)
- [Scaling Up Temporal Domain Generalization (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1432.pdf)
- [WILDS benchmark (Koh 2021)](https://proceedings.mlr.press/v139/koh21a/koh21a.pdf)
