# LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study

- **Authors**: Antonio Briola, Silvia Bartolucci, Tomaso Aste 等（UCL group）
- **arXiv**: 2308.01915 (Aug 2023)
- **Links**: https://arxiv.org/abs/2308.01915 ; https://arxiv.org/html/2308.01915

## 核心贡献

1. **公平 benchmark**：在 FI-2010、LOB-2021、LOB-2022 三个数据集上，**标准化训练协议**测了 15 个 LOB 深度学习模型。
2. 比较的模型清单（必看）：
   - Baseline: MLP, LSTM
   - SOTA (13): CNN1, CTABL, **DeepLOB**, DAIN, CNNLSTM, CNN2, **TransLOB**, TLONBoF, **BiN-CTABL**, DeepLOBATT, DLA, ATNBoF, **Axial-LOB**
   - Ensemble (2): MAJORITY voting, METALOB（meta-classifier）
3. **关键 finding**：
   - **BiN-CTABL 是最稳定的 SOTA**：99.7% robustness、FI-2010 上 82.6 F1。
   - top-6 模型有 5 个用了 attention，CNN-only 模型 std dev > 5 points（不稳定）。
   - **TransLOB 在 FI-2010 强（87.3 F1），但跨数据集严重退化（→59.4 F1）**——overfit to FI-2010 distribution。
4. **泛化危机**：所有模型从 FI-2010 → 真实 NASDAQ 数据 F1 平均下降 19.6%。BiN-CTABL 也不例外。
5. **股票级差异**：CSCO（高平稳性，class balance 18-65-17%）所有模型表现都好；SOFI、SHLS 这类波动股票预测大幅崩盘。

## 实验设置

- FI-2010：5 只 Helsinki Nasdaq 股票（2010 年 6 月，10 trading days）
- LOB-2021：6 只 NASDAQ 股票（2021 年 7 月）— SOFI, NFLX, CSCO, WING, SHLS, LSTR
- LOB-2022：同 6 只（2022 年 2 月）
- Horizons: K=10, 50, 100 (events)

## 对我们比赛的可借鉴点

1. **不要相信 FI-2010 上的 87% F1**：跨股票/年份后大幅退化，我们要 in-distribution 严格 CV。
2. **优先选 attention-based 模型**（top-6 中 5 个有 attention），但**不要过度迷信 Transformer**：BiN-CTABL（bilinear + temporal attention）反而比 TransLOB 更鲁棒。
3. **METALOB（meta-classifier on stacked predictions）有效**：把多个模型的 logits 喂给 LightGBM 做最后分类——这就是经典 stacking，比 majority voting 强。
4. **跨股票训练时要按 stock 分层**——CSCO 类高平稳股票会主导 loss，volatile 股票被欺压。要么 sym embedding 要么按 sym 加权。
5. **MCC（Matthews Correlation Coefficient）比 accuracy 更可靠**——3 类不均衡场景下 accuracy 会被 majority class 推高。我们应该并行追踪 MCC 和 PnL。

## 我的评分

- 实现成本：N/A（这是 benchmark 论文，不是新模型）
- 优先级：**high**（决定我们模型选型方向：BiN-CTABL > TransLOB 在 LOB 任务上）
