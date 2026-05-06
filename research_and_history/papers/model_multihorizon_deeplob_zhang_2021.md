# Multi-Horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches and Hardware Acceleration using Intelligent Processing Units

- **Authors**: Zihao Zhang, Bryan Lim, Stefan Zohren (Oxford)
- **arXiv**: 2105.10430 (May 2021)
- **Links**: https://arxiv.org/abs/2105.10430 ; https://arxiv.org/pdf/2105.10430

## 核心贡献

1. 把原始 DeepLOB 扩展为 **多 horizon 预测** 的 encoder-decoder 架构：
   - **DeepLOB-Seq2Seq**：encoder 用 DeepLOB conv-LSTM 提特征，decoder 用 LSTM 逐步生成多个 horizon 的预测，autoregressive。
   - **DeepLOB-Attention**：decoder 加注意力机制，在生成每个 horizon 的预测时动态关注历史特征。
2. 单次 forward 同时输出 K 个 horizon（k=10/20/30/50/100 events）的预测，**不用为每个 horizon 单独训模型**——比赛极有用。
3. **关键发现**：
   - 在短 horizon 上（k=10/20）三个模型差不多。
   - 在长 horizon 上（k=50/100）DeepLOB-Attention > DeepLOB-Seq2Seq > 单 horizon DeepLOB。
   - Attention 权重大多集中在最近 tick——**说明 long-range 历史不重要**，证实"短窗口足矣"。
4. 在 Intel IPU 上推理加速分析（对我们意义不大）。

## 实验设置

- LSE（伦敦交易所）数据，跨股票训练 → 跨股票测试。
- Horizons: k ∈ {10, 20, 30, 50, 100} events。
- 标签同 DeepLOB（mean ratio + ±α 阈值切三类）。
- Metric: F1 (3-class)。

## 对我们比赛的可借鉴点

**这篇论文几乎为我们量身定做**——比赛要 5 个 horizon (5/10/20/40/60)，正好对应 multi-horizon 预测。

具体借鉴：
1. **multi-task = single backbone + 5 heads** 比 5 个独立模型省 5× 参数和训练时间。
2. **Attention decoder 在长 horizon 上明显赢**——我们 label_60 是 PnL 上限最高的 horizon，长 horizon 优化更关键。
3. **Attention 权重集中在最近 tick** 印证我们的输入约束（≤100 tick）合理；可以试更短窗（如 50/64 tick）观察性能。
4. **autoregressive decoder** 思路：先预测 t+5, 用 t+5 的预测作为 hidden state 输入预测 t+10... 让 horizon 之间共享信息。

## 我的评分

- 实现成本：medium（在 DeepLOB 之上加 decoder）
- 优先级：**high**（5-horizon 多任务结构是比赛刚需）
