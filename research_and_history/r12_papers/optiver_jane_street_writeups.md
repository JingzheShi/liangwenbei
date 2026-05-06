# Kaggle Winner Ensemble Writeups: Optiver & Jane Street

> 补充 paper 摘要——这些是 Kaggle writeup 而非正式 paper，但对我们的 ensemble 设计直接相关。

---

## 1. Optiver — Trading at the Close 2023, 1st place (HYD)

**官方 writeup**：[1st place solution](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
**slogan**：*"Feature is all you need"*

### 三大杀手锏特征（社区共识）

1. **Triplet imbalance**：3 个价格相关 feature 之间的两两 imbalance ratio，遍历所有三元组
2. **Pairwise price imbalance**：两个 price 之间的 (a-b)/(a+b)
3. **Synthetic index features**：合成市场指数与单股票的 deviation

### Ensemble 配方（社区报告）

- **CatBoost ≈ 50% weight + GRU ≈ 30% + Transformer ≈ 20%**
- 不是 stacking，而是简单 weighted average（权重凭 OOF 验证手 tune）
- CatBoost 是单权重最重的部分——它对**短-medium horizon、含 noise category 的 LOB feature 最稳**
- GRU 提供 sequential pattern；Transformer 提供 long-range attention
- 三者 OOF 相关 < 0.85（diversity 充足）

### 训练 / 后处理

- **online retrain 5 次 over 12-day test period**（接受最新数据 fine-tune）——这是商赛优势，**我们不能用**（评测无 state）
- threshold post-processing 没有公开细节，但社区推测是按 PnL 调 confidence floor

### 对我们的启示

- ✅ CatBoost 在 LOB 场景应当成 first-class base（方案 1）
- ✅ 多算法 simple mean 已能接近 stacking，**无需复杂 meta-learner 也行**
- ❌ Online retrain 用不上
- ✅ Feature engineering 与 ensemble 是互补的—— Scheme E (T8 z-score features) + ensemble 应组合做

---

## 2. Jane Street — Market Prediction 2020, 1st place (Yirun Zhang / gogo827jz)

**官方 writeup**：[Yirun's Solution (1st place)](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
**关键论文**：Yirun's *Supervised Autoencoder + MLP* notebook ([gogo827jz/jane-street-supervised-autoencoder-mlp](https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp))

### 架构

**Supervised Autoencoder (SAE)**：
- Encoder: input → bottleneck (low-dim representation)
- Decoder: bottleneck → input reconstruction (auxiliary)
- Predictor head: bottleneck → 5 resp targets (multi-output)

5 个 resp target 共享同一 bottleneck → 类似 multi-task multi-horizon shared backbone（与我们 Scheme C iter_002 思想一致）。

### Ensemble 配方

5 个不同初始化 / 不同时间 fold 的模型 → **middle-60% averaging**：
1. 每个测试点收集 5 个预测
2. 排序，**砍掉最高 1 个和最低 1 个**
3. 中间 3 个取均值

### CV 协议

- **3-fold time-grouped CV with 10-day embargo**
- 训练初期用 RAdam / Adam + cosine annealing scheduler
- **Utility function regularizer fine-tune every 10 epochs**——直接优化交易收益的 surrogate

### 对我们的启示

- ✅ **Middle-60% averaging** 是 robust ensemble 的核心 trick（我们方案 7）；T11 一旦 simple mean 失败，立即试 middle-60%
- ✅ **Multi-task shared bottleneck**（5 resp 共享 representation）= 我们 Scheme C multi-horizon 思想，已采纳
- ✅ **Utility-aware fine-tuning** = 对我们的 PnL-aware loss / threshold 后处理（已部分实现）
- ⚠️ Embargo 我们用 LOSO 自动满足；但**时间方向**的 embargo 还没做——iter_002 LOSO 只 cross-sym，没 cross-time。这是潜在 leakage 来源（如果 sym 内部时间相关性强）。

---

## 3. Optiver — Realized Volatility 2021, 1st place

**核心 trick**（不是 ensemble，但顺带记下）：

- **kNN over time_ids + tick-size 反推真实价格**：把官方匿名化打乱的 time_id 重排回真正时间顺序，恢复时间结构
- LightGBM/MLP simple ensemble at end

→ 与我们关系不大，但提醒：**有时打乱顺序背后藏 trick**。我们的评测打乱测试点顺序——任何时间 patten 都不能 cross-sample reconstruct。

---

## 4. M. Kim 在 Optiver 写的 ensemble 评论（社区帖）

- **"trim 极端预测胜过 simple mean"**——多次比赛验证
- **rank averaging 在跨算法时强于 mean**，但 **跨 seed 同算法时 mean 即可**
- ensemble weight 用 Nelder-Mead 调 OOF score 比 grid search 快 100×

---

## 关键 takeaway 总结

| Trick | 来源 | 对我们价值 |
|---|---|---|
| CatBoost + GBM + NN simple weighted mean | Optiver Close 1st | ✅ 方案 1 直接抄作业 |
| Middle-60% averaging | Jane Street 1st | ✅ 方案 7，T11 后必试 |
| Multi-task shared bottleneck | Jane Street 1st | ✅ 已在 iter_002 用 |
| Online retrain over rolling test | Optiver Close 1st | ❌ 评测无 state，跳过 |
| time-grouped CV + embargo | Jane Street 1st | ⚠️ 我们 LOSO 是 sym 维，时间维 embargo 待补 |
| Utility-aware loss / fine-tune | Jane Street 1st | ✅ 与 threshold post-proc 互补 |
| Nelder-Mead weight search on OOF | M. Kim Optiver | ✅ 方案 10 |
