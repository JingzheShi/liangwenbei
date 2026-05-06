# Kaggle 比赛 Winner 模型选型 + 训练 trick + 后处理

> 与 R-feature worker 的 `competition_insights.md` 区分：那边偏特征工程，这边偏模型架构 / 训练 / 后处理。

整体 takeaway（先看结论）：

| 比赛 | 数据类型 | 1st 用什么模型 | NN/GBDT 比例 | 主要 trick |
|---|---|---|---|---|
| Optiver Realized Volatility 2021 | 10 分钟窗口 LOB 统计 | **Nearest Neighbors**（数据 leak） + LGBM/NN ensemble | 主流方案 60% LGBM + 40% NN | time-id reverse engineering |
| Optiver Trading at the Close 2023 | 10 分钟收盘竞价 | **CatBoost (0.5) + GRU (0.3) + Transformer (0.2)** | 50% GBDT + 50% NN | online learning + post-processing |
| Jane Street Market Prediction 2020 | tabular 130 anonymized features | **AE-MLP + XGBoost** | NN 主导（autoencoder pretrain）+ XGB stacking | utility-based loss, denoised target |
| Jane Street Real-Time 2024 | tabular real-time market | **Ensemble of MLP + Residual MLP + variants** | NN 主导，无 GBDT | dynamic blending, market regime aware |

下面分别展开每个比赛。

---

## 1. Optiver Realized Volatility Prediction 2021

- 比赛页：https://www.kaggle.com/competitions/optiver-realized-volatility-prediction
- 1st 方案讨论页：https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
- 评分：RMSPE on realized volatility (10 min ahead)

### 1st place — Nearest Neighbors

**关键 trick**：用 **价格 tick size + LOB 微结构** **逆向工程 time_id 的真实顺序**——比赛数据被打乱了 time_id，1st place 还原顺序后能用相邻 time 的信息（强 leak）。这是数据漏洞，不是真正的模型胜出。

**模型**：还原顺序后用 KNN 在还原的时间序列上做 lookup-based 预测。

**对我们没直接借鉴价值**（我们没有 leak）。

### Top 5 主流方案（more representative）

学术综述（IEEE Xplore "LightGBM Based Optiver Realized Volatility Prediction"）和高排名讨论中，**主流是 LightGBM + Neural Network ensemble**：

1. **特征工程**（这是冠军共识）：
   - 1 个 time bucket 内有 ~600 个 3 秒 LOB 快照
   - 每 stock × time_id 提取 ~400-700 derived features
   - 特征族：spread/depth 统计（mean, std, skew, kurt）、log return 的 realized volatility、order book imbalance、bucket 内多个时间窗口（30s/60s/120s/300s/600s）的衍生统计
2. **模型**：单 LightGBM 强基线（dart 或 gbdt），再加 Neural Network 补充。NN 通常是简单 MLP。
3. **CV**：K-Fold by time_id（一种特殊 group fold，每 fold 划分一致）。
4. **Ensemble**：LightGBM + NN 加权平均，权重 0.7/0.3 或 0.6/0.4 是常见。
5. **来源**：https://ieeexplore.ieee.org/document/9543438/ ; https://www.atlantis-press.com/proceedings/icfied-22/125971589

### 对我们比赛的启示

- 高频预测里**特征工程仍然是 alpha 主源**——700 个 derived features 是常态。
- LightGBM 是 default first-shot，dart booster 比 gbdt 更稳。
- NN 作为 GBDT 的 ensemble 队友提分，但不会是单独冠军。

---

## 2. Optiver Trading at the Close 2023

- 比赛页：https://www.kaggle.com/competitions/optiver-trading-at-the-close
- 1st 方案：https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- 1st author Twitter: https://x.com/hydantess1993/status/1773982572537581799
- 评分：MAE on closing price movement vs synthetic index (10 min)

### 1st place — hyd

**模型 ensemble**：CatBoost (0.5) + GRU (0.3) + Transformer (0.2)。
- 注意：**CatBoost 权重最高**，但 NN 占了 50%——典型 GBDT-NN hybrid。
- 300 features（重度 feature engineering）。

**关键 trick**：
1. **Online learning every 12 days**（5 次 retrain）—— 因为评测数据是 forward in time，分布有漂移。
2. **Post-processing** 对 score 提升明显（具体没公开但确认有用）。
3. CV strategy: purged K-Fold（5 folds，2 day purge gap）。

### 高排名其他方案（如 fan2goa1 第 186 名 LightGBM only）

- 单 LightGBM (n_estimators=6300-7000, subsample=0.7)。
- 三层特征工程：基线 → 高级 momentum/imbalance → 时序 shift/pct_change/rolling stats。
- Purged K-Fold CV。
- Score：5.3341 (公开榜)。

### 对我们比赛的启示

- **CatBoost 在 LOB 类金融 tabular 上是最强 GBDT 之一**——比 LightGBM 更稳，handles categorical 更好。
- **小比例 NN ensemble (~30-50%) 加分**，但单 NN 不会是冠军。
- **Online retrain** 思路对我们也适用——120 天数据，可以模拟"训练前 90 天 + last 30 天 fine-tune"。
- **Purged K-Fold（带 gap）** 是 LOB CV 的标配（防 look-ahead）。
- **Post-processing 对 final score 影响巨大**，应作为独立优化目标。

---

## 3. Jane Street Market Prediction 2020-2021

- 比赛页：https://www.kaggle.com/competitions/jane-street-market-prediction
- 1st place repo: https://github.com/MingjieWang0606/Kaggle-Jane-Street-AE-MLP-xgb-TOP1
- 评分：utility function on weighted action selection

### 1st place — AE-MLP + XGBoost

**模型**：
1. **Autoencoder（pretrain）**：在所有 features 上做无监督 reconstruction 学嵌入。
2. **MLP head**：把 AE 的 latent 接 MLP 多任务分类（多 target ensemble）。
3. **XGBoost stacking**：把 MLP 输出 + 原 features 喂 XGBoost 做 final blend。

**Key insight**：AE pretrain 把 130 个匿名 features 压成 disentangled latent，MLP 学得更稳。

**Training tricks**：
- Multitask loss（5 个相关 target 同时学，shared encoder）。
- Skip connections in MLP。
- 高 dropout (0.4-0.5)。

### 其他高排方案（scaomath, evgeniavolkova）

- **scaomath（241st, top 5.7%）**：AE+MLP，3-fold ensemble，TF + PyTorch 混合。
- **evgeniavolkova（Jane Street 2024）**：utility regularizer fine-tuning every 10 epochs，denoised target，dynamic ensemble based on market regime。

### 对我们比赛的启示

- **AE pretrain 对匿名 feature 有用**——我们的 154 LOB feature 是已知含义的，AE 收益小（不需要"理解"什么 feature 重要）。
- **Multitask shared encoder 对 5 horizon 多 head 有效**——可以借鉴：5 个 horizon 共享 backbone + 5 个 head + multitask loss。
- **Utility/PnL-based regularizer**（直接在 cross-entropy 之外加 PnL 项）——对我们 PnL 评分的比赛**直接相关**。
- **Dropout 高一点（0.3-0.5）** 是金融 NN 防过拟合标配。

---

## 4. Jane Street Real-Time Market Data Forecasting 2024

- 比赛页：https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting
- 启动：2024-10-14
- 评分：weighted utility on real-time market data
- 1st place 来源未完全公开（截至 2026-05），但 GitHub 有 evgeniavolkova 的开源 baseline solution

### evgeniavolkova 高排方案（2024 版）

**模型** (from solution.md / discussion 556542)：
1. **Tensorflow Autoencoder + small MLP w/ skip connection**
2. **PyTorch ResMLP with embeddings for high-cardinality features (~300-400k params)**
3. **Tensorflow Residual MLP with high-dropout filter layer**

**训练 tricks**：
- **Utility function regularizer**（competitive metric directly into loss），每 10 epochs 应用。
- **Denoised target**：通过 covariance eigen-decomposition 去噪 target 后训练（reduce overfitting）。
- **10-day gap grouped temporal CV**（防 leakage）。
- **40% zero-weight rows replaced with 1e-7**（特殊正则）。
- **Dynamic ensemble**：根据 `feature_64 average gradient + previous-day trade count` 切换不同模型组合（市场状态自适应）。
- **CPU inference faster than GPU**（这点对我们比赛 ≤3h CPU/GPU 很重要）。

### 对我们比赛的启示

- **NN 主导 + 多模型 dynamic blending**——市场状态切换 trick 在 PnL 类比赛上有用。
- **Utility-aware loss** 在 Jane Street 2024 也证实有效。
- **小 MLP + 高 dropout + skip connection** 是简单稳健的金融 baseline，参数 300-400k。

---

## 5. （Bonus）Jane Street 2024 关键讨论 takeaways

- 高排队伍普遍用 **ensemble of N model variants**（typical N = 5-10）。
- **No GBDT in top NN solutions**（Jane Street 数据是匿名 dense features，GBDT 在 anonymized continuous 数据上弱于 NN）。
- 但 **Optiver 2023** 这类显式 LOB 类比赛 **GBDT 是核心**。
- 我们的 LOB 比赛**更像 Optiver 2023**（有明确语义 feature） → **GBDT 应是 baseline**，NN 作为 ensemble 队友。

---

## 6. 综合 takeaways（给我们模型选型）

1. **GBDT (CatBoost/LightGBM) 是几乎所有显式金融 tabular 比赛的 baseline**，在 LOB 任务上几乎不会输。
2. **NN 作为 ensemble 提分项 (20-40% 权重)**，但单 NN 不会是冠军。
3. **Heterogeneous ensemble** 比同质强：CatBoost + LightGBM + GRU + DeepLOB 优于 5 seed CatBoost 平均。
4. **Online learning / fine-tune on tail** 在长时序漂移数据上有用（Optiver 2023 5 轮 retrain）。
5. **Purged K-Fold CV with gap** 是金融 ML 的标配。
6. **Utility/PnL-aware loss** + **denoised target** + **dynamic blending** 是 Jane Street 系列 winner 共识。
7. **Dropout 0.3-0.5 + skip connection + 高正则** 是金融 NN 防过拟合标配。
8. **Post-processing 对 final score 的影响在比赛中常常 = 多个 model variant 加起来的提升**——这是 PnL 评分比赛的最大杠杆，必须独立调优。

---

## 7. 引用清单

- Kaggle Optiver Realized Vol 2021 1st place writeup: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
- Kaggle Optiver Trading at the Close 2023 1st place: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- Kaggle Jane Street 2020 1st place repo: https://github.com/MingjieWang0606/Kaggle-Jane-Street-AE-MLP-xgb-TOP1
- Kaggle Jane Street 2024 evgeniavolkova solution: https://github.com/evgeniavolkova/kagglejanestreet
- LightGBM-NN paper (Optiver vol 2021 inspired): https://www.atlantis-press.com/proceedings/icfied-22/125971589
- fan2goa1 Optiver 2023 blog: https://fan2goa1.github.io/mkdocs-material/blog/2023/12/24/kaggle-optiver---trading-at-the-close/
