# GBDT vs NN for LOB 高频预测 — 系统对比

> 这是个对我们建模优先级最关键的问题：先把工程精力 all-in GBDT，还是先 NN？
>
> TL;DR：**先 GBDT，再 NN ensemble**。结论基于 Kaggle 历届 winner + 学术 benchmark + 我们任务特点。

---

## 1. 两类方法在金融 ML 的角色对比

|  | **GBDT (LightGBM/XGBoost/CatBoost)** | **NN (DeepLOB/Transformer/MLPLOB)** |
|---|---|---|
| 输入要求 | tabular（任意维度、可混合 cat+num） | 序列 tensor（T × F），需归一化 |
| 是否需要 feature engineering | **必须**（这是 alpha 主源） | 理论上 end-to-end，但实际加 hand-crafted 也提分 |
| 训练时长 | 单卡 CPU 几分钟到几十分钟 | GPU 数小时到一天 |
| 推理时长 | 极快（树 depth × n_trees 的分支判断） | 取决于模型大小，可能需要 GPU |
| 可解释性 | 高（feature importance, SHAP） | 低（需要 attention vis） |
| 类别不均衡处理 | class_weight 直接给 | focal loss / sampler / class_weight |
| Out-of-domain 泛化 | 强（基于规则的 split） | 弱（容易 overfit 到训练分布） |
| 在新股票上推理 | 通常稳健 | 需要 sym embedding / per-sym norm |
| Early stopping | 自带 valid metric early stop | 需要手动 + EarlyStopping callback |
| 超参敏感度 | 较低 | 高 |

---

## 2. 实证证据（具体数字）

### 2.1 Kaggle 比赛 winner 选型

| 比赛 | 数据类型 | 1st 模型 | NN/GBDT 比 |
|---|---|---|---|
| Optiver Realized Vol 2021 | 10 min LOB 统计 | NN/KNN（leak） + 主流 LGBM+NN | LGBM 60%+ |
| Optiver Trading at the Close 2023 | 收盘竞价 | **CatBoost (0.5) + GRU (0.3) + Transformer (0.2)** | 50/50 hybrid |
| Jane Street Market Prediction 2020 | 匿名 130 dense feature | **AE-MLP + XGB stacking** | NN 主导 |
| Jane Street RTM 2024 | 匿名 real-time | **MLP variants ensemble** | NN 100% |
| LOB Benchmark Study 2023 (Briola) | LOB raw 40 dim × 100 tick | **BiN-CTABL** (NN, attention bilinear) | NN 100%（这是 LOB 专用 benchmark，无 GBDT 比较） |

**模式**：
- **匿名 dense feature** → NN 占优（Jane Street 系列）
- **显式 tabular feature** → GBDT 至少 50% 权重（Optiver 系列）
- **raw LOB tensor** 学术 benchmark → NN 占优，但**很少有人在这上面跑 GBDT 严格对比**

### 2.2 学术 benchmark

#### Shwartz-Ziv & Armon 2021 "Tabular Data: Deep Learning is Not All You Need"
- 4 篇 tabular DL 论文 (TabNet, NODE, DNF-Net, 1D-CNN) vs XGBoost。
- **11 个数据集 XGBoost 全胜或并列**，DL 只在自己原论文数据集上接近。
- **5 模型 ensemble (含 XGBoost) > XGBoost alone**——多样性提分。

#### Gorishniy et al. 2021 (FT-Transformer)
- FT-Transformer ≈ ResNet-Tab > NODE > TabNet。
- FT-Transformer 缩小了 vs GBDT 差距，但**没有全面超越**。
- ~50% 数据集 GBDT 仍领先。

#### LOB 专用 benchmark (Briola 2023, arxiv 2308.01915)
- 在 FI-2010 + LOB-2021 + LOB-2022 上 15 个 NN 模型严格对比。
- 没有直接 GBDT 比较，但发现：
  - **几乎所有 NN 模型在跨数据集后 F1 下降 19.6%**（BiN-CTABL 也是）。
  - 类别不均衡严重（CSCO 中 class 1 占 65%）。
  - 这种 distribution shift + class imbalance 是 GBDT 的传统强项。

#### Lucchese et al. 2024 "Deep LOB Forecasting Microstructural Guide" (arxiv 2403.09267)
- DeepLOB 在 large-tick 股票上 MCC=0.29，small-tick 上 0.11。
- **预测能力的 80% 来自 microstructure 信号**——这种 signal 是 hand-engineered features 也能 capture 的。
- 暗示：**不需要 NN 也能拿到大部分 alpha**，前提是特征工程做对。

### 2.3 实战 benchmark 数据（来自 LSTM_LightGBM 论文等）

- `LSTM_LightGBM hybrid` > 单 LSTM > 单 LightGBM（边际差 ~10-15% accuracy）。
- 但**单 LightGBM > 单 LSTM** 在大多数股票预测任务上。
- 来源：MATEC Conferences 2021，CSAIEE 2021。

---

## 3. 在我们这个比赛的特点分析

### 比赛 setup
- **数据**：5 sym × 120 day × AM/PM × 2001 tick ≈ 240 万样本。
- **特征**：154 个语义清晰的 LOB feature（10 档量价 + 订单流统计）。
- **窗口**：≤100 tick。
- **任务**：3 分类 × 5 horizon。
- **评分**：cumulative PnL（带 0.01% 双边手续费）。
- **泛化压力**：评测时 sym 可能为训练外的股票。

### 任务特点决定 GBDT 优势
1. ✅ **154 维 explicit feature**（非匿名）—— GBDT 主场。
2. ✅ **handcrafted feature engineering 空间大**——R-feature worker 在做这块。
3. ✅ **跨股票泛化要求高**——GBDT 基于规则的分裂比 NN 更稳。
4. ✅ **类别不均衡严重**（label=1 占大多数）——GBDT 的 class_weight 直接处理。
5. ✅ **CPU 评测预算 3h**——GBDT 推理比 NN 快得多。
6. ✅ **手续费阈值后处理需要 calibrated probability**——GBDT 输出的 `predict_proba` 通常比 NN softmax 更 calibrated（特别是 LightGBM with proper isotonic calibration）。

### 同时 NN 的合理位置
1. **DeepLOB 类专门架构**捕捉 LOB 的 spatial-temporal 结构，这是 flatten 后 GBDT 难以完全替代的。
2. **Multi-horizon shared encoder** (DeepLOB-Attention) 对 5 head 任务自然。
3. **Ensemble 多样性**——单 GBDT 永远比不上 GBDT + NN 多样性 ensemble。
4. **bid-ask Siamese 对称性 prior**（参考 arxiv 2505.22678）只有 NN 能利用。

---

## 4. 推荐的执行顺序

### Stage 0：Baseline 比较（1 周）
1. **DeepLOB** 复刻 mmpc_demo（已有）—— 验证 PnL pipeline 能跑通。
2. **LightGBM 基线**：把 100×154 窗口 → flatten 成 vector + last_tick + rolling stats → LGBM 多分类。
3. **CatBoost 基线**：同 LGBM 但用 CatBoost（金融场景常更稳）。
4. **比 PnL**——直接在我们 PnL evaluator 上看 5 horizon 累计 PnL。

### Stage 1：GBDT 深度优化（2 周）
5. 重特征工程（R-feature worker 给的 100+ 候选 feature）。
6. **5 个 horizon 各训一个 GBDT** vs **多任务 GBDT (mc-multioutput)**——对比。
7. **Class weight + Focal loss for LightGBM** 处理不均衡。
8. **Purged K-Fold (with 1-day gap)** CV。
9. Optuna 调超参。

### Stage 2：NN 补强（1-2 周）
10. **MLPLOB（TLOB paper 的 MLP-only baseline）**—— 简单且强，先跑。
11. **BiN-CTABL**（Briola 2023 benchmark 上最稳的 SOTA）。
12. **DeepLOB-Attention** 加 multi-horizon decoder 共享 backbone。
13. **Siamese parameter sharing** 加 bid-ask 对称性 prior。

### Stage 3：Ensemble + 后处理（1 周）
14. **Stacking**：[GBDT, NN] 各自 prob → meta-learner LightGBM 融合。
15. **Threshold tuning by PnL**：不是 argmax，而是按 PnL 找最优阈值。
16. **Calibration** (Platt / Isotonic) on validation set。
17. **Multi-seed averaging** 各模型 5-10 seed 平均。

---

## 5. 核心结论

**先 GBDT，再 NN，最后 ensemble**。

### 我们应该首先投入 GBDT 的 5 个理由

1. **Kaggle 历届 LOB 类比赛 GBDT 至少占 50%**（Optiver 2023）——经验数据。
2. **154 explicit feature 是 GBDT 主场**——`Tabular DL is not all you need`。
3. **类别不均衡 + 跨股票泛化**对 GBDT 友好——LOB benchmark 显示 NN 在跨域时崩盘 19.6%。
4. **手续费阈值后处理**需要 calibrated probability，GBDT 优势。
5. **CPU 评测限制 + 推理速度** 让 GBDT 是 2GB / 3h 模型限制下最稳的选择。

### 但不放弃 NN 的 3 个理由

1. **Spatial-temporal 结构** 100×154 窗口 flatten 后会丢信息，专门 NN 能恢复。
2. **Ensemble 多样性提分** 是高分必经之路。
3. **bid-ask 对称性、跨 horizon 共享** 等 inductive bias 只有 NN 能塞进去。

### 一句话决策

> 把第一个月的 70% 精力放在特征工程 + GBDT 调参 + post-processing；30% 探索 DeepLOB-Attention / MLPLOB。第二个月 ensemble。

---

## 6. 引用

- Shwartz-Ziv & Armon 2021: https://arxiv.org/abs/2106.03253
- FT-Transformer (Gorishniy 2021): https://arxiv.org/abs/2106.11959
- LOB Benchmark Study (Briola 2023): https://arxiv.org/abs/2308.01915
- Deep LOB Forecasting Microstructural Guide (Lucchese 2024): https://arxiv.org/abs/2403.09267
- TLOB / MLPLOB (Berti 2025): https://arxiv.org/abs/2502.15757
- Siamese LOB (2025): https://arxiv.org/abs/2505.22678
- Kaggle Optiver 2021 1st place: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970
- Kaggle Optiver 2023 1st place: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- LightGBM-NN ensemble paper: https://www.atlantis-press.com/proceedings/icfied-22/125971589
- A Comparative Study of TabNet and XGBoost: https://www.researchgate.net/publication/392509412
