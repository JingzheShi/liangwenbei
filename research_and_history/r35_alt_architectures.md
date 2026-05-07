# R35 — 替代模型架构调研

> 目标：iter_006 = LightGBM 5-seed h_60 + DE 4D thresh, LOSO sum **+13.61**。
> NN 已 3 次失败（T1 / T19 h_10 → T32 h_60 单 seed = **−16.97**）。
> 226-d Scheme C 已饱和（T39）。本文评估 12 种替代架构，看下一个 +Δ 应该来自哪里。
>
> 数据规模：1.47 M train + 0.29 M val + 0.44 M test，**单 fold ~88 k**，h_60 horizon。
> label 3-class，60-80 % flat。**硬约束**：sym-agnostic / stateless / no-date-feature。

---

## 0. 评估表（一览）

| # | 架构 | OOD robustness | 预期 LOSO Δ vs iter_006 | GPU/CPU 时间 | 代码量 | 是否值得做 |
|---|---|---|---|---|---|---|
| 1 | TabPFN v2 / 2.5 | ⭐⭐⭐ (bayesian posterior) | **不可行**（行数超限） | — | — | ❌ |
| 2 | FT-Transformer / SAINT | ⭐⭐ | −5 ~ +1（NN 反复失败有先例） | 8-15 h GPU × 5 fold | 中 | ⚠️ 仅 ensemble 队友 |
| 3 | MLP + LayerNorm + dropout（MLPLOB style） | ⭐⭐ | −10 ~ +2 | 4-8 h GPU × 5 fold | 小 | ⚠️ 低成本 NN 重试 |
| 4 | **CatBoost h_60 5-seed** | ⭐⭐⭐⭐ | **+1 ~ +4** | 1-2 h CPU × 5 seed × 5 fold | 极小 | ✅✅✅ TOP-1 |
| 5 | **XGBoost h_60 5-seed (+monotone)** | ⭐⭐⭐ | **0 ~ +3** | 1-2 h CPU/GPU × 5 seed × 5 fold | 极小 | ✅✅ |
| 6 | **LightGBM Dart h_60 5-seed** | ⭐⭐⭐⭐ | **+1 ~ +3** | 2-4 h CPU × 5 seed × 5 fold | 极小 | ✅✅✅ |
| 7 | NGBoost / Soft Trees | ⭐⭐ | −2 ~ +1 | 2-4 h × 5 fold | 小-中 | ❌ |
| 8 | Mamba / SSM for tabular（MambaTab/Mambular） | ⭐⭐ | −5 ~ +2 | 6-12 h GPU × 5 fold | 中 | ❌ |
| 9 | LightGBM + 浅 MLP gate | ⭐⭐ | −5 ~ +2（依赖一个能学的 MLP） | NN 失败连带失败 | 中 | ❌ |
| 10 | LOBster pre-train + fine-tune | ⭐ | −5 ~ +3（domain shift 风险大） | 24+ h GPU + 数据下载 | 大 | ❌ |
| 11 | Multi-task NN（mid/vol/spread aux） | ⭐⭐ | −5 ~ +3 | 6-10 h GPU × 5 fold | 中 | ⚠️ 仅 NN 走通后 |
| 12 | **Conformal / Venn-Abers wrapper** | ⭐⭐⭐⭐ | **0 ~ +3** | < 1 h CPU | 极小 | ✅✅ |

---

## 1. TabPFN v2 / TabPFN-2.5

- **论文**：Hollmann et al. NeurIPS 2024（v2，arxiv 2502.17361 closer-look）；TabPFN-2.5（arxiv 2511.08667）
- **GitHub**：[PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)
- **核心**：In-context learning 的 prior-fitted Transformer，把训练集塞进 attention context 直接输出后验。
- **是否适合我们**：
  - **TabPFN v2 上限 10 000 samples × 500 features**。我们 train ≈ 1.47 M，**超 147×**。
  - **TabPFN-2.5 上限 50 000 samples × 2 000 features**（24 年 11 月）。仍 **超 30×**。
  - 唯一可行：**子采样 + 多次 ICL ensemble**，但每次 inference 都需要重新 attend 训练 context → 不符合 stateless 推理（评测要 ms 级）。
  - 还存在 in-context 长度对内存的 N² 开销，单条 inference 可能 100 ms+。
- **预期 LOSO 提升**：**不可行**，跳过。
- **实施成本**：—

---

## 2. FT-Transformer / SAINT

- **论文**：Gorishniy 2021（arxiv 2106.11959）；SAINT 2021。Booking.com 2024 复测（arxiv 2405.13692）：FT-T 不调参也接近 LightGBM 调参。
- **GitHub**：[yandex-research/rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models)
- **核心**：Feature tokenizer + Transformer encoder（FT-T）；SAINT 加 row-wise attention。
- **是否适合我们**：
  - 226-d 维度 OK，1.47 M 行 OK（NN 数据量足够）。
  - **但**：T1 / T19 / T32 都是 NN，且都比 LightGBM 弱 10+ 分，T32 NN-LGB ensemble alpha=0 最优 → 信号是 "NN 在我们 setup 里学不到 GBDT 学不到的 alpha"。
  - FT-T 也是 NN，理论上"缩小 vs GBDT 差距但不全面超越"（Booking.com 2024 结论）。
- **预期 LOSO Δ**：单模型 −5 ~ +1；ensemble 队友角色 0 ~ +1。
- **实施成本**：8-15 h GPU × 5 fold，代码量中等。
- **结论**：⚠️ 仅作为 NN 多样性队友（前提是先有一个跑得通的 NN）；不优先。

---

## 3. MLP with attention / ResNet for tabular（MLPLOB style）

- **论文**：MLPLOB / TLOB Berti 2025（arxiv 2502.15757）— 纯 MLP-Mixer 在 FI-2010 上接近/超过 DeepLOB。
- **核心**：feature-mixing MLP + temporal-mixing MLP + LayerNorm + GeLU + dropout 0.3-0.5；可选 SWA、cosine LR、label smoothing。
- **是否适合我们**：
  - DeepLOB（CNN）和 deep dropout NN 都失败了。**MLPLOB 没试过**——它简单到几乎没有 inductive bias 错误。
  - 但 T32（DeepLOB h_60，dropout 0.4/0.5、wd 5e-3、label_smooth 0.1、aug_a）已是强正则版本，**结果仍 OOF gap 巨大**（fold 1 val +2.97 → test −10.81）。
  - 暗示：**问题可能不是 NN 架构，而是 226-d 特征的"非 sym-agnostic 信号"被 NN 过度利用**。GBDT 通过分裂阈值容忍噪声更好。
- **预期 LOSO Δ**：单模型 −10 ~ +2；NN-LGB ensemble 0 ~ +1。
- **实施成本**：4-8 h GPU × 5 fold（MLPLOB 比 DeepLOB 还小），代码量小。
- **结论**：⚠️ 一次低成本 NN 重试值得做，但优先级 < CatBoost/Dart。把它列入 top5 备胎。

---

## 4. CatBoost (h_60 完整) ⭐⭐⭐

- **论文**：Prokhorenkova et al. NeurIPS 2018；Optiver 2023 1st 用它（权重 0.5）。
- **GitHub**：https://github.com/catboost/catboost
- **核心**：Ordered boosting（每棵树用之前 sample 的 OOF residual） → 减少 **target leakage** 与 overfit；内置 categorical handling；支持 GPU。
- **是否适合我们**：
  - T17 仅在 **h_10 + 单 seed** 跑过，sum cum_pnl ≈ 11.32（LightGBM h_10 单 seed 同等水平）。
  - **核心 horizon h_60 完全没试**——iter_006 是 5-seed h_60，CatBoost h_60 5-seed 是最 obvious 的下一步。
  - Ordered boosting 理论上对**跨 sym OOD 更稳**（每棵树的树叶估计不依赖当前 sample 自己）。
  - 与 LightGBM 的 split 策略不同 → **天然 ensemble 多样性**。
- **预期 LOSO Δ**：CatBoost 单独 LOSO sum ≈ iter_006 ± 2；**ensemble (LGB + CB) 5-seed h_60 → +1 ~ +4**。
- **实施成本**：
  - 训练：CatBoost-GPU h_60 ≈ 5-10 min/seed/fold × 5 × 5 = 2-4 h GPU。
  - 推理：CatBoost CPU 推理（提交端）每窗 < 5 ms，OK。
  - 代码：复用 T17 的 `train_loso.py`，改 horizon 和 seed list。
- **结论**：✅✅✅ **TOP-1 优先级，必做**。

---

## 5. XGBoost (h_60 完整) + monotone constraints

- **论文**：Chen & Guestrin 2016；XGBoost 2.0 (2024) 加 GPU device API。
- **核心**：与 LightGBM 类似的 GBDT，但 split policy（depth-wise）不同；可设 `monotone_constraints` 强制某 feature 单调（如 OFI 单调正）。
- **是否适合我们**：
  - T18 h_10 单 seed 跑过，sum ≈ 6.45（fold 0 折戟 −4.15）→ XGBoost 默认参数对 OOD fold 0 不稳。
  - **h_60 没试**。XGBoost h_60 完整 5-seed 仍是个 obvious 的 ensemble 队友。
  - **monotone constraints** 的 trick：T39 显示 226-d 特征里有几个高重要度的方向性特征（best_bid/ask price spread, OFI），强制单调可减 OOD 噪声。
- **预期 LOSO Δ**：XGB 单独 ≈ iter_006 − 2 ~ + 1；**LGB + CB + XGB 三 GBDT 加权 ensemble → +1 ~ +3**。
- **实施成本**：1-2 h GPU × 5 seed × 5 fold；代码极小（复用 T18，加 horizon=60 和 monotone_constraints dict）。
- **结论**：✅✅ 高优先级。先做 CatBoost，确认 GBDT 多样性能加分后再推 XGBoost。

---

## 6. LightGBM Dart mode（drop-tree boosting）⭐⭐⭐

- **论文**：Vinayak & Gilad-Bachrach 2015 "DART"；多篇 2024 金融时序文章选 DART 作为 LGBM booster（Nature HSS Comm 2025、MDPI Systems 2025）。
- **核心**：每次 iter 随机 drop 一部分已加入的树，强制后续树补偿被 drop 的部分 → **类似 NN dropout 的 ensemble averaging 效应**。
- **是否适合我们**：
  - 我们 LightGBM 一直用 `boosting_type='gbdt'`。**Dart 模式没系统试过**。
  - Dart 已被多篇金融时序文章证明 **比 gbdt 减 13% MSE / 缩小 train-test gap 22%**。
  - h_60 LGB 单 seed 已饱和 → Dart 内置正则可能挖出额外 +1 ~ +3。
  - **风险**：Dart 训练慢 2-3×，且不能用 `early_stopping_rounds`（需要固定 num_iter）。
- **预期 LOSO Δ**：单 Dart 5-seed h_60 比 gbdt 5-seed +0 ~ +2；**Dart + gbdt 互补 ensemble → +1 ~ +3**。
- **实施成本**：2-4 h CPU × 5 seed × 5 fold（Dart 慢 2-3×）；代码极小（只改 params）。
- **结论**：✅✅✅ **TOP-1 并列**。低代码、高确定性。

---

## 7. NGBoost / Soft Decision Trees / Sparse Decision Forest

- **NGBoost**（Duan 2019, arxiv 1910.03225）：natural gradient boosting，输出 distribution。
- **Soft Decision Tree**（Frosst & Hinton 2017）：可微决策树。
- **Sparse Decision Forest**：DT 的 L1 reg 版。
- **是否适合我们**：
  - NGBoost 优势是输出 calibrated 分布，但我们 3 分类不是回归，且 calibrated prob 可由 #12 conformal 更便宜地拿到。
  - Soft tree / Sparse DF 在 LOB tabular 上没有强 evidence（搜不到 2024+ 强 result）。
  - 训练慢 5-10×，代码量中等。
- **预期 LOSO Δ**：−2 ~ +1。
- **结论**：❌ 跳过。NGBoost 的好处可由 #12 替代。

---

## 8. Mamba / SSM for tabular（MambaTab / Mambular）

- **论文**：MambaTab（PMC 2025）；Mambular（OpenReview 2025）。
- **核心**：Mamba SSM 把 features 当 sequence pass。声称 "outperforms FT-Transformer / TabTransformer with fewer params"。
- **是否适合我们**：
  - Tabular 特征顺序无意义，Mamba 在"feature 顺序"上没有 inductive bias。MambaTab 论文里也是"plug-and-play"非主推方案。
  - 我们 100-tick 窗口 **是序列**，但已被 flatten 成 226-d Scheme C feature → Mamba 拿到的不是序列。
  - 如果要把 100-tick 给 Mamba，需要重做特征 pipeline，成本高。
- **预期 LOSO Δ**：单模型 −5 ~ +2。
- **实施成本**：6-12 h GPU × 5 fold。
- **结论**：❌ 不优先。NN 路线整体已 3 次失败，再换 NN 架构边际期望低。

---

## 9. 混合 GBDT + 浅 NN（gating）

- **思路**：LightGBM 输出 3 个 logit + 浅 MLP（226 dim → 64 → 3）做 gate，最终预测 = α·LGB + (1-α)·MLP，α 由 gate net 学。
- **是否适合我们**：
  - T32 已证明：单 NN 预测加权进 LGB 的 ensemble，**alpha = 0 是最优**（NN 任何权重都拉低 LOSO）。
  - Gating 比简单 ensemble 更强，但前提 **gate net 能学到"何时信 NN"**，而我们的 NN 在所有 fold 都比 LGB 差 → gate 学到的最优策略也是"永远不信 NN"。
- **预期 LOSO Δ**：−5 ~ +2。
- **结论**：❌ 跳过。NN 走通前不要碰 gating。

---

## 10. Pre-train on LOBster + fine-tune

- **数据源**：LOBster（NASDAQ ITCH Level-3 message data，开源）；LOB-Bench (arxiv 2502.09172) 也聚合了这类数据。
- **核心**：在大量 NASDAQ 股票 LOB 上 self-supervised pre-train（masked LOB modeling 或 autoregressive next-event），然后 fine-tune 我们 5 sym。
- **是否适合我们**：
  - **Domain shift 巨大**：NASDAQ 是 message-level + 美股 microstructure；我们是 100-tick snapshot + A 股（推测）。Tick size、撮合规则、disposition effect 都不同。
  - LOBERT 2025（arxiv 2511.12563）和 TradeFM 2026（arxiv 2602.23784）都 reportedly 训练成本 10-100 GPU-day。我们 1 张卡复现不现实。
  - 即使只下载 pretrained checkpoint，"sym-agnostic + 100-tick window" 的 finetune 接口要重写。
- **预期 LOSO Δ**：−5 ~ +3（高方差，fine-tune 容易过拟合 5 sym）。
- **实施成本**：24+ h GPU + 100+ GB 数据下载 + 大量 glue code。
- **结论**：❌ 性价比极低。

---

## 11. Multi-task NN（auxiliary tasks: next mid / vol / spread）

- **思路**：主头 h_60 三分类；aux 头预测 t+10 mid_return、t+10 volume、t+10 spread → 共享 backbone。
- **是否适合我们**：
  - Multi-task 正则化 idea 是对的（DeepLOB-Attention 已是 multi-horizon）。
  - 但 backbone 仍然是 NN，所以基础 capability 不变。我们 NN 失败的原因不是 head 设计，而是 backbone OOD overfit。
  - aux task 加进来反而可能 **分散 capacity**，让 main head 更弱。
- **预期 LOSO Δ**：−5 ~ +3。
- **结论**：⚠️ 仅 NN 走通后再考虑（即 MLPLOB 单任务 LOSO ≥ 0 之后）。

---

## 12. Conformal prediction / Venn-Abers wrapper ⭐⭐

- **论文**：Vovk 2005；Venn-Abers Manokhin 2017；2024 finance 应用 ScienceDirect S2666548625001030。
- **核心**：用 calibration set 上的 nonconformity score，把 LightGBM 的 raw `predict_proba` **重映射成有覆盖率保证的真实概率**。Venn-Abers 比 isotonic 更稳（双 isotonic）。
- **是否适合我们**：
  - T30 试过 conformal 的早期版本（`conformal_results.json`），但**没广用**——这次重做有空间。
  - iter_006 的 4D thresh DE 优化 **隐含假设 LGB prob 是 calibrated 的**。Tree boosters 的 prob 实际偏离 sigmoid 曲线 → 阈值不在最优 sweet spot。
  - Conformal/Venn-Abers 输出 `[0,1]` 区间真概率 → 阈值可设成"要求 P(up) > 0.55 才下单"这种 **economic interpretable** 形式。
  - 完全 stateless，符合硬约束。
- **预期 LOSO Δ**：0 ~ +3（concrete 期望：阈值在**少数关键 fold** 上能改善 trade selection）。
- **实施成本**：< 1 h CPU。Sklearn 的 `CalibratedClassifierCV(method='isotonic')` 或 `mapie` 库或自写 Venn-Abers ~50 行。
- **结论**：✅✅ 高优先级。低成本、零风险（最差 = iter_006 不变）、与 #4-#6 GBDT ensemble 正交。

---

## 总结：为什么这次不优先 NN

| 已试 NN | h | 配方 | LOSO sum |
|---|---|---|---|
| T1 | h_10 | 默认 NN | 早 epoch 过拟合 |
| T19 | h_10 | dropout + wd + label smooth | val +8.99 → test −7.56（巨 OOD gap）|
| T32 | h_60 | DeepLOB + aug_a + dropout 0.4/0.5 + wd 5e-3 + LS 0.1 + SWA | **−16.97**（vs LGB +11.46） |

3 次都同一规律：**val 看起来 OK，test (held-out sym) 崩**。
说明问题在 NN 难以做到 sym-agnostic：**NN 用 226-d feature 的子空间识别"这是 sym=2 的 LOB"** → train 5 个 sym 中 4 个的隐式 sym signature → test 第 5 个崩。
GBDT 用阈值切分自然不会构造"组合特征代表 sym"，OOD 更稳。

**结论**：下一阶段优先级 = **GBDT 多样性 (CatBoost / XGBoost / Dart) + Conformal calibration**，把 ensemble + threshold 这两个杠杆推到极致。NN 再试一次（MLPLOB）作为 sanity check，但不抱预期。

---

## Sources

- [TabPFN-2.5 (arxiv 2511.08667)](https://arxiv.org/abs/2511.08667)
- [A Closer Look at TabPFN v2 (arxiv 2502.17361)](https://arxiv.org/html/2502.17361v1)
- [Revisiting Deep Learning Models for Tabular Data (FT-Transformer, arxiv 2106.11959)](https://arxiv.org/pdf/2106.11959)
- [Challenging GBDT with Tabular Transformers — Booking.com 2024 (arxiv 2405.13692)](https://arxiv.org/html/2405.13692v2)
- [TLOB / MLPLOB Berti 2025 (arxiv 2502.15757)](https://arxiv.org/abs/2502.15757)
- [LightGBM DART params](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Hybrid LightGBM credit risk 2025 (Nature HSS Comm)](https://www.nature.com/articles/s41599-025-05230-y)
- [MambaTab (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11755428/)
- [Mambular OpenReview 2025](https://openreview.net/forum?id=wElgE9qBb5)
- [LOB-Bench Nagy et al. 2025 (arxiv 2502.09172)](https://arxiv.org/abs/2502.09172)
- [LOBERT 2025 (arxiv 2511.12563)](https://arxiv.org/abs/2511.12563)
- [TradeFM JP Morgan 2026 (arxiv 2602.23784)](https://arxiv.org/abs/2602.23784)
- [Conformal prediction for electricity price forecasting 2025](https://www.sciencedirect.com/science/article/pii/S266654682500103X)
- [Venn-Abers calibration (Manokhin)](https://valeman.medium.com/how-to-calibrate-your-classifier-in-an-intelligent-way-a996a2faf718)
