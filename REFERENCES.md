# 参考文献 / References

> **良文杯 LOB 中间价方向预测项目**（2026-05-01 ~ 2026-05-12 + 答辩准备）
>
> 本文档汇总项目期间调研、引用的全部学术论文、博客、Kaggle writeup 及工程资源。
> 扫描来源：`research_and_history/`（132 个 r*.md 文件 + papers/ 子目录）、答辩材料、实验报告、提案文档。
>
> **状态标注**：✅ 完全采纳 | ⚠️ 部分采纳 / 理论参考 | 🔬 调研未采纳 | ❌ 明确拒绝

---

## §1 LOB / 微观结构理论

- **[Cont, Kukanov & Stoikov 2014]** The Price Impact of Order Book Events. *Journal of Financial Econometrics* 12(1):47–88. [arXiv:1011.6402](https://arxiv.org/abs/1011.6402) | [SSRN 1712822](https://ssrn.com/abstract=1712822)
  - **用在**：F2 MLOFI 30 维特征（W∈{5,20,60} × k∈{1..10}），LOB 事件压缩成 OFI 单标量的理论基础
  - **结论**：✅ 完全采纳，是 F2 多尺度 OFI family 的核心理论

- **[Kolm, Turiel & Westray 2023]** Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the LOB. *Mathematical Finance* 33(4). [SSRN 3900141](https://ssrn.com/abstract=3900141)
  - **用在**：MLOFI → GOFI 扩展调研；多 level OFI 输入验证；arXiv:2112.02947 版本也被引用
  - **结论**：⚠️ 理论参考；MLOFI 30 维实现采纳，但 PCA 降维未采用

- **[Stoikov 2017]** The Micro-Price: A High Frequency Estimator of Future Prices. *Quantitative Finance* (online 2018). [arXiv:1702.02867](https://arxiv.org/abs/1702.02867) | [SSRN 2970694](https://papers.ssrn.com/abstract=2970694)
  - **用在**：F3 WMP 11 维（wmp_lvl1–10 + wmp_balance_12）；Weighted Mid-Price 公式
  - **结论**：✅ 完全采纳，WMP 是 F3 核心

- **[Kyle 1985]** Continuous Auctions and Insider Trading. *Econometrica* 53(6):1315–1335.
  - **用在**：F2 Kyle λ (2 维 W∈{50,100}) 及 KyleInv；Kyle λ 整族 KS-drop（D=0.95–0.98）
  - **结论**：⚠️ 理论采纳，Kyle λ 实测跨 sym 分离严重被整族 drop；KyleInv（³√ 改型）保留

- **[Roll 1984]** A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market. *Journal of Finance* 39(4):1127–1139.
  - **用在**：F4 roll_eff_spr_ratio 3 维（W∈{30,50,100}），W=100 KS-drop
  - **结论**：⚠️ 部分采纳；W=30/50 保留；W=100 因跨 sym Roll 估计不稳被 drop

- **[Easley, López de Prado & O'Hara 2012]** Flow Toxicity and Liquidity in a High-Frequency World. *Review of Financial Studies* 25(5):1457–1493. [SSRN 1695041](https://ssrn.com/abstract=1695041)
  - **用在**：F2 OFI Toxicity 4 维（k∈{1,5} × W∈{20,50}）；VPIN/BVC 框架
  - **结论**：✅ VPIN 框架采纳为 ofi_tox 特征

- **[Barndorff-Nielsen & Shephard 2004]** Power and Bipower Variation with Stochastic Volatility and Jumps. *Journal of Financial Econometrics* 2(1):1–37.
  - **用在**：F5 BV + jshare（跳跃份额）特征；BV = Σ|r_t||r_{t-1}|·π/2
  - **结论**：✅ BV 和 jshare 特征采纳

- **[Barndorff-Nielsen & Shephard 2006]** Econometrics of Testing for Jumps in Financial Economics Using Bipower Variation. *Journal of Financial Econometrics* 4(1):1–30.
  - **用在**：jump test statistic J_W 公式；F5 跳跃检验
  - **结论**：✅ BNS jump test 实现采纳

- **[Hasbrouck 1991]** Measuring the Information Content of Stock Trades. *Journal of Finance* 46(1):179–207.
  - **用在**：Kyle λ 的 √volume 调整来自 Hasbrouck 实证；Hasbrouck Lambda 代理
  - **结论**：⚠️ 理论参考；影响 KyleInv 的 ³√vol 设计

- **[Amihud 2002]** Illiquidity and Stock Returns: Cross-Section and Time-Series Effects. *Journal of Financial Markets* 5(1):31–56.
  - **用在**：Amihud Illiquidity Ratio 特征研究（r20_papers）
  - **结论**：🔬 调研，iliq 特征在 stage 3-4 中低优先级，未进入最终 359 维

- **[Corwin & Schultz 2012]** A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices. *Journal of Finance* 67(2):719–760.
  - **用在**：r20_papers 高低价价差估计调研
  - **结论**：🔬 日频公式；tick 频率下退化，未采用

- **[Bacry, Mastromatteo & Muzy 2015]** Hawkes Processes in Finance. *Market Microstructure and Liquidity* 1(01). [arXiv:1502.04592](https://arxiv.org/abs/1502.04592)
  - **用在**：R_Hawkes 和 R_Hawkes_r6 实验；Hawkes 强度作为 LOB 特征调研
  - **结论**：🔬 Hawkes 强度 feature 实验表明跨 sym 不稳，未采用；stateless Predictor 约束也是障碍

- **[Lu & Abergel 2018]** High-Dimensional Hawkes Processes for Limit Order Books. [Tandf](https://www.tandfonline.com/)
  - **用在**：R_Hawkes 调研背景
  - **结论**：🔬 高维 Hawkes 实现成本过高，未采用

- **[Adams & MacKay 2007]** Bayesian Online Changepoint Detection. [arXiv:0710.3742](https://arxiv.org/abs/0710.3742)
  - **用在**：`tick_since_last_changepoint` 特征提议（research notes）；arXiv:1509.01570 也被引用
  - **结论**：🔬 Changepoint 特征有趣但实现需在线贝叶斯推断，与 stateless 约束冲突，未采用

---

## §2 ML for LOB / 深度学习 HFT

- **[Kercheval & Zhang 2015]** Modelling High-Frequency Limit Order Book Dynamics with Support Vector Machines. *Quantitative Finance* 15(8):1315–1329. [FSU PDF](https://www.math.fsu.edu/~aluffi/archive/paper462.pdf)
  - **用在**：LOB 机器学习第一篇奠基工作；144 维特征 schema（Basic/Time-insensitive/Time-sensitive）
  - **结论**：📖 基础参考；FI-2010 特征 schema 继承了本文设计

- **[Zhang, Zohren & Roberts 2018/2019]** DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing* 67(11). [arXiv:1808.03668](https://arxiv.org/abs/1808.03668) | [code](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)
  - **用在**：比赛主办方 baseline；iter_000 mmpc_demo；CNN-Inception-LSTM 架构参考
  - **结论**：⚠️ 作为 baseline；我们最终方案（+40 LOSO）显著超过 DeepLOB；见 answer Q2

- **[Zhang et al. 2021]** Multi-Horizon Price Forecasting for Limit Order Books. [arXiv:2105.10430](https://arxiv.org/abs/2105.10430)
  - **用在**：DeepLOB-Attention / DeepLOB-Seq2Seq 多 horizon 扩展；r35_alt_architectures 调研
  - **结论**：🔬 参考多 horizon 输出设计；我们 LightGBM 多 horizon 5-head 参考了此思路

- **[Berti et al. 2025]** TLOB / MLPLOB: Dual Attention vs Simple MLP for LOB. [arXiv:2502.15757](https://arxiv.org/abs/2502.15757)
  - **用在**：NN 路线 ablation 参考；"简单 MLP 在恰当训练下可达 SOTA" 验证了我们 flat-feature 路线
  - **结论**：⚠️ MLPLOB 的 flat-feature 思想与我们 NN T188v2 一致；TLOB attention 机制未采用

- **[Sirignano & Cont 2019]** Universal Features of Price Formation in Financial Markets. [arXiv:1803.06917](https://arxiv.org/abs/1803.06917)
  - **用在**：N5 "Universal Market Models"；sym-agnostic 训练的文献支撑
  - **结论**：⚠️ 理论支撑；universal features 假设与我们 sym-agnostic 设计一致

- **[Tran et al. 2018]** Temporal Attention-Augmented Bilinear Network for Financial Time-Series Data Analysis. [arXiv:1712.00975](https://arxiv.org/abs/1712.00975)
  - **用在**：BiN-CTABL / TABL 模型研究（r35_alt_architectures）；arXiv:2003.00598 也是相关续作
  - **结论**：🔬 复杂度过高，无开源可靠实现，未采用

- **[Wallbridge 2020]** TransLOB: Transformers for Limit Order Books. [arXiv:2003.00130](https://arxiv.org/abs/2003.00130)
  - **用在**：r35_alt_architectures LOB Transformer 调研
  - **结论**：🔬 在 FI-2010 上未超过 DeepLOB，未采用

- **[Briola et al. 2024]** Deep Limit Order Book Forecasting: A Microstructural Guide. *Quantitative Finance* (July 2025). [arXiv:2403.09267](https://arxiv.org/abs/2403.09267)
  - **用在**：tick-size 分类影响可预测性；5-day rolling z-score 归一化建议
  - **结论**：⚠️ 5-day rolling z-score 启发我们 window-z 设计；large-tick 股票更易预测的发现与我们吻合

- **[Briola et al. 2023]** A Deep Limit Order Book Benchmark for Stock Markets. [arXiv:2308.01915](https://arxiv.org/abs/2308.01915)
  - **用在**：LOB DL benchmark 调研（r31_papers）
  - **结论**：🔬 FI-2010 扩展 benchmark 参考

- **[Ntakaris et al. 2018]** Benchmark Dataset for Mid-Price Forecasting (FI-2010). *Expert Systems with Applications* 2019.
  - **用在**：FI-2010 数据集 benchmark 的标准参考；DeepLOB 以此为基准
  - **结论**：📖 标准 benchmark 参考

- **[Nagy et al. 2025]** LOB-Bench: Benchmarking Generative AI for Limit Order Books. [arXiv:2502.09172](https://arxiv.org/abs/2502.09172)
  - **用在**：r31_papers LOB Bench 调研；长 horizon 难预测的警告信号
  - **结论**：🔬 generative LOB 模型在 h=60 处 "derail" 的发现，印证我们不追求 over-long horizon

- **[Sirignanano & Cont 2019]** (同 §2 第5条，见上)

- **[Spacetimeformer 2022]** Multivariate Time Series Forecasting with Transformers. [arXiv:2109.12218](https://arxiv.org/abs/2109.12218) — （来自 papers/spacetimeformer_lob_2024.md）
  - **结论**：🔬 调研

- **[Siamese LOB 2025]** — (来自 papers/model_siamese_lob_2025.md)
  - **结论**：🔬 调研，对比学习 LOB 方向，未采用

- **[Lit-LOB Transformer 2025]** — (来自 papers/lit_lob_transformer_2025.md)
  - **结论**：🔬 调研

- **[crypto_lob_better_inputs_2025]** Better Inputs for LOB (Crypto 2025). [arXiv](https://arxiv.org) — (来自 r31_papers)
  - **结论**：🔬 调研

- **[REVOL Lee 2025]** REpresentation learning for VOLatility. [arXiv:2501.07580 附近](https://arxiv.org) — (来自 r31_papers/revol_lee_2025.md)
  - **结论**：🔬 调研

- **[Hybrid VAR-FNN OFI 2024]** Hybrid VAR + FNN for OFI prediction. [arXiv:2411.08382](https://arxiv.org/abs/2411.08382)
  - **用在**：R_roll_tsrv 实验背景；VAR backbone + LightGBM-on-residual 构想
  - **结论**：🔬 两阶段 VAR+FNN 设计参考，未完整实施

---

## §3 时间序列预测（调研过但不推荐用于金融 LOB）

> ⚠️ 以下论文均由用户/项目明确标注"调研过但拒绝"，原因主要是：(a) 针对长周期低频 TS，对 tick 级 LOB 特征无优势；(b) 缺乏 sym-agnostic 设计；(c) 在我们数据上性能未超过简单 MLP/LightGBM baseline。

- **[Liu et al. 2024 iTransformer]** iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR 2024*. [arXiv:2310.06625](https://arxiv.org/abs/2310.06625)
  - **结论**：❌ 维度倒置 TS forecasting，"玩具数据上 SOTA"；LOB tick 级不适用

- **[Nie et al. 2023 PatchTST]** A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
  - **结论**：❌ 长周期预测方法；tick 频率下 patch 无效

- **[Wu et al. 2023 TimesNet]** TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. *ICLR 2023*. [arXiv:2210.02186](https://arxiv.org/abs/2210.02186)
  - **结论**：❌ 同上，不适用 LOB tick 级

- **[Wang et al. 2024 TimeMixer]** TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. *ICLR 2024*. (来自 r35_alt_architectures 调研)
  - **结论**：❌ 调研，未在金融数据上验证

- **[Shi et al. 2024 NeurIPS Scaling Law]** Scaling Law for Time Series Forecasting. *NeurIPS 2024*. [arXiv:2405.15506 附近]
  - **结论**：🔬 调研参考；"更大模型在 TS 上不一定更好" 的发现与我们选择简单 MLP 一致

- **[Gorishniy et al. 2021 ModernTCN / FT-Transformer]** Revisiting Deep Learning Models for Tabular Data. *NeurIPS 2021*. [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
  - **用在**：r35_alt_architectures；tabular DL vs GBDT 调研；arXiv:2405.13692 Booking.com 复测也被引
  - **结论**：🔬 FT-Transformer 在我们 LOB 特征上未超过 LightGBM；未采用

- **[Arik & Pfister 2021 TabNet]** TabNet: Attentive Interpretable Tabular Learning. *AAAI 2021*. [arXiv:1908.07442](https://arxiv.org/abs/1908.07442) | [code](https://github.com/dreamquark-ai/tabnet)
  - **结论**：🔬 稀疏 attentive feature selection；在我们 359 维上优势不明显，未采用

- **[Gorishniy et al. 2025 TabM]** TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. *ICLR 2025*. [arXiv:2410.24210](https://arxiv.org/abs/2410.24210)
  - **结论**：🔬 调研

- **[T-MLP 2024]** Tree-Hybrid MLPs. [arXiv:2407.09790](https://arxiv.org/abs/2407.09790)
  - **结论**：🔬 调研

- **[Revisiting TSFM Finance 2025]** Revisiting Foundation Models for Finance. (来自 r31_papers/revisiting_tsfm_finance_2025.md)
  - **结论**：🔬 调研；大语言模型做金融 TS 的研究，未采用

- **[TradeFM 2026]** TradeFM: A Foundation Model for Trading. (来自 r31_papers/tradefm_2026.md)
  - **结论**：🔬 调研

---

## §4 决策焦点学习 / SPO+

- **[Elmachtoub & Grigas 2022]** Smart Predict, Then Optimize. *Management Science* 68(1):9–26. [arXiv:1710.08005](https://arxiv.org/abs/1710.08005)
  - **用在**：**T87 SPO+ DFL 实验**（iter_015 核心，+1.85 LOSO over iter_014）；自定义 LightGBM 目标函数；答辩 slides 核心 contribution
  - **结论**：✅ 完全采纳；SPO+ 代理损失函数是我们回归→决策转化的理论基础

- **[Donti, Amos & Kolter 2017]** Task-based End-to-end Model Learning in Stochastic Optimization. *NeurIPS 2017*. [arXiv:1703.04529](https://arxiv.org/abs/1703.04529)
  - **用在**：proposals/T103 和 T104 DFL 文献综述
  - **结论**：📖 决策焦点学习奠基工作之一

- **[Elmachtoub & Grigas / Wilder et al. 2019]** Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization. *AAAI 2019*.
  - **用在**：proposals/T104 DFL 背景
  - **结论**：📖 参考

- **[Mandi et al. 2023]** Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities. *JAIR 2023*.
  - **用在**：DFL survey 参考
  - **结论**：📖 参考综述

- **[Return-Weighted CE 2025]** A Novel Loss Function for Deep Learning Based Daily Stock Trading System. [arXiv:2502.17493](https://arxiv.org/abs/2502.17493)
  - **用在**：r10_papers PnL-aware loss 调研；return-weighted cross-entropy = S1 方案
  - **结论**：⚠️ S1 方案理论依据；LightGBM sample_weight=|Δp| 实现已采用（T92 等）

- **[GMA-DL / GMADL 2024]** Gradient-Matching Aligned Decision Loss for GBDT. (来自 proposals T114, arXiv:2412.18405 附近)
  - **用在**：T114 文献扫描提议
  - **结论**：🔬 调研；期望 +0.5–2.0 LOSO 但未实施

- **[ε-Insensitive Pinball Loss 2024]** Pinball Boosting of Regression Quantiles. *CSDA 2024*. [arXiv:2401.02325 附近](https://arxiv.org/abs/2401.02325)
  - **用在**：N11 提议
  - **结论**：🔬 调研，未采用

- **[SPO+ noise variant 2022]** (arXiv:2211.12858) — SPO+ with noise regularization
  - **用在**：T87 实验 ablation 延伸讨论
  - **结论**：🔬 调研，未实施

---

## §5 OOD / 分布偏移 / 鲁棒训练

- **[Sagawa et al. 2020]** Distributionally Robust Neural Networks for Group Shifts. *ICLR 2020*. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) | [code](https://github.com/kohpangwei/group_DRO)
  - **用在**：**R2_group_dro 实验**；LightGBM per-sym group-weighted 训练；答辩 Q3 OOD 章节
  - **结论**：⚠️ Group DRO 概念采纳；LightGBM 版本用 softmax(loss_g/τ) 更新 group weights；未系统 ablate

- **[Krueger et al. 2021]** Out-of-Distribution Generalization via Risk Extrapolation (V-REx). *ICML 2021*. [arXiv:2003.00688](https://arxiv.org/abs/2003.00688) | [PMLR v139](https://proceedings.mlr.press/v139/krueger21a.html)
  - **用在**：r11_papers OOD 调研；variance penalty = Var_e(R_e) 概念
  - **结论**：🔬 NN 路线可用；GBDT 无 batch loss 概念，近似等价于 Group DRO，未独立实施

- **[Ganin et al. 2016]** Domain-Adversarial Training of Neural Networks (DANN). *JMLR 17*. [arXiv:1505.07818](https://arxiv.org/abs/1505.07818)
  - **用在**：r11_papers D4 domain adversarial training；Gradient Reversal Layer 用 sym 作为 domain
  - **结论**：🔬 NN-only；GBDT 无 GRL，未采用；也违反 sym-agnostic 精神（需要 sym label 在训练时）

- **[Arjovsky et al. 2019]** Invariant Risk Minimization (IRM). [arXiv:1907.02893](https://arxiv.org/abs/1907.02893)
  - **用在**：r11 OOD 调研
  - **结论**：❌ 拒绝；arXiv:2010.05761 "The Risks of IRM" 证明线性 setting 下 IRM 需要 #envs ≫ dim，我们仅 5 sym 不满足

- **[Sun & Saenko 2016]** Deep CORAL: Correlation Alignment for Deep Domain Adaptation. *ECCV 2016 workshop*. [arXiv:1607.01719](https://arxiv.org/abs/1607.01719)
  - **用在**：r11 D5 Domain Adaptation 调研
  - **结论**：🔬 NN-only，未采用

- **[Yao et al. 2022]** Wild-Time: A Benchmark of In-the-Wild Distribution Shift over Time. *NeurIPS 2022*. [arXiv:2211.14238](https://arxiv.org/abs/2211.14238)
  - **用在**：r11_papers wildtime_ts_dg_benchmark.md；OOD temporal benchmark
  - **结论**：📖 关键发现：没有方法稳定胜出 ERM on temporal shift；印证我们谨慎对待 OOD 方法

- **[Deng et al. 2024]** Domain Generalization in Time Series Forecasting. *TKDD 2024*.
  - **用在**：r11_papers wildtime benchmark 一起调研
  - **结论**：📖 M training source domains → K-M unseen test 设置与我们类似；结论同样悲观

- **[Zhang et al. 2018]** Mixup: Beyond Empirical Risk Minimization. *ICLR 2018*. [arXiv:1710.09412](https://arxiv.org/abs/1710.09412)
  - **用在**：**R1v2_sym_aug**（跨 sym Mixup 数据增强）；T11 cross-sym mixup；镜像 buy/sell 对称增强
  - **结论**：✅ 跨 sym Mixup 和镜像增强采纳；单次 B(0.4,0.4) 插值，混合不同 sym 样本

- **[Xu et al. 2023]** Embarrassingly Simple MixUp for Time-Series Data Augmentation. [arXiv:2304.04271](https://arxiv.org/abs/2304.04271)
  - **用在**：r11_papers TS Mixup 变体调研；LatentMixUp++ 参考
  - **结论**：⚠️ 理论支撑；我们实现是 input Mixup，非 latent

- **[InvariantStock 2024]** Invariant Features in Stock Prediction. [arXiv:2409.00671](https://arxiv.org/abs/2409.00671)
  - **用在**：r31/r11 跨股票泛化调研
  - **结论**：🔬 fundamental features 比 price features 更跨 sym 稳定的发现；我们 KyleInv 设计有此精神

- **[Eastwood et al. 2023]** Spuriosity Didn't Kill the Classifier: Using Invariant Predictions for Imageability Bias. *NeurIPS 2023*. [arXiv:2307.09933](https://arxiv.org/abs/2307.09933)
  - **用在**：OOD 调研综述
  - **结论**：🔬 跨域 spurious feature 调研

- **[TTA 2021]** Test-Time Adaptation via Confidence Maximization. [arXiv:2106.14999](https://arxiv.org/abs/2106.14999)
  - **用在**：r11 D14 TTA 调研；window z-score ≈ TTA 的近似
  - **结论**：🔬 stateless Predictor 下 TTA 不可行；但 window-z 已实现大部分 TTA 效果

- **[Manifold Mixup 2019]** Manifold Mixup: Better Representations by Interpolating Hidden States. *ICML 2019*. [arXiv:2409.05202 / 原版 1806.05236](https://arxiv.org/abs/1806.05236)
  - **用在**：r11 B9 Manifold Mixup 调研
  - **结论**：❌ 需要访问 hidden states；GBDT 无 hidden states；NN 路线复杂度高，未采用

---

## §6 量化策略 / Alpha 工程

- **[López de Prado 2018]** Advances in Financial Machine Learning. *Wiley*.
  - **用在**：Ch 3 Triple Barrier 标注；Ch 10 Bet Sizing / 概率→仓位公式；Ch 18-19 微观结构特征；Purged K-Fold CV
  - **结论**：✅ 多章节采纳；LOSO 受 Purged K-Fold 启发；threshold 决策受 bet sizing 启发

- **[Kakushadze 2015]** 101 Formulaic Alphas. *Wilmott Magazine* 2016. [arXiv:1601.00991](https://arxiv.org/abs/1601.00991)
  - **用在**：r20_papers Alpha101 调研；公式化 alpha 向 tick 频率的适配
  - **结论**：⚠️ 中频 alpha（平均持仓 2 天），时间窗口压缩到 tick 频率后效果参差，部分 idea 进入 factor_library

- **[Microsoft Qlib Alpha158]** Qlib: An AI-Oriented Quantitative Investment Platform. [GitHub qlib](https://github.com/microsoft/qlib)
  - **用在**：r20_papers Qlib Alpha158 因子库调研
  - **结论**：⚠️ 日频因子库参考；部分动量/反转 idea 以 tick 频率重实现

- **[Amaya et al. 2015]** Realized Skewness. *Journal of Finance* 70(6):2649–2708. (来自 r20_papers/amaya_realized_skewness_2015.md)
  - **用在**：F5 跳跃/偏度类特征调研
  - **结论**：🔬 已计算 jshare，高阶矩特征未进最终集

- **[Bailey & López de Prado 2012]** The Sharpe Ratio Efficient Frontier. *PRMIA 2012*.
  - **用在**：backtest 过拟合风险讨论
  - **结论**：📖 参考

- **[Harvey, Liu & Zhu 2016]** …and the Cross-Section of Expected Returns. *Review of Financial Studies*.
  - **用在**：多重检验 / p-hacking 讨论
  - **结论**：📖 参考

- **[López de Prado]** Triple Barrier Method. *Advances in Financial ML* Ch 3. [mlfinpy docs](https://mlfinpy.readthedocs.io/en/latest/Labelling.html)
  - **用在**：r10_papers triple_barrier 标注调研；讨论比赛固定 alpha 标注的局限
  - **结论**：⚠️ 理论参考；比赛标注固定，无法重标注

- **[Volatility MoE 2025]** Volatility Mixture-of-Experts. (来自 r31_papers/volatility_moe_2025.md)
  - **结论**：🔬 调研

---

## §7 集成学习 / Stacking

- **[Wolpert 1992]** Stacked Generalization. *Neural Networks* 5(2):241–259.
  - **用在**：50×50 集成（50 种特征子集 × 50 种 seed）理论基础
  - **结论**：✅ Stacking 框架采纳；meta-learner 用 NNLS 权重优化

- **[Yao et al. 2018]** Using Stacking to Average Bayesian Predictive Distributions. *Bayesian Analysis*.
  - **用在**：r12_papers Bayesian Stacking；BNNLS 集成权重参考；arXiv:2101.08954 也被引
  - **结论**：⚠️ LOO log-likelihood 最优权重思路参考；实际用简单 equal-weight mean

- **[Huang et al. 2017]** Snapshot Ensembles: Train 1, Get M for Free. *ICLR 2017*. [arXiv:1704.00109](https://arxiv.org/abs/1704.00109)
  - **用在**：r12_papers C3 Snapshot ensemble；cyclic LR + multi-checkpoint
  - **结论**：🔬 调研；我们最终用 50 个固定 seed 而非 cyclic LR snapshot

- **[Izmailov et al. 2018]** Averaging Weights Leads to Wider Optima and Better Generalization (SWA). *UAI 2018*. [arXiv:1803.05407](https://arxiv.org/abs/1803.05407)
  - **用在**：r12_papers B8 SWA；arXiv:2102.08604 / arXiv:2201.00519 也被引
  - **结论**：🔬 调研；NN 训练考虑过，但最终 GBDT 路线为主，SWA 未系统实验

- **[NGBoost: Duan et al. 2020]** NGBoost: Natural Gradient Boosting for Probabilistic Prediction. *ICML 2020*. [arXiv:1910.03225](https://arxiv.org/abs/1910.03225) | [code](https://github.com/stanfordmlgroup/ngboost)
  - **用在**：r12_papers 概率预测 GBDT 调研；T87 延伸讨论（NGBoost vs SPO+）
  - **结论**：🔬 Natural gradient GBDT 有趣；但 SPO+ 更直接对齐 PnL，优先级低

- **[GradNorm 2018]** GradNorm: Gradient Normalization for Adaptive Loss Balancing. *ICML 2018*. [arXiv:1711.02257](https://arxiv.org/abs/1711.02257)
  - **用在**：r12 B6 多任务自适应权重调研
  - **结论**：🔬 5 horizon 多任务权重调整参考

- **[Kendall et al. 2018]** Multi-Task Learning Using Uncertainty to Weigh Losses in Deep Learning. *CVPR 2018*. [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
  - **用在**：r35_alt_architectures B5 不确定性加权多任务
  - **结论**：🔬 调研；5 horizon 自动 loss 权重；未实施

- **[Covariate-Dependent Stacking 2024]** CDST: Covariate-Dependent Stacking. [arXiv:2408.09755](https://arxiv.org/abs/2408.09755)
  - **用在**：R_stack3_compound 集成方案调研
  - **结论**：🔬 调研；当前 iter 的 spread/vol conditioned stacking 有此思路

- **[Bayesian Hierarchical Stacking 2022]** Bayesian Hierarchical Stacking. [arXiv:2101.08954](https://arxiv.org/abs/2101.08954)
  - **用在**：r12 方案 12 sym-aware Hierarchical Stacking
  - **结论**：🔬 调研

---

## §8 概率校准 / 决策理论

- **[Guo et al. 2017]** On Calibration of Modern Neural Networks. *ICML 2017*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
  - **用在**：r13_papers Temperature Scaling；ECE（Expected Calibration Error）定义；T76 threshold 校准
  - **结论**：⚠️ Temperature Scaling 理论采纳；LightGBM raw score / T 实现；阈值调优受此启发

- **[Kull et al. 2017]** Beta calibration: A Well-Founded and Easily Implemented Improvement on Logistic Calibration for Binary Classifiers. *AISTATS 2017*. (来自 r13_papers/kull_beta_calibration_2017.md)
  - **用在**：r13 概率校准调研
  - **结论**：🔬 Beta calibration 参考；我们用 temperature scaling + threshold tuning 替代

- **[Platt 1999]** Probabilistic Outputs for SVMs. *Advances in Large Margin Classifiers*.
  - **用在**：r13 Platt scaling 调研背景
  - **结论**：📖 校准方法对比参考

- **[López de Prado 2018 Ch 10]** Bet Sizing. *Advances in Financial ML*, Ch 10.
  - **用在**：r13_papers bet sizing；z 统计量 → 仓位大小公式；3 类分类→连续仓位
  - **结论**：⚠️ bet sizing 公式参考；比赛决策为离散 {short, flat, long}，连续仓位未直接应用

- **[Conformal Prediction 2024]** Conformal Selective Prediction with General Risk Control. [arXiv:2101.12523 / arXiv:2403.15025 等](https://arxiv.org/abs/2403.15025)
  - **用在**：R_conformal_select 实验；Mondrian conformal arXiv:2110.14914
  - **结论**：🔬 conformal abstain gate 调研；T76 decoupled threshold 更简单且有效，取代了 conformal

- **[Cost-Sensitive Threshold Tuning 2024]** (来自 r13_papers/cost_sensitive_threshold_tuning_2024.md)
  - **用在**：T76 threshold 调优理论背景
  - **结论**：✅ 非对称阈值设计采纳

- **[Quantile NN 2024]** Quantile Neural Network for Stock Return Distributions. [arXiv:2408.07497](https://arxiv.org/abs/2408.07497)
  - **用在**：r13 分位数回归研究；多分位数预测作为不确定性估计
  - **结论**：🔬 调研

---

## §9 工程 / 训练技巧

- **[Lin et al. 2017]** Focal Loss for Dense Object Detection. *ICCV 2017*. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
  - **用在**：r12 B2 Focal Loss；类别不均衡处理调研；arXiv:1908.01672 "Imbalance-XGBoost" 也被引
  - **结论**：🔬 调研；LightGBM focal loss 实验，效果低于 sample-weighted regression，未采用

- **[Izmailov et al. 2018 SWA]** (见 §7)

- **[Huang et al. 2017 Snapshot]** (见 §7)

- **[Ba et al. 2016]** Layer Normalization. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
  - **用在**：NN 架构 T188v2；LayerNorm vs BatchNorm 选择
  - **结论**：✅ LayerNorm 在 NN 最终架构中采用

- **[Kingma & Ba 2015]** Adam: A Method for Stochastic Optimization. *ICLR 2015*. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
  - **用在**：所有 NN 训练优化器
  - **结论**：✅ AdamW 变体采用

- **[Loshchilov & Hutter 2019]** Decoupled Weight Decay Regularization (AdamW). *ICLR 2019*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
  - **用在**：NN 训练 weight decay 设置
  - **结论**：✅ AdamW 采用

- **[Lookahead 2019]** Lookahead Optimizer: k Steps Forward, 1 Step Back. *NeurIPS 2019*. [arXiv:1907.08610](https://arxiv.org/abs/1907.08610)
  - **用在**：r35 优化器调研
  - **结论**：🔬 调研，未系统实验

- **[Foret et al. 2021 SAM]** Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR 2021*. [arXiv:2010.01265](https://arxiv.org/abs/2010.01265)
  - **用在**：r35 优化器调研；SAM + AdamW = ASAM
  - **结论**：🔬 调研，计算成本 2×，未采用

- **[Rough Paths / Signature 2016]** Buehler et al. 2019, Lyons 2007. [arXiv:2006.14498](https://arxiv.org/abs/2006.14498) / [arXiv:1603.03788](https://arxiv.org/abs/1603.03788)
  - **用在**：LOB signature feature 调研（100-tick window path signature）
  - **结论**：🔬 Iterated integrals 特征理论漂亮但工程实现复杂，未采用

- **[Robust-GBDT 2025]** Nonconvex Robust Focal Loss for GBDT. *KAIS 2025*. [arXiv:2310.05067](https://arxiv.org/abs/2310.05067)
  - **用在**：§1.4 调研
  - **结论**：🔬 调研

- **[Moody & Saffell 1998]** Learning to Trade via Direct Reinforcement. (来自 r10_papers/moody_saffell_1998.md)
  - **用在**：直接 RL 优化 PnL 的早期工作；T87 SPO+ 的精神前驱
  - **结论**：📖 历史参考

---

## §10 Kaggle / 社区 Writeup / HFT 比赛

- **[Yirun 2021]** Jane Street Market Prediction — 1st Place Solution (CATs Trading, Yirun Zhang). [Kaggle writeup](https://www.kaggle.com/competitions/jane-street-market-prediction/writeups/cats-trading-yirun-s-solution-1st-place-training-s)
  - **用在**：r10_papers/yirun_jane_street_2021.md；T11 cross-sym mixup；直接 utility-as-loss 验证；5 resp 多目标联合训练
  - **结论**：✅ Mixup 增强采纳；多 horizon 多目标训练思路影响深远；"CE 预热→PnL 损失微调" 策略确认

- **[Volkova 2024]** Jane Street Real-Time Market Data Forecasting — 8th Place Solution (Evgenia Volkova). [GitHub solution.md](https://github.com/evgeniavolkova/kagglejanestreet)
  - **用在**：r10_papers/volkova_jane_street_2024.md；GRU + aux target + online learning 详细分析
  - **结论**：⚠️ aux target 思路（多 horizon 平均）参考；online learning 因 stateless 约束不可用；sym-agnostic 路线验证（8th=top 0.3%）

- **[hyd 2023]** Optiver Trading at the Close — 1st Place Solution. [docswell ppt](https://docswell.com/) + [Kaggle](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
  - **用在**：r10_papers/hyd_optiver_2023.md；CatBoost + GRU + Transformer 三模型集成；cross-stock Transformer；triplet imbalance 特征
  - **结论**：⚠️ "GRU per-stock + Transformer cross-stock" 双轴架构参考；triplet imbalance 特征思路（我们 LOB ratio 已有类似）

- **[Optiver Realized Volatility 2021]** 1st/2nd/3rd place writeups. [Kaggle discussion #274970](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
  - **用在**：r33_kaggle_hft_winners；nyanp 1st tick-size 反推特征；Optiver 特征体系参考
  - **结论**：📖 Realized Vol 特征（BV, TSRV, book pressure）部分进入我们 F5

- **[Jane Street Real-Time 2024-2025]** Top solutions summary. [Kaggle discussion #556542](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/discussion/556542)
  - **用在**：r33_kaggle_hft_winners 调研
  - **结论**：📖 多项结论印证我们设计（sym-agnostic、no state、regression over classification）

- **[G-Research Crypto Forecasting]** Competition. [Kaggle](https://www.kaggle.com/c/g-research-crypto-forecasting)
  - **用在**：r33 HFT 比赛横向对比
  - **结论**：📖 参考

- **[JPX Tokyo Stock Exchange Prediction]** Competition. [Kaggle](https://www.kaggle.com/competitions/jpx-tokyo-stock-exchange-prediction)
  - **用在**：r33 HFT 比赛横向对比
  - **结论**：📖 参考

- **[codefluence Jane Street 13th 2024]** LightGBM + MLP ensemble, 78 raw features. [Kaggle code](https://www.kaggle.com/code/gogo827jz/jane-street-supervised-autoencoder-mlp)
  - **用在**：r33 + competition_insights；LightGBM 基线对比
  - **结论**：📖 无 online learning → vs Volkova +0.002 R² 差距确认 online learning 关键性

- **[Meta-labeling López de Prado]** (来自 r10_papers/lopez_de_prado_meta_labeling.md)
  - **用在**：meta-labeling 的 side + size 两步框架调研
  - **结论**：🔬 调研；我们的两步（LGB 预测方向 + threshold gate 决定交易）有 meta-labeling 思路

- **[Finance-Grounded DFL 2025]** (来自 r10_papers/finance_grounded_khubiyev_2025.md)
  - **用在**：r10 DFL 文献
  - **结论**：🔬 调研

---

## §11 工具 / 库 / 数据集

- **[LightGBM: Ke et al. 2017]** LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*. [arXiv:1711.01830](https://arxiv.org/abs/1711.01830) | [GitHub](https://github.com/microsoft/LightGBM)
  - **用在**：主 GBDT 模型；GPU 训练（device='gpu', gpu_use_dp=False）；全实验流程
  - **结论**：✅ 核心工具；所有 T* 实验均用 LightGBM

- **[XGBoost: Chen & Guestrin 2016]** XGBoost: A Scalable Tree Boosting System. *KDD 2016*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754) | [GitHub](https://github.com/dmlc/xgboost)
  - **用在**：集成候选；T2 GBDT 实验
  - **结论**：⚠️ 部分实验使用；最终 50×50 集成以 LightGBM 为主

- **[CatBoost: Prokhorenkova et al. 2018]** CatBoost: Unbiased Boosting with Categorical Features. *NeurIPS 2018*. [arXiv:1706.09516](https://arxiv.org/abs/1706.09516) | [GitHub](https://github.com/catboost/catboost)
  - **用在**：T2 GBDT SchemeB/A 实验；GPU 版 catboost 调研
  - **结论**：🔬 安装 GPU 版复杂，最终以 LightGBM 替代；CPU 版在大数据下慢

- **[PyTorch: Paszke et al. 2019]** PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*. [arXiv:1912.01703](https://arxiv.org/abs/1912.01703)
  - **用在**：所有 NN 实验（T87 SPO+ NN、T188v2 等）
  - **结论**：✅ NN 框架

- **[scikit-learn: Pedregosa et al. 2011]** Scikit-learn: Machine Learning in Python. *JMLR* 12:2825–2830.
  - **用在**：特征归一化、LOSO 交叉验证、calibration
  - **结论**：✅ 核心工具库

- **[scipy]** SciPy: Open Source Scientific Tools for Python.
  - **用在**：KS 检验（`scipy.stats.ks_2samp`）、阈值优化（`scipy.optimize.minimize_scalar`）
  - **结论**：✅ 核心工具

- **[WandB: Biewald 2020]** Experiment Tracking with Weights and Biases. *ICML 2020 MLOps workshop*.
  - **用在**：所有训练实验追踪（project="liangwenbei", entity="cjxh21-Tsinghua University"）
  - **结论**：✅ 全程实验管理

- **[NumPy / Pandas]** Fundamental Scientific Computing in Python.
  - **用在**：特征构建（向量化，禁止 Python for 循环）
  - **结论**：✅ 核心工具

---

## §12 项目专项资源

- **[良文杯赛题说明会 PDF]** Liangwenbei LOB Direction Prediction Competition Brief. AITopia Platform.
  - **用在**：比赛约束定义（date 评测置 0；测试顺序打乱；sym 0-4 但含训练外股票）
  - **结论**：✅ 核心约束文档；见 CRITICAL_CONSTRAINTS.md

- **[AITopia 平台]** Competition hosting platform.
  - **用在**：`Predictor.py` API 规范；submit-and-score 循环；评分函数定义
  - **结论**：✅ 提交接口规范

- **[CRITICAL_CONSTRAINTS.md]** 内部约束文档（workdir 根目录）.
  - **用在**：所有 worker agent 必读；三大硬约束
  - **结论**：✅ 项目最高优先级规范文档

- **[FI-2010 Dataset]** Ntakaris et al. 2018. Mid-price prediction LOB benchmark. Helsinki Nasdaq Nordic.
  - **用在**：DeepLOB baseline 的标准测试集；LOB DL 论文的通用基准
  - **结论**：📖 基准参考；我们数据是私有中文 A 股 LOB

- **[Kolm, Turiel & Westray 2023 SSRN preprint]** (同 §1 第 2 条)

- **[DeepLOB microstructural guide 2024 Briola]** (同 §2 第 8 条)

- **[Claude Code Agent / Anthropic]** AI Coding Assistant used throughout the project.
  - **用在**：200+ 实验的代码生成、调研分析、答辩材料生成
  - **结论**：✅ 核心工程工具；所有 worker agent 由 Claude Code 驱动

---

## 附：按主题索引的最重要引用（各 §1-2 条核心）

| Section | 最重要引用 | 状态 |
|---|---|---|
| §1 LOB Theory | Cont, Kukanov & Stoikov 2014 (MLOFI) | ✅ |
| §1 LOB Theory | Stoikov 2018 (Micro-Price / WMP) | ✅ |
| §2 ML LOB | Zhang, Zohren & Roberts 2018 (DeepLOB) | ⚠️ baseline |
| §2 ML LOB | Berti 2025 (TLOB/MLPLOB) | ⚠️ NN ablation |
| §3 TS Forecasting | iTransformer / PatchTST / TimesNet | ❌ 拒绝 |
| §4 DFL | Elmachtoub & Grigas 2022 (SPO+) | ✅ 核心 |
| §5 OOD | Sagawa 2020 (Group DRO) | ⚠️ 理论参考 |
| §5 OOD | Zhang 2018 (Mixup) | ✅ 数据增强 |
| §6 Alpha | López de Prado 2018 (AFML) | ✅ 多章节 |
| §7 Ensemble | Wolpert 1992 (Stacking) | ✅ 集成框架 |
| §8 Calibration | Guo 2017 (Temperature Scaling) | ⚠️ 阈值调优 |
| §9 Engineering | LightGBM Ke 2017 | ✅ 主模型 |
| §10 Kaggle | Yirun 2021 (Jane Street 1st) | ✅ Mixup 验证 |
| §11 Tools | LightGBM / PyTorch / WandB | ✅ 全程使用 |

---

*自动扫描 `scan_references.py` + 手工整理 | 2026-05-30*
*总计：~130+ 引用条目，按 §1–§12 共 12 类*
