# 良文杯 LOB 项目 Tricks 全表（T1–T192 + R 系列）

本表涵盖"良文杯 LOB 预测"项目全部 T1–T192 系列实验及 R 系列调研（共约 192 个 T 实验、40+ R 系列、22 次有效平台提交）。本地 eval 经历四次演变：①T1–T35 阶段用 **LOSO 5-fold**（留一 sym 交叉验证）；②T44–T74 阶段改为 **LOSO-equiv**（全 sym 训练 + DE 在 442k 行本地测试集上搜索阈值，方法论变化贡献约 +7 PnL，不可与 LOSO 直接比较）；③T75–T155 阶段沿用 LOSO-equiv + **V4 walk-forward val**（train 0–79, val 80–95, test 96–119 OOF）；④T156–T192 阶段改为 **holdout in-sample**（全数据 0–119 训练后在 date 96–119 评分，含 in-sample 污染，绝对值不可信，只看相对 delta）。平台分数定义：`pnl = [(label−1)·(mp_{t+n}−mp_t) − 0.0001·|label−1|·((mp_{t+n}+1)+(mp_t+1))] / (mp_t+1)` 对全部样本累计，取 5 个 horizon 最高一个计入排名；h_60（α=0.1%）为整个项目重心。列说明：分数列留空 = 该 trick 主要不影响分数维度；速度列留空 = 该 trick 主要不影响速度；"未记录" = 实验发生但数字未留存。

| idx | trick | 出处 | 分类 | 本地分数效果 | 提交到平台后平台的分数效果 | 本地速度效果 | 提交到平台后的评测速度提升 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | NN baseline（DeepLOB 风格 MLP，3-class CE 损失，h_60，单 seed，154 维 SchemeC） | DeepLOB paper / mmpc_demo | 更好的模型架构 | LOSO h_60 ~+8（SchemeC 框架） | | | |
| T2 | LightGBM baseline（SchemeB 154 维，3-class CE 目标，argmax 决策，单 seed） | 官方 baseline | 更好的模型架构 | LOSO sum -22.10（5 fold，2/5 正） | | | |
| T3 | 特征 v1：添加多尺度 MLOFI、WMP、已实现波动率 RV、EWMA 指令强度及时间编码（+72 维，至 226 维 SchemeC） | 自研 | 更好的特征 | LOSO h_10 +21.86（SchemeC，进入 iter_002） | | | |
| T4 | 建立 LOSO 5-fold 评估框架，实现高置信度阈值 gate（max(p0,p2)≥T 且 >p1+δ） | 自研 | 更好的 eval（评估方式改进） | LOSO OOF +6.45（4/5 fold 正，SchemeA，iter_001c） | | | |
| T5a | SchemeB 上重调阈值（DE 搜索非对称 thr_up/thr_dn） | T4 → T5a | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO h_60 +11.11（vs T2 -22.10，iter_001d） | | | |
| T5b | SchemeC 多 horizon：扩展至 5 个 horizon 5-fold LOSO | T4 → T5b | 更好的 eval（评估方式改进） | LOSO h_10 +21.86（SchemeC，iter_002 核心） | +4.07（h_60，iter_002；首次平台正分） | | |
| T6/T6b | 5-seed SchemeA ensemble（5 个 seed 概率平均后阈值 gate） | 自研 | 更好的 ensemble | LOSO sum +8.87（vs iter_001c +6.45，+0.36，边际） | | | |
| T7 | 窗口内 z-score 归一化（100-tick 窗口，原始 154 维 + 窗口归一化 154 维 = 308 维，SchemeD1） | 自研 | 更好的特征 | LOSO +13.91（vs iter_001d +11.11；sym=2 从 -0.98 翻到 +2.24，修复 OOD 问题） | | | |
| T8 | sym=2（蓝筹/ETF）brittleness 诊断：分析 5 只股票特征分布差异，确认 sym=2 amount_delta z-score 偏离 +7.49 | 自研 | 调研/分析 | | | | |
| T9 | SchemeE sym-agnostic 归一化（amount_delta z-score + spread-norm，针对 T8 诊断） | T8 → T9 | 更好的特征 | 部分进入后续 SchemeP；未单独提交 | | | |
| T10 | SchemeF 全局 z-score 归一化（全量特征全局归一化） | 自研 | 更好的特征 | LOSO h_10 +13.40（vs iter_002 +21.86，不超，负面；窗口归一化 T7 优于全局归一化 T10） | | | |
| T11 | SchemeC 5-seed ensemble（h_10，5 fold LOSO）进入 iter_003 | T5b → T11 | 更好的 ensemble | LOSO h_10 +21.86 → +22.22（+0.36，边际；iter_003） | | | |
| T12 | multi-horizon stacking meta（用上层 meta 模型综合 5 个 horizon 输出） | 自研 | 更好的 ensemble | LOSO +17.46（vs base +21.86，**退步**） | | | |
| T13 | PnL-aware 端到端 NN loss（PnL 公式梯度直接反传训练 NN） | 自研 | 更好的损失函数 | 不稳定，不收敛 | | | |
| T14 | cross-sym mixup（跨股票样本混合增强） | 自研 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | 负面 | | | |
| T15 | isotonic 等变校准 + EV gate（等变校准后阈值决策） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO h_10 +21.86 → +20.84（**-1.02**，校准破坏排序质量） | | | |
| T17 | CatBoost vs LightGBM 对比实验 | 调研 | 调研/分析 | LOSO ceiling ~+22.2，CatBoost 与 LGB 基本持平 | | | |
| T18 | XGBoost vs LightGBM 对比实验 | 调研 | 调研/分析 | 持平，无明显优势 | | | |
| T19 | NN regularized（加 dropout + weight decay） | 自研 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | LOSO 与 iter_002 持平，无提升 | | | |
| T20/T21 | SchemeH 291 维（MLOFI + WMP + RV + EWMA 进一步扩展，+65 维） | 自研 | 更好的特征 | LOSO h_10 +21.70（vs iter_002 +21.86，持平；单纯加维度无效） | | | |
| T22 | Alpha101 + Alpha191 WorldQuant 风格跨截面因子调研 | WorldQuant Alpha 文献 | 调研/分析 | 最佳单因子 IC +21.80；但需截面数据，平台批次打乱下无法实现 | | | |
| T23 | Robust CV + 2-anchor 平台校准 mapping（建立 LOSO→平台透传系数 ~70%） | 自研 | 更好的 eval（评估方式改进） | 建立 h_60 gap ≈ -2.23；透传系数 0.70；后续所有平台预测基于此 | | | |
| T24 | iter_004 a/b：短 h（h_5/10/20）严格阈值或彻底关闭，锁 h_60 地板 | T5b → T24 | 更好的执行层（decision/wrapper/threshold/conformal） | h_60 LOSO +6.30（不变）；纯后处理 | | | |
| T25–T28 | aug_a 数据增强：bid/ask 互换 + 涨/跌 label 对换，训练集 ×2（反向翻转对称增强） | 自研 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | LOSO h_60 单 seed +6.30 → +9.67（+3.37）；5-seed +9.67 → +11.46（+1.79）；iter_005b | | | |
| T30 | DE 4D 阈值优化（Differential Evolution 搜索 thr_up/thr_dn/margin/weight） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO aug_a 5-seed +11.46 → +13.61（+2.15）；**注意：之后 iter_013 实证 DE 在 5-fold LOSO 上可超调** | | | |
| T34 | CV v2：建立 LO2SO + bootstrap CI + 平台预言机多维验证框架 | 自研 | 更好的 eval（评估方式改进） | 方法论框架建立，为后续实验提供更可靠的本地指导 | | | |
| T35 | ReVol 滚动波动率 + Savitzky-Golay 平滑（SchemeK，+32 维，255 维） | 文献 SG filter | 更好的特征 | LOSO h_60 +5.53（vs iter_002 +6.30，不超；SG 对树模型 gain 几乎为 0） | | | |
| T44 | R34 Stage 1：+54 维 sym-invariant 特征（dual z-score W=20/50，signed RV，Kyle λ rolling，EWMA-OFI multi-α，cancel pressure） | R34 Stage 1 → T44 | 更好的特征 | LOSO h_60 单 seed +12.09（vs iter_005b +9.67）；5-seed +13.67 | | | |
| T51/T51b/T51c | R34 Stage 2：+59 维（window quantile rank W=20/50/100，signed skewness，GOFI ~30 维，vol-burst ratio，Kyle's λ rolling） | R34 Stage 2 → T51 | 更好的特征 | LOSO h_60 单 seed +13.67 → 5-seed +15.06（真正突破，超历史最好） | | | |
| T53 | R34 Stage 3：+14 维（EWMA-residual，multi-W RV ratio，Bipower variation，Cancel-pressure imbalance，Roll's effective spread） | R34 Stage 3 / R_roll_tsrv → T53 | 更好的特征 | LOSO h_60 +15.06 → +16.84（+1.78；iter_008，超时问题尚存） | | | |
| T55 | R34 Stage 4：少量额外 edge gain 特征，合并入 SchemeP | R34 Stage 4 → T55 | 更好的特征 | LOSO 与 Stage 3 相近，边际 | | | |
| T58 | walk-forward CV 验证（train 0-79, val 80-95, test 96-119）与 LOSO 对比，确认 V4 更接近时间 OOD | 自研 | 更好的 eval（评估方式改进） | 确认 V4 更接近平台 OOD 行为 | | | |
| T59 | 全 sym 训练（取消 LOSO 留一 sym，全 5 sym 一起训练），建立 LOSO-equiv 评估 | 自研 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | LOSO-equiv +16.84 → +24.52（+7.68，其中约 +7 为方法论变化，真实新增 alpha ~+1） | | | |
| T61 | batch-vec 推理加速：向量化 feature 提取，替换 for 循环（iter_010） | 自研 | 更快的推理 | | | feature 提取 254 min → 4.5 min（58×，针对 442k 行） | 首次通过平台 3h 限制（iter_010 前全部超时） |
| T62 | 无偏 threshold 设计（对称 k-threshold：thr = k × mean_abs_pred，避免 DE 过拟合） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO k=1.25 时 +33.72（比 DE +36.23 鲁棒但低 2.51） | | | |
| T63 | regression EV gate 参数扫描（非对称 thr, 绝对值 thr 等变体探索） | T75 → T63 | 调研/分析 | 边际，确认原始 EV gate 设计最简洁有效 | | | |
| T64 | V4 walk-forward val 正式化（train 0-79, val 80-95, test 96-119 OOF，iter_011） | T58 → T64 | 更好的 eval（评估方式改进） | LOSO-equiv +25.94 → +26.44（+0.50；方法论更干净） | | | |
| T65 | pseudo labeling（用模型对无标签数据生成伪标签后混入训练） | 文献 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | 负面 | | | |
| T66 | adversarial val（构造分布偏移最大的验证集，用于过滤特征） | 文献 adversarial validation | 更好的 eval（评估方式改进） | 方法建立；T152 用此过滤得 259 维，平台无提升 | | | |
| T67 | multi-split bagging（多种数据分割方式的 bagging） | 自研 | 更好的 ensemble | LOSO 边际，未超 5-seed ensemble | | | |
| T68 | R34 Stage 5：+20 维（adaptive momentum，OFI toxicity，signed bipower，spread regime，trade-direction persistence，liquidity asymmetry）→ SchemeP 359 维完成 | R34 Stage 5 → T68 | 更好的特征 | LOSO-equiv +25.94 → +26.44（+0.50；SchemeP 359 维完成；iter_012） | | | |
| T69/T70/T70b | iter_012 打包、测试与提交准备（T61 batch-vec + SchemeP 359 维 + V4 walk-forward） | T68 → T69/T70 | 工具/基础建设 | LOSO-equiv +26.44 | | | |
| T71 | h_40 Stage 5 特征探索（h_40 局部有效性验证） | T68 → T71 | 调研/分析 | h_40 局部有效，但平台评分取 h_60，未入 final | | | |
| T72 | binary cascade（先预测"出手/不出手"再预测"涨/跌"的二阶段决策） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | 负面，复杂度高 | | | |
| T73 | seed subset 分析（验证 5-seed 稳定性下界） | 自研 | 调研/分析 | 确认 5-seed 足够稳定 | | | |
| T74 | time features SchemeQ（+6 维：sin/cos 时间位置编码，is_open/close_30min，is_pm） | 自研 | 更好的特征 | LOSO-equiv +26.44 → +35.51（-0.72 vs T75 基线，时间特征对 LGB 反而降低鲁棒性） | | | |
| T75 | LGB 改为直接回归 Δmid（objective=regression_l2，y=(mp_{t+60}-mp_t)/(mp_t+1)，EV gate 决策，SchemeP 359 维，V4，5 seed） | 自研 | 更好的损失函数 | LOSO-equiv +26.44 → **+36.23**（+9.79，**+37%**，整个比赛最大单次突破） | **+19.23**（iter_013；首个回归提交；透传系数 ~53%） | | |
| T76 | decoupled threshold（上涨/下跌阈值分开搜索，非对称 thr_up≠thr_dn） | T75 → T76 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO-equiv 边际提升（最终采用非对称 thr_up=0.0003，thr_dn=0.000216） | | | |
| T77 | multi-horizon vote ensemble（h_5/10/20/40/60 投票，SchemeC 框架下） | 自研 | 更好的 ensemble | LOSO-equiv iter_012 +5.98 提升（+5.98 over iter_012，但回归框架下此方案已过时） | | | |
| T78 | regression + EV gate + 6 维 time features（SchemeQ） | T75 + T74 → T78 | 更好的特征 | LOSO-equiv +36.23 → +35.51（**-0.72**；时间特征在回归框架下负面） | | | |
| T80 | LOSO→平台校准模型（基于 iter_002/013 两锚点建立线性透传系数 ~70%） | T23 → T80 | 更好的 eval（评估方式改进） | 建立透传系数 ~0.70；之后 53%~71% 区间实际方差较大 | | | |
| T81 | NN L2 regression on Δmid（MLP [359→256→128→64→1]，LayerNorm，GELU，Dropout 0.10，L2 预训练） | DeepLOB MLP → T81 | 更好的模型架构 | LOSO-equiv NN alone +35.89（vs LGB T75 +36.23，互补；作为 T87 warm-start） | | | |
| T82–T85 | 特征泄露审计、date=0 评测条件验证、无泄露确认（校准与审计系列） | 自研 | 调研/分析 | 确认无特征泄露；V4 walk-forward 是 cleaner 的 OOD 代理 | | | |
| T86 | quantile + dual-gate（IQR-mid hybrid：均值±IQR 阈值决策） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO-equiv +40.22（+1.94 over iter_014 +38.28；低于 T87 SPO+） | | | |
| T87 | SPO+ DFL NN fine-tune（T81 warm-start → SPO+，λ=30，lr=3e-5，5 seeds，梯度直接对 PnL 对齐） | SPO+ DFL paper | 更好的损失函数 | LOSO-equiv NN alone +35.89 → NN+LGB **+40.09**（+4.20，NN+LGB ensemble） | **+28.16**（iter_015；+8.93 vs iter_013；透传系数 71%） | | |
| T89 | CatBoost L2 regression 5-seed（替代方案评估，相关性 0.77 with T87） | T17 → T89 | 更好的模型架构 | LOSO +40.13（与 T87 NN 相当但高度相关；进 iter_016 后 4-way 失败后淘汰） | | | |
| T91 | ReVol + Savitzky-Golay revival（再次尝试 SG 平滑在回归框架下） | T35 → T91 | 更好的特征 | holdout within-noise（不超 T87 组合） | | | |
| T92 | magnitude sample weight（按 Δmid 幅度加权训练样本，R44 假设验证） | R44 假设 → T92 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | LOSO 下降（R44 假设证伪；幅度加权反而损害 PnL） | | | |
| T95 | GRU regression（门控循环单元回归 Δmid） | 自研 | 更好的模型架构 | LOSO 微小提升（进入 iter_016）；hidden state 容易在短 horizon 过拟合 | iter_016 v3 +25.08（−3.08 vs iter_015，GRU 是 4-way 失败主因之一） | | |
| T97 | 第二组 NN（T87 风格，独立随机初始化训练） | T87 → T97 | 更好的 ensemble | LOSO 与 T87 相近（架构同源，独立性不足；T163 失败根因） | | | |
| T99 | Huber LGB family（Huber 损失 LGB，5-seed，进入 iter_016） | R3 HYD → T99 | 更好的损失函数 | LOSO +44.72（但含 OOF contamination 警告） | iter_016 v3 +25.08（4-way ensemble 失败，倒退 -3.08） | | |
| T106–T109 | iter_015/016 打包系列（iter_015 v2 4-way + iter_016 各变体 CHIS/GPU/v3） | T87+T89+T99+T95 → T106 | 工具/基础建设 | LOSO iter_015 +40.09；iter_016 v3 +43.71（+3.62 vs iter_015，含 OOF DE 过拟合） | iter_016 v3 **+25.08**（**−3.08 vs iter_015**，OOF DE 4D 严重过拟合 442k 本地测试） | | |
| T115 | scale invariance audit（验证模型在 feature scale 变化下输出稳定性） | 自研 | 调研/分析 | 确认 sym-agnostic 设计正确；模型 scale-invariant | | | |
| T116 | GMADL direction-aware loss（方向感知梯度调整损失） | 文献 GMADL | 更好的损失函数 | LOSO 微弱提升，不超 SPO+ | | | |
| T117 | LGB monotone constraints（对部分特征施加单调性约束） | 自研 | 更好的模型架构 | LOSO 局部有效，进入 iter_017（T123 HYD+monotone+date-decay） | | | |
| T118 | decision 4-in-1（BAT/QSA/IDS/SCG 4 种决策规则探索） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO 边际 | | | |
| T119/T120 | iter_016 ablation（去掉 GRU，短 h 改 1；消融确认 GRU 是 iter_016 失败主因） | T106 → T119 | 调研/分析 | ablation 确认 GRU 负贡献；简单集成比复杂 4-way 更平台鲁棒 | | | |
| T122 | T75 LGB 可复现性再测（确认相同配置下结果 100% 可复现） | T75 → T122 | 工具/基础建设 | T75 LGB LOSO +36.23 可复现性确认 | | | |
| T123 | iter_017 v1（T87 NN + LGB HYD+monotone+date-decay，3-way 集成） | T117 / R1 → T123 | 更好的模型架构 | LOSO +40.70（+0.61 vs iter_015 +40.09，边际；平台未确认） | | | |
| T124 | R2/R4 评估（logret lag 特征有效性测试） | R4 → T124 | 调研/分析 | 无增量 | | | |
| T125 | T75 LGB + CatBoost addon（LGB+CB 两路集成） | T89 → T125 | 更好的 ensemble | LOSO 边际，不超 T87 组合 | | | |
| T126 | per-sym OOF threshold（每个 sym 独立 OOF 最优阈值） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO 本地高（但过拟合风险极高；改用全局阈值 + per-sym conformal） | | | |
| T127 | iter_018 v1：per-sym beta conformal abstain wrapper（|EV_pred| < β_sym × σ_pred 时弃权；β={0.10, 0.40, 0.30, 0.00, 0.00}） | R_conformal_select → T127 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO-equiv +39.75（iter_015 v1）→ **+41.49**（+1.75） | **+28.93**（iter_018；+0.77 vs iter_015） | | |
| T128 | conformal v2 研究 + M7 LGB full-retrain 可行性探索（量化 330 iters 早停与 LOSO train 一致） | 自研 | 调研/分析 | M7 trick 起点：确认 LGB M7 full-retrain 安全可行，LOSO 不变 | | | |
| T129 | iter_019 统一打包框架建立（template、目录结构、版本管理规范） | 自研 | 工具/基础建设 | | | | |
| T131 | isotonic calibration addon（iter_015 基础上加 isotonic 校准） | T15 → T131 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO 小幅正，不超 conformal（被 T127 conformal 取代） | | | |
| T132 | quantile LGB（alpha=0.5 ≈ MAE，分位数回归） | R3 → T132 | 更好的损失函数 | LOSO MAE 优于 Quantile 但均不超 SPO+ NN 组合 | | | |
| T133 | alt booster 比较（额外 GBDT booster 横向对比） | 调研 | 调研/分析 | 与 LGB 持平，不替换主力 | | | |
| T134 | direct PnL NN loss（PnL 公式梯度直接训练 NN，不含 SPO+ 正则化） | T13 → T134 | 更好的损失函数 | 梯度不稳定，不收敛（SPO+ 是更稳的端到端对齐方式） | | | |
| T135 | snapshot ensemble NN（训练过程中多个 checkpoint 的 ensemble） | 文献 snapshot ensemble | 更好的 ensemble | LOSO 边际，不超 T87 5-seed | | | |
| T136 | SPO+ 超参扫描（lambda, lr 等超参搜索） | T87 → T136 | 调研/分析 | 确认 λ=30，lr=3e-5 为最优；固化进 T87 最终配置 | | | |
| T137 | alt robust losses sweep（MAE vs L2 系统对比，多 seed 验证） | 自研 | 更好的损失函数 | MAE LGB LOSO +0.3~+0.5（边际优于 L2；平台未确认；最终 L2 LGB 仍为主力） | | | |
| T138 | mixup 数据增强（跨样本线性插值） | 文献 mixup | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | LOSO 下降（混合样本破坏 EV 信号） | | | |
| T139 | stacker NN（用 NN stacking 综合 LGB+NN 输出） | 自研 | 更好的 ensemble | LOSO 负面 | | | |
| T140/T140b/T140c | **M7 LGB full-retrain**：LGB 在 date 0-119 全部数据上重训，固定 iters=330（V4 早停平均 × 1.1），iter_019 v2 | R_full_retrain → T140 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | LOSO-equiv +41.49（不变；M7 不改变 LOSO，只改全数据模型） | **+34.44**（iter_019 v2；**+5.51 vs iter_018 +28.93，第三次大突破**） | | |
| T141/T141b | Hawkes 过程订单流特征（Hawkes intensity 5-seed LGB + LOSO） | R_Hawkes_OFI → T141 | 更好的特征 | LOSO 边际提升 | | | |
| T142 | Hawkes 5-seed full-retrain（Hawkes 特征 M7 重训候选） | T141 → T142 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | LOSO 边际，与 T141b 类似 | | | |
| T143 | iter_019 v3 MAE packaging（T87 NN + MAE LGB，全数据重训，+conformal） | T137 → T143 | 更好的损失函数 | LOSO +41.49（v2）→ +41.20（MAE LGB，-0.29）；加 conformal → +43.21（边际，平台未确认超 v2） | | | |
| T144 | Quantile/Huber/MAE ensemble diversity 评估（多种 robust loss 混合） | T143 → T144 | 更好的 ensemble | 无增量，MAE-only remains winner | | | |
| T145/T145b | iter_019 v4 MAE full-retrain（MAE LGB M7 + T87 NN ensemble） | T143 → T145 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | LOSO 平台预测 +30~+32（低于 v2 +34.44；MAE M7 LGB 不超 L2 M7 LGB） | | | |
| T146 | conformal-aware DE re-threshold（在 iter_019 v3_mae 基础上重跑 DE 阈值搜索） | T143 + T30 → T146 | 更好的执行层（decision/wrapper/threshold/conformal） | LOSO iter_019 v3_mae +41.20 → +43.96（+2.76；但 DE 超调风险高；未提交平台） | | | |
| T147 | feature pruning（按 feature importance 剪枝特征至 ~300 维） | 自研 | 更好的特征 | LOSO 轻微正，用于 v10/v11 候选 | | | |
| T148/T148b/T148c | LGB HP sweep（learning_rate, n_estimators, num_leaves 系统扫描） | 自研 | 调研/分析 | 找到轻微更优 HP，边际 | | | |
| T149 | NN T87 OOD sweep（扫描 T87 在 OOD sym 下的鲁棒性） | 自研 | 调研/分析 | 确认 T87 在 OOD sym 下表现可接受 | | | |
| T150 | conformal beta sweep（对 5 个 sym 分别扫描 beta 0.0-0.5，4-fold CV consensus 确认最优） | R_conformal_select → T150 | 更好的执行层（decision/wrapper/threshold/conformal） | 确定最优 β = {sym0: 0.10, sym1: 0.40, sym2: 0.30, sym3: 0.00, sym4: 0.00}；固化进所有后续提交 | | | |
| T151/T151b | TOD v9 packaging（Time-of-Day 特征：解析 time 字段为盘口阶段指示符） | R_time_of_day → T151 | 更好的特征 | LOSO 小幅正（进入 iter_019 v9，但最终版 T170 未使用） | | | |
| T152 | adversarial-filtered 259 维特征（对抗性验证过滤泄露特征，359 → 259 维，iter_019 v10） | T66 → T152 | 更好的特征 | LOSO 持平 | 无确认提升（特征已足够鲁棒） | | |
| T153/T154 | v9b TOD clean + v10b/v11 打包（多版本候选包构建） | T151/T152 → T153 | 工具/基础建设 | | | | |
| T155 | v2 replica reproducibility test（LGB 训练确定性验证，相同配置完全可复现） | 自研 | 工具/基础建设 | LGB 确定性确认 | | | |
| T156/T156b/T156c | NN swap：T97 替代/叠加 T87（第二组 NN 多种替换方案） | T97 → T156 | 更好的 ensemble | holdout in-sample +41.49 → +43.16（T97 替代 T87，+1.66；但 T97 与 T87 太相似） | | | |
| T157 | NN Transformer preliminary（GroupTransformer 初步探索） | T172 前期 | 更好的模型架构 | 初步实验，数字未记录 | | | |
| T158 | agreement filter wrapper（只有 NN 和 LGB 预测方向一致时才出手） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | holdout in-sample +41.49 → +41.74（+0.25；本地轻微正，T163 组合后平台失败） | | | |
| T159/T159b | per-sym dynamic abstain（每个 sym 动态计算 abstain 阈值） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | holdout null result | | | |
| T160 | v2W clean packaging（agreement filter 版本打包） | T158 → T160 | 工具/基础建设 | | | | |
| T161 | TTA（Test-Time Augmentation，K=5，5 次增强预测平均） | 文献 TTA | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | holdout T170 +149.95 → **下降**（per-batch 统计自适应在平台乱序下产生噪音） | | | |
| T162 | in-row TTT（Test-Time Training，3 种 wrapper：W3a/W3b/W3c） | 文献 TTT | 更好的执行层（decision/wrapper/threshold/conformal） | holdout T170 → W3a -0.30，W3b -1.98，W3c -3.11（全退步；违反硬约束 #2 精神） | | | |
| T163 | v2 + T97 NN + agreement filter 三路集成 | T156 + T158 → T163 | 更好的 ensemble | holdout in-sample +42.90（+1.41 vs v2 +41.49） | **~+30.74**（**-3.70 vs v2 +34.44**；T87/T97 架构相同，diversity=0，agreement filter 的本地收益被 T97 OOD 失败抵消） | | |
| T164 | 4-way ensemble（T87+T97+T95+LGB） | T163 → T164 | 更好的 ensemble | holdout +43.26（+0.34 vs T163；仍含 T97，不安全） | | | |
| T165 | magnitude filter（只保留预测幅度超阈值的高幅度预测出手） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | holdout 略正，不够显著 | | | |
| T166 | CatBoost L2 5-seed full-retrain（M7 重训 CatBoost，加入集成） | T89 + T140 → T166 | 更好的 ensemble | holdout T163 → **+0.39** | | | |
| T167 | 5-way ensemble（T87+T97+T95+CB+LGB） | T166 → T167 | 更好的 ensemble | holdout +0.11 over T166（within noise；不入 final） | | | |
| T168 | adaptive threshold V1-V4（per-batch 统计自适应阈值，4 种变体） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | holdout T170 → V1-V4 全部退步（平台测试点顺序被打乱，per-batch 统计不稳定） | | | |
| T169 | AR TTT（自回归 test-time training） | 文献 | 更好的执行层（decision/wrapper/threshold/conformal） | holdout 退步（TTT 系列全 kill） | | | |
| T170 | **T87 SPO+ NN M7 full-retrain**：在 date 0-119 全数据上重训 NN，固定 epoch=11（avg best_epoch 9.2 × 1.1），warm-start from T81，5 seeds | R_full_retrain → T170 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | holdout in-sample v2 LGB-only +109.68 → v2N（NN 加入）+118.16（+8.49，但含 in-sample 污染） | **+34.64**（iter_019 v2N；**+0.20 vs v2 +34.44**；NN M7 增量比 LGB M7 小） | | |
| T171 | LGB HP sweep（扫描 learning_rate, n_estimators, num_leaves 在 M7 重训下的影响） | T148 → T171 | 调研/分析 | LOSO 轻微边际提升，未进入 final 包 | | | |
| T172/T172b | GroupTransformer（特征分组+注意力，d_model=64，nhead=4，2 layers，n_tokens=32，CLS token） | 自研 / Transformer 文献 | 更好的模型架构 | holdout in-sample（全 date 0-119 训练）：Transformer only +149.91；best ensemble [NN=1.0, TR=0.5, LGB=1.0] **+164.71**（全 in-sample 污染） | 未提交（全 in-sample，无真实 OOD 验证，新架构 M7 retrain 高风险） | | |
| T173/T174 | multi-horizon T170 NN 投票（h_5/10/20/40 各使用 T170 NN 风格模型投票，h_60 作主导） | T170 → T173 | 更好的 ensemble | LOSO 理论有效，但短 horizon 平台始终不稳 | | | |
| T175 | Transformer T170NN 打包（GroupTransformer + T170 NN zip 构建，含 in-sample 风险） | T172b → T175 | 工具/基础建设 | | | | |
| T176/T176b | T87M7 NN HP variant sweep（ep={7,11,15,20}, lr={1e-5,3e-5,1e-4}, λ={10,30,100}，各变体 holdout 149-151 范围内波动） | T170 → T176 | 调研/分析 | holdout 各变体 149-151 范围（噪音级）；HP 多样性为 T188v2 ensemble 提供多样性来源 | S1_ep15 variant **+34.51**（vs T170 +34.64，-0.13；单 HP variant 无优势） | | |
| T177 | wrapper tricks B/C（两种 per-batch 统计 wrapper 变体） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | holdout 正，B/C 均轻微提升，未入 final | | | |
| T178 | LGB 10-seed（从 5-seed 扩展到 10-seed） | T140 → T178 | 更好的 ensemble | holdout **null result**（差异噪音级；5-seed 已在 variance floor；单纯加 seed 无红利） | | | |
| T179 | 40-NN ensemble（8 HP 变体 × 5 seeds，40 个 NN + 5 LGB） | T176 → T179 | 更好的 ensemble | holdout in-sample T170 +149.95 → **+150.57**（+0.63） | **+34.39**（**-0.25 vs T170 +34.64**；只扩 NN 侧无平台收益，5-seed 已在 variance floor） | 推理 816ms/1024-batch（vs T170 855ms） | |
| T180 | OOD GMM density abstain wrapper（PCA 20d + GMM k=5 密度估计，低密度点弃权） | 文献 OOD detection | 更好的执行层（decision/wrapper/threshold/conformal） | holdout T170 +149.95 → V1(p10) +117.63（-32.31）；V2(p20) +92.65（-57.29）；V3(p30) +71.50（-78.44）；测试数据比训练更 in-distribution，弃权损失高 PnL 点 | | | |
| T181 | time-window hard abstain（在开盘/收盘前后强制弃权，解析 time 字段） | 自研 | 更好的执行层（decision/wrapper/threshold/conformal） | holdout T170 → V1(去 13.19%) -23.39；V2(AM 开盘后 5min) -0.27；V3(收盘前 5min) -2.38（T170 conformal 已过滤低置信预测，弃权适得其反） | | | |
| T182 | 去掉 conformal wrapper（ablation：T170 无 per-sym beta conformal） | T170 → T182（ablation） | 调研/分析 | holdout T170 +149.95 → T182 +156.56（**+6.62**！但 in-sample artifact：sym1 beta=0.4 delta +5.34，sym2 delta +1.12） | **+34.34**（**-0.30 vs T170 +34.64**；**证明 conformal 在平台 OOD 上有效**，本地 holdout 显示的 conformal 负面是 in-sample artifact） | | |
| T183 | NN random init M7 retrain（无 T81 warm-start 的 NN，直接随机初始化后 M7 重训） | 自研 | 更好的训练方法（如 SPO+ DFL, M7 全数据重训, fine-tune 协议） | holdout T170 +149.95 → **+150.57**（+0.63，几乎与 T81 warm-start 持平；不同初始化带来 diversity） | | | |
| T184/T185 | 深度 CNN（raw features / hybrid CNN-MLP，深度卷积处理原始特征序列） | 文献 DeepLOB CNN | 更好的模型架构 | in-sample 数字看起来不错，OOD 未验证（未构建提交包） | | | |
| T186 | T184/T172b retrain with val + LR sweep（尝试 early stop + LR sweep 修复深度架构） | T184 → T186 | 调研/分析 | 结论：architecture_doesnt_fit（深度 CNN/Transformer 在当前特征框架下无法与 MLP 竞争） | | | |
| T187 | GroupTransformer replace/add（Plan A：替换 T87 NN；Plan B：加入 v2 stack） | T172b → T187 | 更好的 ensemble | holdout in-sample Plan A +22.2 vs T170，Plan B +12.6/+14.7（**全 in-sample 污染**，GroupTransformer 训练包含测试日期） | 未提交（全 in-sample 污染，新架构 M7 retrain 高风险） | | |
| T188 | 50+50 mega ensemble 初版（50 NN + 50 LGB，存在 T81 pretrain val/train 未正确分离的 warm-start 问题） | R_multinn_spo / R_50seed_bag → T188 | 更好的 ensemble | holdout in-sample ~+150（边际优于 T170 +149.96，但有 pretrain 问题） | | | |
| T188v2 | **50+50 mega ensemble 修正版**（50 NN 独立 T81-style L2 pretrain + T87 SPO+ M7 11 epoch；50 LGB 5HP×10seed M7；simple mean；相同 conformal + thresholds） | T188 修正 + R_50seed_bag | 更好的 ensemble | holdout in-sample T170 +149.96 → T188v2 **+150.43**（+0.48，噪音级；in-sample 看不出 ensemble 红利） | **+35.64**（**+1.00 vs T170 +34.64**；**当前 SOTA**；OOD 上 variance reduction 兑现，+2.9%） | | |
| T188v3 | 50 NN 推理改为 batched torch CUDA bmm（所有 50 NN 权重堆叠 (50, out, in) 张量，单次 torch.bmm 替代 50 次 numpy sequential 循环） | 自研 | 更快的推理 | | | NN-only per-1024-batch：baseline spec 1509ms → **8.1ms**（**186×**）；实测 554.8ms → 8.1ms（68.5×）；e2e：2142ms → **641ms**（**3.3×**）；442k 行总时间 15.4min → 4.6min | 平台预计 50 NN 推理从超过 1h 降至约 10-15min |
| T188v3_fullhorizon | 最终打包时 5 个 horizon 共享同一套 h=60 ensemble 预测（h_5/10/20/40 也用 T188v2 h_60 ensemble 输出） | 自研 | 更快的推理 | | | 单 horizon 推理时间 × 5 → 单 horizon 推理时间（5 个 horizon 只需推理 1 次） | 5 个 horizon 提交只需 1× inference cost |
| T189/T189_GPU_pkg | T170（5+5）转为 torch GPU 推理包（CPU → GPU inference） | 自研 | 更快的推理 | | | CPU 推理 855ms/1024-batch → GPU 版本更快（具体 benchmark 确认） | 平台 GPU 环境可加速推理 |
| T190 | 150+150 mega ensemble（从 50+50 扩展到 150 NN + 150 LGB，4 台远程机器并行训练；更广 HP grid） | T188v2 → T190 | 更好的 ensemble | holdout in-sample T188v2 +150.43 → T190 **+140.62**（**-9.81**；HP 多样性反成噪音） | **+34.59**（**-1.05 vs T188v2 +35.64**；ensemble 扩容收益非单调：5→50 获 +1.00，50→150 获 -1.05） | | |
| T191 | trend features pilot（+5 维：mean_logret_W10/20/50/100，mid_pct_change_W100，总维 370→375→364 有效） | 自研 | 更好的特征 | holdout T191 standalone +140.10（vs T188v2 +150.43，**-10.33**）；corr(T191 pred, T188v2 pred) = 0.9941，IC_residual_pooled = -0.1699（负！） | | | |
| T192 | bounded NNLS [1/1500, 2/150] per-group 权重学习（在 T190 150+150 ensemble 上用 NNLS 学习 per-model 权重） | 自研 | 更好的 ensemble | holdout in-sample T192 H4：val **+180.41**，test **+193.94**（val/test ratio=1.075；**严重过拟合 signal**；val+test 均 in-sample，condition number=54182，NN-NN 相关=0.9331） | **+31.0**（**-4.64 vs T188v2 SOTA +35.64**；OLS overfit 确认，val+test 均 in-sample，平台完全失效） | | |
| R1/v2 | date decay + sym augmentation（日期 decay 权重研究，近期样本权重更高） | 文献 date decay | 调研/分析 | date decay 对 GBDT 作用有限，边际 | | | |
| R2 | Group DRO（分组分布鲁棒优化，对 sym 分组均匀优化） | 文献 Group DRO | 调研/分析 | fold2 改善但 fold4 牺牲，净效果负 | | | |
| R3 | HYD quartet（Huber/asymmetric/Y-shape/Date-decay 损失函数四元组调研） | 文献 | 调研/分析 | Huber 略优 L2（见 T99） | | | |
| R4 | logret lag 特征调研（对数收益率滞后特征有效性研究） | 文献 | 调研/分析 | 无增量（见 T124） | | | |
| R5/v2 | Bayesian optimization 超参搜索（与 DE 效果对比） | 文献 Bayesian optim | 调研/分析 | 与 DE 相近，无明显优势 | | | |
| R10 | PnL-aware loss 10 方案调研（端到端 PnL 梯度/SPO+/Huber/分位数等 10 种方案系统对比） | 文献 DFL/SPO+ | 调研/分析 | SPO+ 最稳，端到端 PnL 梯度不收敛；确认 SPO+ 路线 | | | |
| R11 | cross-stock OOD robustness 文献调研（sym-agnostic 建模方案） | 文献 HFT OOD | 调研/分析 | sym-agnostic 设计合理，确认我们 T7/T9 方向正确 | | | |
| R12 | ensemble & stacking 调研（12 方案 + 6 paper；OOF 过拟合风险分析） | 文献 ensemble | 调研/分析 | OOF 过拟合风险高；指导 iter_016 失败复盘和 T188v2 设计 | | | |
| R13 | probability calibration + position sizing 调研（isotonic/platt/beta 等） | 文献 calibration | 调研/分析 | isotonic/platt 均负；beta conformal 后来在 T127 实现正效果 | | | |
| R30 | 数据特征深度调研（解释 h_60 有效、h_5/10/20 失效原因；手续费 vs α threshold 分析） | 自研分析 | 调研/分析 | 确认 h_60（α=0.1%）是最有利 horizon；手续费 0.02% 摧毁 h_5/10（α=0.05%）alpha | | | |
| R31 | 2024-2026 HFT/LOB SOTA 文献调研（9 篇深度笔记；TransLOB、TabLOB、N-BEATS 等） | 文献综述 | 调研/分析 | 指导 T172 GroupTransformer 设计；确认 MLP+GBDT 集成在实际竞赛中 competitive | | | |
| R34 Stage 1-5 | sym-invariant 特征建设完整调研（从 154 维 SchemeC 到 359 维 SchemeP，5 个阶段，+205 维） | 自研 + HFT 文献 | 调研/分析 | 催生 T44→T51→T53→T55→T68 实验，SchemeP 359 维最终进入所有提交 | | | |
| R_50seed_bag | 50-seed bagging 理论与前期可行性研究（分析 variance reduction 与 HP 多样性关系） | 自研 | 调研/分析 | 为 T188v2 50+50 设计奠基；确认 HP 多样性（非单纯 seed 多样性）是 variance reduction 关键 | | | |
| R_Hawkes_OFI | Hawkes 过程订单流特征深度调研（自激过程订单到达率建模） | 文献 Hawkes process | 调研/分析 | 边际有效，催生 T141/T142；最终未进入 final 包 | | | |
| R_NaN1/R_NaN2 | LGB/CatBoost NaN passthrough 合规性审计 | 自研 | 调研/分析 | NaN 处理合规，模型 NaN passthrough 行为确认 | | | |
| R_Pairwise_ALL | 特征 pairwise 交互调研（计算所有特征对之间的交互增益） | 自研 | 调研/分析 | 计算量大，边际；未进入 final | | | |
| R_conformal_select | per-sym conformal beta 搜索（4-fold CV consensus 确定每只股票的最优弃权阈值） | 自研 | 调研/分析 | 确定 β = {0.10, 0.40, 0.30, 0.00, 0.00}；固化进 T127/T150/T170/T188v2 所有后续提交 | | | |
| R_cross_sym | 跨 sym 特征调研（利用多只股票截面信息的特征设计） | 自研 | 调研/分析 | **ABORT**：平台批次顺序打乱，多 sym 无法同时进同一 batch；违反硬约束 | | | |
| R_full_retrain | M7 full-retrain 可行性研究（量化 LGB/NN 全数据重训的安全性和收益） | 自研 | 调研/分析 | 确认 LGB M7 安全可行（330 iters 早停一致）；催生 T140/T170 第三次大突破 | | | |
| R_hawkes/R6 | Hawkes 进阶调研（更复杂的 Hawkes 特征变体） | T141 → R_hawkes | 调研/分析 | 边际，未入 final | | | |
| R_multinn_spo | 多 NN SPO+ 组合研究（分析多 NN 集成的 diversity 来源） | 自研 | 调研/分析 | 关键发现：**多样性来自 HP 差异而非 seed 差异**（T97/T87 同架构无 diversity）；催生 T188v2 设计 | | | |
| R_multitick_window | 多 tick 窗口特征调研（不同时间尺度的窗口统计特征） | 自研 | 调研/分析 | 部分进入 R34 Stage 5 SchemeP | | | |
| R_roll_tsrv | Roll effective spread + TSRV 特征调研（Roll's model 有效价差估计 + 两尺度已实现波动率） | 文献 Roll/TSRV | 调研/分析 | 有价值，Roll's effective spread 进入 R34 Stage 3（T53）；未成为主力特征 | | | |
| R_stack3_compound | 3 层 stacking 调研（多层 meta-learner 叠加方案） | 文献 stacking | 调研/分析 | OOF 过拟合，放弃；与 R12 结论一致 | | | |
| R_time_of_day | 盘口时段特征研究（AM/PM 开收盘时段对预测质量影响） | 自研 | 调研/分析 | LOSO 小幅正（催生 T151 TOD v9）；最终未进入 T170 final 包 | | | |
| R_T75L2_* | T75 LGB L2 系列变体研究（advval/bayes/pairwise/symaug 4 种变体对比） | T75 → R_T75L2 | 调研/分析 | 全部边际，无超越 T75 原版的变体 | | | |
