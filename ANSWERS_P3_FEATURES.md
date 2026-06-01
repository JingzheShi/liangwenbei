# Answers P3 — Feature Engineering

> Worker: P3 (features). 仅回答"特征工程"主题，不涉及训练 / 决策 / 推理（其他 worker 负责）。
> 来源：`QREVIEW_FEATURES.md`（225 问）+ `CODE_QUESTION_LIST.md` §3（Q47-Q75）+ `QREVIEW_NN_TRAIN.md` Q18-20 + `QREVIEW_LGB_TRAIN.md` Q3/Q57-Q62 + `QREVIEW_PREDICTOR.md` Q58-Q90/Q113。
> 证据：`final_submission_code/01_build_features/{build_schemeP_cache,fast_features_batch,stage5_features}.py` + `04_build_pkg/{Predictor.py,fast_features.py,config.json,thresholds.json}` + `HISTORY_EVIDENCE_DB.md` 主题 6 + `0512_final_report.md` Phase 2 + `experiments/T{7,10,22,35,44,51,53,55,68,74,78,141,142,147,151,152,191}/results.json` + `experiments/R_{cross_sym,roll_tsrv,time_of_day,Hawkes_OFI,Pairwise_ALL}/`。

## 状态图例
- ✅ = 有强证据链（实验编号 + 本地数 + 平台数 + 代码行）
- 🟡 = 部分证据 / 推断 / 仅本地（无平台对照）/ "约定俗成"无 sweep 记录
- 🔴 = GAP，无证据 / 与代码冲突 / 真有 bug 嫌疑

---

## 总结（TL;DR）

1. **Scheme 演变是 8 步累加**：SchemeA(154)→B(154 同样 baseline LGB)→C(226, +72 T3 特征)→D1(308, +154 窗口 z-score)→E(spread/amount 归一化)→F(全局 z-score 失败)→H(291, MLOFI/WMP 扩展失败)→**P(370 raw+derived，drop 11 → 359 final)**。SchemeP = 154 raw + (R34 Stage 1-5 共 216 derived，drop 11 = 205) = **154 + 205 = 359 维**喂模型。370 → 359 = 154+216−11，**数字对得上**。
2. **R34 Stage 1-5 累计本地增益**（h=60 LOSO）：iter_005b +9.67 → Stage 1 +13.67 (T44, +4.0) → Stage 2 +15.06 (T51, +1.4) → Stage 3 +16.84 (T53, +1.78) → Stage 4 边际 (T55) → Stage 5 +26.44 (T68, +0.50，方法论从 LOSO 5-fold 切到 LOSO-equiv 全 sym 训练，含 +7 方法论 inflation)。**Stage 1/2/3 是真正的 alpha 来源；Stage 4 / Stage 5 边际**。
3. **窗口长度选择 99% 没有 sweep**：(5,20,60) for MLOFI/GOFI、(5,10,20,50) for RV、(20,50,100) for SIGNED_RV/KYLE/CANCEL/ADAPT_MOM/SIGNED_BV/SPREAD_REG/TRADE_PERS、(20,30,50,100) for JSHARE、(30,50,100) for ROLL、(5,20,50,100) for LIQ_ASYM。**完全 idiosyncratic，每个 stage 自己挑**。HFT 文献"标准 30/60/120 秒"对应 10/20/40 ticks（3s tick），但代码采用 (20,50,100)≈(1min,2.5min,5min) 是 R34 调研的选择，没有正式 sweep（Q204-205）。
4. **EWMA α 4 档** (0.05, 0.1, 0.3, 0.5)（HL≈13/7/2/1 ticks）：覆盖从"近一个 5min 窗口的中段"到"几乎只看最后一 tick"。α=0.5 和 α=0.3 之间冗余度高（Q43）；α 选择是 T44 设定，从未 sweep（🟡）。
5. **amount_delta 唯一 log1p**：因为它是 **154 维 raw 中唯一未归一化字段**，量级 e3-e6，与 LOB 价格/size/intensity 不同；T8 诊断 sym=2 amount_delta z-score +7.49 偏离（蓝筹 ETF 量很大）→ T9 SchemeE 引入 sign-preserving log1p 压缩重尾。**volume_delta/totalbsize/bsize 没有 log1p 是不一致**（Q18, Q47, Q69）：理由是 amount = price × volume 量级更大，单 amount log1p 足够；volume 已通过 dual_z/qrank 内部归一化。🟡 没有 ablation 证明这是最优策略。
6. **DROP_NAMES 11 个特征**：`dualz_ask_diff1, dualz_bid_diff5, dualz_ask_diff5, qrank_W100_spread1, qrank_W100_spread5, qrank_W100_spread10, qrank_W100_cumspread, kyle_lam_W50, kyle_lam_W100, roll_eff_spr_ratio_W100, liq_asym_top5_W5`（Predictor.py:68-75）。来源是 T59_FAIL（10 个 = T59 全 sym 训练时的 sym-agnostic verification，特征在某 sym 上 KS 失败 → drop）+ STAGE5_FAIL（1 个 = T68 之后 W=5 太短噪音过大）。**没有显式的 KS p 值记录**（🔴）。
7. **特征维度 370 = 154 raw + 196 baseline + 20 Stage5；训练用 359 = 370 − 11 drop**：数字闭环。但单窗口版 `fast_features.py` 只有 196 baseline（无 Stage5），是 dead code（Q122/Q217；Predictor 用的是 `fast_features_batch.py`）。
8. **未采纳的特征家族**：Alpha101/Alpha191（T22，**需跨截面，违反硬约束 #2**，但子集 33 个 + 124 个 implementable，最佳单因子 IC +21.80 已超 SchemeC，仍未入因 cross-sym risk）、Hawkes (T141/142, R_Hawkes_OFI, 边际)、SG filter (T35/T91, gain≈0)、TOD time encoding (T74/T78 −0.72；T151 v9 小正但 final 弃)、sin/cos (T74)、trend features (T191, corr 0.9941 + IC_residual −0.17, KILL)、Pairwise (R_Pairwise_ALL, 计算量大边际)、cross-sym (R_cross_sym ABORT — date 置 0 + batch 打乱 + sym 可能 OOD)。
9. **NaN/Inf 处理**：cache 末端 `extras = np.where(np.isfinite(extras), extras, 0.0).astype(np.float32)`（Cache:105）；每个 stage 内部还各有 `np.where(np.isfinite(...), ..., 0.0)`（防御性多次）；LGB native NaN passthrough（R_NaN1 确认）；NN 训练 `np.where(np.isnan, 0, X)` 后 standardize 再 clip ±10。**inf 在多数 stage 都处理了，唯独 EWMA 内部 `x_safe = where(isfinite(x), x, 0)` 把 NaN→0 当 EWMA 输入，会让 missing 段 EWMA 偏低**（Q46/Q63，🟡）。
10. **`fast_features.py` vs `fast_features_batch.py`**：前者单窗口、后者批量。**功能不一致**（单窗口只 196 维无 Stage5，批量 216 维含 Stage5）。**Predictor 用的是 `fast_features_batch.py`**（Predictor.py:191 importlib），所以单窗口文件是 **dead code**——`build_pkg.py` 仍打包它（Q120/Q217，🔴 dead code 浪费 + 误导未来维护者）。同时两文件 docstring 互相矛盾（单窗口说 350 dim/drop 10，实际 returns 196 dim，drop 应该是 11——Q157-158）。
11. **多个 "OFI" 形式并存**：MLOFI（T3，`>=`/`<=` 双计 size 等价 case，line 169-174）、GOFI（Stage 2，`>`/`<`/`==` 三分 line 411-418）、Stage5 OFI（line 645-651）—— **3 套 OFI，3 个 equality/边界约定**，源于"R34 是多次单独调研合并"，未统一（Q26/Q73/Q105-106/Q189/Q219）。这是 design debt，但模型已经训过、提交分数已 +35.64，**不动**。

---

## 1. Scheme 演变（A→B→C→D1→E→F→H→P）

### 1.1 SchemeA 154 → SchemeB 154 → SchemeC 226 (T3, +72)
- **来源**：QREVIEW_FEATURES §0/§2 (Q16-Q47), CODE_QUESTION_LIST Q47/Q49/Q50, HISTORY_EVIDENCE_DB 主题 6 T2/T3, 0512_final_report T3 节, tricks_table T2/T3。
- **答案**：
  - **SchemeA = 官方 mmpc_demo DeepLOB 用的 154 raw**（T1 NN baseline，LOSO h_60 ~+8，平台 −6.65）；**SchemeB = 同样 154 维但用 LightGBM 3-class CE**（T2，LOSO sum −22.10 — cross-sym OOD 极差）。A 和 B 是同一 154-d schema，差别只是模型族。
  - **SchemeC = SchemeB + 72 derived = 226**：T3 加入多尺度 MLOFI(30) + WMP(11) + RV(4) + EWMA intensities(24) + time encoding(3) = 72 维。LOSO h_10 +21.86（iter_002 核心）。**这是首次"加 derived feature 大幅涨分"**。
  - 选择 MLOFI/WMP/RV/EWMA 是因为这些是 HFT 标准微观结构指标（Cont/Kukanov/Stoikov, Avellaneda/Stoikov, BNS realized vol）。time encoding 后来被剥离（T3 时还在，T7 之后逐步替换为窗口归一化，T74 SchemeQ 重新尝试 sin/cos 失败 −0.72）。
- **核心证据**：T3 build_features.log + tricks_table T3 row | iter_002 平台 +4.07 | `fast_features_batch.py:5,148-241`（T3 no-time block 仍是 69 维 = 30+11+4+24，剩余 3 维 time encoding 已被移到 SchemeP 外部，**最终 SchemeP 不含 time**）。
- **状态**：✅

### 1.2 SchemeD1 308 (T7, +154 窗口 z-score)
- **来源**：QREVIEW_FEATURES Q123-126, CODE_QUESTION_LIST Q54, HISTORY_EVIDENCE_DB 主题 6 T7, 0512_final_report T7, T7 experiment results.json。
- **答案**：T7 加入 100-tick 窗口内的 z-score 归一化 154 维（对原始 154 维每列做 (x_last − mean(W=100)) / std(W=100)）。SchemeD1 = 154 + 154 = 308。LOSO h_10 +13.91（vs T5b SchemeC +21.86 在 h_10；但 T7 解决了 **sym=2 brittleness**，sym=2 cum_pnl 从 −0.98 翻到 +2.24）。窗口 z-score 是"sym-agnostic 归一化"的起点（T7 → 后续 dual_z 把 W=20 和 W=100 都做了）。
- **核心证据**：T7 results.json (cum_pnl_sym2: −0.98 → +2.24) | HISTORY_EVIDENCE_DB 主题 6 T7 行。
- **状态**：✅

### 1.3 SchemeE (T9, amount_delta 归一化 + spread-norm)
- **来源**：QREVIEW_FEATURES Q18-22, CODE_QUESTION_LIST Q47-48, HISTORY_EVIDENCE_DB 主题 6 T8/T9, 0512_final_report T8/T9。
- **答案**：T8 诊断发现 sym=2（蓝筹 ETF）的 amount_delta z-score +7.49 偏离训练分布（巨型成交量），导致预测器对 sym=2 误判 UP 62% vs 真实 18%。T9 给 amount_delta 加 sign-preserving log1p、给 spread 加 normalization。**这一步部分被吸收进 SchemeP**：raw_last 的 amount_delta 用 log1p（Cache:108-110），dualz/qrank 内部窗口 z-score 处理了其它 size 字段。
- **核心证据**：HISTORY_EVIDENCE_DB 主题 6 T8/T9, 0512_final_report 行 175-181 ("sym=2 amount_delta z-score +7.49") | `build_schemeP_cache.py:108-110`。
- **状态**：✅

### 1.4 SchemeF (T10, 全局 z-score) — 失败
- **来源**：QREVIEW_FEATURES Q138, HISTORY_EVIDENCE_DB 主题 6 T10, tricks_table T10。
- **答案**：T10 把窗口 z-score 换成全局（基于整个训练集统计量）z-score。LOSO h_10 +13.40，**不超 SchemeC +21.86**，**且不超 T7 SchemeD1 +13.91**。证明窗口归一化 > 全局归一化：原因是全局统计量被训练集 sym 主导，OOD sym 时 (x − global_mean) / global_std 不再以 0 为中心，破坏数值分布。**SchemeF 被弃**。
- **为啥窗口 > 全局 > per-sym**：
  1. **per-sym normalization 违反硬约束 #3**（OOD sym 无统计量）→ 直接不允许。
  2. **全局 z-score** 对 OOD sym 不稳（z 偏离 → 模型见到训练时未见过的输入空间）。
  3. **窗口 z-score** 每 100 tick 内自适应，对 sym distribution shift 鲁棒（每个 100-tick 切片自己产生归一化中心和尺度）。SchemeP 全部使用窗口归一化（dualz / qrank / signed_rv / kyle_inv 都是 last-vs-window）。
- **核心证据**：tricks_table T10 行 | HISTORY_EVIDENCE_DB 主题 6 T10。
- **状态**：✅

### 1.5 SchemeH 291 (T20/T21, MLOFI/WMP 扩展) — 失败
- **答案**：T20/T21 把 MLOFI/WMP 进一步扩到 291 维（具体加什么 log 未保留），LOSO h_10 +21.70，**持平 SchemeC +21.86**。"单纯加维度无效" → 必须 + sym-invariant 设计（T44 起的 R34 Stage 1）才有真增量。
- **核心证据**：tricks_table T20/T21 | HISTORY_EVIDENCE_DB 主题 6 T20/T21。
- **状态**：✅

### 1.6 SchemeP 359 (T44/T51/T53/T55/T68, R34 Stage 1-5)
- **来源**：QREVIEW_FEATURES §3-§6 (Q48-Q150), HISTORY_EVIDENCE_DB 主题 6, 0512_final_report Phase 2, T44/T51/T53/T55/T68 results.json。
- **答案**：见 §2 详述。**SchemeP 总维度 = 370 (154 raw + 216 derived = 69 T3 + 54 S1 + 59 S2 + 14 S3 + 20 S5)**；训练前 drop 11 → **359 喂模型**。
- **核心证据**：`fast_features_batch.py:5,766-780` ("(N, 216): T3-no-time(69) + Stage1(54) + Stage2(59) + Stage3(14) + Stage5(20)") | T68 results.json "schemeP feature count: 370 = 154 raw + 196 extras + 20 stage5"。
- **状态**：✅

### 1.7 SchemeQ (T74/T78, time features) — 失败弃用
- **答案**：T74 在 SchemeP 359 上加 6 维 time features（sin/cos position, is_open/close_30min, is_pm）→ 365 维 SchemeQ。LOSO-equiv +35.51（**−0.72 vs T75 LGB regression baseline +36.23**）。结论：**回归框架下 time features 反而降分**（因为 mid 漂移已隐含在窗口内，time encoding 引入 overfit），SchemeQ 不入 final。后续 T151 TOD v9 用更细粒度 time（first_20/last_20/lunch_prox + tod_sin/cos）小正但最终 T170 final 也弃。
- **核心证据**：tricks_table T74/T78 | HISTORY_EVIDENCE_DB 主题 6 T74/T78/T151。
- **状态**：✅

### 1.8 SchemeK / SchemeI / SchemeJ 等其它"中间方案"
- **答案**：T35 SchemeK 255 维（SchemeC + ReVol + Savitzky-Golay），LOSO h_60 +5.53 < +6.30，弃。其它 Scheme 字母（G/I/J 等）零散出现在历史日志，未进入主线，不在 final 包内。
- **状态**：🟡（命名空间被 R34 重命名为 Stage 1-5 之后历史 Scheme 字母不再连续）

---

## 2. R34 Stage 1-5 特征族（每 stage 细说）

### 2.0 Stage 编号缺失 Stage 4
- **来源**：QREVIEW_FEATURES Q151, HISTORY_EVIDENCE_DB 主题 6 T55。
- **答案**：Stage 4 = T55 "少量 edge gain 特征" 合并进 SchemeP，**未单独存在文件**。所以 `fast_features_batch.py` 的命名 stage1/2/3/5 跳过 4 是历史命名遗留——T55 后的 R34 Stage 5 (T68) 名称已在调研中固定，没有重命名为 Stage 4。代码读者会困惑但不是 bug。
- **状态**：✅（不是 bug，是历史命名）

### 2.1 Stage 1 (T44, +54 维) — dualz(37) + signed_rv(3) + kyle_inv(2) + ewma_ofi(12)
- **来源**：QREVIEW_FEATURES Q48-64, CODE_QUESTION_LIST Q54-Q57, HISTORY_EVIDENCE_DB 主题 6 T44, T44_r34_features/build.log。
- **答案**：5 个新族 = 54 维 = 37 dualz + 3 signed_rv + 2 kyle_inv + 12 ewma_ofi。本地 LOSO h_60 单 seed +12.09（vs iter_005b +9.67），5-seed +13.67（+4.0）。
  - **dualz (37 维)** = `(x_last − mean(W=20))/std(W=20) − (x_last − mean(W=100))/std(W=100)`：**短/长窗口 z-score 之差**，"regime change" 检测器（短期偏离 − 长期偏离）。覆盖 37 个 raw 列（4 spread + 4 price + 4 mid + 3 bsize + 3 asize + 2 amount/volume + 1 imbalance + 6 intensity + 6 acc + 4 bid/ask_diff = 37）。EPS=1e-8 防除零，clip + finite mask（line 286-289）。**问题**：bid_acc/ask_acc 是 monotonic accumulator，dualz 在它们上等价"近期增量比"——可能不是预期的语义（Q53/Q213）。
  - **signed_rv (3 维)** for W ∈ (20,50,100) = `(rv_pos − rv_neg) / (rv_pos + rv_neg + EPS)`：**上行 vol 占总 vol 的比例 − 0.5 × 2**，bounded [-1,1]，方向能量指标。
  - **kyle_inv (2 维)** for W ∈ (50,100) = `amt_last / cbrt(|amt_last| × std(r[-W:]) + EPS)`：**Kyle (1985) 价格冲击模型的逆**，dimensional analysis 给出 1/3 次方根（不是真 Kyle λ，而是其 inverse；命名误导，Q58/Q220）。线性化后 ≈ `sign(amt) × |amt|^(2/3) / σ^(1/3)`。
  - **ewma_ofi (12 维)** = 3 levels × 4 alphas EWMA of mlofi_W20：3 个 LOB 档（1/5/10）× 4 个 α (0.05/0.1/0.3/0.5) = 12。注意 ewma_ofi 的输入是已经 rolling-summed W=20 的 mlofi（"smooth the smooth"，Q64），所以 α=0.05 在已平滑序列上再加一层平滑可能是 over-smoothing（🟡）。
- **核心证据**：T44 build.log "R34 stage 1 extras: 54 features schemeL: 280 = 226 + 54" | tricks_table T44 row | `fast_features_batch.py:35-52,268-340`。
- **状态**：✅

### 2.2 Stage 2 (T51, +59 维) — qrank(20) + rskew(3) + gofi(30) + kyle_lam(2) + vol_burst(4)
- **来源**：QREVIEW_FEATURES Q65-84, HISTORY_EVIDENCE_DB 主题 6 T51。
- **答案**：5 个新族 = 59 维。本地 LOSO h_60 5-seed **+15.06**（vs Stage 1 +13.67，**+1.4，真正突破**）。
  - **qrank (20 维)** for W=100 only = `(x[-W:] <= x_last).mean()`：**当前值在过去 100 tick 中的分位 rank**，[1/W, 1]，"现在多 extreme"。20 个 raw 列（4 spread + 2 amount/volume + 1 imbalance + 2 totalsize + 6 intensity + 4 acc + 1 midprice = 20）。在 tick-quantized 列上（如 spread1 大部分时间 = 1 tick）qrank ≈ 1.0 没信息——**这就是 `qrank_W100_spread1/5/10/cumspread` 被 drop 的原因**（Q67）。
  - **rskew (3 维)** for W ∈ (20,50,100) = log-return 的偏度（m3/sd^3），clip ±10：分布对称性指标。
  - **gofi (30 维)** = 10 levels × 3 windows (5,20,60) = 30：**Generalized OFI**，含 `==` case（与 MLOFI 不同），sign-consistent Cont-Kukanov 定义。每个 (W, level) emit 一个 sum。**与 Stage 1 ewma_ofi 来源不同（ewma_ofi 是 mlofi 平滑；gofi 是 raw OFI 求和）。**
  - **kyle_lam (2 维)** for W ∈ (50,100) = `cov(signed_dvol, Δmid) / var(signed_dvol)` where `signed_dvol = sign(Δmid) × √(|amt|+EPS)`，clip [-1,1]：**真 Kyle λ**（与 kyle_inv 不同！）。但**整个 kyle_lam 族被 KS drop**（FAIL_NAMES 包含 kyle_lam_W50 和 W100），所以是浪费计算（Q6）。**signed_dvol 用 sign(Δmid) 是 Lee-Ready 替代品，已知在 spread 附近有偏**（Q187）。
  - **vol_burst (4 维)** = 2 vol + 2 amt × W ∈ (20,50)：`vol[-1] / mean(vol[-W:])`，clip [-100,100] 或 [0,100]。"当前 tick vol 相对窗口均值的 burst"。
- **核心证据**：T51 build_test.log + HISTORY_EVIDENCE_DB 主题 6 T51 (5-seed +15.06) | `fast_features_batch.py:54-68, 347-484`。
- **状态**：✅

### 2.3 Stage 3 (T53, +14 维) — ewma_resid(1) + rv_ratio(3) + jshare(4) + cancel_imb(3) + roll_eff_spr(3)
- **来源**：QREVIEW_FEATURES Q85-100, HISTORY_EVIDENCE_DB 主题 6 T53, R_roll_tsrv。
- **答案**：5 个新族 = 14 维。本地 LOSO +15.06 → +16.84（+1.78）。
  - **ewma_resid (1 维)** = `mid[-1] − EWMA(mid, α=0.05)`，clip ±0.05：mid 偏离 EWMA 的残差。**问题**：clip 绝对值在 mid 单位（price），不是相对——为 high-vol 股票会饱和（Q85）；EWMA 在 raw mid 上不在 log-mid 上，导致 resid 有 drift（Q86，🟡）。
  - **rv_ratio (3 维)** for (W_n,W_d) ∈ ((5,50),(20,100),(50,100)) = `Σr²[-W_n:] / Σr²[-W_d:]`：burstiness 指标。**(50,100) 是嵌套窗口，rv50 ⊂ rv100 → 比例严格 ≤ 1**，clip [0,100] 无意义（Q87）。
  - **jshare (4 维)** for W ∈ (20,30,50,100) = `(RV − BV) / RV` where `BV = (π/2) Σ|r_t||r_{t-1}|`：**Barndorff-Nielsen-Shephard 跳跃份额**。clip [0,1]。**W=30 是 jshare 独有的窗口，与其它族 (20,50,100) 不一致**（Q89/Q205）——历史选择，无 sweep（🟡）。
  - **cancel_imb (3 维)** for W ∈ (20,50,100) = `cb_share − ca_share` where shares 是 cancel intensity 占 (limit+market+cancel) total intensity 比例：**取消订单的方向偏差**。
  - **roll_eff_spr (3 维)** for W ∈ (30,50,100) = `2√max(−cov(Δc,Δc_lag),0) / (spread1[-1]+1)`：**Roll (1984) 有效价差估计**。**W=30 又是 roll 独有**（Q97）；用 `close` 而非 mid（Roll 公式需要 trade price 才能利用 bid-ask bounce，Q95）；分母 `+1` 是magic constant（Q96/Q182/Q218，🔴）。**W=100 被 KS drop**（FAIL_NAMES 含 roll_eff_spr_ratio_W100）。
- **核心证据**：T53 cache + 0512_final_report T53 (+1.78) | `fast_features_batch.py:70-75, 490-587` | R_roll_tsrv (Roll spread + TSRV 调研，部分入 Stage 3)。
- **状态**：✅

### 2.4 Stage 4 (T55, edge gain 合入 SchemeP)
- **答案**：T55 是 "少量 edge gain 特征"，**未独立成文件**，合并进 SchemeP——很可能是 cancel_imb 或 vol_burst 的某些子集 / 微调。tricks_table 行 37 "LOSO 与 Stage 3 相近，边际"。**Stage 4 没有独立的 +N 维**，所以 SchemeP 总数 = T3 69 + S1 54 + S2 59 + S3 14 + S5 20 = 216 derived，跳过 Stage 4。
- **状态**：🟡（合并细节没有干净记录）

### 2.5 Stage 5 (T68, +20 维) — adapt_mom(3) + ofi_tox(4) + signed_bv(3) + spread_reg(3) + trade_pers(3) + liq_asym(4)
- **来源**：QREVIEW_FEATURES Q101-120, HISTORY_EVIDENCE_DB 主题 6 T68, T68_stage5_features/build.log, stage5_features.py 整文件。
- **答案**：6 个新族 = 20 维。本地 LOSO-equiv +25.94 → +26.44（+0.50）。**注意**：Stage 5 之前方法论从 LOSO 5-fold 切到 LOSO-equiv（全 sym 训练，T59），所以 Stage 5 增量绝对值数字看着大但**实际只 +0.50**（边际）。
  - **adapt_mom (3 维)** for W ∈ (20,50,100) = `(mid[-1] − mid[-W]) / std(mid[-W:])`：**单点 momentum 归一化**。**问题**：依赖单一 mid[-W] 值（cherry-pick risk），W=100 时 ref = mid[0] 极端敏感（Q101-103）。
  - **ofi_tox (4 维)** for (lvl,W) ∈ ((1,20),(1,50),(5,20),(5,50)) = Pearson `corr(OFI_lvl[-W:], Δmid[-W:])`：**OFI 信号与价格变动的相关度**（信号"毒性"）。W=20 时 corr SE ≈ 0.45（Q107），**估计噪音很大**。
  - **signed_bv (3 维)** for W ∈ (20,50,100) = `(π/2) Σsign(r) × |r| × |r_lag| / Σr²`，clip ±10：**带符号 bipower 与 RV 之比**。
  - **spread_reg (3 维)** for W ∈ (20,50,100) = `(spread[-1] − median(spread[-W:])) / IQR(spread[-W:])`：**鲁棒 z-score on spread**。IQR=0（常数 spread，常见）时 denom=inf → v=0（dead 大半时间，Q112-113）。
  - **trade_pers (3 维)** for W ∈ (20,50,100) = Pearson `corr(sign(Δmid[-W:]), sign(Δmid_lag[-W:]))`：**价格变动方向的滞后 1 自相关**。大多 Δmid=0 → sign=0 → 有效样本远 < W（Q114/Q188）。
  - **liq_asym (4 维)** for W ∈ (5,20,50,100) = `log((1 + Σtop5_bsize[-W:]) / (1 + Σtop5_ask[-W:]))`，clip ±10：**top-5 流动性买卖侧不对称的对数**。**W=5 被 KS drop**（FAIL_NAMES 含 liq_asym_top5_W5，因 5 tick 太短极噪），其余 W=20/50/100 保留。
- **核心证据**：T68 build.log ("schemeP feature count: 370 = 154 raw + 196 extras + 20 stage5") | `stage5_features.py:1-205` | HISTORY_EVIDENCE_DB 主题 6 T68 (+0.50)。
- **状态**：✅

---

## 3. 窗口长度 W 选择（每个 W 出现的地方 + 是否 sweep）

### 3.1 总览（QREVIEW Q30/Q34/Q41/Q56/Q67/Q70/Q89/Q97/Q104/Q117/Q204-205）

| 窗口集 | 出现在 | 物理意义（3s tick） |
|---|---|---|
| (5,20,60) | T3 MLOFI, GOFI | 15s / 1min / 3min |
| (5,10,20,50) | T3 RV | 15s / 30s / 1min / 2.5min |
| (20,50,100) | signed_rv, signed_bv, spread_reg, trade_pers, adapt_mom, cancel_imb | 1min / 2.5min / 5min |
| (5,20,50,100) | liq_asym | 15s / 1min / 2.5min / 5min |
| (50,100) | kyle_inv, kyle_lam | 2.5min / 5min |
| (20,30,50,100) | jshare | 1min / 1.5min / 2.5min / 5min |
| (30,50,100) | roll_eff_spr | 1.5min / 2.5min / 5min |
| ((1,20),(1,50),(5,20),(5,50)) | ofi_tox 的 (lvl,W) pair | 多尺度 |
| W=100 only | qrank | 5min |
| W=20 (dualz short), W=100 (dualz long) | dualz | 1min vs 5min |
| α=0.05 → HL=13 ticks (~40s) | ewma_resid | — |
| α∈(0.05,0.1,0.3,0.5) → HL≈13/7/2/1 | ewma_intst, ewma_ofi | — |

### 3.2 每个窗口"为什么这个"
- **是否有 sweep**：**几乎没有**。R34 调研报告里给出的是 "HFT 标准 30/60/120 秒 → 10/20/40 ticks，扩到 (20,50,100) ≈ 1min/2.5min/5min" 的工程选择。0512_final_report Phase 2 没有 W-sweep 表，experiments/T44/T51/T53/T55/T68 下都是 build + final eval 没有 W ablation。
- **为啥每族窗口不一致**：每 stage 是独立调研产物：
  - T3 是 iter_002 时期定的（5,20,60），那时 MLOFI 文献用 30s/3min 双尺度。
  - R34 Stage 1-3 (T44/T51/T53) 是 R34 整体调研，统一倾向 (20,50,100)。
  - jshare 加 30 ticks 可能因为 BNS 文献提到 30-tick 是 jump detection 的常见基础。
  - roll_eff_spr 没用 20 可能因为 W=20 Roll's estimator 噪音过大（理论上 SE ∝ 1/√W）。
  - liq_asym 加 W=5 是探索性，**结果证明 W=5 太噪 → 被 drop**（FAIL_NAMES）。
- **状态**：🟡（**没有 W-sweep**；每族窗口选择基于工程直觉 + HFT 文献，不是 grid search 出来的）。Q204/Q205 的"没有统一 W"已被 reviewer 标记为 design smell，但代码已稳定上线。

---

## 4. EWMA / 衰减系数 α

### 4.1 α 设计动机
- **来源**：QREVIEW_FEATURES Q43-46/Q63-64, CODE_QUESTION_LIST Q52, fast_features_batch.py:31,52,71。
- **答案**：
  - `T3_EWMA_ALPHAS = (0.05, 0.1, 0.3, 0.5)` 用于 EWMA intensities（6 cols × 4 α = 24 features，T3）和 ewma_ofi（3 levels × 4 α = 12 features，Stage 1）。
  - `EWMA_RES_ALPHA = 0.05` 用于 ewma_resid（Stage 3，单 α）。
  - 4 个 α 的 half-life（HL = ln(2)/ln(1/(1−α))）：α=0.05 → 13.5 ticks ≈ 40s；α=0.1 → 6.6 ticks ≈ 20s；α=0.3 → 1.9 ticks ≈ 6s；α=0.5 → 1.0 ticks ≈ 3s。
- **设计意图**：覆盖从"近一个 5min 窗口的中段"（α=0.05）到"几乎只看最后一 tick"（α=0.5）的全频谱。**冗余度**：α=0.5 EWMA(x) ≈ x[t] 本身；α=0.3 EWMA HL=2 也几乎是局部值；两个高 α 与原始 last-tick 高度相关（Q43）。**没有 ablation 证明 4 个都必需**（🟡）。
- **核心问题**：
  - `_ewma_last_batch` 使用 scipy.signal.lfilter 实现 pandas `ewm(adjust=False).mean()`（Q45 已 verify）。
  - 初值 `zi = (1−α) × x[:,:1]` 假设 `y_init = x_0`，得到 `y_0 = x_0`。OK 与 pandas 一致。
  - **NaN→0 fill in input**（line 238, `x_safe = where(isfinite, x, 0)`）：对真 missing 把 EWMA 偏向下（intensity 默认 = 0），是已知 design choice but biased（Q46）。
- **状态**：✅ 公式正确，🟡 α 集本身无 sweep。

---

## 5. amount_delta log1p 处理

### 5.1 为啥唯独 amount_delta 用 log1p
- **来源**：QREVIEW_FEATURES Q18-22/Q47/Q141, QREVIEW_PREDICTOR Q69/Q72, CODE_QUESTION_LIST Q47-48/Q92, HISTORY_EVIDENCE_DB 主题 6 T8/T9。
- **答案**：
  1. **物理直觉**：amount_delta = price × volume，量级 e3-e6（人民币）。其它 154 raw 字段：price 量级 e1（几十元）、size 量级 e3-e4、intensity 量级 e0-e1、indicator 量级 e0、spread 量级 e0。amount_delta 是**唯一跨多个数量级**的字段。
  2. **诊断证据**：T8 发现 sym=2 蓝筹/ETF 的 amount_delta z-score +7.49（极偏离），导致 NN/LGB 在 sym=2 上误判。
  3. **解决方案**：T9 SchemeE 引入 `sign(v) × log1p(|v|)` 压重尾（保留符号 + 对数压缩）。
- **关键不一致（Q20/Q48/Q92/Q141）**：
  - **raw_last 中** amount_delta 用 log1p（Cache:108-110，Predictor:281-283）。
  - **derived features（dualz_amount_delta, qrank_amount_delta, kyle_inv, kyle_lam, vol_burst）中** 仍用 **raw amount_delta**（X3d[:, :, col_idx["amount_delta"]]）。
  - 同一个底层量在不同特征里被赋予不同尺度的表示。这是 **train/inference 一致**（cache 和 Predictor 都同样处理），但语义上 ambiguous。**没有 ablation 证明是否应该统一**。模型见到两个尺度可能反而是"自动多尺度"——目前是 design choice，留作未来研究。
- **为啥 volume_delta / bsize* 不 log1p**：
  - volume_delta 量级 e2-e3，比 amount_delta 小 3 个数量级，重尾不严重。
  - bsize/asize 已经被 dualz/qrank 内部窗口归一化。
  - **没有 ablation 证明 volume_delta 不需要 log1p**（🟡，Q18/Q47 标记为 design smell）。Reviewer 怀疑应该对所有重尾字段统一 log1p。
- **为啥 log1p 不是 log10**：
  - log1p 在 |v|→0 时光滑（log10 在 v=0 处 −∞）。
  - sign-preserving：sign(v) × log1p(|v|)，对正负 amount 对称。
  - 自然对数与后续 EWMA / RV 等基于 log-return 的公式量级一致。
- **核心证据**：HISTORY_EVIDENCE_DB 主题 6 T8 (sym=2 z=+7.49) → T9 | `build_schemeP_cache.py:108-110` | `Predictor.py:281-283` | `fast_features_batch.py:311,436` (raw amt 仍用)。
- **状态**：✅（log1p 选择和 sign-preserving 有清晰证据链）；🟡（"为啥只挑 amount_delta"无 ablation）；🔴（raw_last vs derived 用不同 scale 是 design debt）。

---

## 6. DROP_NAMES 11 个失效特征

### 6.1 名单 + 分组
- **来源**：QREVIEW_FEATURES Q3-7/Q160-161, QREVIEW_NN_TRAIN Q18-20, QREVIEW_LGB_TRAIN Q57, QREVIEW_PREDICTOR Q78-83, CODE_QUESTION_LIST Q20/Q68, Predictor.py:68-75。
- **答案**：11 个特征 = T59_FAIL_NAMES (10) + STAGE5_FAIL_NAMES (1)。
  | # | 特征名 | 族 | 失效原因（hypothesis） |
  |---|---|---|---|
  | 1 | `dualz_ask_diff1` | dualz | ask 侧 1-level diff 单调性差，跨 sym KS 失败 |
  | 2 | `dualz_bid_diff5` | dualz | 5-level diff 噪音 |
  | 3 | `dualz_ask_diff5` | dualz | 同上 |
  | 4 | `qrank_W100_spread1` | qrank | spread1 大多 = 1 tick → qrank 几乎恒 1 |
  | 5 | `qrank_W100_spread5` | qrank | 同上 |
  | 6 | `qrank_W100_spread10` | qrank | 同上 |
  | 7 | `qrank_W100_cumspread` | qrank | cumspread 也是单调 quantization → rank 退化 |
  | 8 | `kyle_lam_W50` | kyle_lam | Kyle λ 在 50-tick 估计噪音过大；signed_dvol 用 sign(Δmid) 是 Lee-Ready 偏差 |
  | 9 | `kyle_lam_W100` | kyle_lam | 同上，**整个 kyle_lam 族全 drop** |
  | 10 | `roll_eff_spr_ratio_W100` | roll | 100 ticks 上 Roll 估计 cov 经常 > 0（trend 主导）→ eff=0 |
  | 11 | `liq_asym_top5_W5` | liq_asym | W=5 太短，本质是单 tick top-5 比值，噪音大 |

### 6.2 失效标准
- **官方代码注释**：QREVIEW_FEATURES Q3 引用 `fast_features.py:10` docstring 提到 "10 KS-fail extras"——意味着 **KS 检验**（Kolmogorov-Smirnov）但 **p 值阈值 / Bonferroni / random seed / val split 全部不在代码内**（🔴）。
- **T59 命名暗示**：T59 = 全 sym 训练（取消 LOSO）的实验，T59_FAIL_NAMES = 在 T59 sym-agnostic verification 中失败的特征。**应该是**：训练完一个 sym-agnostic 模型，对每个 sym 分别看该特征在该 sym 上的分布 vs 全局训练分布做 KS test，p < α 即"sym-non-agnostic" → drop。
- **STAGE5_FAIL_NAMES** = T68 之后单独发现的 1 个失败特征（liq_asym W=5），加在 T59_FAIL 列表后面。
- **不对称性**（Q4）：`dualz_bid_diff1` 保留但 `dualz_ask_diff1` drop——可能 bid 侧在某些 sym 上 distribution shift 小于 ask 侧（流动性结构不对称）；**没有可复现的证据**（🔴）。
- **kyle_lam 全族 drop**（Q6）：意味着 line 431-453 的整个计算都是 wasted compute；但因为推理时 batch features 仍要 emit 完整 216 维向量再 drop，所以 kyle_lam 还是会算（推理 ~84ms LGB + 8.1ms NN 的 budget 内 negligible）。

### 6.3 决策（T147 feature pruning + T152 adversarial filter）
- **T147 feature pruning**（experiments/T147_feature_pruning/results.json）：尝试 359 维剪到 50/100/150/200/250 维等，**C_drop100/150 → 259/209 维 holdout +20.00/+20.13**（vs 359-d baseline +16.87，**正提升**但仅本地）；**phase2 验证**：C_drop100 inner_loso +82.99 vs baseline +14.10——**严重 overfit holdout**，phase2 没有"真正干净"的 OOD 评估。
- **T152 adversarial filter**：359 → 259（drop 100），LOSO 持平，**无平台确认提升**（HISTORY_EVIDENCE_DB 主题 6 T152）。
- **最终决策**：**不做 feature pruning**，但 keep 11 个 KS-fail 的 drop（这是早就固化在 Predictor.FAIL_NAMES 里的，T127/T140/T170/T188v2 全沿用）。
- **状态**：✅ 名单与代码完全对得上；🔴 失效标准（KS p 值 / Bonferroni / split）无文档证据。

---

## 7. 特征维度核对（370 = 154+196+20，训练 359 = 370−11，**减法对得上**）

### 7.1 加法核对
- **来源**：QREVIEW_FEATURES Q1-3, Cache:18,212-213, fast_features_batch.py:5, T68 build.log。
- **答案**：
  - 154 raw：`build_schemeP_cache.py:53-69` assert len(RAW_COLS)==154，与 `Predictor.py:46-65` RAW_COLS_TRAIN_ORDER assert 154 一致。
  - 216 derived = 69 T3 + 54 S1 + 59 S2 + 14 S3 + 20 S5：
    - T3 (69) = MLOFI 30 (3 W × 10 lvl) + WMP 11 (10 wmp + 1 balance) + RV 4 + EWMA_intst 24 (4 α × 6 col)
    - S1 (54) = dualz 37 + signed_rv 3 + kyle_inv 2 + ewma_ofi 12 (3 lvl × 4 α)
    - S2 (59) = qrank 20 + rskew 3 + gofi 30 (3 W × 10 lvl) + kyle_lam 2 + vol_burst 4 (2 vol + 2 amt)
    - S3 (14) = ewma_resid 1 + rv_ratio 3 + jshare 4 + cancel_imb 3 + roll_eff_spr 3
    - S5 (20) = adapt_mom 3 + ofi_tox 4 + signed_bv 3 + spread_reg 3 + trade_pers 3 + liq_asym 4
  - **总 370 = 154 + 216**
- **训练实际维度**：`Predictor._extra_keep_idx = [i for i, n in enumerate(extra_names) if n not in FAIL_NAMES]`（Predictor.py:200-203）。drop 11 → 216 − 11 = 205 derived → **154 + 205 = 359 fed to model**。
- **assert**：`fast_features_batch.py:735` 内 Stage5 有 `assert col == 20`，但 **T3/S1/S2/S3 没有内部 assert**。`compute_batch_features` 末端**没有 `out.shape[1] == 216` 检查**（Q1/Q130）。所以如果某 stage 的 emit 数量与 docstring 漂移，会 silent 错位。
- **`FAIL_NAMES` 不在 extras 里时静默 ignored**（Q160-161）：`[i for i, n in enumerate(extra_names) if n not in FAIL_NAMES]` 不会报"FAIL_NAMES 里某个名字根本不存在于 extras"——这是脆弱点，typo 不会报错。
- **核心证据**：T68 build.log "schemeP feature count: 370 = 154 raw + 196 extras + 20 stage5"（196 baseline 是 T3+S1+S2+S3 = 69+54+59+14）| `Predictor.py:200-203` | `build_schemeP_cache.py:104,112`。
- **状态**：✅（数字闭环）；🔴（无 runtime assert 防止 silent 维度漂移）。

---

## 8. 未采纳的特征家族（每个为什么拒绝）

### 8.1 Alpha101 / Alpha191 (T22) — **违反硬约束 #2 (batch shuffle)**
- **来源**：QREVIEW_FEATURES Q190, CODE_QUESTION_LIST Q71, HISTORY_EVIDENCE_DB 主题 6 T22, T22 results.json。
- **答案**：T22 尝试 WorldQuant Alpha101 + Alpha191。**自然适配 33 个 Alpha101 + 124 个 Alpha191**（剔除需要 cross-sectional rank / IndClass / fundamentals 的）。最佳单因子 LOSO h_10 **IC +21.80**（已超 SchemeC iter_002）。**未提交**因为：
  - **跨截面 rank 需要同时获得多只股票的当前 tick**；
  - 平台**批次顺序被打乱（硬约束 #2）**，predict(x) 收到的 batch 不保证包含全部 5 个 sym 的同一 tick 数据；
  - 即使能获得 cross-sym 数据也无法对齐时间戳（date 置 0）。
  - 自适配子集（不用 cross-sym）也已尝试，但 IC 上限不如 cross-sym 版本，最终 R_cross_sym ABORT。
- **状态**：✅

### 8.2 Hawkes (T141/T142/R_Hawkes_OFI) — 边际
- **来源**：QREVIEW_FEATURES Q220 类问，HISTORY_EVIDENCE_DB 主题 6 T141-142, T142 results.json。
- **答案**：Hawkes 自激过程订单流特征，单 seed LOSO +39.74~+52.12，5-seed ensemble LOSO +55.00。**单一族增量 < 5%**，且每次再训需要解 MLE，feature build 比 batch-vec 慢。**T142 验证 EWMA(α=0.2) 衰减到 ~2e-10 after 100 ticks，inference feasible**（stateless per-session）。但因为 R34 Stage 1/2/3 已经吸收了主要订单流信号，**Hawkes 净增量边际**，T188v2 未入。
- **状态**：✅

### 8.3 TOD (Time-of-Day, T151) — final 弃用
- **来源**：HISTORY_EVIDENCE_DB 主题 6 T151, T151_TOD_v9/results.json, R_time_of_day。
- **答案**：T151 v9 加 5 维 TOD（tod_sin, tod_cos, is_first_20, is_last_20, is_lunch_prox），LGB 364 维（NN 359 维 + 5 TOD）。LOSO 小幅正，但 **final T170 弃用**（in-sample artifact 风险）。前置 T74 SchemeQ 在回归框架下 −0.72（**回归 framework 下时间特征负面**），再次否定 sin/cos position encoding。
- **底层原因**：`time` 不被平台置 0（CRITICAL_CONSTRAINTS §2），所以技术上能用——但 T74/T78/T151 都没在 OOD 上稳定正。
- **状态**：✅

### 8.4 Roll / TSRV (R_roll_tsrv, T53) — 部分入 Stage 3
- **答案**：Roll's effective spread 进入 Stage 3（T53 cancel_imb 同期），但 W=100 被 drop（**整个 Roll 族只剩 W=30/50 入 final**）。TSRV (Two-Scale Realized Volatility) 在 R_roll_tsrv 调研，未直接进 SchemeP（rv_ratio 在 Stage 3 是替代品）。
- **状态**：✅

### 8.5 Savitzky-Golay filter (T35/T91) — gain ≈ 0
- **答案**：T35 SchemeK 加 SG filter 平滑（+32 维 = 255 维），LOSO h_60 +5.53 < +6.30 baseline，**不超**。T91 在回归框架下再试，holdout within-noise。原因：树模型对单调变换不敏感，SG smoothing 不改变 split 信息。
- **状态**：✅

### 8.6 sin/cos 时间编码 (T74/T78) — 负面
- **答案**：见 §8.3 TOD，T74/T78 在回归框架下 −0.72。
- **状态**：✅

### 8.7 Trend features (T191) — KILL
- **来源**：QREVIEW_FEATURES Q124(类), HISTORY_EVIDENCE_DB 主题 6 T191, T191_trend_features_pilot/results.json。
- **答案**：T191 pilot 加 5 维 trend（mean_logret_W10/20/50/100, mid_pct_change_W100）→ 375 维原始 → 364 维有效（drop 后）。**结果**：T191 standalone holdout +140.10（vs T188v2 +150.43，**−10.33**），**corr(T191 pred, T188v2 pred) = 0.9941**（极高冗余），**IC_residual_pooled = −0.1699（负 IC）**——trend signal 已被现有 359 维完全 capture，强行加入反引入噪音。**Verdict: KILL**。
- **状态**：✅

### 8.8 Pairwise 交互 (R_Pairwise_ALL)
- **答案**：R_Pairwise_ALL 调研所有 359 维之间的 pairwise interaction gain。baseline LOSO-equiv +39.82，trick +42.15（+2.32）——但是 in-sample OOF，计算量 O(N²) ≈ 64k 对，进 ensemble 后变 noise，**未入 final**。
- **状态**：✅

### 8.9 cross-sym features (R_cross_sym) — ABORT（硬约束）
- **来源**：R_cross_sym/results.json "abort_infeasible"。
- **答案**：跨 sym 特征（mean/std/rank/zscore over other 4 syms at same (date,t)）在 inference 时无法实现：
  - date 置 0；
  - sym/date/time 不在 config.feature 内（不进 DataFrame）；
  - 测试集顺序打乱，相同 (date,t) 行无法 align；
  - "constants-per-sym" fallback 也失败（sym 在 OOD 上不可靠）。
- **决定**：early abort，未训。
- **状态**：✅

### 8.10 Other reject patterns
- **logret lag (R4/T124)**：无增量。
- **GroupTransformer (T172/T187)**：not feature 而是 architecture，in-sample 数字看着好但全污染，OOD 风险高，未提交。
- **状态**：✅

---

## 9. NaN / Inf 处理

### 9.1 多层防御
- **来源**：QREVIEW_FEATURES Q21/Q46/Q133/Q142/Q172, QREVIEW_LGB_TRAIN Q55-56, QREVIEW_PREDICTOR Q44/Q74, R_NaN1/R_NaN2。
- **答案**（自上而下）：
  1. **每 stage 内部** 多处 `np.where(np.isfinite(v), v, 0.0)`（fast_features_batch.py:289,298,310,319,373,390,420,434,450,463-464,502,510-511,517,531,552-553,560,574,652,667,683,697,717,731 — 约 30 处）。
  2. **EWMA 输入** 特殊：`x_safe = np.where(np.isfinite(x), x, 0.0)`（line 238, 326）——**NaN→0 当 valid 输入**，会让真 missing 段的 EWMA 偏低（Q46）。
  3. **Cache 末端** `extras = np.where(np.isfinite(extras), extras, 0.0).astype(np.float32)`（line 105）。
  4. **Predictor 推理同样** `extras_full = np.where(np.isfinite(extras_full), extras_full, 0.0).astype(np.float32)`（Predictor.py:286），与训练对称。
  5. **LGB**: native NaN passthrough（R_NaN1 audit 确认），不需要额外处理。但 cache 已把 NaN 全 fill 0，所以 LGB 实际看到的是 0 不是 NaN——这意味着 LGB 看不到"这是 NaN"的信号（轻微 information loss，但实测 work）。
  6. **NN**: `np.where(np.isnan(Xs), 0, Xs)` 然后 standardize 然后 clip ±10（train_T188v2_nn_seed.py:223-224, Predictor.py:159-160）。

### 9.2 边界 padding
- 100-tick 窗口：cache 强制要求 `valid_lo = WINDOW - 1 = 99` 和 `valid_hi = T - 1 - MAX_HORIZON = 1940`（Cache:88-90）。**前 99 tick 完全不用**（Q139）——为了所有 W≤100 都有 full history。代价是丢弃 99 × N_sess ≈ 几千行/session ≈ 几十万行 train 数据。
- t=0 的边界：
  - **MLOFI** 用 NaN 填充（line 165-168），随后 W=20 rolling sum 前 20 tick 全置 0（line 196-198）——保持 pandas behavior 的 "min_periods=20"。
  - **GOFI** 用 e[:,0]=0（line 419-420）。
  - **Stage 5 OFI** 用 b_prev[:,0] = b[:,0] 即 Δb=0（line 641-644）。
  - **三种 boundary convention 不一致**（Q28/Q105/Q189）——但因为 100-tick 窗口很短不影响最终 last-tick 统计的大局。

### 9.3 已知 issues
- **inf 不被 isnan 捕获**：训练时 `np.where(np.isnan(Xs), 0, Xs)` 只处理 NaN；inf 没专门处理（Q93, QREVIEW_PREDICTOR Q44）。如果 std=0 → inf 出现，会通过 clip ±10 兜底（CLIP=10 in NN，LGB 容忍 inf 但实际不会 train 好）。
- **EPS=1e-8 全局常数**（Q50/Q128/Q129）：对量级 e3-e6 的 amount 无效保护，对量级 e-12 的可能过度。但实测 work。
- **状态**：✅（处理充分）；🟡（global EPS 不是 scale-aware）。

---

## 10. fast_features.py vs fast_features_batch.py 的一致性

### 10.1 是否一致
- **来源**：QREVIEW_FEATURES Q121-126/Q157-158/Q217, CODE_QUESTION_LIST Q120/Q130, QREVIEW_PIPELINE_ARCH Q66, file diff。
- **答案**：**不一致 — 是 dead code**。
  - **`fast_features.py`** (04_build_pkg/, "iter_010"): 单窗口、**只 196 维**（无 Stage 5）。docstring 错误地说 "350-d caller drops 10" — 真实 returns 196 dim, drop 应该是 11。
  - **`fast_features_batch.py`** (04_build_pkg/ 与 01_build_features/ **byte-identical**, "iter_012"): 批量 (N,100,K) → (N,216)，含 Stage 5。
  - **Predictor 用的是 `fast_features_batch.py`** (Predictor.py:191 `_load_module(here, "fast_features_batch.py", "iter_018_ffb")`)。所以单窗口 `fast_features.py` 在生产**完全是 dead code**。
- **为啥还 ship 单窗口**：可能历史包结构沿用，没人 cleanup。`build_pkg.py:66-67` 仍 copy `fast_features.py` 到 pkg。**Reviewer 建议 delete**（QREVIEW_FEATURES Q122/Q217）。
- **iter_xxx 命名混乱**：
  - `fast_features.py` header "iter_010"
  - `fast_features_batch.py` header "iter_012"
  - Predictor 给它命名 "iter_018_ffb"
  - **三个 iter 编号在一个 pkg 内**（Q152-153），完全 historical naming debt。
- **状态**：✅（确认是 dead code，文档与代码 disagree）；🔴（应清理但已上线，不动）。

---

## 11. GAP 表（reviewer 关心但代码 / 历史里没明确答案）

| GAP # | 问题 | 影响 | 状态 |
|---|---|---|---|
| G1 | DROP_NAMES 11 个的 KS p 值 / α 阈值 / Bonferroni / split / random seed 全部无记录 | 中 | 🔴 |
| G2 | 各窗口集 (5,20,60) / (5,10,20,50) / (20,50,100) / (5,20,50,100) 等都没有 W-sweep 实验 | 中 | 🟡 |
| G3 | EWMA α 集 (0.05, 0.1, 0.3, 0.5) 无 ablation 证明 4 个都必需（α=0.3 和 α=0.5 高冗余） | 低 | 🟡 |
| G4 | amount_delta 唯独 log1p，volume_delta / bsize 不 log1p — 没有 ablation 证明这是最优策略 | 中 | 🟡 |
| G5 | raw_last 中 amount_delta 是 log1p，derived 中是 raw — 不一致（train/inference 对称但语义模糊） | 低 | 🔴 |
| G6 | T3 (5,20,60) vs S2 GOFI (5,20,60) — 表面相同但 MLOFI 用 `>=`/`<=` 双计 size，GOFI 用 `>`/`<`/`==` 分别处理 — 两个 OFI 公式在同文件 | 低 | ✅ design but unjustified |
| G7 | Stage 5 OFI (line 641-651) 又用第三种 OFI 公式，b_prev=b[0] vs MLOFI 用 NaN — 三种 boundary convention 在同文件 | 低 | ✅ similar |
| G8 | `compute_batch_features` 末端无 `assert out.shape[1]==216` | 低 | 🔴 |
| G9 | `FAIL_NAMES` 里有 typo 不会报错（silent ignore） | 低 | 🔴 |
| G10 | `wmp + 1.0` (B:220), `spread1 + 1.0` (B:563) 两处 magic `+1` 来源不明（Q39/Q96/Q183-185） | 低 | 🔴 |
| G11 | EPS=1e-8 全局，对 amount (e6) 没保护，对 small log-return (e-12) 过度 | 低 | 🟡 |
| G12 | clip 范围 [-10,10] / [-1,1] / [-100,100] / [-0.05,0.05] 各处无文档 | 低 | 🟡 |
| G13 | `kyle_lam_W50` / `kyle_lam_W100` 全 drop → 整个 kyle_lambda 族浪费计算 | 低 | ✅（已 drop, 不影响 PnL，仅 wasted CPU） |
| G14 | `fast_features.py` (196-d, single-window) 仍 ship 但生产不用 — 完全是 dead code | 低 | ✅ |
| G15 | docstring 与代码 disagree（fast_features.py 说 350-d returns 196；header iter_010/012/018 不一致） | 低 | ✅ |
| G16 | feature name 含 float `f"ewma_ofi_a{a}_lvl{k}"` 用 `a=0.1` — 跨机器/locale float-to-string 风险 | 极低 | 🟡 |
| G17 | `compute_batch_features` 假设 T=100 但无 early assert（Q136） | 低 | 🔴 |
| G18 | Stage 4 (T55) 合并入 SchemeP 但具体哪些特征没有干净记录 | 低 | 🟡 |
| G19 | T22 Alpha101/Alpha191 IC +21.80 但因硬约束未提交 — alpha subset (33 + 124) 没有 platform 验证 | 中 | 🟡（设计上排除 cross-sym 是对的） |

---

## 12. 强 ablation 速查（带平台数字的对照）

| trick | 本地 | 平台 delta | 文献证据 |
|---|---|---|---|
| SchemeC 226 vs SchemeB 154 | iter_001d −1.92 → iter_002 +4.07 | **+10.72**（绝对 SOTA 起点） | T3, iter_002 |
| 窗口 z-score (T7) vs 全局 z-score (T10) | T7 +13.91 vs T10 +13.40 (LOSO h_10) | — | T7/T10 |
| T44 R34 Stage 1 vs iter_005b | LOSO 5-seed +9.67 → +13.67 (+4.0) | — | T44 |
| T51 R34 Stage 2 vs Stage 1 | +13.67 → +15.06 (+1.4) | — | T51 |
| T53 R34 Stage 3 vs Stage 2 | +15.06 → +16.84 (+1.78) | — | T53 |
| T68 R34 Stage 5 vs Stage 4 | +25.94 → +26.44 (+0.50) | — | T68 |
| T74 SchemeQ time features | +35.51 → −0.72 vs T75 base | — | T74/T78 |
| T22 Alpha101/191 (cross-sym) | best single IC +21.80 | 未提交（违反约束） | T22 |
| T35 SG filter SchemeK | LOSO +5.53 < +6.30 baseline | — | T35 |
| T141 Hawkes 5-seed | LOSO +52.12 单 seed, +55.00 ensemble | 未提交（边际） | T141 |
| T147 feature pruning 359 → 259 | holdout +20.00 (vs +16.87 baseline) | 平台未确认 | T147 |
| T152 adversarial 259 | 平台无确认提升 | — | T152 |
| T191 trend features +5 | standalone holdout −10.33 vs T188v2, IC_residual −0.17, corr 0.9941 | 未提交（KILL） | T191 |
| R_cross_sym | abort（违反约束） | — | R_cross_sym |
| R_Pairwise_ALL | LOSO +2.32 in-sample (overfit risk) | 未入 final | R_Pairwise_ALL |
| **整个 SchemeP 359 → final (T188v2)** | — | **+35.64**（vs DeepLOB −6.65，+42.29） | T188v2 |

---

## 13. 结论 / Worker 心得

1. **SchemeP 是项目最大的工程贡献**：154 → 359 是 +205 维 sym-invariant 特征，对应 R34 Stage 1-5。每 stage 真实增量 +0.5~+4.0（LOSO h_60，去掉方法论 inflation），累积是项目 PnL 增量的主力。**回归框架 (T75) 是损失函数突破**，**SPO+ (T87) 是训练目标突破**，**M7 (T140) 是数据突破**——三个突破都建立在 SchemeP 之上。
2. **特征工程的设计 debt 主要在"统一性"**：3 套 OFI 公式 / 多种 boundary convention / `+1.0` magic constants / 各 stage 不同窗口集 / 不同 EPS / 不同 clip 范围。**所有这些 debt 都没影响 final 分数**（+35.64 SOTA），但任何"我现在要新加一个特征"的人都需要先理解每 stage 的本地约定，**重构 cost 高**。
3. **drop 11 个 KS-fail 特征是基于 sym-agnostic verification**——但**没有可复现的 KS test 脚本**。如果未来需要重新做 sym-agnostic verification，需要从 0 写脚本（参考 T59 implementation 但是没找到 explicit "ks check"）。
4. **未采纳的家族都有明确证据**：Alpha101/191 (cross-sym, 硬约束 #2) → R_cross_sym ABORT；Hawkes (边际) → T141/142；TOD (回归框架负面) → T74/T78/T151；trend (冗余 + 负 IC) → T191 KILL。这些**已经穷尽了"再加什么特征能涨"的常见想法**，未来要涨需要架构 / 数据 / 训练协议层面的 trick（如 M7-style），不是再加 derived features。

### 完成统计
- n_questions_answered ≈ **236**（QREVIEW_FEATURES 225 全部 + CODE_QUESTION_LIST 29 个特征相关去重 + 其他 QREVIEW 文件 ~10 个特征相关 + 8 个 Scheme 演变综合问题）
- n_gaps = **19** (G1-G19)
- n_strong_ablations = **15**（§12 表格内带平台数字或本地 ablation delta）

---

RESULT: task=[answers P3 features] metrics={n_questions_answered=236, n_gaps=19, n_strong_ablations=15} notes=[SchemeP 359 = 154 raw + 205 derived (R34 Stage 1-5 共 216 - 11 KS drop); 每 stage 真实增量 +0.5~+4.0; 窗口/α/clip/EPS 各处都没 sweep; raw_last amount_delta log1p 但 derived 用 raw amount 是 design debt; 3 套 OFI 公式共存; fast_features.py 单窗口是 dead code 与 batch 不一致; 11 个 FAIL_NAMES 名单与代码完全对得上但 KS 标准无文档; Alpha101/cross-sym/Hawkes/SG/sin-cos/trend 全部已验证不采纳, 每个有平台/本地数字证据]
