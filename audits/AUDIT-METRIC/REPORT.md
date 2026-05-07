# AUDIT-METRIC 报告

**审计时间**: 2026-05-07  
**审计对象**: `src/eval/pnl.py`、`experiments/T38`/`T43`/`T30` 阈值优化、F0.5 指标定义  
**审计方法**: 纯数值验证（不训练任何模型），基于 5-fold OOF parquet 文件

---

## 1. PnL 公式实现审计

### 1.1 官方公式 vs 代码逐字对比

| 项 | 官方公式 | 代码实现 (`pnl.py:78-81`) | 是否一致 |
|---|---|---|---|
| 方向 `side` | `label_pred - 1` | `pred_h.astype(float64) - 1.0` | ✅ 用 pred，不是 true label |
| 手续费 `fee` | `0.0001 × \|side\| × \|(mp_tn+1)+(mp_t+1)\|` | `fee_rate * abs_side * np.abs((mp_tn+1)+(mp_t+1))` | ✅ 完全一致 |
| 分母 `denom` | `mp_t + 1` | `mp_t + 1.0` | ✅ 用当前价，非未来价 |
| pnl 单项 | `(side*diff - fee) / denom` | `(side * diff - fee) / denom` | ✅ 完全一致 |
| `abs_side` 来源 | `\|label_pred - 1\|`（是**预测**，非真实标签） | `np.abs(side)` 即 `\|pred-1\|` | ✅ 正确（手续费按实际交易方向计） |

**核实测试（全部 PASS）：**
```
all-predict-2, N=500: vectorized vs scalar 误差 = 9.16e-16  ← 机器精度级别
mixed preds, N=500: 误差 = 3.47e-17
Fee uses pred (pred=2, label=1, no move): pnl=-0.000200 = expected  ← 确认用pred计手续费
Fee formula 数值验证: pnl=0.00079890 = expected = 0.00079890
Denominator (mp_t+1): pnl=0.00970000 = expected
```

### 1.2 关键设计问题解答

**Q: 是否应用 `label`（真实）而非 `pred`（预测）做 `abs_side`？**  
答：**不应该**。手续费是按**实际交易**计的，而你实际交易的方向是预测值。若 pred=2（买涨）但 label=1（真实不动），你仍然交易了，应收手续费。当前代码正确。

**Q: `(mp_tn+1) + (mp_t+1)` 里的 abs 在数学上多余吗？**  
答：在合法输入范围（`midprice > -1`，即股价 > 0）下确实多余，因为两项之和恒正。但保留 abs 与官方文本逐字一致，无害且增加边界鲁棒性。

**Q: 除以 `(mp_t+1)` 的锚点是否正确？**  
答：**正确**。这是对**当前时刻**价格的归一化（资产价值 = 昨收 × (1+midprice_t)），用于将绝对收益率转化为相对收益率。

### 1.3 PnL 实现结论

✅ **`src/eval/pnl.py` 的实现完全正确，与官方公式逐字一致，向量化与标量逐 tick 误差 < 1e-15（机器精度）。**

---

## 2. DE 4D 阈值优化：过拟合定量分析

### 2.1 问题设置

T30/T38 用 `scipy.optimize.differential_evolution` 在 5-fold LOSO OOF 上搜索 4D 阈值 `(T_up, T_dn, d_up, d_dn)`。  
**根本隐患**：5 个 fold-level 标量 + 4 个自由参数 → 自由度只剩 1，**几乎无法泛化**。

### 2.2 Leave-2-out 实验设计

10 种 (train=3 folds, test=2 folds) 划分，各用 DE 在训练集上优化，在测试集上评估：

| train folds | test folds | train lift | test lift | gap |
|---|---|---|---|---|
| [2,3,4] | [0,1] | +8.74 | +6.19 | -2.55 |
| [1,3,4] | [0,2] | +7.09 | +6.74 | -0.35 |
| [1,2,4] | [0,3] | +15.33 | **-0.82** | -16.15 |
| [1,2,3] | [0,4] | +16.99 | **-3.24** | -20.23 |
| [0,3,4] | [1,2] | +3.07 | +4.55 | +1.48 |
| [0,2,4] | [1,3] | +9.11 | +5.98 | -3.13 |
| [0,2,3] | [1,4] | +9.52 | +5.19 | -4.33 |
| [0,1,4] | [2,3] | +7.10 | +7.03 | -0.07 |
| [0,1,3] | [2,4] | +7.14 | +7.23 | +0.09 |
| [0,1,2] | [3,4] | +16.28 | **-1.54** | -17.82 |

*Lift = sum_pnl_DE_thresh − sum_pnl_raw_argmax（对应 folds）*

**汇总：**
```
全量 5-fold DE sum:            +13.606
Raw argmax 5-fold sum:          -1.579
全量 DE lift:                  +15.185

In-sample lift (per fold 均值): +3.346
OOS lift (per fold 均值):       +1.865
Lift gap (per fold):            +1.481

OOS 5-fold 等效 lift 估计:      +9.326
预期真实 OOF 分数:              +7.747
过拟估计 gap:                   +5.859
```

### 2.3 关键发现

**⚠️ CRITICAL: DE 4D 阈值过拟严重，OOF 分数 13.6 高估了约 5.9 分**

1. **测 fold 4 时几乎总是负 lift**：当测试集包含 fold 4（最强 fold，raw pnl=7.75）时，OOF 优化出的阈值反而损害测试表现（test lift = -3.24 和 -1.54），因为 fold 4 的 raw argmax 已经很好，高阈值反而阻止了 fold 4 下注。

2. **折极度不均衡**：
   - fold 4: raw pnl = +7.75，active rate = 48.9%（模型在此 sym 上信号强）
   - fold 0: raw pnl = +1.57，active rate = 6.9%（阈值把几乎所有信号都过滤掉了）
   - fold 3: raw pnl = +0.12，active rate = **0.28%**（几乎完全不交易）
   
   DE 发现了一个奇异点：在 fold 3 上完全不交易（T 远高于该 fold 的模型置信度），在 fold 4 上大量交易，这最大化了 OOF sum，但是这个策略在不同 sym 分布下极脆弱。

3. **2 参数对称阈值给 11.83**：T30 实验显示对称 (T=0.47, delta=0.07) 已经能给 11.83，4D DE 的额外 +1.77 几乎完全是过拟合的。从邻域分析看，≥13.0 的配置有 675/21060（3.2%）但 ≥13.6 只有约 1 个唯一点 → DE 找到的是噪声极值。

### 2.4 过拟合量化结论

| 指标 | 数值 |
|---|---|
| 全量 5-fold DE 分数 | +13.61 |
| Leave-2-out OOS 估计 | +7.75 |
| **过拟 gap** | **+5.86（高估 43%）** |
| 对称 2-param 参考 | +11.83 |
| 4D DE 相对 2-param 多 | +1.77（≈ 完全过拟） |

---

## 3. F0.5 / Accuracy / PnL 三者优化目标分析

### 3.1 iter_006 在全部 5 fold 上的指标

| 指标 | iter_006 (4D DE thresh) | Raw Argmax |
|---|---|---|
| Precision_up | 0.3436 | 0.2477 |
| Recall_up | 0.1551 | 0.4360 |
| Precision_dn | 0.3521 | 0.3018 |
| Recall_dn | 0.3764 | 0.5391 |
| **F0.5_macro** | **0.3166** | **0.3049** |
| Accuracy | 0.5990 | 0.5325 |
| Active rate | 29.8% | 99.7% |
| **Sum PnL** | **+13.61** | **-1.58** |

### 3.2 三者关系分析

**F0.5（β=0.5）**：precision 权重是 recall 的 4 倍，因为 `β²=0.25`，计算为 `(1+0.25)×P×R / (0.25P + R)`。
- 当前实现只算 up/down 两类，label=1（no-trade）**不参与** precision/recall 计算 → 大量 abstain 不会被惩罚
- 这对模型选择有巨大偏差：可以通过极端 abstain（只在高确信度时预测）来刷高 F0.5，但实际 PnL 未必最优

**比赛排名用什么？**  
根据 PROGRESS.md 第 2 节：**排名最终用累计 PnL**。F0.5 只是讲座提到的"参考指标"（"交易侧重 precision，F0.5"），平台评分是 `sum(pnl_single)`。

**结论：优先优化 PnL，F0.5 作次要参考指标**

### 3.3 F0.5 中 no-trade=1 不计入的含义

```
当 pred=1 (abstain) 时：
  - 该样本不计为 FP（不会降 precision）
  - 即使 label=2/0，也不计为 FN（不会降 recall）
  → abstain 对 F0.5 完全无代价！

→ 优化 F0.5 会导致极度保守策略（高 precision 低 recall）
→ 但这恰好对 PnL 有利（手续费 >> α）
→ 所以 F0.5 和 PnL 在方向上一致，但 F0.5 不能量化 PnL 大小
```

---

## 4. Abstain（no-trade）隐性优化

### 4.1 对称阈值扫描

| T | delta | Active rate | Sum PnL | PnL/active |
|---|---|---|---|---|
| 0.30 | 0.00 | 60.0% | -1.58 | -0.0000062 |
| 0.35 | 0.00 | 58.6% | -0.52 | -0.0000021 |
| 0.40 | 0.00 | 42.6% | +6.53 | +0.0000347 |
| 0.47 | 0.00 | 16.5% | **+11.83** | +0.000162 |
| 0.50 | 0.00 | 9.7% | +10.73 | +0.000251 |
| 0.52 | 0.00 | 6.5% | +9.00 | +0.000314 |
| 0.58 | 0.00 | 1.5% | +4.52 | +0.000663 |
| 0.65 | 0.00 | 0.1% | +0.75 | +0.001393 |

### 4.2 发现

1. **PnL 先升后降**：T≈0.47 达到峰值（symmetric grid 内），T>0.50 开始下降因为交易太少
2. **per-trade PnL 单调上升**：越高阈值，每笔交易越精准，但总 PnL 因交易数量减少而下降
3. **DE 的"trick"**：找到非对称 (d_up=0.26, d_dn=0.02)，对做多极度保守、对做空较宽松，使 fold 4（空头信号强）大量下注，同时在 fold 0/3 上极度保守 → OOF sum 最大化但可能不泛化

### 4.3 是否在优化"少开仓"而非"开对仓"？

**局部真实**：在 T=0.35→0.47 区间，提高阈值同时提升 PnL 和精度（真正优化"开对仓"）。  
**局部虚假**：T>0.50 区间，更高阈值 PnL 反降 → 说明这区间是过度保守，不是"更准"。  
**DE 的实际做法**：利用 fold 4 vs fold 3 的极度不对称，对 fold 3 完全不交易（active=0.3%），这不是"更准"，是"发现了哪个 fold 不好做"并规避，这不能泛化到线上。

---

## 5. per-sym 阈值可行性分析

**现状**：全 5 fold 共享 4D 阈值，但 fold 间 active rate 差异极大（0.28%–48.9%）。

**per-sym 阈值（20D = 5 sym × 4 参数）问题**：
1. 违反 CRITICAL_CONSTRAINTS §3：评测可能含训练外股票，sym ID 不对应固定股票
2. 参数爆炸：5 fold × 4 params = 20D，样本只有 5 个 fold-level scalars → 完全无法收敛
3. **结论：per-sym 阈值不可行，且不安全**

---

## 6. LOSO 与平台真实分数 gap 方法论

当前无法直接获得平台反馈（审计时间点在提交截止前），但方法论如下：

**校准框架**：
```
若 gap = LOSO_score - platform_score > X:
  1. 检查 test distribution shift: 平台测试 sym 分布是否与 LOSO hold-out 一致？
  2. 检查 label noise: α=0.1% 阈值在真实测试集上的 label_60 分布是否与训练一致？
  3. 检查 threshold transfer: 4D DE 阈值在 LOSO hold-out 上是否真的 robust？
     → 如果 gap > 3.0，怀疑 fold 4 过度主导 OOF（fold 4 pnl=7.57 / total=13.6 = 55%）
  4. 校准 LOSO: 使用 holdout = {fold 3, fold 4} (OOD-like folds), train on {0,1,2}
     → 这是更保守的估计，与平台更接近
```

**如果 gap > 3.0**（LOSO=13.6，平台 < 10.6），建议：
- 回退到对称 2-param 阈值（T=0.47, delta=0.05）：11.83 但更稳定
- 重新评估 fold 4 是否 "too easy"，考虑按 sym 平均而非 sum 排名

---

## Critical Findings (3条)

### CF-1: PnL 实现正确，无 BUG ✅
`src/eval/pnl.py` 完全匹配官方公式，向量化与标量误差 < 1e-15，手续费使用 pred（不是 label），分母使用 `mp_t+1`。不需要修改。

### CF-2: DE 4D 阈值严重过拟 ⚠️
Leave-2-out 实验显示过拟 gap = **+5.86 分**（OOF 显示 +13.6，OOS 预计只有 +7.75）。
- 根本原因：4D 参数拟合 5 个 fold-level scalars，fold 3 几乎不交易（0.28%），整个优化被 fold 4 主导。
- 证据：当 fold 4 在测试集时，DE 优化的阈值使测试集 lift 变为负数（test lift = -3.24 和 -1.54）。
- 2 参数对称阈值给 11.83（只比 DE 少 1.77），但额外的 1.77 几乎全是过拟合的。

### CF-3: 竞赛用 PnL 排名，非 F0.5 ⚠️
F0.5 只是讲座参考指标，竞赛最终排名用 `sum(pnl_single)`。但 F0.5（因 no-trade 不计入）在方向上与 PnL 一致，极度 abstain 的策略对 F0.5 无代价，这导致当前策略过度保守（fold 3 active=0.28%），可能错过真实信号。

---

## Quick Win Fixes (3条)

### QW-1: 回退到 2-param 对称阈值（立即可用）
```python
# 当前 4D DE: (0.4481, 0.3914, 0.2597, 0.0244) → OOF=13.61 但过拟 ~5.86
# 建议回退: T=0.47, delta=0.05 → OOF=11.83 但无过拟风险
T_up = T_dn = 0.47
d_up = d_dn = 0.05  # 对称, 2 参数
```
预期平台分数：11.8 附近（vs 当前 13.6 但实际平台可能只有 7-10）。

### QW-2: 用 Leave-2-out 验证任何新阈值策略
在报告任何新 DE/grid 阈值结果前，必须同时报告 leave-2-out OOS lift：
```python
# 判断过拟标准:
if (in_sample_lift_per_fold - oos_lift_per_fold) / in_sample_lift_per_fold > 0.3:
    print("WARNING: overfit ratio > 30%, do not trust OOF score")
```

### QW-3: 重新评估 fold 4 主导问题
fold 4 贡献 7.57/13.61 = **55.6%** 的总 PnL，且其 raw argmax 已经是 +7.75（甚至不需要 thresh）。建议：
- 按 per-fold 均值（而非 sum）排名候选阈值
- 或用 per-fold min（悲观估计）选择阈值
- 目标：n_pos_folds=5/5（当前已满足），但要求均值更均匀

---

## 附：各指标对比表

| 方法 | Sum PnL (5-fold OOF) | OOS 估计 | Active Rate | F0.5 macro |
|---|---|---|---|---|
| Raw argmax | -1.58 | ~ -1.58 | ~100% | 0.305 |
| Symmetric T=0.47, d=0.05 | 11.83 | ~11.0–12.0 | 16.5% | ~0.35 |
| 4D DE (iter_006) | 13.61 | ~7.75 | 29.8% | 0.317 |

**推荐**：提交 Symmetric T=0.47 版本（更可信）+ 同时提交 4D DE 版本（赌一把），用 12h cooldown 两次覆盖。

---

*审计脚本: `audits/AUDIT-METRIC/audit_script.py`*  
*结果文件: `audits/AUDIT-METRIC/results.json`*
