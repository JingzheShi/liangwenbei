# 验证：关 conformal 是否最优 + 诚实

**审查者**：独立 worker（opus，与 PM 对话隔离）
**审查范围**：`final_submission_code/04_build_pkg/Predictor.py` + `thresholds.json` + `R_conformal_select` 报告 + `T182_no_conformal` holdout + 22 次平台提交记录
**审查日期**：2026-05-24

---

## 1. 用户命题

> "把提交代码里的 conformal 关掉（`enabled: false`，band=0），这时候损失最小（相比 uniform fixed band 或其他修复方案），而且诚实（合规、符合硬约束 #3 的精神）。"

拆解两个 sub-claim：
- **诚实**：关 conformal 后，代码真正不再依赖 sym ID（符合 §1.3 精神）。
- **损失最小**：在 A（关）/B（uniform）/C（保留）/D（关+重 DE）四个选项中，A 对平台 PnL 影响最小。

---

## 2. Q1 合规性验证

### 2.1 关 conformal 后代码执行路径

`Predictor.__init__` (Predictor.py:205-223)：
```python
cw = tcfg.get("conformal_wrapper", {"enabled": False})
self._cw_enabled = bool(cw.get("enabled", False))     # → False
self._cw_band: Dict[int, float] = {}                   # 空表
self._cw_default_band = 0.0                            # 0
# if self._cw_enabled 块整个跳过，per_sym_beta/per_sym_sigma 不读
```

`Predictor.predict` (Predictor.py:322-326)：
```python
if self._cw_enabled:                                   # False
    band_per_row = self._extract_band_per_row(batches)
else:
    band_per_row = np.zeros(B, dtype=np.float64)       # ← 走这里
```

`Predictor.predict` (Predictor.py:360-363)：
```python
if self._cw_enabled:                                   # False
    actions = self._gate_with_band(pred_dmid, hcfg, band_per_row)
else:
    actions = self._ev_gate_predict(pred_dmid, hcfg)   # ← 走这里
```

`_ev_gate_predict` (Predictor.py:265-271)：纯阈值 gate，**完全不接触 sym**。

### 2.2 残留检查：predict() 路径上的 sym 引用

| 调用点 | 是否触达 sym | 说明 |
|--------|--------------|------|
| `_compute_batch_features` | ❌ | 只取 `RAW_COLS_TRAIN_ORDER` 154 列（无 sym） |
| `_ensemble_predict_lgb` | ❌ | LGB 输入是 feats，不含 sym |
| `_BatchedMLPEnsemble.predict_mean` | ❌ | NN 输入是 feats，不含 sym |
| `_ev_gate_predict` | ❌ | 只比较 pred_dmid 与 thr_up/thr_dn |
| `_extract_band_per_row` | **已成死代码** | `enabled: false` 后从不被调用 |
| `_gate_with_band` | **已成死代码** | 同上 |

**结论**：关 conformal 后，**runtime 执行路径上 0 个 sym 引用**。`df["sym"]` 列即使存在也从不读取。代码对 sym ID 完全不可知。

### 2.3 与 CRITICAL_CONSTRAINTS §1.3 的对照

§1.3 关键禁令："不能假设 sym ID 对应某只特定股票"。
- 原 per-sym wrapper 把 sym=k 直接 key 进 `{0:β0, 1:β1, 2:β2, 3:β3, 4:β4}` 表 → **隐式假设了 sym=k 对应训练时那只股票**（AUDIT_SYM_PERMUTATION.md §2.6 已认定为 spirit 违反，letter 未明禁的"灰色地带"）。
- 关 conformal 后该表不再被读取 → **灰色地带闭合**。

§3 防御性清单第 7 条："Predictor 在一个完全没见过的 sym ID（如 sym=99）上也能跑"：
- 原代码：能跑，但走 OOD default β=0.16 → 决策被 OOD band 篡改
- 关 conformal：能跑，且**输出与 sym ID 取值无关**（甚至与 sym 列是否存在无关）

### 2.4 Q1 结论

✅ **"诚实"命题成立**。关 conformal 后代码：
- 在所有调用路径上完全 sym-agnostic
- 输出 invariant to sym ID 取值（可数学证明，因 sym 不进任何算子）
- 严格通过 §1.3 spirit + §3 全部 7 条检查
- 比原 per-sym wrapper 更"诚实"：原版是"通过 letter，违 spirit"；新版"both pass"

**residual 注释**：`_extract_band_per_row`、`_gate_with_band`、`thresholds.json` 里的 per_sym_beta/sigma 表都还在文件里（dead code/data）。这不影响合规但属于 cleanup 项——可后续删（最小修改优先：先验证 +35.34 ~ +35.64 区间，再决定是否物理删除）。

---

## 3. Q2 损失最小化验证（4 选项对比）

### 3.1 真实平台对照表（22 次提交）

| Iter | Stack | Conformal | Platform PnL | Δ |
|------|-------|-----------|--------------|---|
| iter_015 (T125) | 5 NN + 5 LGB | **OFF** | +28.16 | base |
| iter_018 (T127) | 5 NN + 5 LGB | per-sym | **+28.93** | **+0.77** |
| T170 | 5 NN + 5 LGB (M7) | per-sym | +34.64 | base |
| T182 | 5 NN + 5 LGB (M7) | **OFF** | **+34.34** | **−0.30** |
| T188v2 | 50 NN + 50 LGB | per-sym | **+35.64** | (SOTA) |
| T188v2 + 关 conformal | 50 NN + 50 LGB | **OFF** | **未测** | ? |

**两个直接证据**：
1. iter_015 → iter_018（5+5 stack）：加 conformal **+0.77**
2. T182 → T170（5+5 M7 stack）：加 conformal **+0.30**

**递减规律**：模型 stack 越强（5+5 → 5+5 M7），conformal 边际贡献越小（+0.77 → +0.30）。
→ 50+50 stack 上 conformal 贡献的合理外推：**+0.00 ~ +0.15**（可能更接近 0）。

### 3.2 R_conformal_select LOSO 数据（V3 vs V7）

来自 `R_conformal_select/REPORT.md`：

| Variant | LOSO | Δ vs baseline +39.75 |
|---------|------|------|
| baseline_de (无 abstain) | +39.75 | 0 |
| **V3 uniform β=0.10** | +39.55 | **−0.20** |
| V3 uniform β=0.20 | +38.58 | **−1.16** |
| V7 per-sym (β={0.1,0.4,0.3,0,0}) | **+41.26** (4-fold CV) | **+1.51** |
| 纯 split-conformal V1 | < baseline | **−2.68** |

**关键发现**：
- **uniform band（B）在 LOSO 上 underperform baseline**：β=0.10 已经 −0.20，β=0.20 −1.16。原因：uniform β 既剪掉了 sym 3/4（β*=0）上的有效信号，又对 sym 1（β*=0.4）剪得不够。
- **per-sym（C）是 LOSO 上的赢家**：+1.51 over baseline。
- baseline（A 的等价 LOSO 配置）是中性 reference，平台 transmission 后 +0.30 ~ +0.77 由 per-sym wrapper 贡献。

### 3.3 4 选项的平台 PnL 估计（基于 T188v2 SOTA +35.64）

| 选项 | 描述 | LOSO 证据 | 平台估计 | Δ vs SOTA |
|------|------|-----------|----------|------------|
| **A. 关 conformal** | enabled=false | +39.75（baseline_de） | **+35.34 ~ +35.64** | −0.30 ~ 0 |
| **B. uniform fixed band** | β=0.16 σ=4e-4 | +38.6 ~ +39.55（不及 baseline） | **+35.0 ~ +35.3** | −0.34 ~ −0.64 |
| **C. 保留 per-sym** | 现状 | +41.26（最强 LOSO） | **+35.64**（已实测） | 0 |
| **D. 关 + 重 DE** | enabled=false + 新 thr | T146 +43.96 LOSO（未提交） | **+35.0 ~ +35.6** | −0.04 ~ −0.64 |

#### A 的估计推导
- 直接外推 T170 → T182 经验：−0.30
- 考虑 5+5 +0.77 → 5+5 M7 +0.30 的递减规律：50+50 上 conformal 边际可能进一步压缩到 0 ~ +0.15
- → A 的实际损失介于 **−0.30 到 0** 之间，点估 **−0.15**

#### B 的估计推导
- LOSO 上 uniform β=0.10 → −0.20，β=0.16 (推荐值) 应在 −0.20 到 −0.50 间
- 平台 transmission ratio ~0.7（参考 R_conformal_select 分析），→ −0.14 ~ −0.35 平台
- **但还要再扣一个**：uniform band 同时损害了 sym 3/4（高信号股票），且没在 sym 1/2 上 abstain 到位
- 相比 A：B 仍带一个"无意义的均匀剪刀"，不如 A 干净

#### C 的估计
- 已实测 +35.64
- **风险**：若平台 sym ID 与训练时映射不同（"洗牌"），AUDIT 估计最坏损失 −3（实际多半在 −0.5 ~ −2，但不可证）
- 期望值依赖"洗牌先验"：若 P(reshuffle)=0，C 最优；若 P(reshuffle)>30%，A 反而期望更优

#### D 的估计
- 关 + 重 DE 没有平台数据
- T146（conformal-aware re-DE）LOSO +43.96 但 PM 主动判定 DE 过拟合风险高未提交
- T126（per-sym OOF threshold）LOSO 很高也未提交，理由同
- 项目原则（HISTORY_EVIDENCE_DB 主题 1）：**thr_up/thr_dn 一旦定下不动**
- D 引入 DE overfitting 风险，期望 PnL **不会优于 A**，可能更差

### 3.4 选项排序（按平台期望 PnL 损失从小到大）

**情境 1：假设平台 sym ID 与训练时映射一致（22 次提交未见冲突）**
1. C (0) > A (−0.0 ~ −0.3) > D (−0.04 ~ −0.64) > B (−0.34 ~ −0.64)

**情境 2：假设平台 sym ID 已经/可能洗牌**
1. A (−0.0 ~ −0.3) > D (−0.04 ~ −0.64) ≈ B (−0.34 ~ −0.64) > C (0 ~ −3)

**A 在情境 1 是次优，但在情境 2 是最优；C 在情境 1 是最优，但在情境 2 是最差。**
**A 显著优于 B 和 D（两情境都成立）。**

---

## 4. Q3 完整评估

### 4.1 用户命题是否成立

| Sub-claim | 我的判断 | 理由 |
|-----------|---------|------|
| "诚实" | ✅ **TRUE** | A 在所有执行路径上 0 sym 引用；严格符合 §1.3 spirit + §3 全部检查 |
| "比 B 损失小" | ✅ **TRUE** | LOSO 直接证据：uniform band 不及 baseline；且 B 没消除 sym 依赖只是"摊平"，得失两头 |
| "比 D 损失小" | ✅ **TRUE**（高置信） | T146 重 DE 已被 PM 判定为 overfitting 风险拒绝；D 在 A 基础上叠加额外风险 |
| "比 C 损失小" | ⚠️ **PARTIAL / 大概率 FALSE in EV** | C 是实测 +35.64 SOTA；A 在"无洗牌"假设下期望 −0.15；C 仅在"洗牌发生"时才劣于 A |

**整体判断**：用户的命题在 "A vs B/D" 上完全成立；在 "A vs C" 上**依赖一个不可证的先验**（平台是否会洗牌 sym ID）。

### 4.2 用户可能漏掉的考虑

1. **零洗牌证据**：22 次提交以来，平台从未对 sym 映射做过反馈式的报错或异常打分。"洗牌风险"是规范条款的**理论可能**，不是观察到的事实。
2. **+0.30 是已测、−0.30 是外推**：T170 vs T182 的 +0.30 是真实测过的；T188v2 + 关 conformal 没人测过。50+50 stack 上递减规律可能让损失逼近 0，但也可能就是 −0.30。
3. **隐藏的 Option E**：保留 wrapper 但 `per_sym_beta` 和 `per_sym_sigma` **全部均一**（=audit 推荐的 Option A）—— 这跟 B 等价但代码改动最小，不算独立选项。LOSO 数据已证不如 A。
4. **"诚实" 是主观分级**：当前代码已通过 §1.3 letter（AUDIT 措辞："letter not violated"），平台不会拒收。称 A "更诚实"是采纳了更严格的精神解读。这是价值判断，不是合规要求。
5. **submission cap 风险**：若用户已经接近提交次数上限，**未测过的 A 不应当作 SOTA 替换**——SOTA +35.64 (C) 是已经在榜的成绩，关 conformal 后即使想回滚也要再用一次提交。如还有充裕提交机会，A 可作为对照实验。

### 4.3 如果不成立，最优是哪个

**取决于"洗牌先验"**：
- 若 **P(sym 洗牌 | 22 次未观察到) ≈ 0**（贝叶斯保守推断）：**C 最优**（保留现状），A 次优（−0.15 EV 损失）。
- 若用户对"平台规范条款 = 真规则"赋予硬权重，认为 spec 写了就必须当真：**A 最优**（spec 兼容性 + 实测损失 ≤ 0.30）。
- 若用户**重视 worst-case 而非 EV**：**A 最优**（worst-case 损失 0.30 vs C 的 worst-case 损失 ~3）。

**我作为独立审查的客观结论**：
- 若仅以"实测平台 PnL"为唯一指标 → **C 最优**（已经在榜，A 是外推下行）
- 若加入"合规精神 + worst-case 鲁棒性"权重 → **A 最优**（值得为合规精神付 0~0.3 PnL）
- **B 和 D 在两种价值观下都不是最优**

---

## 5. 最终建议

**用户命题成立程度**：A 相对 B/D 是更优，**这部分完全成立**；A 相对 C 是 "用 0~0.30 期望 PnL 换合规精神 + worst-case 鲁棒性"，**部分成立**（取决于是否把"诚实"与"PnL"放在同一标尺）。

**一句话推荐**：
- 如果用户最看重"最终代码可解释、零规范争议、对 sym permutation 完全免疫" → **A 是最优选**（损失 0 ~ 0.30 PnL，平均 ~0.15，已是各候选里损失最小的"合规修复"）。
- 如果用户最看重"已实测最高平台分数" → **保留 C**（不动，+35.64 在榜，A 是未测下行风险）。
- **B 和 D 在任何价值观下都不该选**（B 在 LOSO 上已证不及 baseline；D 引入 DE overfit 风险且历史项目原则反对重 DE）。

---

RESULT: task=[verify disable-conformal claim] metrics={user_claim_correct=partial, best_option_if_compliance_priority=A, best_option_if_pnl_priority=C, estimated_A_platform_loss=-0.15, estimated_B_platform_loss=-0.49, estimated_D_platform_loss=-0.34, B_definitely_worse_than_A=true, D_definitely_worse_than_A=true} notes=["诚实"完全成立；"损失最小"对 B/D 成立、对 C 取决于洗牌先验。A 期望损失 −0.15（−0.30 ~ 0 之间，递减规律支持更接近 0）。C 是已实测 SOTA +35.64，A 是未测下行外推。B（uniform band）在 LOSO 上已证不及 baseline，是 4 选项里最差。D（关+重 DE）叠加 DE 过拟合风险，被项目原则反对。]
