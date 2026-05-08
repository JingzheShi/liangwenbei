# T113 — 基础但颠覆性的 target / loss / training-objective 提案（tree-only 轴）

> **作者**: Worker (opus xhigh, research-only, 不训模型)
> **日期**: 2026-05-08
> **范围**: target / loss / sample-weight / training-schedule，必须 LightGBM / CatBoost 兼容（builtin objective 或可写出 grad+hess 的 custom obj）。**不依赖 NN-only loss。**
> **Baseline**: iter_016 候选 = T87 SPO+ NN + T75/T99 LGB Huber + CB Huber + T95 GRU = +44.72 LOSO；GBDT 部分的"loss winner"是 **LGB Huber α=1e-3 + class-balanced weight + (R1) date-decay**，单模型 +40.79 LOSO，已是 5-seed 最强单 GBDT。
> **目标**: 找出在 GBDT 那一支上还没被压榨的 target/loss/weight lever，给 +0.5 ~ +3 LOSO（含 ensemble 多样性收益）。

---

## Part 1 — 现状诊断（loss/target/weight 维度）

### 1.1 已确认 winner

| 维度 | 当前 SOTA | 验证来源 | LOSO lift vs 上游 |
|---|---|---|---|
| **target** | `y = (mp_th - mp_t) / (mp_t + 1)`（Δmid_norm，纯 regression） | T75 | +9.79 vs 3-class CE |
| **loss** | LGB `objective='huber', alpha=1e-3`（≈0.5σ_y） | T99 | +4.56 vs L2，+6.74 vs MAE |
| **CB analog** | CB `loss_function='Huber:delta=0.001'` | T99 | +1.76 vs CB RMSE，但与 LGB Huber corr 0.95 → marginal in ensemble |
| **sample weight** | class-balanced on 3-class label（flat ≈ 0.55x，up/down ≈ 1.7x） | T75 / iter_015 | 隐式编码"|Δmid|>τ"的粗 magnitude 先验 |
| **date weight (新)** | linear decay × class-balanced，d=(0.6, 2.4) | R1 | **+0.72** confirmed unbiased，与 Huber 完全 stackable |

### 1.2 已被证伪 / 触底的方向（不要重做）

| 方向 | 实验 | 结果 |
|---|---|---|
| **gross |Δmid| sample weight** (linear/sqrt/cap/offset/floor) | T57, T92 | -3 ~ -16 LOSO，全部劣于 class-balanced。**根因**: L2 已经按 residual² 加权，再乘 \|y\| 是 \|y\|² 双重 overweight tail；EV gate 决策不要求 magnitude 精度（只要过 thr 即可） |
| **NN 直接 PnL loss (tanh-soft action)** | T60, T81-B | 梯度在 \|c\|≪τ 和 \|c\|≫τ 都塌缩到 0 → 退化为 L2，没多余信号 |
| **LGB quantile q=0.5 (单 seed)** | T99 | +32.71 vs Huber +39.11 → 单一 quantile 不够 |
| **dual-gate "q30 与 q70 同号才交易"** | T86 | q30/q70 在 93.7% 行 disagree on sign → 几乎不交易；只取 IQR midpoint 才能用，且 midpoint 与 mean prediction corr 0.85 (15% orthogonal) |
| **CB MAE / CB Quantile on GPU** | T99 | GPU kernel 限制；CPU 慢且劣于 Huber |

### 1.3 关键洞察 — 现在还有 4 个未充分压榨的"基础 lever"

经过 T75 → T99 → R1 的递进，**剩余 leverage 集中在 4 个轴**：

1. **"target 的几何形状"还没被扫过 EV gate 的真实形状**——current target = Δmid_norm 是 **PnL 的 numerator**，但 EV gate 真正要决策的是 **"net PnL 是否 > 0"，即 max(0, |Δmid| - fee_eff)**。Huber 缓解了 L2 的 tail bias，但 target 本身仍把 fee 以下的 "noise band" 当成有效信号在拟合。
2. **decision-aligned tree custom objective 还没 5-seed 跑过**——T99 写了 `make_spoplus_obj` 但 5-seed full run 只在 NN (T87) 上做过；**LGB SPO+ 的 5-seed**还在 "future work" 里。SPO+ 的次梯度 ±2 在 tree 上同样应该工作，且与 Huber 5-seed 的 corr 应该是 0.7-0.8 区间——**ensemble 多样性的真正剩余源**。
3. **counterfactual 全反馈 expected-PnL custom obj**（M1 in T105）——**从未在 tree 上跑过**。我们在每行训练数据上能算 PnL(z=+1, c)、PnL(z=-1, c)、PnL(z=0, c)（cost = 2·FEE 全已知），所以 expected-PnL loss `L = -Σ_a softmax(p)[a] · pnl(a, c)` 的 grad/hess **可解析**，**不需要 sampling**——这是 RL 没有 variance 的版本，T60 NN 失败是 tanh smoothing 的特定问题，tree custom obj 完全绕过。
4. **sample weight 的"PnL-margin"形式**还没试过——T57/T92 用 gross |Δmid| 失败；**用 net PnL margin `max(|Δmid| - fee_eff, 0)` 当权重，与 gross magnitude 区别巨大**：gross 给 fee 以下的"无效区"还赋 0.0001 级别权重把模型拉向噪声；clipped margin 直接把"不可能盈利的 ~70% 行"权重压到 0，让模型把 capacity 全用在 trade-able 区。

剩余 6 个候选 lever（Tweedie / asymmetric Huber / log-return target / two-head sign×magnitude / fine-grained 9-bin / curriculum 两阶段）每个都有具体动机但 leverage 较小（< +0.5 LOSO），归到 Part 2 的 4-10 名。

---

## Part 2 — 10 个 trick 提案（按预估 LOSO lift × stackability 排序）

### 提案 1 ⭐⭐⭐ — Counterfactual Expected-PnL Custom Obj for LightGBM

**1 句**: 把 GBDT objective 换成"3-action expected PnL"的 surrogate，**所有反事实 PnL 全已知，grad+hess 解析、PSD、零 variance**。

**公式**（per-sample，pred = scalar logit-ish 输出）:
```
fee_eff_i ≈ 2·FEE      (constant approx; per-sample 也可 vectorize)
pnl(z=+1, c) = c - fee_eff
pnl(z=-1, c) = -c - fee_eff
pnl(z= 0, c) = 0

# softmax over 3 actions, temperature τ:
logits = [-pred/τ, 0, +pred/τ]    # [-1=short, 0=hold, +1=long]
p_a = softmax(logits)             # [p_short, p_hold, p_long]
L = -Σ_a p_a · pnl(a, c)          # negative expected PnL
```
对 `pred` 求梯度（解析）:
```
dL/dpred = (1/τ) · [p_long·(pnl_long - E_pnl) - p_short·(pnl_short - E_pnl)]
d²L/dpred² = (1/τ²) · variance of pnl under p_a   ≥ 0  (PSD by Jensen)
```

**为什么基础**: 它就是 cost-sensitive multi-class CE 在 LGB custom obj 槽里展开。grad/hess 都是 6 行 numpy，推导 5 行，PSD 自动满足。

**为什么颠覆性**:
1. **去掉 target = Δmid_norm 这个 intermediate**——T60/T81-B/T87 都是"先回归 c 再用 EV gate"；本提案让 GBDT 直接最大化 expected PnL。Tree 不需要 tanh smoothing，所以 T60 的塌缩不复现。
2. **每行的反事实 PnL 都已知**（数据集给 mp_t、mp_th，cost 是常数 1bp）→ "expected PnL loss" 在我们设定下**等价于**真实 PnL，**variance = 0**。这是 RL 文献里梦寐以求的 zero-variance setting。
3. 与 Huber 5-seed 的 prediction 在结构上不同：Huber 优化的是"逼近 c 的鲁棒回归量"；本 obj 优化的是"使 EV gate 期望 PnL 最大的 logit"。两个 5-seed ensemble 的 corr 期望 0.6-0.75（参考 T87 NN SPO+ vs T75 LGB L2 = 0.71；但 tree-tree 通常更高，预估 0.75-0.80），**显著低于 LGB Huber vs CB Huber 的 0.95**——纯净的 ensemble diversity 来源。
4. tree-friendly: 不需要 LP，不需要 sampling，不需要 reparameterization。

**实现成本**: 在 `train_lgb.py:139` `make_spoplus_obj` 旁边加一个 `make_expected_pnl_obj(fee_eff, tau)`。τ ∈ {0.3σ_y, 0.5σ_y, 1.0σ_y} = {6e-4, 1e-3, 2e-3} sweep；总训练 5 seed × 3 τ = 15 runs ≈ 15 min on GPU。

**预估 LOSO 改进**: 单模型 +1 ~ +2 vs LGB Huber +40.79（保守 +41.5，乐观 +42.5）；**真正价值在 ensemble**：(T87 + 0.7·CB_Huber + 1.0·LGB_Huber + **0.5·LGB_ExpPnL** + 1.0·T95) 加一支低 corr 的 GBDT，预估 +0.5 ~ +1.5 over iter_016 +44.72。

**风险**:
- τ 太小会把所有梯度塞到 thr 附近的薄带（类 hard-min PnL surface），训练不稳；τ 太大退化成线性 regression-on-PnL。需要 sweep。
- Hessian 是 expected_PnL 的 variance-under-softmax，在 |c| 极大处接近 0（softmax 已饱和），LGB 会自动加 ε，不会崩。
- 决策仍然走 EV gate（`pred > thr_up → long`），thr 要重新 DE-tune（不增加新维度）。

---

### 提案 2 ⭐⭐⭐ — Net-PnL-Clipped Target with Huber

**1 句**: 把 target 从 `c = Δmid_norm` 换成 `y = sign(c)·max(0, |c| - fee_eff)`（**fee 以下置 0**），其他不变（5 seed × LGB Huber α=1e-3 × class-balanced × date-decay）。

**公式**:
```
fee_eff ≈ 2·FEE = 2e-4
y_train = sign(c) · max(0, |c| - fee_eff)        # 70% 行 → y=0；剩下 30% 行保留 net PnL
```

**为什么基础**: 1 行 target 替换。"训练模型预测它要决策的那个量"是 ML 常识；当前 target c 把 EV gate 的 "fee 死区"也喂给模型当成可学信号。

**为什么颠覆性**:
1. **target 与 EV gate 的决策完全对齐**——`pred > 0 → long, pred < 0 → short, pred ≈ 0 → hold`。EV gate 的 thr_up/thr_dn 仍然由 DE 调，但**调出来的最优 thr 应该接近 0**（因为 fee 已经从 target 中扣除）。这种 "gate at 0" rule 比 "gate at 3.7e-4" 简单且对分布 shift 更鲁棒。
2. **70% 死区行 y=0**——Huber loss 对 y=0 的 fit 自然是 "pred → 0"，再叠加 class-balanced 把这些行权重压到 0.55x → **模型 capacity 自动从 noise band 转移到 trade-able 区**。这是 T57/T92 想做但做错了的事（它们用 gross |c| 反而把权重往 tail 拉）。
3. **天然解决 0.70 透传系数的"thr 调参不稳"问题**——iter_013/014/015 的 thr_up ≠ thr_dn 不对称是因为 ensemble pred 有微小 bias；新 target 把 fee 抠掉后，**只剩 sign 决策**，thr 抽穿 zero 即可，对称 fallback `thr_up = thr_dn = 0` 应该接近最优。
4. **与 Huber loss 高度协同**——Huber 在 |y|<α 区间用 L2、|y|≥α 用 L1；y=0 占 70% 后，|y|<α 区间被 70% 死区行主导（α=1e-3 ≫ 0），L2 部分给"压向 0"提供强信号，L1 部分稳定 tail。

**实现成本**: 1 行 target 替换 + 重训 5 seed Huber ≈ 8 min on GPU。可选：**target ≠ inference target**——训练用 clipped target，evaluation 还是用原 c 算 PnL（不需要 inverse-transform）。

**预估 LOSO 改进**:
- 单 LGB 模型: +0.5 ~ +1.5 vs LGB Huber +40.79（**因为 target 与 decision 对齐，每个 seed 的 thr 调参噪声减小**——R1 已观察到 d=(1,1) 等权 +36.42 接近 d=(0.6, 2.4) 的 +36.32 → 训练目标更对齐时 DE thr 更不重要）。
- ensemble: 与 LGB Huber-on-c 的 corr 估计 0.85-0.9（同 family，target 微变），**ensemble lift +0.3 ~ +0.8**。stackable on 提案 1。

**风险**:
- 70% 行 y=0 会让 LGB 倾向于学到"大多数样本 pred=0"的偏置。class-balanced 已经反向加权（flat 类 0.55x），但需要监控 pred 分布。
- 如果 fee_eff 估计偏差（per-sample fee_eff 不严格等于 2·FEE），target 边界会有抖动；ablation: y_train 用 1.5·FEE / 2·FEE / 2.5·FEE 三档 sweep。
- 如果 ensemble 在新 target 上 thr_up + thr_dn 接近 0，DE 退化为找平移量，方差极小——**但这正好是稳健性的来源**。

---

### 提案 3 ⭐⭐⭐ — PnL-Margin Sample Weight (clipped, Stack on Class-Balanced × Date-Decay)

**1 句**: sample weight 从 `class_balanced(y_cls) × date_decay(date)` 改为 `class_balanced × date_decay × max(|c| - fee_eff, fee_eff)`，**只对净 PnL > 0 的可交易行加成**，fee 以下行不再 over-weight tail。

**公式**:
```
margin_i = max(|c_i| - fee_eff, fee_eff)         # clamp at fee_eff so 死区行权重不为 0
margin_i = margin_i / mean(margin)               # mean-normalize
w_i = class_balanced(y_cls_i) · date_decay(date_i) · margin_i
```

**为什么基础**: 加权采样按"sample 对最终 metric 的贡献量"加权，是 supervised learning 教科书做法。当前 metric 是 PnL，每行的 PnL 贡献上限就是 `max(|c| - fee_eff, 0)`。

**为什么颠覆性 vs T57/T92 失败**: T57/T92 用 **gross** `|c|` (linear / sqrt / cap)，**没有 clip 在 fee_eff**。结果是：
- |c| < fee_eff 的"无可能盈利"行（占 70%）仍被赋 |c| 这种小但非零的权重 → L2 损失里它们叠加成 0.7 × Σ|c|² 的 base，主导早期 boost rounds，把模型拉向预测小值。
- |c| ≫ fee_eff 的极端尾部行被 sqrt 放缓但没 cap，仍 over-fit。

**Clipped margin 的根本不同**:
- 死区行权重 ≈ fee_eff（被 mean-normalize 后≈ 0.4），class_balanced 又把它们拉到 0.55x → **死区行 effective weight ≈ 0.22x**，几乎被忽略。
- 边缘行（|c| 刚过 fee_eff）权重接近 1.0x，这是 EV gate 真正要分得清的样本。
- 远 tail 行（|c| ≫ fee_eff）权重大但被 Huber α=1e-3 的 L1 部分截断，不会主导梯度。

**与提案 2 的区别**: 提案 2 改 target，提案 3 改 weight。**两者可叠加**——提案 2 让 target=0 占 70%，提案 3 让 weight=0.22 在那 70%；模型 capacity 集中在 30% trade-able 区且 target 对齐 PnL 形状。

**实现成本**: 5 行 weight 计算 patch + 重训 5 seed ≈ 8 min。

**预估 LOSO 改进**: 单模型 +0.3 ~ +1.0 vs LGB Huber + R1 date-decay；与提案 2 stack +0.5 ~ +1.5。

**风险**:
- 如果 fee_eff 跨 sym 差异大（midprice 不同时 fee_eff 略不同），用全局 mean fee_eff 会有偏差；可改成 per-row fee_eff = `FEE × ((mp_th + mp_t + 2)/(mp_t + 1))`。
- 与 R1 date-decay 直接相乘已经验证 stackable（class_bal × date_decay 已是当前 SOTA）；再乘 margin 是第三层 stack，需要监控权重 dynamic range（mean-normalize 后 std 应 < 2）。

---

### 提案 4 ⭐⭐ — LightGBM SPO+ Custom Obj (5-seed full run)

**1 句**: `train_lgb.py` 已经有 `make_spoplus_obj` 草稿但只跑过单 seed pilot；**5 seed × λ_anchor sweep ∈ {1, 5, 10, 30}**，纯 GBDT 版的 T87。

**公式**: T87 REPORT 已给出
```
ℓ_SPO+(p, c) = max(|2p - c| - fee_eff, 0) - z*(c)·(2p-c) + fee_eff·|z*(c)|
g(p) = 2·1{|2p-c|>fee_eff}·sign(2p-c) - 2·z*(c) + λ·(p-c)   # +L2 anchor
h(p) = λ                                                     # PSD via anchor
```
SPO+ 部分次梯度 ±2，恰好在 EV gate 翻转的边缘提供恒定信号；anchor 项保证 h>0、防止单边发散。

**为什么基础 vs 颠覆性**: SPO+ (Elmachtoub & Grigas 2017) 是 predict-then-optimize 文献的 canonical convex surrogate；T87 验证它在 NN 上 +1.85 LOSO。**没在 tree 上正式跑** = 文献已证实但工程未做。

**与 T99 SPO+ 的区别**: T99 的 SPO+ 只做了 single-seed pilot 验证 grad 形状；本提案是**5-seed 完整 pipeline + λ sweep**。

**实现成本**: `make_spoplus_obj` 已存在；只需 launcher 跑 5 seed × 4 λ = 20 runs ≈ 20 min on GPU。

**预估 LOSO 改进**: 单模型 +0.3 ~ +1.5 vs LGB Huber（不会大幅超过，因为 Huber 已经处理了 tail；但 corr 与 Huber 应 0.7-0.8 区间）；ensemble +0.3 ~ +1.0.

**风险**:
- λ_anchor 选过大退化为 L2；过小时 tree 不稳。T87 选 λ=30 但那是 NN scale，tree 的 anchor 系数需要重新校准（参考 T99 草稿 λ=10）。
- Hessian = const λ 对 LGB 是合法的，但 LGB 内部 leaf score = -Σg / (Σh + reg_λ) 在小 λ 下波动大，建议 λ ≥ 5。
- corr 与 Huber 较高（同样在 EV-aware）；ensemble 边际 lift 不一定显著，需要 ablation。

---

### 提案 5 ⭐⭐ — Asymmetric Huber (long-side / short-side different α)

**1 句**: LGB custom obj 实现 Huber，但 `α_up` (`y > 0` 时用) 与 `α_dn` (`y < 0` 时用) 不同——**显式建模 iter_013/014/015 一直观察到的 "thr_up ≠ thr_dn" 不对称**。

**公式**:
```
res = pred - y
α_i = α_up if res >= 0 else α_dn
g = res                if |res| <= α_i
    α_i · sign(res)    otherwise
h = 1                  if |res| <= α_i
    0 (or ε)           otherwise   # use min h = ε for LGB stability
```

**为什么基础**: Asymmetric Huber 是经典 quantile + Huber 的折中；统计文献几十年。

**为什么颠覆性**:
1. iter_015 DE 调出 thr_up=3.58e-4, thr_dn=2.16e-4 → **下行误差比上行误差更敏感**。当前对称 Huber 平等惩罚两侧，模型梯度天平倾向于"上下都拟合得差不多"。Asymmetric Huber 让 capacity 倾斜 → 上行预测更聚焦、下行预测更稳。
2. T75 / T87 都隐式用 DE 调 threshold 来吸收这个不对称；本提案在 loss 层面直接吸收，把"调 thr 才能补偿"的负担转移到训练，**减小 thr DE-tuning 的过拟合风险**（即 0.70 透传系数的一部分）。
3. 与 T86 quantile (q=0.3, q=0.7) 的关系: quantile 是极端版的 asymmetric Huber（α=∞），但 corr q30↔q70 = -0.22 极不稳；asymmetric Huber 是中间地带，预期 corr 与 LGB Huber ~ 0.93-0.95，**ensemble 边际有限但稳健**。

**实现成本**: `make_asym_huber_obj(alpha_up, alpha_dn)`，10 行 numpy；sweep (α_up, α_dn) ∈ {(2e-3, 5e-4), (1.5e-3, 7e-4), (1e-3, 5e-4), ...} 6-8 组合 × 5 seed ≈ 40 runs ≈ 40 min。

**预估 LOSO 改进**: 单模型 +0.2 ~ +0.7；ensemble 微小（与 LGB Huber 同 family）。

**风险**:
- 不对称参数过大会引入 systematic bias；DE thr 仍然能吸收，但失去了"target 对齐"的正面价值。
- 与提案 2 (clipped target) 部分重叠：clipped target 把不对称从 fee 边界拉到 0，需要二选一或叠加最小版本。

---

### 提案 6 ⭐⭐ — Multi-Target Joint Training: (Δmid_h60, Δmid_h60 - Δmid_h40)

**1 句**: LightGBM `multi_output_regression`（或 5 seed × 2 head 实现）同时拟合 h=60 和 (h60 - h40) 两个 target，**额外给一个 "短/中 horizon 差异" 信号正则化主 head**。

**公式**:
```
target_1 = c60 = (mp_t60 - mp_t) / (mp_t + 1)
target_2 = c60 - c40 = (mp_t60 - mp_t40) / (mp_t + 1)    # tail trajectory
Loss = Huber(p1, c60) + λ_aux · Huber(p2, c60 - c40)     # λ_aux ∈ {0.3, 0.5, 1.0}
```

**为什么基础**: Multi-task learning 标配；用相关辅助 target 稳定主 target。

**为什么颠覆性**:
1. h=60 单 target 的"30 round saturate"（T75 best_iter 70-124）说明 EV gate 有效信号集中在前几十棵树；后续棵树都在拟合噪声。引入 (c60 - c40) 把"轨迹方向"作为辅助信号，强迫 model 区分"先涨后回 c60>0" vs "持续涨 c60>0"——两种 c60 一致但实际 microstructure 完全不同。
2. 与 T47 multi-horizon aux 的区别: T47 是 stacking（先训 h40 模型再用其 pred 当输入），需要 stateful；本提案是 LGB native multi-output（要么用 sklearn multi-output wrapper，要么 2 个独立 head + shared features）。**仍然 sym/state-agnostic**。
3. 数据集允许（已观察 mp_t40 在数据集中），**没有违反任何硬约束**——只在训练时多读一列 mp_t40，inference 时不需要 (单 target)。

**实现成本**: 60 行（auxiliary loss 加在 Huber 主 loss 旁，joint backprop）；5 seed × 3 λ ≈ 30 min。

**预估 LOSO 改进**: 单模型 +0.3 ~ +1.0；ensemble 与主 LGB Huber corr 0.85-0.9，边际 +0.2 ~ +0.5。

**风险**:
- λ_aux 调过大主 target 拟合质量下降；λ=0.3 是安全起点。
- LightGBM native multi-output 较 fragile；可改为 `train` 两个独立 booster 共享 init 参数（更稳）。

---

### 提案 7 ⭐ — Tweedie Regression on Δmid_norm

**1 句**: LGB builtin `objective='tweedie', tweedie_variance_power=1.5`，专为 zero-inflated 连续 target 设计——70% 行 |c|≪fee_eff 接近 0，30% 行 |c|>fee_eff 是真实信号，正是 Tweedie 1.5 的命中区。

**公式**: Tweedie 1.5 family `Var(y) ∝ μ^p` with p=1.5，loss = log-likelihood under Tweedie 分布；LGB 已 builtin（`objective='tweedie'`）。

**为什么基础**: Tweedie regression 在保险/广告 click 等"大量零样本 + 少量大值"场景标配；金融 LOB return 也有完全相同结构。

**为什么颠覆性**:
1. T75 L2 / T99 Huber 把 c 当成对称分布拟合；实际 c 是 **zero-inflated heavy-tailed**：70% 行 |c| < fee_eff 几乎是噪声 + 30% 行有真实方向 + 极少数 tail extreme。Huber 用 α 截断了 tail 但没显式建模 zero-inflation。
2. Tweedie 1.5 的"零附近高密度 + 长尾"恰好匹配 c 的实证分布（建议先做 KS 检验，但视觉上 c 的直方图就是"尖峰瘦腰长尾"）。
3. **对偶性**: target 必须 > 0 才能用纯 Tweedie。需要先 `y_pos = c + offset, offset = -min(c)`（或 `y_pos = max(c, 0)` 然后训正/负各一支），稍麻烦但可行。
4. **更基础的版本**: builtin `objective='gamma'` 也是 zero-inflated friendly，但 Gamma 要求 y > 0 严格——需要 split-target trick。

**实现成本**: builtin 切换 + offset 处理 + 5 seed ≈ 10 min。

**预估 LOSO 改进**: 单模型 -0.5 ~ +1.0（高方差，因为 target 变换 + offset 引入额外训练偏移）；ensemble 与 Huber corr 估计 0.7-0.8，**diversity 来源**。

**风险**:
- Tweedie 强假设 y 分布形状；如果实证不匹配会显著恶化。需要 ablation 比较 LGB Huber vs Tweedie at iso-seed.
- offset 处理对 target normalize 敏感。

---

### 提案 8 ⭐ — Two-Stage Curriculum: pretrain on dates 0-59 → finetune on dates 60-79

**1 句**: 用 LGB `init_model` 接力训练: stage1 train on dates 0-59 全权 200 round 学普世微结构; stage2 用 init_model = stage1 booster, 在 dates 60-79 用低 lr (0.02) 继续 100 round, **强迫晚期 boosting tree 全部专注最近 regime**。

**为什么基础**: Curriculum + transfer learning 标配。

**为什么颠覆性**:
1. R1 date-decay 用 weight 平滑施加 date 偏好；本提案用 init_model 显式两阶段——**stage2 的所有 trees 100% 见到 dates 60-79**。如果 R1 +0.72 来自 0.7-1.0x 权重对比，本提案 stage2 = 100% 权重对比，理论上更激进。
2. T101 #9 提议的版本但还没实施。
3. 与 R1 date-decay 互斥（同一 lever 不同实施）；ablation 对照可定哪一种更优。

**实现成本**: 80 行 patch（拆分 stage1/stage2 + init_model）；5 seed × 2 stage ≈ 15 min。

**预估 LOSO 改进**: 单模型 +0.3 ~ +1.0 vs no-decay；vs R1 d=(0.6,2.4) +0.72 应在同 magnitude 但 robustness 更好（pure recent trees）。

**风险**:
- LGB `init_model` + `keep_training_booster=True` API 跨版本敏感；先 sanity check。
- stage2 100 round 在 16 day 数据上可能过拟合；早停在 dates 76-79 val 上。

---

### 提案 9 ⭐ — Sign × Magnitude Two-Head Factorization

**1 句**: 训两个独立 5-seed booster: head A = binary classifier `1{|c| > fee_eff}`（"会不会 trade-able"），head B = regression `c · 1{|c| > fee_eff}`（trade-able 区的 magnitude）；**inference time pred = P(trade) × E[c | trade]**。

**公式**:
```
y_A = 1 if |c| > fee_eff else 0     # binary
y_B = c if |c| > fee_eff else 0     # regression on filtered subset
pred = P_A(x) · pred_B(x)            # combined posterior
```

**为什么基础**: Mixture-of-experts / gated 模型的最简版本。

**为什么颠覆性**:
1. 当前 single regression 同时学"是否过 fee"和"过多远"两件事；分离后每个 head 都用最优 loss（A 用 logistic CE，B 用 Huber）。
2. **决策对齐**: EV gate 的真问题是"|c|>fee 吗"，head A 直接训这个二分类；head B 仅在过 fee 区训练（不被 70% 死区行污染）。
3. 与 T42 OvA binary ensemble 的区别: T42 训 3 个二分类（up vs rest, down vs rest, flat vs rest），用类标签；本提案用 thr=fee_eff（更细，匹配 EV gate 真实阈值）。

**实现成本**: 2 套独立 5-seed pipeline；总训练 ~20 min。

**预估 LOSO 改进**: 单模型 +0.2 ~ +0.8；与 LGB Huber corr 0.8-0.85，ensemble +0.1 ~ +0.4。

**风险**:
- head B 的训练样本只有 30%，可能 overfit；需要监控 val_corr。
- inference 时两个 booster 都要跑（load + predict 时间 2x）；仍在 platform CPU 预算内（iter_016 候选 1024-batch ~1s, 2x→2s）。

---

### 提案 10 ⭐ — Self-Distillation / Pseudo-Soft Target

**1 句**: stage1 train teacher = LGB Huber 5-seed → predict on train set get `c_hat`; stage2 train student = LGB Huber on `y_soft = (1-λ)·c + λ·c_hat`，**用 teacher 的预测平滑 ground-truth 中的噪声**。

**公式**:
```
teacher: standard LGB Huber on c, get c_hat per train row
student: same architecture, target y = (1-λ)·c + λ·c_hat, λ ∈ {0.1, 0.3, 0.5}
```

**为什么基础**: Self-distillation (Mobahi 2020), label smoothing 在 regression 上的扩展。

**为什么颠覆性**:
1. c 在 |c|<fee_eff 区域几乎是纯噪声（次 tick microstructure 不可预测）；Huber 试图鲁棒拟合但仍然受到噪声影响。teacher prediction `c_hat` 已经把 "可学信号" 部分提取出来，学生从 mix target 学的是 `c_hat`'s pattern + 残差中的真实信息。
2. 文献证据: Mobahi 2020 NeurIPS 显示 self-distillation 等价于隐式正则化；金融 ML 上很少做但理论支持。
3. 与 T65 pseudo-labeling 的区别: T65 在 unlabeled test 上做 pseudo-label（违反"不能泄漏 test"原则），本提案在 train 上做（训练时间正常 supervised）。

**实现成本**: stage1 已存在 LGB Huber（直接用 T99 输出）；stage2 重训 ≈ 5-10 min × 5 seed × 3 λ。

**预估 LOSO 改进**: 单模型 +0.1 ~ +0.5（小但稳）；ensemble 边际更小。

**风险**:
- teacher 的 systematic bias 会传染给 student（self-distillation 的固有风险）。
- λ 太大学生不学真实信号，太小没区别；需要 sweep。

---

## Part 3 — Top 3 推荐（impact × LGB-Huber-stackable × low-risk）

排序原则: 必须能与 iter_016 候选（T87 SPO+ NN + LGB Huber + CB Huber + T95 GRU）在 GBDT 那一支上 stack；优先 **未跑过 + 文献坚实 + 1-day 实施**。

### 🥇 #1: Counterfactual Expected-PnL Custom LGB Obj（提案 1）

**为什么 #1**: 唯一一个**未在 tree 上跑过**且有**完整理论保障**（grad/hess 解析、PSD、零 variance）的 lever。其他提案大多是已知 lever 的微调；本提案是新一轴。预期 corr 与 Huber 0.7-0.8 → ensemble 真正多样性来源（vs 现有 LGB Huber↔CB Huber 0.95 的同质化）。**T87 SPO+ NN 的成功暗示决策对齐对树同样有效**。

**Step-by-step（≤ 1 day）**:

1. **添加 obj 函数到 `experiments/T99_e2e_execution_gbdt/train_lgb.py`**（在 `make_spoplus_obj` 后）:
```python
def make_expected_pnl_obj(fee_eff, tau):
    """Counterfactual full-feedback expected-PnL custom objective.
       LGB pred is a single scalar; we softmax-ify into 3-action distribution."""
    fe = float(fee_eff)
    t = float(tau)
    def obj(preds, train_data):
        c = train_data.get_label().astype(np.float64)
        p = preds.astype(np.float64)
        # 3-action softmax with logits [-p/t, 0, +p/t] for [short, hold, long]
        logits = np.stack([-p/t, np.zeros_like(p), p/t], axis=1)
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        s = ex.sum(axis=1, keepdims=True)
        pa = ex / s            # shape (N, 3): [p_short, p_hold, p_long]
        # per-row PnL per action
        pnl = np.stack([-c - fe, np.zeros_like(c), c - fe], axis=1)  # (N, 3)
        E_pnl = (pa * pnl).sum(axis=1)
        # Loss = -E_pnl
        # dL/dp = -(d/dp) E_pnl
        # d softmax / dp at action a: pa * (delta_{a,a'} - pa') · (dlogit_a/dp)
        # dlogit / dp: short -> -1/t, hold -> 0, long -> +1/t
        coef = np.array([-1.0/t, 0.0, 1.0/t])  # (3,)
        # gradient: sum_a pa * (pnl_a - E_pnl) * coef_a   (negate for descent)
        grad = -((pa * (pnl - E_pnl[:, None])) * coef[None, :]).sum(axis=1)
        # hessian (use action-variance under pa): >= 0 by construction
        hess = (pa * (pnl - E_pnl[:, None])**2).sum(axis=1) / (t**2)
        hess = np.maximum(hess, 1e-8)
        return grad, hess
    return obj
```

2. **添加命令行 arg**: `--objective expected_pnl --tau {6e-4, 1e-3, 2e-3}`，wire 到 `train_lgb.py:329` 的 `custom_obj` 槽。

3. **跑 5 seed × 3 τ = 15 runs**:
```bash
for tau in 6e-4 1e-3 2e-3; do
  for seed in 42 1 7 13 100; do
    python train_lgb.py --seed $seed --objective expected_pnl --tau $tau \
                       --tag exppnl_t${tau}_s${seed} --use-gpu
  done
done
```

4. **EV-gate eval** 复用 `eval_pnl.py`（pred 直接走 EV gate，不需要 inverse-transform）：
```bash
python eval_pnl.py --pattern "model_T113_exppnl_t*_s*.txt" --report ev_t113.json
```

5. **Ensemble integration**: 在 iter_016 候选基础上加 0.5 × T113_ExpPnL stream，DE-tune (w_T87, w_CB_H, w_LGB_H, w_T113, w_T95, thr_up, thr_dn) 7 维。

6. **决策**: ship 当 **5-seed avg DE LOSO ≥ +41.5** 且 **5-way ensemble ≥ +45.2**（+0.5 over iter_016）。

**预期**: 单模型 +1 ~ +2 LOSO vs LGB Huber +40.79；ensemble +0.5 ~ +1.5 vs iter_016 +44.72。

**风险监控**:
- τ=6e-4 时如训练崩溃（Newton step 过大），加 `min_data_in_leaf=200` 或 LGB `learning_rate=0.03`。
- 单 seed pred 分布如严重偏（mean offset > 1e-4），换 `tau=1e-3` 默认。

---

### 🥈 #2: Net-PnL-Clipped Target with Huber（提案 2）

**为什么 #2**: 1 行 target 替换、零额外代码；与提案 1 完全 stackable（提案 1 用新 obj，提案 2 用新 target，可以同时换或单独换）。target ↔ decision 对齐是最基础的 ML 原则之一，竟然还没在本项目试过——是个明显遗漏。

**Step-by-step（≤ 0.5 day）**:

1. **修改 `train_lgb.py:121` 附近的 `y_regr` 计算**:
```python
fee_eff = 2.0 * 1e-4   # match FEE_EFF_APPROX
c = (mp_th - mp_t) / (mp_t + 1.0)
y_clipped = np.sign(c) * np.maximum(np.abs(c) - fee_eff, 0.0)
```

2. **跑 5 seed × LGB Huber α=1e-3 用 y_clipped 当 target**:
```bash
for seed in 42 1 7 13 100; do
  python train_lgb.py --seed $seed --objective huber --alpha 1e-3 \
                     --target-mode clipped --tag clipped_huber_s${seed}
done
```

3. **EV-gate eval**: 由于 target = clipped PnL（不是 c），`pred` 已经"扣过 fee"，**最优 thr 应接近 0**。验证：
```bash
python eval_pnl.py --pattern "model_T113_clipped_huber_s*.txt" --thr-search-range 0.0,2e-4
```
确认 DE-tuned thr_up, thr_dn ∈ [0, 1.5e-4]（如果跑出 thr_up = 3.5e-4 则 target 替换没起到对齐作用）。

4. **Ablation: fee_eff sweep ∈ {1.5e-4, 2e-4, 2.5e-4}**——验证对 fee_eff 估计的 robustness。

5. **Ensemble integration**: 加入 iter_016 候选作为第 5 stream（替代或补充 LGB Huber-on-c）。Target 形状变了 → corr 与 LGB Huber-on-c 估计 0.85-0.92；ensemble +0.2 ~ +0.6。

6. **决策**: ship 当 5-seed DE LOSO ≥ +41.0（vs LGB Huber-on-c +40.79）且 ensemble ≥ +45.0。

**预期**: 单模型 +0.5 ~ +1.5；与 #1 stack 后 ensemble 总 lift +1 ~ +2.5。

**风险**: 70% 行 y=0 可能让 booster 早期偏向 pred=0；监控 5-seed 内 best_iter 是否 < 50（过早收敛）。如发生，把 class_balanced 权重对 flat 类从 0.55x 提升到 0.7x。

---

### 🥉 #3: PnL-Margin Sample Weight + R1 Date-Decay（提案 3）

**为什么 #3**: 最小代码改动（5 行 weight 计算），无新 obj、无新 target；和 R1 date-decay 完全同 framework；和 提案 1 / 提案 2 都正交（提案 1 改 obj，提案 2 改 target，提案 3 改 weight）。3 个一起 stack 是 GBDT 那一支唯一未压榨的"基础三件套"。

**Step-by-step（≤ 0.5 day）**:

1. **修改 `train_lgb.py:241` 附近的 sample weight 计算**:
```python
# old:
# sw_tr = sw_class_bal * date_decay
# new:
fee_eff = 2.0 * 1e-4
c_train = (mp_th_train - mp_t_train) / (mp_t_train + 1.0)
margin = np.maximum(np.abs(c_train) - fee_eff, fee_eff)   # clamp at fee_eff
margin = margin / margin.mean()                            # mean-normalize
sw_tr = sw_class_bal * date_decay * margin
```

2. **重训 5 seed × LGB Huber α=1e-3 with margin weight**：
```bash
for seed in 42 1 7 13 100; do
  python train_lgb.py --seed $seed --objective huber --alpha 1e-3 \
                     --weight-mode class_bal_x_decay_x_margin \
                     --tag margin_huber_s${seed}
done
```

3. **Sanity check 1**: 确认死区行 (|c|<fee_eff) effective weight = class_bal(flat) × date_decay × (fee_eff/mean_margin) ≈ 0.55 × 1.0 × 0.4 ≈ 0.22x；trade-able 行 effective weight ≈ 1.0-1.5x。

4. **Sanity check 2**: train log 中 5-seed best_iter 应 ≈ 100-150（vs LGB Huber α=1e-3 当前 70-130；权重重分布让模型有更多有效 boost rounds）。如 best_iter 反而下降 → 死区行 over-suppressed，把 fee_eff clamp 调到 1.5e-4。

5. **Ensemble integration**: 替换 iter_016 中 LGB Huber 流（同 obj 同 target，仅 weight 改）；预期 corr 与 LGB Huber 0.92-0.95（小变），ensemble 边际 +0.1 ~ +0.4。**真正价值在与 #1 / #2 叠加后**——3 个 GBDT 流 (T113-ExpPnL, T113-Clipped, T113-Margin) 形成新的 GBDT triplet，corr 矩阵预期对角占优 0.85-0.92。

6. **决策**: ship 当 5-seed DE LOSO ≥ +41.0 且与 #1 / #2 ensemble + 现有 iter_016 stack ≥ +45.5。

**预期**: 单模型 +0.3 ~ +1.0；3 个 stack 后 ensemble +1 ~ +2.5 over iter_016 +44.72。

**风险**:
- 与 #2 (clipped target) 叠加时，y=0 行的 weight 也接近 0（margin clamp at fee_eff 但 mean_margin 因有大 tail 行被拉高）→ 70% 死区行实际 weight ≈ 0.1x，可能太激进。建议 #2 + #3 同时启用前先 ablation 单独 #3。

---

## Part 4 — 推荐执行顺序（1 day sprint）

| 时段 | 内容 | 期望产出 |
|---|---|---|
| 0:00-2:00 | 实施 #2 (clipped target) 5-seed run | LGB Huber-on-clipped 5-seed avg LOSO; 验证 thr 是否近 0 |
| 2:00-3:00 | 实施 #3 (margin weight) 5-seed run | LGB Huber-on-c with margin weight 5-seed avg LOSO |
| 3:00-7:00 | 实施 #1 (expected-PnL custom obj) 5 seed × 3 τ run | 选 best τ 的 5-seed avg LOSO |
| 7:00-8:00 | DE-tune 7-way ensemble (T87, CB_H, LGB_H, T113-ExpPnL, T113-Clipped, T113-Margin, T95) | iter_017 候选 |
| 8:00-9:00 | sanity_and_timing.py + Predictor end_to_end_check | iter_017 packaged |

**全部叠加预期 LOSO**: +44.72 → +45.5 ~ +47.0（保守 +0.8，乐观 +2.3）。**对应平台真增益（×0.7 透传）**：+0.6 ~ +1.6。

---

## 附录 A — 不要做的事（明确 rule out）

| 不推荐 | 原因 |
|---|---|
| Gross \|c\| sample weight | T57/T92 已证伪 -3 ~ -16 LOSO |
| Pure quantile q=0.5 (median) | T99 +32.71 < Huber +39.11 |
| LGB MAE / regression_l1 全替换 | T99 +35.04 < Huber，且与 Huber corr 高 |
| dual quantile gate (q30 同号 q70) | T86 验证 sign agreement 仅 6.3% → 不可用 |
| NN-only loss（focal regression / SCA / Gumbel-ST） | user 暗示纯树；NN PnL loss 已在 T60/T81 失败 |
| Pseudo-label on test (date 96-119) | T65 -0.61 + 违反 "不泄漏 test" 原则 |
| Sym-specific weight / per-sym fee_eff | 违反 CRITICAL_CONSTRAINTS §1.3（sym-agnostic） |

---

## 附录 B — 关键代码位置

| 文件 | 修改点 |
|---|---|
| `experiments/T99_e2e_execution_gbdt/train_lgb.py:139` | 提案 1: 添加 `make_expected_pnl_obj` |
| `experiments/T99_e2e_execution_gbdt/train_lgb.py:121` | 提案 2: y_clipped = sign(c) · max(|c|-fee_eff, 0) |
| `experiments/T99_e2e_execution_gbdt/train_lgb.py:241` | 提案 3: sw = class_bal × date_decay × margin |
| `experiments/T99_e2e_execution_gbdt/train_lgb.py:139 + arg` | 提案 4: SPO+ 5-seed full run（`make_spoplus_obj` 已存在） |
| `experiments/T99_e2e_execution_gbdt/train_lgb.py:139` | 提案 5: `make_asym_huber_obj(alpha_up, alpha_dn)` |
| 新建 `experiments/T113_target_loss_basic/` | 所有产物 |

---

```
RESULT: task=target_loss_basic top_3=[counterfactual_expected_pnl_custom_lgb_obj, net_pnl_clipped_target_with_huber, pnl_margin_sample_weight_x_class_bal_x_date_decay] notes=GBDT 一支的剩余 leverage 集中在 4 个未压榨的 lever：counterfactual expected-PnL custom obj (M1 for trees, never run)、net-PnL-clipped target (target↔decision 对齐 1 行替换)、PnL-margin sample weight (修正 T57/T92 gross |c| 失败的核心错误：缺 fee_eff clamp)、LGB SPO+ 5-seed (T99 草稿就位但未跑)；T57/T92 magnitude weight 已证伪不要重做，dual-gate quantile 已证伪不要重做，NN-only loss 不在范围；3 个 top 推荐全部 1-day 实施可同时 stack on iter_016，预期 +0.8 ~ +2.3 LOSO，对应平台 +0.6 ~ +1.6
```
