# 提交迭代历史

> 每次准备好一个 iter 都自动追加一行（由 `submission/scripts/prepare.py` 维护）。
> "公榜 PnL" 与"单次收益"等列由人工在收到平台反馈后填回。
>
> 所有 iter 的本地产物在 `submission/iter_<NNN>_<name>/` 目录里。
>
> 注意：**同账号 12 小时只能提交 1 次**——确认上次提交时间再准备下一次。

## 速览表

| iter | 日期 | 模型 | 关键 trick | 本地 PnL (best) | best horizon | 公榜 PnL | 单次收益 | F0.5 (best h) | 备注 |
| ---- | ---- | ---- | ---------- | --------------- | ------------ | -------- | -------- | ------------- | ---- |
| 000  | 2026-05-06 | mmpc_demo (DeepLOB 5-head, 142 features) | 无（直接用 examples/mmpc_demo 跑通流程） | **0.001119** | label_10 | **-6.6492** (label_20, best of 5) | -0.000134~-0.000212 (≈ -fee 全 5 horizon) | 0.302 (label_60) | 平台跑通 ✓；**全 5 horizon 都是负分**，model 是 zero-alpha 乱猜（每笔 ≈ -fee 完美吻合）；label_60 最差 -23.13 因为长 horizon p_flat 更低 → 模型激活率更高 → 乱猜笔数更多 |
| 001  | 2026-05-06 | LightGBM Scheme B (2002 features = 154 last-tick + 1848 rolling stats，label_60 only) | argmax 1/3 + 其他 4 horizon 全 1 | +19.25 (Scheme B IID) / **LOSO -22.10** | label_60 | _未提交（不推荐）_ | _LOSO -5e-5 ≈ -fee_ | 0.353 IID | **brittle**：LOSO 5 fold sum -22.10 完美匹配 mmpc_demo 平台 -23.13；不要单独提，用 iter_001c 替代 |
| 001c | 2026-05-06 | LightGBM Scheme B + threshold (T=0.50, δ=0.15)（threshold 是在 Scheme A LOSO 上调的）| Scheme A's threshold transferred to Scheme B model | LOSO **+8.15** on Scheme B's OOF (5/5 folds positive!) | label_60 | _待提交_ | LOSO +0.000248 | LOSO 0.608 | 5/5 folds positive；最稳；`submission_050602_iter1c.zip` |
| 001d | 2026-05-06 | LightGBM Scheme B + **Scheme B's own tuned threshold** (T=0.45, δ=0.05) | 在 Scheme B LOSO OOF 上重新 sweep 出来的最优 threshold | **LOSO +11.11** (4/5 folds positive) | label_60 | _待提交_ | LOSO +1.34e-04 | LOSO 0.609 | 比 iter_001c 高 +2.95；最差 fold sym=2 -0.98；`submission_050603_iter1d.zip` 1.60MB / 22 sanity ✓ |
| 001f | 2026-05-06 | LightGBM **Scheme D1** (raw 154 + window-zscore 154 = **308 features**) + threshold (T=0.45, δ=0.05) | sym-agnostic z-score normalization 修 sym=2 brittleness（per T8 诊断 F1） | **LOSO +13.91** (**5/5 folds positive!**) | label_60 | _待提交_ | LOSO +1.74e-04 | LOSO 0.615 | **强烈推荐提交**：sym=2 从 -0.98 翻到 **+2.24**！比 iter_001d 高 +2.80；`submission_050605_iter1f.zip` 1.89MB / 22 sanity ✓ |
| 001d | 2026-05-06 | 同 iter_001 模型（Scheme B 训练）+ threshold post-proc (T=0.45, delta=0.05) re-tuned on Scheme B LOSO OOF | 高置信度 gating：max(p0,p2)≥0.45 且 > p1+0.05 才出手；其他 horizon 全 1 | LOSO Scheme B **+11.11** (sum 5 fold, 4/5 正) | label_60 | _待提交_ | LOSO +1.34e-04 | LOSO 0.609 | **推荐替换 iter_001c**；T5a sweep 在 Scheme B 自己的 LOSO OOF 上重调 (T,δ)，比 iter_001c 透传 Scheme A 阈值好 +2.95 PnL；唯一负 fold sym=2 仅 -0.98（vs raw -7.07）；`submission_050603_iter1d.zip` 1.60MB / 22 sanity ✓ |
| 001  | 2026-05-06 | LightGBM Scheme B (last-tick + rolling{mean,std,min,max} W=5/20/60, 2002 dim, label_60 only; 其他 horizon=1 不出手) | 无（raw argmax） | +19.25 (IID) / **-22.10 (LOSO sum)** | label_60 | _未提交_ | _未提交_ | _未提交_ | 已打包 + 通过 sanity_check 19/19 + 契约校验。**T4 LOSO 显示 raw argmax 跨 sym 失败 sum=-22**（mean=-4.42，2/5 fold 正）→ **不建议提交**；改用 iter_001c。 |
| 001c | 2026-05-06 | 同 iter_001 模型 | + 置信度阈值 (T=0.50, delta=0.15) 后处理：max(p0,p2)>T 且 >p1+δ 才出手 | LOSO OOF +6.45 (4/5 fold 正) / IID schemeA test +14.88 vs +9.36 baseline | label_60 | _未提交_ | _未提交_ | _未提交_ | T4 推荐提交。LightGBM model.txt 复用 iter_001 同一权重；sanity_check 19/19 + 契约校验通过。Scheme B IID 显示 T=0.50 偏严，可能 T=0.42-0.46 + δ=0.05 更优 — 待 LOSO B 完成后再 tune。 |
| 001f | 2026-05-06 | LightGBM **Scheme D1** (raw 154 + window-zscore 154 = 308 dim, sym-agnostic adaptive normalization) + threshold (T=0.45, δ=0.05) | window-内 z-score：(x[t] - mean100) / std100 — 完全用 100-tick 窗口内的 mean/std 归一化，无 per-sym/全局统计 | **LOSO +13.91** (5/5 folds positive) / val thresh +16.84 | label_60 | _待提交_ | LOSO +0.000172 | – | **替换 iter_001d 推荐**：T7 sym-agnostic adaptive norm，比 iter_001d (+11.11) 高 +2.80；sym 2 从 −0.98 翻为 +2.24；最弱 fold sym 3 +0.16；`submission_050605_iter1f.zip` 1.89MB / 22 sanity ✓ + contract validator ✓ |

## 列含义

- **iter**：从 000 起单调递增的 3 位整数。
- **本地 PnL (best)**：用 `src/eval/pnl.py` 在本地测试 session（默认 `sym=0 date=119 am`）上算的累计 PnL，5 个 horizon 中最高的一个。
- **best horizon**：上一列对应的 horizon（label_5 / label_10 / label_20 / label_40 / label_60）。
- **公榜 PnL**：平台 dashboard 显示的累计收益率。
- **单次收益**：cum_pnl / 活跃预测数（`pred ≠ 1` 的样本数）。
- **F0.5 (best h)**：在 best horizon 上的 F0.5 macro 值（涨跌平均）。这是平台官方监控指标之一。

## 详细记录

### iter_000_mmpc_demo — 走通流程基线（已提交平台，5 horizon 全负）

#### 平台公榜回填（任务 ID 2340）

| label | accuracy | recall | F0.5 | cum_pnl (=模型评分) | single_pnl (per-trade) | n_active 推算 |
|---|---|---|---|---|---|---|
| label_5  | 0.306 | 0.155 | 0.256 | -9.079  | -0.000200 | ~45,400 |
| label_10 | 0.336 | 0.171 | 0.282 | -13.319 | -0.000199 | ~66,930 |
| label_20 | 0.314 | 0.139 | 0.251 | **-6.649** | -0.000134 | ~49,620 |
| label_40 | 0.339 | 0.159 | 0.276 | -16.065 | -0.000212 | ~75,778 |
| label_60 | 0.333 | 0.220 | 0.302 | -23.132 | -0.000184 | ~125,717 |

**Best score (即排名分)** = max over 5 horizons = **-6.649 at label_20**（最不亏的那个）。

**核心诊断（保存到 history_and_important_notes.md）**：
1. **全 5 horizon 都是负分** — mmpc_demo 在平台上零 alpha
2. **per-trade pnl 跨 horizon 都 ≈ -fee（-0.000134 到 -0.000212）** — 数学上完美吻合"随机预测期望损失 = fee"
3. **label_60 最差不是最好**——长 horizon p_flat 更低（54% vs label_5 的 76%）→ 模型激活率更高（~60% vs ~22%）→ 乱猜笔数翻倍 → 总损失最大
4. **平台 test set 推算 ~210k 评测点**（基于 label_60 ~125k active / ~60% rate）
5. **本地 1-session 评测严重低估了模型 catastrophic 失败的风险**——本地 sym0_date0_am 几乎全预测 1（PnL≈0），但平台全集 test 下分布漂移大到模型乱预测

**iter_000 教训**：
- 不要再提交 mmpc_demo（任何路径都比这差）
- 全预测 1（不交易）即可拿 0 分，已经胜过 mmpc_demo
- 真正的 baseline 必须在我们 train 集上从头训练
- 本地评测必须跑全 240-session test 集，不能只跑 1 session

#### 原 iter_000 描述（保留）

- **日期**：2026-05-06
- **模型**：DeepLOB（多任务 5-head 三分类），142 个特征，100 行窗口，单 head 模型 ≈ 4MB。
- **训练**：见 `examples/mmpc_demo/best_model.pt`（已存在的产物，不是这一轮训的）。
- **目的**：把现成的 demo 走完 zip 打包 + 本地评测 + 历史登记 全流程，**作为后续 iter 的参考样例**。
- **本地评测**：用 `src/eval/pnl.py` 在 `analysis/mmpc_demo_baseline_predictions.parquet`（1842 个评测点，sym=0 date=119 am）上算 PnL。

  | horizon | accuracy | pred_dist (0/1/2) | precision_macro | recall_macro | F0.5_macro | cum_pnl | single_pnl |
  | ------- | -------- | ----------------- | --------------- | ------------ | ---------- | ------- | ---------- |
  | label_5 | 0.864 | 0/1841/1 | 1.0 | 0.0038 | 0.0373 | 0.000560 | 0.000560 |
  | label_10 | 0.790 | 0/1840/2 | 1.0 | 0.0049 | 0.0469 | **0.001119** | 0.000560 |
  | label_20 | 0.950 | 0/1842/0 | NaN | 0.0 | NaN | 0.0 | 0.0 |
  | label_40 | 0.892 | 0/1842/0 | NaN | 0.0 | NaN | 0.0 | 0.0 |
  | label_60 | 0.855 | 0/1842/0 | NaN | 0.0 | NaN | 0.0 | 0.0 |

- **关键发现**：
  - 模型严重欠拟合 / 塌缩到全预测 1（"平"），accuracy 看起来高 (85–95%) 仅因为真实分布也以 1 为主（label=1 占 86–95%）。
  - PnL 几乎全 0，因为没什么活跃预测（`pred=1` → 不交易）。
  - 这是 **baseline floor**——任何后续 iter 的 PnL 应该高于这个。
- **提交状态**：⛔ 未提交（流程演练，不浪费 12h 提交配额）。
| 002  | 2026-05-06 | lgbm-schemeC-multihorizon | _(待填)_ | _(待跑 PnL)_ | _(待填)_ | _未提交_ | _未提交_ | _(待填)_ | zip=15.19MB, 10 files; pred_dist=label_5=[133,1610,99], label_10=[173,1537,132], label_20=[26,1771,45], label_40=[8,1793,41], label_60=[3,1830,9]; note=Scheme C1 (T2 raw 154 + T3 69 = 223-d, no time-encoding) multi-horizon LightGBM. All 5 horizons thresholded. LOSO best sums: h5=+17.5, h10=+21.9, h20=+19.7, h40=+11.7, h60=+6.3 (vs iter_001c +6.45) |
