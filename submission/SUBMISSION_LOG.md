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
| 000  | 2026-05-06 | mmpc_demo (DeepLOB 5-head, 142 features) | 无（直接用 examples/mmpc_demo 跑通流程） | **0.001119** | label_10 | _未提交_ | _未提交_ | 0.0469 | 走通流程；模型严重塌缩到 1（pred dist ≈ 全 1） |

## 列含义

- **iter**：从 000 起单调递增的 3 位整数。
- **本地 PnL (best)**：用 `src/eval/pnl.py` 在本地测试 session（默认 `sym=0 date=119 am`）上算的累计 PnL，5 个 horizon 中最高的一个。
- **best horizon**：上一列对应的 horizon（label_5 / label_10 / label_20 / label_40 / label_60）。
- **公榜 PnL**：平台 dashboard 显示的累计收益率。
- **单次收益**：cum_pnl / 活跃预测数（`pred ≠ 1` 的样本数）。
- **F0.5 (best h)**：在 best horizon 上的 F0.5 macro 值（涨跌平均）。这是平台官方监控指标之一。

## 详细记录

### iter_000_mmpc_demo — 走通流程基线

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
