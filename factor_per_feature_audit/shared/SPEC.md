# Per-Feature Audit — 共享规范

> 3 个 worker 并行做同一件事（不同 family 子集），最终输出必须严格遵循本规范，方便 PM 合并。

## 任务目标

为指定 family 下的**每一个** feature 产出一条结构化记录，包括：
1. **identity**：中文名 / 英文全称 / 缩写 / 维度（如果是带 W、k、α 的，列出所有 (W,k,α) 组合）
2. **公式**：数学公式（LaTeX） + 代码定位（文件:行号）+ "如果 R 自己实现一次该怎么写" 的 numpy 伪代码
3. **物理含义**：1-3 行说清楚信号是什么、为什么有方向预测力（不能只是公式翻译）
4. **NaN 处理**：
   - 公式层兜底（np.where(denom==0, fallback, normal) 这类）
   - 训练管线层兜底（window-z 是否启用？nan_to_num 兜底？）
5. **实测 NaN 比例**：在 train/val/test 三个 cache 上分别测，给绝对个数 + 占比 (%)
6. **跨 date 敏感性**：按 date 分组算 mean/std，给 PSI（vs train 全局基线）和 KS 统计
7. **跨 stock 敏感性**：按 sym 分组算 mean/std，给 PSI 和 KS 统计

## 关键路径

- **特征 cache**（X 是 (N, 370) float32）：
  - `/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p/schemeP_train.npz` (1.47M rows, date 0-79)
  - `/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p/schemeP_val.npz` (295K rows, date 80-95)
  - `/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p/schemeP_test.npz` (442K rows, date 96-119)
  - cache 的 keys: `X, mp_t, sym (int8 ∈{0..4}), date (int16 ∈{0..119}), sess_idx, t, y5, y10, y20, y40, y60, mp_t{5,10,20,40,60}`
- **特征名**：`/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p/schemeP_feat_names.txt`（370 行，与 X 列对齐）
- **family 划分**：`/root/projects/liangwenbei_workdir/factor_families_report/family_assignment.json`（dict: `name → "F1"...|"F6"`）
- **特征计算代码**：
  - `/root/projects/liangwenbei_workdir/final_submission_code/01_build_features/fast_features_batch.py`（主路径，216 派生特征）
  - `/root/projects/liangwenbei_workdir/final_submission_code/01_build_features/stage5_features.py`（Stage 5 的 20 维）
  - `/root/projects/liangwenbei_workdir/final_submission_code/01_build_features/build_schemeP_cache.py`（cache 构建）
- **现有报告**（不要复制，只参考）：
  - `/root/projects/liangwenbei_workdir/factor_families_report_v4.pdf`（family 级别总览）
  - `/root/projects/liangwenbei_workdir/ANSWERS_P3_FEATURES.md`（feature 演变史 + KS-drop 细节）
- **KS-drop 11 名单**（这些 feature 在 SchemeP 中仍存在于 X 但训练时被剔除，需要标注）：
  ```python
  KS_FAIL = [
      "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
      "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
      "qrank_W100_cumspread",
      "kyle_lam_W50", "kyle_lam_W100",
      "roll_eff_spr_ratio_W100",
      "liq_asym_top5_W5",
  ]
  ```

## PSI（Population Stability Index）算法

PSI 衡量两个分布的差异，越大越不稳定：
```
PSI(P, Q) = Σ_i (P_i − Q_i) · log(P_i / Q_i)
```
经验阈值：< 0.10 稳定 / 0.10–0.25 中等漂移 / > 0.25 强漂移。

**实现**（必须用这个版本，保证 worker 间一致）：
```python
def psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """PSI on quantile bins of `ref`."""
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 100 or len(cur) < 100:
        return float("nan")
    # 用 ref 的分位数当 bin 边界
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:  # 常数列
        return float("nan")
    qs[0], qs[-1] = -np.inf, np.inf
    p, _ = np.histogram(ref, bins=qs)
    q, _ = np.histogram(cur, bins=qs)
    p = (p + 1e-6) / (p.sum() + 1e-6 * len(p))
    q = (q + 1e-6) / (q.sum() + 1e-6 * len(q))
    return float(((p - q) * np.log(p / q)).sum())
```

## 跨 date / 跨 stock 敏感性的具体定义

- **跨 stock PSI**：以 train 全集分布为 ref（采样 200K 行），逐 sym (0..4) 取该 sym 在 train 上的所有行为 cur，计算 5 个 PSI；输出：`psi_by_sym = [psi_s0, psi_s1, psi_s2, psi_s3, psi_s4]`、`psi_by_sym_max`、`psi_by_sym_mean`
- **跨 date PSI**：把 train (0-79) 当 ref（采样 200K 行），val (80-95) 和 test (96-119) 各自为 cur，得到 `psi_train_vs_val`、`psi_train_vs_test`；额外按 6 个 date bucket（每 20 日一组）算 PSI 序列
- **跨 stock KS**：每只 sym vs 其他 4 只 sym 的 `scipy.stats.ks_2samp` 统计量；输出 `ks_by_sym = [...]`、`ks_max`
- **跨 date KS**：train vs val、train vs test 的 KS 统计

> 抽样规则：所有 PSI / KS 都先按 5% 均匀抽样压缩（节省时间），如果列是有限值少于 1000 个 → 标 `insufficient_data` 跳过。

## NaN 处理审计

每个 feature 必须填三项：
1. **公式层 (in `fast_features_batch.py` / `stage5_features.py`)**：`np.where`、`np.nan_to_num`、`+ε denom`、isfinite mask 等代码片段（贴出关键 3-5 行）
2. **NN 管线层 (`ablation_runs/rerun_13rows.py:271-285`)**：window-z 路径下走 `nanmean / nanstd → nan_to_num(0) → clip(-10, 10)`；raw 路径走 `nan_to_num(0)`
3. **LGB 管线层**：完全不动 NaN，由 LightGBM 内置 missing 路由处理

## 输出文件

每个 worker 必须产出（路径见各自 prompt）：

1. **`per_feature_audit.json`** — 数组，每条对应一个 feature：
   ```json
   {
     "name": "ma_intst",
     "name_cn": "市价卖单强度",
     "name_en_full": "Market-Ask Order-Flow Intensity",
     "family": "F3",
     "raw_or_derived": "raw",
     "dim_in_X": 53,
     "in_ks_drop_11": false,
     "formula_latex": "...",
     "code_loc": "fast_features_batch.py:N/A (raw, 主办方直接给)",
     "numpy_pseudo": "ma_intst = raw_input[:, 53]",
     "physical_meaning": "...",
     "nan_handling": {
       "formula_level": "raw 字段，无公式层兜底",
       "nn_pipeline": "window-z: nanmean/nanstd → nan_to_num(0) → clip(±10)",
       "lgb_pipeline": "不动，LGB 内置 missing routing"
     },
     "nan_pct": {"train": 0.0, "val": 0.0, "test": 0.0},
     "cross_stock_psi": {"by_sym": [...], "max": ..., "mean": ...},
     "cross_stock_ks": {"by_sym": [...], "max": ...},
     "cross_date_psi": {"train_vs_val": ..., "train_vs_test": ..., "by_bucket": [...]},
     "cross_date_ks": {"train_vs_val": ..., "train_vs_test": ...},
     "verdict_cross_stock": "stable | mild_drift | strong_drift",
     "verdict_cross_date": "stable | mild_drift | strong_drift",
     "notes": "可选：补充说明，如 'KS drop 11 之一'"
   }
   ```

2. **`per_feature_audit.md`** — 把上面 JSON 渲染成人类可读的 markdown，按 family 分章节，每个 feature 一节，包含表格 + 公式 + 解读

3. **`{F1F2|F3F4|F5F6}_summary.json`** — 高层汇总：每个 family 的最稳定 / 最不稳定 top-5、NaN 比例最高的 top-5、KS-drop 11 命中情况

4. **`run.log`** — 主要计算步骤的日志（NaN 比例、PSI/KS 计算耗时等）

## 共用工具脚本（推荐先写）

写一个 `audit_utils.py` 放在自己的 workdir 下，包含 `psi`, `ks`, `load_cache`, `feature_stats_by_sym`, `feature_stats_by_date_bucket` 等函数。

## 实验/质量要求

- **必须用 wandb 记录**：`wandb.init(project="liangwenbei-feat-audit", entity="cjxh21-Tsinghua University", name="F1F2-audit" 或 "F3F4-audit" 或 "F5F6-audit")`，log 关键 metric（最高 PSI、平均 NaN 比例、KS-drop 命中数等）。
- WandB API key: `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`
- 计算耗时控制：每个 worker 完整跑下来应 < 30 min。如果某个 PSI/KS 计算很慢，先采样到 50K 行。
- 不要重复读 cache；一次 load 进内存，重复用 view。

## 完成 checklist

- [ ] feature 数量与 family 维度吻合（F1=105, F2=50, F3=45, F4=29, F5=67, F6=74）
- [ ] KS-drop 11 中属于本 worker 的 feature 都标了 `in_ks_drop_11: true`
- [ ] 每个 feature 都有公式 + 代码定位 + 物理含义（不能只填 "raw"）
- [ ] NaN 比例 / PSI / KS 三套数都齐全
- [ ] verdict 字段对每个 feature 都给了判断
- [ ] markdown 报告渲染好，可以直接看
- [ ] wandb run 完成 + git commit 自己的产出文件
