# AUDIT: sym-ID-permutation risk on `final_submission_code/`

**审查日期**: 2026-05-24
**审查范围**: `final_submission_code/` 全部源码（Predictor.py / fast_features*.py / train_*.py / build_schemeP_cache.py / build_pkg.py / config.json / thresholds.json / run_pipeline.sh / stage5_features.py）
**审查者**: 独立 worker，未读任何之前的 PM/审计文档
**唯一聚焦**: 平台 sym ID 仍为 0..4，但 sym=k 实际对应股票身份可能与训练时 sym=k 不同（"洗牌"场景）

---

## 1. 总结结论

**结论**: 代码在 sym 洗牌假设下 **"部分有问题"**，且问题点**唯一**——只在 conformal abstain band 查表那一处。模型本体（NN + LGB）完全不受 sym 洗牌影响，特征工程也完全不受影响。

**最严重发现**:
`Predictor._extract_band_per_row()` (Predictor.py:291–304) 用输入 DataFrame 中的 `sym` 列直接 key 进 `thresholds.json` 里 calibrated 的 `per_sym_beta * per_sym_sigma` 表。该表中的 β·σ 值是**按训练时 5 只股票身份逐一 calibrate 出来的**。如果平台 sym=k 对应的实际股票不是训练时 sym=k 那只，**代码会把张三的 abstain band 套到李四头上**，导致 effective threshold 错位（最大 1.88e-4，相对 base threshold 3e-4 是 +63%；对 h=5 base threshold 8.66e-5 来说是 +217%）。

**严重度**: 🟡 **中等**（不会 crash、不会归零，但会令 conformal 这一层从"正贡献"变成"零或负贡献"；估计 PnL 影响 -0.3 ~ -3 之间，相对 h=60 ≈ 35.64 PnL 是 ~1%~8% 量级）。

---

## 2. 逐点列举：所有 sym 引用 + 洗牌影响

### 2.1 模型 forward 路径 — `Predictor._compute_batch_features` (Predictor.py:273–289)

```python
def _compute_batch_features(self, batches):
    K = len(self._raw_feat_cols)            # = 154，不含 sym
    X3d = np.empty((N, WINDOW, K), dtype=np.float64)
    for n, df in enumerate(batches):
        X3d[n] = df[self._raw_feat_cols].to_numpy(...)  # 只取 RAW_COLS_TRAIN_ORDER 154 列
```

- `_raw_feat_cols = RAW_COLS_TRAIN_ORDER` (Predictor.py:46–66)，共 154 列：open/high/low/close + volume_delta + amount_delta + bid/ask/bsize/asize 1..10 + 各 intst/ind/acc + midprice/spread/diff 1..10 + 各 mean/rate。**没有 sym**。
- `_col_idx`（Predictor.py:225–227）只 mirror `_raw_col_to_idx`，外加一个 `"midprice" → midprice1` 别名。**没有 sym**。
- `compute_batch_features(X3d, col_idx)`（fast_features_batch.py）和 raw_last 都不用 sym。

**洗牌影响**: ✅ **完全不受影响**。模型 input 张量里根本没有 sym 维度。

### 2.2 模型权重和 normalization — NN/LGB 训练 (train_T188v2_{nn,lgb}_seed.py)

```python
# 两份训练脚本都有：
forbidden = {"date", "sym", "time"}
assert not (forbidden & set(feat_names))  # 训练就会 panic
feat_mean = np.nanmean(X_tr, axis=0)      # 全局 mean，不分 sym
feat_std  = np.nanstd(X_tr, axis=0)       # 全局 std，不分 sym
target_scale = 1.0 / max(y_regr_tr.std(), 1e-8)  # 全局，不分 sym
```

- 训练阶段会 `assert` 一刀，sym 永远不会进入 feature 矩阵。
- NN 没有 `nn.Embedding`、没有 sym-specific 分支；LGB 没有把 sym 当 categorical column。
- normalization 统计量是**全局**的：所有 5 只训练股票数据混算 mean/std，不存在 per-sym normalization。

**洗牌影响**: ✅ **完全不受影响**。即使平台 sym=1 是个完全没见过的新股票，模型 forward 计算流程是一样的；只是泛化精度可能差（这是另一个问题，与 sym 洗牌无关）。

### 2.3 特征工程 — `fast_features.py` / `fast_features_batch.py` / `stage5_features.py`

每份文件的 docstring 都明示"sym never used"、"sym-agnostic"，代码也确实只读 100×K 的 LOB 窗口张量（K 不含 sym）。`compute_batch_features` 的签名是 `(X3d, col_idx)`，col_idx 在 Predictor 里就是 `_raw_col_to_idx`（无 sym）。

**洗牌影响**: ✅ **完全不受影响**。

### 2.4 训练数据 cache 构造 — `build_schemeP_cache.py`

```python
for i, (sym, date, sess) in enumerate(sym_dates):
    path = os.path.join(data_dir, f"snapshot_sym{sym}_date{date}_{sess}.parquet")
    ...
    syms_list.append(np.full(n, sym, dtype=np.int8))
```

sym 仅用于 (a) 文件路径解析、(b) 作为 metadata 写入 npz（"sym" 数组，独立于 "X" 特征矩阵）。训练脚本只读 "X"，不读 "sym" metadata。

**洗牌影响**: ✅ **完全不受影响**（仅训练阶段使用，与推理无关）。

### 2.5 `config.json` 的 `feature` 列表 — line 159 含 "sym"

```json
"feature": [..., "asize_rate10", "sym"]
```

这是告诉平台"DataFrame 里给我这些列"，**不**是模型 input 维度。Predictor 收到 DataFrame 后只取 `self._raw_feat_cols` 那 154 列做特征，"sym" 列只被 `_extract_band_per_row` 单独用。

**洗牌影响**: ✅ **schema 层面无影响**——只是确保 sym 列被传入 Predictor，至于 Predictor 怎么用是另一个问题（见 §2.6）。

### 2.6 🟡 **唯一问题点**: `Predictor._extract_band_per_row` (Predictor.py:291–304) + `thresholds.json` per_sym_beta/sigma 表

```python
def _extract_band_per_row(self, batches):
    out = np.full(B, self._cw_default_band, dtype=np.float64)  # OOD default
    for i, df in enumerate(batches):
        if "sym" not in df.columns: continue
        try: s = int(df["sym"].iloc[-1])
        except: continue
        if s in self._cw_band:
            out[i] = self._cw_band[s]    # <<< 用 sym ID 直接查表
    return out
```

`_cw_band` 在 `__init__` 里由 `thresholds.json` 构造（Predictor.py:218–223）：

```json
"per_sym_beta":  {"0":0.1,    "1":0.4,    "2":0.3,    "3":0.0,    "4":0.0},
"per_sym_sigma": {"0":2.4e-4, "1":4.71e-4,"2":4.52e-4,"3":4.24e-4,"4":4.28e-4}
```

→ band[sym] = β·σ，per ID：

| sym | β   | σ        | band = β·σ | effective_thr_up @ h=60 (base=3e-4)   | effective_thr_up @ h=5 (base=8.66e-5)  |
|-----|-----|----------|------------|---------------------------------------|-----------------------------------------|
| 0   | 0.1 | 2.40e-4  | **2.40e-5**| 3.24e-4 (+8%)                         | 1.106e-4 (+28%)                         |
| 1   | 0.4 | 4.71e-4  | **1.88e-4**| **4.88e-4 (+63%)**                    | **2.75e-4 (+217%)** (最极端)            |
| 2   | 0.3 | 4.52e-4  | **1.36e-4**| 4.36e-4 (+45%)                        | 2.22e-4 (+156%)                         |
| 3   | 0.0 | 4.24e-4  | **0**      | 3.00e-4 (+0%)                         | 8.66e-5 (+0%)                           |
| 4   | 0.0 | 4.28e-4  | **0**      | 3.00e-4 (+0%)                         | 8.66e-5 (+0%)                           |
| OOD | 0.16| 4.00e-4  | 6.40e-5    | 3.64e-4 (+21%)                        | 1.51e-4 (+74%)                          |

**洗牌场景的后果**：

- 平台 sym 范围**仍是 0..4**（不是 5..9），所以 `s in self._cw_band` **永远命中**，OOD 默认值**永远不会被用到**。
- 假设训练时 sym=3 那只股票（真实 β 应为 0）现在被平台标为 sym=1，则 Predictor 会给它 **1.88e-4 的 band**——这相当于把 h=60 阈值从 3e-4 抬到 4.88e-4（+63%），把 h=5 阈值从 8.66e-5 抬到 2.75e-4（+217%）。
- 结果：该股票上 trade 数大幅减少，许多本该 long/short 的信号被 abstain 掉，**漏单**。
- 反向：训练时 sym=1 那只（真实需要 0.4 的保守带）现在被平台标为 sym=3，则 band=0，**所有信号无 abstain 直接 trade**，模型 noise 较大的预测也被强制下单 → **多吃噪声、PnL 受损**。

**严重度判断**: 🟡 **中等**
- (a) **不会 crash**，输出 shape 永远对（5 个 action ∈ {0,1,2}）。
- (b) 决策层失真（thresholds 错位），但失真有上限：单 sym 最多 +63% 或 -0%（h=60），即决策侧的扰动是**有界**的。
- (c) 模型本体的预测 quality 不受影响，所以下游 PnL 不会**完全**被毁，只是 conformal 那一层从正贡献变成可能负贡献。

### 2.7 README.md / `__main__` smoke test

仅文档与自测代码，运行时无影响。

---

## 3. 量化估计（PnL 影响范围）

依据：

- 文档（READ ME 第 233 行）声称 conformal per-sym abstain 在 validation 上贡献 **+0.77 PnL** 单位（占 h=60 PnL ~35.64 的约 **2.2%**）。
- 这 +0.77 单位的来源是"对每只股票施加正确 calibrated 的 band"。如果 band 错配，最坏情况是**反向损失同等量级**。
- 短 horizon（h=5/10/20/40）band 未按 sqrt(H/60) 缩放，因此 band 错配在短 horizon 上**相对扰动倍数**更大。但平台 metric = max(per-horizon PnL)，若 h=60 仍是最高那个，短 horizon 损坏不直接影响最终分数；若短 horizon 的 contribution 也参与 max，则负贡献会进一步扩大。

**估算区间**（仅基于 conformal wrapper 这一层）:

| 场景                                            | 估计 PnL 影响        | 占 h=60 PnL 比 |
|------------------------------------------------|----------------------|------------------|
| **零洗牌**（id 映射恰好正确）                   | +0.77（保持原状）    | +2.2%            |
| **完全随机洗牌**（5! 个置换平均）              | ~ 0 ± 0.5             | 0 ± 1.4%         |
| **最坏对抗性洗牌**（β=0.4 stock ↔ β=0 stock）  | -1.0 ~ -2.5（最坏）  | -3% ~ -7%        |
| **平台真的有训练外股票**（实际股票不属于 5 只之一）| -0.5 ~ -2.0           | -1.4% ~ -5.6%   |

**最坏总情况估计**: 约 **-3 PnL 单位**（相对 h=60 ≈ 35.64 是 ~8.4%）。

注意：
- 这只覆盖 conformal wrapper 这一层；模型本体在面对训练外股票时另有泛化风险，但**那部分**与 sym ID 洗牌无关（不在本审计范围）。
- 估算未做实际回测，是基于 threshold 扰动幅度 × wrapper 原始贡献的粗略上下界。
- 若 platform sym 与 training sym 恰好同分布（即各 sym 上 β·σ 的"最优值"差异很小），实际损失会接近 0。

---

## 4. 建议（按修改成本由小到大排序）

### 选项 A（最小代价，推荐）: 在 thresholds.json 把 per_sym_beta 都设为同一个值

```json
"per_sym_beta":  {"0":0.16, "1":0.16, "2":0.16, "3":0.16, "4":0.16},
"per_sym_sigma": {"0":4.00e-4, "1":4.00e-4, "2":4.00e-4, "3":4.00e-4, "4":4.00e-4}
```

- 等价于在所有 sym 上用 OOD 默认 band (6.40e-5)；
- 移除 sym 洗牌敏感性（band 与 sym ID 解耦）；
- 失去 sym-specific 贡献的 +0.77 PnL，但避免 -3 的最坏损失；
- **改 thresholds.json 一行，不动 Predictor.py，零 build-pkg.py 风险**。

### 选项 B: 彻底关掉 conformal wrapper

```json
"conformal_wrapper": {"enabled": false, ...}
```

- Predictor 走 `_ev_gate_predict` 分支，纯 EV gate；
- 完全消除 sym 相关代码路径；
- 失去 conformal 全部贡献（包括正贡献），但完全免疫 sym 洗牌；
- 0 修改成本（只改 1 个字段）。

### 选项 C（最保守，但需重训）: 重新校准一个 sym-pooled β

- 在 validation 数据上把 5 个 sym 合并 calibrate 出一个全局 β·σ；
- 等价于"承认我不知道哪个 sym 对应哪个股票"；
- 比 A/B 多一步离线 calibration 工作，但保留正向贡献的一部分；
- 仍可能比当前正确映射下的 0.77 略低（信息少了）。

### 不推荐: 修 Predictor.py 加 sym 推断逻辑

试图从 mid-price level / volume 等 LOB 特征推断"这是哪只训练股票"是新增 100+ 行复杂代码，引入新风险且增益不确定。**不推荐**。

---

## 5. 与 CRITICAL_CONSTRAINTS §1.3 的契约对照

| §1.3 禁止条款                              | 当前代码      | 评估        |
|--------------------------------------------|---------------|-------------|
| 把 sym 当 categorical feature 喂给模型     | ✅ 没有       | 通过        |
| `nn.Embedding(num_sym=5, ...)` 索引 sym    | ✅ 没有       | 通过        |
| LightGBM/XGBoost 中把 sym 加进 feature 列  | ✅ 没有（forbidden 校验）| 通过|
| 每个 sym 各自训练的 sym-specific 模型集合  | ✅ 没有       | 通过        |
| per-sym 的 normalization 统计量            | ✅ 没有（global）| 通过    |
| **额外**: 把 sym ID 用做 calibration 查表 key | ⚠️ 有（conformal band） | **契约未明禁，但 spirit 违反** |

CRITICAL_CONSTRAINTS §1.3 的最后一段写得很明确：

> "评测 sym 范围仍然 0-4（不是 5+），平台只是说 sym=0..4 的数据可能不属于训练集那 5 只股票（即同一 ID 可能映射到不同股票）。所以即使 ID 是 0-4，也不能假设它是某只特定股票。"

`_extract_band_per_row` **正好假设了** sym=k 对应某只特定股票。这与契约 spirit 直接冲突。

虽然契约文字里没有显式禁止"把 sym ID 当 lookup key"，但根据契约的 spirit（"不能假设 sym ID 对应某只特定股票"）和 §3 防御性检查清单第 7 条（"Predictor 在一个完全没见过的 sym ID 上也能跑"——当前代码确实"跑得动"，但"跑出错误的 band"），这处代码处于灰色地带：**不会触发平台报错或归零，但会做出基于错误假设的决策**。

---

## 6. 复测建议

提交前应用 §1 §3 防御性检查里的自测脚本，但要扩展一项：

```python
# 当前 Predictor 在 OOD sym=99 上正常输出（用 default band），通过
# 但需要新增一个"sym permutation"测试：
df_orig = build_window(stock_A, sym_id=0)
df_swap = df_orig.copy(); df_swap["sym"] = 1   # 把 stock_A 标成 sym=1
out_orig = p.predict([df_orig])
out_swap = p.predict([df_swap])
# 期望: out_orig != out_swap 但都"合理"（不 crash、shape OK）
# 当前代码: out_swap 会用 sym=1 的更保守 band，决策可能更倾向 1 (flat)
# 如果应用选项 A/B，则: out_orig == out_swap （band 与 sym 解耦）
```

---

## 7. 一句话结论

代码本体（NN + LGB + 154 特征）**完全 sym-agnostic**，唯一 sym ID 依赖在 conformal abstain band 查表（Predictor.py:291–304 + thresholds.json:54–60），最坏情况 PnL 损失 ~3 单位（占 h=60 PnL 的 ~8%）；建议将 thresholds.json 的 per_sym_beta 改为单一全局值（选项 A）或关闭 conformal wrapper（选项 B）以彻底免疫 sym 洗牌。

---

RESULT: task=[sym permutation audit] metrics={n_sym_dependencies=1, n_red=0, n_yellow=1, n_green=6, estimated_worst_pnl_loss=3.0} notes=[只有 conformal abstain band 查表会被 sym 洗牌影响；模型本体和特征完全免疫；最小修复=thresholds.json per_sym_beta 设单一全局值]
