# ⚠️ 评测协议硬性约束（必读，违反 = 提交直接报错或拿 0 分）

> 来源：用户从平台官网原文复制（"评测指南 → 测试方法"，2026-05-06）。
> 本文是 workdir 的**强制契约**——任何模型/特征/Predictor 设计必须严格遵守。
> 任何 worker 派发的 prompt 都必须复制本文 §1 那 3 条进 prompt 顶部。

---

## §1 三条必须遵守的硬性约束

### 1️⃣ `date` 字段在评测时被置 0（无意义）

> "date：日期会置为 0，无意义"

**禁止**：把 date 当 feature、用 date 推理时段、用 date 做 embedding。
**允许**：完全不用 date。
**影响**：周一/周五效应、季末效应、首日/末日等 date-derived 特征**全部不能用**。

### 2️⃣ 评测程序会**打乱测试点输入顺序**，不同次输入数据无先后关系

> "评测程序会打乱测试点输入顺序，不同次输入的数据无先后关系"

**禁止**：
- 在 Predictor 里维护跨 batch 的 hidden state / cache buffer
- LSTM/Transformer 的 hidden state 跨 predict 调用累计
- 在 `__init__` 里建 `sym → buffer` 之类的运行时映射
- 任何"上一个 t 是 t-1，所以可以增量计算"假设

**允许**：每次 `predict(x)` 调用是**完全独立**的，只用当前 100 行窗口做预测。

### 3️⃣ `sym` 范围 0-4，但**可能含训练外股票**

> "sym：可能的范围 0-4（可能有不来自于 5 只训练集股票的数据）"

**禁止**：
- 把 sym 当 categorical feature 喂给模型
- `nn.Embedding(num_sym=5, ...)` 用 sym ID 索引（训练外 sym 会 IndexError）
- LightGBM/XGBoost 中把 sym 加进 feature 列
- 用每个 sym 各自训练的 sym-specific 模型集合（训练外 sym 没 model 可用）
- per-sym 的 normalization 统计量，因为评测 sym 可能是新的

**允许**（推荐）：
- 模型完全 sym-agnostic：只看 100×D 的 LOB 切片本身
- 全局统计量做 normalization（不分 sym）
- 如果一定要"sym embedding"，用 hash trick + 训练时随机 mask 一定比例 sym 强迫泛化

**特别提示**：评测 sym 范围**仍然 0-4**（不是 5+），平台只是说 sym=0..4 的数据可能不属于训练集那 5 只股票（即同一 ID 可能映射到不同股票）。所以即使 ID 是 0-4，也**不能**假设它是某只特定股票。

---

## §2 可以用的字段

- `time`：保留实际时间戳（"HH:MM:SS"），可用于推理时段（开盘/收盘附近 dynamics 不同）
- 所有 154 维特征（量价、LOB 10 档、订单流、衍生）
- 100-tick 滑窗内的所有时序信息

注意：**`time` 不在 config.feature 列表里默认就不会被送进 DataFrame**。如果想用 time 特征，要么把 time 编码成"分钟数"等加进 feature 列，要么完全放弃 time-based 特征。**稳妥起见，不要依赖 time 在 DataFrame 里**——在 T3 特征工程的 build_cache 阶段把 time-derived 列预先计算并加到 feature 列表，从那里走。

---

## §3 防御性检查清单（每次提交前过一遍）

```
[ ] 模型 forward 不接收 sym 作为输入
[ ] 模型 forward 不接收 date 作为输入
[ ] Predictor.predict 不依赖 self 上的任何"上次调用"状态
[ ] Predictor.predict 不读取 batch 里 row 顺序的物理意义（可被 shuffle）
[ ] Normalization 统计量是全局的（不分 sym）
[ ] requirements.txt 不含会从网络下载东西的库（评测无网）
[ ] Predictor 在一个完全没见过的 sym ID（如 sym=99）上也能跑（防御性测试）
```

**自测脚本建议**（提交前必跑）：

```python
# 把 zip 解压到 tempdir，模拟平台行为：
# 1. shuffle 测试点
# 2. date 全置 0
# 3. 故意把 1/4 测试点的 sym 改为训练外（如 sym=99）
# 4. 单跑 Predictor.predict，确认不报错且输出形状正确
```

如果第 4 步报错（IndexError / KeyError / OOR），说明模型还在用 sym ID。修。

---

## §4 已知历史踩坑（持续追加）

（暂无；每次发现新坑加一行）

---

**最后强调**：本文的 §1 三条是**硬约束**——不是建议、不是优化方向、不是可选。
违反任意一条，平台评测会直接报错（IndexError 等）或正常跑但拿 0 分（被打乱顺序后预测无意义）。
