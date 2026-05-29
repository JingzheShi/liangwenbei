# F1+F2 Per-Feature Audit Report

Scope: F1 LOB派生 (105) + F2 多尺度OFI (50) = **155** features.

Sampling: ref_pool=200,000 train rows, per-sym/per-bucket/val/test sub-sample=50,000, seed=7.


## High-Level Summary

- **Max PSI cross-stock**: 19.1537
- **Max PSI cross-date**:  7.4570
- **Mean NaN% (train) overall**: 0.0256%
- **Mean NaN% (train) F1**:    0.0378%
- **Mean NaN% (train) F2**:    0.0000%
- **KS-drop 11 in scope**: ['kyle_lam_W50', 'kyle_lam_W100']; observed hits in this run: ['kyle_lam_W100', 'kyle_lam_W50']

### Top-5 most unstable across stocks (PSI max)

| name | family | psi_stock_max | psi_date_max | verdict_stock |
|------|--------|---------------|--------------|----------------|
| `bid_mean` | F1 | 19.1537 | 7.5691 | strong_drift |
| `ask_mean` | F1 | 19.1536 | 7.6682 | strong_drift |
| `spread10` | F1 | 19.1203 | 2.6257 | strong_drift |
| `spread9` | F1 | 19.1155 | 2.6302 | strong_drift |
| `ask_diff10` | F1 | 19.1148 | 2.5812 | strong_drift |

### Top-5 most unstable across date (PSI max)

| name | family | psi_stock_max | psi_date_max | verdict_date |
|------|--------|---------------|--------------|---------------|
| `ask_mean` | F1 | 19.1536 | 7.6682 | strong_drift |
| `bid_mean` | F1 | 19.1537 | 7.5691 | strong_drift |
| `ask_diff5` | F1 | 16.9267 | 3.6939 | strong_drift |
| `ask_diff6` | F1 | 16.9042 | 3.6194 | strong_drift |
| `ask_diff7` | F1 | 16.9535 | 3.6585 | strong_drift |

### Top-5 highest NaN% (train)

| name | family | nan%_train | raw/derived |
|------|--------|------------|-------------|
| `ask10` | F1 | 0.1686% | raw |
| `asize10` | F1 | 0.1686% | raw |
| `spread10` | F1 | 0.1686% | raw |
| `ask_diff9` | F1 | 0.1686% | raw |
| `ask_diff10` | F1 | 0.1686% | raw |

---

## F1 — LOB 派生 (105 features)

### F1.A 原始字段 (94 raw)


### `open` — 本 tick 开盘价

- **English**: Open price of current tick (3-second bar)
- **Family / type**: F1 / raw
- **dim_in_X**: 0  (raw idx in 154: 0)
- **In KS-drop 11**: no

**Formula**

$$
\text{open}_t \;=\; \text{raw input (3s OHLC)}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last = X3d[:, -1, :]`

**Numpy pseudo**

```python
open = raw_input[:, 0]
```

**Physical meaning**: 主办方给出的 3 秒级 K 线 本 tick 开盘价；这 4 列共同刻画了短期价格振幅与瞬时方向。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.565422, 0.513008, 0.096877, 0.533721, 0.073588]; max=0.5654, mean=0.3565
**Cross-stock KS** by sym: [0.1181, 0.10318, 0.0969, 0.096025, 0.06312]; max=0.1181
**Cross-date PSI** train→val=0.3594, train→test=0.3024, by_bucket=[0.07728, 0.544138, 0.14211, 0.072488, 0.272774, 0.456042]
**Cross-date KS** train→val=0.1820, train→test=0.1292

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `high` — 本 tick 最高价

- **English**: High price of current tick
- **Family / type**: F1 / raw
- **dim_in_X**: 1  (raw idx in 154: 1)
- **In KS-drop 11**: no

**Formula**

$$
\text{high}_t \;=\; \text{raw input (3s OHLC)}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last = X3d[:, -1, :]`

**Numpy pseudo**

```python
high = raw_input[:, 1]
```

**Physical meaning**: 主办方给出的 3 秒级 K 线 本 tick 最高价；这 4 列共同刻画了短期价格振幅与瞬时方向。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.053484, 0.10633, 0.157956, 0.15218, 0.212236]; max=0.2122, mean=0.1364
**Cross-stock KS** by sym: [0.1139, 0.08279, 0.07697, 0.07271, 0.13942]; max=0.1394
**Cross-date PSI** train→val=0.1539, train→test=0.2027, by_bucket=[0.070988, 0.087309, 0.107874, 0.025421, 0.170328, 0.209991]
**Cross-date KS** train→val=0.1123, train→test=0.1505

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `low` — 本 tick 最低价

- **English**: Low price of current tick
- **Family / type**: F1 / raw
- **dim_in_X**: 2  (raw idx in 154: 2)
- **In KS-drop 11**: no

**Formula**

$$
\text{low}_t \;=\; \text{raw input (3s OHLC)}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last = X3d[:, -1, :]`

**Numpy pseudo**

```python
low = raw_input[:, 2]
```

**Physical meaning**: 主办方给出的 3 秒级 K 线 本 tick 最低价；这 4 列共同刻画了短期价格振幅与瞬时方向。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.228863, 0.052579, 0.141277, 0.051155, 0.123882]; max=0.2289, mean=0.1196
**Cross-stock KS** by sym: [0.172575, 0.07832, 0.119605, 0.060105, 0.14491]; max=0.1726
**Cross-date PSI** train→val=0.1599, train→test=0.1687, by_bucket=[0.027573, 0.033656, 0.044254, 0.059528, 0.142747, 0.197543]
**Cross-date KS** train→val=0.1112, train→test=0.1246

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `close` — 本 tick 收盘价

- **English**: Close price of current tick
- **Family / type**: F1 / raw
- **dim_in_X**: 3  (raw idx in 154: 3)
- **In KS-drop 11**: no

**Formula**

$$
\text{close}_t \;=\; \text{raw input (3s OHLC)}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last = X3d[:, -1, :]`

**Numpy pseudo**

```python
close = raw_input[:, 3]
```

**Physical meaning**: 主办方给出的 3 秒级 K 线 本 tick 收盘价；这 4 列共同刻画了短期价格振幅与瞬时方向。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.101136, 0.04469, 0.059661, 0.040372, 0.099848]; max=0.1011, mean=0.0691
**Cross-stock KS** by sym: [0.08831, 0.073525, 0.093765, 0.040845, 0.08498]; max=0.0938
**Cross-date PSI** train→val=0.0343, train→test=0.1495, by_bucket=[0.055607, 0.030807, 0.0346, 0.020736, 0.060678, 0.169518]
**Cross-date KS** train→val=0.0482, train→test=0.0901

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `bid1` — 第 1 档买盘报价

- **English**: Level-1 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 6  (raw idx in 154: 6)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid1}_t = \text{level-1 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid1 = raw_input[:, 6]  # level-1 bid side quote
```

**Physical meaning**: 第 1 档买盘的报价（bid side L1）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.10671, 0.047309, 0.053972, 0.039898, 0.097111]; max=0.1067, mean=0.0690
**Cross-stock KS** by sym: [0.09153, 0.07632, 0.0856, 0.039965, 0.083005]; max=0.0915
**Cross-date PSI** train→val=0.0313, train→test=0.1503, by_bucket=[0.055229, 0.028019, 0.03502, 0.020044, 0.05902, 0.168231]
**Cross-date KS** train→val=0.0499, train→test=0.0884

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `bid2` — 第 2 档买盘报价

- **English**: Level-2 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 7  (raw idx in 154: 7)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid2}_t = \text{level-2 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid2 = raw_input[:, 7]  # level-2 bid side quote
```

**Physical meaning**: 第 2 档买盘的报价（bid side L2）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.114571, 0.056158, 0.044185, 0.080075, 0.087551]; max=0.1146, mean=0.0765
**Cross-stock KS** by sym: [0.09557, 0.08389, 0.069925, 0.066955, 0.07072]; max=0.0956
**Cross-date PSI** train→val=0.0368, train→test=0.1632, by_bucket=[0.05234, 0.026028, 0.033239, 0.021102, 0.064505, 0.181075]
**Cross-date KS** train→val=0.0490, train→test=0.0863

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `bid3` — 第 3 档买盘报价

- **English**: Level-3 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 8  (raw idx in 154: 8)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid3}_t = \text{level-3 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid3 = raw_input[:, 8]  # level-3 bid side quote
```

**Physical meaning**: 第 3 档买盘的报价（bid side L3）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.115649, 0.060355, 0.039412, 0.075607, 0.078491]; max=0.1156, mean=0.0739
**Cross-stock KS** by sym: [0.09582, 0.09179, 0.0524, 0.107295, 0.07234]; max=0.1073
**Cross-date PSI** train→val=0.0319, train→test=0.1807, by_bucket=[0.052269, 0.026054, 0.034489, 0.021524, 0.071722, 0.200905]
**Cross-date KS** train→val=0.0506, train→test=0.0860

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `bid4` — 第 4 档买盘报价

- **English**: Level-4 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 9  (raw idx in 154: 9)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid4}_t = \text{level-4 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid4 = raw_input[:, 9]  # level-4 bid side quote
```

**Physical meaning**: 第 4 档买盘的报价（bid side L4）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.118792, 0.069118, 0.046729, 0.127212, 0.070948]; max=0.1272, mean=0.0866
**Cross-stock KS** by sym: [0.097795, 0.09759, 0.054535, 0.143095, 0.07625]; max=0.1431
**Cross-date PSI** train→val=0.0307, train→test=0.1671, by_bucket=[0.05245, 0.022892, 0.034871, 0.022642, 0.061591, 0.19134]
**Cross-date KS** train→val=0.0557, train→test=0.0853

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `bid5` — 第 5 档买盘报价

- **English**: Level-5 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 10  (raw idx in 154: 10)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid5}_t = \text{level-5 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid5 = raw_input[:, 10]  # level-5 bid side quote
```

**Physical meaning**: 第 5 档买盘的报价（bid side L5）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.12943, 0.070472, 0.049992, 0.161315, 0.062111]; max=0.1613, mean=0.0947
**Cross-stock KS** by sym: [0.099505, 0.10269, 0.069695, 0.178285, 0.07938]; max=0.1783
**Cross-date PSI** train→val=0.0379, train→test=0.1833, by_bucket=[0.054721, 0.022857, 0.033928, 0.025581, 0.070108, 0.213512]
**Cross-date KS** train→val=0.0531, train→test=0.0853

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `bid6` — 第 6 档买盘报价

- **English**: Level-6 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 11  (raw idx in 154: 11)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid6}_t = \text{level-6 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid6 = raw_input[:, 11]  # level-6 bid side quote
```

**Physical meaning**: 第 6 档买盘的报价（bid side L6）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.131763, 0.07566, 0.059078, 0.2364, 0.057799]; max=0.2364, mean=0.1121
**Cross-stock KS** by sym: [0.101375, 0.107455, 0.0855, 0.21338, 0.08221]; max=0.2134
**Cross-date PSI** train→val=0.0263, train→test=0.1588, by_bucket=[0.059794, 0.025642, 0.036952, 0.026646, 0.055765, 0.18404]
**Cross-date KS** train→val=0.0551, train→test=0.0854

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `bid7` — 第 7 档买盘报价

- **English**: Level-7 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 12  (raw idx in 154: 12)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid7}_t = \text{level-7 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid7 = raw_input[:, 12]  # level-7 bid side quote
```

**Physical meaning**: 第 7 档买盘的报价（bid side L7）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.138326, 0.083574, 0.073978, 0.347684, 0.053556]; max=0.3477, mean=0.1394
**Cross-stock KS** by sym: [0.10327, 0.10901, 0.10152, 0.24689, 0.086145]; max=0.2469
**Cross-date PSI** train→val=0.0207, train→test=0.1332, by_bucket=[0.059026, 0.026946, 0.035486, 0.030901, 0.053257, 0.155086]
**Cross-date KS** train→val=0.0579, train→test=0.0841

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `bid8` — 第 8 档买盘报价

- **English**: Level-8 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 13  (raw idx in 154: 13)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid8}_t = \text{level-8 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid8 = raw_input[:, 13]  # level-8 bid side quote
```

**Physical meaning**: 第 8 档买盘的报价（bid side L8）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.150419, 0.083974, 0.094475, 0.418551, 0.054342]; max=0.4186, mean=0.1604
**Cross-stock KS** by sym: [0.106355, 0.109335, 0.116065, 0.284505, 0.09029]; max=0.2845
**Cross-date PSI** train→val=0.0211, train→test=0.1285, by_bucket=[0.060641, 0.025841, 0.03779, 0.0346, 0.047189, 0.146557]
**Cross-date KS** train→val=0.0572, train→test=0.0840

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `bid9` — 第 9 档买盘报价

- **English**: Level-9 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 14  (raw idx in 154: 14)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid9}_t = \text{level-9 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid9 = raw_input[:, 14]  # level-9 bid side quote
```

**Physical meaning**: 第 9 档买盘的报价（bid side L9）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.165221, 0.084464, 0.122047, 0.524076, 0.060734]; max=0.5241, mean=0.1913
**Cross-stock KS** by sym: [0.109225, 0.11077, 0.13252, 0.321485, 0.094095]; max=0.3215
**Cross-date PSI** train→val=0.0165, train→test=0.1043, by_bucket=[0.065353, 0.029803, 0.040546, 0.039896, 0.035412, 0.121559]
**Cross-date KS** train→val=0.0582, train→test=0.0846

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `bid10` — 第 10 档买盘报价

- **English**: Level-10 bid side price
- **Family / type**: F1 / raw
- **dim_in_X**: 15  (raw idx in 154: 15)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid10}_t = \text{level-10 bid side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
bid10 = raw_input[:, 15]  # level-10 bid side quote
```

**Physical meaning**: 第 10 档买盘的报价（bid side L10）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.174611, 0.078746, 0.150209, 0.641865, 0.069334]; max=0.6419, mean=0.2230
**Cross-stock KS** by sym: [0.112655, 0.10927, 0.149015, 0.354165, 0.097815]; max=0.3542
**Cross-date PSI** train→val=0.0205, train→test=0.1054, by_bucket=[0.062939, 0.027832, 0.038303, 0.036493, 0.039368, 0.12135]
**Cross-date KS** train→val=0.0607, train→test=0.0836

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `bsize1` — 第 1 档买盘挂单量

- **English**: Level-1 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 16  (raw idx in 154: 16)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize1}_t = \text{level-1 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize1 = raw_input[:, 16]  # bid-size L1
```

**Physical meaning**: 第 1 档买盘挂单量（bid-size L1）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.732968, 0.428003, 0.709991, 2.367757, 1.318457]; max=2.3678, mean=1.1114
**Cross-stock KS** by sym: [0.162305, 0.163935, 0.27851, 0.633325, 0.38505]; max=0.6333
**Cross-date PSI** train→val=0.0295, train→test=0.0333, by_bucket=[0.001669, 0.001373, 0.003583, 0.005929, 0.043238, 0.033946]
**Cross-date KS** train→val=0.0517, train→test=0.0556

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `bsize2` — 第 2 档买盘挂单量

- **English**: Level-2 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 17  (raw idx in 154: 17)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize2}_t = \text{level-2 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize2 = raw_input[:, 17]  # bid-size L2
```

**Physical meaning**: 第 2 档买盘挂单量（bid-size L2）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.363431, 0.657696, 0.982935, 4.594953, 1.962737]; max=4.5950, mean=1.9124
**Cross-stock KS** by sym: [0.19534, 0.1833, 0.33704, 0.724295, 0.4465]; max=0.7243
**Cross-date PSI** train→val=0.0564, train→test=0.1150, by_bucket=[0.001107, 0.0027, 0.006584, 0.012415, 0.092777, 0.113352]
**Cross-date KS** train→val=0.0617, train→test=0.0949

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `bsize3` — 第 3 档买盘挂单量

- **English**: Level-3 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 18  (raw idx in 154: 18)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize3}_t = \text{level-3 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize3 = raw_input[:, 18]  # bid-size L3
```

**Physical meaning**: 第 3 档买盘挂单量（bid-size L3）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.446385, 0.580968, 0.94359, 5.152112, 2.009108]; max=5.1521, mean=2.0264
**Cross-stock KS** by sym: [0.210015, 0.18132, 0.340405, 0.7297, 0.455545]; max=0.7297
**Cross-date PSI** train→val=0.0648, train→test=0.1464, by_bucket=[0.000833, 0.007109, 0.002938, 0.009142, 0.103833, 0.154832]
**Cross-date KS** train→val=0.0731, train→test=0.0959

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `bsize4` — 第 4 档买盘挂单量

- **English**: Level-4 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 19  (raw idx in 154: 19)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize4}_t = \text{level-4 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize4 = raw_input[:, 19]  # bid-size L4
```

**Physical meaning**: 第 4 档买盘挂单量（bid-size L4）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.252346, 0.549071, 0.98674, 6.793538, 3.455364]; max=6.7935, mean=2.6074
**Cross-stock KS** by sym: [0.21878, 0.18477, 0.3495, 0.725495, 0.461]; max=0.7255
**Cross-date PSI** train→val=0.0729, train→test=0.1510, by_bucket=[0.004155, 0.007028, 0.0046, 0.009756, 0.0826, 0.165112]
**Cross-date KS** train→val=0.0713, train→test=0.0937

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `bsize5` — 第 5 档买盘挂单量

- **English**: Level-5 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 20  (raw idx in 154: 20)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize5}_t = \text{level-5 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize5 = raw_input[:, 20]  # bid-size L5
```

**Physical meaning**: 第 5 档买盘挂单量（bid-size L5）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.257595, 0.540355, 0.966701, 6.498628, 3.482855]; max=6.4986, mean=2.5492
**Cross-stock KS** by sym: [0.22054, 0.190445, 0.35349, 0.706435, 0.46359]; max=0.7064
**Cross-date PSI** train→val=0.0710, train→test=0.0839, by_bucket=[0.014594, 0.003983, 0.008154, 0.004556, 0.067745, 0.091074]
**Cross-date KS** train→val=0.0661, train→test=0.0722

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `bsize6` — 第 6 档买盘挂单量

- **English**: Level-6 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 21  (raw idx in 154: 21)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize6}_t = \text{level-6 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize6 = raw_input[:, 21]  # bid-size L6
```

**Physical meaning**: 第 6 档买盘挂单量（bid-size L6）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.133766, 0.488749, 0.944562, 5.96478, 1.809465]; max=5.9648, mean=2.0683
**Cross-stock KS** by sym: [0.21717, 0.18928, 0.35257, 0.67524, 0.45827]; max=0.6752
**Cross-date PSI** train→val=0.0872, train→test=0.0452, by_bucket=[0.006198, 0.002562, 0.004013, 0.00287, 0.04546, 0.052265]
**Cross-date KS** train→val=0.0768, train→test=0.0389

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `bsize7` — 第 7 档买盘挂单量

- **English**: Level-7 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 22  (raw idx in 154: 22)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize7}_t = \text{level-7 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize7 = raw_input[:, 22]  # bid-size L7
```

**Physical meaning**: 第 7 档买盘挂单量（bid-size L7）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.962086, 0.440462, 0.853438, 3.532786, 1.737624]; max=3.5328, mean=1.5053
**Cross-stock KS** by sym: [0.21208, 0.19481, 0.346645, 0.63666, 0.45702]; max=0.6367
**Cross-date PSI** train→val=0.0439, train→test=0.0384, by_bucket=[0.002689, 0.001745, 0.00315, 0.003761, 0.028157, 0.043922]
**Cross-date KS** train→val=0.0598, train→test=0.0318

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `bsize8` — 第 8 档买盘挂单量

- **English**: Level-8 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 23  (raw idx in 154: 23)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize8}_t = \text{level-8 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize8 = raw_input[:, 23]  # bid-size L8
```

**Physical meaning**: 第 8 档买盘挂单量（bid-size L8）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.871134, 0.411596, 0.797613, 2.769686, 1.597451]; max=2.7697, mean=1.2895
**Cross-stock KS** by sym: [0.205905, 0.19695, 0.3422, 0.60316, 0.44777]; max=0.6032
**Cross-date PSI** train→val=0.0434, train→test=0.0335, by_bucket=[0.008708, 0.004586, 0.00309, 0.006055, 0.007994, 0.04752]
**Cross-date KS** train→val=0.0590, train→test=0.0521

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `bsize9` — 第 9 档买盘挂单量

- **English**: Level-9 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 24  (raw idx in 154: 24)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize9}_t = \text{level-9 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize9 = raw_input[:, 24]  # bid-size L9
```

**Physical meaning**: 第 9 档买盘挂单量（bid-size L9）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.725926, 0.371585, 0.734385, 2.306062, 1.47997]; max=2.3061, mean=1.1236
**Cross-stock KS** by sym: [0.197175, 0.201705, 0.335275, 0.567465, 0.43509]; max=0.5675
**Cross-date PSI** train→val=0.0240, train→test=0.0278, by_bucket=[0.006725, 0.006088, 0.006305, 0.007785, 0.007013, 0.036156]
**Cross-date KS** train→val=0.0620, train→test=0.0602

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `bsize10` — 第 10 档买盘挂单量

- **English**: Level-10 bid-size
- **Family / type**: F1 / raw
- **dim_in_X**: 25  (raw idx in 154: 25)
- **In KS-drop 11**: no

**Formula**

$$
\text{bsize10}_t = \text{level-10 bid-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize10 = raw_input[:, 25]  # bid-size L10
```

**Physical meaning**: 第 10 档买盘挂单量（bid-size L10）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.556236, 0.321695, 0.638957, 2.148584, 1.335901]; max=2.1486, mean=1.0003
**Cross-stock KS** by sym: [0.1803, 0.198225, 0.32488, 0.55047, 0.429305]; max=0.5505
**Cross-date PSI** train→val=0.0342, train→test=0.0192, by_bucket=[0.008159, 0.008675, 0.009059, 0.011123, 0.0192, 0.024052]
**Cross-date KS** train→val=0.0557, train→test=0.0552

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ask1` — 第 1 档卖盘报价

- **English**: Level-1 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 26  (raw idx in 154: 26)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask1}_t = \text{level-1 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask1 = raw_input[:, 26]  # level-1 ask side quote
```

**Physical meaning**: 第 1 档卖盘的报价（ask side L1）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0391% (576), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.111446, 0.049228, 0.063668, 0.075772, 0.100284]; max=0.1114, mean=0.0801
**Cross-stock KS** by sym: [0.086179, 0.071079, 0.102279, 0.062778, 0.089991]; max=0.1023
**Cross-date PSI** train→val=0.0349, train→test=0.1480, by_bucket=[0.058145, 0.030063, 0.039338, 0.02015, 0.061955, 0.168421]
**Cross-date KS** train→val=0.0542, train→test=0.0939

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `ask2` — 第 2 档卖盘报价

- **English**: Level-2 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 27  (raw idx in 154: 27)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask2}_t = \text{level-2 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask2 = raw_input[:, 27]  # level-2 ask side quote
```

**Physical meaning**: 第 2 档卖盘的报价（ask side L2）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0546% (805), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.098866, 0.046687, 0.07673, 0.08378, 0.108596]; max=0.1086, mean=0.0829
**Cross-stock KS** by sym: [0.083714, 0.06573, 0.119262, 0.098971, 0.102504]; max=0.1193
**Cross-date PSI** train→val=0.0419, train→test=0.1473, by_bucket=[0.058376, 0.034414, 0.038108, 0.017685, 0.052751, 0.167435]
**Cross-date KS** train→val=0.0581, train→test=0.0932

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `ask3` — 第 3 档卖盘报价

- **English**: Level-3 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 28  (raw idx in 154: 28)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask3}_t = \text{level-3 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask3 = raw_input[:, 28]  # level-3 ask side quote
```

**Physical meaning**: 第 3 档卖盘的报价（ask side L3）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0679% (1000), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.102097, 0.036275, 0.100095, 0.095089, 0.116008]; max=0.1160, mean=0.0899
**Cross-stock KS** by sym: [0.082365, 0.05981, 0.136522, 0.134438, 0.110148]; max=0.1365
**Cross-date PSI** train→val=0.0440, train→test=0.1386, by_bucket=[0.058081, 0.032489, 0.033514, 0.019079, 0.054185, 0.157541]
**Cross-date KS** train→val=0.0538, train→test=0.0913

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `ask4` — 第 4 档卖盘报价

- **English**: Level-4 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 29  (raw idx in 154: 29)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask4}_t = \text{level-4 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask4 = raw_input[:, 29]  # level-4 ask side quote
```

**Physical meaning**: 第 4 档卖盘的报价（ask side L4）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0743% (1095), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.114242, 0.039914, 0.113955, 0.139379, 0.116053]; max=0.1394, mean=0.1047
**Cross-stock KS** by sym: [0.085353, 0.054093, 0.152749, 0.172716, 0.118222]; max=0.1727
**Cross-date PSI** train→val=0.0363, train→test=0.1124, by_bucket=[0.057458, 0.03599, 0.032775, 0.020103, 0.045716, 0.128035]
**Cross-date KS** train→val=0.0496, train→test=0.0890

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `ask5` — 第 5 档卖盘报价

- **English**: Level-5 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 30  (raw idx in 154: 30)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask5}_t = \text{level-5 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask5 = raw_input[:, 30]  # level-5 ask side quote
```

**Physical meaning**: 第 5 档卖盘的报价（ask side L5）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0808% (1191), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.11087, 0.038409, 0.13401, 0.190219, 0.118906]; max=0.1902, mean=0.1185
**Cross-stock KS** by sym: [0.086596, 0.04834, 0.169689, 0.201946, 0.122864]; max=0.2019
**Cross-date PSI** train→val=0.0437, train→test=0.1087, by_bucket=[0.058024, 0.032686, 0.026578, 0.020986, 0.053661, 0.119041]
**Cross-date KS** train→val=0.0512, train→test=0.0870

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `ask6` — 第 6 档卖盘报价

- **English**: Level-6 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 31  (raw idx in 154: 31)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask6}_t = \text{level-6 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask6 = raw_input[:, 31]  # level-6 ask side quote
```

**Physical meaning**: 第 6 档卖盘的报价（ask side L6）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0882% (1300), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.122681, 0.03952, 0.155826, 0.239161, 0.120563]; max=0.2392, mean=0.1356
**Cross-stock KS** by sym: [0.088322, 0.043015, 0.186928, 0.237226, 0.128756]; max=0.2372
**Cross-date PSI** train→val=0.0355, train→test=0.0909, by_bucket=[0.057997, 0.033257, 0.026061, 0.021458, 0.040596, 0.101763]
**Cross-date KS** train→val=0.0506, train→test=0.0826

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `ask7` — 第 7 档卖盘报价

- **English**: Level-7 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 32  (raw idx in 154: 32)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask7}_t = \text{level-7 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask7 = raw_input[:, 32]  # level-7 ask side quote
```

**Physical meaning**: 第 7 档卖盘的报价（ask side L7）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0981% (1446), val=0.4248% (1252), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.128805, 0.046819, 0.177728, 0.300011, 0.120165]; max=0.3000, mean=0.1547
**Cross-stock KS** by sym: [0.08935, 0.045747, 0.198479, 0.268851, 0.133208]; max=0.2689
**Cross-date PSI** train→val=0.0379, train→test=0.0880, by_bucket=[0.058229, 0.034038, 0.02867, 0.027092, 0.038744, 0.096957]
**Cross-date KS** train→val=0.0501, train→test=0.0769

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ask8` — 第 8 档卖盘报价

- **English**: Level-8 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 33  (raw idx in 154: 33)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask8}_t = \text{level-8 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask8 = raw_input[:, 33]  # level-8 ask side quote
```

**Physical meaning**: 第 8 档卖盘的报价（ask side L8）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1237% (1823), val=0.4251% (1253), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.142123, 0.05154, 0.198948, 0.420696, 0.126636]; max=0.4207, mean=0.1880
**Cross-stock KS** by sym: [0.08925, 0.052264, 0.208667, 0.30038, 0.138318]; max=0.3004
**Cross-date PSI** train→val=0.0407, train→test=0.0689, by_bucket=[0.062286, 0.032084, 0.029154, 0.02394, 0.029524, 0.083684]
**Cross-date KS** train→val=0.0505, train→test=0.0722

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ask9` — 第 9 档卖盘报价

- **English**: Level-9 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 34  (raw idx in 154: 34)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask9}_t = \text{level-9 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask9 = raw_input[:, 34]  # level-9 ask side quote
```

**Physical meaning**: 第 9 档卖盘的报价（ask side L9）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1446% (2131), val=0.4265% (1257), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.157803, 0.059853, 0.218839, 0.494658, 0.126776]; max=0.4947, mean=0.2116
**Cross-stock KS** by sym: [0.09165, 0.060138, 0.21732, 0.330191, 0.144504]; max=0.3302
**Cross-date PSI** train→val=0.0396, train→test=0.0540, by_bucket=[0.059605, 0.031828, 0.029513, 0.024988, 0.025519, 0.067148]
**Cross-date KS** train→val=0.0505, train→test=0.0670

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ask10` — 第 10 档卖盘报价

- **English**: Level-10 ask side price
- **Family / type**: F1 / raw
- **dim_in_X**: 35  (raw idx in 154: 35)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask10}_t = \text{level-10 ask side quote}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方原始字段)`

**Numpy pseudo**

```python
ask10 = raw_input[:, 35]  # level-10 ask side quote
```

**Physical meaning**: 第 10 档卖盘的报价（ask side L10）；构造 LOB 价格梯度与挂单深度的基础维度。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1686% (2484), val=0.4268% (1258), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.169561, 0.067212, 0.244516, 0.59089, 0.127255]; max=0.5909, mean=0.2399
**Cross-stock KS** by sym: [0.092378, 0.066476, 0.226523, 0.357689, 0.151858]; max=0.3577
**Cross-date PSI** train→val=0.0459, train→test=0.0634, by_bucket=[0.057149, 0.023738, 0.029776, 0.023947, 0.022011, 0.079754]
**Cross-date KS** train→val=0.0480, train→test=0.0634

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `asize1` — 第 1 档卖盘挂单量

- **English**: Level-1 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 36  (raw idx in 154: 36)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize1}_t = \text{level-1 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize1 = raw_input[:, 36]  # ask-size L1
```

**Physical meaning**: 第 1 档卖盘挂单量（ask-size L1）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0391% (576), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.523031, 0.403452, 0.591102, 2.384319, 1.312769]; max=2.3843, mean=1.0429
**Cross-stock KS** by sym: [0.170433, 0.16054, 0.237064, 0.637881, 0.367296]; max=0.6379
**Cross-date PSI** train→val=0.0195, train→test=0.0159, by_bucket=[0.010282, 0.004079, 0.001145, 0.011165, 0.007602, 0.025207]
**Cross-date KS** train→val=0.0393, train→test=0.0438

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `asize2` — 第 2 档卖盘挂单量

- **English**: Level-2 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 37  (raw idx in 154: 37)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize2}_t = \text{level-2 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize2 = raw_input[:, 37]  # ask-size L2
```

**Physical meaning**: 第 2 档卖盘挂单量（ask-size L2）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0546% (805), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.968657, 0.580591, 0.855541, 4.889122, 3.216154]; max=4.8891, mean=2.1020
**Cross-stock KS** by sym: [0.193228, 0.18019, 0.286579, 0.716751, 0.420226]; max=0.7168
**Cross-date PSI** train→val=0.0172, train→test=0.0485, by_bucket=[0.018823, 0.003521, 0.002743, 0.019817, 0.006492, 0.073201]
**Cross-date KS** train→val=0.0435, train→test=0.0703

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `asize3` — 第 3 档卖盘挂单量

- **English**: Level-3 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 38  (raw idx in 154: 38)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize3}_t = \text{level-3 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize3 = raw_input[:, 38]  # ask-size L3
```

**Physical meaning**: 第 3 档卖盘挂单量（ask-size L3）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0679% (1000), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.057725, 0.510468, 0.800302, 6.154487, 1.841249]; max=6.1545, mean=2.0728
**Cross-stock KS** by sym: [0.194585, 0.168858, 0.285205, 0.717222, 0.41778]; max=0.7172
**Cross-date PSI** train→val=0.0231, train→test=0.0786, by_bucket=[0.01869, 0.003084, 0.0037, 0.017099, 0.01324, 0.11189]
**Cross-date KS** train→val=0.0439, train→test=0.0749

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `asize4` — 第 4 档卖盘挂单量

- **English**: Level-4 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 39  (raw idx in 154: 39)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize4}_t = \text{level-4 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize4 = raw_input[:, 39]  # ask-size L4
```

**Physical meaning**: 第 4 档卖盘挂单量（ask-size L4）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0743% (1095), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.11045, 0.566545, 0.897245, 5.019652, 1.631312]; max=5.0197, mean=1.8450
**Cross-stock KS** by sym: [0.196533, 0.172565, 0.293971, 0.722383, 0.414709]; max=0.7224
**Cross-date PSI** train→val=0.0304, train→test=0.0694, by_bucket=[0.022366, 0.004938, 0.006234, 0.04094, 0.012581, 0.100272]
**Cross-date KS** train→val=0.0429, train→test=0.0668

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `asize5` — 第 5 档卖盘挂单量

- **English**: Level-5 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 40  (raw idx in 154: 40)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize5}_t = \text{level-5 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize5 = raw_input[:, 40]  # ask-size L5
```

**Physical meaning**: 第 5 档卖盘挂单量（ask-size L5）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0808% (1191), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.189595, 0.56888, 0.919176, 8.013245, 1.563031]; max=8.0132, mean=2.4508
**Cross-stock KS** by sym: [0.191942, 0.180525, 0.300166, 0.716163, 0.416679]; max=0.7162
**Cross-date PSI** train→val=0.0279, train→test=0.0626, by_bucket=[0.022403, 0.002934, 0.004448, 0.043934, 0.009766, 0.090153]
**Cross-date KS** train→val=0.0467, train→test=0.0651

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `asize6` — 第 6 档卖盘挂单量

- **English**: Level-6 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 41  (raw idx in 154: 41)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize6}_t = \text{level-6 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize6 = raw_input[:, 41]  # ask-size L6
```

**Physical meaning**: 第 6 档卖盘挂单量（ask-size L6）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0882% (1300), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.139112, 0.531633, 0.888757, 9.235099, 1.492361]; max=9.2351, mean=2.6574
**Cross-stock KS** by sym: [0.19253, 0.183003, 0.306415, 0.704988, 0.416198]; max=0.7050
**Cross-date PSI** train→val=0.0337, train→test=0.0302, by_bucket=[0.01396, 0.004427, 0.008159, 0.035706, 0.010095, 0.044768]
**Cross-date KS** train→val=0.0490, train→test=0.0377

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `asize7` — 第 7 档卖盘挂单量

- **English**: Level-7 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 42  (raw idx in 154: 42)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize7}_t = \text{level-7 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize7 = raw_input[:, 42]  # ask-size L7
```

**Physical meaning**: 第 7 档卖盘挂单量（ask-size L7）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0981% (1446), val=0.4248% (1252), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.022344, 0.504157, 0.860602, 9.197151, 1.5366]; max=9.1972, mean=2.6242
**Cross-stock KS** by sym: [0.187483, 0.190252, 0.3134, 0.694923, 0.418815]; max=0.6949
**Cross-date PSI** train→val=0.0430, train→test=0.0215, by_bucket=[0.011863, 0.003863, 0.011019, 0.021152, 0.016159, 0.034675]
**Cross-date KS** train→val=0.0547, train→test=0.0339

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `asize8` — 第 8 档卖盘挂单量

- **English**: Level-8 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 43  (raw idx in 154: 43)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize8}_t = \text{level-8 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize8 = raw_input[:, 43]  # ask-size L8
```

**Physical meaning**: 第 8 档卖盘挂单量（ask-size L8）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1237% (1823), val=0.4251% (1253), test=0.0000% (0)

**Cross-stock PSI** by sym: [1.036532, 0.556786, 0.919315, 7.403937, 1.607986]; max=7.4039, mean=2.3049
**Cross-stock KS** by sym: [0.187793, 0.200821, 0.320728, 0.677462, 0.423448]; max=0.6775
**Cross-date PSI** train→val=0.0365, train→test=0.0092, by_bucket=[0.008792, 0.003818, 0.007076, 0.017081, 0.019142, 0.016728]
**Cross-date KS** train→val=0.0538, train→test=0.0316

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `asize9` — 第 9 档卖盘挂单量

- **English**: Level-9 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 44  (raw idx in 154: 44)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize9}_t = \text{level-9 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize9 = raw_input[:, 44]  # ask-size L9
```

**Physical meaning**: 第 9 档卖盘挂单量（ask-size L9）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1446% (2131), val=0.4265% (1257), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.895192, 0.505714, 1.176534, 4.826525, 1.617878]; max=4.8265, mean=1.8044
**Cross-stock KS** by sym: [0.181245, 0.200827, 0.327304, 0.673928, 0.424323]; max=0.6739
**Cross-date PSI** train→val=0.0320, train→test=0.0078, by_bucket=[0.007803, 0.01125, 0.009827, 0.017287, 0.009504, 0.008733]
**Cross-date KS** train→val=0.0480, train→test=0.0318

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `asize10` — 第 10 档卖盘挂单量

- **English**: Level-10 ask-size
- **Family / type**: F1 / raw
- **dim_in_X**: 45  (raw idx in 154: 45)
- **In KS-drop 11**: no

**Formula**

$$
\text{asize10}_t = \text{level-10 ask-size}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize10 = raw_input[:, 45]  # ask-size L10
```

**Physical meaning**: 第 10 档卖盘挂单量（ask-size L10）的挂单总量；衡量该价位被动流动性。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1686% (2484), val=0.4268% (1258), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.868595, 0.47428, 1.143734, 4.466111, 1.505502]; max=4.4661, mean=1.6916
**Cross-stock KS** by sym: [0.173238, 0.200332, 0.331582, 0.665292, 0.424274]; max=0.6653
**Cross-date PSI** train→val=0.0142, train→test=0.0065, by_bucket=[0.010475, 0.014303, 0.004121, 0.019001, 0.006497, 0.010531]
**Cross-date KS** train→val=0.0396, train→test=0.0301

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `avgbid` — 买盘均价

- **English**: Volume-weighted bid quote average across 10 levels
- **Family / type**: F1 / raw
- **dim_in_X**: 46  (raw idx in 154: 46)
- **In KS-drop 11**: no

**Formula**

$$
\text{avgbid}_t \;=\; \frac{\sum_{k=1}^{10} \text{bid}_k \cdot \text{bsize}_k}{\sum_{k=1}^{10} \text{bsize}_k}\;(\text{exchange-provided})
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
avgbid = raw_input[:, 46]  # exchange-provided
```

**Physical meaning**: 交易所计算的 10 档买盘加权均价，等价于‘卖出全部 10 档可成交的平均价’，衡量买盘综合价位中心。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.134898, 0.033099, 0.15807, 0.092188, 0.070708]; max=0.1581, mean=0.0978
**Cross-stock KS** by sym: [0.14595, 0.05872, 0.182435, 0.108255, 0.106375]; max=0.1824
**Cross-date PSI** train→val=0.0388, train→test=0.0554, by_bucket=[0.041273, 0.027026, 0.034569, 0.075709, 0.057308, 0.062777]
**Cross-date KS** train→val=0.0585, train→test=0.0957

**Verdict**: cross_stock=**mild_drift**, cross_date=**stable**

### `avgask` — 卖盘均价

- **English**: Volume-weighted ask quote average across 10 levels
- **Family / type**: F1 / raw
- **dim_in_X**: 47  (raw idx in 154: 47)
- **In KS-drop 11**: no

**Formula**

$$
\text{avgask}_t \;=\; \frac{\sum_{k=1}^{10} \text{ask}_k \cdot \text{asize}_k}{\sum_{k=1}^{10} \text{asize}_k}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
avgask = raw_input[:, 47]
```

**Physical meaning**: 10 档卖盘加权均价，与 avgbid 配对刻画买卖双方综合价格中心。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [3.601027, 0.105523, 0.685588, 0.120394, 0.238587]; max=3.6010, mean=0.9502
**Cross-stock KS** by sym: [0.47486, 0.111935, 0.35162, 0.11507, 0.169805]; max=0.4749
**Cross-date PSI** train→val=0.1213, train→test=0.1775, by_bucket=[0.106249, 0.053666, 0.029482, 0.051696, 0.147691, 0.170549]
**Cross-date KS** train→val=0.1038, train→test=0.1395

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `totalbsize` — 总买盘挂单量

- **English**: Total bid depth across 10 levels
- **Family / type**: F1 / raw
- **dim_in_X**: 48  (raw idx in 154: 48)
- **In KS-drop 11**: no

**Formula**

$$
\text{totalbsize}_t = \sum_{k=1}^{10}\text{bsize}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方提供，与 sum(bsize_k) 应一致)`

**Numpy pseudo**

```python
totalbsize = bsize1 + bsize2 + ... + bsize10
```

**Physical meaning**: 10 档买方流动性合计；体现接盘力量总规模。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [7.906674, 0.347715, 6.2485, 4.826507, 11.73224]; max=11.7322, mean=6.2123
**Cross-stock KS** by sym: [0.47354, 0.145175, 0.50396, 0.49744, 0.56153]; max=0.5615
**Cross-date PSI** train→val=0.2311, train→test=0.1949, by_bucket=[0.149607, 0.132685, 0.081733, 0.101178, 0.215972, 0.201921]
**Cross-date KS** train→val=0.1027, train→test=0.0744

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `totalasize` — 总卖盘挂单量

- **English**: Total ask depth across 10 levels
- **Family / type**: F1 / raw
- **dim_in_X**: 49  (raw idx in 154: 49)
- **In KS-drop 11**: no

**Formula**

$$
\text{totalasize}_t = \sum_{k=1}^{10}\text{asize}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
totalasize = asize1 + asize2 + ... + asize10
```

**Physical meaning**: 10 档卖方流动性合计；与 totalbsize 配合得到 imbalance 信号的基础量。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.95003, 0.45457, 5.453131, 4.543492, 12.280618]; max=12.2806, mean=4.7364
**Cross-stock KS** by sym: [0.346855, 0.133305, 0.4084, 0.49691, 0.648675]; max=0.6487
**Cross-date PSI** train→val=0.1677, train→test=0.0297, by_bucket=[0.134641, 0.057849, 0.067314, 0.117868, 0.132482, 0.044239]
**Cross-date KS** train→val=0.1136, train→test=0.0466

**Verdict**: cross_stock=**strong_drift**, cross_date=**mild_drift**

### `midprice1` — 第 1 档中间价

- **English**: Level-1 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 68  (raw idx in 154: 68)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice1 = (bid1 + ask1) / 2
```

**Physical meaning**: 第 1 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.110189, 0.051531, 0.055951, 0.068516, 0.096665]; max=0.1102, mean=0.0766
**Cross-stock KS** by sym: [0.089465, 0.07488, 0.094705, 0.045465, 0.088385]; max=0.0947
**Cross-date PSI** train→val=0.0348, train→test=0.1477, by_bucket=[0.056118, 0.032086, 0.038711, 0.017821, 0.049663, 0.172868]
**Cross-date KS** train→val=0.0508, train→test=0.0920

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice2` — 第 2 档中间价

- **English**: Level-2 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 69  (raw idx in 154: 69)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice2 = (bid2 + ask2) / 2
```

**Physical meaning**: 第 2 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.110885, 0.052419, 0.055464, 0.068263, 0.096394]; max=0.1109, mean=0.0767
**Cross-stock KS** by sym: [0.089275, 0.07475, 0.095405, 0.045125, 0.088205]; max=0.0954
**Cross-date PSI** train→val=0.0344, train→test=0.1480, by_bucket=[0.056274, 0.031965, 0.038785, 0.018041, 0.049061, 0.173728]
**Cross-date KS** train→val=0.0509, train→test=0.0920

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice3` — 第 3 档中间价

- **English**: Level-3 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 70  (raw idx in 154: 70)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice3 = (bid3 + ask3) / 2
```

**Physical meaning**: 第 3 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.110268, 0.053618, 0.05616, 0.068225, 0.096098]; max=0.1103, mean=0.0769
**Cross-stock KS** by sym: [0.08887, 0.07492, 0.09534, 0.04464, 0.087075]; max=0.0953
**Cross-date PSI** train→val=0.0347, train→test=0.1484, by_bucket=[0.056097, 0.032354, 0.039353, 0.01789, 0.049479, 0.17432]
**Cross-date KS** train→val=0.0507, train→test=0.0918

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice4` — 第 4 档中间价

- **English**: Level-4 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 71  (raw idx in 154: 71)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice4 = (bid4 + ask4) / 2
```

**Physical meaning**: 第 4 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.110238, 0.053272, 0.055911, 0.06667, 0.095644]; max=0.1102, mean=0.0763
**Cross-stock KS** by sym: [0.088175, 0.075305, 0.094925, 0.04407, 0.08644]; max=0.0949
**Cross-date PSI** train→val=0.0336, train→test=0.1487, by_bucket=[0.05574, 0.033759, 0.039869, 0.017694, 0.047969, 0.174791]
**Cross-date KS** train→val=0.0511, train→test=0.0919

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice5` — 第 5 档中间价

- **English**: Level-5 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 72  (raw idx in 154: 72)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice5 = (bid5 + ask5) / 2
```

**Physical meaning**: 第 5 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.110016, 0.053799, 0.056786, 0.064869, 0.094838]; max=0.1100, mean=0.0761
**Cross-stock KS** by sym: [0.08814, 0.075605, 0.09463, 0.04333, 0.085775]; max=0.0946
**Cross-date PSI** train→val=0.0334, train→test=0.1478, by_bucket=[0.055439, 0.033995, 0.040283, 0.017829, 0.048141, 0.172484]
**Cross-date KS** train→val=0.0515, train→test=0.0920

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice6` — 第 6 档中间价

- **English**: Level-6 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 73  (raw idx in 154: 73)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice6 = (bid6 + ask6) / 2
```

**Physical meaning**: 第 6 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.107716, 0.054767, 0.056472, 0.063086, 0.092689]; max=0.1077, mean=0.0749
**Cross-stock KS** by sym: [0.08773, 0.076895, 0.094405, 0.042405, 0.085]; max=0.0944
**Cross-date PSI** train→val=0.0337, train→test=0.1490, by_bucket=[0.055926, 0.034585, 0.040382, 0.0172, 0.04813, 0.173794]
**Cross-date KS** train→val=0.0517, train→test=0.0914

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice7` — 第 7 档中间价

- **English**: Level-7 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 74  (raw idx in 154: 74)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice7 = (bid7 + ask7) / 2
```

**Physical meaning**: 第 7 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.10533, 0.055389, 0.055702, 0.055024, 0.090614]; max=0.1053, mean=0.0724
**Cross-stock KS** by sym: [0.087205, 0.077475, 0.09398, 0.0414, 0.08383]; max=0.0940
**Cross-date PSI** train→val=0.0343, train→test=0.1517, by_bucket=[0.055672, 0.035015, 0.040987, 0.017172, 0.050226, 0.175815]
**Cross-date KS** train→val=0.0517, train→test=0.0914

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice8` — 第 8 档中间价

- **English**: Level-8 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 75  (raw idx in 154: 75)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice8 = (bid8 + ask8) / 2
```

**Physical meaning**: 第 8 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.10303, 0.054615, 0.056482, 0.057443, 0.088855]; max=0.1030, mean=0.0721
**Cross-stock KS** by sym: [0.086675, 0.07841, 0.093645, 0.04066, 0.08228]; max=0.0936
**Cross-date PSI** train→val=0.0320, train→test=0.1495, by_bucket=[0.055602, 0.035646, 0.040408, 0.018048, 0.04992, 0.173184]
**Cross-date KS** train→val=0.0518, train→test=0.0908

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice9` — 第 9 档中间价

- **English**: Level-9 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 76  (raw idx in 154: 76)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice9 = (bid9 + ask9) / 2
```

**Physical meaning**: 第 9 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.100638, 0.054942, 0.056197, 0.056019, 0.088316]; max=0.1006, mean=0.0712
**Cross-stock KS** by sym: [0.085965, 0.078965, 0.093165, 0.03961, 0.0806]; max=0.0932
**Cross-date PSI** train→val=0.0323, train→test=0.1490, by_bucket=[0.055773, 0.035693, 0.039955, 0.018277, 0.04982, 0.172806]
**Cross-date KS** train→val=0.0517, train→test=0.0905

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `midprice10` — 第 10 档中间价

- **English**: Level-10 midprice = (bid_k + ask_k)/2
- **Family / type**: F1 / raw
- **dim_in_X**: 77  (raw idx in 154: 77)
- **In KS-drop 11**: no

**Formula**

$$
\text{midprice}_k = (\text{bid}_k + \text{ask}_k)/2
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方给出)`

**Numpy pseudo**

```python
midprice10 = (bid10 + ask10) / 2
```

**Physical meaning**: 第 10 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.099639, 0.054454, 0.055855, 0.056762, 0.086637]; max=0.0996, mean=0.0707
**Cross-stock KS** by sym: [0.085045, 0.078885, 0.09275, 0.03851, 0.07953]; max=0.0927
**Cross-date PSI** train→val=0.0318, train→test=0.1482, by_bucket=[0.054987, 0.03547, 0.040472, 0.018032, 0.049526, 0.172731]
**Cross-date KS** train→val=0.0513, train→test=0.0903

**Verdict**: cross_stock=**stable**, cross_date=**mild_drift**

### `spread1` — 第 1 档买卖价差

- **English**: Level-1 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 78  (raw idx in 154: 78)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread1 = ask1 - bid1
```

**Physical meaning**: 第 1 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0391% (576), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [10.44032, 6.215416, 2.016201, 16.828677, 1.032757]; max=16.8287, mean=7.3067
**Cross-stock KS** by sym: [0.452782, 0.293149, 0.557067, 0.786332, 0.331354]; max=0.7863
**Cross-date PSI** train→val=0.5364, train→test=0.7888, by_bucket=[0.134599, 0.12438, 0.12136, 0.203241, 0.664205, 0.79065]
**Cross-date KS** train→val=0.1812, train→test=0.1759

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread2` — 第 2 档买卖价差

- **English**: Level-2 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 79  (raw idx in 154: 79)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread2 = ask2 - bid2
```

**Physical meaning**: 第 2 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0546% (805), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [11.002332, 8.57912, 6.339596, 19.10371, 2.29697]; max=19.1037, mean=9.4643
**Cross-stock KS** by sym: [0.481957, 0.341265, 0.618476, 0.798267, 0.419466]; max=0.7983
**Cross-date PSI** train→val=1.3114, train→test=1.2077, by_bucket=[0.142358, 0.102117, 0.144664, 0.10142, 1.15572, 1.190039]
**Cross-date KS** train→val=0.1807, train→test=0.1745

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread3` — 第 3 档买卖价差

- **English**: Level-3 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 80  (raw idx in 154: 80)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread3 = ask3 - bid3
```

**Physical meaning**: 第 3 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0679% (1000), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.538514, 9.048823, 6.86224, 19.105745, 4.04447]; max=19.1057, mean=10.5200
**Cross-stock KS** by sym: [0.510333, 0.362905, 0.631157, 0.798338, 0.478484]; max=0.7983
**Cross-date PSI** train→val=1.0581, train→test=2.6363, by_bucket=[0.045136, 0.07312, 0.100532, 0.098235, 1.211549, 1.241124]
**Cross-date KS** train→val=0.1808, train→test=0.1746

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread4` — 第 4 档买卖价差

- **English**: Level-4 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 81  (raw idx in 154: 81)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread4 = ask4 - bid4
```

**Physical meaning**: 第 4 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0743% (1095), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [15.147342, 9.074206, 7.336273, 19.107545, 5.708986]; max=19.1075, mean=11.2749
**Cross-stock KS** by sym: [0.513035, 0.373299, 0.642679, 0.798399, 0.510841]; max=0.7984
**Cross-date PSI** train→val=2.4843, train→test=2.6236, by_bucket=[0.050545, 0.101015, 0.097304, 0.143866, 1.242079, 2.634011]
**Cross-date KS** train→val=0.1810, train→test=0.1749

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread5` — 第 5 档买卖价差

- **English**: Level-5 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 82  (raw idx in 154: 82)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread5 = ask5 - bid5
```

**Physical meaning**: 第 5 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0808% (1191), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.53224, 10.458949, 7.855412, 19.107824, 5.9811]; max=19.1078, mean=11.9871
**Cross-stock KS** by sym: [0.51788, 0.379131, 0.655001, 0.798412, 0.532814]; max=0.7984
**Cross-date PSI** train→val=2.4771, train→test=2.6784, by_bucket=[0.072718, 0.087607, 0.098061, 0.160968, 2.603109, 2.673133]
**Cross-date KS** train→val=0.1812, train→test=0.1751

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread6` — 第 6 档买卖价差

- **English**: Level-6 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 83  (raw idx in 154: 83)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread6 = ask6 - bid6
```

**Physical meaning**: 第 6 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0882% (1300), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.518557, 10.526997, 8.196672, 19.107973, 6.238709]; max=19.1080, mean=12.1178
**Cross-stock KS** by sym: [0.517641, 0.382782, 0.665435, 0.798415, 0.547062]; max=0.7984
**Cross-date PSI** train→val=2.4525, train→test=1.2891, by_bucket=[0.082853, 0.087261, 0.084294, 0.180972, 2.572796, 2.698631]
**Cross-date KS** train→val=0.1813, train→test=0.1751

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread7` — 第 7 档买卖价差

- **English**: Level-7 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 84  (raw idx in 154: 84)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread7 = ask7 - bid7
```

**Physical meaning**: 第 7 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0981% (1446), val=0.4248% (1252), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.528749, 11.943534, 9.730814, 19.108163, 6.447725]; max=19.1082, mean=12.7518
**Cross-stock KS** by sym: [0.517362, 0.385926, 0.676171, 0.798428, 0.556579]; max=0.7984
**Cross-date PSI** train→val=2.4444, train→test=1.1952, by_bucket=[0.096202, 0.092208, 0.087012, 0.202038, 2.574854, 1.273289]
**Cross-date KS** train→val=0.1814, train→test=0.1754

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread8` — 第 8 档买卖价差

- **English**: Level-8 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 85  (raw idx in 154: 85)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread8 = ask8 - bid8
```

**Physical meaning**: 第 8 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1237% (1823), val=0.4251% (1253), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.521765, 11.961731, 11.439376, 19.112286, 6.621872]; max=19.1123, mean=13.1314
**Cross-stock KS** by sym: [0.516351, 0.387818, 0.685653, 0.798594, 0.564507]; max=0.7986
**Cross-date PSI** train→val=2.4227, train→test=2.6434, by_bucket=[0.112431, 0.093785, 0.09223, 0.22953, 2.573134, 2.646335]
**Cross-date KS** train→val=0.1817, train→test=0.1755

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread9` — 第 9 档买卖价差

- **English**: Level-9 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 86  (raw idx in 154: 86)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread9 = ask9 - bid9
```

**Physical meaning**: 第 9 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1446% (2131), val=0.4265% (1257), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.511176, 11.976276, 11.535965, 19.115459, 6.783786]; max=19.1155, mean=13.1845
**Cross-stock KS** by sym: [0.518076, 0.389394, 0.69405, 0.798726, 0.570545]; max=0.7987
**Cross-date PSI** train→val=2.4190, train→test=2.6295, by_bucket=[0.136005, 0.085248, 0.09464, 0.25539, 2.560532, 2.630239]
**Cross-date KS** train→val=0.1816, train→test=0.1754

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `spread10` — 第 10 档买卖价差

- **English**: Level-10 bid-ask spread
- **Family / type**: F1 / raw
- **dim_in_X**: 87  (raw idx in 154: 87)
- **In KS-drop 11**: no

**Formula**

$$
\text{spread}_k = \text{ask}_k - \text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
spread10 = ask10 - bid10
```

**Physical meaning**: 第 10 档买卖价差，是流动性成本与做市意愿的直接刻画。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1686% (2484), val=0.4268% (1258), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.505112, 11.997065, 13.100836, 19.120251, 6.918315]; max=19.1203, mean=13.5283
**Cross-stock KS** by sym: [0.518618, 0.390924, 0.701187, 0.798918, 0.57437]; max=0.7989
**Cross-date PSI** train→val=2.4174, train→test=2.6257, by_bucket=[0.153464, 0.08616, 0.097851, 0.265907, 2.566862, 2.623774]
**Cross-date KS** train→val=0.1815, train→test=0.1754

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff1` — 第 1-1 档买价价差

- **English**: Level-1 minus level-0 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 88  (raw idx in 154: 88)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff1 = bid1 - bid1  # 内档间报价跨度
```

**Physical meaning**: 第 1 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [12.889873, 8.551117, 3.021286, 16.992488, 4.135195]; max=16.9925, mean=9.1180
**Cross-stock KS** by sym: [0.54134, 0.36402, 0.59685, 0.79672, 0.505015]; max=0.7967
**Cross-date PSI** train→val=2.9902, train→test=3.2207, by_bucket=[0.116848, 0.154757, 0.160549, 0.508296, 3.166359, 3.247195]
**Cross-date KS** train→val=0.1817, train→test=0.1765

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff2` — 第 2-1 档买价价差

- **English**: Level-2 minus level-1 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 89  (raw idx in 154: 89)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff2 = bid2 - bid1  # 内档间报价跨度
```

**Physical meaning**: 第 2 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.088619, 8.566672, 3.45752, 17.067025, 4.379896]; max=17.0670, mean=9.3119
**Cross-stock KS** by sym: [0.547225, 0.371695, 0.603145, 0.79719, 0.52316]; max=0.7972
**Cross-date PSI** train→val=3.2397, train→test=3.4437, by_bucket=[0.214849, 0.205995, 0.19523, 2.485757, 3.393348, 3.355095]
**Cross-date KS** train→val=0.1820, train→test=0.1763

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff3` — 第 3-1 档买价价差

- **English**: Level-3 minus level-2 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 90  (raw idx in 154: 90)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff3 = bid3 - bid2  # 内档间报价跨度
```

**Physical meaning**: 第 3 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.216437, 8.546322, 3.459272, 19.0776, 4.383765]; max=19.0776, mean=9.7367
**Cross-stock KS** by sym: [0.54663, 0.37221, 0.605705, 0.79736, 0.521095]; max=0.7974
**Cross-date PSI** train→val=3.2206, train→test=3.3619, by_bucket=[0.216675, 0.194831, 0.185053, 2.454535, 3.392828, 3.374963]
**Cross-date KS** train→val=0.1820, train→test=0.1764

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff4` — 第 4-1 档买价价差

- **English**: Level-4 minus level-3 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 91  (raw idx in 154: 91)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff4 = bid4 - bid3  # 内档间报价跨度
```

**Physical meaning**: 第 4 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.192661, 8.538919, 3.615188, 19.078621, 4.360004]; max=19.0786, mean=9.7571
**Cross-stock KS** by sym: [0.544595, 0.37181, 0.605745, 0.797395, 0.518175]; max=0.7974
**Cross-date PSI** train→val=3.2858, train→test=3.5118, by_bucket=[0.199792, 0.190926, 0.195065, 2.477837, 3.426978, 3.419648]
**Cross-date KS** train→val=0.1819, train→test=0.1762

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff5` — 第 5-1 档买价价差

- **English**: Level-5 minus level-4 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 92  (raw idx in 154: 92)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff5 = bid5 - bid4  # 内档间报价跨度
```

**Physical meaning**: 第 5 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.314753, 8.485963, 3.545239, 19.080412, 4.384887]; max=19.0804, mean=9.7623
**Cross-stock KS** by sym: [0.54502, 0.37118, 0.605885, 0.797455, 0.51997]; max=0.7975
**Cross-date PSI** train→val=3.2789, train→test=3.4339, by_bucket=[0.188083, 0.19177, 0.193107, 2.467164, 3.464984, 3.42533]
**Cross-date KS** train→val=0.1818, train→test=0.1763

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff6` — 第 6-1 档买价价差

- **English**: Level-6 minus level-5 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 93  (raw idx in 154: 93)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff6 = bid6 - bid5  # 内档间报价跨度
```

**Physical meaning**: 第 6 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.344771, 8.458232, 3.460625, 19.076285, 4.396309]; max=19.0763, mean=9.7472
**Cross-stock KS** by sym: [0.543345, 0.371135, 0.605755, 0.797305, 0.514915]; max=0.7973
**Cross-date PSI** train→val=3.1363, train→test=3.4714, by_bucket=[0.167863, 0.175925, 0.189619, 2.496747, 3.360687, 3.416763]
**Cross-date KS** train→val=0.1818, train→test=0.1763

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff7` — 第 7-1 档买价价差

- **English**: Level-7 minus level-6 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 94  (raw idx in 154: 94)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff7 = bid7 - bid6  # 内档间报价跨度
```

**Physical meaning**: 第 7 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.415159, 8.404336, 3.47666, 19.077339, 4.335193]; max=19.0773, mean=9.7417
**Cross-stock KS** by sym: [0.542265, 0.370955, 0.607165, 0.797355, 0.516135]; max=0.7974
**Cross-date PSI** train→val=3.1028, train→test=3.3912, by_bucket=[0.125125, 0.156087, 0.177217, 2.451612, 3.307525, 3.465499]
**Cross-date KS** train→val=0.1818, train→test=0.1763

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff8` — 第 8-1 档买价价差

- **English**: Level-8 minus level-7 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 95  (raw idx in 154: 95)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff8 = bid8 - bid7  # 内档间报价跨度
```

**Physical meaning**: 第 8 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.337319, 8.390538, 4.964638, 17.042474, 4.356247]; max=17.0425, mean=9.6182
**Cross-stock KS** by sym: [0.543155, 0.371275, 0.607905, 0.797205, 0.51633]; max=0.7972
**Cross-date PSI** train→val=3.0954, train→test=3.4471, by_bucket=[0.171953, 0.177201, 0.188831, 2.493106, 3.299507, 3.492721]
**Cross-date KS** train→val=0.1819, train→test=0.1761

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff9` — 第 9-1 档买价价差

- **English**: Level-9 minus level-8 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 96  (raw idx in 154: 96)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff9 = bid9 - bid8  # 内档间报价跨度
```

**Physical meaning**: 第 9 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.310851, 8.357826, 3.565236, 17.024699, 4.331321]; max=17.0247, mean=9.3180
**Cross-stock KS** by sym: [0.542335, 0.37126, 0.607465, 0.797185, 0.516725]; max=0.7972
**Cross-date PSI** train→val=3.0409, train→test=3.3787, by_bucket=[0.135344, 0.158365, 0.174598, 2.445733, 3.231124, 3.387587]
**Cross-date KS** train→val=0.1820, train→test=0.1760

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_diff10` — 第 10-1 档买价价差

- **English**: Level-10 minus level-9 bid price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 97  (raw idx in 154: 97)
- **In KS-drop 11**: no

**Formula**

$$
\text{bid\_diff}_k = \text{bid}_k - \text{bid}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_diff10 = bid10 - bid9  # 内档间报价跨度
```

**Physical meaning**: 第 10 档与上一档买盘报价的跨度，刻画 LOB 买盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.571926, 12.108985, 8.214356, 19.113106, 6.481318]; max=19.1131, mean=12.4979
**Cross-stock KS** by sym: [0.537235, 0.389435, 0.644945, 0.798595, 0.568775]; max=0.7986
**Cross-date PSI** train→val=2.5295, train→test=2.7097, by_bucket=[0.145244, 0.063273, 0.097121, 0.256841, 2.680632, 2.708877]
**Cross-date KS** train→val=0.1815, train→test=0.1761

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff1` — 第 1-1 档卖价价差

- **English**: Level-1 minus level-0 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 98  (raw idx in 154: 98)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff1 = ask1 - ask1  # 内档间报价跨度
```

**Physical meaning**: 第 1 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0546% (805), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [12.895591, 8.40107, 2.740238, 17.03396, 3.931055]; max=17.0340, mean=9.0004
**Cross-stock KS** by sym: [0.529325, 0.354928, 0.588169, 0.795806, 0.483791]; max=0.7958
**Cross-date PSI** train→val=2.9880, train→test=3.4412, by_bucket=[0.105605, 0.126203, 0.145987, 0.238754, 3.237324, 3.373198]
**Cross-date KS** train→val=0.1830, train→test=0.1763

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff2` — 第 2-1 档卖价价差

- **English**: Level-2 minus level-1 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 99  (raw idx in 154: 99)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff2 = ask2 - ask1  # 内档间报价跨度
```

**Physical meaning**: 第 2 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0679% (1000), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.013667, 8.27207, 2.892195, 17.041714, 4.00857]; max=17.0417, mean=9.0456
**Cross-stock KS** by sym: [0.528251, 0.357921, 0.595815, 0.795511, 0.491486]; max=0.7955
**Cross-date PSI** train→val=2.9955, train→test=3.3686, by_bucket=[0.122026, 0.125679, 0.159671, 0.235116, 3.212504, 3.399172]
**Cross-date KS** train→val=0.1825, train→test=0.1758

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff3` — 第 3-1 档卖价价差

- **English**: Level-3 minus level-2 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 100  (raw idx in 154: 100)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff3 = ask3 - ask2  # 内档间报价跨度
```

**Physical meaning**: 第 3 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0743% (1095), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.099685, 8.168106, 2.954397, 17.005477, 3.937521]; max=17.0055, mean=9.0330
**Cross-stock KS** by sym: [0.52464, 0.35647, 0.596222, 0.794576, 0.477487]; max=0.7946
**Cross-date PSI** train→val=3.0382, train→test=3.4258, by_bucket=[0.109888, 0.12836, 0.137955, 0.214563, 3.22096, 3.466545]
**Cross-date KS** train→val=0.1823, train→test=0.1752

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff4` — 第 4-1 档卖价价差

- **English**: Level-4 minus level-3 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 101  (raw idx in 154: 101)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff4 = ask4 - ask3  # 内档间报价跨度
```

**Physical meaning**: 第 4 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0808% (1191), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.267351, 10.213521, 2.966076, 16.918652, 3.83861]; max=16.9187, mean=9.4408
**Cross-stock KS** by sym: [0.518941, 0.353184, 0.596013, 0.793557, 0.470578]; max=0.7936
**Cross-date PSI** train→val=2.9781, train→test=3.4680, by_bucket=[0.118313, 0.096977, 0.124704, 0.21696, 3.186278, 3.429314]
**Cross-date KS** train→val=0.1819, train→test=0.1745

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff5` — 第 5-1 档卖价价差

- **English**: Level-5 minus level-4 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 102  (raw idx in 154: 102)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff5 = ask5 - ask4  # 内档间报价跨度
```

**Physical meaning**: 第 5 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0882% (1300), val=0.4238% (1249), test=0.0000% (0)

**Cross-stock PSI** by sym: [14.826822, 10.156709, 2.905975, 16.926675, 3.809444]; max=16.9267, mean=9.7251
**Cross-stock KS** by sym: [0.515239, 0.349341, 0.596176, 0.792669, 0.45916]; max=0.7927
**Cross-date PSI** train→val=2.9167, train→test=3.6939, by_bucket=[0.124719, 0.110027, 0.12171, 0.197472, 3.157222, 3.66244]
**Cross-date KS** train→val=0.1808, train→test=0.1745

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff6` — 第 6-1 档卖价价差

- **English**: Level-6 minus level-5 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 103  (raw idx in 154: 103)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff6 = ask6 - ask5  # 内档间报价跨度
```

**Physical meaning**: 第 6 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0981% (1446), val=0.4248% (1252), test=0.0000% (0)

**Cross-stock PSI** by sym: [14.96094, 10.161009, 2.985657, 16.904169, 3.794898]; max=16.9042, mean=9.7613
**Cross-stock KS** by sym: [0.513403, 0.346752, 0.599966, 0.792412, 0.45018]; max=0.7924
**Cross-date PSI** train→val=3.0591, train→test=3.6194, by_bucket=[0.139032, 0.110872, 0.116544, 0.206613, 3.205292, 3.61434]
**Cross-date KS** train→val=0.1814, train→test=0.1741

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff7` — 第 7-1 档卖价价差

- **English**: Level-7 minus level-6 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 104  (raw idx in 154: 104)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff7 = ask7 - ask6  # 内档间报价跨度
```

**Physical meaning**: 第 7 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1237% (1823), val=0.4251% (1253), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.277942, 10.223934, 3.069645, 16.9535, 3.7929]; max=16.9535, mean=9.4636
**Cross-stock KS** by sym: [0.512491, 0.345284, 0.5988, 0.792572, 0.444277]; max=0.7926
**Cross-date PSI** train→val=3.1200, train→test=3.6083, by_bucket=[0.154234, 0.116774, 0.125747, 0.22155, 3.187641, 3.658526]
**Cross-date KS** train→val=0.1821, train→test=0.1744

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff8` — 第 8-1 档卖价价差

- **English**: Level-8 minus level-7 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 105  (raw idx in 154: 105)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff8 = ask8 - ask7  # 内档间报价跨度
```

**Physical meaning**: 第 8 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1446% (2131), val=0.4265% (1257), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.303481, 10.201109, 4.434644, 16.924719, 3.753604]; max=16.9247, mean=9.7235
**Cross-stock KS** by sym: [0.510841, 0.343587, 0.600514, 0.792717, 0.436643]; max=0.7927
**Cross-date PSI** train→val=3.3216, train→test=3.5321, by_bucket=[0.152204, 0.119179, 0.131015, 0.217868, 3.266289, 3.580997]
**Cross-date KS** train→val=0.1819, train→test=0.1745

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff9` — 第 9-1 档卖价价差

- **English**: Level-9 minus level-8 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 106  (raw idx in 154: 106)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff9 = ask9 - ask8  # 内档间报价跨度
```

**Physical meaning**: 第 9 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1686% (2484), val=0.4268% (1258), test=0.0000% (0)

**Cross-stock PSI** by sym: [13.232511, 10.249472, 2.903819, 16.949433, 3.763734]; max=16.9494, mean=9.4198
**Cross-stock KS** by sym: [0.511385, 0.34266, 0.600751, 0.793343, 0.435844]; max=0.7933
**Cross-date PSI** train→val=3.1527, train→test=3.5486, by_bucket=[0.14303, 0.116041, 0.13043, 0.209079, 3.223343, 3.551828]
**Cross-date KS** train→val=0.1824, train→test=0.1743

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_diff10` — 第 10-1 档卖价价差

- **English**: Level-10 minus level-9 ask price diff
- **Family / type**: F1 / raw
- **dim_in_X**: 107  (raw idx in 154: 107)
- **In KS-drop 11**: no

**Formula**

$$
\text{ask\_diff}_k = \text{ask}_k - \text{ask}_{k-1}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_diff10 = ask10 - ask9  # 内档间报价跨度
```

**Physical meaning**: 第 10 档与上一档卖盘报价的跨度，刻画 LOB 卖盘的价格台阶密度（密=深度好）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.1686% (2484), val=0.4268% (1258), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.547572, 10.095948, 7.604141, 19.114843, 5.594428]; max=19.1148, mean=11.7914
**Cross-stock KS** by sym: [0.501818, 0.368154, 0.651253, 0.798728, 0.4917]; max=0.7987
**Cross-date PSI** train→val=2.4382, train→test=2.5742, by_bucket=[0.14783, 0.110143, 0.097051, 0.175936, 2.58115, 2.555316]
**Cross-date KS** train→val=0.1825, train→test=0.1762

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bid_mean` — 10 档买价均值

- **English**: Mean of 10 bid prices
- **Family / type**: F1 / raw
- **dim_in_X**: 108  (raw idx in 154: 108)
- **In KS-drop 11**: no

**Formula**

$$
\frac{1}{10}\sum_{k=1}^{10}\text{bid}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bid_mean = (bid1+...+bid10)/10
```

**Physical meaning**: 对应方向的算术均值——与 totalbsize/totalasize 等价信息，但更稳健（不受单档异常挂单影响）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.745986, 14.725799, 19.153737, 17.153409, 17.197182]; max=19.1537, mean=16.9952
**Cross-stock KS** by sym: [0.565075, 0.398405, 0.801045, 0.798595, 0.601595]; max=0.8010
**Cross-date PSI** train→val=6.9936, train→test=7.4084, by_bucket=[0.958952, 0.281758, 0.397599, 4.912494, 7.248858, 7.569079]
**Cross-date KS** train→val=0.1975, train→test=0.1992

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `ask_mean` — 10 档卖价均值

- **English**: Mean of 10 ask prices
- **Family / type**: F1 / raw
- **dim_in_X**: 109  (raw idx in 154: 109)
- **In KS-drop 11**: no

**Formula**

$$
\frac{1}{10}\sum_{k=1}^{10}\text{ask}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
ask_mean = (ask1+...+ask10)/10
```

**Physical meaning**: 对应方向的算术均值——与 totalbsize/totalasize 等价信息，但更稳健（不受单档异常挂单影响）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0391% (576), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.751291, 14.74231, 19.153645, 17.150277, 17.206147]; max=19.1536, mean=17.0007
**Cross-stock KS** by sym: [0.565445, 0.39856, 0.800967, 0.798516, 0.60144]; max=0.8010
**Cross-date PSI** train→val=7.0093, train→test=7.4570, by_bucket=[0.97818, 0.281026, 0.397721, 4.928206, 7.267986, 7.668221]
**Cross-date KS** train→val=0.1976, train→test=0.1993

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `bsize_mean` — 10 档买量均值

- **English**: Mean of 10 bid-sizes
- **Family / type**: F1 / raw
- **dim_in_X**: 110  (raw idx in 154: 110)
- **In KS-drop 11**: no

**Formula**

$$
\frac{1}{10}\sum_{k=1}^{10}\text{bsize}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
bsize_mean = totalbsize/10
```

**Physical meaning**: 对应方向的算术均值——与 totalbsize/totalasize 等价信息，但更稳健（不受单档异常挂单影响）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [11.87015, 8.396751, 10.19309, 17.145322, 12.125779]; max=17.1453, mean=11.9462
**Cross-stock KS** by sym: [0.389785, 0.39098, 0.58723, 0.79797, 0.60179]; max=0.7980
**Cross-date PSI** train→val=0.4380, train→test=0.4797, by_bucket=[0.028059, 0.022081, 0.008734, 0.055326, 0.417233, 0.489492]
**Cross-date KS** train→val=0.1382, train→test=0.1381

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `asize_mean` — 10 档卖量均值

- **English**: Mean of 10 ask-sizes
- **Family / type**: F1 / raw
- **dim_in_X**: 111  (raw idx in 154: 111)
- **In KS-drop 11**: no

**Formula**

$$
\frac{1}{10}\sum_{k=1}^{10}\text{asize}_k
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
asize_mean = totalasize/10
```

**Physical meaning**: 对应方向的算术均值——与 totalbsize/totalasize 等价信息，但更稳健（不受单档异常挂单影响）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0391% (576), val=0.4235% (1248), test=0.0000% (0)

**Cross-stock PSI** by sym: [11.851302, 8.346316, 8.591164, 17.08836, 12.039579]; max=17.0884, mean=11.5833
**Cross-stock KS** by sym: [0.387588, 0.395599, 0.588164, 0.796696, 0.59695]; max=0.7967
**Cross-date PSI** train→val=0.1249, train→test=0.2082, by_bucket=[0.087013, 0.030296, 0.024527, 0.098181, 0.120318, 0.277436]
**Cross-date KS** train→val=0.0708, train→test=0.0789

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `cumspread` — 累计 10 档价差

- **English**: Cumulative spread across 10 levels (ask_mean - bid_mean equivalent)
- **Family / type**: F1 / raw
- **dim_in_X**: 112  (raw idx in 154: 112)
- **In KS-drop 11**: no

**Formula**

$$
\text{cumspread}_t = \sum_{k=1}^{10}(\text{ask}_k - \text{bid}_k) = 10\,(\text{ask\_mean} - \text{bid\_mean})
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last`

**Numpy pseudo**

```python
cumspread = sum(spread1..spread10) = 10*(ask_mean - bid_mean)
```

**Physical meaning**: 10 档价差累积；衡量整个 LOB 的整体宽度（流动性/做市意愿宏观指标）。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [16.50466, 8.563622, 8.428948, 17.243451, 6.486733]; max=17.2435, mean=11.4455
**Cross-stock KS** by sym: [0.517045, 0.381075, 0.674725, 0.79655, 0.55427]; max=0.7965
**Cross-date PSI** train→val=2.4235, train→test=2.6286, by_bucket=[0.089671, 0.095961, 0.095972, 0.215376, 2.56161, 2.637612]
**Cross-date KS** train→val=0.1802, train→test=0.1749

**Verdict**: cross_stock=**strong_drift**, cross_date=**strong_drift**

### `imbalance` — 买卖盘力量失衡

- **English**: Order-book imbalance (bid vs ask depth, exchange-provided)
- **Family / type**: F1 / raw
- **dim_in_X**: 113  (raw idx in 154: 113)
- **In KS-drop 11**: no

**Formula**

$$
\text{imbalance}_t = \frac{\text{totalbsize} - \text{totalasize}}{\text{totalbsize} + \text{totalasize}}
$$

- **Code location**: `build_schemeP_cache.py:107 raw_last (主办方提供)`

**Numpy pseudo**

```python
imbalance = (totalbsize - totalasize) / (totalbsize + totalasize)
```

**Physical meaning**: 买卖盘挂单不平衡比例 ∈ (−1, +1)，>0 买方占优、价格倾向上行；最经典的 LOB 方向预测器之一。

**NaN handling**:
  - 公式层: raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）
  - NN 管线: wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，LightGBM 内置 missing routing

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.79942, 0.297965, 0.674345, 2.051992, 3.614496]; max=3.6145, mean=1.4876
**Cross-stock KS** by sym: [0.210875, 0.128165, 0.17211, 0.336255, 0.27889]; max=0.3363
**Cross-date PSI** train→val=0.0441, train→test=0.0208, by_bucket=[0.054809, 0.006152, 0.008257, 0.069749, 0.083417, 0.017861]
**Cross-date KS** train→val=0.0527, train→test=0.0389

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### F1.B 派生字段 (11 derived: 10 wmp_lvl + wmp_balance_12)


### `wmp_lvl1` — 第 1 档加权中价 (WMP)

- **English**: Level-1 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 184  (derived idx in 216: 30)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.105254, 0.050448, 0.057833, 0.059037, 0.10027]; max=0.1053, mean=0.0746
**Cross-stock KS** by sym: [0.086285, 0.06802, 0.091415, 0.04306, 0.086965]; max=0.0914
**Cross-date PSI** train→val=0.0399, train→test=0.1587, by_bucket=[0.056267, 0.030239, 0.037214, 0.018212, 0.063701, 0.180075]
**Cross-date KS** train→val=0.0535, train→test=0.0912

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `wmp_lvl2` — 第 2 档加权中价 (WMP)

- **English**: Level-2 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 185  (derived idx in 216: 31)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.104536, 0.051564, 0.060661, 0.065876, 0.107046]; max=0.1070, mean=0.0779
**Cross-stock KS** by sym: [0.08584, 0.068585, 0.09573, 0.05768, 0.091915]; max=0.0957
**Cross-date PSI** train→val=0.0514, train→test=0.1656, by_bucket=[0.055363, 0.028406, 0.036515, 0.017512, 0.066367, 0.186778]
**Cross-date KS** train→val=0.0551, train→test=0.0941

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `wmp_lvl3` — 第 3 档加权中价 (WMP)

- **English**: Level-3 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 186  (derived idx in 216: 32)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.106036, 0.053918, 0.063219, 0.082752, 0.115634]; max=0.1156, mean=0.0843
**Cross-stock KS** by sym: [0.088545, 0.06939, 0.0986, 0.071395, 0.097905]; max=0.0986
**Cross-date PSI** train→val=0.0432, train→test=0.1490, by_bucket=[0.055123, 0.023338, 0.034904, 0.016238, 0.050847, 0.170921]
**Cross-date KS** train→val=0.0500, train→test=0.0929

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `wmp_lvl4` — 第 4 档加权中价 (WMP)

- **English**: Level-4 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 187  (derived idx in 216: 33)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.1075, 0.050553, 0.063884, 0.037185, 0.11415]; max=0.1142, mean=0.0747
**Cross-stock KS** by sym: [0.095045, 0.07006, 0.100335, 0.047995, 0.093335]; max=0.1003
**Cross-date PSI** train→val=0.0516, train→test=0.1487, by_bucket=[0.0524, 0.034545, 0.04004, 0.018741, 0.04634, 0.177037]
**Cross-date KS** train→val=0.0555, train→test=0.0970

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `wmp_lvl5` — 第 5 档加权中价 (WMP)

- **English**: Level-5 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 188  (derived idx in 216: 34)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.122416, 0.050799, 0.063288, 0.025941, 0.109262]; max=0.1224, mean=0.0743
**Cross-stock KS** by sym: [0.10374, 0.07324, 0.09988, 0.03006, 0.089685]; max=0.1037
**Cross-date PSI** train→val=0.0579, train→test=0.1345, by_bucket=[0.05067, 0.027204, 0.041632, 0.019149, 0.04852, 0.153543]
**Cross-date KS** train→val=0.0574, train→test=0.0883

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `wmp_lvl6` — 第 6 档加权中价 (WMP)

- **English**: Level-6 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 189  (derived idx in 216: 35)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.138969, 0.05377, 0.056415, 0.018412, 0.101355]; max=0.1390, mean=0.0738
**Cross-stock KS** by sym: [0.113135, 0.07727, 0.0944, 0.028695, 0.082125]; max=0.1131
**Cross-date PSI** train→val=0.0501, train→test=0.1377, by_bucket=[0.047426, 0.027249, 0.040799, 0.012595, 0.046441, 0.158107]
**Cross-date KS** train→val=0.0554, train→test=0.0799

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `wmp_lvl7` — 第 7 档加权中价 (WMP)

- **English**: Level-7 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 190  (derived idx in 216: 36)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.148042, 0.053348, 0.053793, 0.019946, 0.089576]; max=0.1480, mean=0.0729
**Cross-stock KS** by sym: [0.118, 0.07593, 0.08802, 0.045075, 0.076815]; max=0.1180
**Cross-date PSI** train→val=0.0371, train→test=0.1166, by_bucket=[0.049333, 0.027506, 0.036494, 0.009518, 0.045753, 0.129338]
**Cross-date KS** train→val=0.0476, train→test=0.0762

**Verdict**: cross_stock=**mild_drift**, cross_date=**mild_drift**

### `wmp_lvl8` — 第 8 档加权中价 (WMP)

- **English**: Level-8 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 191  (derived idx in 216: 37)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.158572, 0.053699, 0.04745, 0.035168, 0.081408]; max=0.1586, mean=0.0753
**Cross-stock KS** by sym: [0.11546, 0.079635, 0.08088, 0.054385, 0.0715]; max=0.1155
**Cross-date PSI** train→val=0.0232, train→test=0.0770, by_bucket=[0.055565, 0.02894, 0.028085, 0.007676, 0.042806, 0.078866]
**Cross-date KS** train→val=0.0387, train→test=0.0733

**Verdict**: cross_stock=**mild_drift**, cross_date=**stable**

### `wmp_lvl9` — 第 9 档加权中价 (WMP)

- **English**: Level-9 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 192  (derived idx in 216: 38)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.174294, 0.053339, 0.043444, 0.063668, 0.07559]; max=0.1743, mean=0.0821
**Cross-stock KS** by sym: [0.127965, 0.081395, 0.07476, 0.102075, 0.060675]; max=0.1280
**Cross-date PSI** train→val=0.0094, train→test=0.0672, by_bucket=[0.04962, 0.026602, 0.032614, 0.007798, 0.030287, 0.067844]
**Cross-date KS** train→val=0.0264, train→test=0.0751

**Verdict**: cross_stock=**mild_drift**, cross_date=**stable**

### `wmp_lvl10` — 第 10 档加权中价 (WMP)

- **English**: Level-10 weighted mid price (size-weighted micro price)
- **Family / type**: F1 / derived
- **dim_in_X**: 193  (derived idx in 216: 39)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}{\text{bsize}_k + \text{asize}_k}\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}
$$

- **Code location**: `fast_features_batch.py:202-217`

**Numpy pseudo**

```python
denom = bsize_k + asize_k
wmp_k = where(denom==0, (ask_k+bid_k)/2, (ask_k*bsize_k + bid_k*asize_k) / denom)
```

**Physical meaning**: Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom==0 时回退到 (ask+bid)/2 (公式层兜底)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.183951, 0.048936, 0.046681, 0.094951, 0.065506]; max=0.1840, mean=0.0880
**Cross-stock KS** by sym: [0.12668, 0.078795, 0.066165, 0.117095, 0.05954]; max=0.1267
**Cross-date PSI** train→val=0.0122, train→test=0.0673, by_bucket=[0.048561, 0.030055, 0.032355, 0.006995, 0.02194, 0.077122]
**Cross-date KS** train→val=0.0284, train→test=0.0715

**Verdict**: cross_stock=**mild_drift**, cross_date=**stable**

### `wmp_balance_12` — 1-2 档加权中价之差

- **English**: WMP balance level-1 minus level-2
- **Family / type**: F1 / derived
- **dim_in_X**: 194  (derived idx in 216: 40)
- **In KS-drop 11**: no

**Formula**

$$
\text{wmp\_balance\_12} = \text{wmp\_lvl1} - \text{wmp\_lvl2}
$$

- **Code location**: `fast_features_batch.py:215-216`

**Numpy pseudo**

```python
wmp_balance_12 = wmp_lvl1[-1] - wmp_lvl2[-1]
```

**Physical meaning**: 1 档与 2 档加权中价之差，正负号反映 1 档相对 2 档的局部失衡——当 lvl1 的加权中价显著高于 lvl2 表示 ask 一侧近端挂单量更大或将吸吃，预测短期下行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.149993, 0.143158, 0.923506, 0.786677, 0.166655]; max=0.9235, mean=0.4340
**Cross-stock KS** by sym: [0.121485, 0.06447, 0.230305, 0.238075, 0.120075]; max=0.2381
**Cross-date PSI** train→val=0.0011, train→test=0.0037, by_bucket=[0.012383, 0.005009, 0.000173, 0.010278, 0.013683, 0.003641]
**Cross-date KS** train→val=0.0082, train→test=0.0223

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

---

## F2 — 多尺度 OFI (50 features, all derived)


### `mlofi_W5_lvl1` — 多档 OFI 窗口5-档1

- **English**: Multi-level OFI window=5 ticks, level=1 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 154  (derived idx in 216: 0)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid1, ask1, bsize1, asize1
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl1 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.139123, 0.059112, 0.035399, 0.194334, 0.230344]; max=0.2303, mean=0.1317
**Cross-stock KS** by sym: [0.078155, 0.059565, 0.039545, 0.105105, 0.09833]; max=0.1051
**Cross-date PSI** train→val=0.0099, train→test=0.0219, by_bucket=[0.01613, 0.013914, 0.002268, 0.005704, 0.006271, 0.026299]
**Cross-date KS** train→val=0.0160, train→test=0.0357

**Verdict**: cross_stock=**mild_drift**, cross_date=**stable**

### `mlofi_W5_lvl2` — 多档 OFI 窗口5-档2

- **English**: Multi-level OFI window=5 ticks, level=2 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 155  (derived idx in 216: 1)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid2, ask2, bsize2, asize2
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl2 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.092673, 0.160451, 0.19388, 0.06143, 0.210821]; max=0.2108, mean=0.1439
**Cross-stock KS** by sym: [0.077985, 0.07907, 0.096005, 0.07063, 0.06684]; max=0.0960
**Cross-date PSI** train→val=0.0047, train→test=0.0237, by_bucket=[0.007817, 0.013153, 0.001091, 0.002266, 0.006285, 0.026653]
**Cross-date KS** train→val=0.0160, train→test=0.0358

**Verdict**: cross_stock=**mild_drift**, cross_date=**stable**

### `mlofi_W5_lvl3` — 多档 OFI 窗口5-档3

- **English**: Multi-level OFI window=5 ticks, level=3 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 156  (derived idx in 216: 2)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid3, ask3, bsize3, asize3
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl3 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.101973, 0.162942, 0.286922, 0.213513, 0.257503]; max=0.2869, mean=0.2046
**Cross-stock KS** by sym: [0.077235, 0.079515, 0.11995, 0.113235, 0.08329]; max=0.1200
**Cross-date PSI** train→val=0.0023, train→test=0.0201, by_bucket=[0.007503, 0.010577, 0.00098, 0.000976, 0.004281, 0.024175]
**Cross-date KS** train→val=0.0151, train→test=0.0404

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W5_lvl4` — 多档 OFI 窗口5-档4

- **English**: Multi-level OFI window=5 ticks, level=4 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 157  (derived idx in 216: 3)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid4, ask4, bsize4, asize4
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl4 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.102005, 0.178207, 0.359885, 0.352356, 0.339308]; max=0.3599, mean=0.2664
**Cross-stock KS** by sym: [0.07313, 0.082485, 0.130905, 0.14134, 0.096475]; max=0.1413
**Cross-date PSI** train→val=0.0026, train→test=0.0252, by_bucket=[0.008219, 0.009415, 0.001653, 0.000617, 0.005329, 0.029106]
**Cross-date KS** train→val=0.0157, train→test=0.0430

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W5_lvl5` — 多档 OFI 窗口5-档5

- **English**: Multi-level OFI window=5 ticks, level=5 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 158  (derived idx in 216: 4)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid5, ask5, bsize5, asize5
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl5 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.117333, 0.200719, 0.441778, 0.420099, 0.362563]; max=0.4418, mean=0.3085
**Cross-stock KS** by sym: [0.07012, 0.08412, 0.139805, 0.15314, 0.105495]; max=0.1531
**Cross-date PSI** train→val=0.0015, train→test=0.0239, by_bucket=[0.00828, 0.009926, 0.001178, 0.000923, 0.004917, 0.028698]
**Cross-date KS** train→val=0.0176, train→test=0.0436

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W5_lvl6` — 多档 OFI 窗口5-档6

- **English**: Multi-level OFI window=5 ticks, level=6 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 159  (derived idx in 216: 5)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid6, ask6, bsize6, asize6
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl6 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.099666, 0.209099, 0.460851, 0.571863, 0.480303]; max=0.5719, mean=0.3644
**Cross-stock KS** by sym: [0.071085, 0.08491, 0.1535, 0.17799, 0.119995]; max=0.1780
**Cross-date PSI** train→val=0.0030, train→test=0.0254, by_bucket=[0.008033, 0.011254, 0.00109, 0.000684, 0.007121, 0.029519]
**Cross-date KS** train→val=0.0184, train→test=0.0432

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W5_lvl7` — 多档 OFI 窗口5-档7

- **English**: Multi-level OFI window=5 ticks, level=7 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 160  (derived idx in 216: 6)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid7, ask7, bsize7, asize7
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl7 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.130299, 0.23066, 0.61259, 0.709054, 0.526662]; max=0.7091, mean=0.4419
**Cross-stock KS** by sym: [0.069795, 0.085515, 0.158685, 0.195475, 0.12848]; max=0.1955
**Cross-date PSI** train→val=0.0023, train→test=0.0248, by_bucket=[0.006467, 0.00997, 0.000707, 0.00099, 0.007172, 0.027374]
**Cross-date KS** train→val=0.0171, train→test=0.0446

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W5_lvl8` — 多档 OFI 窗口5-档8

- **English**: Multi-level OFI window=5 ticks, level=8 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 161  (derived idx in 216: 7)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid8, ask8, bsize8, asize8
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl8 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.119262, 0.229932, 0.629512, 0.802082, 0.555366]; max=0.8021, mean=0.4672
**Cross-stock KS** by sym: [0.073135, 0.08711, 0.16383, 0.203835, 0.13611]; max=0.2038
**Cross-date PSI** train→val=0.0019, train→test=0.0237, by_bucket=[0.005893, 0.010148, 0.000654, 0.000307, 0.006524, 0.026624]
**Cross-date KS** train→val=0.0183, train→test=0.0428

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W5_lvl9` — 多档 OFI 窗口5-档9

- **English**: Multi-level OFI window=5 ticks, level=9 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 162  (derived idx in 216: 8)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid9, ask9, bsize9, asize9
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl9 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.123244, 0.230094, 0.629808, 0.90542, 0.625318]; max=0.9054, mean=0.5028
**Cross-stock KS** by sym: [0.071195, 0.087, 0.165405, 0.2128, 0.141325]; max=0.2128
**Cross-date PSI** train→val=0.0035, train→test=0.0247, by_bucket=[0.00693, 0.010832, 0.001189, 0.000533, 0.007893, 0.02834]
**Cross-date KS** train→val=0.0164, train→test=0.0399

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W5_lvl10` — 多档 OFI 窗口5-档10

- **English**: Multi-level OFI window=5 ticks, level=10 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 163  (derived idx in 216: 9)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid10, ask10, bsize10, asize10
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W5_lvl10 = nansum(e_t[last_5_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=5-tick (≈15s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.137233, 0.234314, 0.627107, 0.967805, 0.701074]; max=0.9678, mean=0.5335
**Cross-stock KS** by sym: [0.07289, 0.086815, 0.171095, 0.216385, 0.14429]; max=0.2164
**Cross-date PSI** train→val=0.0027, train→test=0.0226, by_bucket=[0.006485, 0.011204, 0.000291, 0.001247, 0.005478, 0.025352]
**Cross-date KS** train→val=0.0185, train→test=0.0424

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl1` — 多档 OFI 窗口20-档1

- **English**: Multi-level OFI window=20 ticks, level=1 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 164  (derived idx in 216: 10)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid1, ask1, bsize1, asize1
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl1 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.182684, 0.059999, 0.030035, 0.280761, 0.321063]; max=0.3211, mean=0.1749
**Cross-stock KS** by sym: [0.086965, 0.06289, 0.038375, 0.121395, 0.115875]; max=0.1214
**Cross-date PSI** train→val=0.0096, train→test=0.0186, by_bucket=[0.014798, 0.01286, 0.003213, 0.004947, 0.006457, 0.026099]
**Cross-date KS** train→val=0.0170, train→test=0.0415

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl2` — 多档 OFI 窗口20-档2

- **English**: Multi-level OFI window=20 ticks, level=2 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 165  (derived idx in 216: 11)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid2, ask2, bsize2, asize2
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl2 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.090867, 0.126578, 0.082862, 0.114445, 0.242591]; max=0.2426, mean=0.1315
**Cross-stock KS** by sym: [0.050105, 0.09029, 0.070935, 0.06839, 0.092105]; max=0.0921
**Cross-date PSI** train→val=0.0077, train→test=0.0228, by_bucket=[0.009409, 0.01186, 0.003991, 0.004842, 0.00585, 0.029859]
**Cross-date KS** train→val=0.0183, train→test=0.0356

**Verdict**: cross_stock=**mild_drift**, cross_date=**stable**

### `mlofi_W20_lvl3` — 多档 OFI 窗口20-档3

- **English**: Multi-level OFI window=20 ticks, level=3 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 166  (derived idx in 216: 12)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid3, ask3, bsize3, asize3
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl3 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.087085, 0.143041, 0.158855, 0.279977, 0.267823]; max=0.2800, mean=0.1874
**Cross-stock KS** by sym: [0.049525, 0.09355, 0.07755, 0.09967, 0.09003]; max=0.0997
**Cross-date PSI** train→val=0.0052, train→test=0.0279, by_bucket=[0.009235, 0.010254, 0.003084, 0.004278, 0.005535, 0.035437]
**Cross-date KS** train→val=0.0131, train→test=0.0372

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl4` — 多档 OFI 窗口20-档4

- **English**: Multi-level OFI window=20 ticks, level=4 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 167  (derived idx in 216: 13)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid4, ask4, bsize4, asize4
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl4 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.077607, 0.151474, 0.240289, 0.445655, 0.325021]; max=0.4457, mean=0.2480
**Cross-stock KS** by sym: [0.043015, 0.095635, 0.08715, 0.12596, 0.088275]; max=0.1260
**Cross-date PSI** train→val=0.0061, train→test=0.0338, by_bucket=[0.009453, 0.010169, 0.00264, 0.00391, 0.006705, 0.042376]
**Cross-date KS** train→val=0.0136, train→test=0.0453

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl5` — 多档 OFI 窗口20-档5

- **English**: Multi-level OFI window=20 ticks, level=5 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 168  (derived idx in 216: 14)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid5, ask5, bsize5, asize5
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl5 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.079419, 0.150647, 0.259694, 0.549071, 0.367312]; max=0.5491, mean=0.2812
**Cross-stock KS** by sym: [0.04246, 0.095755, 0.090015, 0.13502, 0.089155]; max=0.1350
**Cross-date PSI** train→val=0.0064, train→test=0.0412, by_bucket=[0.010378, 0.011078, 0.003378, 0.004331, 0.008868, 0.049731]
**Cross-date KS** train→val=0.0191, train→test=0.0411

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl6` — 多档 OFI 窗口20-档6

- **English**: Multi-level OFI window=20 ticks, level=6 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 169  (derived idx in 216: 15)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid6, ask6, bsize6, asize6
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl6 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.067683, 0.17605, 0.343025, 0.752393, 0.467265]; max=0.7524, mean=0.3613
**Cross-stock KS** by sym: [0.037475, 0.102575, 0.10231, 0.15718, 0.09064]; max=0.1572
**Cross-date PSI** train→val=0.0058, train→test=0.0491, by_bucket=[0.011686, 0.013401, 0.0029, 0.003457, 0.010709, 0.054168]
**Cross-date KS** train→val=0.0169, train→test=0.0449

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl7` — 多档 OFI 窗口20-档7

- **English**: Multi-level OFI window=20 ticks, level=7 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 170  (derived idx in 216: 16)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid7, ask7, bsize7, asize7
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl7 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.061569, 0.17773, 0.456, 0.881528, 0.564164]; max=0.8815, mean=0.4282
**Cross-stock KS** by sym: [0.0369, 0.106455, 0.104395, 0.17038, 0.09166]; max=0.1704
**Cross-date PSI** train→val=0.0066, train→test=0.0520, by_bucket=[0.009006, 0.012807, 0.002708, 0.003339, 0.01685, 0.057833]
**Cross-date KS** train→val=0.0179, train→test=0.0478

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl8` — 多档 OFI 窗口20-档8

- **English**: Multi-level OFI window=20 ticks, level=8 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 171  (derived idx in 216: 17)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid8, ask8, bsize8, asize8
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl8 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.048273, 0.184787, 0.502626, 1.020995, 0.699445]; max=1.0210, mean=0.4912
**Cross-stock KS** by sym: [0.036075, 0.108565, 0.110535, 0.182485, 0.091995]; max=0.1825
**Cross-date PSI** train→val=0.0063, train→test=0.0511, by_bucket=[0.009278, 0.011864, 0.00181, 0.002238, 0.014623, 0.056685]
**Cross-date KS** train→val=0.0190, train→test=0.0508

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl9` — 多档 OFI 窗口20-档9

- **English**: Multi-level OFI window=20 ticks, level=9 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 172  (derived idx in 216: 18)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid9, ask9, bsize9, asize9
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl9 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.053654, 0.176465, 0.596957, 1.105927, 0.772519]; max=1.1059, mean=0.5411
**Cross-stock KS** by sym: [0.03372, 0.10956, 0.10828, 0.18819, 0.09224]; max=0.1882
**Cross-date PSI** train→val=0.0085, train→test=0.0548, by_bucket=[0.00979, 0.012869, 0.002242, 0.002234, 0.016734, 0.06307]
**Cross-date KS** train→val=0.0190, train→test=0.0488

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W20_lvl10` — 多档 OFI 窗口20-档10

- **English**: Multi-level OFI window=20 ticks, level=10 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 173  (derived idx in 216: 19)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid10, ask10, bsize10, asize10
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W20_lvl10 = nansum(e_t[last_20_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=20-tick (≈60s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.046789, 0.184203, 0.266299, 1.049844, 0.376679]; max=1.0498, mean=0.3848
**Cross-stock KS** by sym: [0.03349, 0.110045, 0.115065, 0.198105, 0.09991]; max=0.1981
**Cross-date PSI** train→val=0.0059, train→test=0.0376, by_bucket=[0.008965, 0.014413, 0.001641, 0.001815, 0.009632, 0.045018]
**Cross-date KS** train→val=0.0235, train→test=0.0494

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl1` — 多档 OFI 窗口60-档1

- **English**: Multi-level OFI window=60 ticks, level=1 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 174  (derived idx in 216: 20)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid1, ask1, bsize1, asize1
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl1 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.231358, 0.052922, 0.026962, 0.376102, 0.368164]; max=0.3761, mean=0.2111
**Cross-stock KS** by sym: [0.098235, 0.066955, 0.04468, 0.136655, 0.1269]; max=0.1367
**Cross-date PSI** train→val=0.0105, train→test=0.0190, by_bucket=[0.017581, 0.015169, 0.006954, 0.00391, 0.008426, 0.026754]
**Cross-date KS** train→val=0.0183, train→test=0.0402

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl2` — 多档 OFI 窗口60-档2

- **English**: Multi-level OFI window=60 ticks, level=2 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 175  (derived idx in 216: 21)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid2, ask2, bsize2, asize2
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl2 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.165518, 0.088241, 0.056207, 0.254601, 0.314464]; max=0.3145, mean=0.1758
**Cross-stock KS** by sym: [0.075355, 0.083435, 0.06904, 0.11298, 0.10867]; max=0.1130
**Cross-date PSI** train→val=0.0086, train→test=0.0242, by_bucket=[0.011686, 0.01218, 0.00603, 0.004706, 0.007187, 0.034012]
**Cross-date KS** train→val=0.0158, train→test=0.0324

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl3` — 多档 OFI 窗口60-档3

- **English**: Multi-level OFI window=60 ticks, level=3 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 176  (derived idx in 216: 22)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid3, ask3, bsize3, asize3
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl3 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.147883, 0.111365, 0.097915, 0.411877, 0.339031]; max=0.4119, mean=0.2216
**Cross-stock KS** by sym: [0.07, 0.09018, 0.07281, 0.11419, 0.109255]; max=0.1142
**Cross-date PSI** train→val=0.0063, train→test=0.0304, by_bucket=[0.010721, 0.012001, 0.006817, 0.0046, 0.004556, 0.049151]
**Cross-date KS** train→val=0.0180, train→test=0.0429

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl4` — 多档 OFI 窗口60-档4

- **English**: Multi-level OFI window=60 ticks, level=4 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 177  (derived idx in 216: 23)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid4, ask4, bsize4, asize4
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl4 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.142696, 0.117711, 0.127458, 0.592212, 0.357735]; max=0.5922, mean=0.2676
**Cross-stock KS** by sym: [0.067085, 0.089035, 0.06939, 0.115925, 0.10843]; max=0.1159
**Cross-date PSI** train→val=0.0067, train→test=0.0379, by_bucket=[0.011501, 0.011139, 0.005884, 0.004468, 0.006866, 0.051077]
**Cross-date KS** train→val=0.0167, train→test=0.0457

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl5` — 多档 OFI 窗口60-档5

- **English**: Multi-level OFI window=60 ticks, level=5 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 178  (derived idx in 216: 24)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid5, ask5, bsize5, asize5
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl5 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.128084, 0.121608, 0.12621, 0.649771, 0.390635]; max=0.6498, mean=0.2833
**Cross-stock KS** by sym: [0.06189, 0.088965, 0.065225, 0.11456, 0.1132]; max=0.1146
**Cross-date PSI** train→val=0.0092, train→test=0.0416, by_bucket=[0.012705, 0.01122, 0.006945, 0.004762, 0.010104, 0.05061]
**Cross-date KS** train→val=0.0232, train→test=0.0454

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl6` — 多档 OFI 窗口60-档6

- **English**: Multi-level OFI window=60 ticks, level=6 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 179  (derived idx in 216: 25)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid6, ask6, bsize6, asize6
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl6 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.111085, 0.134585, 0.152639, 0.823009, 0.408453]; max=0.8230, mean=0.3260
**Cross-stock KS** by sym: [0.05828, 0.097025, 0.074015, 0.11288, 0.11295]; max=0.1129
**Cross-date PSI** train→val=0.0075, train→test=0.0468, by_bucket=[0.014095, 0.014073, 0.005398, 0.002995, 0.009197, 0.060145]
**Cross-date KS** train→val=0.0196, train→test=0.0430

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl7` — 多档 OFI 窗口60-档7

- **English**: Multi-level OFI window=60 ticks, level=7 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 180  (derived idx in 216: 26)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid7, ask7, bsize7, asize7
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl7 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.096512, 0.137441, 0.184862, 0.870148, 0.437416]; max=0.8701, mean=0.3453
**Cross-stock KS** by sym: [0.054425, 0.09996, 0.077725, 0.122465, 0.117875]; max=0.1225
**Cross-date PSI** train→val=0.0102, train→test=0.0546, by_bucket=[0.012391, 0.014409, 0.004666, 0.004383, 0.017796, 0.065732]
**Cross-date KS** train→val=0.0232, train→test=0.0470

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl8` — 多档 OFI 窗口60-档8

- **English**: Multi-level OFI window=60 ticks, level=8 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 181  (derived idx in 216: 27)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid8, ask8, bsize8, asize8
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl8 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.081217, 0.153879, 0.208524, 0.920063, 0.460708]; max=0.9201, mean=0.3649
**Cross-stock KS** by sym: [0.049245, 0.10369, 0.07926, 0.136095, 0.11901]; max=0.1361
**Cross-date PSI** train→val=0.0081, train→test=0.0564, by_bucket=[0.013113, 0.014472, 0.003999, 0.003862, 0.014972, 0.068863]
**Cross-date KS** train→val=0.0222, train→test=0.0528

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl9` — 多档 OFI 窗口60-档9

- **English**: Multi-level OFI window=60 ticks, level=9 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 182  (derived idx in 216: 28)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid9, ask9, bsize9, asize9
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl9 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.080135, 0.166596, 0.207435, 0.985946, 0.474781]; max=0.9859, mean=0.3830
**Cross-stock KS** by sym: [0.04813, 0.10448, 0.07565, 0.145675, 0.12145]; max=0.1457
**Cross-date PSI** train→val=0.0123, train→test=0.0621, by_bucket=[0.013536, 0.01438, 0.005109, 0.004308, 0.017574, 0.076792]
**Cross-date KS** train→val=0.0234, train→test=0.0568

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `mlofi_W60_lvl10` — 多档 OFI 窗口60-档10

- **English**: Multi-level OFI window=60 ticks, level=10 (Cont 2014)
- **Family / type**: F2 / derived
- **dim_in_X**: 183  (derived idx in 216: 29)
- **In KS-drop 11**: no

**Formula**

$$
e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t -\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} -\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t +\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}\;,\quad \text{mlofi}_W^{(k)} = \sum_{\tau=t-W+1}^{t} e_\tau^{(k)}
$$

- **Code location**: `fast_features_batch.py:156-184 (e_per_lvl + W-sum)`

**Numpy pseudo**

```python
# Cont 2014 OFI 单档 e_t (with prev-tick lag):
# b, a, bs, as = bid10, ask10, bsize10, asize10
ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)
ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)
e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag
mlofi_W60_lvl10 = nansum(e_t[last_60_ticks])
```

**Physical meaning**: Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。W=60-tick (≈180s) 平滑后的本档主动方向流，正值预示短期上行。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；窗口内 NaN→0 后 sum (fast_features_batch:182)
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.062509, 0.176821, 0.226, 0.980382, 0.500325]; max=0.9804, mean=0.3892
**Cross-stock KS** by sym: [0.04481, 0.104075, 0.07794, 0.15093, 0.11656]; max=0.1509
**Cross-date PSI** train→val=0.0144, train→test=0.0593, by_bucket=[0.012133, 0.015185, 0.003252, 0.004635, 0.019501, 0.07437]
**Cross-date KS** train→val=0.0263, train→test=0.0551

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `kyle_inv_W50` — Kyle 逆冲击 W=50

- **English**: Kyle inverse-impact proxy at last tick over W=50
- **Family / type**: F2 / derived
- **dim_in_X**: 263  (derived idx in 216: 109)
- **In KS-drop 11**: no

**Formula**

$$
\sigma_W = \mathrm{std}(r_{t-W+1:t}),\;a = (|\text{amt}_t| + \epsilon)\,\sigma_W + \epsilon\;,\;\text{kyle\_inv}_W = \text{amt}_t / (\sqrt[3]{a} + \epsilon)
$$

- **Code location**: `fast_features_batch.py:310-321`

**Numpy pseudo**

```python
# r = log-return of midprice; amt = signed amount_delta
sigma_W = std(r[-50:])
a = |amt_last| * sigma_W + EPS
kyle_inv_W50 = amt_last / (cbrt(a) + EPS)
```

**Physical meaning**: Kyle (1985) λ 的逆方向变体：单位 ‘信息含量’ amount × σ 折算后归一化的金额——若 |amount| 足够大且 σ 较小（信息冲击型），信号值大；正号表示买方驱动。W=50 短期、W=100 中期。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；分母 +EPS、isfinite 兜底 →0
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.05108, 0.195125, 0.449403, 0.184513, 0.113933]; max=0.4494, mean=0.1988
**Cross-stock KS** by sym: [0.08474, 0.177005, 0.24542, 0.0956, 0.090345]; max=0.2454
**Cross-date PSI** train→val=0.0073, train→test=0.0572, by_bucket=[0.028955, 0.016142, 0.006013, 0.006414, 0.011558, 0.065551]
**Cross-date KS** train→val=0.0268, train→test=0.1109

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `kyle_inv_W100` — Kyle 逆冲击 W=100

- **English**: Kyle inverse-impact proxy at last tick over W=100
- **Family / type**: F2 / derived
- **dim_in_X**: 264  (derived idx in 216: 110)
- **In KS-drop 11**: no

**Formula**

$$
\sigma_W = \mathrm{std}(r_{t-W+1:t}),\;a = (|\text{amt}_t| + \epsilon)\,\sigma_W + \epsilon\;,\;\text{kyle\_inv}_W = \text{amt}_t / (\sqrt[3]{a} + \epsilon)
$$

- **Code location**: `fast_features_batch.py:310-321`

**Numpy pseudo**

```python
# r = log-return of midprice; amt = signed amount_delta
sigma_W = std(r[-100:])
a = |amt_last| * sigma_W + EPS
kyle_inv_W100 = amt_last / (cbrt(a) + EPS)
```

**Physical meaning**: Kyle (1985) λ 的逆方向变体：单位 ‘信息含量’ amount × σ 折算后归一化的金额——若 |amount| 足够大且 σ 较小（信息冲击型），信号值大；正号表示买方驱动。W=50 短期、W=100 中期。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；分母 +EPS、isfinite 兜底 →0
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.046359, 0.169364, 0.423321, 0.276104, 0.192184]; max=0.4233, mean=0.2215
**Cross-stock KS** by sym: [0.08435, 0.17095, 0.24451, 0.100285, 0.09434]; max=0.2445
**Cross-date PSI** train→val=0.0079, train→test=0.0610, by_bucket=[0.030055, 0.016574, 0.005876, 0.007191, 0.014456, 0.067764]
**Cross-date KS** train→val=0.0268, train→test=0.1108

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.05_lvl1` — OFI EWMA α=0.05 档1

- **English**: EWMA of rolling-W20 OFI level-1, alpha=0.05
- **Family / type**: F2 / derived
- **dim_in_X**: 265  (derived idx in 216: 111)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl1}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.05
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl1_series  # rolling-20 OFI 完整序列
y_t = 0.05*x_t + (1-0.05)*y_{t-1};  y_0=0.05*x_0
out = y[-1]
```

**Physical meaning**: 对档 1 的 20-tick OFI 序列做 EWMA(α=0.05) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈13.9 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.212898, 0.048343, 0.024702, 0.357173, 0.339004]; max=0.3572, mean=0.1964
**Cross-stock KS** by sym: [0.09563, 0.063455, 0.042675, 0.13519, 0.123335]; max=0.1352
**Cross-date PSI** train→val=0.0102, train→test=0.0172, by_bucket=[0.016069, 0.013409, 0.006031, 0.005951, 0.007858, 0.025134]
**Cross-date KS** train→val=0.0191, train→test=0.0406

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.1_lvl1` — OFI EWMA α=0.1 档1

- **English**: EWMA of rolling-W20 OFI level-1, alpha=0.1
- **Family / type**: F2 / derived
- **dim_in_X**: 266  (derived idx in 216: 112)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl1}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.1
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl1_series  # rolling-20 OFI 完整序列
y_t = 0.1*x_t + (1-0.1)*y_{t-1};  y_0=0.1*x_0
out = y[-1]
```

**Physical meaning**: 对档 1 的 20-tick OFI 序列做 EWMA(α=0.1) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈6.9 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.196967, 0.054409, 0.02273, 0.32023, 0.328507]; max=0.3285, mean=0.1846
**Cross-stock KS** by sym: [0.09108, 0.062495, 0.037845, 0.12756, 0.117535]; max=0.1276
**Cross-date PSI** train→val=0.0107, train→test=0.0186, by_bucket=[0.016397, 0.013495, 0.00497, 0.005305, 0.007651, 0.025415]
**Cross-date KS** train→val=0.0170, train→test=0.0417

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.3_lvl1` — OFI EWMA α=0.3 档1

- **English**: EWMA of rolling-W20 OFI level-1, alpha=0.3
- **Family / type**: F2 / derived
- **dim_in_X**: 267  (derived idx in 216: 113)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl1}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.3
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl1_series  # rolling-20 OFI 完整序列
y_t = 0.3*x_t + (1-0.3)*y_{t-1};  y_0=0.3*x_0
out = y[-1]
```

**Physical meaning**: 对档 1 的 20-tick OFI 序列做 EWMA(α=0.3) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈2.3 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.184303, 0.058134, 0.027854, 0.290714, 0.31987]; max=0.3199, mean=0.1762
**Cross-stock KS** by sym: [0.08699, 0.06286, 0.03733, 0.122805, 0.115715]; max=0.1228
**Cross-date PSI** train→val=0.0098, train→test=0.0186, by_bucket=[0.014533, 0.01329, 0.003396, 0.005553, 0.006215, 0.025317]
**Cross-date KS** train→val=0.0177, train→test=0.0427

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.5_lvl1` — OFI EWMA α=0.5 档1

- **English**: EWMA of rolling-W20 OFI level-1, alpha=0.5
- **Family / type**: F2 / derived
- **dim_in_X**: 268  (derived idx in 216: 114)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl1}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.5
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl1_series  # rolling-20 OFI 完整序列
y_t = 0.5*x_t + (1-0.5)*y_{t-1};  y_0=0.5*x_0
out = y[-1]
```

**Physical meaning**: 对档 1 的 20-tick OFI 序列做 EWMA(α=0.5) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈1.4 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.184533, 0.059638, 0.026669, 0.285223, 0.322035]; max=0.3220, mean=0.1756
**Cross-stock KS** by sym: [0.08693, 0.06228, 0.037405, 0.12135, 0.11544]; max=0.1213
**Cross-date PSI** train→val=0.0096, train→test=0.0190, by_bucket=[0.01452, 0.013669, 0.003084, 0.005886, 0.006114, 0.025309]
**Cross-date KS** train→val=0.0173, train→test=0.0414

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.05_lvl5` — OFI EWMA α=0.05 档5

- **English**: EWMA of rolling-W20 OFI level-5, alpha=0.05
- **Family / type**: F2 / derived
- **dim_in_X**: 269  (derived idx in 216: 115)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl5}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.05
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl5_series  # rolling-20 OFI 完整序列
y_t = 0.05*x_t + (1-0.05)*y_{t-1};  y_0=0.05*x_0
out = y[-1]
```

**Physical meaning**: 对档 5 的 20-tick OFI 序列做 EWMA(α=0.05) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈13.9 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.105076, 0.11013, 0.079284, 0.339038, 0.389365]; max=0.3894, mean=0.2046
**Cross-stock KS** by sym: [0.06053, 0.081285, 0.055815, 0.10389, 0.12074]; max=0.1207
**Cross-date PSI** train→val=0.0176, train→test=0.0452, by_bucket=[0.013475, 0.011502, 0.007597, 0.003667, 0.013925, 0.057184]
**Cross-date KS** train→val=0.0276, train→test=0.0494

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.1_lvl5` — OFI EWMA α=0.1 档5

- **English**: EWMA of rolling-W20 OFI level-5, alpha=0.1
- **Family / type**: F2 / derived
- **dim_in_X**: 270  (derived idx in 216: 116)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl5}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.1
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl5_series  # rolling-20 OFI 完整序列
y_t = 0.1*x_t + (1-0.1)*y_{t-1};  y_0=0.1*x_0
out = y[-1]
```

**Physical meaning**: 对档 5 的 20-tick OFI 序列做 EWMA(α=0.1) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈6.9 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.07175, 0.132968, 0.096475, 0.356593, 0.337624]; max=0.3566, mean=0.1991
**Cross-stock KS** by sym: [0.05074, 0.08686, 0.06143, 0.0859, 0.108765]; max=0.1088
**Cross-date PSI** train→val=0.0185, train→test=0.0482, by_bucket=[0.012415, 0.010856, 0.005772, 0.004955, 0.015845, 0.057534]
**Cross-date KS** train→val=0.0256, train→test=0.0485

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.3_lvl5` — OFI EWMA α=0.3 档5

- **English**: EWMA of rolling-W20 OFI level-5, alpha=0.3
- **Family / type**: F2 / derived
- **dim_in_X**: 271  (derived idx in 216: 117)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl5}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.3
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl5_series  # rolling-20 OFI 完整序列
y_t = 0.3*x_t + (1-0.3)*y_{t-1};  y_0=0.3*x_0
out = y[-1]
```

**Physical meaning**: 对档 5 的 20-tick OFI 序列做 EWMA(α=0.3) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈2.3 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.051912, 0.157146, 0.137364, 0.449913, 0.312723]; max=0.4499, mean=0.2218
**Cross-stock KS** by sym: [0.04163, 0.09423, 0.06993, 0.10839, 0.09738]; max=0.1084
**Cross-date PSI** train→val=0.0141, train→test=0.0478, by_bucket=[0.009727, 0.011174, 0.003951, 0.004828, 0.013372, 0.05554]
**Cross-date KS** train→val=0.0251, train→test=0.0461

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.5_lvl5` — OFI EWMA α=0.5 档5

- **English**: EWMA of rolling-W20 OFI level-5, alpha=0.5
- **Family / type**: F2 / derived
- **dim_in_X**: 272  (derived idx in 216: 118)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl5}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.5
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl5_series  # rolling-20 OFI 完整序列
y_t = 0.5*x_t + (1-0.5)*y_{t-1};  y_0=0.5*x_0
out = y[-1]
```

**Physical meaning**: 对档 5 的 20-tick OFI 序列做 EWMA(α=0.5) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈1.4 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.056876, 0.153072, 0.167851, 0.476459, 0.311089]; max=0.4765, mean=0.2331
**Cross-stock KS** by sym: [0.04048, 0.094575, 0.07594, 0.119855, 0.09381]; max=0.1199
**Cross-date PSI** train→val=0.0110, train→test=0.0446, by_bucket=[0.009545, 0.011476, 0.003242, 0.004442, 0.0116, 0.05326]
**Cross-date KS** train→val=0.0219, train→test=0.0451

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.05_lvl10` — OFI EWMA α=0.05 档10

- **English**: EWMA of rolling-W20 OFI level-10, alpha=0.05
- **Family / type**: F2 / derived
- **dim_in_X**: 273  (derived idx in 216: 119)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl10}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.05
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl10_series  # rolling-20 OFI 完整序列
y_t = 0.05*x_t + (1-0.05)*y_{t-1};  y_0=0.05*x_0
out = y[-1]
```

**Physical meaning**: 对档 10 的 20-tick OFI 序列做 EWMA(α=0.05) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈13.9 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.054392, 0.177774, 0.101974, 0.382387, 0.382253]; max=0.3824, mean=0.2198
**Cross-stock KS** by sym: [0.042465, 0.103135, 0.063155, 0.093285, 0.118805]; max=0.1188
**Cross-date PSI** train→val=0.0211, train→test=0.0701, by_bucket=[0.012031, 0.017413, 0.004764, 0.004396, 0.023199, 0.090033]
**Cross-date KS** train→val=0.0270, train→test=0.0648

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.1_lvl10` — OFI EWMA α=0.1 档10

- **English**: EWMA of rolling-W20 OFI level-10, alpha=0.1
- **Family / type**: F2 / derived
- **dim_in_X**: 274  (derived idx in 216: 120)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl10}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.1
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl10_series  # rolling-20 OFI 完整序列
y_t = 0.1*x_t + (1-0.1)*y_{t-1};  y_0=0.1*x_0
out = y[-1]
```

**Physical meaning**: 对档 10 的 20-tick OFI 序列做 EWMA(α=0.1) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈6.9 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.042786, 0.211618, 0.140486, 0.500747, 0.351055]; max=0.5007, mean=0.2493
**Cross-stock KS** by sym: [0.03612, 0.106785, 0.07044, 0.12185, 0.109605]; max=0.1218
**Cross-date PSI** train→val=0.0209, train→test=0.0692, by_bucket=[0.012528, 0.016636, 0.004089, 0.004736, 0.024452, 0.085197]
**Cross-date KS** train→val=0.0294, train→test=0.0606

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.3_lvl10` — OFI EWMA α=0.3 档10

- **English**: EWMA of rolling-W20 OFI level-10, alpha=0.3
- **Family / type**: F2 / derived
- **dim_in_X**: 275  (derived idx in 216: 121)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl10}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.3
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl10_series  # rolling-20 OFI 完整序列
y_t = 0.3*x_t + (1-0.3)*y_{t-1};  y_0=0.3*x_0
out = y[-1]
```

**Physical meaning**: 对档 10 的 20-tick OFI 序列做 EWMA(α=0.3) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈2.3 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.029743, 0.230096, 0.235596, 0.75993, 0.385542]; max=0.7599, mean=0.3282
**Cross-stock KS** by sym: [0.03038, 0.109325, 0.08384, 0.16681, 0.09835]; max=0.1668
**Cross-date PSI** train→val=0.0190, train→test=0.0635, by_bucket=[0.01006, 0.015626, 0.003245, 0.004283, 0.021949, 0.074399]
**Cross-date KS** train→val=0.0304, train→test=0.0545

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ewma_ofi_a0.5_lvl10` — OFI EWMA α=0.5 档10

- **English**: EWMA of rolling-W20 OFI level-10, alpha=0.5
- **Family / type**: F2 / derived
- **dim_in_X**: 276  (derived idx in 216: 122)
- **In KS-drop 11**: no

**Formula**

$$
x_t = \text{mlofi\_W20\_lvl10}_t,\;\text{ewma\_ofi}^{(k)}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{ewma\_ofi}^{(k)}_\alpha(t-1)\;,\;\alpha=0.5
$$

- **Code location**: `fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)`

**Numpy pseudo**

```python
x = mlofi_W20_lvl10_series  # rolling-20 OFI 完整序列
y_t = 0.5*x_t + (1-0.5)*y_{t-1};  y_0=0.5*x_0
out = y[-1]
```

**Physical meaning**: 对档 10 的 20-tick OFI 序列做 EWMA(α=0.5) 平滑：α 大→更看重近 1-2 tick (短记忆)，α 小→长记忆（半衰期≈1.4 ticks）。同方向（买推升）OFI 的指数平均，比 raw mlofi 更平滑也更鲁棒。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.032696, 0.218646, 0.308391, 0.872474, 0.4503]; max=0.8725, mean=0.3765
**Cross-stock KS** by sym: [0.029025, 0.10947, 0.09373, 0.18131, 0.094905]; max=0.1813
**Cross-date PSI** train→val=0.0153, train→test=0.0593, by_bucket=[0.009595, 0.016097, 0.003125, 0.003937, 0.0189, 0.070597]
**Cross-date KS** train→val=0.0272, train→test=0.0535

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `kyle_lam_W50` — Kyle λ 滚动 W=50

- **English**: Rolling Kyle lambda over W=50 ticks
- **Family / type**: F2 / derived
- **dim_in_X**: 330  (derived idx in 216: 176)
- **In KS-drop 11**: YES — 训练时被剔除

**Formula**

$$
x_t = \mathrm{sign}(\Delta m_t)\sqrt{|\text{amt}_t|+\epsilon},\;y_t = \Delta m_t,\;\lambda_W = \frac{\mathrm{Cov}_W(x,y)}{\mathrm{Var}_W(x)+\epsilon}, \text{clip}[-1,1]
$$

- **Code location**: `fast_features_batch.py:431-453`

**Numpy pseudo**

```python
dmid = diff(midprice1)
x = sign(dmid) * sqrt(|amount_delta| + EPS)
y = dmid
lam = cov(x[-50:], y[-50:]) / (var(x[-50:]) + EPS)
kyle_lam_W50 = clip(lam, -1, 1)
```

**Physical meaning**: 经典 Kyle (1985) 价格冲击系数滚动估计：单位 ‘签名根成交量’ 引致的 mid-price 变化。λ 大表示 LOB 浅、价格易被推动。**KS-drop 命中**：训练集分布与 val/test 偏差过大被剔除。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；clip 到 [-1, 1]; isfinite 兜底 →0
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.852448, 1.722595, 6.561344, 4.540286, 1.387271]; max=6.5613, mean=3.0128
**Cross-stock KS** by sym: [0.30898, 0.42952, 0.554245, 0.379315, 0.268575]; max=0.5542
**Cross-date PSI** train→val=0.0175, train→test=0.0307, by_bucket=[0.031847, 0.008235, 0.012708, 0.042548, 0.019774, 0.032019]
**Cross-date KS** train→val=0.0695, train→test=0.0899

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

*Notes*: KS-drop 11 之一

### `kyle_lam_W100` — Kyle λ 滚动 W=100

- **English**: Rolling Kyle lambda over W=100 ticks
- **Family / type**: F2 / derived
- **dim_in_X**: 331  (derived idx in 216: 177)
- **In KS-drop 11**: YES — 训练时被剔除

**Formula**

$$
x_t = \mathrm{sign}(\Delta m_t)\sqrt{|\text{amt}_t|+\epsilon},\;y_t = \Delta m_t,\;\lambda_W = \frac{\mathrm{Cov}_W(x,y)}{\mathrm{Var}_W(x)+\epsilon}, \text{clip}[-1,1]
$$

- **Code location**: `fast_features_batch.py:431-453`

**Numpy pseudo**

```python
dmid = diff(midprice1)
x = sign(dmid) * sqrt(|amount_delta| + EPS)
y = dmid
lam = cov(x[-100:], y[-100:]) / (var(x[-100:]) + EPS)
kyle_lam_W100 = clip(lam, -1, 1)
```

**Physical meaning**: 经典 Kyle (1985) 价格冲击系数滚动估计：单位 ‘签名根成交量’ 引致的 mid-price 变化。λ 大表示 LOB 浅、价格易被推动。**KS-drop 命中**：训练集分布与 val/test 偏差过大被剔除。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；clip 到 [-1, 1]; isfinite 兜底 →0
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [3.106111, 2.195957, 10.515249, 5.427535, 3.496565]; max=10.5152, mean=4.9483
**Cross-stock KS** by sym: [0.319635, 0.4131, 0.62782, 0.41338, 0.33002]; max=0.6278
**Cross-date PSI** train→val=0.0291, train→test=0.0423, by_bucket=[0.041103, 0.008322, 0.021993, 0.061625, 0.035005, 0.046909]
**Cross-date KS** train→val=0.0541, train→test=0.0727

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

*Notes*: KS-drop 11 之一

### `ofi_tox_lvl1_W20` — OFI 毒性 (corr) 档1 W=20

- **English**: OFI toxicity = corr(OFI_l1, Δmid) over last W=20 ticks
- **Family / type**: F2 / derived
- **dim_in_X**: 353  (derived idx in 216: 199)
- **In KS-drop 11**: no

**Formula**

$$
\text{ofi\_tox}_{l=1,W=20} = \mathrm{corr}_W(\text{OFI}^{(lvl)}_t,\;\Delta m_t),\;\text{clip}[-1,1]
$$

- **Code location**: `stage5_features.py / fast_features_batch.py:634-669`

**Numpy pseudo**

```python
ofi_l1 = (Cont OFI per-tick e at level 1)
dmid = diff(midprice1)
ofi_tox_lvl1_W20 = pearson_corr(ofi_l1[-20:], dmid[-20:])
```

**Physical meaning**: 档 1 的 OFI 与 Δmid 在最近 W=20 上的 Pearson 相关——近期订单流方向与价格变动是否一致（即 ‘有信息’ 程度）。高正相关 = 订单流推得动价格 = 信息毒性高。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom +EPS、clip[-1,1]、isfinite 兜底
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.010511, 0.096317, 0.216266, 0.846727, 0.200116]; max=0.8467, mean=0.2740
**Cross-stock KS** by sym: [0.021265, 0.07002, 0.12585, 0.254535, 0.10115]; max=0.2545
**Cross-date PSI** train→val=0.0172, train→test=0.0320, by_bucket=[0.001705, 0.002414, 0.000271, 0.001995, 0.016329, 0.03785]
**Cross-date KS** train→val=0.0371, train→test=0.0634

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ofi_tox_lvl1_W50` — OFI 毒性 (corr) 档1 W=50

- **English**: OFI toxicity = corr(OFI_l1, Δmid) over last W=50 ticks
- **Family / type**: F2 / derived
- **dim_in_X**: 354  (derived idx in 216: 200)
- **In KS-drop 11**: no

**Formula**

$$
\text{ofi\_tox}_{l=1,W=50} = \mathrm{corr}_W(\text{OFI}^{(lvl)}_t,\;\Delta m_t),\;\text{clip}[-1,1]
$$

- **Code location**: `stage5_features.py / fast_features_batch.py:634-669`

**Numpy pseudo**

```python
ofi_l1 = (Cont OFI per-tick e at level 1)
dmid = diff(midprice1)
ofi_tox_lvl1_W50 = pearson_corr(ofi_l1[-50:], dmid[-50:])
```

**Physical meaning**: 档 1 的 OFI 与 Δmid 在最近 W=50 上的 Pearson 相关——近期订单流方向与价格变动是否一致（即 ‘有信息’ 程度）。高正相关 = 订单流推得动价格 = 信息毒性高。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom +EPS、clip[-1,1]、isfinite 兜底
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.03257, 0.160976, 0.21795, 0.71941, 0.204356]; max=0.7194, mean=0.2671
**Cross-stock KS** by sym: [0.044275, 0.05692, 0.094805, 0.20228, 0.05985]; max=0.2023
**Cross-date PSI** train→val=0.0349, train→test=0.0571, by_bucket=[0.002635, 0.003456, 0.002355, 0.006632, 0.033611, 0.064991]
**Cross-date KS** train→val=0.0396, train→test=0.0668

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ofi_tox_lvl5_W20` — OFI 毒性 (corr) 档5 W=20

- **English**: OFI toxicity = corr(OFI_l5, Δmid) over last W=20 ticks
- **Family / type**: F2 / derived
- **dim_in_X**: 355  (derived idx in 216: 201)
- **In KS-drop 11**: no

**Formula**

$$
\text{ofi\_tox}_{l=5,W=20} = \mathrm{corr}_W(\text{OFI}^{(lvl)}_t,\;\Delta m_t),\;\text{clip}[-1,1]
$$

- **Code location**: `stage5_features.py / fast_features_batch.py:634-669`

**Numpy pseudo**

```python
ofi_l5 = (Cont OFI per-tick e at level 5)
dmid = diff(midprice1)
ofi_tox_lvl5_W20 = pearson_corr(ofi_l5[-20:], dmid[-20:])
```

**Physical meaning**: 档 5 的 OFI 与 Δmid 在最近 W=20 上的 Pearson 相关——近期订单流方向与价格变动是否一致（即 ‘有信息’ 程度）。高正相关 = 订单流推得动价格 = 信息毒性高。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom +EPS、clip[-1,1]、isfinite 兜底
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.26522, 0.212989, 0.6118, 1.490079, 0.600099]; max=1.4901, mean=0.6360
**Cross-stock KS** by sym: [0.16473, 0.13125, 0.11802, 0.29615, 0.117715]; max=0.2962
**Cross-date PSI** train→val=0.0267, train→test=0.0379, by_bucket=[0.00947, 0.003453, 0.001909, 0.006017, 0.026542, 0.040065]
**Cross-date KS** train→val=0.0336, train→test=0.0500

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**

### `ofi_tox_lvl5_W50` — OFI 毒性 (corr) 档5 W=50

- **English**: OFI toxicity = corr(OFI_l5, Δmid) over last W=50 ticks
- **Family / type**: F2 / derived
- **dim_in_X**: 356  (derived idx in 216: 202)
- **In KS-drop 11**: no

**Formula**

$$
\text{ofi\_tox}_{l=5,W=50} = \mathrm{corr}_W(\text{OFI}^{(lvl)}_t,\;\Delta m_t),\;\text{clip}[-1,1]
$$

- **Code location**: `stage5_features.py / fast_features_batch.py:634-669`

**Numpy pseudo**

```python
ofi_l5 = (Cont OFI per-tick e at level 5)
dmid = diff(midprice1)
ofi_tox_lvl5_W50 = pearson_corr(ofi_l5[-50:], dmid[-50:])
```

**Physical meaning**: 档 5 的 OFI 与 Δmid 在最近 W=50 上的 Pearson 相关——近期订单流方向与价格变动是否一致（即 ‘有信息’ 程度）。高正相关 = 订单流推得动价格 = 信息毒性高。

**NaN handling**:
  - 公式层: fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换；denom +EPS、clip[-1,1]、isfinite 兜底
  - NN 管线: wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)
  - LGB 管线: 完全不动，但 derived 已经无 NaN，所以 missing routing 不触发

**NaN%**: train=0.0000% (0), val=0.0000% (0), test=0.0000% (0)

**Cross-stock PSI** by sym: [0.33471, 0.285039, 0.935763, 1.580966, 0.887356]; max=1.5810, mean=0.8048
**Cross-stock KS** by sym: [0.221495, 0.1272, 0.168305, 0.249395, 0.16106]; max=0.2494
**Cross-date PSI** train→val=0.0422, train→test=0.0569, by_bucket=[0.017345, 0.003987, 0.006029, 0.018754, 0.040599, 0.061353]
**Cross-date KS** train→val=0.0404, train→test=0.0508

**Verdict**: cross_stock=**strong_drift**, cross_date=**stable**
