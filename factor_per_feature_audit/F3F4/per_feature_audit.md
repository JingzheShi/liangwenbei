# F3+F4 Per-Feature Audit

> 良文杯（中间价方向预测）feature audit. 74 个 F3+F4 features.
> 公榜+35.64 / 私榜#1 提交的 Predictor 用了所有这 74 个 feature。

## 全局指标

- 特征总数: **74** (F3=45, F4=29)
- 跨 stock 最大 PSI: **1.1444**
- 跨 date 最大 PSI: **0.1463**
- train 平均 NaN%: **0.0%**
- KS-drop 11 命中: **1**

## Top-4 intst 特征稳定性（命根）
| feature | PSI sym_max | PSI by sym | PSI tr→va | PSI tr→te | KS tr→te | verdict |
|---|---|---|---|---|---|---|
| `ma_intst` | 0.225 | [0.0694, 0.0211, 0.225, 0.0367, 0.0219] | 0.0024 | 0.0199 | 0.0879 | stock=mild_drift / date=stable |
| `mb_intst` | 0.2555 | [0.0785, 0.0346, 0.2555, 0.045, 0.036] | 0.0085 | 0.0332 | 0.0979 | stock=strong_drift / date=stable |
| `la_intst` | 0.2411 | [0.1288, 0.0526, 0.2411, 0.0152, 0.0433] | 0.0039 | 0.0171 | 0.0917 | stock=mild_drift / date=stable |
| `lb_intst` | 0.2732 | [0.1257, 0.0517, 0.2732, 0.0511, 0.0885] | 0.0109 | 0.0259 | 0.0909 | stock=strong_drift / date=stable |

## Top-5 最不稳定（跨 stock）
- `amount_delta` (family=F4) — PSI_max=1.1444
- `roll_eff_spr_ratio_W100` (family=F4) — PSI_max=1.0539
- `roll_eff_spr_ratio_W30` (family=F4) — PSI_max=0.8758
- `roll_eff_spr_ratio_W50` (family=F4) — PSI_max=0.8519
- `ewma_a0.05_la_intst` (family=F3) — PSI_max=0.7895

## Top-5 最不稳定（跨 date）
- `roll_eff_spr_ratio_W100` (family=F4) — PSI_max=0.1463
- `rv_w50` (family=F4) — PSI_max=0.1392
- `ewma_a0.05_mb_intst` (family=F3) — PSI_max=0.1321
- `ewma_a0.05_ma_intst` (family=F3) — PSI_max=0.1267
- `rv_w20` (family=F4) — PSI_max=0.1227

## Top-5 train NaN% 最高
- `lb_intst` (family=F3) — NaN%=0.0
- `la_intst` (family=F3) — NaN%=0.0
- `mb_intst` (family=F3) — NaN%=0.0
- `ma_intst` (family=F3) — NaN%=0.0
- `cb_intst` (family=F3) — NaN%=0.0

## KS-drop 11 命中
- `roll_eff_spr_ratio_W100` — PSI tr→te = 0.1446, KS tr→te = 0.1747

---

## F3 — 45 features

### `lb_intst` (raw)
- 中文名: 限价买强度
- 英文: Limit-Bid Order-Flow Intensity (count or amount)
- X 列号 (0-based): 50
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{intst})} = N^{(\cdot)}_t \quad \text{or aggregated amount of this order type at tick t}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
lb_intst = raw_X[:, col_idx["lb_intst"]]  # 3s tick aggregation
```
- 物理含义: intst = 该类委托/成交在 3s tick 内的频次或量；信息密度最高。lb/la 反映报价新增，mb/ma 反映主动成交（信息含量最强：Top-4 都是 mb_intst/ma_intst/la_intst/lb_intst），cb/ca 反映撤单博弈。intst > ind > acc：强度保留幅度信息，二值化 (ind) 损失幅度而只保留方向，二阶差分 (acc) 是噪声/突变检测，信噪比通常最低。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2732 / mean=0.118 / by_sym=[0.1257, 0.0517, 0.2732, 0.0511, 0.0885]
- 跨 stock KS: max=0.3769 / by_sym=[0.1782, 0.1415, 0.3769, 0.1081, 0.0827]
- 跨 date PSI: tr→va=0.0109 / tr→te=0.0259
- 跨 date PSI by bucket: [('tr_d0_19', 0.0111), ('tr_d20_39', 0.0082), ('tr_d40_59', 0.0017), ('tr_d60_79', 0.001), ('va_d80_95', 0.0069), ('te_d96_119', 0.0236)]
- 跨 date KS: tr→va=0.0404 / tr→te=0.0909
- 结论: cross_stock=strong_drift / cross_date=stable

### `la_intst` (raw)
- 中文名: 限价卖强度
- 英文: Limit-Ask Order-Flow Intensity (count or amount)
- X 列号 (0-based): 51
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{intst})} = N^{(\cdot)}_t \quad \text{or aggregated amount of this order type at tick t}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
la_intst = raw_X[:, col_idx["la_intst"]]  # 3s tick aggregation
```
- 物理含义: intst = 该类委托/成交在 3s tick 内的频次或量；信息密度最高。lb/la 反映报价新增，mb/ma 反映主动成交（信息含量最强：Top-4 都是 mb_intst/ma_intst/la_intst/lb_intst），cb/ca 反映撤单博弈。intst > ind > acc：强度保留幅度信息，二值化 (ind) 损失幅度而只保留方向，二阶差分 (acc) 是噪声/突变检测，信噪比通常最低。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2411 / mean=0.0962 / by_sym=[0.1288, 0.0526, 0.2411, 0.0152, 0.0433]
- 跨 stock KS: max=0.3066 / by_sym=[0.2062, 0.1158, 0.3066, 0.0738, 0.1102]
- 跨 date PSI: tr→va=0.0039 / tr→te=0.0171
- 跨 date PSI by bucket: [('tr_d0_19', 0.0087), ('tr_d20_39', 0.0082), ('tr_d40_59', 0.0014), ('tr_d60_79', 0.0028), ('va_d80_95', 0.0023), ('te_d96_119', 0.0185)]
- 跨 date KS: tr→va=0.0369 / tr→te=0.0917
- 结论: cross_stock=mild_drift / cross_date=stable

### `mb_intst` (raw)
- 中文名: 市价买强度
- 英文: Market-Bid Order-Flow Intensity (count or amount)
- X 列号 (0-based): 52
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{intst})} = N^{(\cdot)}_t \quad \text{or aggregated amount of this order type at tick t}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
mb_intst = raw_X[:, col_idx["mb_intst"]]  # 3s tick aggregation
```
- 物理含义: intst = 该类委托/成交在 3s tick 内的频次或量；信息密度最高。lb/la 反映报价新增，mb/ma 反映主动成交（信息含量最强：Top-4 都是 mb_intst/ma_intst/la_intst/lb_intst），cb/ca 反映撤单博弈。intst > ind > acc：强度保留幅度信息，二值化 (ind) 损失幅度而只保留方向，二阶差分 (acc) 是噪声/突变检测，信噪比通常最低。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2555 / mean=0.0899 / by_sym=[0.0785, 0.0346, 0.2555, 0.045, 0.036]
- 跨 stock KS: max=0.3617 / by_sym=[0.1563, 0.1425, 0.3617, 0.1288, 0.0818]
- 跨 date PSI: tr→va=0.0085 / tr→te=0.0332
- 跨 date PSI by bucket: [('tr_d0_19', 0.0154), ('tr_d20_39', 0.0086), ('tr_d40_59', 0.0022), ('tr_d60_79', 0.0014), ('va_d80_95', 0.0067), ('te_d96_119', 0.036)]
- 跨 date KS: tr→va=0.0461 / tr→te=0.0979
- 结论: cross_stock=strong_drift / cross_date=stable

### `ma_intst` (raw)
- 中文名: 市价卖强度
- 英文: Market-Ask Order-Flow Intensity (count or amount)
- X 列号 (0-based): 53
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{intst})} = N^{(\cdot)}_t \quad \text{or aggregated amount of this order type at tick t}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
ma_intst = raw_X[:, col_idx["ma_intst"]]  # 3s tick aggregation
```
- 物理含义: intst = 该类委托/成交在 3s tick 内的频次或量；信息密度最高。lb/la 反映报价新增，mb/ma 反映主动成交（信息含量最强：Top-4 都是 mb_intst/ma_intst/la_intst/lb_intst），cb/ca 反映撤单博弈。intst > ind > acc：强度保留幅度信息，二值化 (ind) 损失幅度而只保留方向，二阶差分 (acc) 是噪声/突变检测，信噪比通常最低。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.225 / mean=0.0748 / by_sym=[0.0694, 0.0211, 0.225, 0.0367, 0.0219]
- 跨 stock KS: max=0.3234 / by_sym=[0.1546, 0.0972, 0.3234, 0.1326, 0.0748]
- 跨 date PSI: tr→va=0.0024 / tr→te=0.0199
- 跨 date PSI by bucket: [('tr_d0_19', 0.012), ('tr_d20_39', 0.0092), ('tr_d40_59', 0.002), ('tr_d60_79', 0.0007), ('va_d80_95', 0.0037), ('te_d96_119', 0.0233)]
- 跨 date KS: tr→va=0.0291 / tr→te=0.0879
- 结论: cross_stock=mild_drift / cross_date=stable

### `cb_intst` (raw)
- 中文名: 撤买强度
- 英文: Cancel-Bid Order-Flow Intensity (count or amount)
- X 列号 (0-based): 54
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{intst})} = N^{(\cdot)}_t \quad \text{or aggregated amount of this order type at tick t}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
cb_intst = raw_X[:, col_idx["cb_intst"]]  # 3s tick aggregation
```
- 物理含义: intst = 该类委托/成交在 3s tick 内的频次或量；信息密度最高。lb/la 反映报价新增，mb/ma 反映主动成交（信息含量最强：Top-4 都是 mb_intst/ma_intst/la_intst/lb_intst），cb/ca 反映撤单博弈。intst > ind > acc：强度保留幅度信息，二值化 (ind) 损失幅度而只保留方向，二阶差分 (acc) 是噪声/突变检测，信噪比通常最低。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4264 / mean=0.1378 / by_sym=[0.0829, 0.103, 0.4264, 0.0499, 0.0267]
- 跨 stock KS: max=0.3959 / by_sym=[0.1448, 0.1596, 0.3959, 0.1093, 0.0886]
- 跨 date PSI: tr→va=0.0031 / tr→te=0.012
- 跨 date PSI by bucket: [('tr_d0_19', 0.0066), ('tr_d20_39', 0.0043), ('tr_d40_59', 0.0021), ('tr_d60_79', 0.0011), ('va_d80_95', 0.0033), ('te_d96_119', 0.0143)]
- 跨 date KS: tr→va=0.0248 / tr→te=0.0696
- 结论: cross_stock=strong_drift / cross_date=stable

### `ca_intst` (raw)
- 中文名: 撤卖强度
- 英文: Cancel-Ask Order-Flow Intensity (count or amount)
- X 列号 (0-based): 55
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{intst})} = N^{(\cdot)}_t \quad \text{or aggregated amount of this order type at tick t}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
ca_intst = raw_X[:, col_idx["ca_intst"]]  # 3s tick aggregation
```
- 物理含义: intst = 该类委托/成交在 3s tick 内的频次或量；信息密度最高。lb/la 反映报价新增，mb/ma 反映主动成交（信息含量最强：Top-4 都是 mb_intst/ma_intst/la_intst/lb_intst），cb/ca 反映撤单博弈。intst > ind > acc：强度保留幅度信息，二值化 (ind) 损失幅度而只保留方向，二阶差分 (acc) 是噪声/突变检测，信噪比通常最低。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2824 / mean=0.0985 / by_sym=[0.0651, 0.0827, 0.2824, 0.0382, 0.0239]
- 跨 stock KS: max=0.3228 / by_sym=[0.1222, 0.1473, 0.3228, 0.0913, 0.0599]
- 跨 date PSI: tr→va=0.0009 / tr→te=0.0131
- 跨 date PSI by bucket: [('tr_d0_19', 0.0106), ('tr_d20_39', 0.0071), ('tr_d40_59', 0.0011), ('tr_d60_79', 0.0004), ('va_d80_95', 0.001), ('te_d96_119', 0.0157)]
- 跨 date KS: tr→va=0.0234 / tr→te=0.0748
- 结论: cross_stock=strong_drift / cross_date=stable

### `lb_ind` (raw)
- 中文名: 限价买二值指示
- 英文: Limit-Bid Binary Indicator (presence)
- X 列号 (0-based): 56
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{ind})} = \mathbf{1}\{N^{(\cdot)}_t > 0\}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
lb_ind = raw_X[:, col_idx["lb_ind"]]  # 3s tick aggregation
```
- 物理含义: ind = 是否发生此类委托的二值指示。把幅度信息压成 {0,1}，信号粒度变粗；在 LGB 树模型里 ind 几乎可被 intst 完全替代（intst==0 ↔ ind==0），所以 ind 重要性远低于 intst。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0169 / mean=0.0066 / by_sym=[0.0007, 0.0035, 0.0169, 0.0095, 0.0023]
- 跨 stock KS: max=0.072 / by_sym=[0.0139, 0.0313, 0.072, 0.0514, 0.026]
- 跨 date PSI: tr→va=0.0007 / tr→te=0.0026
- 跨 date PSI by bucket: [('tr_d0_19', 0.0008), ('tr_d20_39', 0.0006), ('tr_d40_59', 0.0), ('tr_d60_79', 0.0002), ('va_d80_95', 0.0004), ('te_d96_119', 0.0017)]
- 跨 date KS: tr→va=0.0107 / tr→te=0.0213
- 结论: cross_stock=stable / cross_date=stable

### `la_ind` (raw)
- 中文名: 限价卖二值指示
- 英文: Limit-Ask Binary Indicator (presence)
- X 列号 (0-based): 57
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{ind})} = \mathbf{1}\{N^{(\cdot)}_t > 0\}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
la_ind = raw_X[:, col_idx["la_ind"]]  # 3s tick aggregation
```
- 物理含义: ind = 是否发生此类委托的二值指示。把幅度信息压成 {0,1}，信号粒度变粗；在 LGB 树模型里 ind 几乎可被 intst 完全替代（intst==0 ↔ ind==0），所以 ind 重要性远低于 intst。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0151 / mean=0.008 / by_sym=[0.0057, 0.0022, 0.0151, 0.0092, 0.0078]
- 跨 stock KS: max=0.0663 / by_sym=[0.0392, 0.0245, 0.0663, 0.0486, 0.0469]
- 跨 date PSI: tr→va=0.0007 / tr→te=0.0037
- 跨 date PSI by bucket: [('tr_d0_19', 0.0011), ('tr_d20_39', 0.0003), ('tr_d40_59', 0.0), ('tr_d60_79', 0.0001), ('va_d80_95', 0.0007), ('te_d96_119', 0.0034)]
- 跨 date KS: tr→va=0.013 / tr→te=0.0265
- 结论: cross_stock=stable / cross_date=stable

### `mb_ind` (raw)
- 中文名: 市价买二值指示
- 英文: Market-Bid Binary Indicator (presence)
- X 列号 (0-based): 58
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{ind})} = \mathbf{1}\{N^{(\cdot)}_t > 0\}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
mb_ind = raw_X[:, col_idx["mb_ind"]]  # 3s tick aggregation
```
- 物理含义: ind = 是否发生此类委托的二值指示。把幅度信息压成 {0,1}，信号粒度变粗；在 LGB 树模型里 ind 几乎可被 intst 完全替代（intst==0 ↔ ind==0），所以 ind 重要性远低于 intst。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0186 / mean=0.0087 / by_sym=[0.0023, 0.005, 0.0186, 0.0108, 0.0067]
- 跨 stock KS: max=0.071 / by_sym=[0.0243, 0.036, 0.071, 0.0505, 0.0408]
- 跨 date PSI: tr→va=0.0008 / tr→te=0.0031
- 跨 date PSI by bucket: [('tr_d0_19', 0.0005), ('tr_d20_39', 0.0005), ('tr_d40_59', 0.0), ('tr_d60_79', 0.0), ('va_d80_95', 0.0004), ('te_d96_119', 0.0033)]
- 跨 date KS: tr→va=0.0117 / tr→te=0.0221
- 结论: cross_stock=stable / cross_date=stable

### `ma_ind` (raw)
- 中文名: 市价卖二值指示
- 英文: Market-Ask Binary Indicator (presence)
- X 列号 (0-based): 59
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{ind})} = \mathbf{1}\{N^{(\cdot)}_t > 0\}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
ma_ind = raw_X[:, col_idx["ma_ind"]]  # 3s tick aggregation
```
- 物理含义: ind = 是否发生此类委托的二值指示。把幅度信息压成 {0,1}，信号粒度变粗；在 LGB 树模型里 ind 几乎可被 intst 完全替代（intst==0 ↔ ind==0），所以 ind 重要性远低于 intst。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0121 / mean=0.0072 / by_sym=[0.0038, 0.0012, 0.0115, 0.0121, 0.0072]
- 跨 stock KS: max=0.0557 / by_sym=[0.0309, 0.0171, 0.0557, 0.0528, 0.0427]
- 跨 date PSI: tr→va=0.0 / tr→te=0.0018
- 跨 date PSI by bucket: [('tr_d0_19', 0.0006), ('tr_d20_39', 0.0005), ('tr_d40_59', 0.0), ('tr_d60_79', 0.0), ('va_d80_95', 0.0), ('te_d96_119', 0.0023)]
- 跨 date KS: tr→va=0.0022 / tr→te=0.0175
- 结论: cross_stock=stable / cross_date=stable

### `cb_ind` (raw)
- 中文名: 撤买二值指示
- 英文: Cancel-Bid Binary Indicator (presence)
- X 列号 (0-based): 60
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{ind})} = \mathbf{1}\{N^{(\cdot)}_t > 0\}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
cb_ind = raw_X[:, col_idx["cb_ind"]]  # 3s tick aggregation
```
- 物理含义: ind = 是否发生此类委托的二值指示。把幅度信息压成 {0,1}，信号粒度变粗；在 LGB 树模型里 ind 几乎可被 intst 完全替代（intst==0 ↔ ind==0），所以 ind 重要性远低于 intst。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0232 / mean=0.0089 / by_sym=[0.0009, 0.0097, 0.0232, 0.0083, 0.0025]
- 跨 stock KS: max=0.0802 / by_sym=[0.0152, 0.0474, 0.0802, 0.0429, 0.0256]
- 跨 date PSI: tr→va=0.0003 / tr→te=0.0011
- 跨 date PSI by bucket: [('tr_d0_19', 0.0009), ('tr_d20_39', 0.0002), ('tr_d40_59', 0.0002), ('tr_d60_79', 0.0), ('va_d80_95', 0.0002), ('te_d96_119', 0.003)]
- 跨 date KS: tr→va=0.0048 / tr→te=0.0113
- 结论: cross_stock=stable / cross_date=stable

### `ca_ind` (raw)
- 中文名: 撤卖二值指示
- 英文: Cancel-Ask Binary Indicator (presence)
- X 列号 (0-based): 61
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{ind})} = \mathbf{1}\{N^{(\cdot)}_t > 0\}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
ca_ind = raw_X[:, col_idx["ca_ind"]]  # 3s tick aggregation
```
- 物理含义: ind = 是否发生此类委托的二值指示。把幅度信息压成 {0,1}，信号粒度变粗；在 LGB 树模型里 ind 几乎可被 intst 完全替代（intst==0 ↔ ind==0），所以 ind 重要性远低于 intst。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0203 / mean=0.0088 / by_sym=[0.0026, 0.0081, 0.0203, 0.0087, 0.0044]
- 跨 stock KS: max=0.0752 / by_sym=[0.0236, 0.041, 0.0752, 0.0426, 0.0334]
- 跨 date PSI: tr→va=0.0003 / tr→te=0.003
- 跨 date PSI by bucket: [('tr_d0_19', 0.0015), ('tr_d20_39', 0.0005), ('tr_d40_59', 0.0001), ('tr_d60_79', 0.0), ('va_d80_95', 0.0001), ('te_d96_119', 0.0041)]
- 跨 date KS: tr→va=0.0089 / tr→te=0.0228
- 结论: cross_stock=stable / cross_date=stable

### `lb_acc` (raw)
- 中文名: 限价买二阶差分
- 英文: Limit-Bid 2nd-order Difference (acceleration)
- X 列号 (0-based): 62
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{acc})} = N^{(\cdot)}_t - 2 N^{(\cdot)}_{t-1} + N^{(\cdot)}_{t-2}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
lb_acc = raw_X[:, col_idx["lb_acc"]]  # 3s tick aggregation
```
- 物理含义: acc = 二阶差分，捕捉强度的二阶突变（如订单流加速/减速）。由于 3s tick 强度本身波动大、二阶差分放大噪声，信噪比最低；ablation 中删除 acc 几乎不影响指标。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.3635 / mean=0.2072 / by_sym=[0.0564, 0.3152, 0.3635, 0.0483, 0.2527]
- 跨 stock KS: max=0.1682 / by_sym=[0.0782, 0.0736, 0.1682, 0.0461, 0.0386]
- 跨 date PSI: tr→va=0.0067 / tr→te=0.03
- 跨 date PSI by bucket: [('tr_d0_19', 0.0074), ('tr_d20_39', 0.0048), ('tr_d40_59', 0.0008), ('tr_d60_79', 0.0013), ('va_d80_95', 0.0053), ('te_d96_119', 0.0283)]
- 跨 date KS: tr→va=0.0189 / tr→te=0.0384
- 结论: cross_stock=strong_drift / cross_date=stable

### `la_acc` (raw)
- 中文名: 限价卖二阶差分
- 英文: Limit-Ask 2nd-order Difference (acceleration)
- X 列号 (0-based): 63
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{acc})} = N^{(\cdot)}_t - 2 N^{(\cdot)}_{t-1} + N^{(\cdot)}_{t-2}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
la_acc = raw_X[:, col_idx["la_acc"]]  # 3s tick aggregation
```
- 物理含义: acc = 二阶差分，捕捉强度的二阶突变（如订单流加速/减速）。由于 3s tick 强度本身波动大、二阶差分放大噪声，信噪比最低；ablation 中删除 acc 几乎不影响指标。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0944 / mean=0.0434 / by_sym=[0.0731, 0.0307, 0.0944, 0.0026, 0.0164]
- 跨 stock KS: max=0.1348 / by_sym=[0.0829, 0.0634, 0.1348, 0.0355, 0.0602]
- 跨 date PSI: tr→va=0.0026 / tr→te=0.0148
- 跨 date PSI by bucket: [('tr_d0_19', 0.0061), ('tr_d20_39', 0.0049), ('tr_d40_59', 0.0008), ('tr_d60_79', 0.0008), ('va_d80_95', 0.0025), ('te_d96_119', 0.0178)]
- 跨 date KS: tr→va=0.0196 / tr→te=0.041
- 结论: cross_stock=stable / cross_date=stable

### `mb_acc` (raw)
- 中文名: 市价买二阶差分
- 英文: Market-Bid 2nd-order Difference (acceleration)
- X 列号 (0-based): 64
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{acc})} = N^{(\cdot)}_t - 2 N^{(\cdot)}_{t-1} + N^{(\cdot)}_{t-2}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
mb_acc = raw_X[:, col_idx["mb_acc"]]  # 3s tick aggregation
```
- 物理含义: acc = 二阶差分，捕捉强度的二阶突变（如订单流加速/减速）。由于 3s tick 强度本身波动大、二阶差分放大噪声，信噪比最低；ablation 中删除 acc 几乎不影响指标。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0902 / mean=0.0391 / by_sym=[0.0441, 0.0516, 0.0902, 0.005, 0.0045]
- 跨 stock KS: max=0.1519 / by_sym=[0.0656, 0.0667, 0.1519, 0.0519, 0.0372]
- 跨 date PSI: tr→va=0.0011 / tr→te=0.0165
- 跨 date PSI by bucket: [('tr_d0_19', 0.0079), ('tr_d20_39', 0.0066), ('tr_d40_59', 0.0006), ('tr_d60_79', 0.0007), ('va_d80_95', 0.0009), ('te_d96_119', 0.0207)]
- 跨 date KS: tr→va=0.0194 / tr→te=0.0388
- 结论: cross_stock=stable / cross_date=stable

### `ma_acc` (raw)
- 中文名: 市价卖二阶差分
- 英文: Market-Ask 2nd-order Difference (acceleration)
- X 列号 (0-based): 65
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{acc})} = N^{(\cdot)}_t - 2 N^{(\cdot)}_{t-1} + N^{(\cdot)}_{t-2}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
ma_acc = raw_X[:, col_idx["ma_acc"]]  # 3s tick aggregation
```
- 物理含义: acc = 二阶差分，捕捉强度的二阶突变（如订单流加速/减速）。由于 3s tick 强度本身波动大、二阶差分放大噪声，信噪比最低；ablation 中删除 acc 几乎不影响指标。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0873 / mean=0.0321 / by_sym=[0.0368, 0.0235, 0.0873, 0.0081, 0.0047]
- 跨 stock KS: max=0.1314 / by_sym=[0.0662, 0.0558, 0.1314, 0.053, 0.046]
- 跨 date PSI: tr→va=0.0013 / tr→te=0.0148
- 跨 date PSI by bucket: [('tr_d0_19', 0.0076), ('tr_d20_39', 0.0063), ('tr_d40_59', 0.0012), ('tr_d60_79', 0.0003), ('va_d80_95', 0.0017), ('te_d96_119', 0.0157)]
- 跨 date KS: tr→va=0.0096 / tr→te=0.034
- 结论: cross_stock=stable / cross_date=stable

### `cb_acc` (raw)
- 中文名: 撤买二阶差分
- 英文: Cancel-Bid 2nd-order Difference (acceleration)
- X 列号 (0-based): 66
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{acc})} = N^{(\cdot)}_t - 2 N^{(\cdot)}_{t-1} + N^{(\cdot)}_{t-2}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
cb_acc = raw_X[:, col_idx["cb_acc"]]  # 3s tick aggregation
```
- 物理含义: acc = 二阶差分，捕捉强度的二阶突变（如订单流加速/减速）。由于 3s tick 强度本身波动大、二阶差分放大噪声，信噪比最低；ablation 中删除 acc 几乎不影响指标。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2248 / mean=0.0711 / by_sym=[0.0349, 0.0741, 0.2248, 0.0171, 0.0046]
- 跨 stock KS: max=0.1599 / by_sym=[0.0584, 0.0694, 0.1599, 0.0401, 0.0382]
- 跨 date PSI: tr→va=0.0005 / tr→te=0.0095
- 跨 date PSI by bucket: [('tr_d0_19', 0.0043), ('tr_d20_39', 0.0019), ('tr_d40_59', 0.0013), ('tr_d60_79', 0.0007), ('va_d80_95', 0.0005), ('te_d96_119', 0.0116)]
- 跨 date KS: tr→va=0.0104 / tr→te=0.03
- 结论: cross_stock=mild_drift / cross_date=stable

### `ca_acc` (raw)
- 中文名: 撤卖二阶差分
- 英文: Cancel-Ask 2nd-order Difference (acceleration)
- X 列号 (0-based): 67
- KS-drop 11: no
- 公式: $$x_t^{(\mathrm{acc})} = N^{(\cdot)}_t - 2 N^{(\cdot)}_{t-1} + N^{(\cdot)}_{t-2}$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
ca_acc = raw_X[:, col_idx["ca_acc"]]  # 3s tick aggregation
```
- 物理含义: acc = 二阶差分，捕捉强度的二阶突变（如订单流加速/减速）。由于 3s tick 强度本身波动大、二阶差分放大噪声，信噪比最低；ablation 中删除 acc 几乎不影响指标。
- NaN 处理:
  - 公式层: raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1502 / mean=0.0507 / by_sym=[0.0302, 0.0622, 0.1502, 0.0085, 0.0023]
- 跨 stock KS: max=0.1295 / by_sym=[0.0496, 0.0662, 0.1295, 0.034, 0.0282]
- 跨 date PSI: tr→va=0.0006 / tr→te=0.0121
- 跨 date PSI by bucket: [('tr_d0_19', 0.0082), ('tr_d20_39', 0.0046), ('tr_d40_59', 0.0006), ('tr_d60_79', 0.0003), ('va_d80_95', 0.0004), ('te_d96_119', 0.0151)]
- 跨 date KS: tr→va=0.013 / tr→te=0.0299
- 结论: cross_stock=mild_drift / cross_date=stable

### `ewma_a0.05_lb_intst` (derived)
- 中文名: 限价买强度-EWMA(α=0.05, HL≈13.5tick)
- 英文: EWMA of Limit-Bid Intensity (alpha=0.05, half-life≈13.5 ticks)
- X 列号 (0-based): 199
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['lb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.05], [1, -(1-0.05)], x_safe, axis=-1, zi=(1-0.05)*x_safe[:, :1])[0]
ewma_a0.05_lb_intst = y[:, -1]
```
- 物理含义: 对 lb_intst 做平滑（α=0.05, half-life≈13.5 ticks ≈ 41s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.7431 / mean=0.3666 / by_sym=[0.7431, 0.0785, 0.6106, 0.1468, 0.2542]
- 跨 stock KS: max=0.4303 / by_sym=[0.4303, 0.1343, 0.3372, 0.1914, 0.2496]
- 跨 date PSI: tr→va=0.0717 / tr→te=0.0813
- 跨 date PSI by bucket: [('tr_d0_19', 0.0472), ('tr_d20_39', 0.0404), ('tr_d40_59', 0.0102), ('tr_d60_79', 0.0116), ('va_d80_95', 0.0721), ('te_d96_119', 0.0858)]
- 跨 date KS: tr→va=0.0629 / tr→te=0.1038
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.05_la_intst` (derived)
- 中文名: 限价卖强度-EWMA(α=0.05, HL≈13.5tick)
- 英文: EWMA of Limit-Ask Intensity (alpha=0.05, half-life≈13.5 ticks)
- X 列号 (0-based): 200
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['la_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.05], [1, -(1-0.05)], x_safe, axis=-1, zi=(1-0.05)*x_safe[:, :1])[0]
ewma_a0.05_la_intst = y[:, -1]
```
- 物理含义: 对 la_intst 做平滑（α=0.05, half-life≈13.5 ticks ≈ 41s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.7895 / mean=0.3446 / by_sym=[0.7895, 0.0691, 0.464, 0.1625, 0.2379]
- 跨 stock KS: max=0.4608 / by_sym=[0.4608, 0.1274, 0.2942, 0.2026, 0.2055]
- 跨 date PSI: tr→va=0.0537 / tr→te=0.0831
- 跨 date PSI by bucket: [('tr_d0_19', 0.0284), ('tr_d20_39', 0.0447), ('tr_d40_59', 0.0156), ('tr_d60_79', 0.0156), ('va_d80_95', 0.0593), ('te_d96_119', 0.0895)]
- 跨 date KS: tr→va=0.0551 / tr→te=0.1036
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.05_mb_intst` (derived)
- 中文名: 市价买强度-EWMA(α=0.05, HL≈13.5tick)
- 英文: EWMA of Market-Bid Intensity (alpha=0.05, half-life≈13.5 ticks)
- X 列号 (0-based): 201
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['mb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.05], [1, -(1-0.05)], x_safe, axis=-1, zi=(1-0.05)*x_safe[:, :1])[0]
ewma_a0.05_mb_intst = y[:, -1]
```
- 物理含义: 对 mb_intst 做平滑（α=0.05, half-life≈13.5 ticks ≈ 41s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.6442 / mean=0.2815 / by_sym=[0.4371, 0.0878, 0.6442, 0.0218, 0.2164]
- 跨 stock KS: max=0.3525 / by_sym=[0.3525, 0.1506, 0.3342, 0.0684, 0.2023]
- 跨 date PSI: tr→va=0.0709 / tr→te=0.1278
- 跨 date PSI by bucket: [('tr_d0_19', 0.0462), ('tr_d20_39', 0.0467), ('tr_d40_59', 0.0176), ('tr_d60_79', 0.0077), ('va_d80_95', 0.081), ('te_d96_119', 0.1321)]
- 跨 date KS: tr→va=0.0734 / tr→te=0.1326
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `ewma_a0.05_ma_intst` (derived)
- 中文名: 市价卖强度-EWMA(α=0.05, HL≈13.5tick)
- 英文: EWMA of Market-Ask Intensity (alpha=0.05, half-life≈13.5 ticks)
- X 列号 (0-based): 202
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['ma_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.05], [1, -(1-0.05)], x_safe, axis=-1, zi=(1-0.05)*x_safe[:, :1])[0]
ewma_a0.05_ma_intst = y[:, -1]
```
- 物理含义: 对 ma_intst 做平滑（α=0.05, half-life≈13.5 ticks ≈ 41s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.5898 / mean=0.2759 / by_sym=[0.3743, 0.1045, 0.5898, 0.0212, 0.2896]
- 跨 stock KS: max=0.3338 / by_sym=[0.312, 0.1546, 0.3338, 0.0694, 0.2311]
- 跨 date PSI: tr→va=0.058 / tr→te=0.1262
- 跨 date PSI by bucket: [('tr_d0_19', 0.0484), ('tr_d20_39', 0.0381), ('tr_d40_59', 0.0117), ('tr_d60_79', 0.0113), ('va_d80_95', 0.0632), ('te_d96_119', 0.1267)]
- 跨 date KS: tr→va=0.0612 / tr→te=0.1304
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `ewma_a0.05_cb_intst` (derived)
- 中文名: 撤买强度-EWMA(α=0.05, HL≈13.5tick)
- 英文: EWMA of Cancel-Bid Intensity (alpha=0.05, half-life≈13.5 ticks)
- X 列号 (0-based): 203
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['cb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.05], [1, -(1-0.05)], x_safe, axis=-1, zi=(1-0.05)*x_safe[:, :1])[0]
ewma_a0.05_cb_intst = y[:, -1]
```
- 物理含义: 对 cb_intst 做平滑（α=0.05, half-life≈13.5 ticks ≈ 41s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.7777 / mean=0.3587 / by_sym=[0.5565, 0.0333, 0.7777, 0.18, 0.2461]
- 跨 stock KS: max=0.3891 / by_sym=[0.3891, 0.0914, 0.3418, 0.2064, 0.2529]
- 跨 date PSI: tr→va=0.0418 / tr→te=0.055
- 跨 date PSI by bucket: [('tr_d0_19', 0.0282), ('tr_d20_39', 0.0339), ('tr_d40_59', 0.0111), ('tr_d60_79', 0.0112), ('va_d80_95', 0.0416), ('te_d96_119', 0.0593)]
- 跨 date KS: tr→va=0.0535 / tr→te=0.0822
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.05_ca_intst` (derived)
- 中文名: 撤卖强度-EWMA(α=0.05, HL≈13.5tick)
- 英文: EWMA of Cancel-Ask Intensity (alpha=0.05, half-life≈13.5 ticks)
- X 列号 (0-based): 204
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['ca_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.05], [1, -(1-0.05)], x_safe, axis=-1, zi=(1-0.05)*x_safe[:, :1])[0]
ewma_a0.05_ca_intst = y[:, -1]
```
- 物理含义: 对 ca_intst 做平滑（α=0.05, half-life≈13.5 ticks ≈ 41s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4673 / mean=0.2655 / by_sym=[0.4673, 0.0136, 0.4298, 0.1762, 0.2404]
- 跨 stock KS: max=0.3543 / by_sym=[0.3543, 0.0638, 0.2806, 0.2182, 0.2216]
- 跨 date PSI: tr→va=0.0354 / tr→te=0.0647
- 跨 date PSI by bucket: [('tr_d0_19', 0.0271), ('tr_d20_39', 0.0398), ('tr_d40_59', 0.01), ('tr_d60_79', 0.008), ('va_d80_95', 0.0389), ('te_d96_119', 0.0672)]
- 跨 date KS: tr→va=0.0655 / tr→te=0.0912
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.1_lb_intst` (derived)
- 中文名: 限价买强度-EWMA(α=0.1, HL≈6.6tick)
- 英文: EWMA of Limit-Bid Intensity (alpha=0.1, half-life≈6.6 ticks)
- X 列号 (0-based): 205
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['lb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.1], [1, -(1-0.1)], x_safe, axis=-1, zi=(1-0.1)*x_safe[:, :1])[0]
ewma_a0.1_lb_intst = y[:, -1]
```
- 物理含义: 对 lb_intst 做平滑（α=0.1, half-life≈6.6 ticks ≈ 20s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.5824 / mean=0.2833 / by_sym=[0.5824, 0.0449, 0.5083, 0.0891, 0.1919]
- 跨 stock KS: max=0.3796 / by_sym=[0.3796, 0.1071, 0.3183, 0.1539, 0.2149]
- 跨 date PSI: tr→va=0.0551 / tr→te=0.0703
- 跨 date PSI by bucket: [('tr_d0_19', 0.0409), ('tr_d20_39', 0.0343), ('tr_d40_59', 0.007), ('tr_d60_79', 0.0087), ('va_d80_95', 0.0536), ('te_d96_119', 0.0719)]
- 跨 date KS: tr→va=0.0576 / tr→te=0.0977
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.1_la_intst` (derived)
- 中文名: 限价卖强度-EWMA(α=0.1, HL≈6.6tick)
- 英文: EWMA of Limit-Ask Intensity (alpha=0.1, half-life≈6.6 ticks)
- X 列号 (0-based): 206
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['la_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.1], [1, -(1-0.1)], x_safe, axis=-1, zi=(1-0.1)*x_safe[:, :1])[0]
ewma_a0.1_la_intst = y[:, -1]
```
- 物理含义: 对 la_intst 做平滑（α=0.1, half-life≈6.6 ticks ≈ 20s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.6358 / mean=0.2684 / by_sym=[0.6358, 0.0399, 0.383, 0.0997, 0.1836]
- 跨 stock KS: max=0.4088 / by_sym=[0.4088, 0.104, 0.2787, 0.1653, 0.1762]
- 跨 date PSI: tr→va=0.0445 / tr→te=0.0735
- 跨 date PSI by bucket: [('tr_d0_19', 0.0235), ('tr_d20_39', 0.0395), ('tr_d40_59', 0.0107), ('tr_d60_79', 0.0121), ('va_d80_95', 0.0463), ('te_d96_119', 0.0771)]
- 跨 date KS: tr→va=0.0505 / tr→te=0.1007
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.1_mb_intst` (derived)
- 中文名: 市价买强度-EWMA(α=0.1, HL≈6.6tick)
- 英文: EWMA of Market-Bid Intensity (alpha=0.1, half-life≈6.6 ticks)
- X 列号 (0-based): 207
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['mb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.1], [1, -(1-0.1)], x_safe, axis=-1, zi=(1-0.1)*x_safe[:, :1])[0]
ewma_a0.1_mb_intst = y[:, -1]
```
- 物理含义: 对 mb_intst 做平滑（α=0.1, half-life≈6.6 ticks ≈ 20s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.6144 / mean=0.2423 / by_sym=[0.3475, 0.0512, 0.6144, 0.0237, 0.1746]
- 跨 stock KS: max=0.3281 / by_sym=[0.3138, 0.1198, 0.3281, 0.0517, 0.1703]
- 跨 date PSI: tr→va=0.0531 / tr→te=0.1154
- 跨 date PSI by bucket: [('tr_d0_19', 0.04), ('tr_d20_39', 0.0404), ('tr_d40_59', 0.0122), ('tr_d60_79', 0.0055), ('va_d80_95', 0.0602), ('te_d96_119', 0.1202)]
- 跨 date KS: tr→va=0.066 / tr→te=0.1295
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `ewma_a0.1_ma_intst` (derived)
- 中文名: 市价卖强度-EWMA(α=0.1, HL≈6.6tick)
- 英文: EWMA of Market-Ask Intensity (alpha=0.1, half-life≈6.6 ticks)
- X 列号 (0-based): 208
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['ma_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.1], [1, -(1-0.1)], x_safe, axis=-1, zi=(1-0.1)*x_safe[:, :1])[0]
ewma_a0.1_ma_intst = y[:, -1]
```
- 物理含义: 对 ma_intst 做平滑（α=0.1, half-life≈6.6 ticks ≈ 20s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.5231 / mean=0.2266 / by_sym=[0.2988, 0.0696, 0.5231, 0.0243, 0.217]
- 跨 stock KS: max=0.3203 / by_sym=[0.2805, 0.1345, 0.3203, 0.0494, 0.1952]
- 跨 date PSI: tr→va=0.0424 / tr→te=0.1095
- 跨 date PSI by bucket: [('tr_d0_19', 0.0441), ('tr_d20_39', 0.0344), ('tr_d40_59', 0.0087), ('tr_d60_79', 0.0085), ('va_d80_95', 0.0445), ('te_d96_119', 0.108)]
- 跨 date KS: tr→va=0.0535 / tr→te=0.1267
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `ewma_a0.1_cb_intst` (derived)
- 中文名: 撤买强度-EWMA(α=0.1, HL≈6.6tick)
- 英文: EWMA of Cancel-Bid Intensity (alpha=0.1, half-life≈6.6 ticks)
- X 列号 (0-based): 209
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['cb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.1], [1, -(1-0.1)], x_safe, axis=-1, zi=(1-0.1)*x_safe[:, :1])[0]
ewma_a0.1_cb_intst = y[:, -1]
```
- 物理含义: 对 cb_intst 做平滑（α=0.1, half-life≈6.6 ticks ≈ 20s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.685 / mean=0.2855 / by_sym=[0.4301, 0.0193, 0.685, 0.1083, 0.1846]
- 跨 stock KS: max=0.3416 / by_sym=[0.3416, 0.0711, 0.3309, 0.1692, 0.2192]
- 跨 date PSI: tr→va=0.0265 / tr→te=0.0423
- 跨 date PSI by bucket: [('tr_d0_19', 0.0249), ('tr_d20_39', 0.0285), ('tr_d40_59', 0.0074), ('tr_d60_79', 0.0091), ('va_d80_95', 0.0244), ('te_d96_119', 0.0459)]
- 跨 date KS: tr→va=0.045 / tr→te=0.0746
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.1_ca_intst` (derived)
- 中文名: 撤卖强度-EWMA(α=0.1, HL≈6.6tick)
- 英文: EWMA of Cancel-Ask Intensity (alpha=0.1, half-life≈6.6 ticks)
- X 列号 (0-based): 210
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['ca_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.1], [1, -(1-0.1)], x_safe, axis=-1, zi=(1-0.1)*x_safe[:, :1])[0]
ewma_a0.1_ca_intst = y[:, -1]
```
- 物理含义: 对 ca_intst 做平滑（α=0.1, half-life≈6.6 ticks ≈ 20s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.369 / mean=0.2016 / by_sym=[0.353, 0.0075, 0.369, 0.1075, 0.1712]
- 跨 stock KS: max=0.3038 / by_sym=[0.3038, 0.0436, 0.2683, 0.1734, 0.1853]
- 跨 date PSI: tr→va=0.0213 / tr→te=0.0542
- 跨 date PSI by bucket: [('tr_d0_19', 0.0256), ('tr_d20_39', 0.0347), ('tr_d40_59', 0.0062), ('tr_d60_79', 0.0065), ('va_d80_95', 0.0247), ('te_d96_119', 0.0559)]
- 跨 date KS: tr→va=0.052 / tr→te=0.083
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.3_lb_intst` (derived)
- 中文名: 限价买强度-EWMA(α=0.3, HL≈1.9tick)
- 英文: EWMA of Limit-Bid Intensity (alpha=0.3, half-life≈1.9 ticks)
- X 列号 (0-based): 211
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['lb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.3], [1, -(1-0.3)], x_safe, axis=-1, zi=(1-0.3)*x_safe[:, :1])[0]
ewma_a0.3_lb_intst = y[:, -1]
```
- 物理含义: 对 lb_intst 做平滑（α=0.3, half-life≈1.9 ticks ≈ 6s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4125 / mean=0.1849 / by_sym=[0.3569, 0.0142, 0.4125, 0.0299, 0.1107]
- 跨 stock KS: max=0.2995 / by_sym=[0.2975, 0.0618, 0.2995, 0.0892, 0.1565]
- 跨 date PSI: tr→va=0.0353 / tr→te=0.0557
- 跨 date PSI by bucket: [('tr_d0_19', 0.0312), ('tr_d20_39', 0.0243), ('tr_d40_59', 0.0047), ('tr_d60_79', 0.0045), ('va_d80_95', 0.0312), ('te_d96_119', 0.0538)]
- 跨 date KS: tr→va=0.0475 / tr→te=0.0903
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.3_la_intst` (derived)
- 中文名: 限价卖强度-EWMA(α=0.3, HL≈1.9tick)
- 英文: EWMA of Limit-Ask Intensity (alpha=0.3, half-life≈1.9 ticks)
- X 列号 (0-based): 212
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['la_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.3], [1, -(1-0.3)], x_safe, axis=-1, zi=(1-0.3)*x_safe[:, :1])[0]
ewma_a0.3_la_intst = y[:, -1]
```
- 物理含义: 对 la_intst 做平滑（α=0.3, half-life≈1.9 ticks ≈ 6s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4045 / mean=0.1735 / by_sym=[0.4045, 0.0091, 0.2931, 0.0367, 0.124]
- 跨 stock KS: max=0.3244 / by_sym=[0.3244, 0.0558, 0.259, 0.102, 0.1265]
- 跨 date PSI: tr→va=0.0251 / tr→te=0.0567
- 跨 date PSI by bucket: [('tr_d0_19', 0.0188), ('tr_d20_39', 0.0287), ('tr_d40_59', 0.0058), ('tr_d60_79', 0.0067), ('va_d80_95', 0.0264), ('te_d96_119', 0.0627)]
- 跨 date KS: tr→va=0.0413 / tr→te=0.0947
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.3_mb_intst` (derived)
- 中文名: 市价买强度-EWMA(α=0.3, HL≈1.9tick)
- 英文: EWMA of Market-Bid Intensity (alpha=0.3, half-life≈1.9 ticks)
- X 列号 (0-based): 213
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['mb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.3], [1, -(1-0.3)], x_safe, axis=-1, zi=(1-0.3)*x_safe[:, :1])[0]
ewma_a0.3_mb_intst = y[:, -1]
```
- 物理含义: 对 mb_intst 做平滑（α=0.3, half-life≈1.9 ticks ≈ 6s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.5554 / mean=0.1907 / by_sym=[0.2137, 0.0258, 0.5554, 0.04, 0.1184]
- 跨 stock KS: max=0.3263 / by_sym=[0.2488, 0.0756, 0.3263, 0.0811, 0.1208]
- 跨 date PSI: tr→va=0.028 / tr→te=0.0875
- 跨 date PSI by bucket: [('tr_d0_19', 0.0302), ('tr_d20_39', 0.0304), ('tr_d40_59', 0.0057), ('tr_d60_79', 0.0034), ('va_d80_95', 0.0316), ('te_d96_119', 0.0906)]
- 跨 date KS: tr→va=0.0523 / tr→te=0.1182
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.3_ma_intst` (derived)
- 中文名: 市价卖强度-EWMA(α=0.3, HL≈1.9tick)
- 英文: EWMA of Market-Ask Intensity (alpha=0.3, half-life≈1.9 ticks)
- X 列号 (0-based): 214
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['ma_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.3], [1, -(1-0.3)], x_safe, axis=-1, zi=(1-0.3)*x_safe[:, :1])[0]
ewma_a0.3_ma_intst = y[:, -1]
```
- 物理含义: 对 ma_intst 做平滑（α=0.3, half-life≈1.9 ticks ≈ 6s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.451 / mean=0.1702 / by_sym=[0.1921, 0.031, 0.451, 0.0429, 0.1341]
- 跨 stock KS: max=0.3037 / by_sym=[0.228, 0.093, 0.3037, 0.0861, 0.1387]
- 跨 date PSI: tr→va=0.0222 / tr→te=0.0822
- 跨 date PSI by bucket: [('tr_d0_19', 0.0374), ('tr_d20_39', 0.0281), ('tr_d40_59', 0.0045), ('tr_d60_79', 0.0036), ('va_d80_95', 0.0215), ('te_d96_119', 0.0805)]
- 跨 date KS: tr→va=0.0397 / tr→te=0.1166
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.3_cb_intst` (derived)
- 中文名: 撤买强度-EWMA(α=0.3, HL≈1.9tick)
- 英文: EWMA of Cancel-Bid Intensity (alpha=0.3, half-life≈1.9 ticks)
- X 列号 (0-based): 215
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['cb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.3], [1, -(1-0.3)], x_safe, axis=-1, zi=(1-0.3)*x_safe[:, :1])[0]
ewma_a0.3_cb_intst = y[:, -1]
```
- 物理含义: 对 cb_intst 做平滑（α=0.3, half-life≈1.9 ticks ≈ 6s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.5976 / mean=0.205 / by_sym=[0.2479, 0.0351, 0.5976, 0.0393, 0.1048]
- 跨 stock KS: max=0.3303 / by_sym=[0.2602, 0.0574, 0.3303, 0.1021, 0.1592]
- 跨 date PSI: tr→va=0.0125 / tr→te=0.0344
- 跨 date PSI by bucket: [('tr_d0_19', 0.0189), ('tr_d20_39', 0.0187), ('tr_d40_59', 0.0052), ('tr_d60_79', 0.0052), ('va_d80_95', 0.0116), ('te_d96_119', 0.0347)]
- 跨 date KS: tr→va=0.0334 / tr→te=0.0672
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.3_ca_intst` (derived)
- 中文名: 撤卖强度-EWMA(α=0.3, HL≈1.9tick)
- 英文: EWMA of Cancel-Ask Intensity (alpha=0.3, half-life≈1.9 ticks)
- X 列号 (0-based): 216
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['ca_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.3], [1, -(1-0.3)], x_safe, axis=-1, zi=(1-0.3)*x_safe[:, :1])[0]
ewma_a0.3_ca_intst = y[:, -1]
```
- 物理含义: 对 ca_intst 做平滑（α=0.3, half-life≈1.9 ticks ≈ 6s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.335 / mean=0.1388 / by_sym=[0.2005, 0.0253, 0.335, 0.0399, 0.0934]
- 跨 stock KS: max=0.267 / by_sym=[0.2272, 0.0595, 0.267, 0.105, 0.1302]
- 跨 date PSI: tr→va=0.0104 / tr→te=0.0426
- 跨 date PSI by bucket: [('tr_d0_19', 0.0232), ('tr_d20_39', 0.0249), ('tr_d40_59', 0.0026), ('tr_d60_79', 0.0038), ('va_d80_95', 0.0106), ('te_d96_119', 0.0448)]
- 跨 date KS: tr→va=0.0366 / tr→te=0.0828
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.5_lb_intst` (derived)
- 中文名: 限价买强度-EWMA(α=0.5, HL≈1.0tick)
- 英文: EWMA of Limit-Bid Intensity (alpha=0.5, half-life≈1.0 ticks)
- X 列号 (0-based): 217
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['lb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.5], [1, -(1-0.5)], x_safe, axis=-1, zi=(1-0.5)*x_safe[:, :1])[0]
ewma_a0.5_lb_intst = y[:, -1]
```
- 物理含义: 对 lb_intst 做平滑（α=0.5, half-life≈1.0 ticks ≈ 3s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.3745 / mean=0.1496 / by_sym=[0.2591, 0.0153, 0.3745, 0.0172, 0.0818]
- 跨 stock KS: max=0.298 / by_sym=[0.2566, 0.0421, 0.298, 0.0608, 0.1231]
- 跨 date PSI: tr→va=0.0247 / tr→te=0.05
- 跨 date PSI by bucket: [('tr_d0_19', 0.0258), ('tr_d20_39', 0.0188), ('tr_d40_59', 0.0035), ('tr_d60_79', 0.0031), ('va_d80_95', 0.022), ('te_d96_119', 0.0493)]
- 跨 date KS: tr→va=0.0402 / tr→te=0.0853
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.5_la_intst` (derived)
- 中文名: 限价卖强度-EWMA(α=0.5, HL≈1.0tick)
- 英文: EWMA of Limit-Ask Intensity (alpha=0.5, half-life≈1.0 ticks)
- X 列号 (0-based): 218
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['la_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.5], [1, -(1-0.5)], x_safe, axis=-1, zi=(1-0.5)*x_safe[:, :1])[0]
ewma_a0.5_la_intst = y[:, -1]
```
- 物理含义: 对 la_intst 做平滑（α=0.5, half-life≈1.0 ticks ≈ 3s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.3062 / mean=0.137 / by_sym=[0.3062, 0.0065, 0.2547, 0.0222, 0.0956]
- 跨 stock KS: max=0.2843 / by_sym=[0.2843, 0.0342, 0.2521, 0.0712, 0.0989]
- 跨 date PSI: tr→va=0.018 / tr→te=0.0555
- 跨 date PSI by bucket: [('tr_d0_19', 0.017), ('tr_d20_39', 0.0221), ('tr_d40_59', 0.0036), ('tr_d60_79', 0.0048), ('va_d80_95', 0.021), ('te_d96_119', 0.0613)]
- 跨 date KS: tr→va=0.0388 / tr→te=0.0898
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.5_mb_intst` (derived)
- 中文名: 市价买强度-EWMA(α=0.5, HL≈1.0tick)
- 英文: EWMA of Market-Bid Intensity (alpha=0.5, half-life≈1.0 ticks)
- X 列号 (0-based): 219
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['mb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.5], [1, -(1-0.5)], x_safe, axis=-1, zi=(1-0.5)*x_safe[:, :1])[0]
ewma_a0.5_mb_intst = y[:, -1]
```
- 物理含义: 对 mb_intst 做平滑（α=0.5, half-life≈1.0 ticks ≈ 3s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.5059 / mean=0.1664 / by_sym=[0.1587, 0.0326, 0.5059, 0.0463, 0.0884]
- 跨 stock KS: max=0.3294 / by_sym=[0.2167, 0.0558, 0.3294, 0.1001, 0.0963]
- 跨 date PSI: tr→va=0.0184 / tr→te=0.0756
- 跨 date PSI by bucket: [('tr_d0_19', 0.0254), ('tr_d20_39', 0.0254), ('tr_d40_59', 0.0038), ('tr_d60_79', 0.0028), ('va_d80_95', 0.0184), ('te_d96_119', 0.0794)]
- 跨 date KS: tr→va=0.047 / tr→te=0.1085
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.5_ma_intst` (derived)
- 中文名: 市价卖强度-EWMA(α=0.5, HL≈1.0tick)
- 英文: EWMA of Market-Ask Intensity (alpha=0.5, half-life≈1.0 ticks)
- X 列号 (0-based): 220
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['ma_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.5], [1, -(1-0.5)], x_safe, axis=-1, zi=(1-0.5)*x_safe[:, :1])[0]
ewma_a0.5_ma_intst = y[:, -1]
```
- 物理含义: 对 ma_intst 做平滑（α=0.5, half-life≈1.0 ticks ≈ 3s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4055 / mean=0.1444 / by_sym=[0.1467, 0.025, 0.4055, 0.0494, 0.0956]
- 跨 stock KS: max=0.3002 / by_sym=[0.2008, 0.0743, 0.3002, 0.1067, 0.1102]
- 跨 date PSI: tr→va=0.0153 / tr→te=0.0718
- 跨 date PSI by bucket: [('tr_d0_19', 0.034), ('tr_d20_39', 0.0244), ('tr_d40_59', 0.003), ('tr_d60_79', 0.0025), ('va_d80_95', 0.0141), ('te_d96_119', 0.0721)]
- 跨 date KS: tr→va=0.0354 / tr→te=0.1106
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.5_cb_intst` (derived)
- 中文名: 撤买强度-EWMA(α=0.5, HL≈1.0tick)
- 英文: EWMA of Cancel-Bid Intensity (alpha=0.5, half-life≈1.0 ticks)
- X 列号 (0-based): 221
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['cb_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.5], [1, -(1-0.5)], x_safe, axis=-1, zi=(1-0.5)*x_safe[:, :1])[0]
ewma_a0.5_cb_intst = y[:, -1]
```
- 物理含义: 对 cb_intst 做平滑（α=0.5, half-life≈1.0 ticks ≈ 3s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.5695 / mean=0.1808 / by_sym=[0.1773, 0.0604, 0.5695, 0.0256, 0.071]
- 跨 stock KS: max=0.3395 / by_sym=[0.2176, 0.0988, 0.3395, 0.0695, 0.1263]
- 跨 date PSI: tr→va=0.0078 / tr→te=0.0322
- 跨 date PSI by bucket: [('tr_d0_19', 0.0165), ('tr_d20_39', 0.0137), ('tr_d40_59', 0.0039), ('tr_d60_79', 0.0035), ('va_d80_95', 0.0076), ('te_d96_119', 0.0336)]
- 跨 date KS: tr→va=0.0277 / tr→te=0.0683
- 结论: cross_stock=strong_drift / cross_date=stable

### `ewma_a0.5_ca_intst` (derived)
- 中文名: 撤卖强度-EWMA(α=0.5, HL≈1.0tick)
- 英文: EWMA of Cancel-Ask Intensity (alpha=0.5, half-life≈1.0 ticks)
- X 列号 (0-based): 222
- KS-drop 11: no
- 公式: $$\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}$$
- 代码定位: `fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — 底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)`
- numpy 伪代码:
```python
x = raw_X3d[:, :, col_idx['ca_intst']]
x_safe = np.where(np.isfinite(x), x, 0.0)
# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{t-1}
y = lfilter([0.5], [1, -(1-0.5)], x_safe, axis=-1, zi=(1-0.5)*x_safe[:, :1])[0]
ewma_a0.5_ca_intst = y[:, -1]
```
- 物理含义: 对 ca_intst 做平滑（α=0.5, half-life≈1.0 ticks ≈ 3s）。4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。
- NaN 处理:
  - 公式层: np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动，LightGBM 内置 missing routing
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.3261 / mean=0.1215 / by_sym=[0.1438, 0.0454, 0.3261, 0.0245, 0.0676]
- 跨 stock KS: max=0.2797 / by_sym=[0.1917, 0.097, 0.2797, 0.0715, 0.0997]
- 跨 date PSI: tr→va=0.0063 / tr→te=0.0433
- 跨 date PSI by bucket: [('tr_d0_19', 0.0215), ('tr_d20_39', 0.0209), ('tr_d40_59', 0.002), ('tr_d60_79', 0.0028), ('va_d80_95', 0.0066), ('te_d96_119', 0.0443)]
- 跨 date KS: tr→va=0.0264 / tr→te=0.0832
- 结论: cross_stock=strong_drift / cross_date=stable

### `cancel_imb_W20` (derived)
- 中文名: 撤单不平衡(窗口W=20tick≈60s)
- 英文: Cancellation Imbalance over W=20 ticks
- X 列号 (0-based): 344
- KS-drop 11: no
- 公式: $$\mathrm{cancel\_imb}_W = \frac{\sum_{s=t-W+1}^{t} cb_s}{\sum (lb_s+mb_s+cb_s)+\epsilon}-\frac{\sum_{s=t-W+1}^{t} ca_s}{\sum (la_s+ma_s+ca_s)+\epsilon}$$
- 代码定位: `fast_features_batch.py:535-554 (compute_stage3_batch --- cancel_imb)`
- numpy 伪代码:
```python
cb_sum=cb[:, -20:].sum(-1); ca_sum=ca[:, -20:].sum(-1)
bt_sum=(lb+mb+cb)[:, -20:].sum(-1); at_sum=(la+ma+ca)[:, -20:].sum(-1)
imb=(cb_sum/(bt_sum+1e-8))-(ca_sum/(at_sum+1e-8))
cancel_imb_W20 = np.clip(np.nan_to_num(imb), -1, 1)
```
- 物理含义: 窗口 W=20 内『买侧撤单占买侧总下单的比例』减去『卖侧撤单占卖侧总下单的比例』，∈[-1,1]。+1 = 买侧撤单密集（看跌信号，参与者放弃挂单）；-1 = 卖侧撤单密集（看涨信号）。与 ma_intst/mb_intst 互补：intst 看主动成交方向，cancel_imb 看『谁先怂』。3 个 W 覆盖短/中/长尺度（W=20≈60s 短期博弈，W=100≈300s 中期持仓压力）。
- NaN 处理:
  - 公式层: denom +EPS=1e-8 避免除零；np.where(np.isfinite, ..., 0.0) 兜底；clip(-1, 1)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1143 / mean=0.0507 / by_sym=[0.0216, 0.0079, 0.1143, 0.0651, 0.0447]
- 跨 stock KS: max=0.0826 / by_sym=[0.0503, 0.0375, 0.0826, 0.0742, 0.0535]
- 跨 date PSI: tr→va=0.0075 / tr→te=0.0238
- 跨 date PSI by bucket: [('tr_d0_19', 0.007), ('tr_d20_39', 0.0021), ('tr_d40_59', 0.0014), ('tr_d60_79', 0.0013), ('va_d80_95', 0.0072), ('te_d96_119', 0.0245)]
- 跨 date KS: tr→va=0.0291 / tr→te=0.0416
- 结论: cross_stock=mild_drift / cross_date=stable

### `cancel_imb_W50` (derived)
- 中文名: 撤单不平衡(窗口W=50tick≈150s)
- 英文: Cancellation Imbalance over W=50 ticks
- X 列号 (0-based): 345
- KS-drop 11: no
- 公式: $$\mathrm{cancel\_imb}_W = \frac{\sum_{s=t-W+1}^{t} cb_s}{\sum (lb_s+mb_s+cb_s)+\epsilon}-\frac{\sum_{s=t-W+1}^{t} ca_s}{\sum (la_s+ma_s+ca_s)+\epsilon}$$
- 代码定位: `fast_features_batch.py:535-554 (compute_stage3_batch --- cancel_imb)`
- numpy 伪代码:
```python
cb_sum=cb[:, -50:].sum(-1); ca_sum=ca[:, -50:].sum(-1)
bt_sum=(lb+mb+cb)[:, -50:].sum(-1); at_sum=(la+ma+ca)[:, -50:].sum(-1)
imb=(cb_sum/(bt_sum+1e-8))-(ca_sum/(at_sum+1e-8))
cancel_imb_W50 = np.clip(np.nan_to_num(imb), -1, 1)
```
- 物理含义: 窗口 W=50 内『买侧撤单占买侧总下单的比例』减去『卖侧撤单占卖侧总下单的比例』，∈[-1,1]。+1 = 买侧撤单密集（看跌信号，参与者放弃挂单）；-1 = 卖侧撤单密集（看涨信号）。与 ma_intst/mb_intst 互补：intst 看主动成交方向，cancel_imb 看『谁先怂』。3 个 W 覆盖短/中/长尺度（W=20≈60s 短期博弈，W=100≈300s 中期持仓压力）。
- NaN 处理:
  - 公式层: denom +EPS=1e-8 避免除零；np.where(np.isfinite, ..., 0.0) 兜底；clip(-1, 1)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1145 / mean=0.0541 / by_sym=[0.0169, 0.011, 0.1145, 0.0789, 0.0494]
- 跨 stock KS: max=0.0857 / by_sym=[0.0427, 0.0523, 0.0857, 0.0809, 0.0671]
- 跨 date PSI: tr→va=0.0083 / tr→te=0.0304
- 跨 date PSI by bucket: [('tr_d0_19', 0.0058), ('tr_d20_39', 0.0028), ('tr_d40_59', 0.0018), ('tr_d60_79', 0.0018), ('va_d80_95', 0.0077), ('te_d96_119', 0.0281)]
- 跨 date KS: tr→va=0.0298 / tr→te=0.0485
- 结论: cross_stock=mild_drift / cross_date=stable

### `cancel_imb_W100` (derived)
- 中文名: 撤单不平衡(窗口W=100tick≈300s)
- 英文: Cancellation Imbalance over W=100 ticks
- X 列号 (0-based): 346
- KS-drop 11: no
- 公式: $$\mathrm{cancel\_imb}_W = \frac{\sum_{s=t-W+1}^{t} cb_s}{\sum (lb_s+mb_s+cb_s)+\epsilon}-\frac{\sum_{s=t-W+1}^{t} ca_s}{\sum (la_s+ma_s+ca_s)+\epsilon}$$
- 代码定位: `fast_features_batch.py:535-554 (compute_stage3_batch --- cancel_imb)`
- numpy 伪代码:
```python
cb_sum=cb[:, -100:].sum(-1); ca_sum=ca[:, -100:].sum(-1)
bt_sum=(lb+mb+cb)[:, -100:].sum(-1); at_sum=(la+ma+ca)[:, -100:].sum(-1)
imb=(cb_sum/(bt_sum+1e-8))-(ca_sum/(at_sum+1e-8))
cancel_imb_W100 = np.clip(np.nan_to_num(imb), -1, 1)
```
- 物理含义: 窗口 W=100 内『买侧撤单占买侧总下单的比例』减去『卖侧撤单占卖侧总下单的比例』，∈[-1,1]。+1 = 买侧撤单密集（看跌信号，参与者放弃挂单）；-1 = 卖侧撤单密集（看涨信号）。与 ma_intst/mb_intst 互补：intst 看主动成交方向，cancel_imb 看『谁先怂』。3 个 W 覆盖短/中/长尺度（W=20≈60s 短期博弈，W=100≈300s 中期持仓压力）。
- NaN 处理:
  - 公式层: denom +EPS=1e-8 避免除零；np.where(np.isfinite, ..., 0.0) 兜底；clip(-1, 1)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0992 / mean=0.0519 / by_sym=[0.0128, 0.0183, 0.0992, 0.077, 0.0523]
- 跨 stock KS: max=0.0859 / by_sym=[0.0404, 0.065, 0.086, 0.0853, 0.0766]
- 跨 date PSI: tr→va=0.0123 / tr→te=0.0344
- 跨 date PSI by bucket: [('tr_d0_19', 0.0043), ('tr_d20_39', 0.0022), ('tr_d40_59', 0.0038), ('tr_d60_79', 0.0022), ('va_d80_95', 0.0104), ('te_d96_119', 0.0342)]
- 跨 date KS: tr→va=0.0317 / tr→te=0.0529
- 结论: cross_stock=stable / cross_date=stable

## F4 — 29 features

### `volume_delta` (raw)
- 中文名: 成交量增量
- 英文: Volume Delta (Δ shares per tick)
- X 列号 (0-based): 4
- KS-drop 11: no
- 公式: $$\Delta V_t = V_t - V_{t-1}\;\;(\text{shares})$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
volume_delta = raw_X[:, col_idx["volume_delta"]]
```
- 物理含义: 3s tick 内的成交量/成交额增量。F4 微结构波动的两个 raw 输入。vol_burst_W / amt_burst_W 直接以它们为基础做窗口冲击检测；kyle_lam / signed_dvol / OFI toxicity 也用 amount_delta 衡量主动成交强度。
- NaN 处理:
  - 公式层: raw 字段；cache 已 log1p 处理，保证无 NaN/inf
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.3491 / mean=0.1185 / by_sym=[0.1166, 0.0205, 0.3491, 0.0417, 0.0645]
- 跨 stock KS: max=0.3609 / by_sym=[0.1979, 0.1366, 0.3609, 0.1284, 0.1015]
- 跨 date PSI: tr→va=0.0068 / tr→te=0.0401
- 跨 date PSI by bucket: [('tr_d0_19', 0.0199), ('tr_d20_39', 0.0158), ('tr_d40_59', 0.0025), ('tr_d60_79', 0.0013), ('va_d80_95', 0.0065), ('te_d96_119', 0.0392)]
- 跨 date KS: tr→va=0.0492 / tr→te=0.1185
- 结论: cross_stock=strong_drift / cross_date=stable

### `amount_delta` (raw)
- 中文名: 成交额增量
- 英文: Amount Delta (Δ cash per tick)
- X 列号 (0-based): 5
- KS-drop 11: no
- 公式: $$\Delta A_t = A_t - A_{t-1}\;\;(\text{cash})$$
- 代码定位: `fast_features_batch.py:N/A (raw, 主办方原始 column)`
- numpy 伪代码:
```python
amount_delta = raw_X[:, col_idx["amount_delta"]]
```
- 物理含义: 3s tick 内的成交量/成交额增量。F4 微结构波动的两个 raw 输入。vol_burst_W / amt_burst_W 直接以它们为基础做窗口冲击检测；kyle_lam / signed_dvol / OFI toxicity 也用 amount_delta 衡量主动成交强度。
- NaN 处理:
  - 公式层: raw 字段；cache 已 log1p 处理，保证无 NaN/inf
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=1.1444 / mean=0.3253 / by_sym=[0.0253, 0.1602, 1.1444, 0.2362, 0.0602]
- 跨 stock KS: max=0.4422 / by_sym=[0.0989, 0.2088, 0.4422, 0.2741, 0.1974]
- 跨 date PSI: tr→va=0.0077 / tr→te=0.062
- 跨 date PSI by bucket: [('tr_d0_19', 0.029), ('tr_d20_39', 0.013), ('tr_d40_59', 0.0021), ('tr_d60_79', 0.0118), ('va_d80_95', 0.0069), ('te_d96_119', 0.0627)]
- 跨 date KS: tr→va=0.0444 / tr→te=0.1177
- 结论: cross_stock=strong_drift / cross_date=stable

### `rv_w5` (derived)
- 中文名: 实现波动率(窗口W=5tick≈15s)
- 英文: Realized Volatility over W=5 ticks (wmp_lvl1-based)
- X 列号 (0-based): 195
- KS-drop 11: no
- 公式: $$\mathrm{RV}_W = \sqrt{\max\left(0, \sum_{s=t-W+1}^{t} r_s^2\right)},\;r_s = \log\frac{P_s+1}{P_{s-1}+1}$$
- 代码定位: `fast_features_batch.py:216-230 (T3 RV loop on wmp_lvl1+1)`
- numpy 伪代码:
```python
safe = wmp_lvl1 + 1.0  # +1 to guard zero/negative
log_ret = np.log(safe[:,1:]/safe[:,:-1])
log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)
rv_w5 = np.sqrt(np.maximum((log_ret**2)[:, -5:].sum(-1), 0.0))
```
- 物理含义: T3 stage 计算的实现波动率：基于 wmp_lvl1 (1档加权中间价) 的对数收益平方和。W=5 ticks ≈ 15s 时间窗。RV 是经典 Andersen-Bollerslev 类指标，反映该窗口内中间价的累积波动幅度；高 RV 通常对应剧烈波动 / 大额成交 / 重大新闻冲击。NN/LGB 利用 RV 调整对 mid 走势的『风险下注幅度』。
- NaN 处理:
  - 公式层: np.where(safe>0, safe, np.nan) 防 log 负数; np.where(isfinite, lr, 0.0) 填零; sqrt(max(.,0))
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4206 / mean=0.1687 / by_sym=[0.0268, 0.0317, 0.1485, 0.4206, 0.2159]
- 跨 stock KS: max=0.3516 / by_sym=[0.0904, 0.1086, 0.1538, 0.3516, 0.2397]
- 跨 date PSI: tr→va=0.0205 / tr→te=0.1035
- 跨 date PSI by bucket: [('tr_d0_19', 0.0042), ('tr_d20_39', 0.0157), ('tr_d40_59', 0.0018), ('tr_d60_79', 0.0046), ('va_d80_95', 0.0207), ('te_d96_119', 0.0962)]
- 跨 date KS: tr→va=0.054 / tr→te=0.1266
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `rv_w10` (derived)
- 中文名: 实现波动率(窗口W=10tick≈30s)
- 英文: Realized Volatility over W=10 ticks (wmp_lvl1-based)
- X 列号 (0-based): 196
- KS-drop 11: no
- 公式: $$\mathrm{RV}_W = \sqrt{\max\left(0, \sum_{s=t-W+1}^{t} r_s^2\right)},\;r_s = \log\frac{P_s+1}{P_{s-1}+1}$$
- 代码定位: `fast_features_batch.py:216-230 (T3 RV loop on wmp_lvl1+1)`
- numpy 伪代码:
```python
safe = wmp_lvl1 + 1.0  # +1 to guard zero/negative
log_ret = np.log(safe[:,1:]/safe[:,:-1])
log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)
rv_w10 = np.sqrt(np.maximum((log_ret**2)[:, -10:].sum(-1), 0.0))
```
- 物理含义: T3 stage 计算的实现波动率：基于 wmp_lvl1 (1档加权中间价) 的对数收益平方和。W=10 ticks ≈ 30s 时间窗。RV 是经典 Andersen-Bollerslev 类指标，反映该窗口内中间价的累积波动幅度；高 RV 通常对应剧烈波动 / 大额成交 / 重大新闻冲击。NN/LGB 利用 RV 调整对 mid 走势的『风险下注幅度』。
- NaN 处理:
  - 公式层: np.where(safe>0, safe, np.nan) 防 log 负数; np.where(isfinite, lr, 0.0) 填零; sqrt(max(.,0))
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4801 / mean=0.1966 / by_sym=[0.0237, 0.069, 0.1496, 0.4801, 0.2604]
- 跨 stock KS: max=0.3793 / by_sym=[0.0659, 0.1484, 0.1245, 0.3793, 0.2329]
- 跨 date PSI: tr→va=0.0317 / tr→te=0.1119
- 跨 date PSI by bucket: [('tr_d0_19', 0.0037), ('tr_d20_39', 0.0158), ('tr_d40_59', 0.0026), ('tr_d60_79', 0.0074), ('va_d80_95', 0.0271), ('te_d96_119', 0.1063)]
- 跨 date KS: tr→va=0.0633 / tr→te=0.1236
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `rv_w20` (derived)
- 中文名: 实现波动率(窗口W=20tick≈60s)
- 英文: Realized Volatility over W=20 ticks (wmp_lvl1-based)
- X 列号 (0-based): 197
- KS-drop 11: no
- 公式: $$\mathrm{RV}_W = \sqrt{\max\left(0, \sum_{s=t-W+1}^{t} r_s^2\right)},\;r_s = \log\frac{P_s+1}{P_{s-1}+1}$$
- 代码定位: `fast_features_batch.py:216-230 (T3 RV loop on wmp_lvl1+1)`
- numpy 伪代码:
```python
safe = wmp_lvl1 + 1.0  # +1 to guard zero/negative
log_ret = np.log(safe[:,1:]/safe[:,:-1])
log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)
rv_w20 = np.sqrt(np.maximum((log_ret**2)[:, -20:].sum(-1), 0.0))
```
- 物理含义: T3 stage 计算的实现波动率：基于 wmp_lvl1 (1档加权中间价) 的对数收益平方和。W=20 ticks ≈ 60s 时间窗。RV 是经典 Andersen-Bollerslev 类指标，反映该窗口内中间价的累积波动幅度；高 RV 通常对应剧烈波动 / 大额成交 / 重大新闻冲击。NN/LGB 利用 RV 调整对 mid 走势的『风险下注幅度』。
- NaN 处理:
  - 公式层: np.where(safe>0, safe, np.nan) 防 log 负数; np.where(isfinite, lr, 0.0) 填零; sqrt(max(.,0))
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4932 / mean=0.2138 / by_sym=[0.0253, 0.1237, 0.1345, 0.4932, 0.2926]
- 跨 stock KS: max=0.3898 / by_sym=[0.0752, 0.1748, 0.0968, 0.3898, 0.2401]
- 跨 date PSI: tr→va=0.0434 / tr→te=0.1227
- 跨 date PSI by bucket: [('tr_d0_19', 0.0054), ('tr_d20_39', 0.0177), ('tr_d40_59', 0.0026), ('tr_d60_79', 0.0096), ('va_d80_95', 0.0383), ('te_d96_119', 0.1179)]
- 跨 date KS: tr→va=0.0734 / tr→te=0.1306
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `rv_w50` (derived)
- 中文名: 实现波动率(窗口W=50tick≈150s)
- 英文: Realized Volatility over W=50 ticks (wmp_lvl1-based)
- X 列号 (0-based): 198
- KS-drop 11: no
- 公式: $$\mathrm{RV}_W = \sqrt{\max\left(0, \sum_{s=t-W+1}^{t} r_s^2\right)},\;r_s = \log\frac{P_s+1}{P_{s-1}+1}$$
- 代码定位: `fast_features_batch.py:216-230 (T3 RV loop on wmp_lvl1+1)`
- numpy 伪代码:
```python
safe = wmp_lvl1 + 1.0  # +1 to guard zero/negative
log_ret = np.log(safe[:,1:]/safe[:,:-1])
log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)
rv_w50 = np.sqrt(np.maximum((log_ret**2)[:, -50:].sum(-1), 0.0))
```
- 物理含义: T3 stage 计算的实现波动率：基于 wmp_lvl1 (1档加权中间价) 的对数收益平方和。W=50 ticks ≈ 150s 时间窗。RV 是经典 Andersen-Bollerslev 类指标，反映该窗口内中间价的累积波动幅度；高 RV 通常对应剧烈波动 / 大额成交 / 重大新闻冲击。NN/LGB 利用 RV 调整对 mid 走势的『风险下注幅度』。
- NaN 处理:
  - 公式层: np.where(safe>0, safe, np.nan) 防 log 负数; np.where(isfinite, lr, 0.0) 填零; sqrt(max(.,0))
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.4398 / mean=0.2238 / by_sym=[0.0404, 0.1958, 0.0966, 0.4398, 0.3462]
- 跨 stock KS: max=0.3566 / by_sym=[0.1007, 0.1825, 0.083, 0.3566, 0.2698]
- 跨 date PSI: tr→va=0.0565 / tr→te=0.1392
- 跨 date PSI by bucket: [('tr_d0_19', 0.0091), ('tr_d20_39', 0.0242), ('tr_d40_59', 0.0035), ('tr_d60_79', 0.0142), ('va_d80_95', 0.0524), ('te_d96_119', 0.1367)]
- 跨 date KS: tr→va=0.0828 / tr→te=0.1493
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `signed_rv_W20` (derived)
- 中文名: 方向化实现波动率(W=20tick≈60s)
- 英文: Signed Realized Volatility over W=20 ticks
- X 列号 (0-based): 260
- KS-drop 11: no
- 公式: $$\mathrm{signedRV}_W = \frac{\sum r_s^2\mathbf{1}\{r_s>0\}-\sum r_s^2\mathbf{1}\{r_s<0\}}{\sum r_s^2\mathbf{1}\{r_s>0\}+\sum r_s^2\mathbf{1}\{r_s<0\}+\epsilon}\in[-1,1]$$
- 代码定位: `fast_features_batch.py:293-308 (compute_stage1_batch --- signed_rv)`
- numpy 伪代码:
```python
r = log_ret(midprice)  # 基于 midprice1
r2_pos = r**2 * (r>0); r2_neg = r**2 * (r<0)
rp = r2_pos[:, -20:].sum(-1); rn = r2_neg[:, -20:].sum(-1)
signed_rv_W20 = (rp-rn)/(rp+rn+1e-8)
```
- 物理含义: W=20 窗口内『涨方差 - 跌方差』除以总方差，∈[-1,1]。+1 = 该窗口完全由上涨贡献波动（涨主导），-1 = 完全由下跌贡献（跌主导）。是 mid 方向预测的强信号：与 mb_intst/ma_intst 互补——后者看意图（主动买/卖），signed_rv 看实际成交后的中间价方向。
- NaN 处理:
  - 公式层: (rp-rn)/(rp+rn+EPS) 防除零；上游 r=np.where(isfinite, r, 0)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1938 / mean=0.0491 / by_sym=[0.0009, 0.039, 0.0039, 0.1938, 0.0082]
- 跨 stock KS: max=0.1364 / by_sym=[0.0397, 0.0787, 0.0313, 0.1364, 0.0509]
- 跨 date PSI: tr→va=0.0094 / tr→te=0.0193
- 跨 date PSI by bucket: [('tr_d0_19', 0.0179), ('tr_d20_39', 0.0106), ('tr_d40_59', 0.0017), ('tr_d60_79', 0.0002), ('va_d80_95', 0.0076), ('te_d96_119', 0.0205)]
- 跨 date KS: tr→va=0.032 / tr→te=0.0437
- 结论: cross_stock=mild_drift / cross_date=stable

### `signed_rv_W50` (derived)
- 中文名: 方向化实现波动率(W=50tick≈150s)
- 英文: Signed Realized Volatility over W=50 ticks
- X 列号 (0-based): 261
- KS-drop 11: no
- 公式: $$\mathrm{signedRV}_W = \frac{\sum r_s^2\mathbf{1}\{r_s>0\}-\sum r_s^2\mathbf{1}\{r_s<0\}}{\sum r_s^2\mathbf{1}\{r_s>0\}+\sum r_s^2\mathbf{1}\{r_s<0\}+\epsilon}\in[-1,1]$$
- 代码定位: `fast_features_batch.py:293-308 (compute_stage1_batch --- signed_rv)`
- numpy 伪代码:
```python
r = log_ret(midprice)  # 基于 midprice1
r2_pos = r**2 * (r>0); r2_neg = r**2 * (r<0)
rp = r2_pos[:, -50:].sum(-1); rn = r2_neg[:, -50:].sum(-1)
signed_rv_W50 = (rp-rn)/(rp+rn+1e-8)
```
- 物理含义: W=50 窗口内『涨方差 - 跌方差』除以总方差，∈[-1,1]。+1 = 该窗口完全由上涨贡献波动（涨主导），-1 = 完全由下跌贡献（跌主导）。是 mid 方向预测的强信号：与 mb_intst/ma_intst 互补——后者看意图（主动买/卖），signed_rv 看实际成交后的中间价方向。
- NaN 处理:
  - 公式层: (rp-rn)/(rp+rn+EPS) 防除零；上游 r=np.where(isfinite, r, 0)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1006 / mean=0.0284 / by_sym=[0.0065, 0.0319, 0.0014, 0.1006, 0.0016]
- 跨 stock KS: max=0.1225 / by_sym=[0.0299, 0.079, 0.0222, 0.1225, 0.0285]
- 跨 date PSI: tr→va=0.0059 / tr→te=0.0175
- 跨 date PSI by bucket: [('tr_d0_19', 0.0167), ('tr_d20_39', 0.0071), ('tr_d40_59', 0.0027), ('tr_d60_79', 0.0002), ('va_d80_95', 0.0062), ('te_d96_119', 0.0166)]
- 跨 date KS: tr→va=0.0288 / tr→te=0.0549
- 结论: cross_stock=mild_drift / cross_date=stable

### `signed_rv_W100` (derived)
- 中文名: 方向化实现波动率(W=100tick≈300s)
- 英文: Signed Realized Volatility over W=100 ticks
- X 列号 (0-based): 262
- KS-drop 11: no
- 公式: $$\mathrm{signedRV}_W = \frac{\sum r_s^2\mathbf{1}\{r_s>0\}-\sum r_s^2\mathbf{1}\{r_s<0\}}{\sum r_s^2\mathbf{1}\{r_s>0\}+\sum r_s^2\mathbf{1}\{r_s<0\}+\epsilon}\in[-1,1]$$
- 代码定位: `fast_features_batch.py:293-308 (compute_stage1_batch --- signed_rv)`
- numpy 伪代码:
```python
r = log_ret(midprice)  # 基于 midprice1
r2_pos = r**2 * (r>0); r2_neg = r**2 * (r<0)
rp = r2_pos[:, -100:].sum(-1); rn = r2_neg[:, -100:].sum(-1)
signed_rv_W100 = (rp-rn)/(rp+rn+1e-8)
```
- 物理含义: W=100 窗口内『涨方差 - 跌方差』除以总方差，∈[-1,1]。+1 = 该窗口完全由上涨贡献波动（涨主导），-1 = 完全由下跌贡献（跌主导）。是 mid 方向预测的强信号：与 mb_intst/ma_intst 互补——后者看意图（主动买/卖），signed_rv 看实际成交后的中间价方向。
- NaN 处理:
  - 公式层: (rp-rn)/(rp+rn+EPS) 防除零；上游 r=np.where(isfinite, r, 0)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1557 / mean=0.0554 / by_sym=[0.0082, 0.066, 0.0275, 0.1557, 0.0197]
- 跨 stock KS: max=0.1071 / by_sym=[0.0334, 0.0832, 0.0174, 0.1071, 0.014]
- 跨 date PSI: tr→va=0.017 / tr→te=0.0362
- 跨 date PSI by bucket: [('tr_d0_19', 0.0249), ('tr_d20_39', 0.0126), ('tr_d40_59', 0.0046), ('tr_d60_79', 0.0004), ('va_d80_95', 0.0154), ('te_d96_119', 0.0327)]
- 跨 date KS: tr→va=0.0231 / tr→te=0.0543
- 结论: cross_stock=mild_drift / cross_date=stable

### `rskew_W20` (derived)
- 中文名: 对数收益偏度(W=20tick≈60s)
- 英文: Realized Skewness of log-returns over W=20 ticks
- X 列号 (0-based): 297
- KS-drop 11: no
- 公式: $$\mathrm{rskew}_W = \mathrm{clip}\left(\frac{\mu_3}{\sigma^3+\epsilon}, -10, 10\right),\;\mu_3 = \overline{r^3} - 3\overline{r}\,\overline{r^2}+2\overline{r}^3$$
- 代码定位: `fast_features_batch.py:374-393 (compute_stage2_batch --- skew)`
- numpy 伪代码:
```python
r = log_ret(midprice)
seg = r[:, -20:]; mu = seg.mean(-1); sd = seg.std(-1)+1e-8
m3 = (seg**3).mean(-1) - 3*mu*(seg**2).mean(-1) + 2*mu**3
rskew_W20 = np.clip(np.nan_to_num(m3/(sd**3+1e-8)), -10, 10)
```
- 物理含义: W=20 窗口内 log 收益的偏度（3 阶矩 / 3 阶标准差）。正偏 = 大涨少而显著、小跌频繁；负偏反之。对 mid 方向预测有领先性：负偏暗示『下跌尾部风险积累』，往往伴随后续向下突破。采用 sample 中心矩公式避免数值不稳，clip(±10) 防长尾爆炸。
- NaN 处理:
  - 公式层: denom (sd**3+EPS) 防除零；np.where(isfinite, skew, 0)；clip(±10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.189 / mean=0.0494 / by_sym=[0.0013, 0.0453, 0.0031, 0.189, 0.0084]
- 跨 stock KS: max=0.1346 / by_sym=[0.0311, 0.0742, 0.03, 0.1346, 0.0468]
- 跨 date PSI: tr→va=0.0089 / tr→te=0.0212
- 跨 date PSI by bucket: [('tr_d0_19', 0.0138), ('tr_d20_39', 0.01), ('tr_d40_59', 0.0016), ('tr_d60_79', 0.0001), ('va_d80_95', 0.0084), ('te_d96_119', 0.02)]
- 跨 date KS: tr→va=0.0288 / tr→te=0.0423
- 结论: cross_stock=mild_drift / cross_date=stable

### `rskew_W50` (derived)
- 中文名: 对数收益偏度(W=50tick≈150s)
- 英文: Realized Skewness of log-returns over W=50 ticks
- X 列号 (0-based): 298
- KS-drop 11: no
- 公式: $$\mathrm{rskew}_W = \mathrm{clip}\left(\frac{\mu_3}{\sigma^3+\epsilon}, -10, 10\right),\;\mu_3 = \overline{r^3} - 3\overline{r}\,\overline{r^2}+2\overline{r}^3$$
- 代码定位: `fast_features_batch.py:374-393 (compute_stage2_batch --- skew)`
- numpy 伪代码:
```python
r = log_ret(midprice)
seg = r[:, -50:]; mu = seg.mean(-1); sd = seg.std(-1)+1e-8
m3 = (seg**3).mean(-1) - 3*mu*(seg**2).mean(-1) + 2*mu**3
rskew_W50 = np.clip(np.nan_to_num(m3/(sd**3+1e-8)), -10, 10)
```
- 物理含义: W=50 窗口内 log 收益的偏度（3 阶矩 / 3 阶标准差）。正偏 = 大涨少而显著、小跌频繁；负偏反之。对 mid 方向预测有领先性：负偏暗示『下跌尾部风险积累』，往往伴随后续向下突破。采用 sample 中心矩公式避免数值不稳，clip(±10) 防长尾爆炸。
- NaN 处理:
  - 公式层: denom (sd**3+EPS) 防除零；np.where(isfinite, skew, 0)；clip(±10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.0933 / mean=0.0282 / by_sym=[0.0087, 0.0357, 0.0016, 0.0933, 0.0017]
- 跨 stock KS: max=0.1245 / by_sym=[0.0294, 0.0857, 0.0208, 0.1245, 0.026]
- 跨 date PSI: tr→va=0.0056 / tr→te=0.0187
- 跨 date PSI by bucket: [('tr_d0_19', 0.0126), ('tr_d20_39', 0.0067), ('tr_d40_59', 0.0024), ('tr_d60_79', 0.0002), ('va_d80_95', 0.0062), ('te_d96_119', 0.0199)]
- 跨 date KS: tr→va=0.0312 / tr→te=0.0526
- 结论: cross_stock=stable / cross_date=stable

### `rskew_W100` (derived)
- 中文名: 对数收益偏度(W=100tick≈300s)
- 英文: Realized Skewness of log-returns over W=100 ticks
- X 列号 (0-based): 299
- KS-drop 11: no
- 公式: $$\mathrm{rskew}_W = \mathrm{clip}\left(\frac{\mu_3}{\sigma^3+\epsilon}, -10, 10\right),\;\mu_3 = \overline{r^3} - 3\overline{r}\,\overline{r^2}+2\overline{r}^3$$
- 代码定位: `fast_features_batch.py:374-393 (compute_stage2_batch --- skew)`
- numpy 伪代码:
```python
r = log_ret(midprice)
seg = r[:, -100:]; mu = seg.mean(-1); sd = seg.std(-1)+1e-8
m3 = (seg**3).mean(-1) - 3*mu*(seg**2).mean(-1) + 2*mu**3
rskew_W100 = np.clip(np.nan_to_num(m3/(sd**3+1e-8)), -10, 10)
```
- 物理含义: W=100 窗口内 log 收益的偏度（3 阶矩 / 3 阶标准差）。正偏 = 大涨少而显著、小跌频繁；负偏反之。对 mid 方向预测有领先性：负偏暗示『下跌尾部风险积累』，往往伴随后续向下突破。采用 sample 中心矩公式避免数值不稳，clip(±10) 防长尾爆炸。
- NaN 处理:
  - 公式层: denom (sd**3+EPS) 防除零；np.where(isfinite, skew, 0)；clip(±10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1369 / mean=0.0549 / by_sym=[0.0092, 0.0568, 0.0341, 0.1369, 0.0375]
- 跨 stock KS: max=0.1135 / by_sym=[0.0322, 0.0982, 0.0224, 0.1135, 0.0156]
- 跨 date PSI: tr→va=0.011 / tr→te=0.034
- 跨 date PSI by bucket: [('tr_d0_19', 0.0244), ('tr_d20_39', 0.0146), ('tr_d40_59', 0.0036), ('tr_d60_79', 0.0007), ('va_d80_95', 0.0099), ('te_d96_119', 0.0326)]
- 跨 date KS: tr→va=0.0265 / tr→te=0.0555
- 结论: cross_stock=mild_drift / cross_date=stable

### `vol_burst_W20` (derived)
- 中文名: 成交量冲击比(W=20tick≈60s)
- 英文: Volume Burst Ratio (last tick vs mean over W=20 ticks)
- X 列号 (0-based): 332
- KS-drop 11: no
- 公式: $$\mathrm{vol\_burst}_W = \mathrm{clip}\left(\frac{\Delta V_t}{\overline{\Delta V}_{[t-W+1,t]}+\epsilon}, -100, 100\right)$$
- 代码定位: `fast_features_batch.py:455-466 (compute_stage2_batch --- vol_burst)`
- numpy 伪代码:
```python
mean_v = volume_delta[:, -20:].mean(-1) + 1e-8
vol_burst_W20 = np.clip(np.nan_to_num(volume_delta[:, -1]/mean_v), -100, 100)
```
- 物理含义: 当前 tick 成交量相对窗口均值的『冲击倍数』；W=20 给短期 (60s) 窗口。>>1 = 突发大单/连续放量（机构进出场信号）；<<1 = 当前 tick 异常清淡。对短期价格突变有领先性，常与 amt_burst / kyle_lam 联合解释流动性事件。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, ., 0)；clip(-100, 100)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2165 / mean=0.0726 / by_sym=[0.0219, 0.0419, 0.2165, 0.0558, 0.0268]
- 跨 stock KS: max=0.2452 / by_sym=[0.0839, 0.1352, 0.2452, 0.1357, 0.0891]
- 跨 date PSI: tr→va=0.0026 / tr→te=0.0278
- 跨 date PSI by bucket: [('tr_d0_19', 0.0094), ('tr_d20_39', 0.0068), ('tr_d40_59', 0.0006), ('tr_d60_79', 0.0003), ('va_d80_95', 0.0018), ('te_d96_119', 0.0319)]
- 跨 date KS: tr→va=0.0248 / tr→te=0.1067
- 结论: cross_stock=mild_drift / cross_date=stable

### `amt_burst_W20` (derived)
- 中文名: 成交额冲击比(W=20tick≈60s)
- 英文: Amount Burst Ratio (|Δamt_t| vs mean |Δamt| over W=20 ticks)
- X 列号 (0-based): 333
- KS-drop 11: no
- 公式: $$\mathrm{amt\_burst}_W = \mathrm{clip}\left(\frac{|\Delta A_t|}{\overline{|\Delta A|}_{[t-W+1,t]}+\epsilon},\,0,\,100\right)$$
- 代码定位: `fast_features_batch.py:455-468 (compute_stage2_batch --- amt_burst, 配对 vol_burst)`
- numpy 伪代码:
```python
mean_a = np.abs(amount_delta)[:, -20:].mean(-1) + 1e-8
amt_burst_W20 = np.clip(np.nan_to_num(np.abs(amount_delta[:, -1])/mean_a), 0, 100)
```
- 物理含义: 当前 tick |成交额| 相对窗口均值的冲击倍数 (∈[0,100])。与 vol_burst 互补——vol_burst 看股数（不分价格），amt_burst 看金额（含价格信息）。金额冲击比 = 异动检测，对识别大单/批量交易尤其敏感。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, ., 0)；clip(0, 100)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2164 / mean=0.0725 / by_sym=[0.0219, 0.0419, 0.2164, 0.0558, 0.0268]
- 跨 stock KS: max=0.245 / by_sym=[0.0838, 0.1372, 0.245, 0.1349, 0.0918]
- 跨 date PSI: tr→va=0.0026 / tr→te=0.0278
- 跨 date PSI by bucket: [('tr_d0_19', 0.0094), ('tr_d20_39', 0.0067), ('tr_d40_59', 0.0006), ('tr_d60_79', 0.0003), ('va_d80_95', 0.0018), ('te_d96_119', 0.032)]
- 跨 date KS: tr→va=0.0248 / tr→te=0.1067
- 结论: cross_stock=mild_drift / cross_date=stable

### `vol_burst_W50` (derived)
- 中文名: 成交量冲击比(W=50tick≈150s)
- 英文: Volume Burst Ratio (last tick vs mean over W=50 ticks)
- X 列号 (0-based): 334
- KS-drop 11: no
- 公式: $$\mathrm{vol\_burst}_W = \mathrm{clip}\left(\frac{\Delta V_t}{\overline{\Delta V}_{[t-W+1,t]}+\epsilon}, -100, 100\right)$$
- 代码定位: `fast_features_batch.py:455-466 (compute_stage2_batch --- vol_burst)`
- numpy 伪代码:
```python
mean_v = volume_delta[:, -50:].mean(-1) + 1e-8
vol_burst_W50 = np.clip(np.nan_to_num(volume_delta[:, -1]/mean_v), -100, 100)
```
- 物理含义: 当前 tick 成交量相对窗口均值的『冲击倍数』；W=50 给短期 (150s) 窗口。>>1 = 突发大单/连续放量（机构进出场信号）；<<1 = 当前 tick 异常清淡。对短期价格突变有领先性，常与 amt_burst / kyle_lam 联合解释流动性事件。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, ., 0)；clip(-100, 100)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2187 / mean=0.0751 / by_sym=[0.0222, 0.0395, 0.2187, 0.0661, 0.0288]
- 跨 stock KS: max=0.2506 / by_sym=[0.0845, 0.1358, 0.2506, 0.1509, 0.097]
- 跨 date PSI: tr→va=0.0023 / tr→te=0.0262
- 跨 date PSI by bucket: [('tr_d0_19', 0.0085), ('tr_d20_39', 0.0061), ('tr_d40_59', 0.0007), ('tr_d60_79', 0.0004), ('va_d80_95', 0.002), ('te_d96_119', 0.0294)]
- 跨 date KS: tr→va=0.0248 / tr→te=0.1068
- 结论: cross_stock=mild_drift / cross_date=stable

### `amt_burst_W50` (derived)
- 中文名: 成交额冲击比(W=50tick≈150s)
- 英文: Amount Burst Ratio (|Δamt_t| vs mean |Δamt| over W=50 ticks)
- X 列号 (0-based): 335
- KS-drop 11: no
- 公式: $$\mathrm{amt\_burst}_W = \mathrm{clip}\left(\frac{|\Delta A_t|}{\overline{|\Delta A|}_{[t-W+1,t]}+\epsilon},\,0,\,100\right)$$
- 代码定位: `fast_features_batch.py:455-468 (compute_stage2_batch --- amt_burst, 配对 vol_burst)`
- numpy 伪代码:
```python
mean_a = np.abs(amount_delta)[:, -50:].mean(-1) + 1e-8
amt_burst_W50 = np.clip(np.nan_to_num(np.abs(amount_delta[:, -1])/mean_a), 0, 100)
```
- 物理含义: 当前 tick |成交额| 相对窗口均值的冲击倍数 (∈[0,100])。与 vol_burst 互补——vol_burst 看股数（不分价格），amt_burst 看金额（含价格信息）。金额冲击比 = 异动检测，对识别大单/批量交易尤其敏感。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, ., 0)；clip(0, 100)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2186 / mean=0.0751 / by_sym=[0.0223, 0.0395, 0.2186, 0.0662, 0.0288]
- 跨 stock KS: max=0.2505 / by_sym=[0.0854, 0.137, 0.2505, 0.1511, 0.0983]
- 跨 date PSI: tr→va=0.0023 / tr→te=0.0262
- 跨 date PSI by bucket: [('tr_d0_19', 0.0085), ('tr_d20_39', 0.0061), ('tr_d40_59', 0.0007), ('tr_d60_79', 0.0004), ('va_d80_95', 0.002), ('te_d96_119', 0.0294)]
- 跨 date KS: tr→va=0.0248 / tr→te=0.1068
- 结论: cross_stock=mild_drift / cross_date=stable

### `rv_ratio_W5_W50` (derived)
- 中文名: 波动率比(5/50tick)
- 英文: Realized Variance Ratio short(W=5)/long(W=50)
- X 列号 (0-based): 337
- KS-drop 11: no
- 公式: $$\mathrm{rv\_ratio}_{W_n/W_d} = \mathrm{clip}\left(\frac{\sum_{s=t-W_n+1}^t r_s^2}{\sum_{s=t-W_d+1}^t r_s^2 + \epsilon}, 0, 100\right)$$
- 代码定位: `fast_features_batch.py:506-519 (compute_stage3_batch --- rv_ratio)`
- numpy 伪代码:
```python
r2 = log_ret(midprice)**2
rv_5 = r2[:, -5:].sum(-1); rv_50 = r2[:, -50:].sum(-1)
rv_ratio_W5_W50 = np.clip(np.nan_to_num(rv_5/(rv_50+1e-8)), 0, 100)
```
- 物理含义: 短窗 RV / 长窗 RV ∈ [0, 100]。>> Wn/Wd（无量纲尺度比）= 短期波动突增（波动率冲击），对识别『盘整→爆发』非常有效；NN 通过它做时间尺度的相对放大检测。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, ratio, 0)；clip(0, 100)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2155 / mean=0.0537 / by_sym=[0.0038, 0.0256, 0.0136, 0.2155, 0.0099]
- 跨 stock KS: max=0.2417 / by_sym=[0.0504, 0.0878, 0.1029, 0.2417, 0.1235]
- 跨 date PSI: tr→va=0.0049 / tr→te=0.0175
- 跨 date PSI by bucket: [('tr_d0_19', 0.0145), ('tr_d20_39', 0.0158), ('tr_d40_59', 0.0013), ('tr_d60_79', 0.0001), ('va_d80_95', 0.0044), ('te_d96_119', 0.016)]
- 跨 date KS: tr→va=0.0312 / tr→te=0.062
- 结论: cross_stock=mild_drift / cross_date=stable

### `rv_ratio_W20_W100` (derived)
- 中文名: 波动率比(20/100tick)
- 英文: Realized Variance Ratio short(W=20)/long(W=100)
- X 列号 (0-based): 338
- KS-drop 11: no
- 公式: $$\mathrm{rv\_ratio}_{W_n/W_d} = \mathrm{clip}\left(\frac{\sum_{s=t-W_n+1}^t r_s^2}{\sum_{s=t-W_d+1}^t r_s^2 + \epsilon}, 0, 100\right)$$
- 代码定位: `fast_features_batch.py:506-519 (compute_stage3_batch --- rv_ratio)`
- numpy 伪代码:
```python
r2 = log_ret(midprice)**2
rv_20 = r2[:, -20:].sum(-1); rv_100 = r2[:, -100:].sum(-1)
rv_ratio_W20_W100 = np.clip(np.nan_to_num(rv_20/(rv_100+1e-8)), 0, 100)
```
- 物理含义: 短窗 RV / 长窗 RV ∈ [0, 100]。>> Wn/Wd（无量纲尺度比）= 短期波动突增（波动率冲击），对识别『盘整→爆发』非常有效；NN 通过它做时间尺度的相对放大检测。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, ratio, 0)；clip(0, 100)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2005 / mean=0.0541 / by_sym=[0.0058, 0.0573, 0.0034, 0.2005, 0.0038]
- 跨 stock KS: max=0.2553 / by_sym=[0.0435, 0.1507, 0.0252, 0.2553, 0.0644]
- 跨 date PSI: tr→va=0.0129 / tr→te=0.0259
- 跨 date PSI by bucket: [('tr_d0_19', 0.0168), ('tr_d20_39', 0.0146), ('tr_d40_59', 0.0015), ('tr_d60_79', 0.0002), ('va_d80_95', 0.0106), ('te_d96_119', 0.0254)]
- 跨 date KS: tr→va=0.0559 / tr→te=0.0847
- 结论: cross_stock=mild_drift / cross_date=stable

### `rv_ratio_W50_W100` (derived)
- 中文名: 波动率比(50/100tick)
- 英文: Realized Variance Ratio short(W=50)/long(W=100)
- X 列号 (0-based): 339
- KS-drop 11: no
- 公式: $$\mathrm{rv\_ratio}_{W_n/W_d} = \mathrm{clip}\left(\frac{\sum_{s=t-W_n+1}^t r_s^2}{\sum_{s=t-W_d+1}^t r_s^2 + \epsilon}, 0, 100\right)$$
- 代码定位: `fast_features_batch.py:506-519 (compute_stage3_batch --- rv_ratio)`
- numpy 伪代码:
```python
r2 = log_ret(midprice)**2
rv_50 = r2[:, -50:].sum(-1); rv_100 = r2[:, -100:].sum(-1)
rv_ratio_W50_W100 = np.clip(np.nan_to_num(rv_50/(rv_100+1e-8)), 0, 100)
```
- 物理含义: 短窗 RV / 长窗 RV ∈ [0, 100]。>> Wn/Wd（无量纲尺度比）= 短期波动突增（波动率冲击），对识别『盘整→爆发』非常有效；NN 通过它做时间尺度的相对放大检测。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, ratio, 0)；clip(0, 100)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1142 / mean=0.0349 / by_sym=[0.0085, 0.0486, 0.0026, 0.1142, 0.0006]
- 跨 stock KS: max=0.2113 / by_sym=[0.0597, 0.1473, 0.0296, 0.2113, 0.0238]
- 跨 date PSI: tr→va=0.0104 / tr→te=0.0235
- 跨 date PSI by bucket: [('tr_d0_19', 0.0139), ('tr_d20_39', 0.0117), ('tr_d40_59', 0.0018), ('tr_d60_79', 0.0008), ('va_d80_95', 0.0101), ('te_d96_119', 0.0229)]
- 跨 date KS: tr→va=0.0545 / tr→te=0.09
- 结论: cross_stock=mild_drift / cross_date=stable

### `jshare_W20` (derived)
- 中文名: 跳跃份额(BNS, W=20tick≈60s)
- 英文: Barndorff-Nielsen-Shephard Jump Share over W=20 ticks
- X 列号 (0-based): 340
- KS-drop 11: no
- 公式: $$\mathrm{jshare}_W = \mathrm{clip}\left(\frac{\mathrm{RV}_W - \mathrm{BV}_W}{\mathrm{RV}_W + \epsilon},0,1\right),\;\mathrm{BV}_W = \frac{\pi}{2}\sum_{s=t-W+2}^{t} |r_s||r_{s-1}|$$
- 代码定位: `fast_features_batch.py:521-533 (compute_stage3_batch --- jshare)`
- numpy 伪代码:
```python
abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1)
bv = (np.pi/2) * (abs_r*abs_r_lag)[:, -20:].sum(-1)
rv = (r**2)[:, -20:].sum(-1)
jshare_W20 = np.clip(np.nan_to_num((rv-bv)/(rv+1e-8)), 0, 1)
```
- 物理含义: BNS (2004) 跳跃份额 ∈ [0,1]：RV 包含跳跃 + 连续波动；BV (Bipower Variation) 只对连续部分稳健。(RV-BV)/RV 即跳跃占总方差比例。π/2 是 |r||r_lag| 在 Brownian motion 下的方差归一化因子。高 jshare = 价格存在不连续跳跃（如撮合大单冲击），对预测后续 mean-reversion / 跟随有意义。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, j, 0)；clip(0, 1)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2412 / mean=0.1008 / by_sym=[0.0521, 0.0712, 0.0873, 0.2412, 0.0524]
- 跨 stock KS: max=0.2343 / by_sym=[0.0987, 0.1576, 0.0903, 0.2343, 0.0803]
- 跨 date PSI: tr→va=0.0158 / tr→te=0.0264
- 跨 date PSI by bucket: [('tr_d0_19', 0.0167), ('tr_d20_39', 0.0146), ('tr_d40_59', 0.0016), ('tr_d60_79', 0.0001), ('va_d80_95', 0.0166), ('te_d96_119', 0.0251)]
- 跨 date KS: tr→va=0.0566 / tr→te=0.0793
- 结论: cross_stock=mild_drift / cross_date=stable

### `jshare_W30` (derived)
- 中文名: 跳跃份额(BNS, W=30tick≈90s)
- 英文: Barndorff-Nielsen-Shephard Jump Share over W=30 ticks
- X 列号 (0-based): 341
- KS-drop 11: no
- 公式: $$\mathrm{jshare}_W = \mathrm{clip}\left(\frac{\mathrm{RV}_W - \mathrm{BV}_W}{\mathrm{RV}_W + \epsilon},0,1\right),\;\mathrm{BV}_W = \frac{\pi}{2}\sum_{s=t-W+2}^{t} |r_s||r_{s-1}|$$
- 代码定位: `fast_features_batch.py:521-533 (compute_stage3_batch --- jshare)`
- numpy 伪代码:
```python
abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1)
bv = (np.pi/2) * (abs_r*abs_r_lag)[:, -30:].sum(-1)
rv = (r**2)[:, -30:].sum(-1)
jshare_W30 = np.clip(np.nan_to_num((rv-bv)/(rv+1e-8)), 0, 1)
```
- 物理含义: BNS (2004) 跳跃份额 ∈ [0,1]：RV 包含跳跃 + 连续波动；BV (Bipower Variation) 只对连续部分稳健。(RV-BV)/RV 即跳跃占总方差比例。π/2 是 |r||r_lag| 在 Brownian motion 下的方差归一化因子。高 jshare = 价格存在不连续跳跃（如撮合大单冲击），对预测后续 mean-reversion / 跟随有意义。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, j, 0)；clip(0, 1)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2078 / mean=0.0992 / by_sym=[0.0641, 0.0786, 0.0875, 0.2078, 0.0578]
- 跨 stock KS: max=0.2271 / by_sym=[0.1152, 0.1676, 0.0927, 0.2271, 0.0812]
- 跨 date PSI: tr→va=0.0141 / tr→te=0.0244
- 跨 date PSI by bucket: [('tr_d0_19', 0.0138), ('tr_d20_39', 0.012), ('tr_d40_59', 0.0015), ('tr_d60_79', 0.0001), ('va_d80_95', 0.0149), ('te_d96_119', 0.0228)]
- 跨 date KS: tr→va=0.0593 / tr→te=0.0832
- 结论: cross_stock=mild_drift / cross_date=stable

### `jshare_W50` (derived)
- 中文名: 跳跃份额(BNS, W=50tick≈150s)
- 英文: Barndorff-Nielsen-Shephard Jump Share over W=50 ticks
- X 列号 (0-based): 342
- KS-drop 11: no
- 公式: $$\mathrm{jshare}_W = \mathrm{clip}\left(\frac{\mathrm{RV}_W - \mathrm{BV}_W}{\mathrm{RV}_W + \epsilon},0,1\right),\;\mathrm{BV}_W = \frac{\pi}{2}\sum_{s=t-W+2}^{t} |r_s||r_{s-1}|$$
- 代码定位: `fast_features_batch.py:521-533 (compute_stage3_batch --- jshare)`
- numpy 伪代码:
```python
abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1)
bv = (np.pi/2) * (abs_r*abs_r_lag)[:, -50:].sum(-1)
rv = (r**2)[:, -50:].sum(-1)
jshare_W50 = np.clip(np.nan_to_num((rv-bv)/(rv+1e-8)), 0, 1)
```
- 物理含义: BNS (2004) 跳跃份额 ∈ [0,1]：RV 包含跳跃 + 连续波动；BV (Bipower Variation) 只对连续部分稳健。(RV-BV)/RV 即跳跃占总方差比例。π/2 是 |r||r_lag| 在 Brownian motion 下的方差归一化因子。高 jshare = 价格存在不连续跳跃（如撮合大单冲击），对预测后续 mean-reversion / 跟随有意义。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, j, 0)；clip(0, 1)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1603 / mean=0.1002 / by_sym=[0.0814, 0.0838, 0.1048, 0.1603, 0.0704]
- 跨 stock KS: max=0.2033 / by_sym=[0.1431, 0.1704, 0.1178, 0.2033, 0.0859]
- 跨 date PSI: tr→va=0.0144 / tr→te=0.0202
- 跨 date PSI by bucket: [('tr_d0_19', 0.0097), ('tr_d20_39', 0.0102), ('tr_d40_59', 0.0022), ('tr_d60_79', 0.0), ('va_d80_95', 0.0133), ('te_d96_119', 0.0196)]
- 跨 date KS: tr→va=0.059 / tr→te=0.0862
- 结论: cross_stock=mild_drift / cross_date=stable

### `jshare_W100` (derived)
- 中文名: 跳跃份额(BNS, W=100tick≈300s)
- 英文: Barndorff-Nielsen-Shephard Jump Share over W=100 ticks
- X 列号 (0-based): 343
- KS-drop 11: no
- 公式: $$\mathrm{jshare}_W = \mathrm{clip}\left(\frac{\mathrm{RV}_W - \mathrm{BV}_W}{\mathrm{RV}_W + \epsilon},0,1\right),\;\mathrm{BV}_W = \frac{\pi}{2}\sum_{s=t-W+2}^{t} |r_s||r_{s-1}|$$
- 代码定位: `fast_features_batch.py:521-533 (compute_stage3_batch --- jshare)`
- numpy 伪代码:
```python
abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1)
bv = (np.pi/2) * (abs_r*abs_r_lag)[:, -100:].sum(-1)
rv = (r**2)[:, -100:].sum(-1)
jshare_W100 = np.clip(np.nan_to_num((rv-bv)/(rv+1e-8)), 0, 1)
```
- 物理含义: BNS (2004) 跳跃份额 ∈ [0,1]：RV 包含跳跃 + 连续波动；BV (Bipower Variation) 只对连续部分稳健。(RV-BV)/RV 即跳跃占总方差比例。π/2 是 |r||r_lag| 在 Brownian motion 下的方差归一化因子。高 jshare = 价格存在不连续跳跃（如撮合大单冲击），对预测后续 mean-reversion / 跟随有意义。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, j, 0)；clip(0, 1)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1532 / mean=0.1283 / by_sym=[0.127, 0.0914, 0.1488, 0.1532, 0.1209]
- 跨 stock KS: max=0.1684 / by_sym=[0.1682, 0.1684, 0.1487, 0.1548, 0.1156]
- 跨 date PSI: tr→va=0.0165 / tr→te=0.0332
- 跨 date PSI by bucket: [('tr_d0_19', 0.0226), ('tr_d20_39', 0.0169), ('tr_d40_59', 0.0066), ('tr_d60_79', 0.001), ('va_d80_95', 0.0151), ('te_d96_119', 0.0304)]
- 跨 date KS: tr→va=0.0467 / tr→te=0.09
- 结论: cross_stock=mild_drift / cross_date=stable

### `roll_eff_spr_ratio_W30` (derived)
- 中文名: Roll有效价差比(W=30tick≈90s)
- 英文: Roll (1984) Effective Spread Ratio over W=30 ticks
- X 列号 (0-based): 347
- KS-drop 11: no
- 公式: $$\mathrm{roll\_eff\_spr\_ratio}_W = \mathrm{clip}\left(\frac{2\sqrt{\max(0, -\mathrm{cov}(\Delta P_t,\Delta P_{t-1}))}}{\mathrm{spread1}_t + 1}, 0, 10\right)$$
- 代码定位: `fast_features_batch.py:556-576 (compute_stage3_batch --- roll_eff_spr)`
- numpy 伪代码:
```python
dc = np.diff(close, prepend=close[:,:1])  # ΔP
dc_lag = shift(dc, 1)
cov_W = (dc*dc_lag)[:, -30:].mean(-1) - dc[:, -30:].mean(-1)*dc_lag[:, -30:].mean(-1)
eff = 2 * np.sqrt(np.maximum(-cov_W, 0))
roll_eff_spr_ratio_W30 = np.clip(np.nan_to_num(eff/(spread1[:,-1]+1+1e-8)), 0, 10)
```
- 物理含义: Roll 1984 经典：在『无信息流交易』模型下，价格变化序列的负自协方差直接编码 bid-ask 弹跳幅度。-cov(ΔP, ΔP_lag) > 0 时，2√(-cov) 即有效价差估计；再除以名义 spread1 得『相对成本』。高 ratio = 隐性流动性成本远高于 quote spread → 可能预示反转/参与者撤出。⚠️ KS-drop 11 之一（W=100 版本）：train→test KS 高于阈值，被训练管线 mask；W=30/50 保留。
- NaN 处理:
  - 公式层: max(-cov, 0) 防 sqrt 负值；+EPS 防除零；np.where(isfinite, ., 0)；clip(0, 10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.8758 / mean=0.3743 / by_sym=[0.198, 0.196, 0.4101, 0.8758, 0.1918]
- 跨 stock KS: max=0.3238 / by_sym=[0.1304, 0.1926, 0.3238, 0.2165, 0.1474]
- 跨 date PSI: tr→va=0.0181 / tr→te=0.0981
- 跨 date PSI by bucket: [('tr_d0_19', 0.0212), ('tr_d20_39', 0.0214), ('tr_d40_59', 0.0005), ('tr_d60_79', 0.0082), ('va_d80_95', 0.0159), ('te_d96_119', 0.1045)]
- 跨 date KS: tr→va=0.0527 / tr→te=0.1424
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `roll_eff_spr_ratio_W50` (derived)
- 中文名: Roll有效价差比(W=50tick≈150s)
- 英文: Roll (1984) Effective Spread Ratio over W=50 ticks
- X 列号 (0-based): 348
- KS-drop 11: no
- 公式: $$\mathrm{roll\_eff\_spr\_ratio}_W = \mathrm{clip}\left(\frac{2\sqrt{\max(0, -\mathrm{cov}(\Delta P_t,\Delta P_{t-1}))}}{\mathrm{spread1}_t + 1}, 0, 10\right)$$
- 代码定位: `fast_features_batch.py:556-576 (compute_stage3_batch --- roll_eff_spr)`
- numpy 伪代码:
```python
dc = np.diff(close, prepend=close[:,:1])  # ΔP
dc_lag = shift(dc, 1)
cov_W = (dc*dc_lag)[:, -50:].mean(-1) - dc[:, -50:].mean(-1)*dc_lag[:, -50:].mean(-1)
eff = 2 * np.sqrt(np.maximum(-cov_W, 0))
roll_eff_spr_ratio_W50 = np.clip(np.nan_to_num(eff/(spread1[:,-1]+1+1e-8)), 0, 10)
```
- 物理含义: Roll 1984 经典：在『无信息流交易』模型下，价格变化序列的负自协方差直接编码 bid-ask 弹跳幅度。-cov(ΔP, ΔP_lag) > 0 时，2√(-cov) 即有效价差估计；再除以名义 spread1 得『相对成本』。高 ratio = 隐性流动性成本远高于 quote spread → 可能预示反转/参与者撤出。⚠️ KS-drop 11 之一（W=100 版本）：train→test KS 高于阈值，被训练管线 mask；W=30/50 保留。
- NaN 处理:
  - 公式层: max(-cov, 0) 防 sqrt 负值；+EPS 防除零；np.where(isfinite, ., 0)；clip(0, 10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.8519 / mean=0.3872 / by_sym=[0.2712, 0.2319, 0.4088, 0.8519, 0.1722]
- 跨 stock KS: max=0.3518 / by_sym=[0.1483, 0.1881, 0.3518, 0.2435, 0.1695]
- 跨 date PSI: tr→va=0.0184 / tr→te=0.1083
- 跨 date PSI by bucket: [('tr_d0_19', 0.0253), ('tr_d20_39', 0.0262), ('tr_d40_59', 0.0012), ('tr_d60_79', 0.0118), ('va_d80_95', 0.014), ('te_d96_119', 0.1097)]
- 跨 date KS: tr→va=0.0614 / tr→te=0.1576
- 结论: cross_stock=strong_drift / cross_date=mild_drift

### `roll_eff_spr_ratio_W100` (derived)
- 中文名: Roll有效价差比(W=100tick≈300s)
- 英文: Roll (1984) Effective Spread Ratio over W=100 ticks
- X 列号 (0-based): 349
- KS-drop 11: **YES**
- 公式: $$\mathrm{roll\_eff\_spr\_ratio}_W = \mathrm{clip}\left(\frac{2\sqrt{\max(0, -\mathrm{cov}(\Delta P_t,\Delta P_{t-1}))}}{\mathrm{spread1}_t + 1}, 0, 10\right)$$
- 代码定位: `fast_features_batch.py:556-576 (compute_stage3_batch --- roll_eff_spr)`
- numpy 伪代码:
```python
dc = np.diff(close, prepend=close[:,:1])  # ΔP
dc_lag = shift(dc, 1)
cov_W = (dc*dc_lag)[:, -100:].mean(-1) - dc[:, -100:].mean(-1)*dc_lag[:, -100:].mean(-1)
eff = 2 * np.sqrt(np.maximum(-cov_W, 0))
roll_eff_spr_ratio_W100 = np.clip(np.nan_to_num(eff/(spread1[:,-1]+1+1e-8)), 0, 10)
```
- 物理含义: Roll 1984 经典：在『无信息流交易』模型下，价格变化序列的负自协方差直接编码 bid-ask 弹跳幅度。-cov(ΔP, ΔP_lag) > 0 时，2√(-cov) 即有效价差估计；再除以名义 spread1 得『相对成本』。高 ratio = 隐性流动性成本远高于 quote spread → 可能预示反转/参与者撤出。⚠️ KS-drop 11 之一（W=100 版本）：train→test KS 高于阈值，被训练管线 mask；W=30/50 保留。
- NaN 处理:
  - 公式层: max(-cov, 0) 防 sqrt 负值；+EPS 防除零；np.where(isfinite, ., 0)；clip(0, 10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=1.0539 / mean=0.4779 / by_sym=[0.347, 0.3084, 0.4795, 1.0539, 0.2005]
- 跨 stock KS: max=0.3922 / by_sym=[0.1734, 0.1803, 0.3922, 0.2825, 0.2093]
- 跨 date PSI: tr→va=0.0212 / tr→te=0.1446
- 跨 date PSI by bucket: [('tr_d0_19', 0.0285), ('tr_d20_39', 0.033), ('tr_d40_59', 0.0009), ('tr_d60_79', 0.0182), ('va_d80_95', 0.0194), ('te_d96_119', 0.1463)]
- 跨 date KS: tr→va=0.0647 / tr→te=0.1747
- 结论: cross_stock=strong_drift / cross_date=mild_drift
- 备注: KS-drop 11 member (W=100, fail train→test KS)

### `signed_bv_W20` (derived)
- 中文名: 方向化双幂变差(W=20tick≈60s)
- 英文: Signed Bipower Variation Ratio over W=20 ticks
- X 列号 (0-based): 357
- KS-drop 11: no
- 公式: $$\mathrm{signed\_bv}_W = \mathrm{clip}\left(\frac{\frac{\pi}{2}\sum_{s=t-W+1}^{t}\mathrm{sign}(r_s)|r_s||r_{s-1}|}{\sum r_s^2 + \epsilon},-10,10\right)$$
- 代码定位: `fast_features_batch.py:671-685 (compute_stage5_batch --- C. Multi-scale signed bipower)`
- numpy 伪代码:
```python
abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1); sign_r = np.sign(r)
sbv = (np.pi/2) * (sign_r*abs_r*abs_r_lag)[:, -20:].sum(-1)
rv = (r**2)[:, -20:].sum(-1) + 1e-8
signed_bv_W20 = np.clip(np.nan_to_num(sbv/rv), -10, 10)
```
- 物理含义: 对 jshare 的方向化版本：保留 sign(r_t)，把连续段（BV-like）的方向信号汇总后再除 RV 归一。>0 = 该窗口『连续波动』偏涨主导；<0 = 偏跌主导。与 signed_rv 互补——signed_rv 用 r²（强调极值），signed_bv 用 |r||r_lag|（更稳健、抑制单点跳跃）。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, v, 0)；clip(-10, 10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2726 / mean=0.0777 / by_sym=[0.0184, 0.0177, 0.0335, 0.2726, 0.0464]
- 跨 stock KS: max=0.1241 / by_sym=[0.0429, 0.0438, 0.0612, 0.1241, 0.0741]
- 跨 date PSI: tr→va=0.0052 / tr→te=0.015
- 跨 date PSI by bucket: [('tr_d0_19', 0.0222), ('tr_d20_39', 0.0156), ('tr_d40_59', 0.0005), ('tr_d60_79', 0.0003), ('va_d80_95', 0.0036), ('te_d96_119', 0.0179)]
- 跨 date KS: tr→va=0.0191 / tr→te=0.0305
- 结论: cross_stock=strong_drift / cross_date=stable

### `signed_bv_W50` (derived)
- 中文名: 方向化双幂变差(W=50tick≈150s)
- 英文: Signed Bipower Variation Ratio over W=50 ticks
- X 列号 (0-based): 358
- KS-drop 11: no
- 公式: $$\mathrm{signed\_bv}_W = \mathrm{clip}\left(\frac{\frac{\pi}{2}\sum_{s=t-W+1}^{t}\mathrm{sign}(r_s)|r_s||r_{s-1}|}{\sum r_s^2 + \epsilon},-10,10\right)$$
- 代码定位: `fast_features_batch.py:671-685 (compute_stage5_batch --- C. Multi-scale signed bipower)`
- numpy 伪代码:
```python
abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1); sign_r = np.sign(r)
sbv = (np.pi/2) * (sign_r*abs_r*abs_r_lag)[:, -50:].sum(-1)
rv = (r**2)[:, -50:].sum(-1) + 1e-8
signed_bv_W50 = np.clip(np.nan_to_num(sbv/rv), -10, 10)
```
- 物理含义: 对 jshare 的方向化版本：保留 sign(r_t)，把连续段（BV-like）的方向信号汇总后再除 RV 归一。>0 = 该窗口『连续波动』偏涨主导；<0 = 偏跌主导。与 signed_rv 互补——signed_rv 用 r²（强调极值），signed_bv 用 |r||r_lag|（更稳健、抑制单点跳跃）。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, v, 0)；clip(-10, 10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.2054 / mean=0.0547 / by_sym=[0.0052, 0.0353, 0.0094, 0.2054, 0.018]
- 跨 stock KS: max=0.1373 / by_sym=[0.0244, 0.0887, 0.0368, 0.1373, 0.0557]
- 跨 date PSI: tr→va=0.011 / tr→te=0.0229
- 跨 date PSI by bucket: [('tr_d0_19', 0.0192), ('tr_d20_39', 0.0145), ('tr_d40_59', 0.0009), ('tr_d60_79', 0.0007), ('va_d80_95', 0.0095), ('te_d96_119', 0.0222)]
- 跨 date KS: tr→va=0.03 / tr→te=0.0435
- 结论: cross_stock=mild_drift / cross_date=stable

### `signed_bv_W100` (derived)
- 中文名: 方向化双幂变差(W=100tick≈300s)
- 英文: Signed Bipower Variation Ratio over W=100 ticks
- X 列号 (0-based): 359
- KS-drop 11: no
- 公式: $$\mathrm{signed\_bv}_W = \mathrm{clip}\left(\frac{\frac{\pi}{2}\sum_{s=t-W+1}^{t}\mathrm{sign}(r_s)|r_s||r_{s-1}|}{\sum r_s^2 + \epsilon},-10,10\right)$$
- 代码定位: `fast_features_batch.py:671-685 (compute_stage5_batch --- C. Multi-scale signed bipower)`
- numpy 伪代码:
```python
abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1); sign_r = np.sign(r)
sbv = (np.pi/2) * (sign_r*abs_r*abs_r_lag)[:, -100:].sum(-1)
rv = (r**2)[:, -100:].sum(-1) + 1e-8
signed_bv_W100 = np.clip(np.nan_to_num(sbv/rv), -10, 10)
```
- 物理含义: 对 jshare 的方向化版本：保留 sign(r_t)，把连续段（BV-like）的方向信号汇总后再除 RV 归一。>0 = 该窗口『连续波动』偏涨主导；<0 = 偏跌主导。与 signed_rv 互补——signed_rv 用 r²（强调极值），signed_bv 用 |r||r_lag|（更稳健、抑制单点跳跃）。
- NaN 处理:
  - 公式层: +EPS 防除零；np.where(isfinite, v, 0)；clip(-10, 10)
  - NN 管线: wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)
  - LGB 管线: 不动
- NaN%: train=0.0% / val=0.0% / test=0.0%
- 跨 stock PSI: max=0.1465 / mean=0.04 / by_sym=[0.0071, 0.0351, 0.0025, 0.1465, 0.0089]
- 跨 stock KS: max=0.1309 / by_sym=[0.0284, 0.1095, 0.0177, 0.1309, 0.042]
- 跨 date PSI: tr→va=0.0112 / tr→te=0.0226
- 跨 date PSI by bucket: [('tr_d0_19', 0.0154), ('tr_d20_39', 0.01), ('tr_d40_59', 0.0004), ('tr_d60_79', 0.0013), ('va_d80_95', 0.0118), ('te_d96_119', 0.0248)]
- 跨 date KS: tr→va=0.033 / tr→te=0.047
- 结论: cross_stock=mild_drift / cross_date=stable
