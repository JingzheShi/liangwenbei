# Batch 内统计泄露审计报告

## 总结

- **结论**：✅ **安全**。没有发现任何 batch 内跨样本统计泄露 / 非法状态依赖。
- **审计范围**：9 个 .py 文件，3830 行代码。
  - `04_build_pkg/Predictor.py` (核心 inference, 388 行)
  - `04_build_pkg/fast_features.py` (single-window 旧版, 568 行 —— **未在 inference 路径中调用**)
  - `04_build_pkg/fast_features_batch.py` (inference 实际特征构造, 780 行)
  - `04_build_pkg/build_pkg.py` (构建期, 136 行)
  - `04_build_pkg/config.json` / `thresholds.json` (常量配置)
  - `01_build_features/fast_features_batch.py` (训练特征, 780 行, **byte-identical 与 04_build_pkg 版本**)
  - `01_build_features/stage5_features.py` (205 行)
  - `01_build_features/build_schemeP_cache.py` (241 行)
  - `02_train_lgb/train_T188v2_lgb_seed.py` (208 行)
  - `03_train_nn/train_T188v2_nn_seed.py` (524 行)
- **发现**：0 处 🔴 高风险 / 0 处 🟡 中风险 / 多处 ✅ 安全确认
- **核心判定**：
  - Predictor.predict 不持有跨 batch state（`pred_cache` 是 per-call 局部变量）
  - 所有 reductions 都沿 **time 维（window 内 100 步）** 或 **hidden 维**，从未沿 sample 维
  - 所有 sigma/beta/feat_mean/feat_std/target_scale 都是**训练时算好的常量**，inference 时只 load 不重算
  - 没有 BatchNorm、softmax/dim=0、F.normalize、TTA/TTT、adaptive wrapper、batch-rank、cross-sample agreement vote 等任何 batch-stat 模式

---

## 1. Predictor.py（核心 inference）

### 1.1 `predict()` 主流程逐步追踪（line 316-365）

| 行 | 操作 | 数据形状 | axis | 判定 |
|---|---|---|---|---|
| 319 | `feats = self._compute_batch_features(batches)` | → (B, 154+205) | per-row | ✅ 见 §1.5 |
| 324 | `band_per_row = self._extract_band_per_row(batches)` | (B,) | per-row sym lookup | ✅ 见 §1.2 |
| 329 | `pred_cache: Dict[int, np.ndarray] = {}` | **per-call 局部** | — | ✅ **不跨 predict() 共享** |
| 348 | `_ensemble_predict_lgb(lgbs, feats)` | → (B,) | per-row 树推断 | ✅ |
| 351 | `nn_ens.predict_mean(feats)` | → (B,) | per-row bmm | ✅ 见 §1.4 |
| 355 | `np.stack(preds, axis=0)` | (M=2, B) | stack 模型类型 | ✅ 仅打包 LGB/NN 两条预测 |
| 357 | `(stacked * ws_arr).sum(axis=0) / ws_arr.sum()` | (B,) | sum 沿 **model 维** | ✅ M=2 加权和，不是 batch sum |
| 358 | `pred_cache[src_H] = pred_dmid` | — | 缓存到局部 dict | ✅ 仅给 share_with 重用，依然 per-row |
| 361 / 363 | `_gate_with_band(...)` / `_ev_gate_predict(...)` | (B,) | elementwise compare | ✅ 见 §1.6 |

**🟢 没有任何对 batch 维（axis 0 of (B,...) 形状）的 reduction。**

### 1.2 Conformal abstain (`_extract_band_per_row`, line 291-304)

- 每行独立读取 `df['sym'].iloc[-1]`（窗口最后一 tick 的 sym ID）
- 查 `self._cw_band[s]`，该 dict 由 `__init__` 从 **thresholds.json 常量** 构造：
  - `cw_band[k] = per_sym_beta[k] * per_sym_sigma[k]`，两个都是 thresholds.json 写死的浮点常量
  - per_sym_sigma 是**训练时**算好的（per-sym validation residual std），写入 thresholds.json，inference 期 **只 load 不重算**
- OOD sym → `default_beta_for_ood * default_sigma_for_ood`（也是常量）
- **判定**：✅ sigma/beta 完全来自常量配置；从未对当前 batch 的 pred 做 std/rank/quantile 来构造 band

### 1.3 Gate functions (`_gate_with_band`, `_ev_gate_predict`, line 255-271)

```python
out[pred_dmid > (thr_up + band_per_row)] = 2
out[pred_dmid < -(thr_dn + band_per_row)] = 0
```

- `thr_up` / `thr_dn` 来自 thresholds.json 常量（每个 horizon 独立）
- `band_per_row` 是 (B,) 向量，来自 §1.2，每行只依赖该行的 sym（来自常量表）
- 全是 **elementwise compare**
- **判定**：✅ 完全 per-sample；同一 row 单独放进 batch 或多 batch 共存，结果完全一致

### 1.4 NN Ensemble forward (`_BatchedMLPEnsemble.predict_mean`, line 145-178)

| 行 | 操作 | 形状 | dim/axis | 判定 |
|---|---|---|---|---|
| 154 | `h = torch.from_numpy(Xs).to(device)` | (B, in_dim) | — | ✅ |
| 155 | `h.unsqueeze(0).expand(N, -1, -1)` | (N=50, B, in_dim) | 复制到 N=50 NN 维 | ✅ |
| 158 | `(h - feat_mean.unsqueeze(1)) / feat_std.unsqueeze(1)` | (N, B, in_dim) | **per-NN 训练时常量** broadcast 到所有 row | ✅ 见 §3.1 |
| 159 | `torch.where(torch.isnan(h), zeros, h)` | (N, B, in_dim) | elementwise | ✅ |
| 160 | `torch.clamp(h, -clip, clip)` | (N, B, in_dim) | elementwise | ✅ |
| 164 | `torch.bmm(h, W[i].T) + b[i]` | (N, B, hid_out) | bmm 沿 **in_dim**，bias 沿 hidden | ✅ |
| 167 | `mean = h.mean(dim=-1, keepdim=True)` | (N, B, 1) | **dim=-1 = hidden_out 维** | ✅ LayerNorm，不跨 batch |
| 168 | `var = h.var(dim=-1, keepdim=True, unbiased=False)` | (N, B, 1) | **dim=-1 = hidden 维** | ✅ |
| 169 | `(h - mean) / sqrt(var + 1e-5)` | (N, B, hid) | elementwise | ✅ |
| 170 | `h * LNW[i].unsqueeze(1) + LNb[i].unsqueeze(1)` | (N, B, hid) | gamma/beta 是常量 | ✅ |
| 171 | `F.gelu(h, ...)` | (N, B, hid) | elementwise | ✅ |
| 174 | `torch.bmm(h, WF.T).squeeze(-1)` | (N, B) | bmm 沿 hidden_last | ✅ |
| 175 | `out + bF` | (N, B) | bF (N,1) broadcast | ✅ |
| 176 | `out / target_scale` | (N, B) | target_scale (N,1) 常量 | ✅ |
| 177 | `out = out.mean(0)` | (B,) | **dim=0 = N=50 NN 维** | ✅ 跨 50 个 NN 取平均，**不跨 B** |

**🔍 重点确认 line 177**：`out` 形状是 `(N=50, B)`；`mean(0)` 是沿**模型集成维**取平均，不是沿样本维。Result `(B,)` 的每一项 = 50 个 NN 对该 row 的预测的均值，每 row 独立。✅ Safe。

**没有 BatchNorm**。`_BatchedMLPEnsemble` 是 numpy/torch 手撸的 MLP，只有 Linear + manual LayerNorm + GELU。

### 1.5 `_compute_batch_features` (line 273-289)

- `X3d` 形状 `(N=batch_size, T=100, K=154)`，每 row 独立由 `df[self._raw_feat_cols].to_numpy()` 构造
- `raw_last = X3d[:, -1, :]`：每 row 取 window 最后一 tick（独立）
- `amount_delta` 做 `sign(v) * log1p(|v|)`（elementwise）
- `self._ffb.compute_batch_features(X3d, self._col_idx)`：进入 fast_features_batch.py，见 §2
- `np.where(np.isfinite(extras_full), ...)`（elementwise）
- `extras_kept = extras_full[:, self._extra_keep_idx]`（列切片）
- `np.concatenate([raw_last, extras_kept], axis=1)`：沿**特征维**拼接，不沿 batch
- **判定**：✅ Per-row 操作

### 1.6 5-horizon `share_with` + threshold scaling

- `thresholds.json` 中 h=5/10/20/40 都有 `share_with=60`
- `predict()` 迭代每个 horizon：先看 `pred_cache[60]` 是否已算；首次访问时构造，后续 horizon 复用（**同一 batch 内复用**）
- 即使 horizon 顺序变化（thresholds.json 列表顺序固定）或 share_with 取消，`pred_dmid[i]` 都只依赖第 i 行的 input feature
- 每个 horizon 独立用自己的 `thr_up`/`thr_dn`（thresholds.json 常量）调用 `_gate_with_band`/`_ev_gate_predict`
- **判定**：✅ 5 horizons 各自的 action 完全 per-sample 决定，绝无 batch-rank / batch-quantile 的"出手率控制"

### 1.7 State / cache（`self.*` 一览）

所有 `self.*` 属性都在 `__init__` 中设置一次，predict 期只读：
- `self._device`、`self._raw_feat_cols`、`self._raw_col_to_idx`、`self._extra_keep_idx`、`self._col_idx`、`self._horizons` —— 配置常量
- `self._lgb_lists` / `self._nn_batched` / `self._weights` —— 模型权重
- `self._cw_enabled` / `self._cw_band` / `self._cw_default_band` —— thresholds.json 常量

**唯一一个 dict 在 predict 内**：line 329 `pred_cache: Dict[int, np.ndarray] = {}` —— 函数局部变量，每次 predict 调用都重新建空。**不跨 predict 调用持续**。

**判定**：✅ Stateless across predict() calls，符合"测试点顺序被打乱"约束。

---

## 2. fast_features_batch.py（inference 实际特征构造）

`X3d` 形状约定：**(N=batch, T=100, K=raw_cols)**。**axis 0 = batch（绝对不能跨）**，axis 1 = time（可在窗口内聚合），axis 2 = feature。

所有 reductions 的 axis/dim 检查表（基于 grep 全量结果）：

| 文件:行 | 操作 | seg 形状 | axis/dim 实际方向 | 判定 |
|---|---|---|---|---|
| 105 `_rolling_sum_last_W` | `x[:, -W:].sum(axis=-1)` | (N, W) | **time** | ✅ |
| 130-131 `_rolling_mean_std_lastW` | `seg.mean/std(axis=-1)` | (N, W) | **time** | ✅ |
| 183 T3 mlofi | `window.sum(axis=-1)` | (N, W) | **time** | ✅ |
| 227 T3 RV | `sq[:, -W:].sum(axis=-1)` | (N, W) | **time** | ✅ |
| 282-285 Stage1 dual_z | `seg20.mean(axis=1)`、`seg20.std(axis=1)` | (N, 20, 37) | **time = axis 1**（不是 axis 0！）| ✅ **关键确认** |
| 305-306 Stage1 signed_rv | `r2_pos[:, -W:].sum(axis=-1)` | (N, W) | **time** | ✅ |
| 315 Stage1 kyle_inv | `r[:, -W:].std(axis=-1)` | (N, W) | **time** | ✅ |
| 371 Stage2 qrank | `(seg <= last_val).mean(axis=-1)` | (N, W) | **time** | ✅ window 内 rank |
| 382-387 Stage2 skew | `seg.mean/std(axis=-1)` 等 | (N, W) | **time** | ✅ |
| 426 Stage2 gofi | `seg.sum(axis=-1)` | (N, W) | **time** | ✅ |
| 443-446 Stage2 kyle_lambda | mx/my/mxy/mx2 = `*.mean(axis=-1)` | (N, W) | **time** | ✅ |
| 459-460 Stage2 vol_burst | `vol[:, -W:].mean(axis=-1)` | (N, W) | **time** | ✅ |
| 514 Stage3 rv_ratio | `r2[:, -W:].sum(axis=-1)` | (N, W) | **time** | ✅ |
| 528-529 Stage3 jshare | `prod[:, -W:].sum(axis=-1)` | (N, W) | **time** | ✅ |
| 545-548 Stage3 cancel_imb | `*[:, -W:].sum(axis=-1)` | (N, W) | **time** | ✅ |
| 568-570 Stage3 roll_eff | mx/my/mxy = `*.mean(axis=-1)` | (N, W) | **time** | ✅ |
| 618 Stage5 adapt_mom | `mid[:, -W:].std(axis=-1)` | (N, W) | **time** | ✅ |
| 658-664 Stage5 ofi_tox | mx, my, sxy, sxx, syy = `*.mean/sum(axis=-1)` | (N, W) | **time** | ✅ per-window Pearson |
| 680-681 Stage5 signed_bv | `bv_term[:, -W:].sum(axis=-1)` | (N, W) | **time** | ✅ |
| 691-693 Stage5 spread_reg | `np.median(seg, axis=-1)`, `np.quantile(seg, q, axis=-1)` | (N, W) | **time** | ✅✅✅ **关键：median/quantile 是 per-window，每 row 独立分位数** |
| 708-714 Stage5 trade_pers | mx, my, sxy, sxx, syy = `*.mean/sum(axis=-1)` | (N, W) | **time** | ✅ per-window |
| 728-729 Stage5 liq_asym | `*[:, -W:].sum(axis=-1)` | (N, W) | **time** | ✅ |

### 关键确认：line 282-285 Stage1 dual_z 不是 axis 0

```python
X_dz = X3d[:, :, dz_idx]  # (N=batch, T=100, 37)
seg20 = X_dz[:, -DUAL_Z_W_SHORT:, :]  # (N, 20, 37)
m20 = seg20.mean(axis=1)  # axis 1 = TIME (20), 得 (N, 37)
```

axis=1 是 time，axis=0 是 batch。这里 **axis=1**，所以每个 row 用自己的 last-20-tick 算每个特征的 mean。**不跨 batch**。

### 任何 pandas groupby / quantile / rank？

- ✅ 全文件**没有** `groupby` / `.rank(` / `pd.qcut` / `pd.factorize`
- ✅ 全文件**没有** `np.argsort` / `scipy.stats.rankdata`
- ✅ `np.quantile` / `np.median` 都带 `axis=-1`（time 维内）

### 总判定：fast_features_batch.py 完全 sym-agnostic、shuffle-invariant、stateless ✅

每个 row 在 batch 中的位置不影响输出。同一 row 单独放进 batch 1 和混在 batch B 里得到 **逐 bit 相同**的输出（除浮点误差）。

---

## 3. fast_features.py（single-window 版，**inference 不调用**）

⚠️ 该文件存在于 `04_build_pkg/` 但 **Predictor.py 没有 import 它**（line 191 显式 load `fast_features_batch.py`）。它是 reference / debugging 版，每函数只处理一个 window。

### 3.1 line 267-270 dual_z 的 `axis=0`

```python
X = np.column_stack([df_arr[c]... for c in DUAL_Z_COLS])  # X shape (T=100, 37) — single window
m20 = X[-DUAL_Z_W_SHORT:].mean(axis=0)  # axis 0 = TIME (20), 得 (37,)
```

**单窗口 context 下**，X 形状是 `(T=100, 37)`，axis=0 在这里是 **time**，不是 batch。✅ Safe。

但即便如此，因为该文件**根本不被 inference 调用**，就算有 bug 也不会到提交。

**判定**：✅ Safe；也是 dead-code-in-inference 的双重保险。

---

## 4. build_pkg.py（构建期，不参与 inference）

- 纯 file copy / json read。
- 无任何统计、模型推理、batch 操作。
- **判定**：✅ Irrelevant to inference。

---

## 5. config.json / thresholds.json

### config.json
- `"feature"` 列表（154 个 raw 列名 + "sym"）—— 静态字段
- `"batch": 1024` —— 静态
- **判定**：✅ 无 placeholder

### thresholds.json

所有数值都是 **具体浮点数常量**，无 placeholder：

```json
"horizons": [ {"h": 5, "share_with": 60, "thr_up": 0.0000866, "thr_dn": 0.0000624, ...}, ... ]
"conformal_wrapper": {
  "enabled": true,
  "per_sym_beta": {"0": 0.1, "1": 0.4, "2": 0.3, "3": 0.0, "4": 0.0},
  "per_sym_sigma": {"0": 0.0002403901, "1": 0.0004712397, "2": 0.0004522216, "3": 0.0004235249, "4": 0.0004279811},
  "default_beta_for_ood": 0.16,
  "default_sigma_for_ood": 0.0003998317
}
```

- per_sym_beta 是 5 个浮点
- per_sym_sigma 是 5 个浮点（应来自训练集 5 只 sym 上的 validation residual std，**算一次写死**）
- default_beta_for_ood / default_sigma_for_ood 提供 OOD sym fallback
- **判定**：✅ 全部常量，inference 时只 load 不重算

---

## 6. 训练代码（确认推理时不重复操作 + 训练数据本身无泄露）

### 6.1 `train_T188v2_nn_seed.py`

| 行 | 操作 | 上下文 | 判定 |
|---|---|---|---|
| 63-67 | `class_balanced_weight(y)` — 用 bincount/全样本计数 | 训练前算 sample weight | ✅ 训练阶段已知全集，不污染推理 |
| 254-255 | `feat_mean = np.nanmean(X_tr, axis=0)`，`feat_std = np.nanstd(X_tr, axis=0)` | axis=0 = train rows | ✅ **训练时算一次**，写进 ckpt + npz，inference 期当**常量** broadcast，**不重算** |
| 261, 402 | `nan_mask.any(axis=0)` | 找含 NaN 的列做 imputation | ✅ 训练期，不影响 inference |
| 266 | `target_scale = 1.0 / y_regr_tr.std()` | label 标准化 scalar | ✅ 训练期算一次，写进 ckpt |
| 270-272, 406-414 | `aug_a`：`rng.uniform(0.8, 1.2, size=X.shape)` × X | per-element multiplicative noise | ✅ per-sample，无跨样本依赖 |
| 313, 456 | `xb = torch.clamp((xb - fm_t) / fs_t, -CLIP, CLIP)` | 标准化用预算 fm_t/fs_t（GPU 常量 tensor） | ✅ **不依赖当前 batch** 的统计 |
| 305 | `model.train()` | 训练 mode | OK，反正没 BN/Dropout 不影响 inference 一致性 |
| 80-104 MLPRegr | `nn.Linear → nn.LayerNorm → GELU → Dropout`，**无 BatchNorm** | LayerNorm 沿 hidden | ✅ |
| 108 | `model.eval()` 在 `predict_chunked` | 把 Dropout 关掉 | ✅ |
| 117-127 SPO+ loss | per-sample 计算，sum across batch（用 `wb.sum()` 归一） | 训练 loss reduction | ✅ 仅 loss，未输出回到样本预测 |

**关键确认**：CLIP=10 的 std/mean 用什么算的？答：用 `np.nanmean/nanstd(X_tr, axis=0)` 算（line 254-255），是**全训练集统计**，存到 ckpt['feat_mean'], ckpt['feat_std']。Predictor 通过 `_BatchedMLPEnsemble.__init__` 从 npz load 这俩做常量，inference 期不重算。

✅ **NN 训练无泄露**。

### 6.2 `train_T188v2_lgb_seed.py`

| 行 | 操作 | 判定 |
|---|---|---|
| 47-51 | `class_balanced_weight(y)` | ✅ 训练期全集统计，仅作 sample weight |
| 54-61 | `aug_a_scale_chunked` —— per-element uniform 缩放 | ✅ per-sample |
| 104-105 | `assert not (forbidden & set(feat_names))` —— 防止 sym/date 漏进特征 | ✅ 防御断言 |
| 170-188 | `lgb.train(params, dtrain, ...)` | ✅ LightGBM 标准训练；每棵树独立看每行特征 |
| 188 | `booster.save_model(out_path)` | ✅ Tree-only 模型，inference 是 booster.predict(X)，per-row 独立 |

✅ **LGB 训练无泄露**。

### 6.3 `01_build_features/build_schemeP_cache.py`

| 关键点 | 判定 |
|---|---|
| `build_one_session(df)` 对每个 parquet 独立处理 | ✅ 每个 session（sym, date, am/pm）独立 |
| `compute_batch_features(X3d, ...)` X3d 是单 session 内的所有 100-tick 窗口 | ✅ 调用的是同一份 fast_features_batch.py（与 04_build_pkg 字节一致），axis=-1 全在 time 维 |
| 训练时 X3d 里的 N 维是 same-session 多窗口，不是跨 session ⇒ feature 构造时绝不可能跨 session 泄露 | ✅ |
| 最后 `np.concatenate(..., axis=0)` 把所有 session 的输出沿 row 维拼起来 | ✅ 拼接，不是统计 |

✅ **特征构建无 cross-session 统计泄露**。

### 6.4 `01_build_features/stage5_features.py`

与 `04_build_pkg/fast_features_batch.py` 的 `compute_stage5_batch` 逻辑等价（语义一致，参数同名）。所有 `axis=-1` 都是 time 维。

✅ Safe（同 §2 分析）。

### 6.5 `01_build_features/fast_features_batch.py`

**`diff 01_build_features/fast_features_batch.py 04_build_pkg/fast_features_batch.py` → no output（字节完全相同）**。审计结论沿用 §2。

✅ Safe。

---

## 7. 风险列表

| 严重度 | 位置 | 问题 | 影响 | 建议 |
|---|---|---|---|---|
| — | — | **无任何风险发现** | — | — |

---

## 8. 安全确认表

| 位置 | 检查项 | 结论 |
|---|---|---|
| `Predictor.predict()` | per-batch normalize | ❌ 无 |
| `Predictor.predict()` | 跨 predict 调用的 state / cache | ❌ 无（pred_cache 是 per-call 局部） |
| `Predictor.predict()` | per-batch rank / quantile / argsort | ❌ 无 |
| `Predictor.predict()` | per-batch adaptive threshold (T168 V1-V4) | ❌ 无 |
| `Predictor.predict()` | per-batch agreement / vote / TTA / TTT | ❌ 无 |
| `_BatchedMLPEnsemble` | BatchNorm / running_mean / running_var | ❌ 无 BN，仅 LayerNorm（dim=-1=hidden） |
| `_BatchedMLPEnsemble` line 177 `out.mean(0)` | mean 沿哪一维 | ✅ 沿 N=50 NN 维（dim 0 是 ensemble），**不是 batch** |
| `_BatchedMLPEnsemble` | feat_mean/feat_std 来源 | ✅ 训练时算好，npz 中常量 |
| `_BatchedMLPEnsemble` | LayerNorm gamma/beta 来源 | ✅ 训练学好的常量，npz |
| `_ensemble_predict_lgb` | LGB inference | ✅ 每棵树 per-row 独立 |
| `Predictor.predict()` line 357 `sum(axis=0)` | sum 沿哪一维 | ✅ stacked (M=2, B) 沿 model 维（M=2 即 LGB+NN）求加权和 |
| Conformal sigma 来源 | 来自 thresholds.json `per_sym_sigma` 常量 | ✅ |
| Conformal beta 来源 | 来自 thresholds.json `per_sym_beta` 常量 | ✅ |
| `_gate_with_band` / `_ev_gate_predict` | elementwise compare | ✅ |
| 5-horizon `share_with` + per-horizon `thr_*` 缩放 | 每 horizon 独立 thr，每 row 独立 action | ✅ 无 batch-rank |
| `fast_features_batch.py` | 任何 axis=0 reduction | ❌ 无（grep 0 命中） |
| `fast_features_batch.py` | 任何跨样本 mean/std/sum/median/quantile/rank | ❌ 无（全部 axis=-1 / axis=1 都是 time 维） |
| `fast_features_batch.py` stage5 `np.median/quantile` | 沿 axis=-1（window 内 W ticks） | ✅ per-window 分位数，不跨样本 |
| `fast_features_batch.py` stage1 dual_z `axis=1` | seg20/seg100 形状 (N,T,37)，axis=1=time | ✅ 不是 axis=0=batch |
| `fast_features.py`（不在 inference 路径）`axis=0` | X 是 single-window (100,37)，axis=0=time | ✅（且 dead code） |
| `train_T188v2_nn_seed.py` line 254-255 `axis=0` | 训练时算 feat_mean/feat_std，存 ckpt | ✅ 训练统计 → 推理常量 |
| `train_T188v2_nn_seed.py` aug_a / class_balanced_weight | per-element noise + per-sample weight | ✅ |
| `train_T188v2_lgb_seed.py` aug_a / class_balanced_weight | 同上 | ✅ |
| `train_T188v2_*` 中是否有 BatchNorm | grep 无命中 | ✅ |
| `build_schemeP_cache.py` | 每个 session 独立 build | ✅ 无 cross-session 统计 |
| config/thresholds.json | 全部常量值，无 placeholder | ✅ |

---

## 9. 历史关联

### 9.1 T168 adaptive threshold V1-V4（用户提到）

> T168 试过 V1-V4 全部退步（顺序打乱下不稳定），final 代码不应该含这种 wrapper

✅ **已确认**：grep 全代码 0 次 `adaptive` / `mean_abs_pred` / `batch_stat` 等关键字。Predictor 完全没有 adaptive threshold wrapper。`thr_up`/`thr_dn` 永远是 `thresholds.json` 里的常量浮点，唯一的"动态"是 `band_per_row`，而 band 来自 per_sym 常量表，不依赖 batch 内 pred。

### 9.2 T161 TTA / T162 TTT / T169 AR TTT

> 试过失败的 batch-stat / cross-sample wrapper，final 不应有

✅ **已确认**：grep 全代码 0 次 `TTA` / `TTT`。NN 没有 inference-time 自适应。整个 `_BatchedMLPEnsemble.predict_mean` 完全是前向 bmm，无任何"测试时调整"逻辑。

### 9.3 注释明确声明的兼容性（Predictor.py line 19-28）

代码自带 compliance docstring：
- "No cross-call state held on `self`"
- "sym-agnostic FORWARD"
- "Stateless / shuffle-invariant: no row order dependency, no buffers"
- "W <= 100 for every rolling feature"

✅ 实际代码与声明一致，未发现"嘴上合规手上违规"。

---

## 10. 附：审计方法补充

### 10.1 grep 全文件 0 命中的关键字
- `BatchNorm` / `batchnorm` / `running_mean` / `running_var` / `track_running`
- `softmax` / `groupby` / `F.normalize`
- `adaptive` / `TTA` / `TTT` / `batch_stat` / `cross_sample` / `mean_abs_pred`
- `argsort` / `percentile`（除已分析的 `np.quantile`）

### 10.2 axis/dim 矩阵已逐一确认（见 §1, §2, §3）

无遗漏。

### 10.3 局限性
- 未运行时实测（如静态 shuffle test），但**代码语义清晰**：所有操作均为 per-row 或 within-window 内部，shuffle 不可能改变输出。
- 未对每个 .npz 文件做 schema 检查（feat_mean/feat_std 是否真的是 (in_dim,) 常量向量），但 `_BatchedMLPEnsemble.__init__` 的 `.astype(np.float32)` + `np.stack(..., 0)` 流程保证了 shape 一致性，且 inference 期不重算。

---

**最终判定：整套提交代码在 batch 内统计泄露这条硬约束上 ✅ 完全合规，可放心提交。**

RESULT: task=[batch leakage audit] metrics={n_files_audited=9, n_red_findings=0, n_yellow_findings=0, n_safe_confirmations=25} notes=[完全安全：所有 reductions 均沿 time 维或 hidden 维，无 axis=0 batch 维聚合；所有 sigma/beta/feat_mean/feat_std/target_scale 是训练期算好的常量；无 BatchNorm/adaptive/TTA/TTT；Predictor.predict 不跨调用 state，pred_cache 是 per-call 局部变量；fast_features_batch.py 训练版与推理版字节一致；fast_features.py 虽含 axis=0 但操作的是 single-window 内的 time 维且 inference 不调用]
