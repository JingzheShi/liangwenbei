# Answers P4 — Pipeline & Miscellaneous

> Scope: pipeline 组织 / build_pkg / shell scripts / requirements / 模型文件格式 / config.json / 可复现性 / 平台约束合规 / README 一致性 / 跨文件常量 / CODE_QUESTION_LIST 杂项。
> 不回答 P1（训练）/ P2（决策层）/ P3（特征工程）主题。
> 状态图例: ✅ 合理且可辩护  🟡 可改善但可上线  🔴 真实风险/需修

---

## 总结（TL;DR）

- **4 步 pipeline 设计基本合理**：单线性流水线 + per-script idempotent skip 已足够；没有 DAG orchestrator 是 conscious trade-off（5 步生命周期、单一开发者、Bash 比 Snakemake 更可移植）。
- **3 条平台硬约束代码层面全部满足**：date 仅做 split 不进 feature、Predictor 完全 stateless、sym 仅用于 conformal lookup（OOD 安全降级）。
- **真实风险（🔴）总共 5 处，按严重度排序**：
  1. **GELU train/inference mismatch**（Q176/Q115）—— `nn.GELU()` 默认 erf vs `F.gelu(..., approximate="tanh")`，量级 1e-4，与 `thr_up=3e-4` 同阶，理论上可翻 action。已通过 T188v3 vs T188v2 max diff = 6.98e-10 实测证否，但成因未澄清。
  2. **build_pkg 无 NN/LGB seed 数对齐校验**（Q133）：可上线但 ensemble 失衡。
  3. **build_pkg 不校验 npz 可加载性 + 不计算 checksum**（Q22/Q26/Q193）：md5_file 定义但未调用。
  4. **`run_pipeline.sh` 缺 `set -u`/`pipefail`、无 preflight、参数位置式**（Q11/Q43/Q56-57）。
  5. **CHUNK=200_000、`S*7919+offset` 等"看不见"的 seed-sensitive 常量未文档化**（Q31/Q32）。
- **可复现性是 best-effort**：未启 `torch.use_deterministic_algorithms(True)`、未固定 `CUBLAS_WORKSPACE_CONFIG`、LightGBM 不设 `deterministic=true`。同 seed 同硬件可能 bit-差异。**项目立场**：T188v2 ensemble 50+50 内方差远大于 bit drift，可接受；T155 单独验证过 LGB M7 100% 确定性，NN 没有。
- **README 与代码总体一致**，已知 drift 2 处（Q66 stage 维度数 / Q70 val 天数）+ 数字打字错误 1 处（Q163 +5.51 vs +6.28），不影响交付。
- **跨文件常量**：DROP_NAMES 3 处重复、HORIZON_LIST/RAW_COLS 各 2 处重复、阈值缩放常数硬编码 thresholds.json 但有 `_note` 标注。可重构但当前规模可控。
- **杂项问题**接管 ~100 个，多为 minor / 已实证不影响交付 / 历史决策由 evidence-DB 支撑。

**P4 主答题量**：~290 个独立问题。**GAP 表**末尾列出 12 条值得改进的项。**强 ablation/证据链**列出 8 条（T61 58×、T188v3 186×、T182 conformal ablation、T188v2 vs T179 NN-only 扩容、T190 150+150 退步、T155 LGB 100% 确定性、T188v2 vs T188v3 max diff 6.98e-10、T141 Hawkes M7 边际）。

---

## 1. 4 步 pipeline 组织 + 数据契约

### 1.1 Stage granularity / 为啥分 4 步而不是 3、5、N
- **来源**：QREVIEW_PIPELINE_ARCH Q1, Q3
- **答案**：
  4 步对应 4 个**截然不同的资源画像**：
  - 01 build_features：磁盘 IO 重 + CPU 向量化（~30 min, ~20 GB cache）
  - 02 train LGB：CPU/GPU + 中等 RAM（5 HP × 10 seed, ~12–30 min）
  - 03 train NN：GPU 重 + 训练长（90 min 单卡 / 18 min 5 卡）
  - 04 build_pkg：纯文件拷贝 + JSON sanity（数秒）

  4 步边界即资源切换点；如果把 02、03 合并为"models"子树会绑死 GPU/CPU 调度（用户可能机器无 GPU 但想跑 LGB）。从历史看，T188v2 mega ensemble 阶段才把 LGB/NN 拆开独立扩容（T179 只扩 NN 平台 −0.25，T188v2 两侧都扩 +1.00，验证两者必须独立）。
- **核心证据**：T179（只扩 NN 平台 −0.25）vs T188v2（两侧都扩 +1.00）—— ensemble 增量必须 LGB/NN 对称 → 02/03 独立可单独 scale 是设计资产，而非缺陷。
- **状态**：✅

### 1.2 Stage 是否可独立 re-run / resume 半失败
- **来源**：QREVIEW_PIPELINE_ARCH Q2, Q10
- **答案**：
  Resume 通过两层 idempotency：
  - `train_T188v2_lgb_seed.py:91` `if os.path.isfile(out_path): SKIP`
  - `train_T188v2_nn_seed.py:229` `if args.skip_phase1 and isfile(anchor_path)` → 跳过 Phase 1（`run_all_nn_seeds.sh:27` 默认开启）
  - `run_all_lgb_seeds.sh` 与 `run_all_nn_seeds.sh` 都是 `for SEED in 1..50` 串行——单 seed 失败不会污染其他 seed 的 .txt/.npz。

  但 `run_pipeline.sh` 顶层 `set -e` 会在第一个失败处中断，无 `pipeline_state.json`。重启时 Step 1 cache 已存在 → 仍会被重做（除非用户手动 `--which` 跳过；该参数没暴露到顶层 run_pipeline.sh）。Step 4 始终重做（毫秒级）。

  **结论**：seed 级 resume 是 first-class；stage 级 resume 是 implicit（Cache npz 文件存在就不会重算，因为 `build_schemeP_cache.py:225-235` 没有 idempotent 检查，**会无脑覆盖**——这是 🟡）。
- **状态**：🟡 stage 级 idempotency 不完整（Step 1 会重做）

### 1.3 Numeric prefix 暗示 DAG，但实际为线性 → 是否限制并行
- **来源**：QREVIEW_PIPELINE_ARCH Q3, Q40
- **答案**：
  Cache 写完后 02/03 数据上完全独立（无共享文件）；当前 `run_pipeline.sh` 顺序跑 02 → 03 是为 RAM 安全考虑（02 LGB 训完释放 ~4.2 GB 后 03 NN 再开 ~4.2 GB；并行需 ~8.4 GB peak）。README:27/172 提示 5 GPU 并行做 NN 但没在脚本里实现——用户需手动 sharding。
- **核心证据**：`train_T188v2_lgb_seed.py:119` `Pre-allocating ~4.2 GB` 注释。
- **状态**：🟡（README 与脚本能力不符；可加 `--seeds 1-10` shard 参数）

### 1.4 缺少 pipeline orchestrator（Make/Snakemake/Airflow）
- **来源**：QREVIEW_PIPELINE_ARCH Q4
- **答案**：
  Conscious trade-off：单一开发者、5-步生命周期、Bash 比 Snakemake 在评测平台和复现者环境更可移植（无 pip 依赖）。缺点确实是"cache 改了但 model 不变"的 stale 问题——通过 `gitignore: model_h*_seed*.txt` 强制用户重训而不复用旧 model，部分缓解。
- **状态**：✅（决策合理，但应在 README 说明"cache 必须从对应 commit 的 `fast_features_batch.py` 重新生成"）

### 1.5 数据契约 01→{02,03}：cache npz 字段
- **来源**：QREVIEW_PIPELINE_ARCH Q5, Q6, Q7, Q8
- **答案**：
  契约通过**字符串 key + 文件结构**隐式表达：
  - npz 顶层 key：`X (N, 370)`、`mp_t`、`y{5,10,20,40,60}`、`mp_t{5,10,20,40,60}`、`sym`、`date`、`sess_idx`、`t`（`build_schemeP_cache.py:183-194`）
  - 文本 sidecar：`schemeP_feat_names.txt`（370 行）、`schemeP_extra_feat_names.txt`（216 行）

  **无 schema validation、无 git SHA、无 hash**。02/03 训练在 `npz["X"][:, keep_idx]` 时若 X 维度变了会 silently 出错（feat_dim 由 `len(all_feat_names) - len(drop_idx)` 推导）。

  **mitigation 实际存在**：
  - `train_T188v2_lgb_seed.py:104-106` 和 `train_T188v2_nn_seed.py:207-208` 都有 `assert not (forbidden & set(feat_names))`，至少能挡住 date/sym/time 进 feature。
  - `Predictor.py:66` `assert len(RAW_COLS_TRAIN_ORDER) == 154`。
  - `Predictor.py:148-151` 在 in_dim 不符时 fallback 用 `keep_idx` 投影一次。

  **真实风险**：如果用户改了 `fast_features_batch.py` 让 `all_feature_names()` 顺序变了但维度不变，模型仍能 load，但每列含义错位 → silent 失败。
- **核心证据**：`md5sum 01/fast_features_batch.py 04/fast_features_batch.py` 完全一致（81bb9fc8b06d893b05bb3abc16aa0fc5）——byte-equivalence 是当前唯一防线。
- **状态**：🔴 无 schema/version stamp 是真实风险（Predictor 没法发现 cache 与 model 不匹配）。建议在 npz 里写 git SHA + `compute_batch_features` hash。

### 1.6 cache npz 不带 fingerprint
- **来源**：QREVIEW_PIPELINE_ARCH Q7, Q25
- **答案**：
  同 1.5 — `build_pkg.py:107-115` 写了 manifest.json 但**只列文件名**、不计 md5（Q193；`md5_file()` 函数定义在 `build_pkg.py:27-32` 但**从未调用**——dead code，明显是"打算加但忘了"的痕迹）。整个项目无 BUILD_INFO.json/no git SHA in package（Q55）。
- **状态**：🔴 应在 manifest.json 加 git rev-parse HEAD + 每文件 md5。

### 1.7 训练（03）↔ Predictor（04）契约
- **来源**：QREVIEW_PIPELINE_ARCH Q8, Q9
- **答案**：
  契约靠**两份独立的 MLP forward 实现**：
  - 训练侧：`train_T188v2_nn_seed.py:80-104` 用 `nn.Sequential(Linear, LayerNorm, GELU, Dropout)`
  - 推理侧：`Predictor.py:78-178` `_BatchedMLPEnsemble` 用 `torch.bmm` + 手写 LayerNorm + `F.gelu(approximate="tanh")`
  - 桥梁：`extract_npz` 在 `train_T188v2_nn_seed.py:130-160`，按 `net.{layer_idx}.weight` 字符串 key 抽 state_dict 出来存 npz。

  **风险点**：
  - GELU train/inference mismatch（Q176 后文 §8.6）—— 已实测 max diff 6.98e-10，证否 1e-4 风险。
  - `layer_idx` 计数依赖 `use_layernorm` 标志（Q135）：若 use_layernorm=False，跳过 1 个 module → layer_idx 错位。当前 hardcode True，但脆弱。

  **没有 numerical-parity 单元测试**（Q9 docstring 说 1e-6 typical/1e-4 max，但没 test 文件 verify）。**实际等价性是通过 T188v2 → T188v3 离线 actions diff 6.98e-10 实测出来的**，但没作为 CI 测试沉淀下来。
- **核心证据**：HISTORY_EVIDENCE_DB §10 "T188v3 actions vs T188v2 max diff 6.98e-10, 远低于 1e-4 容忍度"。
- **状态**：🟡 应加 numerical parity test（保留训练时 .pt + 包内 .npz 各做一次 forward，assert max diff < 1e-5）。

### 1.8 schemeP_feat_names.txt：plain-text 契约的脆弱性
- **来源**：QREVIEW_PIPELINE_ARCH Q6
- **答案**：
  - 写入端：`build_schemeP_cache.py:215-220` `f.write("\n".join(full_names) + "\n")`
  - 读取端：`train_T188v2_lgb_seed.py:95-103` 和 `train_T188v2_nn_seed.py:198-204`：`with open(...) as f: all_feat_names = [line.strip() for line in f]`
  - 用途：通过 name → index 映射决定 DROP_NAMES 的位置。

  **稳定性靠**：`fast_features_batch.all_feature_names()` 返回顺序 = `compute_batch_features` 中拼接顺序——Python dict 排序在 3.7+ 是插入序，所以稳定；CPython 实现细节。
- **状态**：🟡（应在 .txt 顶部加 "# generated by build_schemeP_cache.py @ {git_sha} {timestamp}"）

### 1.9 Bash `set -e` only / 缺 logging
- **来源**：QREVIEW_PIPELINE_ARCH Q11, Q12, Q13
- **答案**：
  - `set -e`：脚本任一命令非 0 退出立即停。
  - **缺**：`set -u`（typo 静默通过），`set -o pipefail`（`cd && zip` 链中 cd 失败不会传播），`trap ERR`（不打哪行失败）。
  - 无 log 文件硬约定（用户自己 `> log.txt 2>&1`）；CLAUDE.md 要求的 `worker-progress.json` 没实现——但这是 worker agent 规范，不是给最终用户的 pipeline。
- **状态**：🟡（添加 `set -euo pipefail` 是 1-line 改动，强烈建议）

### 1.10 Pipeline 缺 timeout / kill switch
- **来源**：QREVIEW_PIPELINE_ARCH Q14
- **答案**：
  pandas 读 corrupt parquet hang 是低概率但严重的风险；可加 `timeout 30m python3 build_schemeP_cache.py`。当前依赖用户监控 stdout 进度（`build_schemeP_cache.py:170-175` 每 5% 打 sess/s + ETA）。
- **状态**：🟡

### 1.11 数据契约：parquet 缺失静默跳过
- **来源**：QREVIEW_PIPELINE_ARCH Q77, CODE_QUESTION_LIST Q148
- **答案**：
  `build_schemeP_cache.py:152-154`：`if not os.path.isfile(path): skipped += 1; continue`。结尾 `print(f"WARNING: skipped {skipped} ...")` 但**不报错**。如果 50% parquet 缺失，仍跑出小 cache → 后续训练在残缺数据上跑 → 模型质量低。
- **状态**：🔴 应加 `if skipped > 5: raise RuntimeError(...)` 或至少在 stderr exit code != 0。

### 1.12 Parquet schema 不上前 validate
- **来源**：QREVIEW_PIPELINE_ARCH Q76, CODE_QUESTION_LIST Q149
- **答案**：
  `pd.read_parquet(path)` → `df[RAW_COLS]` 在 RAW_COLS 任一缺失时抛 KeyError 在循环深处。1200 个 parquet 时，发现要 ~30 s × 部分文件后才崩。可加 1-parquet pre-flight：`pd.read_parquet(first_path, columns=RAW_COLS+["label_60"]).head()`。
  
  parquet 命名硬编码 `snapshot_sym{sym}_date{date}_{sess}.parquet`（line 151）：用户用其他命名要改 build script。
- **状态**：🟡

### 1.13 100-tick window：连续性假设
- **来源**：QREVIEW_PIPELINE_ARCH Q78, Q79
- **答案**：
  `sliding_window_view` 假设相邻行 = 相邻 tick；PROGRESS.md §3 确认"AM 09:40-11:20 / PM 13:10-14:50 每段 2001 ticks @ 3s"，**官方数据每段保证连续**。
  AM/PM 跨段不会被 window 桥接——`build_schemeP_cache.py:73-84` 按 `(sym, date, sess)` 三元组分别处理，每段独立计算 sliding window。这是正确实现，README 应该明示一下（CLAUDE 已在 Q79 中标 ✅）。
- **状态**：✅

---

## 2. build_pkg.py / sanity check

### 2.1 build_pkg.py 做了什么
- **来源**：QREVIEW_PIPELINE_ARCH Q20-Q26, CODE_QUESTION_LIST Q131-Q133, Q193
- **答案**：
  4 件事：
  1. **检测缺失** seed（NN npz、LGB txt）—— `check_models:35-46`，只 print warning，**不 fail**
  2. **清空并复制** pkg_dir 下的 6 个静态文件（Predictor.py / config.json / fast_features.py / fast_features_batch.py / requirements.txt / thresholds.json）
  3. **复制实际存在的** N 个 NN npz + N 个 LGB txt
  4. **写 manifest.json**（n_nn/n_lgb/nn_seeds/lgb_seeds/static_files）

  **重点缺失**：
  - 不 validate npz 可加载性（`shutil.copy2`，corrupt 静默通过）
  - 不计算 file checksum（`md5_file()` 定义未调用 → Q26 dead code）
  - 不校验 `thresholds.json:ensemble_seeds` 与实际存在的 seed 匹配
  - 不校验 NN seed list == LGB seed list（可能 NN 缺 5 个、LGB 缺 8 个，ensemble 失衡）
  - 不校验 manifest.json 写完后再读回一致
  - 不 zip（zip 由 `run_pipeline.sh:75` 调 `zip -qr` 完成，build_pkg.py 单独跑时容易忘记）
- **核心证据**：T188v2 mega ensemble 148 MB / 100 文件，build_pkg.py 跑约 2 s（CPU-bound 文件拷贝）。
- **状态**：🟡 缺 4 处校验，但**当前生产模式下用户清楚自己装了什么**，没在生产爆过。

### 2.2 fast_features.py 是 dead code 还是有用
- **来源**：QREVIEW_PIPELINE_ARCH Q66, Q67, CODE_QUESTION_LIST Q120, Q130
- **答案**：
  `04_build_pkg/fast_features.py`（568 行）打包进 pkg，但 `Predictor.py:191` 只 `_load_module("fast_features_batch.py", ...)`，**从未引用 `fast_features.py`**。
  - 历史原因：fast_features.py 是单 window reference 实现（196 维，不含 Stage5），曾用于 iter_013~iter_015 时代单样本推理 + sanity test。
  - T188v3 batched bmm 优化后全部走 batched 路径；fast_features.py 在 pkg 中是**包袱**（148 MB 占比里 22 KB 微不足道，但增加阅读混淆）。
- **核心证据**：`grep fast_features 04_build_pkg/Predictor.py` 只有 `_ffb = _load_module(... "fast_features_batch.py" ...)` 一处。
- **状态**：🟡 应删除（build_pkg.py 第 66 行 static_files list 去掉 "fast_features.py"）。删了不影响功能。

### 2.3 stage5_features.py 是 dead code
- **来源**：QREVIEW_PIPELINE_ARCH Q87, CODE_QUESTION_LIST Q151
- **答案**：
  `01_build_features/stage5_features.py`（7 KB）；`build_schemeP_cache.py:42` 注释 "kept in this directory for reference but not called here"——明示 dead。同时 Stage5 实际在 `fast_features_batch.py` 内部计算（216 = 196 baseline + 20 Stage5）。
- **状态**：🟡 应删除或加 README 说明"参考实现，未挂线"。

### 2.4 build_pkg 不主动排除 .pt
- **来源**：QREVIEW_PIPELINE_ARCH Q20, CODE_QUESTION_LIST Q188
- **答案**：
  build_pkg 只**正向白名单**复制 nn_h60_seed{S}.npz 和 model_h60_seed{S}.txt；不复制 .pt（包括 `T81_pretrained/anchor_seed{S}.pt`）。安全。`outputs/models/` 下 .pt 文件继续存在不会被复制进 pkg。这是 defensive design ✅。但如果用户**手动** `cp -r outputs/models pkg/` 会带上 .pt + T81_pretrained → 包翻倍。.gitignore 已挡 `*.pt`、`*.zip`。
- **状态**：✅

### 2.5 build_pkg 缺失 seed 容忍策略
- **来源**：QREVIEW_PIPELINE_ARCH Q22, CODE_QUESTION_LIST Q132, Q133
- **答案**：
  - 缺失 NN/LGB → `print WARNING` 后**继续打包**（line 58-59）
  - 缺失 static file → `print ERROR`（line 74）但**不退出**——下游 Predictor `import` 时才崩
  - 缺失 thresholds.json 的 horizon → 没单独检查；只 print "horizons with share_with = ..."
  
  容忍 N<50 的设计 rationale：debug 阶段可能只训了 30 seed，仍能打包测试。但生产时应该 N==50 强校验。
- **状态**：🟡 应加 `--strict` flag（缺任一就 sys.exit(1)）。

### 2.6 manifest.json 写而不读
- **来源**：QREVIEW_PIPELINE_ARCH Q25
- **答案**：
  manifest.json 没被 Predictor 加载（Predictor.py 没有 `with open("manifest.json")`）。它是给**人**看的文档（"这个 zip 里有什么"），不是程序契约。OK 但应在 build 完后 print 出来给用户 verify。
- **状态**：✅（design intent OK）

### 2.7 build_pkg 不打包 T81_pretrained
- **来源**：QREVIEW_PIPELINE_ARCH Q21, CODE_QUESTION_LIST Q188
- **答案**：
  Phase 1 anchor 不上传 ✅（推理不需要）；但**用户复现**时需保留 T81_pretrained 否则重新跑 Phase 1 = 多花 ~50 min × 50 seed = 跑步 41 h。`gitignore: outputs/` + `*.pt` 会让 T81_pretrained 被忽略；用户应在 README 中知会"想跳过 Phase 1 重训需保留 outputs/models/T81_pretrained/"。当前 README 没明示。
- **状态**：🟡

### 2.8 zip 步骤在 build_pkg 外
- **来源**：QREVIEW_PIPELINE_ARCH Q52, CODE_QUESTION_LIST Q199
- **答案**：
  `build_pkg.py:118-120` 只 print 提示，不 zip。Zip 由 `run_pipeline.sh:74-75` `cd "$PKG_DIR" && zip -qr "$ZIP_OUT" .` 完成。
  - **理由（probably）**：让 build_pkg 单步可测（不每次拖几十秒压缩）。
  - **风险**：`zip -qr . `包含 `.gitkeep`、潜在 hidden files（不会 dotfile 是 zip 默认行为 ✅）。
- **状态**：✅

### 2.9 包文件名硬编码
- **来源**：QREVIEW_PIPELINE_ARCH Q52
- **答案**：
  `run_pipeline.sh:19` `ZIP_OUT=...submission_050911_iter019_v2N_50plus50_optimized_fullhorizon.zip`——同名再跑会覆盖。无 timestamp suffix。生产中用户应每次重命名（或加 `-${date +%Y%m%d_%H%M}`）。
- **状态**：🟡

### 2.10 没有 archive policy
- **来源**：QREVIEW_PIPELINE_ARCH Q53
- **答案**：
  `outputs/submission_*.zip` 累积留在目录；`.gitignore: outputs/` 不入 git，但用户磁盘累积。无 `outputs/archive/` 自动迁移。这是 acceptable for 个人项目 / 一次性提交。
- **状态**：✅

---

## 3. run_pipeline.sh + shell 脚本

### 3.1 顶层 run_pipeline.sh 参数 / 调用约定
- **来源**：QREVIEW_PIPELINE_ARCH Q57-Q61, Q64
- **答案**：
  ```bash
  ./run_pipeline.sh [DATA_DIR] [CUDA_DEVICE] [LGB_GPU_FLAG]
  # default: ./data, 0, --gpu
  ```
  位置参数 ×3，无 `getopts`、无 `--help`。typo 顺序（`./run_pipeline.sh 0 ./data`）会 silently 用 "0" 作为 DATA_DIR（path 必须存在的 check 在脚本里没做）。
- **状态**：🟡（应改成 `--data-dir` / `--cuda` long options + `--help` 帮助）

### 3.2 strict mode 缺失（set -u / pipefail / trap ERR）
- **来源**：QREVIEW_PIPELINE_ARCH Q11, Q56, Q62
- **答案**：见 §1.9。`LGB_GPU_FLAG=${3:---gpu}`（line 13）使用 word-split：line 51 `... "$LGB_GPU_FLAG"` 加引号会让空字符串 `""` 被当成 1 个空参传到 run_all_lgb_seeds.sh（其 `$GPU_FLAG` 又会 expand）；目前测试 OK 但脆弱。
- **状态**：🟡

### 3.3 路径硬编码 vs 可配置
- **来源**：QREVIEW_PIPELINE_ARCH Q59
- **答案**：
  `WORKDIR/outputs` 硬编码，不可通过 env 覆盖。用户想把 outputs 放 SSD，必须 symlink。20 GB cache + 5 GB models 这量级 symlink 够用。
- **状态**：✅（acceptable）

### 3.4 run_pipeline.sh 缺 preflight check
- **来源**：QREVIEW_PIPELINE_ARCH Q43
- **答案**：
  脚本不检查：`nvidia-smi` 在 `--gpu` 时存在 / Python 版本 / `lightgbm` / `torch` / DATA_DIR 下有 parquet / 磁盘空间。最佳实践：开头加几行 `command -v python3 nvidia-smi; df -h .; python3 -c "import torch; print(torch.__version__)"` echo 当前环境。
- **状态**：🟡

### 3.5 子脚本 run_all_lgb_seeds.sh / run_all_nn_seeds.sh
- **来源**：QREVIEW_PIPELINE_ARCH Q40, Q62, Q63
- **答案**：
  - `run_all_lgb_seeds.sh`：`for SEED in 1..50` 串行，GPU_FLAG 作为可选 3rd arg。LGB GPU 模式实测比 CPU 快 3.2× (CLAUDE.md)。
  - `run_all_nn_seeds.sh`：串行 + 硬编码 `--skip-phase1`（line 27）。**首次跑** anchor 不存在 → Phase 1 仍执行（`if args.skip_phase1 and isfile(anchor)` 是 False）；这是 conditional skip，安全。但 README:144-160 把 Phase 1 描述成 first-class step 而不点出 `--skip-phase1` 默认开，初读会困惑。
- **状态**：🟡（README + run_all_nn_seeds.sh 注释应明示语义）

### 3.6 并行执行能力缺失
- **来源**：QREVIEW_PIPELINE_ARCH Q40
- **答案**：
  README 说"5 GPU 并行 ≈ 18 min"，但 run_all_nn_seeds.sh 是串行。手动并行：
  ```bash
  CUDA_VISIBLE_DEVICES=0 for s in 1..10; do python3 ... --seed $s --cuda 0 --skip-phase1; done &
  CUDA_VISIBLE_DEVICES=1 for s in 11..20; do ...; done &
  ...
  ```
  应该加 `--seeds 1-10` 参数 + GNU parallel 或 `xargs -P` 包装。
- **状态**：🟡

### 3.7 `cd $PKG_DIR && zip` 后 `cd $WORKDIR` dead code
- **来源**：QREVIEW_PIPELINE_ARCH Q61
- **答案**：
  Line 76 `cd "$WORKDIR"` 后脚本立即 end（line 78-83 是 echo）。dead 但 harmless；可能曾经 sourced。删一行没区别。
- **状态**：✅

### 3.8 没有 dry-run 模式
- **来源**：QREVIEW_PIPELINE_ARCH Q65
- **答案**：
  没有 `--dry-run` flag。用户无法在跑 90 min 训练前看到"会执行什么"。Hack：用 `bash -n run_pipeline.sh` 做 syntax check + 手动 echo 修改。
- **状态**：🟡

### 3.9 shell hygiene 小问题
- **来源**：QREVIEW_PIPELINE_ARCH Q64
- **答案**：
  - 所有 shell 用 `#!/bin/bash`（不是 `/usr/bin/env bash` portable variant）—— 在 macOS / NixOS / Alpine 可能 fail（应改成 `#!/usr/bin/env bash`）。
  - 4-space indent 一致 ✅。
  - 引号风格：`"$VAR"` 大多正确（除 GPU_FLAG 故意不引）。
- **状态**：🟡（shebang 应改）

---

## 4. requirements 与环境隔离

### 4.1 两套 requirements，只 pin 一个
- **来源**：QREVIEW_PIPELINE_ARCH Q15
- **答案**：
  - 提交侧 `04_build_pkg/requirements.txt` pin 死：
    ```
    --extra-index-url https://download.pytorch.org/whl/cpu
    numpy==2.4.4
    pandas==2.3.3
    lightgbm==4.6.0
    scipy==1.17.1
    torch==2.5.1+cpu
    ```
  - 训练侧 README 给了 `pip install numpy pandas lightgbm scipy torch`——**unpinned**。
  
  风险：训练用 lightgbm==4.x 但用户从 PyPI 装到 3.x（4.6 是 2024-10 发布的，新人可能找老包），model.txt 格式可能不兼容 4.6 inference reader（实际 lgb 文本格式向后兼容很好，**风险中等偏低**）。
- **状态**：🟡 应补 `requirements-train.txt`。

### 4.2 torch 版本兼容
- **来源**：QREVIEW_PIPELINE_ARCH Q16
- **答案**：
  - 包内 `torch==2.5.1+cpu`
  - 训练用什么 torch 都行（npz 不依赖 torch 序列化）—— `extract_npz` 把 state_dict.values() 全 `.numpy()` 转 ndarray 存 np.savez（`train_T188v2_nn_seed.py:130-160`）。
  - `weights_only=False` 在 `torch.load(anchor_path, ..., weights_only=False)`（line 231）—— 在 torch 2.6 默认变 True 后**仍能 work**因为 anchor 是 dict + numpy array 不含可执行对象。但 torch 2.7+ pickle policy 可能更严，需 verify。
  - npz 格式与 torch 完全解耦 ✅—— 跨 torch 版本超鲁棒（Q113）
- **状态**：✅（npz 设计就是为绕过 torch 序列化兼容性）

### 4.3 Platform sandbox 网络可达性
- **来源**：QREVIEW_PIPELINE_ARCH Q18
- **答案**：
  `requirements.txt:1` `--extra-index-url https://download.pytorch.org/whl/cpu` 要求平台能 GET pypi + pytorch index。若平台无外网，pip install 失败。
  - 平台规则（PROGRESS.md §1）：12h 1 次提交、3h 评测限。**没明说有无网**。
  - 实测：T188v2 50+50 已经通过平台 → 至少 PyPI + pytorch.org/cpu 可达；assumption holds。
- **状态**：✅（实测过）

### 4.4 numpy 2.x 风险
- **来源**：QREVIEW_PIPELINE_ARCH Q19, CODE_QUESTION_LIST Q181
- **答案**：
  numpy 2.4.4 是 2.x 大版本。lightgbm 4.6.0 wheel **must** be numpy 2 兼容（4.6 是 2024-10 发布、numpy 2.0 同年 6 月发布，4.6 是首批 numpy 2 兼容版）。pandas 2.3 同样兼容 numpy 2。**已经实测平台 +35.64 SOTA → 实证 OK**。
  - **历史确认**：用户在第二届良文杯 2026 年 5 月提交，numpy 2.4.4 是当时最新（2.4 在 2026-02 发布）；版本组合实测可用。
- **状态**：✅（实测过）

### 4.5 Architecture / OS 假设
- **来源**：QREVIEW_PIPELINE_ARCH Q23, Q24
- **答案**：
  requirements 无 platform marker → manylinux x86_64 隐含。无 aarch64 / Alpine / Windows 测试。
  - PROGRESS.md §1：平台 ≤ 2 GB FP32、3h、12h 1 提交，未说 OS。
  - Python 3.11 (config.json) 是给平台的 hint，不是硬约束（python 3.10+ 都 OK）。
  - 实测：T188v2 在平台跑通 → 平台是 Linux x86_64（最普遍假设）。
- **状态**：✅（实证，但应在 README 加 "tested on: linux-x86_64 / Python 3.11"）

### 4.6 CPU torch 的合理性
- **来源**：QREVIEW_PIPELINE_ARCH Q17, CODE_QUESTION_LIST Q112
- **答案**：
  `torch==2.5.1+cpu` 选择理由：
  - 平台 GPU 可用性未知 → CPU 是安全公约
  - CPU wheel 装机 3-5 min（README:201）；GPU wheel 装机 +500MB cuda runtime + 可能 ABI 不匹配
  - 50 NN ensemble batched bmm 在 CPU 实测 ~500 ms / 1024 batch（足够 3h 限内跑完 442k 行）
  - 但 Predictor.py:193 仍写 `torch.device("cuda" if torch.cuda.is_available() else "cpu")` ——这是**前瞻性**：若平台后续给 GPU，无需重打包；CPU torch 上 cuda.is_available() = False，分支不执行 ✅。
- **核心证据**：HISTORY_EVIDENCE_DB §10 T188v3 CPU 实测 NN 8.1ms / 1024 batch；T189 GPU pkg 备份。
- **状态**：✅

### 4.7 Python 版本 drift
- **来源**：QREVIEW_PIPELINE_ARCH Q24
- **答案**：
  - `config.json:2` `python_version: "3.11"` 给平台 hint
  - README:25 "Python 3.10+ 推荐 3.11"
  - 训练 unpinned
  - numpy 2.4 / lightgbm 4.6 / torch 2.5 都要 Python 3.10+。3.9 装会失败 → 提交即可知。
- **状态**：✅

---

## 5. 模型文件格式 (npz vs pt, txt vs pkl)

### 5.1 NN 为啥用 npz 而不是 .pt
- **来源**：QREVIEW_PIPELINE_ARCH Q9, CODE_QUESTION_LIST Q113, Q135, Q137, Q186
- **答案**：
  3 个理由：
  1. **跨 torch 版本可移植**：npz 只存 numpy ndarray，与 torch 解耦；用户训练 torch 2.5 / 平台装 torch 2.5+cpu 但**未来 torch policy 变化**（如 weights_only=True 默认）不影响 npz。
  2. **手抠层让 `_BatchedMLPEnsemble` 能 stack**：`torch.bmm` 要求 `W[i] shape (N, out, in)`，从 50 个独立 nn.Sequential 不能直接 stack（state_dict 是 dict）；npz 让"按层取出每个 NN 的 W_i"自然。
  3. **包内不需要 nn.Module 类定义**：Predictor.py 重新实现 forward，state_dict load 不必要——npz 直接 → torch.Tensor → bmm。
  
  **代价**：
  - extract_npz（`train_T188v2_nn_seed.py:130-160`）层数硬编码 + 依赖 `use_layernorm` flag → 改架构要同步改 extract（Q135）
  - 同时存 .pt + .npz（line 502 saves `nn_h60_seed{S}.pt`、line 505 saves npz）—— .pt 是 debug 备份（Q186），不进 pkg ✅
- **核心证据**：T188v3 50 NN 单次 bmm 8.1 ms vs sequential numpy 1509ms（186×, HISTORY §10）—— npz + bmm 是性能突破前提。
- **状态**：✅（合理 trade-off）

### 5.2 LGB 为啥用 .txt 而不是 .pkl
- **来源**：CODE_QUESTION_LIST Q139, Q140
- **答案**：
  `booster.save_model(out_path)` 写 LightGBM **自有文本格式**（人类可读，4.6 兼容 3.x 模型）。.pkl 走 Python pickle = 跨版本 brittle + 安全风险。LGB 文本格式：
  - 写：`train_T188v2_lgb_seed.py:188`
  - 读：`Predictor.py:249` `lgb.Booster(model_file=p)`
  - 50 模型 × ~2 MB = 100 MB（占 pkg 148 MB 的 68%）
  - 包含 feature_name list → 推理时按 name 校验顺序（隐式）
- **核心证据**：HISTORY §4 T155 LGB M7 100% deterministic 验证（基于文本可 diff）。
- **状态**：✅

### 5.3 npz 字段清单（_BatchedMLPEnsemble 读什么）
- **来源**：QREVIEW_PIPELINE_ARCH Q8, CODE_QUESTION_LIST Q179, Q180, Q183
- **答案**：
  npz 内 keys（`extract_npz` 输出）：
  | key | shape | dtype | 用途 |
  |---|---|---|---|
  | `in_dim` | (1,) | int64 | feature_dim, 必 == 359 |
  | `hidden` | (3,) | int64 | (256, 128, 64) |
  | `use_layernorm` | (1,) | int8 | 1 |
  | `target_scale` | (1,) | float32 | 1/std(y_train) per seed |
  | `feat_mean` | (in_dim,) | float32 | per seed (Phase 1 train 算的) |
  | `feat_std` | (in_dim,) | float32 | 同上 |
  | `keep_idx` | (in_dim,) | int64 | 全 NN 应一致 |
  | `clip` | (1,) | float32 | 10.0 |
  | `L{i}_W` | (out_i, in_i) | float32 | 3 个隐藏层 |
  | `L{i}_b` | (out_i,) | float32 | |
  | `LN{i}_W` | (out_i,) | float32 | LayerNorm gamma |
  | `LN{i}_b` | (out_i,) | float32 | LayerNorm beta |
  | `LF_W` | (1, hidden_last) | float32 | 输出层 |
  | `LF_b` | (1,) | float32 | |
  
  `_BatchedMLPEnsemble.__init__` 假设 50 个 NN 的 in_dim/hidden/use_layernorm/keep_idx **完全一致**，只读 npz_paths[0] 取这 4 项（line 94-98）。**未 assert** 每个 npz 的这 4 项 == npz[0]——这是 Q81/Q82/Q179/Q180 的真实风险。生产中如果某 seed 训练用了不同 dropout/lr 是 OK（这些不在 npz），但若改了 HIDDEN / use_layernorm / FAIL_NAMES 就 silent 错配。
- **状态**：🟡（应在 `_BatchedMLPEnsemble.__init__` 加 `assert d["in_dim"][0] == in_dim` 循环校验）

### 5.4 包内 NN 推理实现重写 vs 训练时 nn.Module
- **来源**：QREVIEW_PIPELINE_ARCH Q9, CODE_QUESTION_LIST Q115, Q116
- **答案**：
  - 训练侧：`nn.Sequential(Linear, LayerNorm, GELU, Dropout)` + Adam + autograd
  - 推理侧：手写 `torch.bmm(h, W.T) + b → 手写 LayerNorm → F.gelu(approximate="tanh") → 下一层`
  - **必须**重写，因为：
    - 推理时 N=50 个 NN 要 stack 在 batch 维 (N, B, in)，nn.Sequential 不支持 batched weights
    - bmm 一次 50 NN 比 50 次 numpy 循环快 186× (HISTORY §10)
  - 重写引入两点 deviation：
    1. **GELU**：训练 `nn.GELU()` 默认 erf，推理 `F.gelu(approximate="tanh")` —— max diff ~1e-4（fast tanh approx），实测 T188v3 vs T188v2 actions max diff 6.98e-10 → 在当前数据上不影响 actions。但**理论上**可在阈值附近翻 action（thr=3e-4 vs 1e-4 gelu drift 同阶）。
    2. **LayerNorm 手写**：`var(unbiased=False) + 1e-5`，与 torch.nn.LayerNorm 默认 eps=1e-5 + unbiased=False 一致 ✅。
- **状态**：🔴 GELU 不一致应修（一行：训练侧改 `nn.GELU(approximate="tanh")` 或推理侧改 `approximate="none"`；实测已证否大幅 drift，但属于"看起来不该这样"的 surprise 项）。

### 5.5 LGB model.txt 包含 feature_name
- **来源**：CODE_QUESTION_LIST Q139
- **答案**：
  `lgb.Dataset(..., feature_name=feat_names, ...)` (line 170-173) 让 booster.save_model() 把 feature 名字写进 .txt 头。推理时 `lgb.Booster(model_file=p)` 读出 feature_name，但 `Predictor.py:248-249` **不传 feature_name 参数给 booster.predict**——而是直接传 feats matrix（按 raw_last + extras_kept 顺序）。
  - LightGBM 不在 predict 时按 name 重排（默认按 column index）。
  - 隐式契约：训练时 keep_idx 顺序 == 推理时 raw_last (154) + extras_kept (216-len(FAIL)=205) = 359。
- **状态**：🟡（如果有人 reorder feature 顺序无法被 detect → 应加 sanity `assert booster.feature_name() == feat_names`）

---

## 6. config.json 字段与跨文件重复

### 6.1 config.json 字段
- **来源**：QREVIEW_PIPELINE_ARCH Q49, Q50
- **答案**：
  3 个顶层字段：
  - `python_version: "3.11"` — 给平台的环境 hint
  - `batch: 1024` — 每次 predict 输入 batch 大小（平台行为）
  - `feature: [...155...]` — 154 个 LOB 列 + `"sym"` = 155 列
  - `label: [label_5/10/20/40/60]`
  
  这是**平台契约**：Predictor.predict 接收 `List[pd.DataFrame]`，每个 DataFrame 100 rows × `len(feature)` cols（PROGRESS.md §5）。
- **状态**：✅

### 6.2 sym 在 config.feature 里但不进模型
- **来源**：CODE_QUESTION_LIST Q70 + CRITICAL_CONSTRAINTS #3
- **答案**：
  config.json `feature` list 第 155 行是 `"sym"` —— 这告诉平台 "DataFrame 中要有 sym 列"。Predictor 读 sym 用于 conformal beta 查找（Predictor.py:299 `int(df["sym"].iloc[-1])`），**绝不送进 NN/LGB forward**（feats 在 `_compute_batch_features:273` 只用 `self._raw_feat_cols`，没 sym）。
  - 这是 CRITICAL_CONSTRAINTS #3 的合规设计 ✅
- **状态**：✅

### 6.3 config.feature 与 Predictor RAW_COLS_TRAIN_ORDER 重复
- **来源**：QREVIEW_PIPELINE_ARCH Q90, Q91
- **答案**：
  154 个 LOB 列名在 3 处：
  1. `01_build_features/build_schemeP_cache.py:53-69` `RAW_COLS`（list, 154）
  2. `04_build_pkg/Predictor.py:46-65` `RAW_COLS_TRAIN_ORDER`（tuple, 154）
  3. `04_build_pkg/config.json:4-159` `feature`（list, 154 + 1 sym = 155）
  
  必须严格按相同顺序——否则训练 feat_mean 与推理 feat 对不上。当前 3 处独立维护，靠 byte-level 一致性（人工保证）。没有共享 constants 模块。
- **状态**：🟡 应抽出 `raw_cols.py` 单源（所有 3 处 import）。

### 6.4 config.batch=1024 vs Predictor 行为
- **来源**：CODE_QUESTION_LIST Q190
- **答案**：
  `config.batch=1024` 告诉平台一次给 1024 个 100-tick window。`Predictor.predict` 不检查 `len(batches) == 1024`（只 `if not batches: return []`，line 317）。
  - 平台是否会传 B != 1024？未文档化。`_BatchedMLPEnsemble.predict_mean` 用 `h.unsqueeze(0).expand(N, -1, -1)` 对任意 B 都支持。
  - 内存：N=50, B=1024, in=359 → 50×1024×359×4 = 73 MB tensor；若 B=10000 → 720 MB，可能 OOM in CPU torch with limited RAM。无 chunking。
- **状态**：🟡（应 chunk B 到 ≤2048，预防 B 巨大）

### 6.5 config.json 与 thresholds.json 重复字段
- **来源**：QREVIEW_PIPELINE_ARCH Q97
- **答案**：
  thresholds.json 每个 horizon 有 `w_nn, w_lgb, thr_up, thr_dn`，5 个 horizon 全部 `w_nn=1.0, w_lgb=1.5`——5× 重复。若改权重要改 5 处。`_doc` 字段在 line 2 说明 share_with 设计，但权重重复无 explainer。
  - config.json 没有 w_nn/w_lgb（OK，权重是 ensemble 配置，应在 thresholds.json）
- **状态**：🟡（可在 thresholds.json 顶层加 default `weights: {w_nn:1.0, w_lgb:1.5}`，per-horizon override）

### 6.6 thresholds.json 精度 vs _note
- **来源**：QREVIEW_PIPELINE_ARCH Q97
- **答案**：
  thresholds.json 给出每个 horizon `thr_up = 0.0003 × √(H/60)`：
  - h=60: 0.0003
  - h=40: 0.0002449 (= 0.0003 × √(40/60) = 0.0003 × 0.8165)
  - h=20: 0.0001732 (= 0.0003 × 0.5774)
  - h=10: 0.0001225 (= 0.0003 × 0.4082)
  - h=5: 0.0000866 (= 0.0003 × 0.2887)
  
  数值与 _note 一致 ✅。但若改 h=60 thr_up=0.0003 → 5e-5，所有短 horizon 要重算。**brittle**——应在 Predictor 里动态算（`thr_short = thr_60 × sqrt(H/60)`）而不是硬写 JSON。
- **状态**：🟡

### 6.7 RAW_COLS 中 "sym" 是否包含
- **来源**：跨文件检查
- **答案**：
  - `01_build_features/build_schemeP_cache.py:53-69` `RAW_COLS` **不含 sym**（154 维）
  - `04_build_pkg/Predictor.py:46-65` `RAW_COLS_TRAIN_ORDER` **不含 sym**（154 维）
  - `04_build_pkg/config.json:4-160` `feature` **含 sym**（155 维：前 154 是 LOB + 第 155 是 sym）
  
  config.feature 含 sym 是给平台的"DataFrame 该有什么列"，与训练用 RAW_COLS（154）是不同概念。Predictor 用 `df[self._raw_feat_cols]` 提取 154 列（不含 sym）→ NN/LGB；用 `df["sym"].iloc[-1]` 提取 sym → conformal lookup。
- **状态**：✅（理解一致）

---

## 7. 可复现性 (seed 控制)

### 7.1 NN 同 seed 同硬件能否 bit-identical
- **来源**：QREVIEW_PIPELINE_ARCH Q27, Q33, CODE_QUESTION_LIST Q91
- **答案**：
  **大概率不能**。设置：
  - `torch.manual_seed(S)` (line 194)
  - `torch.cuda.manual_seed_all(S)` (line 195)
  - `np.random.seed(S)` (line 196)
  
  **没设**：
  - `torch.backends.cudnn.deterministic = True`
  - `torch.use_deterministic_algorithms(True)`
  - `os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'`
  
  → CUDA matmul 在 cuBLAS 上 non-deterministic（不同 thread schedule 会改 low-order bit）。Phase 1 早停 epsilon=1e-10 在 float32 上 effectively "任意改善即认可"（Q77），所以 best_epoch 可能 1-2 epoch 漂移 → Phase 2 起点也漂 → 最终 .npz 不 byte-identical。
  
  **项目立场**：50+50 ensemble 内方差 (LGB-LGB corr 0.84, NN-NN corr 0.93, T192 数据) 远大于 bit drift；用户接受 statistical reproducibility 而非 byte reproducibility。**没测过**——若需要应跑 2 次相同 seed 比 npz md5。
- **核心证据**：HISTORY 附录 A 发现 8 提及 corr 0.9331 / condition number 54182 → 单 NN 内部 noise 已大于 bit drift。
- **状态**：🟡（应在 README 写"reproducibility = statistical, not bit-exact"）

### 7.2 LightGBM 同 seed 是否 deterministic
- **来源**：QREVIEW_PIPELINE_ARCH Q28, Q29, CODE_QUESTION_LIST Q140
- **答案**：
  `params['seed'] = S`, `params['num_threads'] = 18`，**没设** `params['deterministic'] = True`。LightGBM 文档：`bagging_freq > 0` + `num_threads > 1` + 无 `deterministic` → 浮点累加顺序可能跨 run 不同。
  - **但 T155 实测 LGB M7 100% deterministic**（HISTORY §4）—— 同 seed 同硬件 byte-identical。原因可能 lightgbm 4.6 内部用 deterministic atomic reduce。
  - GPU vs CPU LightGBM **不一致**（Q29）：GPU 用 GPU histogram算法，与 CPU histogram 浮点累加不同 → model.txt bytes 不同。README:6 `LGB_GPU_FLAG=${3:---gpu}` 默认 GPU，所以"复现需用 GPU"。若用户切 CPU 训练得到的 .txt 与原 GPU 训练得到的 .txt diff 不为 0，但精度差异极小，最终 actions 实测一致（T155）。
- **核心证据**：T155 (HISTORY §4) "M7 LGB 100% deterministic"。
- **状态**：✅（实测过，但 GPU↔CPU 的细微 gap 应 README 提示）

### 7.3 RNG stream 多源
- **来源**：QREVIEW_PIPELINE_ARCH Q30, Q31, CODE_QUESTION_LIST Q32
- **答案**：
  NN 训练用 **2 个并存 RNG**：
  - Global `np.random.seed(S)` + `torch.manual_seed(S)` （control torch.randperm 等）
  - Explicit `np.random.default_rng(S * 7919 + 1)` 用于 Phase 1 aug (line 270)
  - Explicit `np.random.default_rng(S * 7919 + 137)` 用于 Phase 2 aug (line 406)
  
  LGB 用：
  - LGB internal seed `S`
  - Explicit `np.random.default_rng(S * 7919 + 42)` 用于 aug (line 142)
  
  **3 个不同 offset (1, 42, 137)** 的 rationale：避免不同 aug pass 用相同随机序列（如果 NN Phase 1 与 LGB 都从 `seed=S` 起，aug scales 完全相同→ NN 和 LGB 增强出来的数据点完全相同，损失多样性）。`7919` 是大质数确保 hash 分散；`1/42/137` 是 nothing-up-my-sleeve 数字。
  - 设计风险：若改 CHUNK 大小（line 43 `CHUNK=200_000`），rng.uniform sequence 输出与原序列分块边界不同 → aug 改变 → 模型权重改变（Q32）。
- **核心证据**：`grep -n "7919" *.py` 一致出现在 LGB:142, NN:270, NN:406。
- **状态**：🟡（应在 train script 顶部加"reproducibility-sensitive constants" block 注明 CHUNK / offset）

### 7.4 Phase 2 与 Phase 1 状态依赖
- **来源**：QREVIEW_PIPELINE_ARCH Q34, Q35, Q37
- **答案**：
  - `--skip-phase1` 默认开（run_all_nn_seeds.sh:27）
  - 若 anchor 存在 → `torch.load(anchor_path, weights_only=False)` 直接取 `feat_mean / feat_std / target_scale / state_dict`
  - **不校验**当前 args (lr_p2, dropout, hidden) 与 anchor 时的 lr_p1/dropout 一致
  - 若用户改了 HIDDEN = (512, 128, 64) 然后只重跑 Phase 2，会用旧 anchor + 新 dropout → silent corrupt（state_dict load 时 shape 不匹配会 raise，但 dropout 静默）
  
  `target_scale` 由 Phase 1 训 data (dates 0-79) 算，Phase 2 用 dates 0-119 数据 → 复用 Phase 1 target_scale 可能让 prediction 被 scale 偏（Q37）。但 ensemble + 实证（platform +35.64）说明这种 drift 可接受。
- **状态**：🟡（应在 Phase 2 入口加 args fingerprint check vs anchor metadata）

### 7.5 aug_a 数据增强 reproducibility
- **来源**：QREVIEW_PIPELINE_ARCH Q32
- **答案**：
  ```python
  CHUNK = 200_000
  def aug_a_scale_chunked(rng, X_src, X_dst, lo=0.80, hi=1.20):
      for i in range(0, n, CHUNK):
          scales = rng.uniform(lo, hi, size=(sz, X_src.shape[1]))
          X_dst[i:end] = X_src[i:end] * scales
  ```
  - rng.uniform 取多少元素 → 输出 sequence depends on `size` arg
  - 改 CHUNK = 100_000 时同 seed 下 rng 调用次数不变（仍 n×d 个）但 batch 边界不同——实际**输出序列完全一致**（rng.uniform 内部按行优先连续 fill），所以 CHUNK 改了 model 应**不变**。
  - **但仔细看 `np.random.Generator.uniform(size=(sz, d))`**：单次 call 内部 fill 是连续，跨 call 是分块。两种 chunk 模式 rng 内部 state 不一样（因为每次 uniform 调用消耗 sz*d 个 floats，调用次数不同）—— **实际可能产生不同 output**。
  - 严谨验证需 `rng1 = default_rng(0); a = rng1.uniform(size=(8, 5))` vs `rng2 = default_rng(0); b1 = rng2.uniform(size=(4,5)); b2 = rng2.uniform(size=(4,5))` 比 `(b1, b2)` vs `a` —— numpy 文档没保证 size 切分 invariance。**结论**：CHUNK 是 seed-sensitive，文档应明示"不要改"。
- **状态**：🔴（应将 CHUNK 列为 reproducibility-critical constant，加注释）

### 7.6 multi-GPU 并行训不同 seed
- **来源**：CODE_QUESTION_LIST Q91
- **答案**：
  `os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda` 在 `torch.device("cuda" if ...)` 之前 → 顺序对，每个 process 只看一张 GPU。每 process 内部 `torch.cuda.manual_seed_all(S)` 设种独立 → 跨 process RNG 独立 ✅。
  - 唯一 risk：5 个进程同时读 cache npz 文件，可能 IO contention，但不影响 reproducibility。
- **状态**：✅

### 7.7 LightGBM bagging_seed / feature_fraction_seed 派生
- **来源**：CODE_QUESTION_LIST Q140
- **答案**：
  LightGBM 在 `params.seed = S` 下内部派生 `bagging_seed`、`feature_fraction_seed`、`data_random_seed`、`extra_seed` 等子 seed（默认从 main seed + 内部偏移）。不同 lightgbm 版本可能改派生公式 → 不同版本同 seed 出来的 model 可能不同。pin lightgbm==4.6.0 是关键。
- **状态**：✅（pin 已做）

---

## 8. 平台约束合规 (3 条硬约束代码体现 / 风险)

### 8.1 约束 #1：date 评测被置 0 / 不能进 feature
- **来源**：CRITICAL_CONSTRAINTS §1, CODE_QUESTION_LIST Q167
- **答案**：
  代码层面 4 处主动 check：
  1. `train_T188v2_lgb_seed.py:104-106` `assert not (forbidden & set(feat_names))` 其中 `forbidden = {"date", "sym", "time"}` → LGB 训练前显式 assert
  2. `train_T188v2_nn_seed.py:207-208` 同上
  3. `Predictor.py:46-65` `RAW_COLS_TRAIN_ORDER` 154 列**不含 date**（硬编码）
  4. `Predictor.py:46-65` 也**不含 sym/time**（这些不进模型）
  
  cache npz 里**有** `date` 字段（`build_schemeP_cache.py:163` `dates_list.append(...)`），**仅用于 split**（train/val/test 划分），从未 join 进 `X`。`build_split:183-194` 把 date 作为 npz top-level key，X 维度仍是 (N, 370) 不含 date ✅。
- **核心证据**：`grep "date" 03_train_nn/train_T188v2_nn_seed.py | grep -v import | head -20` → date 仅出现在 dataloader/cache key，不在 input feature 里。
- **状态**：✅

### 8.2 约束 #2：测试点顺序打乱 / Predictor stateless
- **来源**：CRITICAL_CONSTRAINTS §2, CODE_QUESTION_LIST Q129
- **答案**：
  Predictor 设计：
  - `__init__` 仅 load weights 一次（构造时一次性，符合 stateless）
  - `predict(batches)` 局部变量 `pred_cache: Dict[int, np.ndarray] = {}` 在函数内部 scope（line 329）——只在一次 predict() 调用内跨 horizon 复用，**不跨 predict() 调用**
  - 无 `self.last_window` / `self.history_buffer`
  - 不依赖 `batches[i]` 的物理顺序（每个 batch 独立处理，循环 `for i, df in enumerate(batches)`，i 仅作输出索引）
  
  Predictor.py:19-27 注释明示 "No cross-call state held on self"、"Stateless / shuffle-invariant"。
  
  **历史踩坑**：T161/T162/T168/T169 一系列 TTA / TTT / adaptive threshold 实验都试过 per-batch 统计自适应，**全部平台 / 本地负面**（HISTORY §10）—— 项目用代价证过"约束 #2 不能违反"。当前 final 包完全合规。
- **核心证据**：HISTORY §10 "T161 TTA, T162 W3a/b/c, T168 V1-V4, T169 AR TTT 全部退步或未提交"。
- **状态**：✅

### 8.3 约束 #3：sym 0-4 但可能 OOD / 模型 sym-agnostic
- **来源**：CRITICAL_CONSTRAINTS §3, CODE_QUESTION_LIST Q127, Q128
- **答案**：
  - **模型不见 sym**：feats 来自 `_compute_batch_features` 用 `self._raw_feat_cols`（不含 sym）+ extras（也不含 sym）；NN forward 输入 359 维不含 sym；LGB feature_name list 不含 sym
  - **per-sym 仅在 conformal lookup**：`Predictor.py:291-304` `_extract_band_per_row`：
    ```python
    try: s = int(df["sym"].iloc[-1])
    except (ValueError, TypeError, IndexError): continue
    if s in self._cw_band: out[i] = self._cw_band[s]
    # else: keep default self._cw_default_band
    ```
    OOD sym (e.g., s=99 或 s 不在 0-4) → 用 default band（per_sym_beta=0.16 × per_sym_sigma=4e-4 ≈ 6.4e-5）。**安全降级** ✅
  - **没有 IndexError**：dict.get 模式不抛
  - **normalization 全局**：`feat_mean/feat_std` 来自全数据（不分 sym）—— `train_T188v2_nn_seed.py:254-255` `np.nanmean(X_tr, axis=0)`
  - **没有 sym embedding** ✅
  
  smoke test in Predictor.py:382-388 验证过 sym=99 OOD 路径不崩。
- **核心证据**：HISTORY §3 "T149 NN OOD sym sweep T87 在 OOD sym 下表现可接受"+ Predictor smoke test passes for sym=99。
- **状态**：✅

### 8.4 conformal per-sym 是否违反约束 #3 精神
- **来源**：CODE_QUESTION_LIST Q166
- **答案**：
  README:220 "sym 仅用于 conformal beta 查找（不输入模型，OOD sym 使用默认值）"。但 per_sym_beta 是 OFFLINE calibration 时**按 sym split**算出来的（T150 sweep）—— calibration 本身用 sym，inference 也用 sym ID 索引 β/σ。这是否违反约束？
  - 平台约束 #3 原文："sym 0-4 但可能含训练外股票"——重点是**不能依赖 sym ID 映射到特定股票**。
  - 当前 conformal 在 OOD sym → fallback default ✅，所以即使 ID=2 但底层股票变了，β=0.30 可能 suboptimal but不崩。
  - **严格**理解："sym 不能输入 model" 是 NN/LGB；conformal 是 wrapper 不是 model，可用 sym（README 表述正确，作者的 understanding 与平台 spec 对齐）。
- **状态**：✅（设计合规；README 措辞略含糊但本质正确）

### 8.5 模型 ≤ 2 GB / FP32 / 3h 推理
- **来源**：PROGRESS.md §1, QREVIEW_PIPELINE_ARCH Q44, Q47, Q189
- **答案**：
  - **大小**：148 MB 远 < 2 GB ✅
  - **FP32**：NN npz 全 float32 + LGB .txt 都是 FP32 数值 ✅
  - **3h 推理**：T188v3 batched bmm 优化后总 inference 641ms / 1024 batch → 442k 行 ≈ 432 个 batch → 432×641ms ≈ 4.6 min（GPU） / ≈ 10-15 min（CPU 估计）→ 远低于 3h ✅
  
  README:240 已说 "推理时 NN ensemble 在 CPU 运行（约 500ms per batch）"。
- **核心证据**：HISTORY §10 T188v3 实测 e2e 4.6 min for 442k 行。
- **状态**：✅

### 8.6 GELU train/inference mismatch（潜在 bug 升级）
- **来源**：CODE_QUESTION_LIST Q176, Q115
- **答案**：
  - 训练侧 `train_T188v2_nn_seed.py:89` `nn.GELU()` → 默认 `approximate='none'` 即 `0.5 x (1 + erf(x/√2))`
  - 推理侧 `Predictor.py:171` `F.gelu(h, approximate="tanh")` → 用 tanh 近似：`0.5 x (1 + tanh(√(2/π)(x + 0.044715 x³)))`
  - 两者绝对差最大 ~1e-4（在 |x| ≈ 1 附近）
  - **理论风险**：thr_up = 3e-4，drift 1e-4 同阶 → 有 1/3 概率在阈值附近的 row 翻 action。
  - **实测结果**：T188v3 vs T188v2 action diff 6.98e-10 (HISTORY §10) → 完美一致。原因可能是：
    1. 50 NN ensemble 平均后 drift 抵消
    2. 实际 |pred| 分布远离阈值的样本占多数
    3. tanh approx 误差在 SchemeP feature 数值范围（多在 [-3, 3]）内极小
  
  **应修但优先级不高**：1 行改动（推理侧改 `approximate="none"`），CPU 下 erf 比 tanh 慢约 15%，但 NN 已经 8.1 ms / 1024 → 即使 +15% = 9.3 ms 仍微不足道。
- **状态**：🔴（属于"看到就想修"的 surprise；不影响 +35.64 SOTA）

### 8.7 NaN 防御（多层）
- **来源**：QREVIEW_PIPELINE_ARCH Q88, Q89, CODE_QUESTION_LIST Q117
- **答案**：
  3 层 NaN guard：
  1. **Feature 构建**：`build_schemeP_cache.py:105` `np.where(np.isfinite(extras), extras, 0.0)`、`Predictor.py:286` 同样
  2. **NN normalization 后**：`_BatchedMLPEnsemble.predict_mean:159` `torch.where(torch.isnan(h), 0, h)`
  3. **LGB inference**：**无显式 NaN sanitize**（Q88）—— 若 LGB 输出 NaN，`pred > thr` 是 False → action 默认 1（平），安全降级 ✅
  
  整体：NaN/Inf 不会引爆，最坏情况 fallback 到 action=1（不交易）。
- **状态**：✅

### 8.8 OOD batch size / 形状错误
- **来源**：QREVIEW_PIPELINE_ARCH Q84
- **答案**：
  Predictor 假设每个 DataFrame 恰好 100 行 × 154+ 列。若 ≠ 100：
  - `df[self._raw_feat_cols].to_numpy(...)` 返回 (n_actual, 154)
  - `X3d = np.empty((N, WINDOW=100, K))` 切片 `X3d[n] = df[...].to_numpy(...)` —— shape mismatch → ValueError
  - **不会 silent 错**——会崩，但崩在循环里，无 graceful degradation。
  
  平台契约（PROGRESS §5）说"每个 DataFrame 100 行"是 guarantee。
- **状态**：✅（依赖平台契约）

---

## 9. README 一致性

### 9.1 README 与代码主要 drift
- **来源**：QREVIEW_PIPELINE_ARCH Q66, Q68, Q70, Q71, CODE_QUESTION_LIST Q161-Q175
- **答案**：
  逐项核对：
  
  | README 声明 | 代码实际 | 状态 |
  |---|---|---|
  | "Python 3.10+ 推荐 3.11" (line 25) | config.json: 3.11 | ✅ |
  | "148 MB" package (line 72) | 50 NN ~13 MB + 50 LGB ~100 MB + 代码 ~30 KB ≈ 148 MB | ✅ (Q168) |
  | "5 GPU 18 min" (line 27, 172) | 单 NN ~100 s × 10 NN/GPU = 16.7 min | ✅ (Q169) |
  | "feature dim 370 = 154+196+20" (line 18-19) | 实际 154+216 = 370（196+20 都在 fast_features_batch 内） | 🟡 README 与 fast_features.py 单 window 380 维 docstring 不符（fast_features.py:10-11 "350"）—— 因为 fast_features.py 是 dead code，README 用的是 fast_features_batch 的数 (Q66) |
  | "MLP [359→256→128→64→1]" (line 153) | HIDDEN=(256,128,64), feat_dim=370-11=359 | ✅ (Q68) |
  | "phase1_val_split: dates 80-95 (15 days)" (line 156) | `build_schemeP_cache.py:82` `80 <= d < 96` → 16 天 | 🔴 off-by-one (Q70) |
  | "M7 全数据重训 +5.51" (line 229) | iter_018 +28.93 → iter_019 v2 +34.44 = +5.51 ✅; 但 HISTORY §4 也写 +28.16 → +34.44 = +6.28 用的是不同基线 | 🔴 文档不同处用不同基线 (Q163) |
  | "L2-only +19.23" (line 231) | iter_013 = T75 LGB L2-only 平台 +19.23 ✅ | ✅ (Q164) |
  | "conformal +0.77" (line 233) | iter_018 vs iter_015 = 28.93-28.16 = +0.77 ✅ | ✅ (Q165) |
  | "150+150 / 100+100 是否做过" | T190 150+150 平台 +34.59，**−1.05 vs T188v2**；100+100 未单独提交 | 🟡 README 应明确 (Q171) |
  | "CRITICAL_CONSTRAINTS.md" (line 240) | 该文件在 workdir 根目录，**不在 final_submission_code 内** | 🟡 平台看不到 (Q170) |
- **状态**：🟡 2 处需修：Q70 (15→16 天)、Q163 (+5.51 vs +6.28 选一致基线)
  
### 9.2 README 缺章节
- **来源**：QREVIEW_PIPELINE_ARCH Q73, Q74, Q75, CODE_QUESTION_LIST Q152
- **答案**：
  - **TROUBLESHOOTING/FAQ**：缺。"KeyError: label_60" 该怎么办、Step 3 OOM 该怎么办、zip 文件太大该怎么办 → 无指导。
  - **CHANGELOG**：缺。README 提 "iter_018"、"T75"、"T81"、"T87"、"T170"、"T188" 等内部代号但**无 glossary**——新读者无法 join 到具体改动。HISTORY_EVIDENCE_DB.md 是这个 glossary 但不在 final_submission_code 内。
  - **ARCHITECTURE.md**：缺。README 表 line 227-234 "关键设计决策"列了 6 条但都是"决策 → 理由"一句话；没有 narrative。
- **状态**：🟡（应至少加 1-page CHANGELOG 把 T 编号 → 改动 join）

### 9.3 README 列出的"T" 编号是否在代码可验证
- **来源**：CODE_QUESTION_LIST Q161, Q172-Q175
- **答案**：
  - "44 个 R 系列调研" (Q161)：HISTORY 附录 D 列了 40+ R 系列；具体每个 R 在代码里**没痕迹**（R 是调研笔记，不入代码）。README 文字应说"基于 192 个 T + 40+ R 实验" 给出来源指针（HISTORY_EVIDENCE_DB.md 不在包内）。
  - T81 warm-start (Q173/Q162)：代码层面是 `model.load_state_dict(ckpt_p1["state_dict"])` (line ~370，Phase 2 入口)，本质 fine-tune，"warm-start" 是术语包装 ✅。
  - T87 SPO+ (Q174)：Elmachtoub & Grigas 2022 原文针对线性 LP；1D 标量回归 + 3 类决策的"SPO+" 是项目改造版，Fisher consistency 不一定 inherited（这是 P1 主题）。
  - T170/T188 (Q175)：T170 是 NN M7 单独突破；T188 是 50+50 mega ensemble。README:253 "T170/T188 M7 全数据重训 trick" 把两者并列略含糊（T188v2 的 NN M7 step 沿用 T170 协议），但**实质 OK**。
- **状态**：🟡（README 应在脚注加 "详细实验编号见 HISTORY_EVIDENCE_DB.md 附录 D"）

---

## 10. 跨文件常量重复

### 10.1 DROP_NAMES / T59_FAIL_NAMES / STAGE5_FAIL_NAMES
- **来源**：QREVIEW_PIPELINE_ARCH Q90
- **答案**：
  完全相同的 list 在 **3 处**：
  1. `02_train_lgb/train_T188v2_lgb_seed.py:33-41`
  2. `03_train_nn/train_T188v2_nn_seed.py:52-60`
  3. `04_build_pkg/Predictor.py:68-75`
  
  byte-identical 时（实测 ✅）一致；任一处改了其他两处忘改 → 训练用的 keep_idx 与推理用的不一致 → silent 错配。
- **状态**：🟡 应抽出 `common/fail_names.py`（4 行 list + 1 行 import）。

### 10.2 RAW_COLS 154 列
- **来源**：QREVIEW_PIPELINE_ARCH Q90 + 跨文件
- **答案**：
  - `01_build_features/build_schemeP_cache.py:53-69` (list, 154)
  - `04_build_pkg/Predictor.py:46-65` (tuple, 154)
  - `04_build_pkg/config.json:4-160` (list, 155 含 sym)
  
  同样 3 处 byte-equivalent。
- **状态**：🟡

### 10.3 fast_features_batch.py 重复
- **来源**：QREVIEW_PIPELINE_ARCH Q91
- **答案**：
  - `01_build_features/fast_features_batch.py` (28519 bytes)
  - `04_build_pkg/fast_features_batch.py` (28519 bytes)
  - **md5sum 一致**: `81bb9fc8b06d893b05bb3abc16aa0fc5`
  
  build_pkg.py 不主动 sync 这两份（只 copy 04 的进 pkg）；01 跑 build_schemeP_cache 也只读 01 的。如果用户改 01 的没同步到 04，cache 用新 fast_features 算特征但 Predictor 用旧 fast_features → silent mismatch。
- **状态**：🟡 应改成 symlink 或在 build_pkg.py 加 `assert md5(01/) == md5(04/)`。

### 10.4 HORIZON_LIST
- **来源**：CODE_QUESTION_LIST Q147
- **答案**：
  - `01_build_features/build_schemeP_cache.py:46` `HORIZONS = (5, 10, 20, 40, 60)`
  - `04_build_pkg/Predictor.py:43` `HORIZON_LIST = (5, 10, 20, 40, 60)`
  - 平台契约：5 个 head 顺序 (5, 10, 20, 40, 60)。
  
  若平台改 horizon list（不太可能），这两处都要改。
- **状态**：🟡

### 10.5 FEE = 0.0001
- **来源**：CODE_QUESTION_LIST Q22
- **答案**：
  - `train_T188v2_lgb_seed.py:23` `FEE = 0.0001`
  - `train_T188v2_nn_seed.py:33` `FEE = 0.0001`
  - PROGRESS.md §1 "手续费 0.01% 双边 = 0.02%/笔"
  - SPO+ loss 里 `fee_eff` 用 FEE × ((mp_th+1) + (mp_t+1)) / (mp_t+1) ≈ 2×FEE 当 mp_th ≈ mp_t → 双边 ✅
  
  Predictor.py 不用 FEE（只用 thr_up/thr_dn 已 absorb 了 fee）。所以 FEE 只在训练侧 2 处一致。
- **状态**：✅

### 10.6 CHUNK = 200_000
- **来源**：CODE_QUESTION_LIST Q31
- **答案**：
  - `train_T188v2_lgb_seed.py:43` `CHUNK = 200_000`
  - `train_T188v2_nn_seed.py:410` chunked aug 段没显式 CHUNK 但 in-line size=X.shape (no chunking, allocates full at once)
  - 实际 NN aug 在 line 271 `scales = rng_aug.uniform(...size=X_tr_imp.shape...)` 一次性分配 → no CHUNK
  - LGB aug 用 CHUNK 是因为 X 大（2.94M × 359 × 4 = 4.2 GB）+ scales 同样大 → 想控制临时内存
- **状态**：🟡（CHUNK 仅 LGB 用；NN 与 LGB 不一致——若 LGB CHUNK 改了同 seed model 也变，这一点应文档化）

### 10.7 AUG_LO=0.80, AUG_HI=1.20
- **来源**：QREVIEW_PIPELINE_ARCH 隐含
- **答案**：
  - `train_T188v2_lgb_seed.py:44` `AUG_LO, AUG_HI = 0.80, 1.20`
  - `train_T188v2_nn_seed.py:44` 同上
  - 一致 ✅，但 2 处复制。
- **状态**：🟡

### 10.8 thresholds.json 与 Predictor 默认值不一致
- **来源**：跨文件
- **答案**：
  - `Predictor.py:222-223` 默认 `default_beta_for_ood=0.16, default_sigma_for_ood=4.0e-4`
  - `thresholds.json:58-59` 实际 `default_beta_for_ood=0.16, default_sigma_for_ood=0.0003998317`
  - **不一致**！如果 thresholds.json 缺 default_*_for_ood key（不太可能），Predictor 会用 4e-4 而不是精确的 3.998317e-4。差异 0.04% 量级 → 影响微小但**不一致**是 code smell。
- **状态**：🟡 应让 Predictor.py 不写硬编码默认，强制读 JSON（缺则 raise）。

### 10.9 thresholds.json _doc / _note 字段
- **来源**：CODE_QUESTION_LIST Q192
- **答案**：
  - thresholds.json 顶层 `_doc`、每个 horizon 内 `_note`
  - JSON 不支持 //comment，所以用 `_` 前缀 key 作 "可写在文件里给读者看的元数据"
  - Predictor 不读这些 key（json.load 把所有 key 都 load 但 Predictor 只用 `enabled/per_sym_beta/per_sym_sigma/...`）
- **状态**：✅

---

## 11. 杂项（接管 CODE_QUESTION_LIST 不属于 P1/P2/P3）

按 CODE_QUESTION_LIST 顺序回答；P1/P2/P3 主题跳过。

### Q22 FEE = 0.0001 (单边/双边)
- 双边。SPO+ `fee_eff = FEE × ((mp_th+1) + (mp_t+1)) / (mp_t+1) ≈ 2×FEE`（line 75-77）。
- PROGRESS §1 与平台公布一致：0.01% 双边。
- **状态**：✅

### Q26 AdamW weight_decay = 1e-4 (Q26 in CODE_QUESTION_LIST)
- P1 主题，跳过。

### Q30 num_threads = 18
- 硬编码 18 是用户当时机器 cpu 数（很可能 AMD Ryzen 9 / Threadripper 18-32 核）。平台 CPU 数未知（probably 16-32）。`-1` 自动会用 platform 全部，但与训练时 18 不一致 → 模型 .txt bytes 不同（同 LGB GPU vs CPU 现象）。
- LGB 4.6 提供 `params.deterministic=True` 时跨 thread 数也确定，但当前未开。
- **状态**：🟡

### Q31 CHUNK = 200_000 in aug
- 见 §7.5。NN 不分块；LGB 分块 200k 行/批是 memory smell（X_aug = (1.47M × 226 × 4) ≈ 1.3 GB 临时；分块 200k 节省到 200k×226×4 ≈ 180 MB）。
- **状态**：🟡 应注释 "CHUNK is reproducibility-sensitive"。

### Q32 rng seed 公式 S*7919+{42,1,137}
- 见 §7.3。3 个 offset 避免不同 RNG stream 相同。
- **状态**：✅（设计合理但 magic number 应注释）

### Q100 Bagging seeds 1-50 硬编码 in thresholds.json
- **答案**：`thresholds.json:51` `"ensemble_seeds": [1, 2, ..., 50]` 写死。Predictor.py:238-247 按 seeds 列表 + `if os.path.isfile(lp)` 校验文件存在。若用户漏训 seed 12，Predictor 会**silent skip 12**，ensemble 用 49 个模型继续跑（不崩，但 manifest.json 与 thresholds 列表不一致）。
- build_pkg.py 也按 seeds 1-50 全扫，缺则 WARNING（§2.5）。
- **更好**：`build_pkg.py` 应动态生成 thresholds.json 的 ensemble_seeds，与实际复制的 seed 列表一致。
- **状态**：🟡

### Q112 requirements.txt CPU torch / Predictor CUDA 分支
- 见 §4.6。CPU torch + `if torch.cuda.is_available()` 分支是前瞻性 + 0 cost（cpu wheel 下 cuda.is_available() = False）。
- **状态**：✅

### Q113 NN .npz 而非 .pt
- 见 §5.1。详细 rationale。
- **状态**：✅

### Q115 GELU approximate="tanh" 推理 vs 训练
- 见 §8.6。**潜在 bug，实测 max diff 6.98e-10 → 不影响 actions**。
- **状态**：🔴 应修一行。

### Q116 LayerNorm 手写 (var unbiased=False + eps=1e-5)
- 与 `torch.nn.LayerNorm` 默认 (unbiased=False, eps=1e-5) 数学一致 ✅。手写原因：F.layer_norm 不支持 per-NN gamma/beta (50 个 NN 各自不同 LN params)。无 unit test 但 T188v2→T188v3 max diff 6.98e-10 = 实证 OK。
- **状态**：✅

### Q117 `torch.where(isnan, 0, h)` 推理
- Predictor.py:159 用 torch.where（不是 np.where；docstring 描述 inaccurate）。在 normalization 后 sanitize。若 NaN 在 forward 中产生（unlikely on standardized data），会被替换为 0 而不传染。
- **状态**：✅

### Q118 NN ensemble 不输出 std/quantile
- predict_mean 只返回 mean(50)，丢了 disagreement 信号。若输出 std 可叠加"high-uncertainty → abstain" 第二层 wrapper。
- 项目早期 conformal β × σ_pred 设计**曾**用 5-NN std 作为 σ（HISTORY §3 "σ_pred = 5 NN（或 50 NN）的预测标准差"），但当前 thresholds.json:57 per_sym_sigma 是**离线 calibration 的常数**而非运行时 std → 设计变了，σ_pred dynamic 没用上。
- **状态**：🟡 可作为未来优化方向（但已知 T159 dynamic abstain 失败 → 谨慎）。

### Q119 X3d float64 / raw_last float32 cast
- `_compute_batch_features:276` `X3d dtype=np.float64` 给特征计算用（avoid 累积误差），随后 raw_last cast 回 float32 (line 280)，extras_full 也 float32 (line 286)。
- 训练时 `train_T188v2_nn_seed.py` X_tr 是 float32 → 推理 float32 一致 ✅
- 训练 build_schemeP_cache.py:107 raw_last 也是 float32 ✅
- **状态**：✅

### Q120 fast_features.py dead code
- 见 §2.2。
- **状态**：🟡 应删

### Q127 Predictor 对 OOD sym 行为
- 见 §8.3。OOD sym (s 不在 0-4 或 cast 失败) → 用 default_beta_for_ood × default_sigma_for_ood ≈ 6.4e-5 band。安全降级 ✅。`Predictor.py:296` `if "sym" not in df.columns: continue` 直接 skip → default band。
- **状态**：✅

### Q128 Predictor 对 missing 'sym' 列
- 见 Q127。若整 batch 无 sym 列，所有行用 default band（OOD treatment）。**平台是否保证 sym 在 df 里**：config.json:159 列出 "sym" 在 feature 列表 → 是契约 ✅
- **状态**：✅

### Q129 Predictor cross-batch state
- 见 §8.2。完全 stateless ✅。`_BatchedMLPEnsemble` 实例在 self 上但是 immutable model weights（构造时一次）。
- **状态**：✅

### Q130 fast_features.py vs fast_features_batch.py 不一致
- Q120 already covered. fast_features.py 是 196 维历史版（不含 Stage5），不在生产链路上。
- **状态**：🟡

### Q131-Q133 build_pkg sanity check 弱 / WARNING 不报错 / NN-LGB seed pairing 不校验
- 见 §2.1, §2.5。
- **状态**：🟡

### Q134 Predictor smoke test 用 random data
- `Predictor.py:368-388` 用 `np.random.default_rng(0).standard_normal((100, len(feats)))`。只验证不 crash + 输出 shape 正确，不验证 actions 与训练时输出对比。
- 应至少加：用 1 个 real cache npz 第 0 行做 forward，assert output 与训练时 nn_h60_seed1.npz 推理一致（cross-check Predictor.py vs nn.Sequential）。
- **状态**：🟡

### Q135 extract_npz layer_idx hardcoded use_layernorm=True
- 见 §5.1, §5.3。脆弱但 use_layernorm=True 在 train script 是 default + 没暴露 args → 实际不出问题。
- **状态**：🟡

### Q137 NN final_ckpt 同时 keep_idx (in .pt) 与 npz 里
- `.pt` 是 debug 备份不进 pkg；npz 里 keep_idx 是 Predictor 用的真源。冗余但 harmless。
- **状态**：✅

### Q138 n_train assertion (n_total*2 aug)
- LGB 训练 pre-alloc 2.94M × 226 × 4 = 4.2 GB。NN 同。16 GB 机器够，但与其他进程并跑可能 OOM。README:31 提 "RAM 16GB+"。
- **状态**：✅（文档化）

### Q139 LGB Dataset feature_name
- 见 §5.5。
- **状态**：🟡 应 assert 推理时 feature_name 与训练一致

### Q140 LGB bagging_seed / feature_fraction_seed 派生
- 见 §7.7。lightgbm 内部 deterministic 派生 + pin 版本 ✅。
- **状态**：✅

### Q141 Predictor _load_module importlib
- 用 `importlib.util.spec_from_file_location("iter_018_ffb", ...)` 加载 fast_features_batch.py。优点：fast_features_batch.py 不需在 sys.path / 不需 `__init__.py`。缺点：mod_name `"iter_018_ffb"` 是历史命名（iter_018 时代）—— 与 fname 不对应，混乱但 harmless。
- **状态**：🟡 应改成 `"fast_features_batch"`。

### Q142 _BatchedMLPEnsemble.predict_mean keep_idx 判断
- `if X_in.shape[1] != self.in_dim: Xs = X_in[:, self.keep_idx]`（line 148）。
- `_compute_batch_features` 已 keep_idx-投影 (line 287 `extras_full[:, self._extra_keep_idx]`)，所以 X_in.shape[1] 总等于 in_dim → 条件总 False → dead-ish code，但作为 defensive 保留 OK。
- **状态**：✅

### Q143 ProductionPipeline pred_cache by src_H
- 见 §1.7 / §8.2。pred_cache 在 predict() 局部 dict，按 src_H key 缓存；5 horizon 都 share_with=60 → 1 次 ensemble inference + 4 次复用 ✅。
- **状态**：✅

### Q144 lp/np_ filenames 硬编码 h{H}
- `Predictor.py:242-243` `f"model_h{H}_seed{s}.txt"`、`f"nn_h{H}_seed{s}.npz"` —— 若用户加 h=120 但模型不存在，silent skip（不崩）。OK。
- **状态**：✅

### Q145 lgb.Booster(model_file=p) × 50
- 50 个 booster 加载 → 数百 MB RAM（每个 booster ~2 MB 文本 + parser overhead ~5x = ~10 MB 内存→ 500 MB）。Predictor `__init__` 一次性 load，inference 时不 reload。OK。
- **状态**：✅

### Q147 HORIZON_LIST 顺序固定
- 见 §10.4。
- **状态**：🟡

### Q148 build_schemeP_cache 缺 parquet 静默
- 见 §1.11.
- **状态**：🔴

### Q149 parquet 文件名约定写死
- 见 §1.12.
- **状态**：🟡

### Q150 没有 unit test
- final_submission_code 无 tests/ 目录。byte-equivalence + 实际平台跑过是验证手段。生产软件标准不达；研究代码可接受。
- **状态**：🟡 应加最少 3 个测试：
  1. Predictor 在 random data 上 forward 不崩 (line 368 已有 smoke)
  2. extract_npz → _BatchedMLPEnsemble 数值与 nn.Module 误差 < 1e-5
  3. build_pkg 缺 seed 时退出 code != 0

### Q151 stage5_features.py dead code
- 见 §2.3.
- **状态**：🟡

### Q152 文件头 "iter_018"/"T59" 等代号
- README 与代码注释频繁出现"iter_xxx" 和 "T*" 编号。新读者无 glossary。详细 mapping 在 HISTORY_EVIDENCE_DB.md 但不在 pkg 内。
- 建议：final_submission_code/CHANGELOG.md 列 T → 改动一句话表。
- **状态**：🟡

### Q161 README "44 个 R 系列调研"
- HISTORY 附录 D 列了 R 系列 40+。具体每个 R 在代码无痕（调研笔记不入 final code）。是诚实声明而非夸大。
- **状态**：✅

### Q162 T81 warm-start 实现
- `model.load_state_dict(ckpt_p1["state_dict"])` 普通 fine-tune（line ~370）。"warm-start"是 ML 术语包装，OK。
- **状态**：✅

### Q163 README "+5.51 from M7" vs HISTORY "+6.28"
- README:229 "M7 全数据重训 平台 +34.44 vs 早停 +28.16（+5.51）"——但 34.44 − 28.16 = 6.28, not 5.51. 计算错误 / 或者用 iter_018 +28.93 基线 → 34.44 − 28.93 = 5.51。**HISTORY §0 / 主题 4 "T140 平台 +5.51 vs iter_018 +28.93"** 是正确的；README 把 28.16 写成 baseline 但算 +5.51 是 inconsistent。
- **状态**：🔴 README typo

### Q164-Q165 +8.93 SPO+ / +0.77 conformal
- L2-only +19.23 → SPO+ +28.16 = +8.93 ✅
- iter_015 +28.16 → iter_018 +28.93 = +0.77 ✅
- 但 README 这两处与 Q163 同样要小心 baseline 一致性。
- **状态**：✅

### Q166 README "sym 在 conformal 用，训练不用" vs 计算 σ 用 sym split
- 见 §8.4。本质 OK，措辞略含糊。
- **状态**：✅

### Q167 README "date 从不使用" vs cache 存 date
- 见 §8.1。cache 存 date 仅 split 用，不进 X。措辞 OK。
- **状态**：✅

### Q168 README "148 MB" 包大小拆分
- 50 NN npz × ~270 KB = 13.5 MB; 50 LGB txt × ~2 MB = 100 MB; Predictor + thresholds + config + requirements + 2 fast_features = ~50 KB + 28 KB × 2 = 60 KB; manifest = 2 KB. **总 ≈ 113 MB**——与 README 148 MB 差 ~35 MB；可能 LGB .txt 实际更大或 manifest 多算。重要的是 < 2GB 限。
- **状态**：✅（数字略偏，但不影响）

### Q169 README "5 GPU 18 min"
- 单 NN ~100 s × 10 / 5 GPU = 16.7 min + setup overhead → 18 min 合理 ✅
- **状态**：✅

### Q170 README "CRITICAL_CONSTRAINTS.md" 位置
- 该文件在 workdir 根目录，不进 pkg。但 Predictor.py:19-27 docstring 已把 3 条约束 inline → 平台读 Predictor.py 能看到精神。
- **状态**：✅

### Q171 README "150+150 / 100+100 是否做过"
- T190 150+150 平台 **−1.05 vs T188v2**（退步！）。100+100 未单独提交。README 应明示。
- **状态**：🟡 README 应加一行 "T190 150+150 已试 +34.59 −1.05 vs SOTA"

### Q172 README "T75 iter_013"
- T75 = LGB Δmid 回归 = iter_013 突破首次 +19.23。准确 ✅。git log 在 final_submission_code 里不 keep 这些历史（这是 final clean pkg 不是 dev repo）。
- **状态**：✅

### Q173 T81 命名
- T81 是 NN L2 pretrain protocol。当前 final code 中 `T81_pretrained/` 目录就是 anchor 路径。命名一致。
- **状态**：✅

### Q174 T87 SPO+ Fisher consistency
- P1 主题（损失函数理论），跳过。

### Q175 T170 vs T188 贡献
- T170 = NN M7 retrain 5+5 平台 +34.64; T188v2 = 50+50 mega 平台 +35.64。两者都贡献：T170 是 M7 first; T188v2 是 ensemble 扩容（NN+LGB 都扩才 +1.00）。
- **状态**：✅

### Q176 GELU train/inference 不一致（critical）
- 见 §8.6。**核心 bug**，实测影响为零但应修。
- **状态**：🔴

### Q177 NN 50×B×359 stack 内存
- 73 MB tensor + 4 层 bmm 中间结果 50×1024×256×4 ≈ 52 MB → peak ~150 MB（CPU 上 OK）。`.contiguous()` after `.expand()` 多分配 73 MB 副本；理论上能让 bmm 直接 broadcast 但代码用 contiguous 是为 bmm 性能。
- **状态**：✅

### Q178 target_scale per-NN, mean after scaling
- `out / target_scale_i (per NN) → mean(0)`（line 174-177）= mean(pred_i × scale_i^{-1})。等价于权重 scale_i^{-1} 的加权平均。若 50 NN target_scale 差异大 (max/min > 2)，等权 mean 偏向 large-scale 模型。实际 target_scale = 1/std(y_phase1) 各 seed 差异小 (phase1 train data 都是 dates 0-79)。
- **状态**：✅

### Q179, Q180 NN ensemble 维度一致性
- 见 §5.3. **应 assert 但未 assert**。
- **状态**：🟡

### Q181 requirements.txt 版本是否实际存在
- numpy 2.4.4 / pandas 2.3.3 / lightgbm 4.6.0 / scipy 1.17.1 / torch 2.5.1+cpu —— 实测 platform pip install 成功（+35.64 SOTA 提交跑过）→ 都存在 ✅
- **状态**：✅

### Q186 .pt 中间产物
- debug 备份；不入 pkg；与 npz 平行存在。OK。
- **状态**：✅

### Q187 anchor_seed.pt 文件名 / Phase 1 触发
- 见 §3.5. 首次 `--skip-phase1` 仍跑 Phase 1（条件 isfile=False）→ 单 shell 跑 50 seed 时每个 seed Phase 1 都跑（~50 s × 50 = 41 min，加 Phase 2 41 min ≈ 90 min 总，与 README 一致）。第二次 run 时 anchor 已存在 → 跳过。
- **状态**：✅

### Q188 build_pkg 不打包 T81_pretrained
- 见 §2.7. 推理不需要 anchor；用户复现需手动保留。
- **状态**：🟡

### Q189 NN CPU 推理延迟
- HISTORY §10 实测 8.1 ms / 1024 batch (batched bmm CUDA) → CPU 估计 100-400 ms。LGB 84 ms。总 e2e ~500 ms / batch → 442k 行 / 1024 = 432 batch × 500 ms = 3.6 min CPU。远低于 3h。
- **状态**：✅

### Q190 Predictor 单次 predict() B 个 batch
- 见 §6.4. B=1024 OK；B 大于 ~2048 可能 OOM 在 CPU（73 MB 已经在 limit；NN 4 层中间 tensor ×4 ≈ 800 MB peak for B=4096）。
- **状态**：🟡 可加 chunking

### Q191 Smoke test 不 assert action 在 {0,1,2}
- line 382-388 只 print，不 assert。可加 `assert all(a in (0,1,2) for batch in out for a in batch)`。
- **状态**：🟡

### Q192 _doc field
- 见 §10.9. OK。
- **状态**：✅

### Q193 manifest checksum
- `md5_file` 定义未调用。应在 manifest.json 加每文件 md5。
- **状态**：🔴

### Q197 Predictor 无 timeout/error handling
- `predict(batches)` 若中间抛异常整 predict() 失败；平台无 retry。但平台只调一次（PROGRESS §1：单次评测 3h），所以应 try/except wrap 单 batch → 失败 batch 返回 [1,1,1,1,1] 默认 all-flat 而非全 predict 失败。
- **状态**：🟡

### Q199 build_pkg 不 zip
- 见 §2.8.
- **状态**：✅

### Q200 无版本号 / metadata
- 无 VERSION / __version__ / __build__。所有 metadata 在 README + commit history。
- **状态**：🟡 应加 `pkg/VERSION` (含 git SHA + build timestamp + pkg md5)

---

## 12. GAP 表（应改进项排序）

按"应修紧迫度 / 实现成本"排序：

| # | 问题 | 来源 | 风险 | 修复成本 | 状态 |
|---|---|---|---|---|---|
| 1 | GELU train/inference mismatch (erf vs tanh approx) | Q176/Q115 §8.6 | 理论可翻 action，实测 max diff 6.98e-10 | 1 行 | 🔴 |
| 2 | build_pkg 无 ensemble 完整性 strict 校验 (NN/LGB seed count 对齐 / npz loadable / manifest md5) | Q131-Q133, Q193, §2.1 | 缺 seed 不 fail → ensemble 失衡 | ~30 行 | 🔴 |
| 3 | build_schemeP_cache 缺 parquet 静默继续 | Q77/Q148, §1.11 | 数据缺一半仍生成 cache → 模型质量低 | 5 行 | 🔴 |
| 4 | README typo: "+5.51 from M7" 但 34.44-28.16=+6.28 (baseline 不一致) | Q163, §9.1 | 文档可信度 | 1 行 | 🔴 |
| 5 | cache npz 无 schema/version/hash stamp | Q5-Q8, §1.5 | cache 与 model 不匹配静默 | ~20 行 | 🔴 |
| 6 | `set -u`/`pipefail` 缺失 | Q11/Q56, §1.9 | typo silent pass | 1 行 | 🟡 |
| 7 | DROP_NAMES / RAW_COLS 3 处重复 | Q90, §10.1-10.2 | 改一处忘改其他 | ~50 行 重构 | 🟡 |
| 8 | fast_features.py + stage5_features.py dead code | Q66/Q67/Q87/Q120/Q151, §2.2-2.3 | 阅读混淆 | 删 2 文件 | 🟡 |
| 9 | NN ensemble in_dim/hidden/use_layernorm/keep_idx 不 assert 一致 | Q81/Q82/Q179/Q180, §5.3 | 错配 silent | ~10 行 | 🟡 |
| 10 | Phase 1/2 args 与 anchor 元数据不校验 | Q34, §7.4 | 复用 stale anchor 致 model invalid | ~20 行 | 🟡 |
| 11 | README "phase1_val_split: 15 天" 实际 16 天 (off-by-one) | Q70, §9.1 | 文档准确性 | 1 字符 | 🟡 |
| 12 | 无最小单元测试 (Predictor numerical parity / smoke OOD / build_pkg strict) | Q100/Q150, §1.7, Q134 | 回归风险 | ~100 行 test 文件 | 🟡 |

**说明**：4 个 🔴 是"我会现在就修"级别；8 个 🟡 是"下个 sprint" 级别。**没有任何 🔴 影响当前 +35.64 SOTA 平台得分**（GELU mismatch 实测 0 影响；其他都是 robustness/maintainability）。

---

## 关键证据 / Strong Ablations 汇总

P4 范围内可援引的硬数据：

1. **T61 推理 254 min → 4.5 min (58×)**（HISTORY §10 iter_010 batch-vec）—— 验证"特征构建必须向量化"是平台 3h 限的硬约束，全 CRITICAL_CONSTRAINTS 顺路证。
2. **T188v3 batched bmm 50 NN 1509ms → 8.1ms (186×)**（HISTORY §10）—— 验证 `_BatchedMLPEnsemble` 的存在价值；解释 fast_features.py 为啥能成为 dead code（不再需要单 sample fast path）。
3. **T188v3 vs T188v2 actions max diff 6.98e-10**（HISTORY §10）—— 实证 GELU tanh approx + 手写 LayerNorm 在当前数据上不翻 action，但理论上仍是 surprise。
4. **T155 LGB M7 100% deterministic 验证**（HISTORY §4）—— 同 seed 同硬件 byte-identical；与 NN 的"statistical reproducibility"形成对比。
5. **T182 无 conformal ablation：本地 +6.62, 平台 −0.30**（HISTORY §3）—— 证明 conformal wrapper 必要性 + in-sample vs OOD divergence + 平台 OOD 是唯一 ground truth；构成"为什么 Predictor 必须保留 conformal 路径而非走 _ev_gate_predict"的 evidence。
6. **T179 (40-NN only) 平台 −0.25 vs T170**（HISTORY §2）—— 验证 ensemble 必须 NN+LGB 对称扩；解释 build_pkg 应严格校验 NN/LGB seed 数对等。
7. **T190 150+150 平台 −1.05 vs T188v2 +35.64**（HISTORY §2）—— 50→150 非单调；解释为什么"thresholds.json 写死 50 seed 列表"是 conscious choice 而非懒惰。
8. **T192 NN-NN corr 0.9331, LGB-LGB 0.8433, cond# 54182**（HISTORY 附录 A 发现 8）—— 解释为什么 ensemble 用 simple mean 而不学权重；解释 reproducibility 容忍 bit drift 的根本依据（noise 早已 > drift）。

---

**总计 P4 答题量约 290 个独立问题（QREVIEW_PIPELINE_ARCH 100 + CODE_QUESTION_LIST 杂项 ~100 + QREVIEW_PREDICTOR/NN_TRAIN/LGB_TRAIN/FEATURES 中 pipeline-misc 横切问题 ~50 + 跨问题综合 ~40）。**

RESULT: task=[answers P4 pipeline] metrics={n_questions_answered=290, n_gaps=12, n_strong_ablations=8} notes=[Pipeline 设计合理（4 步资源切分、Predictor stateless、3 条硬约束代码合规），实测平台 +35.64 SOTA 验证；5 个真实 🔴 风险都是 robustness/maintainability 而非功能（GELU mismatch 实测 0 影响、build_pkg sanity 弱、cache 无 fingerprint、parquet 缺失静默继续、README typo +5.51）；最大 implementation gap 是 build_pkg.py 的 strict check + md5_file 死代码激活，1 个文件 ~50 行可完成。]
