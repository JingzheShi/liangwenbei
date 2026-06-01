# NN 训练脚本疑问清单

脚本是一个两阶段训练流程：Phase 1 在 train(0-79)/val(80-95) 上做 L2 回归 pretrain（AdamW + CosineLR + early stop，HP 通过 (seed-1)%k cycling 跨三个轴变化），Phase 2 在 train+val+test 合并的全数据上做 SPO+ DFL fine-tune 11 个 epoch（lr=3e-5、λ=30 固定）。输出两个 checkpoint（pt 与 npz），其中 npz 是手工逐层抠出的权重以供推理端 numpy load。

下面按主题分组列出 100+ 处会让我想问"为什么"的具体设计点。

---

## A. 顶层常量与超参（行 32–50）

## Q1: NUM_CLASS=3 为什么是 3？
- 位置: 行 32
- 代码: `NUM_CLASS = 3`
- 问题: 三分类是题目给定的标签语义吗（涨/平/跌）？为什么模型却用了回归头（输出 1 维），却仍要构造类别权重？是否对照过 5/7 类的离散分桶？

## Q2: FEE=0.0001 这个值从哪里来？
- 位置: 行 33
- 代码: `FEE = 0.0001`
- 问题: 这是平台给的手续费率还是经验拟合？做过 FEE 取 5e-5、2e-4 的 sensitivity 吗？它直接进入 SPO+ 的死区阈值——如果设错，z* 全为 0，loss 退化。

## Q3: HIDDEN=(256,128,64) 为什么是金字塔型 3 层？
- 位置: 行 34
- 代码: `HIDDEN = (256, 128, 64)`
- 问题: 为什么不是 (512,256,128) 或 (128,128,128) 或 wider/shallower？有 sweep depth/width 的曲线吗？相对输入维度（约 220）只压缩一次到 256 算"宽"吗？

## Q4: CLIP=10.0 标准化后的硬截断阈值
- 位置: 行 36
- 代码: `CLIP = 10.0`
- 问题: 为什么取 10σ 而不是 5、6、8？做过 clip 范围对 val_mse 的实验吗？特征本身做过分布尾部分析吗（多少比例样本会被 clip）？

## Q5: LR_LIST 只有 3 个离散点
- 位置: 行 37
- 代码: `LR_LIST = [1e-4, 3e-4, 1e-3]`
- 问题: 为什么只选 ½-decade 步长这 3 个？跨度 10x 但中间没有 5e-4、5e-5 等点。是先单 seed 扫了然后取附近三个吗？

## Q6: DROPOUT_LIST 选 [0.05,0.10,0.15,0.20]
- 位置: 行 38
- 代码: `DROPOUT_LIST = [0.05, 0.10, 0.15, 0.20]`
- 问题: 为什么排除 0（无 dropout）和 0.30+？步长 0.05 是凭感觉吗？做过单独 dropout sweep 的曲线吗？

## Q7: BATCH_LIST 只 3 个值且全 ≥2048
- 位置: 行 39
- 代码: `BATCH_LIST = [2048, 4096, 8192]`
- 问题: 为什么不试 512/1024 的小 batch（小 batch 通常有 implicit regularization）？8192 在哪种 GPU 上跑过的，会不会显存爆？

## Q8: HP cycling 用 (seed-1)%len 是否会让 LR/dropout/batch 强相关？
- 位置: 行 186-188
- 代码: `lr_p1 = LR_LIST[(S-1)%3]; dropout = DROPOUT_LIST[(S-1)%4]; batch_p1 = BATCH_LIST[(S-1)%3]`
- 问题: lr 和 batch 都 mod 3 → 配对永远相同（seed 1→lr=1e-4 & batch=2048）；dropout 是 mod 4，与前者会形成 LCM=12 的 cycle。这是 grid sweep 而不是均匀采样吗？50 个 seed 里只有 12 种独立组合，剩下 38 个是重复 + 不同 init/aug 噪声——这种结构是有意吗？

## Q9: PHASE1_EPOCHS=50 上限
- 位置: 行 41
- 代码: `PHASE1_EPOCHS = 50`
- 问题: 50 怎么定的？有没有看过 epoch=100 时 val 是否还能继续下降？Cosine T_max=50 让 LR 在 50 ep 时归到 eta_min，若 patience 在 ep 15 就触发，剩下的余弦未用，schedule 失效。

## Q10: PHASE1_PATIENCE=10 太大还是太小？
- 位置: 行 42
- 代码: `PHASE1_PATIENCE = 10`
- 问题: 1/5 的总 epoch 数做 patience，意味着 noise 大的 seed 几乎永远不会早停。对比过 patience=5 / 15 吗？

## Q11: PHASE1_WD=1e-4 weight decay
- 位置: 行 43
- 代码: `PHASE1_WD = 1e-4`
- 问题: AdamW 默认是 0.01；这里 100x 更小。是因为已经有 dropout + early stop 所以削弱 WD？做过 WD ∈ {0, 1e-5, 1e-4, 1e-3} 的对比吗？

## Q12: AUG_LO=0.80, AUG_HI=1.20 不对称 multiplicative noise
- 位置: 行 44
- 代码: `AUG_LO, AUG_HI = 0.80, 1.20`
- 问题: ±20% multiplicative noise；为什么这个幅度？log-uniform 不是更对称（exp(±0.2)≈[0.82,1.22]，几乎相同）？对**负值特征**乘 1.2 会让其更负——这是有意的吗？

## Q13: PHASE2_EPOCHS=11 这个奇数
- 位置: 行 46
- 代码: `PHASE2_EPOCHS = 11`
- 问题: 为什么是 11 而不是 10 或 12？是早期某次实验的 best_epoch 被硬编码了吗？fine-tune 没有 val 监控，11 是怎么选定的？

## Q14: PHASE2_LR=3e-5 与 Phase1 lr 关系
- 位置: 行 47
- 代码: `PHASE2_LR = 3e-5`
- 问题: 相对于 Phase1 的 [1e-4, 3e-4, 1e-3] 是约 3-33x 小。为什么是 3e-5 而不是 phase1_lr/10？为什么所有 seed 都用同一个 lr，不再 cycling？

## Q15: PHASE2_LAMBDA_SPO=30.0 这个 magnitude
- 位置: 行 48
- 代码: `PHASE2_LAMBDA_SPO = 30.0`
- 问题: L2 和 SPO+ 数量级不同（L2 是 squared scale，SPO+ 是线性 scale），30 的权重比是怎么定的？SPO+ 已被 weight 主导后，L2 还能起多大锚定作用？做过 λ ∈ {1, 10, 30, 100} sweep 吗？

## Q16: PHASE2_BATCH=4096 为什么不再 cycle？
- 位置: 行 50
- 代码: `PHASE2_BATCH = 4096`
- 问题: Phase 1 batch 三选一但 Phase 2 固定 4096——是因为 SPO+ loss 需要"足够多样本"才稳定？还是仅仅怕显存？

## Q17: PHASE2_WD=1e-4 与 phase1 同
- 位置: 行 49
- 代码: `PHASE2_WD = 1e-4`
- 问题: fine-tune 阶段是否需要更小的 WD（已经 warm-started）？固定与 phase1 同值的依据？

---

## B. 被丢弃的特征列表（行 52–60）

## Q18: 为什么丢掉 T59_FAIL_NAMES 里这 10 个特征？
- 位置: 行 52-58
- 代码: `T59_FAIL_NAMES = ["dualz_ask_diff1", ...]`
- 问题: "FAIL" 是指什么——sym-agnostic verification 不过？数值不稳？低重要性？每个被剔的特征单独验过吗？

## Q19: STAGE5_FAIL_NAMES 只剩 1 个
- 位置: 行 59
- 代码: `STAGE5_FAIL_NAMES = ["liq_asym_top5_W5"]`
- 问题: 为什么这一个被单独归类，与 T59 列表分开？是不同阶段发现的吗？日后若新加发现的 fail，是否需要重训整个模型？

## Q20: 丢特征 vs masked/regularized
- 位置: 行 60
- 代码: `DROP_NAMES = T59_FAIL_NAMES + STAGE5_FAIL_NAMES`
- 问题: 直接 drop 是否过度——这些特征在某些样本/sym 仍可能有信号。试过保留但加 group-wise dropout 吗？

---

## C. 损失辅助函数（行 63–77）

## Q21: class_balanced_weight 的公式选择
- 位置: 行 63-67
- 代码: `cw = len(y) / (num_class * counts)`
- 问题: 这是 "inverse frequency × N/K"。为什么不用 effective number of samples（Cui 2019）或简单 1/freq？做过 weight 形式对比吗？

## Q22: bincount 在空类别时被替换成 1
- 位置: 行 65
- 代码: `counts = np.where(counts == 0, 1.0, counts)`
- 问题: 如果某类真不存在，给权重无穷大会出问题；但替换成 1 又把它压得极轻（len/3/1 = ~N/3，但乘 0 样本=0）。这逻辑只是防 ÷0 还是有别的考虑？

## Q23: 类别权重用于回归 MSE 是不是 mismatch？
- 位置: 行 315, 460
- 代码: `mse = ((pred - yb) ** 2 * wb).sum() / wb.sum().clamp_min(1.0)`
- 问题: y_regr 是连续的，y_cls 用来分桶后产 sample weight；这等价于给"中性"类的样本权重较低（如果数据偏向 0 类）。试过 unweighted MSE 吗？

## Q24: regr_target 用 (mp_th - mp_t)/(mp_t+1) 而不是 log return
- 位置: 行 70-72
- 代码: `((mp_th - mp_t) / (mp_t + 1.0))`
- 问题: 为什么 +1？防止 mp_t≈0 的除零？mp_t 是什么尺度（如果已是几百量级，+1 几乎无影响）？为什么不直接 log(mp_th/mp_t)？

## Q25: regr_target 强制 float64 中间再转 float32
- 位置: 行 71-72
- 代码: `mp_th.astype(np.float64) - ...`
- 问题: 是否担心 float32 精度不够（mp_t±0.01 的小差）？所有 mp_t 都验证过 +1 偏移是合理的吗？

## Q26: fee_eff 的公式
- 位置: 行 75-77
- 代码: `FEE * ((mp_th+1) + (mp_t+1)) / (mp_t+1)`
- 问题: 推导出来是 FEE*(mp_th+mp_t+2)/(mp_t+1)，约 ≈2*FEE 当 mp_th≈mp_t≈0；这是双边手续费的简化吗？精确公式来源？

## Q27: fee_eff 和 regr_target 应保持单位一致
- 位置: 行 70-77
- 代码: 两个函数都除以 (mp_t+1)
- 问题: 都是"相对 mp_t+1"的尺度——这个一致性是文档化的设计吗？后面 SPO+ 里 fee_scaled 与 y_scaled 用同一 target_scale，依赖单位一致。

---

## D. MLP 架构（行 80–104）

## Q28: 默认 dropout=0.10 但被 cycling 覆盖
- 位置: 行 81
- 代码: `def __init__(self, in_dim, hidden=HIDDEN, dropout=0.10, ...)`
- 问题: 函数签名默认 0.10，但实际调用从 DROPOUT_LIST 取——为什么留个未必触发的默认值？

## Q29: LayerNorm 还是 BatchNorm 还是没有
- 位置: 行 87-88
- 代码: `if use_layernorm: layers.append(nn.LayerNorm(h))`
- 问题: 在小 batch=2048 上 BN 也算稳，为什么选 LN？做过 BN/LN/无 norm 对比吗？LN 用默认 eps=1e-5 是否合适？

## Q30: 顺序 Linear→LN→GELU→Dropout
- 位置: 行 86-90
- 代码: `Linear; LN; GELU; Dropout`
- 问题: 对比过 Linear→GELU→LN→Dropout 或 LN→Linear→GELU 吗？norm-before vs norm-after 在 MLP 上影响不大但仍是设计选择。

## Q31: GELU 没有指定 approximate
- 位置: 行 89
- 代码: `layers.append(nn.GELU())`
- 问题: 默认 approximate='none' 用 erf；要不要用 tanh 近似（更快）？做过 ReLU/SiLU/GELU 比较吗？

## Q32: Kaiming init 用 nonlinearity="relu" 但激活是 GELU
- 位置: 行 99
- 代码: `nn.init.kaiming_normal_(m.weight, nonlinearity="relu")`
- 问题: gain mismatch——GELU 的 gain ≈ relu 的 √2/某常数。这虽 typical practice，是否值得用 `nonlinearity="leaky_relu", a=0` 或自定 gain？

## Q33: bias 全部 zero init
- 位置: 行 100-101
- 代码: `nn.init.zeros_(m.bias)`
- 问题: 标准做法，但最后一层做回归 bias 是否应该 init 到 target mean？

## Q34: 最后一层无 LN/activation/dropout
- 位置: 行 92
- 代码: `layers.append(nn.Linear(d, 1))`
- 问题: regression head 通常这样，但有没有试过加 LN 在 64→1 之前帮助 scaling？

## Q35: forward 用 squeeze(-1)
- 位置: 行 104
- 代码: `return self.net(x).squeeze(-1)`
- 问题: 如果 batch=1 会把 batch 维也 squeeze 掉——这种 edge case 在 phase 2 的 batch 末尾会发生吗（n_train2 % 4096 == 1）？

## Q36: 没有残差/skip 连接
- 位置: 行 80-104
- 代码: 整个 MLPRegr 类
- 问题: 三层金字塔 (256→128→64) 是否考虑过 residual block？小模型常见 baseline，做过对比吗？

## Q37: 模型只有 1 个 head（回归）但用了 class weights
- 位置: 行 92, 315
- 代码: 单 head + sample weight
- 问题: 试过 multi-task（class head + regression head）吗？为什么放弃多 head？

---

## E. 推理函数（行 107–114）

## Q38: predict_chunked 默认 batch=16384
- 位置: 行 107
- 代码: `def predict_chunked(model, X_std_tensor, batch=16384, device="cuda"):`
- 问题: 为什么 16384 而不是 4096 或 32768？基于哪种 GPU 显存预算？这里 X_std_tensor 已经在 CPU（CPU tensor），每 batch .to(device) 是不是浪费 PCIE 带宽？

## Q39: out 用 np.zeros 预分配，但顺序写入可能有 issue
- 位置: 行 109-113
- 代码: `out = np.zeros(...); out[s:s+batch] = model(xb).detach().cpu().numpy()`
- 问题: detach() 已在 no_grad 下，是否多余？.cpu() 之后 numpy() 是否会引入 sync 阻塞导致吞吐降低？

## Q40: predict_chunked 用 .detach() 但是又在 no_grad 里
- 位置: 行 110, 113
- 代码: `with torch.no_grad():` 内 `.detach()`
- 问题: 冗余。是 leftover 还是有理由？

---

## F. SPO+ 损失（行 117–127）

## Q41: z_star 用阈值 ±FEE 而不是 sign(y)
- 位置: 行 118-122
- 代码: `z_star = where(y>fee, 1, where(y<-fee, -1, 0))`
- 问题: 这是 SPO+ 论文里的 "true optimal action"？为什么不在 |y|<fee 时取 0 而是改用 sign(y)？这种"死区"会让一大块样本 z_star=0，loss 退化成 relu(|spread|-fee)，无明确方向梯度。

## Q42: spread = 2*pred - y
- 位置: 行 124
- 代码: `spread = 2.0 * pred_scaled - y_scaled`
- 问题: 这是 SPO+ 的标准 "2 cˆ - c" 形式（来自 Elmachtoub-Grigas 2022）。为什么乘 2 在这里依然写成 2.0*pred-y 而不写 `pred + (pred - y)`？数学等价但可读性差。

## Q43: relu_max = relu(|spread| - fee)
- 位置: 行 125
- 代码: `relu_max = F.relu(torch.abs(spread) - fee_scaled)`
- 问题: 这取自 SPO+ for "interval" action space（[-1,0,1] 决策）的封闭形式吗？非线性绝对值在 0 处不可导，pytorch 自动赋 0 梯度——这会不会让训练在死区卡住？

## Q44: ell = relu_max - z*spread + fee*|z|
- 位置: 行 126
- 代码: `ell = relu_max - z_star * spread + fee_scaled * abs_z_star`
- 问题: 最后 `+ fee*|z|` 在 z=0 时为 0，z=±1 时为 fee——这是补偿"如果 z 是 sell/buy 的实际收益"的常数项吗？为什么这种形式恰好等价于 SPO+ ξ-surrogate？

## Q45: SPO+ 返回 per-sample loss（不 mean）
- 位置: 行 127
- 代码: `return ell`
- 问题: 调用方 (行 459-461) 自己做加权 mean，OK；但若有人改成默认 mean=True，会和现有 caller 不兼容——为什么不做参数控制？

## Q46: SPO+ loss 没有数值稳定性检查
- 位置: 行 117-127
- 代码: 整个函数
- 问题: pred、y 是 scaled 后的数（×target_scale），但若某 batch 出现极端值 pred→1e3，relu(...) 也跟着爆。有 sanity check 吗？

## Q47: SPO+ 梯度有否被显式推导验证
- 位置: 行 117-127
- 代码: 直接 autograd
- 问题: 论文里有解析梯度 -2*(z_star - sign(spread)*1[|spread|>fee]) 之类的，autograd 与之是否一致？做过 gradient check 吗？

## Q48: fee 是否应该在 scaled 空间还是原空间用
- 位置: 行 422, 459
- 代码: `fee_scaled_full = (fee_full * target_scale)`
- 问题: y * target_scale 是无量纲化，但 fee 是同单位的——同时 scale 是对的。但 SPO+ 在 scaled 空间 vs raw 空间会给完全不同的 loss landscape，做过对比吗？

## Q49: 没用 reduction='none' 显式注释
- 位置: 行 127
- 代码: `return ell`
- 问题: 没有 docstring 说明返回形状是 (B,)；下游 caller 用 .sum() 隐式假设——重构时容易出 bug。

---

## G. extract_npz 权重导出（行 130–160）

## Q50: 为什么单独存 npz 而非 torch state_dict
- 位置: 行 130
- 代码: 整个函数
- 问题: 推理端需要 numpy？平台限制 torch 版本？如果是，为什么不直接 `np.savez` torch tensor 转出来——而是手工逐层抠？

## Q51: layer_idx += 2 假定 GELU+Dropout 各占 1 个 module
- 位置: 行 156
- 代码: `layer_idx += 2  # GELU + Dropout`
- 问题: 如果将来改 activation 或加 BatchNorm，这个 magic +2 会错位。为什么不按 named children 迭代？

## Q52: use_layernorm 假定 hidden 每一层都用 LN
- 位置: 行 152-155
- 代码: `if use_layernorm: ...`
- 问题: 不允许"只在部分层用 LN"——架构搜索时会限制空间。是有意的吗？

## Q53: target_scale / clip 同时存为长度 1 的 ndarray
- 位置: 行 141, 145
- 代码: `np.array([target_scale], dtype=np.float32)`
- 问题: 为什么不存 scalar？是 npz 要求？读取端 [0] 索引看起来怪。

---

## H. main() 入口与参数（行 163–197）

## Q54: --seed 是必需的且为整数 index 1..50
- 位置: 行 165
- 代码: `ap.add_argument("--seed", type=int, required=True, help="Seed index 1-50")`
- 问题: 用 (S-1) 做 cycling 意味着 seed=0 不会被支持但也不报错——是否应该 assert 1 ≤ S ≤ 50？为什么 50 上限？

## Q55: --horizon 默认 60 但文件名仍 hardcode "h60"
- 位置: 行 166, 212-213
- 代码: `f"nn_h60_seed{S}.pt"`
- 问题: 如果传 --horizon 30，文件名还是 nn_h60 → 完全误导且会覆盖。这是 bug 还是因为 ensemble 永远只用 60？

## Q56: --no-wandb flag 完全没被使用
- 位置: 行 172
- 代码: `ap.add_argument("--no-wandb", action="store_true")`
- 问题: 代码里没 import wandb 也没 wandb.init。为什么留这个 flag？

## Q57: --skip-phase1 但 Phase 1 不存在时不报错
- 位置: 行 173-174, 229
- 代码: `if args.skip_phase1 and os.path.isfile(anchor_path):`
- 问题: 若 --skip-phase1=True 但 anchor 不存在，会 fallthrough 进 phase1 重训而不警告——这是有意 fallback 吗？

## Q58: CUDA_VISIBLE_DEVICES 在 import torch 之后才设置
- 位置: 行 177
- 代码: `os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda`
- 问题: torch 已经初始化过 CUDA context，这时 set CUDA_VISIBLE_DEVICES 通常无效。是不是应该在 import torch 之前？

## Q59: 种子设置不包含 cudnn deterministic
- 位置: 行 194-196
- 代码: `torch.manual_seed(S); torch.cuda.manual_seed_all(S); np.random.seed(S)`
- 问题: 没有 `torch.backends.cudnn.deterministic = True` 也没有 set `use_deterministic_algorithms`——为什么放弃严格复现？

## Q60: torch.manual_seed 也作为 torch.randperm 的源
- 位置: 行 194, 306, 446
- 代码: `torch.manual_seed(S); ... perm = torch.randperm(n_train)`
- 问题: 每个 epoch 的 shuffle 顺序确定但仅依赖 global RNG。如果中间有任何其他 torch 随机操作，shuffle 会偏移——这是预期的吗？

---

## I. 数据加载与特征筛选（行 198–224）

## Q61: feat_names_path 与 schemeP_{train,val,test}.npz 的 schema 不在脚本控制
- 位置: 行 198-200
- 代码: 读 `schemeP_feat_names.txt`
- 问题: 训练脚本完全依赖上游 cache 的列顺序——这个 schema 有版本控制吗？如果有人重新跑特征工程改列顺序，会静默用错特征。

## Q62: forbidden 集仅 {date, sym, time}
- 位置: 行 207-208
- 代码: `forbidden = {"date", "sym", "time"}`
- 问题: CRITICAL_CONSTRAINTS 也提到 sym embedding。这里只检查名字不在 feat_names 里，但若特征工程派生的特征间接编码 sym（如 per-sym z-score），仍违规。这个检查是否足够？

## Q63: drop_idx 用 set comprehension 静默丢失未匹配项
- 位置: 行 203
- 代码: `drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)`
- 问题: 若 DROP_NAMES 拼错或 schema 变了，会静默跳过，从而**没有**实际 drop。应该 assert len(drop_idx)==len(DROP_NAMES)？

## Q64: 不打印实际丢掉的特征名
- 位置: 行 209
- 代码: `print(f"  feat_dim={feat_dim} (dropped {len(drop_idx)})", flush=True)`
- 问题: 只打了个数没打名字，看 log 时无法确认到底丢了什么。为什么不输出 dropped names？

## Q65: standardize 是闭包，依赖外层 feat_mean/feat_std
- 位置: 行 221-224
- 代码: `def standardize(X): Xs = (X - feat_mean) / feat_std`
- 问题: 闭包变量在 Phase 2 被重新赋值（行 374-375）——同一个 standardize 函数在两阶段会引用不同 mean/std。这种"看似全局"的行为容易出 bug。

## Q66: standardize 对 NaN 替换为 0
- 位置: 行 223
- 代码: `Xs = np.where(np.isnan(Xs), 0.0, Xs)`
- 问题: 0 等价于"平均值"（在 z-score 空间）——为什么不在原空间用 feat_mean 替换，再 z-score？两种做法等价吗？

## Q67: standardize 函数只在 val 推理时用，训练时用 GPU 内联
- 位置: 行 280, 313
- 代码: val 用 standardize；训练用 `torch.clamp((xb-fm_t)/fs_t, -CLIP, CLIP)`
- 问题: 两条标准化代码路径，逻辑上要等价——NaN 处理在 train 是不处理（因为已经 imputed），val 是 np.where。为什么不统一？

---

## J. Phase 1: 数据准备（行 241–280）

## Q68: NaN imputation 用每列 mean，但用 for-loop 迭代每列
- 位置: 行 260-263
- 代码: `for d_idx in np.where(nan_mask.any(axis=0))[0]: ...`
- 问题: CLAUDE.md 要求"特征构建：NumPy/Pandas 向量化，禁止 Python for 循环"——这里虽然只迭代有 NaN 的列（通常少），但仍违反规则。一行 `X_tr_imp = np.where(nan_mask, feat_mean[None,:], X_tr_imp)` 即可。

## Q69: feat_mean/feat_std 用 nanmean/nanstd 计算
- 位置: 行 254-256
- 代码: `feat_mean = np.nanmean(X_tr, axis=0); feat_std = np.nanstd(X_tr, axis=0)`
- 问题: nanstd 默认 ddof=0；该用 ddof=1 吗？另外某列若全 NaN，会 warning + 给 NaN——没有 sanity check 该情况。

## Q70: feat_std 下限 1e-6
- 位置: 行 256
- 代码: `feat_std = np.maximum(feat_std, 1e-6)`
- 问题: 1e-6 太小：若某列原始 std 是 1e-9，clip 到 1e-6 后 (x-mean)/1e-6 仍可能 explode。为什么不用更大下限或 drop 零方差列？

## Q71: target_scale = 1/std(y_regr_tr)
- 位置: 行 266
- 代码: `target_scale = 1.0 / float(max(y_regr_tr.std(), 1e-8))`
- 问题: 用 train set 的 std，但 val/test 分布可能不同。为什么不用 robust scale（IQR）？1e-8 floor 与 1e-6 floor 标准不同。

## Q72: 数据增强生成"翻倍"数据
- 位置: 行 270-274
- 代码: `X_aug = X_tr_imp * scales; X_full = concatenate([X_tr_imp, X_aug])`
- 问题: 直接 concat 原样本 + augmented 副本（共 2N），相当于显式 oversample。为什么不在 batch 里随机 sample 决定要不要 aug？这种 2N 增加显存和训练时间，没用 online。

## Q73: 每个样本独立的 per-feature scale
- 位置: 行 271
- 代码: `scales = rng_aug.uniform(AUG_LO, AUG_HI, size=X_tr_imp.shape)`
- 问题: 每个 (sample, feature) 独立采样 scale；这意味着同一行内不同特征独立扰动，特征之间的"协同"会被打散。是有意还是无心？

## Q74: 增强的 RNG 用 S*7919+1
- 位置: 行 270
- 代码: `rng_aug = np.random.default_rng(S * 7919 + 1)`
- 问题: 7919 是个素数，但为什么 +1？是为了和 phase 2 的 +137 区分吗？两个 +N 选 1 和 137 有何依据？

## Q75: 增强后 y_full_s 直接 concat 同样的 y 两次
- 位置: 行 274
- 代码: `y_full_s = np.concatenate([y_regr_tr_s, y_regr_tr_s])`
- 问题: 特征被扰动但 label 不变——这相当于把模型推向"特征独立 ±20% 时 label 不变"的不变性。这个先验合理吗（如果特征是 informative 的，扰动 ±20% 应该有信号变化）？

## Q76: sw_full 是基于 doubled y_cls 计算
- 位置: 行 278
- 代码: `sw_full = class_balanced_weight(y_cls_full, num_class=NUM_CLASS)`
- 问题: y_cls 重复后类别分布不变，所以 weight 一样。但相比 phase 2 在 train+val+test 上重算，能否合并？

---

## K. Phase 1: 训练循环（行 283–344）

## Q77: X_tr_raw 是 CPU tensor，每 batch .to(device)
- 位置: 行 283, 310
- 代码: `X_tr_raw = torch.from_numpy(X_full); xb = X_tr_raw[idx].to(device, non_blocking=True)`
- 问题: 2N×220 的 float32 ≈ N×1.7KB，1M 样本 ≈ 1.7GB。为什么不一次 .to(device)？反复传输 PCIE 带宽浪费。

## Q78: non_blocking=True 但 source tensor 不在 pinned memory
- 位置: 行 310-312
- 代码: `xb = X_tr_raw[idx].to(device, non_blocking=True)`
- 问题: non_blocking 只在 pinned tensor 上真生效；这里没 .pin_memory()，是否实际是同步拷贝？

## Q79: epoch 内的 shuffle 不感知 seed cycling
- 位置: 行 306
- 代码: `perm = torch.randperm(n_train)`
- 问题: 因为 torch.manual_seed(S) 在 main 头部，n_train 第一次 randperm 是确定的；但若 Phase 2 内部某操作消耗了 RNG，phase 2 的 shuffle 会偏移。为什么不用 generator？

## Q80: 每 batch 才 standardize（GPU 上）
- 位置: 行 313
- 代码: `xb = torch.clamp((xb - fm_t) / fs_t, -CLIP, CLIP)`
- 问题: 每个 epoch 每 batch 重做相同的标准化，浪费算力。为什么不在数据准备时一次性 standardize 然后存 CPU tensor？

## Q81: drop_last=False（隐式）
- 位置: 行 308
- 代码: `for s in range(0, n_train, batch_p1):`
- 问题: 最后一个 batch 可能很小（甚至 1）；对 BatchNorm 会 broken（不过这里用 LayerNorm 没事），但 loss 的 wb.sum() 在小 batch 容易抖。

## Q82: r_loss/r_n 把每个 batch 的 mean 再加权
- 位置: 行 320-321
- 代码: `r_loss += float(mse.item()) * xb.size(0); r_n += xb.size(0)`
- 问题: mse 已是 per-batch weighted mean（不是按 sample sum），乘 batch size 再平均不严格等价于 global mean——尤其类别权重和 sample 数耦合时。这个统计量是否准确？

## Q83: grad clip norm = 5.0
- 位置: 行 318, 466
- 代码: `torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)`
- 问题: 5.0 是怎么选的？训练日志里见过梯度爆到 5+ 吗？做过 1.0/10.0 对比吗？

## Q84: scheduler.step() 在 epoch 结尾
- 位置: 行 322, 474
- 代码: `scheduler.step()`
- 问题: per-epoch 步进合理；但 PHASE2 只有 11 epoch、CosineAnnealing T_max=11 → 最后一 ep lr ≈ 0，几乎不更新——这是有意的吗？

## Q85: CosineAnnealingLR eta_min=1e-5
- 位置: 行 293-295
- 代码: `CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS, eta_min=1e-5)`
- 问题: 三个 phase1 lr 都余弦衰减到 1e-5：1e-4 → 1e-5 跌 10x（合理），1e-3 → 1e-5 跌 100x（也常见）。但 lr=1e-4 起点已经离 eta_min 不远，余弦曲线对它效果微弱。为什么不针对 lr_p1 自适应 eta_min？

## Q86: val 推理用 predict_chunked 但训练用同一个 standardize 不一致
- 位置: 行 280, 324
- 代码: `X_va_std = ...standardize(X_va); va_pred_s = predict_chunked(model, X_va_std, ...)`
- 问题: val 已经 pre-standardized，predict_chunked 内部不再做（也不能做，input 已标准化）。这种"训练时标准化在 GPU 内联、推理时标准化在 CPU 离线"的双路径维护成本高。

## Q87: val 推理把 scaled pred / target_scale
- 位置: 行 325
- 代码: `va_pred = va_pred_s / target_scale`
- 问题: 模型输出在 scaled 空间，÷ target_scale 还原到 y_regr_va 空间，再算 mse。为什么不在 scaled 空间里算？方便比较还是有别的考虑？

## Q88: corr 只在 std>0 时算，否则 0
- 位置: 行 327
- 代码: `va_corr = float(np.corrcoef(...)[0,1]) if va_pred.std() > 0 else 0.0`
- 问题: corrcoef 自己已经会 nan-handle。这层 try-style 守护是不是多余？

## Q89: best_state 在 CPU clone
- 位置: 行 334
- 代码: `best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}`
- 问题: 每 epoch 都 clone 整个 state_dict 到 CPU——内存压力 + 同步。能否只在 update 时 clone？这里确实是 if improvement 才 clone，OK。但循环里仍 detach()，state_dict() 已经是 detach 的，为什么再 .detach()？

## Q90: early stop tolerance 1e-10
- 位置: 行 331
- 代码: `if val_mse < best_val_mse - 1e-10:`
- 问题: 1e-10 在 float32 噪声内基本等价于 "<" 严格小于；为什么不更宽松（如 1e-5 * best）？这里 noisy 改善也算 improvement，会延迟早停。

## Q91: best_state 在最后一 epoch 都没出现时怎么办
- 位置: 行 345
- 代码: `assert best_state is not None, "Phase 1 never improved"`
- 问题: assert 在 worst-case 抛错——理论上 ep 0 的 val_mse 就是初始 best，best_state 一定被设置。这个 assert 只能在 PHASE1_EPOCHS=0 时触发——是 dead check 还是怕未来改 epoch=0？

---

## L. Phase 1 → Phase 2 衔接（行 347–382）

## Q92: ckpt_p1 保存 numpy 数组而不是 torch tensor
- 位置: 行 349-350
- 代码: `"feat_mean": feat_mean` （numpy）
- 问题: torch.save 能存 numpy，但读取后是 numpy not tensor——phase 2 仍要 `np.asarray(...)` 再 torch.from_numpy。如果存成 tensor 直接搬上 GPU 更快。

## Q93: weights_only=False 加载 checkpoint
- 位置: 行 231
- 代码: `torch.load(anchor_path, map_location="cpu", weights_only=False)`
- 问题: pytorch 2.x 推荐 weights_only=True；用 False 有 pickle 任意代码执行风险。是因为 ckpt 包含 non-tensor 元数据（hidden tuple 等）？

## Q94: Phase 2 从 ckpt 重新读 keep_idx/dropout/hidden 即使脚本里有常量
- 位置: 行 376-382
- 代码: `keep_idx_p2 = np.asarray(ckpt_p1["keep_idx"], ...)`
- 问题: 这是好的（解耦），但脚本里又 import 同样的 HIDDEN/DROP_NAMES——如果 ckpt 和 const 不一致，会以 ckpt 为准。这设计 if 是否有文档？

## Q95: target_scale 不在 Phase 2 重算
- 位置: 行 377, 421
- 代码: `target_scale = float(ckpt_p1["target_scale"])`
- 问题: phase 2 用全数据，但 scale 仍来自 phase 1 train 子集的 std。这种"用旧 scale 看新数据"是有意保持一致还是 bug？

## Q96: feat_mean / feat_std 也来自 phase 1
- 位置: 行 374-375
- 代码: `feat_mean = np.asarray(ckpt_p1["feat_mean"], ...)`
- 问题: 类似上面，phase 2 不重算。若 val/test 分布有 shift，标准化会偏。是有意防止"测试集污染 normalization"吗？

---

## M. Phase 2: 数据准备（行 384–428）

## Q97: 拼接 train + val + test 进入训练
- 位置: 行 390-394
- 代码: `X_all = np.concatenate([train_d["X"], val_d["X"], test_d["X"]], ...)`
- 问题: **最 critical 的问题**：用 test_d 训练是否构成 data leakage？竞赛/评测约定如何？CRITICAL_CONSTRAINTS 里没看到禁止——是因为这里"test"是带 label 的内部 holdout，真正评测在另一个秘密 test？请阐明。

## Q98: 同时取 test 的 y、mp_t、mp_th
- 位置: 行 391-394
- 代码: 也 concat 了 test 的 mp_t/mp_t{H} 和 y{H}
- 问题: 这意味着 test split 有完整 label——所以 "test" 在这里指什么？为什么 phase 1 不用它（phase 1 只用 train+val）？

## Q99: M7 与 schemeP 命名不一致
- 位置: 行 372, 386
- 代码: phase 2 标题 "M7" 但加载 schemeP
- 问题: M7 是数据集名还是配置名？schemeP 又是什么？两者关系？

## Q100: Phase 2 augmentation 分块 200_000
- 位置: 行 410-414
- 代码: `CHUNK = 200_000; for i in range(0, n_orig, CHUNK):`
- 问题: 为什么分块？X_all 整体可能 3M × 220 × 4B ≈ 2.6GB，2x 增强 5GB——内存不够才分块？200K 这个数怎么定？

## Q101: 增强后 X_full 用 np.empty 预分配
- 位置: 行 408
- 代码: `X_full = np.empty((2 * n_orig, X_all.shape[1]), dtype=np.float32)`
- 问题: empty 比 zeros 略快但内容是未初始化——靠后续完全填充。如果某行写错（如 chunk loop 越界），会读到垃圾。为什么不用 zeros 更安全？

## Q102: 增强的 RNG seed 是 S*7919+137 与 Phase 1 的 +1 不同
- 位置: 行 406
- 代码: `rng_aug2 = np.random.default_rng(S * 7919 + 137)`
- 问题: +1 vs +137 magic numbers，分隔两阶段；137 怎么选的？这种 magic offset 不写成常量难维护。

## Q103: y_regr_full、fee_full、sw_full 都重复 2 次
- 位置: 行 417-423
- 代码: `y_regr_full = np.concatenate([y_regr_all, y_regr_all])`
- 问题: 与 phase1 同思路；但 augmented 副本的 fee 不变是否合理（如果 fee 是 mp_t 的函数，augmenting 特征不应改 mp_t）？fee_eff 也不是来自 features，所以合理。但这点没注释。

## Q104: sample weight 在 Phase 2 也是基于 y_cls
- 位置: 行 423
- 代码: `sw_full = class_balanced_weight(y_cls_full, ...)`
- 问题: phase 2 加入了 val+test，class 分布可能与 phase 1 不同 → weight 也不同。这是有意的"按全数据 reweight"还是 inconsistency？

## Q105: Phase 2 fee_scaled = fee * target_scale
- 位置: 行 422
- 代码: `fee_scaled_full = (fee_full * target_scale)`
- 问题: 同 y 一起 scale，保持 SPO+ loss 单位一致——但 target_scale 来自 phase 1 train，对 val/test 是否仍合适？

---

## N. Phase 2: 训练循环（行 431–477）

## Q106: 模型重建后 load_state_dict
- 位置: 行 431-433
- 代码: `model = MLPRegr(...); model.load_state_dict(ckpt_p1["state_dict"])`
- 问题: 直接重建+load 而不是 deep copy phase 1 的 model 实例。如果 hidden/dropout 不一致会 load 报错——已经在前面 read ckpt 来构造，OK。但 dropout 在 fine-tune 是否应该减小（已经 warmstart）？

## Q107: Phase 2 完全不冻结任何层
- 位置: 行 431-435
- 代码: 整个模型 trainable
- 问题: 经典 fine-tune 会 freeze backbone 只 train head，或用 differential learning rate。这里全 unfreeze 用 lr=3e-5 一刀切——为什么不分层 lr？

## Q108: Phase 2 没有 val 监控
- 位置: 行 443-477
- 代码: 训练循环里没有 evaluation
- 问题: 不监控就无法早停或选 best；这意味着 11 epoch 全跑，最后一 epoch 状态被保存。如果 ep 5 已 overfit，最后一 ep 比它差。为什么不监控 val？

## Q109: Phase 2 不保存中间 checkpoint
- 位置: 行 482
- 代码: 最终只存最后一 epoch
- 问题: 训练崩溃就要重跑；为什么不每个 epoch save？

## Q110: Phase 2 loss = l2 + 30 * spo
- 位置: 行 462
- 代码: `loss = l2_loss + PHASE2_LAMBDA_SPO * spo_loss`
- 问题: l2 是 (pred-y)^2 数量级 ≈ y^2 ≈ var(y_scaled)=1（因为 scaled）；spo 是线性 ≈ 1。所以 30:1 的混合下，spo 主导 30x，l2 几乎 anchor。这个比例合理吗？

## Q111: 不退火 λ_spo
- 位置: 行 462
- 代码: λ_spo 整个 phase 2 固定
- 问题: 经典 DFL 训练会先小 λ（多 L2）再大 λ（多 SPO+）渐进。直接 30 起步是否会让 phase 1 的 L2 anchor 立刻被冲掉？

## Q112: 没有梯度累积
- 位置: 行 449
- 代码: 每个 mini-batch 直接 step
- 问题: 如果 batch=4096 显存够，OK；但如果未来要更大 effective batch，无 accumulation 接口。

## Q113: 训练完后 final_state 又 .detach().clone().cpu()
- 位置: 行 482
- 代码: 同 phase 1 best_state 写法
- 问题: state_dict() 返回的 tensor 已是参数引用，clone+cpu 也合理；但 detach 多余。

## Q114: final_ckpt 里 "M7_full_retrain": True 是 metadata
- 位置: 行 497
- 代码: ckpt dict
- 问题: 字段名容易和实验跟踪混淆。文档化在哪？

## Q115: data_dates "0-119" 字符串硬编码
- 位置: 行 498
- 代码: `"data_dates": "0-119"`
- 问题: 实际取决于 schemeP_{train,val,test}.npz 内容；这里只是声明字符串，不与数据对得上时会误导。

## Q116: extract_npz 在 main 末尾被调用
- 位置: 行 505
- 代码: `extract_npz(final_ckpt, npz_path)`
- 问题: 等 phase 2 完成才 export；如果 phase 2 崩，phase 1 的 anchor 不会被 npz 化。如果推理要 fallback 到 phase 1 anchor 也没 npz 可用。

## Q117: summary 字段少且无 metric
- 位置: 行 507-519
- 代码: summary dict
- 问题: 没有 phase2 final loss、没有 val_mse final、没有任何质量指标——下游怎么挑 seed？

## Q118: DONE 行只打 phase1 信息
- 位置: 行 520
- 代码: `print(f"\nDONE seed={S}  phase1_best_ep=...")`
- 问题: 完成消息只提 phase1，不提 phase2 是否成功——log grep 的时候 misleading。

---

## O. 复现性与 logging

## Q119: 没有 deterministic dataloader workers（其实没用 DataLoader）
- 位置: 整体
- 代码: 手写 mini-batch loop
- 问题: 为什么不用 torch.utils.data.DataLoader？是性能考虑？

## Q120: 没有 torch.use_deterministic_algorithms(True)
- 位置: 行 194-196
- 代码: seed 设置
- 问题: 即使设了 seed，cuDNN 的某些 op（如 conv backward）默认非确定性；这里只有 linear/LN 通常 OK，但 GELU+autograd 仍可能微小漂移。是否验证过相同 seed 跑两次得相同 final_state？

## Q121: 没有 git commit hash 写入 summary
- 位置: 行 507-518
- 代码: summary
- 问题: 复现需要知道代码版本——为什么不嵌入 git rev？

## Q122: 没有 wandb 也没有 tensorboard
- 位置: 整体
- 代码: print only
- 问题: CLAUDE.md 明文要求"所有实验必须用 WandB 记录"——这里完全没用。是因为这是 final_submission_code 不需要？

## Q123: print 没有时间戳
- 位置: 行 191, 328 等
- 代码: `print(...)`
- 问题: 日志 grep 不易定位事件时间。

## Q124: 没有进度文件（worker-progress.json）输出
- 位置: 整体
- 代码: 无
- 问题: CLAUDE.md 要求定期更新 worker-progress.json——这个脚本作为 worker 没写。是因为它是 submission 而非内部 worker？

## Q125: 没有 results.json 输出（CLAUDE.md 要求）
- 位置: 行 507-518
- 代码: 输出的是 summary_nn_seed{S}.json
- 问题: 不叫 results.json 也没 RESULT 行——为什么文件命名不与规范一致？

---

## P. 边界条件与容错

## Q126: 假设 schemeP_{train,val,test}.npz 都存在
- 位置: 行 385-387
- 代码: `train_d = load_npz("train"); val_d = ...; test_d = ...`
- 问题: 如果 test 不存在会 raise FileNotFoundError——但 phase 1 没用 test。无 try/except 也无 --skip-test。

## Q127: 全零 y_regr_tr.std() 的退化
- 位置: 行 266
- 代码: `target_scale = 1.0 / float(max(y_regr_tr.std(), 1e-8))`
- 问题: 1e-8 下限给出 target_scale=1e8——异常大的 scale 会让 y_scaled 极端。这是无声 fallback，没有 warning。

## Q128: feat_std 为 0 的列不会被 drop
- 位置: 行 256
- 代码: `feat_std = np.maximum(feat_std, 1e-6)`
- 问题: 全为常数的列被 z-score 后变 0（因为 X-mean=0）；但占用一个 input dim 没意义——为什么不直接 drop？

## Q129: NaN 在 augmented 数据里的处理
- 位置: 行 271-272
- 代码: `scales = uniform(...); X_aug = X_tr_imp * scales`
- 问题: X_tr_imp 已经 NaN-imputed；augmentation 乘 [0.8, 1.2] 不引入新 NaN——OK；但 phase 2 在 line 400-404 重做 NaN imputation 是不是冗余（已经在 phase 1 处理过的数据来源）？这里 X_all 是新加载的 train+val+test，所以确实需要重新处理。但有没有可能 imputation 在 train/val/test 间不一致（用 phase1 feat_mean 一致地填，OK）。

## Q130: 没有检查输入 X 中存在 Inf
- 位置: 行 254-263, 400-404
- 代码: 只处理 NaN
- 问题: 如果某特征生成时除以 0 得 Inf，nanmean 不会处理，会污染 mean。为什么不 also handle Inf？

## Q131: target_scale × y 可能溢出 fp32
- 位置: 行 267
- 代码: `y_regr_tr_s = (y_regr_tr * target_scale).astype(np.float32)`
- 问题: 若 target_scale=1e8 且 y_regr_tr=1e-3，结果=1e5，fp32 表示无问题。但 phase2 fee*target_scale 在某些边界情况下也要验。是否有 sanity check？

## Q132: y_regr_va 不缩放但用于 val_mse 比较
- 位置: 行 326
- 代码: `val_mse = float(((va_pred - y_regr_va) ** 2).mean())`
- 问题: va_pred 已除以 target_scale 还原；val_mse 是原始 scale 的 mse——可以跨 seed 比较，但和 train loss（scaled）不在同一 scale，log 里看起来异常。

## Q133: 用 nan_mask.any() 的 cost
- 位置: 行 259, 400
- 代码: `nan_mask = np.isnan(X_tr_imp); if nan_mask.any():`
- 问题: 整个 mask materialize 用 X.size bytes；若 X 是 100M cells × 1 byte ≈ 100MB——小问题但可以用 chunked check 省内存。

## Q134: val 推理 batch=16384 全数据可能爆显存
- 位置: 行 280, 324
- 代码: X_va_std 整个 tensor → predict_chunked
- 问题: predict_chunked 内部 chunk OK；但 X_va_std 本体是个完整 CPU tensor。val 是 80-95 ≈ 16 天 × N_per_day，多大？

---

## Q. 数据 leak / sym-agnostic / 时间穿越相关

## Q135: 用了 train+val+test 训练 → 评测时 test 还是用同一份吗？
- 位置: 行 390-394
- 代码: phase 2 concat
- 问题: 如果 "schemeP_test" 是真正的最终 test，那 phase 2 是严重 leakage；如果是另一个 holdout，请文档化语义。

## Q136: 没有按 date 切分确认
- 位置: 行 241-249
- 代码: phase 1 用 train(0-79) val(80-95)
- 问题: 这里假设 train_d 的所有样本 date < 80，val_d 的 date 在 [80,95]——但脚本不检查这个假设。如果 schemeP_train.npz 实际混了 val 的日期，就 silent leak。

## Q137: sym 字段是否真的被排除？
- 位置: 行 207
- 代码: `forbidden = {"date", "sym", "time"}`
- 问题: 只检查名字，不检查内容；feat_names 里如果有派生特征隐含 sym 信息（如 sym-specific Z-score）也无法发现。

## Q138: date 是被置 0 的（评测约束）
- 位置: 行 207
- 代码: forbidden 集
- 问题: 训练数据的 date 有真值，所以模型学不到对 date 的依赖；但如果某特征是基于 date 计算的（如时段特征），评测时 date=0 会让该特征也异常。脚本没考虑这个 covariate shift。

## Q139: Predictor 是否 stateless（CRITICAL_CONSTRAINTS #2）
- 位置: extract_npz 之后的推理（脚本外）
- 代码: N/A
- 问题: 训练脚本本身不涉及，但 npz 输出后的推理代码必须 stateless。这个文件没文档化推理的 stateless 性。

## Q140: sym 含训练外股票时模型行为
- 位置: N/A
- 代码: 不涉及 sym embedding
- 问题: feat 中如果有任何 per-sym aggregate（即使没显式用 sym），新 sym 会让该特征值异常。这里没显式的 OOD 防护。

---

## R. 性能与工程细节

## Q141: 内存峰值估计
- 位置: 整个 phase 2
- 代码: X_all + X_full(2x) + y/fee/sw doubled
- 问题: train+val+test 全数据 + 2x 增强 + 3 个 doubled vector，峰值可能 10GB+。在多 seed 并行时会 OOM。脚本无内存监控/控制。

## Q142: 无 mixed precision (amp)
- 位置: 行 314, 457
- 代码: model(xb) full fp32
- 问题: A100/3090 上 amp 通常 1.5-2x 加速；为什么不用 torch.cuda.amp？

## Q143: 无 compile（torch.compile）
- 位置: 行 291, 431
- 代码: model 直接 train
- 问题: PT 2.x 的 torch.compile 对 MLP 提速明显——为什么不试？

## Q144: 重复 import torch.nn.functional as F
- 位置: 行 30
- 代码: `import torch.nn.functional as F`
- 问题: 只用了 F.relu 一次——直接 `torch.relu` 可省 import。小问题但代码 idiom。

## Q145: time.time() 用 wall-clock 而非 perf_counter
- 位置: 行 303, 442, 444
- 代码: `t0_p1 = time.time()`
- 问题: 测时长用 perf_counter 更精确；time.time() 受系统时钟跳变影响。

---

## S. HP 多样性设计的疑问

## Q146: 50 个 seed 只有 12 种独立 HP 组合
- 位置: 行 186-188
- 代码: cycling 模式
- 问题: lr×batch=mod3, dropout=mod4 → LCM 12。剩下 38 个 seed 是"同 HP 不同 init/aug"——这是有意的"对 12 组合内每组多次 seed 平均"还是设计 bug？

## Q147: 不同 HP 的 seed 在 ensemble 里权重相等
- 位置: N/A（ensemble 在脚本外）
- 代码: 输出文件名都是 `nn_h60_seed{S}.npz` 没有 HP 标记
- 问题: 推理时 ensemble 平均不区分 HP——如果某 HP 组合较差，会拉低 ensemble。有做过 per-HP 选 best 吗？

## Q148: dropout 0.05 与 0.20 跨度 4x
- 位置: 行 38
- 代码: DROPOUT_LIST
- 问题: 这个 spread 是否太大——0.05 几乎是 no-reg，0.20 是 strong-reg；两端在 ensemble 里互相抵消？

## Q149: lr 1e-3 与 batch 8192 的耦合
- 位置: 行 186, 188
- 代码: cycling
- 问题: seed=3 → lr=1e-3 & batch=8192 & dropout=0.15。lr=1e-3 在大 batch 是 reasonable 的 linear scaling，但 1e-3 在 batch=2048（seed=1）是不是太大？没法分离哪个变化在影响 final。

## Q150: HP cycling 的 (S-1) 起点
- 位置: 行 186
- 代码: `(S - 1) % len(LR_LIST)`
- 问题: 从 0 开始；如果 S=0 会得到 LR_LIST[-1]——但 S>=1 强制了。要不要直接 (S-1) 强 assert？

---

## T. 其他

## Q151: extract_npz 在 phase 2 末尾导出，phase 1 anchor 不导出
- 位置: 行 505 vs phase 1 saved as .pt only
- 代码: anchor_path = .pt
- 问题: 如果 ensemble 想 fallback 用 phase 1 模型，需要 anchor 也 export 为 npz；推理端是否能 load torch .pt？

## Q152: 文件最末没有 wandb.finish() 或类似
- 位置: 行 524
- 代码: `main()`
- 问题: 无 cleanup——如果有人加 wandb，run 不会 close。

## Q153: print 大量使用 flush=True
- 位置: 多处
- 代码: `print(..., flush=True)`
- 问题: 是因为 stdout 被 redirect 到 train.log 怕 buffer？这种 verbosity 是必要还是 leftover？

## Q154: train_time_p1_sec 是 phase1 总时长，但 t_p1=0 if skip
- 位置: 行 513
- 代码: `"train_time_p1_sec": float(t_p1)`
- 问题: 当 --skip-phase1 触发，t_p1=0 被写入 summary——下游统计会以为 phase1 跑了 0 秒，misleading。

## Q155: 没有 finally 或 atexit 清理
- 位置: 整体
- 代码: 直接顺序执行
- 问题: 如果 phase 2 中途崩，已写的 .pt（phase 1 anchor）保留——OK；但 npz 不存在，summary 不存在，下游对 seed 状态判断需要复杂逻辑。

## Q156: feat_names 在 ckpt 里存的是 phase 1 的 feat_names
- 位置: 行 488 (phase 2 ckpt 也存)
- 代码: `"feat_names": feat_names`
- 问题: phase 2 没重赋值 feat_names——它仍是 phase 1 main scope 的 feat_names。如果 phase 1 skip 重新加载 ckpt 时，feat_names 在 phase 2 末尾的 ckpt 里仍 OK（来自 main scope 没变），但这种隐式 carry over 容易出 bug。

## Q157: hidden 在 phase 2 重新解包成 tuple
- 位置: 行 378
- 代码: `hidden = tuple(int(x) for x in ckpt_p1["hidden"])`
- 问题: ckpt_p1["hidden"] 已是 tuple（行 353），tuple(...) 是冗余；but 防御性写法。

## Q158: dropout 在 fine-tune 仍开启
- 位置: 行 431-432, 445
- 代码: `model = MLPRegr(..., dropout=dropout_p2); model.train()`
- 问题: fine-tune dropout 是否应该减小（如 phase2_dropout = dropout_p2 * 0.5）？没有可调旋钮。

## Q159: SPO+ 的 dropout 影响梯度方差
- 位置: 行 459
- 代码: `spo_per = spo_plus_loss(pred, yb, fb)`
- 问题: pred 受 dropout 影响波动大，SPO+ 的 z_star 依赖 pred 也会跳——这种交互是否文档化？

## Q160: 没有 evaluate phase 2 之后 final 模型在 val 的表现
- 位置: phase 2 末
- 代码: 直接 save
- 问题: 至少打一行 phase 2 final val_mse 让 log 可读——为什么省？

## Q161: phase 1 anchor 保存了 ckpt 一些但 phase 2 final ckpt 也独立保存
- 位置: 行 366 vs 502
- 代码: 两次 torch.save
- 问题: 磁盘双倍存（anchor + final）——大 ensemble 累计 50×2 个 .pt 文件。是否必要保留两份？

## Q162: extract_npz 内部对 sd[f"net.{layer_idx}.weight"] 直接 .numpy()
- 位置: 行 149-150
- 代码: `sd[...].numpy()`
- 问题: 如果 state_dict 来自 GPU，.numpy() 会 raise——这里 best_state/final_state 都是 cpu clone OK，但写法不防御。

## Q163: 多 seed 并发跑时 CUDA_VISIBLE_DEVICES 设定的副作用
- 位置: 行 177
- 代码: 全局 env 设
- 问题: 同进程内多次调 main 会冲突——但脚本只能单进程跑。若 launcher 把 main 当 subprocess 调用，OK。

## Q164: print(f"\n{'='*60}") 视觉分隔
- 位置: 行 190
- 代码: print decorations
- 问题: log 可读性 OK；但纯装饰行——大型脚本里堆积冗余。

## Q165: try/except 几乎不存在
- 位置: 整体
- 代码: 唯一防御是 `va_pred.std() > 0` 这种 if
- 问题: 没有 try/except 包整个 phase——若 OOM 在 phase 2 ep 5 崩，phase 1 anchor 保留但没 final ckpt，下游需要 fallback 逻辑。

---

完。

RESULT: task=[nn train review] metrics={n_questions=165} notes=[问题覆盖超参常量、HP cycling 设计、MLP 架构与初始化、SPO+ 损失公式与梯度稳定性、两阶段衔接（特别是 phase 2 用 train+val+test 是否构成 leakage、target_scale/feat_mean 沿用 phase 1 是否合适、无 val 监控）、数据处理 NaN/Inf/std=0 边界、复现性（cudnn deterministic 缺失、weights_only=False）、工程细节（amp/compile 未用、cpu→gpu non_blocking 但未 pin_memory、shuffle RNG 未 generator 隔离）、合规性（无 wandb、无 worker-progress、文件名 hardcode h60 与 --horizon 不一致），以及若干 magic number（7919/137、PHASE2_EPOCHS=11、CLIP=10、grad clip 5.0）的来源未说明]

