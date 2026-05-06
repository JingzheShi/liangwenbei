# DANN — Domain-Adversarial Training of Neural Networks

> Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand, Lempitsky. JMLR 2016
> [arXiv:1505.07818](https://arxiv.org/abs/1505.07818) | [JMLR 17](https://jmlr.org/papers/volume17/15-239/15-239.pdf)

## 一句话

**Adversarial domain confusion**：feature extractor 努力让 domain classifier 区分不出 source/target，迫使 backbone 学到 domain-invariant 表示。Inference 时直接丢掉 domain head。

## 架构

```
                 ┌──→ label predictor (任务头)  ──→ y
input ──→ Φ(x) ──┤
                 └──→ GRL ──→ domain classifier ──→ domain_label
```

- **Feature extractor Φ**：backbone（CNN / MLP / DeepLOB）
- **Label predictor**：原任务 head（我们这里：3-class up/flat/down）
- **Domain classifier**：分类 source vs target 域（我们这里：5-class sym ID）
- **GRL (Gradient Reversal Layer)**：forward = identity，backward = `-λ × grad`
  - 对 domain classifier 的梯度，到达 backbone 时被翻转 → backbone 朝"让 domain 不可分"的方向更新
  - 对 label predictor 的梯度照常传

## Loss

```
L_total = L_label(predictor(Φ(x)), y) - λ · L_domain(classifier(Φ(x)), domain)
```

(注意 minus sign — 我们要最大化 domain loss)

通过 GRL 实现：forward 是 `+λ` mult，backward auto 反 grad。

## λ schedule

paper 推荐：
```
λ(p) = 2 / (1 + exp(-10·p)) - 1
```
其中 p = 当前 step / 总 steps（[0,1]）
- 训练初期 λ ≈ 0（让 backbone 先学到任务相关 features）
- 训练后期 λ → 1（强 domain confusion）

## 我们任务上的应用

**Domain = sym ID (0-4)**

```python
class DANNModel(nn.Module):
    def __init__(self):
        self.backbone = MLPLOB(...)  # 我们的 LOB backbone
        self.task_head = nn.Linear(d_emb, 3)  # 3-class
        self.domain_head = nn.Linear(d_emb, 5)  # 5 sym
    
    def forward(self, x, alpha):
        feat = self.backbone(x)  # [B, d_emb]
        task_logits = self.task_head(feat)
        rev_feat = ReverseLayer.apply(feat, alpha)  # GRL
        domain_logits = self.domain_head(rev_feat)
        return task_logits, domain_logits

# Training step
task_logits, domain_logits = model(x, alpha=lambda_schedule(step))
loss = ce_loss(task_logits, y) + ce_loss(domain_logits, sym)
loss.backward()  # GRL handles the sign
```

**Inference**：只调用 `backbone + task_head`，**完全不传 sym**，符合硬约束 #3 ✅

## 在我们 setup 上的风险

1. **5 个 domain 太少**：domain classifier 太容易（5-class 比 3-class 简单）→ 反向梯度信号噪声大。**对策**：用 LOSO 风格——每个 mini-batch 只 sample 4 个 sym，强迫 domain classifier 在"留一外推"模式下工作。

2. **sym=2 太 outlier**：domain classifier 几乎能 100% 区分 sym=2 → λ 项主导 → backbone 被推向极端去消除 sym=2 信号 → 下游性能可能崩。**对策**：domain classifier 加 label smoothing（ε=0.2），降低过自信。

3. **不能控的 trade-off**：DANN 无法保证 task accuracy 不掉。**对策**：在 λ 上 hyperparam search，监控 LOSO 验证 cum_pnl。

## DANN 在时序 / 金融上的 evidence

- Deng 2024 ([Domain Generalization in Time Series Forecasting](https://staff.fnwi.uva.nl/m.derijke/wp-content/papercite-data/pdf/deng-2024-domain.pdf)) 把 DANN 和 wERM 列为"weaker baselines"——能 work 但不是 SOTA
- Springer 2023 ["DANN for DG: when it works and how to improve"](https://link.springer.com/article/10.1007/s10994-023-06324-x) 讨论 DANN failure modes，建议加 multi-task aux losses

## 实现

- 官方 PyTorch reference：[github.com/fungtion/DANN](https://github.com/fungtion/DANN)
- GRL 是核心，PyTorch one-liner：

```python
class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
```

## 推荐性

- **NN 路线优先级 #2**（在 V-REx / SWA 之后）
- 仅 NN 可用，LightGBM 无对应物
- 实现复杂度：中（半天到一天）

## Sources
- [arXiv:1505.07818](https://arxiv.org/abs/1505.07818)
- [JMLR 17](https://jmlr.org/papers/volume17/15-239/15-239.pdf)
- [github.com/fungtion/DANN](https://github.com/fungtion/DANN)
- [Springer "DANN for DG: when it works"](https://link.springer.com/article/10.1007/s10994-023-06324-x)
- [Counterfactual DANN IEEE 2017](https://ieeexplore.ieee.org/document/8253217/)
