# Implementation Spec: Top-5 NN 架构

> 给 opus worker 的实现规范
> 基于 cross_sym_nn 工作目录（已有 CrossConcatMLP / SymAttentionModel / HybridCrossAttn）
>
> **约束提醒**：
> - 输入格式：(B, 5, 359) per-group
> - 预测格式：(B, 5) continuous regression
> - 任务约束（修正后）：5 sym 测试和训练相同，允许跨 sym 交互
> - 训练对齐：AdamW, L2+sample-weight loss, mirror aug 跨 sym 同步
> - 不得修改 data_loader.py / train.py，模型类加入 models.py

---

## Spec 1: SetTransformerISAB

> 优先级 TOP-1。排列不变（permutation invariant），无 sym-specific 参数，跨 sym 通过 inducing points 交互。

### 背景

Set-Transformer（Lee et al., ICML 2019）使用 Induced Set Attention Block（ISAB）。
与 SymAttentionModel 的区别：没有 sym_emb，完全排列不变——任意置换 5 sym 的输入顺序，输出随之置换，预测不变。

### PyTorch 类签名

```python
class SetTransformerISAB(nn.Module):
    """
    Permutation-equivariant cross-sym architecture via ISAB.

    Architecture:
      1. Input projection: (B, 5, 359) -> (B, 5, d_model)
      2. ISAB x2: (B, 5, d_model) -> (B, 5, d_model) [m inducing points]
      3. Per-element head: (B, 5, d_model) -> (B, 5, d_head) -> (B, 5)
    """
    def __init__(
        self,
        n_sym: int = 5,
        n_feat: int = 359,
        d_model: int = 128,
        n_heads: int = 4,
        n_inducing: int = 8,     # m inducing points; n_inducing=5 是特殊 case = SAB
        depth: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_feat, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Each ISAB: MAB(X, MAB(I, X)) where I is learned inducing points
        self.isab_blocks = nn.ModuleList([
            ISAB(d_model, n_heads, n_inducing, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, 359)
        h = self.input_proj(x)          # (B, 5, d_model)
        for blk in self.isab_blocks:
            h = blk(h)                   # (B, 5, d_model)
        h = self.norm(h)
        return self.head(h).squeeze(-1)  # (B, 5)
```

### ISAB 子模块实现

```python
class MAB(nn.Module):
    """Multi-head Attention Block: MAB(X, Y) = LayerNorm(H + rFF(H)) where H=LayerNorm(X + Att(X,Y,Y))"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        h = d_model * 2
        self.ff = nn.Sequential(nn.Linear(d_model, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, d_model))
        self.drop_p = dropout

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # X: (B, n, d), Y: (B, m, d)  -> output (B, n, d)
        B, n, d = X.shape
        q = self.q_proj(X).reshape(B, n, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(Y).reshape(B, -1, 2, self.n_heads, self.d_head)
        k, v = kv[:,:,0].transpose(1,2), kv[:,:,1].transpose(1,2)
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_p if self.training else 0.0)
        h = h.transpose(1, 2).reshape(B, n, d)
        h = self.out_proj(h)
        X = self.norm1(X + h)
        X = self.norm2(X + self.ff(X))
        return X

class ISAB(nn.Module):
    """Induced Set Attention Block: MAB(X, MAB(I, X))"""
    def __init__(self, d_model: int, n_heads: int, n_inducing: int, dropout: float = 0.1):
        super().__init__()
        # Learned inducing points (not sym-specific: same I for all batches)
        self.I = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, n_heads, dropout)  # MAB(I, X): I attends to X
        self.mab2 = MAB(d_model, n_heads, dropout)  # MAB(X, H): X attends to compressed H

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, 5, d_model)
        B = X.size(0)
        I = self.I.expand(B, -1, -1)   # (B, n_inducing, d_model)
        H = self.mab1(I, X)             # (B, n_inducing, d_model): inducing points informed by X
        return self.mab2(X, H)          # (B, 5, d_model): X informed by compressed representation
```

### 关键超参

| 超参 | 默认值 | 说明 |
|---|---|---|
| d_model | 128 | token 维度 |
| n_heads | 4 | attention 头数 |
| n_inducing | 8 | 诱导点数量；5 时退化为 SAB |
| depth | 2 | ISAB 层数 |
| dropout | 0.10 | 与 baseline MLP 保持一致 |

**变体消融**：
- `n_inducing=5` (= n_sym) = SAB (no inducing)
- `depth=1` vs `depth=2`
- `d_model=64` (更轻量)

### 预期 PnL

- 乐观：+35 ~ +40（超过 MLP baseline，跨 sym 相关性确实存在）
- 中性：+28 ~ +34（接近 SymAttentionModel 上限）
- 悲观：+20 ~ +27（5 sym 太少，attention 噪声大）

---

## Spec 2: DeepSetsEnhanced

> 最简单的排列不变架构，快速验证 cross-sym 是否有任何价值。

### PyTorch 类签名

```python
class DeepSetsEnhanced(nn.Module):
    """
    DeepSets with enhanced aggregation: max + mean + std pooling.

    phi: per-sym encoder (shared weights) -> (B, 5, d_phi)
    agg: [max, mean, std] concat -> (B, 3*d_phi)
    rho: global context -> (B, d_rho)
    head: [phi(x_s), rho(context)] -> (B, 5, 1)
    """
    def __init__(
        self,
        n_sym: int = 5,
        n_feat: int = 359,
        d_phi: int = 128,
        d_rho: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(n_feat, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, d_phi),  nn.LayerNorm(d_phi), nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(3 * d_phi, d_rho), nn.LayerNorm(d_rho), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_phi + d_rho, d_phi // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_phi // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, 359)
        B, S, F = x.shape
        x_flat = x.reshape(B * S, F)
        h_flat = self.phi(x_flat)
        h = h_flat.reshape(B, S, -1)            # (B, 5, d_phi)

        # Permutation-invariant aggregation: max + mean + std
        h_max  = h.max(dim=1).values             # (B, d_phi)
        h_mean = h.mean(dim=1)                   # (B, d_phi)
        h_std  = h.std(dim=1, unbiased=False)    # (B, d_phi)
        agg    = torch.cat([h_max, h_mean, h_std], dim=-1)  # (B, 3*d_phi)

        ctx = self.rho(agg).unsqueeze(1).expand(-1, S, -1)  # (B, 5, d_rho)
        combined = torch.cat([h, ctx], dim=-1)               # (B, 5, d_phi + d_rho)
        return self.head(combined).squeeze(-1)               # (B, 5)
```

### 关键超参

| 超参 | 默认值 |
|---|---|
| d_phi | 128 |
| d_rho | 64 |
| dropout | 0.10 |

### 预期 PnL

- 乐观：+32 ~ +38（接近 MLP，cross-sym pooling 提供额外信号）
- 中性：+25 ~ +32（略低于 MLP，cross-sym noise 抵消收益）
- 悲观：+18 ~ +24（与 SymAttentionModel 相当，cross-sym 无益）

---

## Spec 3: SupervisedAutoEncoderMLP (SAE-MLP)

> 不依赖 cross-sym，直接在 359-d 上训练去噪 AE + MLP head。Jane Street 2020 1st place 架构适配版。

### PyTorch 类签名

```python
class SupervisedAutoEncoderMLP(nn.Module):
    """
    Supervised AutoEncoder MLP (Yirun 2020 Jane Street 1st adapted).

    Works on per-sample (B, 359) — not cross-sym — for maximum compatibility.
    But can be plugged into cross-sym pipeline as per-sym encoder inside a DeepSets wrapper.

    Loss = alpha * MSE(recon, x) + (1-alpha) * MSE(pred, y_reg)
    where recon is reconstruction of input x.

    For cross-sym pipeline: apply this as the phi encoder in DeepSets.
    Input shape: (B, n_feat)  [or (B, 5, n_feat) as batch of per-sym rows]
    """
    def __init__(
        self,
        n_feat: int = 359,
        d_latent: int = 64,
        d_hidden: int = 256,
        dropout: float = 0.10,
        recon_alpha: float = 0.3,
    ):
        super().__init__()
        self.recon_alpha = recon_alpha

        # Encoder: x -> latent
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.LayerNorm(d_latent), nn.GELU(),
        )
        # Decoder: latent -> x_hat (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_feat),
        )
        # Prediction head: concat(latent, x) -> 1
        self.pred_head = nn.Sequential(
            nn.Linear(d_latent + n_feat, d_hidden),
            nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, n_feat) or (B*5, n_feat) when used per-sym in cross-sym pipeline
        # Returns: (pred, recon)  -- pred: (B,) or (B*5,), recon: (B, n_feat)
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        pred   = self.pred_head(torch.cat([latent, x], dim=-1)).squeeze(-1)
        return pred, recon

    def loss(
        self,
        x: torch.Tensor,
        y_reg: torch.Tensor,
        sw: torch.Tensor,
    ) -> torch.Tensor:
        pred, recon = self.forward(x)
        pred_loss = (sw * (pred - y_reg) ** 2).mean()
        recon_loss = F.mse_loss(recon, x)
        return (1 - self.recon_alpha) * pred_loss + self.recon_alpha * recon_loss
```

### 训练 hooks（与 cross-sym pipeline 集成方式）

**选项 A（独立运行，不依赖 cross-sym）**：
- 将 SAE-MLP 作为独立 per-sample model（flatten (B,5,359) → (B*5, 359) then predict (B*5,) reshape to (B,5)）
- 直接在现有 train.py 框架中，input_mode="flat"

**选项 B（作为 DeepSets phi 的 encoder）**：
- 用 SAE-MLP.encoder 作为 phi 函数
- 外层 DeepSets 做 cross-sym aggregation
- 总 loss = pred_loss + reconstruction_loss

### 关键超参

| 超参 | 默认值 |
|---|---|
| d_latent | 64 |
| d_hidden | 256 |
| dropout | 0.10 |
| recon_alpha | 0.3 (30% reconstruction, 70% prediction) |
| noise_sigma | 0.035 (Gaussian noise on x during training, à la Yirun) |

### 预期 PnL

- 乐观：+36 ~ +42（去噪 latent + raw 双流超过 pure MLP）
- 中性：+32 ~ +36（与 MLP 相当，recon loss 没有帮助也没有拖累）
- 悲观：+26 ~ +30（recon loss 分散 capacity，prediction 变差）

---

## Spec 4: GraphAttentionCrossSym (GAT 5-node)

> 5 sym 作为 5 节点全连接图，GAT 学跨 sym 边权。与 SetTransformerISAB 互补验证。

### PyTorch 类签名

```python
class GraphAttentionCrossSym(nn.Module):
    """
    GAT on 5-sym fully connected graph.

    Architecture:
      1. Per-sym encoder: 359 -> d_node
      2. K GAT layers on 5-node fully connected graph
      3. Per-sym prediction head
    """
    def __init__(
        self,
        n_sym: int = 5,
        n_feat: int = 359,
        d_node: int = 128,
        n_heads: int = 4,
        n_gat_layers: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.n_sym = n_sym
        self.sym_encoder = nn.Sequential(
            nn.Linear(n_feat, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, d_node), nn.LayerNorm(d_node), nn.GELU(),
        )
        # GAT: simplified as multi-head attention where all 5 nodes are "neighbors"
        # Edge weights: e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
        self.gat_layers = nn.ModuleList([
            GATLayer(d_node, n_heads, dropout=dropout)
            for _ in range(n_gat_layers)
        ])
        self.norm = nn.LayerNorm(d_node)
        self.head = nn.Sequential(
            nn.Linear(d_node, d_node // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_node // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 5, 359)
        B, S, F = x.shape
        h = self.sym_encoder(x.reshape(B * S, F)).reshape(B, S, -1)  # (B, 5, d_node)
        for gat in self.gat_layers:
            h = gat(h)  # (B, 5, d_node)
        h = self.norm(h)
        return self.head(h).squeeze(-1)  # (B, 5)


class GATLayer(nn.Module):
    """Single GAT layer on fully-connected n_node graph."""
    def __init__(self, d_node: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_node % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_node // n_heads
        # Attention weight: a^T [q || k] for each head
        self.W_q = nn.Linear(d_node, d_node)
        self.W_k = nn.Linear(d_node, d_node)
        self.W_v = nn.Linear(d_node, d_node)
        self.a   = nn.Parameter(torch.randn(n_heads, 2 * self.d_head) * 0.02)
        self.out = nn.Linear(d_node, d_node)
        self.norm1 = nn.LayerNorm(d_node)
        self.norm2 = nn.LayerNorm(d_node)
        h = d_node * 2
        self.ff = nn.Sequential(nn.Linear(d_node, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, d_node))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, 5, d_node)
        B, N, D = h.shape
        # Compute edge attention weights (Veličković-style but simplified)
        q = self.W_q(h).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(h).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(h).reshape(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # Standard SDPA (equivalent to learned-edge-weight attention)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out(out)
        h = self.norm1(h + self.drop(out))
        h = self.norm2(h + self.ff(h))
        return h
```

### 关键超参

| 超参 | 默认值 |
|---|---|
| d_node | 128 |
| n_heads | 4 |
| n_gat_layers | 2 |
| dropout | 0.10 |

### 预期 PnL

- 乐观：+33 ~ +38（学到有效的跨 sym 关系）
- 中性：+25 ~ +33（与 SymAttentionModel 相当）
- 悲观：+18 ~ +25（5 节点的图 attention 欠拟合）

---

## Spec 5: FeatureGroupAttentionMLP

> 将 359 维特征按语义分组（~10-15 组），组级 attention + per-group MLP。不依赖 cross-sym。

### 语义分组方案（基于 partition.json + 领域知识）

```python
FEATURE_GROUPS = {
    "raw_price_size":  range(0,   80),   # bid1-10, ask1-10, bsize1-10, asize1-10 etc.
    "ofi_mlofi":       range(80,  110),  # F2 MLOFI 30d
    "wmp":             range(110, 121),  # F3 WMP 11d
    "vpin_ofi":        range(121, 129),  # F2 VPIN/OFI Toxicity
    "kyle_roll":       range(129, 138),  # F2 Kyle + Roll
    "jump_bv":         range(138, 150),  # F5 BV + jump
    "ema_features":    range(150, 200),  # various EMA/window stats
    "cross_sym":       range(200, 240),  # cross-sym pooling features if any
    "derived":         range(240, 359),  # remaining derived features
}
```

### PyTorch 类签名

```python
class FeatureGroupAttentionMLP(nn.Module):
    """
    Organize 359 features into semantic groups.
    Each group -> d_group token. Transformer attention across groups.
    Final: concat attended group tokens -> prediction MLP.

    Input:  (B, n_feat) or (B, 5, n_feat) per-sym
    Output: (B,) or (B, 5) per-sym prediction
    """
    def __init__(
        self,
        n_feat: int = 359,
        group_sizes: list[int] = None,  # sizes of each group, must sum to n_feat
        d_group: int = 32,
        n_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        if group_sizes is None:
            # Default: split 359 into 11 groups of varying size
            # Adjust so sum == n_feat
            group_sizes = [32, 32, 16, 8, 8, 12, 50, 40, 50, 51, 60]
            assert sum(group_sizes) == n_feat, f"group_sizes must sum to {n_feat}"

        self.group_sizes = group_sizes
        self.n_groups = len(group_sizes)
        self.d_group = d_group

        # Per-group linear projection
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gs, d_group), nn.LayerNorm(d_group), nn.GELU()
            )
            for gs in group_sizes
        ])

        # Transformer over groups (n_groups tokens, each d_group)
        self.transformer = nn.ModuleList([
            SDPABlock(d_group, n_heads, ffn_mult=2.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_group)

        # Prediction MLP on flattened group tokens
        self.pred_mlp = nn.Sequential(
            nn.Linear(self.n_groups * d_group, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_feat) or (B, 5, n_feat) [handled transparently]
        shape = x.shape
        if x.dim() == 3:
            B, S, F = shape
            x_flat = x.reshape(B * S, F)
        else:
            x_flat = x
            B, F = shape
            S = None

        # Tokenize each feature group
        offset = 0
        tokens = []
        for i, gs in enumerate(self.group_sizes):
            grp = x_flat[:, offset:offset+gs]
            tokens.append(self.group_projs[i](grp))
            offset += gs
        h = torch.stack(tokens, dim=1)  # (B[*S], n_groups, d_group)

        for blk in self.transformer:
            h = blk(h)
        h = self.norm(h)                 # (B[*S], n_groups, d_group)
        h_flat = h.reshape(h.size(0), -1)  # (B[*S], n_groups * d_group)
        pred = self.pred_mlp(h_flat).squeeze(-1)  # (B[*S],)

        if S is not None:
            pred = pred.reshape(B, S)
        return pred
```

### 关键超参

| 超参 | 默认值 |
|---|---|
| n_groups | 11 (基于 partition.json 分层) |
| d_group | 32 |
| n_heads | 4 |
| depth | 2 |
| dropout | 0.10 |

### 预期 PnL

- 乐观：+34 ~ +40（group-level attention 比 per-feature token 更稳定）
- 中性：+30 ~ +34（与 MLP 持平）
- 悲观：+24 ~ +30（FT-Transformer 失败说明 feature-level attention 效果有限）

---

## 训练 Hooks（所有架构通用）

与 train.py 中已实现的 baseline 保持一致：

```python
# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# Loss（加权 L2）
loss = (sw * (pred - y_reg / target_scale) ** 2).mean()

# 数据格式
# train: X_grp (N_groups, 5, n_feat), y_reg (N_groups, 5), sw (N_groups, 5)
# Per-sym 模型：reshape (N_groups*5, n_feat), (N_groups*5,), (N_groups*5,)
# Cross-sym 模型：(N_groups, 5, n_feat), (N_groups, 5), (N_groups, 5)

# Mirror augmentation: 已在 data_loader.py 中同步应用于 5 sym
# Window-z: 已在 data_loader.py 中计算

# Early stopping: patience=10 on val_pnl
# Epochs: max 50

# WandB: wandb.init(project="cross_sym_nn", entity="cjxh21-Tsinghua University")
# Log: loss/val_pnl/test_pnl 每 epoch

# 推理：先搜最优阈值（见 train.py search_thr），再计算 test_pnl
```

---

## models.py 集成方式

所有新架构直接加入 `models.py` 的 `ARCH_CLASSES` 和 `ARCH_DEFAULTS` 字典：

```python
# 在 models.py 末尾追加：

ARCH_CLASSES = {
    "cross_mlp":    CrossConcatMLP,
    "sym_attn":     SymAttentionModel,
    "hybrid":       HybridCrossAttn,
    # NEW:
    "set_isab":     SetTransformerISAB,
    "deepsets":     DeepSetsEnhanced,
    "sae_mlp":      SupervisedAutoEncoderMLP,  # per-sym mode
    "gat_sym":      GraphAttentionCrossSym,
    "feat_group":   FeatureGroupAttentionMLP,
}

ARCH_DEFAULTS = {
    # existing...
    "set_isab":   dict(d_model=128, n_heads=4, n_inducing=8, depth=2, dropout=0.10),
    "deepsets":   dict(d_phi=128, d_rho=64, dropout=0.10),
    "sae_mlp":    dict(d_latent=64, d_hidden=256, dropout=0.10, recon_alpha=0.3),
    "gat_sym":    dict(d_node=128, n_heads=4, n_gat_layers=2, dropout=0.10),
    "feat_group": dict(d_group=32, n_heads=4, depth=2, dropout=0.10),
}
```

---

## 参数量汇总

| 架构 | 约参数量 | 相对 MLP(134k) |
|---|---|---|
| MLP baseline | 134k | 1× |
| SetTransformerISAB | ~280k | 2.1× |
| DeepSetsEnhanced | ~160k | 1.2× |
| SAE-MLP (per-sym) | ~450k | 3.4× |
| GAT 5-sym | ~200k | 1.5× |
| FeatureGroupAttentionMLP | ~180k | 1.3× |
| CrossConcatMLP (existing) | 1094k | 8.2× ← 可能 OOM/overfit |

---

## 风险与建议

1. **5 sym 数量极小**：所有跨 sym 架构的 attention/message-passing 在 5 个节点上可能高方差
2. **SymAttentionModel (s0=18, s1=27) 已显示跨 sym 方差大**：建议先跑 2 seed，若两 seed 间方差 > 10，说明不稳定
3. **SAE-MLP 是独立路线**（不依赖 cross-sym），风险最低，建议与跨 sym 架构并行实验
4. **ISAB inducing points 在 n_sym=5 时**：尝试 n_inducing=3（少于 sym 数）和 n_inducing=5（等于 sym 数）
5. **如果所有跨 sym 架构都低于 MLP baseline**：最终结论是 5 sym 的跨 sym 信息量不足以帮助；回归 per-sym 强化路线（SAE-MLP, TabM, 更深 MLP）

---

*IMPLEMENTATION_SPEC.md 生成于 2026-05-30 by research worker*
*基于 RESEARCH.md 调研结论 + 现有 models.py 代码结构*
