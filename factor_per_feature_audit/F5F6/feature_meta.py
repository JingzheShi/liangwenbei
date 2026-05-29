"""F5 + F6 feature identity metadata.

For each of 141 features we provide:
  name, name_cn, name_en_full, family, raw_or_derived, formula_latex,
  code_loc, numpy_pseudo, physical_meaning, nan_handling (formula_level)

The NaN pipeline-level descriptions (NN, LGB) and dim_in_X are filled in
at audit-emit time from the cache feat_names index.
"""
from __future__ import annotations
from typing import Dict


# ---------------------------------------------------------------------------
# F5 — Window statistics (67 dims)
# ---------------------------------------------------------------------------

# Base columns used by dualz (the 37 stage-1 dualz features).
# (column, CN, EN-full)
_DUALZ_BASE: Dict[str, tuple] = {
    "spread1":      ("一档价差",          "Level-1 spread (ask1-bid1)"),
    "spread5":      ("五档价差",          "Level-5 spread (ask5-bid5)"),
    "spread10":     ("十档价差",          "Level-10 spread (ask10-bid10)"),
    "cumspread":    ("累计十档价差",      "Cumulative 10-level spread"),
    "bid1":         ("一档买价",          "Best bid price"),
    "ask1":         ("一档卖价",          "Best ask price"),
    "bid_mean":     ("十档买价均值",      "Mean of 10-level bid prices"),
    "ask_mean":     ("十档卖价均值",      "Mean of 10-level ask prices"),
    "midprice1":    ("一档中间价",        "Level-1 midprice (bid1+ask1)/2"),
    "midprice2":    ("二档中间价",        "Level-2 midprice (bid2+ask2)/2"),
    "midprice5":    ("五档中间价",        "Level-5 midprice"),
    "midprice10":   ("十档中间价",        "Level-10 midprice"),
    "bsize1":       ("一档买量",          "Best-bid size"),
    "bsize_mean":   ("十档买量均值",      "Mean of 10-level bid sizes"),
    "totalbsize":   ("买侧总量",          "Total bid size (sum 10 levels)"),
    "asize1":       ("一档卖量",          "Best-ask size"),
    "asize_mean":   ("十档卖量均值",      "Mean of 10-level ask sizes"),
    "totalasize":   ("卖侧总量",          "Total ask size (sum 10 levels)"),
    "volume_delta": ("成交量增量",        "Volume traded since previous snapshot"),
    "amount_delta": ("成交额增量",        "Notional traded since previous snapshot"),
    "imbalance":    ("一档量不平衡",      "Order-book imbalance (bsize1-asize1)/(bsize1+asize1)"),
    "lb_intst":     ("挂买强度",          "Limit-bid order intensity"),
    "la_intst":     ("挂卖强度",          "Limit-ask order intensity"),
    "mb_intst":     ("市价买强度",        "Market-bid order intensity"),
    "ma_intst":     ("市价卖强度",        "Market-ask order intensity"),
    "cb_intst":     ("撤买强度",          "Cancel-bid order intensity"),
    "ca_intst":     ("撤卖强度",          "Cancel-ask order intensity"),
    "lb_acc":       ("挂买累积",          "Limit-bid cumulative count"),
    "la_acc":       ("挂卖累积",          "Limit-ask cumulative count"),
    "mb_acc":       ("市价买累积",        "Market-bid cumulative count"),
    "ma_acc":       ("市价卖累积",        "Market-ask cumulative count"),
    "cb_acc":       ("撤买累积",          "Cancel-bid cumulative count"),
    "ca_acc":       ("撤卖累积",          "Cancel-ask cumulative count"),
    "bid_diff1":    ("一档买价一阶差分",  "Δbid1 (this tick - prev tick)"),
    "ask_diff1":    ("一档卖价一阶差分",  "Δask1 (this tick - prev tick)"),
    "bid_diff5":    ("五档买价一阶差分",  "Δbid5"),
    "ask_diff5":    ("五档卖价一阶差分",  "Δask5"),
}

# QRANK base columns (20 stage-2 qrank features).
_QRANK_BASE: Dict[str, tuple] = {
    "spread1":      ("一档价差",          "Level-1 spread"),
    "spread5":      ("五档价差",          "Level-5 spread"),
    "spread10":     ("十档价差",          "Level-10 spread"),
    "cumspread":    ("累计十档价差",      "Cumulative 10-level spread"),
    "amount_delta": ("成交额增量",        "Trade notional delta"),
    "volume_delta": ("成交量增量",        "Trade volume delta"),
    "imbalance":    ("一档量不平衡",      "OB imbalance"),
    "totalbsize":   ("买侧总量",          "Total bid size"),
    "totalasize":   ("卖侧总量",          "Total ask size"),
    "lb_intst":     ("挂买强度",          "Limit-bid intensity"),
    "la_intst":     ("挂卖强度",          "Limit-ask intensity"),
    "mb_intst":     ("市价买强度",        "Market-bid intensity"),
    "ma_intst":     ("市价卖强度",        "Market-ask intensity"),
    "cb_intst":     ("撤买强度",          "Cancel-bid intensity"),
    "ca_intst":     ("撤卖强度",          "Cancel-ask intensity"),
    "lb_acc":       ("挂买累积",          "Limit-bid cumulative"),
    "la_acc":       ("挂卖累积",          "Limit-ask cumulative"),
    "mb_acc":       ("市价买累积",        "Market-bid cumulative"),
    "ma_acc":       ("市价卖累积",        "Market-ask cumulative"),
    "midprice":     ("中间价",            "Midprice (bid1+ask1)/2"),
}


def f5_records():
    recs = []
    # --- dualz (37) ---
    KS_FAIL = {"dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5"}
    formula_dz = (
        r"\mathrm{dualz}_t(c) = \frac{c_t - \mu_{20}(c)}{\sigma_{20}(c)+\varepsilon} "
        r"- \frac{c_t - \mu_{100}(c)}{\sigma_{100}(c)+\varepsilon}"
    )
    pseudo_dz = (
        "seg20 = X[:, -20:, col]; seg100 = X[:, -100:, col]\n"
        "z_short = (X[:,-1,col] - seg20.mean(-1)) / (seg20.std(-1)+EPS)\n"
        "z_long  = (X[:,-1,col] - seg100.mean(-1)) / (seg100.std(-1)+EPS)\n"
        "dualz = np.where(np.isfinite(z_short-z_long), z_short-z_long, 0.0)"
    )
    physical_dz_template = (
        "短窗(20) z-score 减长窗(100) z-score。两 z 相减消除了 base column 的全局水平差异和方差差异，"
        "使得跨股(sym)、跨日的 z-score 单位可比 — 这是跨股泛化的几何基础。"
        "信号本身衡量【最近 20 tick 异常程度】相对于【最近 100 tick 基线异常程度】的偏离。"
    )
    for c, (cn, en) in _DUALZ_BASE.items():
        name = f"dualz_{c}"
        in_ks = name in KS_FAIL
        notes = ""
        if in_ks:
            # diff features have heavier ask-side distribution shift
            notes = (
                f"KS-drop 11 之一。{c} 一阶差分高度集中在 0 附近(price-tick 量化)，长短 z 差极易被 outlier 主导；"
                "ask 侧 distribution shift 比 bid 侧更明显(主办方数据中 ask 报价更新更稀疏) → "
                "训练时被 KS 过滤剔除。"
            )
        recs.append({
            "name": name,
            "name_cn": f"双窗口 z-score - {cn}",
            "name_en_full": f"Dual-window z-score on {en}",
            "family": "F5",
            "raw_or_derived": "derived",
            "formula_latex": formula_dz,
            "code_loc": "fast_features_batch.py:275-291 (compute_stage1_batch -> dual_z block)",
            "numpy_pseudo": pseudo_dz,
            "physical_meaning": physical_dz_template,
            "nan_handling": {
                "formula_level": (
                    "EPS=1e-12 加在分母防 0；np.where(np.isfinite(dual), dual, 0.0) 兜底。"
                    "因此恒为 finite (不会出 NaN/Inf)。"
                ),
                "nn_pipeline": "window-z: nanmean/nanstd → nan_to_num(0) → clip(±10) (rerun_13rows.py:271-285)",
                "lgb_pipeline": "不动；LGB 内置 missing routing（实际上 dualz 几乎无 NaN，故无影响）。",
            },
            "in_ks_drop_11": in_ks,
            "notes": notes,
        })

    # --- qrank_W100 (20) ---
    KS_FAIL_Q = {"qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10", "qrank_W100_cumspread"}
    formula_q = r"\mathrm{qrank}_t^W(c) = \frac{1}{W}\sum_{i=t-W+1}^{t} \mathbb{1}[c_i \le c_t]"
    pseudo_q = (
        "seg = X[:, -100:, col].astype(np.float32)\n"
        "last = seg[:, -1:]\n"
        "qrank = (seg <= last).mean(axis=-1)"
    )
    physical_q = (
        "当前值在过去 100 tick 中的分位（empirical rank ∈ [1/W, 1]）。"
        "对绝对水平不敏感，对单调变换(e.g. log)不变；天然 sym-agnostic。"
        "但对 tick-quantized 列(spread 常 = 1 tick)，几乎所有 tick 都等于 last → qrank ≈ 1 → 退化为常数(死特征)。"
    )
    for c, (cn, en) in _QRANK_BASE.items():
        name = f"qrank_W100_{c}"
        in_ks = name in KS_FAIL_Q
        notes = ""
        if in_ks:
            notes = (
                "KS-drop 11 之一。spread 列在 tick-quantized 市场里大多 = 1 tick (恒值)，"
                "(seg <= last_val) 几乎 100% 为 True → qrank 退化为常数 1 → 训练时 KS 检测出与目标分布无关被剔除。"
            )
        recs.append({
            "name": name,
            "name_cn": f"100-tick 分位 rank - {cn}",
            "name_en_full": f"100-tick empirical quantile rank of {en}",
            "family": "F5",
            "raw_or_derived": "derived",
            "formula_latex": formula_q,
            "code_loc": "fast_features_batch.py:354-372 (compute_stage2_batch -> qrank block)",
            "numpy_pseudo": pseudo_q,
            "physical_meaning": physical_q,
            "nan_handling": {
                "formula_level": (
                    "比较运算返回 bool，mean 后必定 finite ∈ [1/W, 1]，无 NaN 风险。"
                ),
                "nn_pipeline": "window-z 或 raw 路径；nan_to_num(0) (rerun_13rows.py:271-285)",
                "lgb_pipeline": "不动；LGB 内置 missing routing（无 NaN）。",
            },
            "in_ks_drop_11": in_ks,
            "notes": notes,
        })

    # --- mid_ewma_resid_a0.05 (1) ---
    recs.append({
        "name": "mid_ewma_resid_a0.05",
        "name_cn": "中间价 EWMA 残差 α=0.05",
        "name_en_full": "Midprice EWMA residual (α=0.05)",
        "family": "F5",
        "raw_or_derived": "derived",
        "formula_latex": r"\mathrm{resid}_t = m_t - \mathrm{EWMA}_{\alpha=0.05}(m)_t,\quad y_t = \alpha m_t + (1-\alpha)y_{t-1}",
        "code_loc": "fast_features_batch.py:499-504 (compute_stage3_batch -> ewma_resid block)",
        "numpy_pseudo": (
            "ewma = _ewma_last_batch(midprice, (0.05,))[:, 0]\n"
            "resid = midprice[:, -1] - ewma\n"
            "resid = np.clip(np.where(np.isfinite(resid), resid, 0.0), -0.05, 0.05)"
        ),
        "physical_meaning": (
            "当前中间价对其长期 EWMA 趋势(α=0.05 → 半衰约 14 tick)的偏离。"
            "正值 = 短期偏强 / 高于慢均线；负值 = 短期偏弱。"
            "经验阈 ±5% clip，剔除极端价跳。"
        ),
        "nan_handling": {
            "formula_level": (
                "lfilter EWMA 在 mid 数值上恒输出 finite；"
                "np.where(np.isfinite, resid, 0.0) + clip(-0.05, 0.05) 双重兜底。"
            ),
            "nn_pipeline": "window-z 或 raw + nan_to_num(0)",
            "lgb_pipeline": "不动；几乎无 NaN。",
        },
        "in_ks_drop_11": False,
        "notes": "",
    })

    # --- adapt_mom_W (3) ---
    for W in (20, 50, 100):
        recs.append({
            "name": f"adapt_mom_W{W}",
            "name_cn": f"自适应动量 W={W}",
            "name_en_full": f"Volatility-adaptive momentum, window={W}",
            "family": "F5",
            "raw_or_derived": "derived",
            "formula_latex": (
                r"\mathrm{adapt\_mom}_t^W = \frac{m_t - m_{t-W}}{\sigma_W(m)+\varepsilon}"
            ),
            "code_loc": "fast_features_batch.py:615-622 (compute_stage5_batch -> adapt_mom block)",
            "numpy_pseudo": (
                f"last = mid[:, -1]; ref = mid[:, -{W}]\n"
                f"sd = mid[:, -{W}:].std(axis=-1) + EPS\n"
                "v = np.clip(np.where(np.isfinite((last-ref)/sd), (last-ref)/sd, 0.0), -10, 10)"
            ),
            "physical_meaning": (
                f"过去 {W} tick 的中间价位移除以同窗口波动率 → 动量信号的 σ-规整版。"
                "波动率高的时段同等位移会得到更小的 |signal|，所以本质是【风险调整后的动量】。"
                "Sym-agnostic：σ 自动按 sym 的波动尺度归一化。"
            ),
            "nan_handling": {
                "formula_level": "EPS 防 0；np.where(isfinite) 兜底；clip(-10,10).",
                "nn_pipeline": "window-z 或 raw + nan_to_num(0)",
                "lgb_pipeline": "不动；几乎无 NaN。",
            },
            "in_ks_drop_11": False,
            "notes": "",
        })

    # --- spread_reg_W (3) ---
    for W in (20, 50, 100):
        recs.append({
            "name": f"spread_reg_W{W}",
            "name_cn": f"价差稳健 z 分位 W={W}",
            "name_en_full": f"Robust spread regime z-score, window={W}",
            "family": "F5",
            "raw_or_derived": "derived",
            "formula_latex": (
                r"\mathrm{spread\_reg}_t^W = \frac{s_t - \mathrm{median}_W(s)}{\mathrm{IQR}_W(s)}"
            ),
            "code_loc": "fast_features_batch.py:689-699 (compute_stage5_batch -> spread regime block)",
            "numpy_pseudo": (
                f"seg = spread1[:, -{W}:]\n"
                "med = np.median(seg, -1); iqr = np.quantile(seg, .75, -1) - np.quantile(seg, .25, -1)\n"
                "denom = np.where(iqr > EPS, iqr, np.inf)\n"
                "v = np.clip((spread1[:, -1] - med) / denom, -10, 10)"
            ),
            "physical_meaning": (
                f"当前 spread 相对最近 {W} tick 的稳健中位数偏离，用 IQR 归一化。"
                "正 = 价差异常加宽(流动性收紧)，负 = 异常窄。"
                "IQR=0(spread 在窗口内全等)时 denom=∞ → v=0，避免 outlier 爆炸；这种 case 在小盘股很常见。"
            ),
            "nan_handling": {
                "formula_level": "iqr==0 → denom=inf → 0；np.where(isfinite) 兜底；clip(±10).",
                "nn_pipeline": "window-z 或 raw + nan_to_num(0)",
                "lgb_pipeline": "不动；几乎无 NaN。",
            },
            "in_ks_drop_11": False,
            "notes": "",
        })

    # --- trade_pers_W (3) ---
    for W in (20, 50, 100):
        recs.append({
            "name": f"trade_pers_W{W}",
            "name_cn": f"成交方向持续度 W={W}",
            "name_en_full": f"Trade-direction persistence, window={W}",
            "family": "F5",
            "raw_or_derived": "derived",
            "formula_latex": (
                r"\mathrm{trade\_pers}_t^W = \mathrm{Corr}\big(\mathrm{sign}(\Delta m)_{t-W:t},\ \mathrm{sign}(\Delta m)_{t-W-1:t-1}\big)"
            ),
            "code_loc": "fast_features_batch.py:701-719 (compute_stage5_batch -> trade_pers block)",
            "numpy_pseudo": (
                "sign_dm = np.sign(np.diff(mid, prepend=mid[:,:1]))\n"
                "sign_lag = np.roll(sign_dm, 1, -1); sign_lag[:,0]=0\n"
                f"x = sign_dm[:, -{W}:]; y = sign_lag[:, -{W}:]\n"
                "v = pearson_corr(x, y) → clip(-1, 1)"
            ),
            "physical_meaning": (
                "中间价方向序列 sign(Δm) 与其滞后 1 tick 的 Pearson 相关 = 一阶自相关。"
                "正 = 方向连续(动量市场)；负 = 方向反转(均值回归 / market-making)；"
                "0 = 随机游走。"
            ),
            "nan_handling": {
                "formula_level": (
                    "denom = sqrt(sxx*syy)+EPS 防 0；np.where(isfinite, v, 0) 兜底；clip(±1)."
                ),
                "nn_pipeline": "window-z 或 raw + nan_to_num(0)",
                "lgb_pipeline": "不动；几乎无 NaN。",
            },
            "in_ks_drop_11": False,
            "notes": "",
        })

    return recs


# ---------------------------------------------------------------------------
# F6 — Asymmetry (74 dims)
# ---------------------------------------------------------------------------

def f6_records():
    recs = []

    # --- bid_rate{k}, ask_rate{k}, bsize_rate{k}, asize_rate{k}  (40 raw) ---
    rate_meta = [
        ("bid_rate",   "买价滑动平均变化率",  "Sliding-mean rate-of-change of bid_{k}"),
        ("ask_rate",   "卖价滑动平均变化率",  "Sliding-mean rate-of-change of ask_{k}"),
        ("bsize_rate", "买量滑动平均变化率",  "Sliding-mean rate-of-change of bsize_{k}"),
        ("asize_rate", "卖量滑动平均变化率",  "Sliding-mean rate-of-change of asize_{k}"),
    ]
    for prefix, cn, en in rate_meta:
        for k in range(1, 11):
            recs.append({
                "name": f"{prefix}{k}",
                "name_cn": f"{cn} 第{k}档",
                "name_en_full": f"{en.replace('{k}', str(k))}",
                "family": "F6",
                "raw_or_derived": "raw",
                "formula_latex": (
                    r"\mathrm{rate}_t^{(k)} = \frac{\bar{x}_{[t-W:t]}^{(k)} - \bar{x}_{[t-2W:t-W]}^{(k)}}"
                    r"{|\bar{x}_{[t-2W:t-W]}^{(k)}|+\varepsilon}"
                ),
                "code_loc": "fast_features_batch.py:N/A (raw, 主办方直接给的 38 个原始字段之一)",
                "numpy_pseudo": (
                    f"# raw input: provided by host as column 'idx({prefix}{k})'\n"
                    f"feat = raw_input[:, name2idx['{prefix}{k}']]"
                ),
                "physical_meaning": (
                    f"主办方给的 {prefix}{k}：第 {k} 档{'买'if 'bid' in prefix or 'bsize' in prefix else '卖'}"
                    f"侧{'价格'if 'rate' in prefix and 'size' not in prefix else '挂量'}的"
                    "滑动平均变化率（具体窗口主办方未披露，经验上 ~10-20 tick）。"
                    "买卖侧分开 → 天然不对称：bid_rate 反映吃买盘速度，ask_rate 反映吃卖盘速度。"
                ),
                "nan_handling": {
                    "formula_level": "raw 字段，主办方上游做了 +ε denom 兜底；输入侧基本无 NaN。",
                    "nn_pipeline": "window-z: nanmean/nanstd → nan_to_num(0) → clip(±10)",
                    "lgb_pipeline": "不动；LGB 内置 missing routing。",
                },
                "in_ks_drop_11": False,
                "notes": "",
            })

    # --- gofi (30) ---
    formula_g = (
        r"e_t^{(k)} = \big[\mathbb{1}(b_t > b_{t-1})\,bs_t - \mathbb{1}(b_t<b_{t-1})\,bs_{t-1} + \mathbb{1}(b_t=b_{t-1})(bs_t-bs_{t-1})\big]"
        r"\,+\,\big[\mathbb{1}(a_t>a_{t-1})\,as_{t-1} - \mathbb{1}(a_t<a_{t-1})\,as_t - \mathbb{1}(a_t=a_{t-1})(as_t-as_{t-1})\big],"
        r"\quad \mathrm{gofi}_W^{(k)} = \sum_{i=t-W+1}^{t} e_i^{(k)}"
    )
    pseudo_g = (
        "for each level k in 1..10:\n"
        "  bid_up = (b > b_lag) * bs; bid_dn = (b < b_lag) * bs_lag\n"
        "  bid_eq = (b == b_lag) * (bs - bs_lag)\n"
        "  ask_up = (a > a_lag) * as_lag; ask_dn = (a < a_lag) * a_size\n"
        "  ask_eq = (a == a_lag) * (a_size - as_lag)\n"
        "  e = (bid_up - bid_dn + bid_eq) - (ask_up - ask_dn - ask_eq)\n"
        "  gofi_W = e[:, -W:].sum(-1)"
    )
    physical_g = (
        "Generalized Order-Flow Imbalance：MLOFI 只算【价格档移动+主动成交】产生的净挂量变化，"
        "GOFI 额外加上【价格档不变但挂量增减】那一项 (bid_eq, ask_eq)，更完整地刻画 cancel/refill。"
        "这就是 F6 ↔ F2 |corr|=0.176 的根源(六族中第二大共线性)。"
        "正 = 净 bid-side 增 → 看多；负 = 净 ask-side 增 → 看空。"
    )
    for W in (5, 20, 60):
        for k in range(1, 11):
            recs.append({
                "name": f"gofi_W{W}_lvl{k}",
                "name_cn": f"GOFI 广义订单流不平衡 W={W} 第{k}档",
                "name_en_full": f"Generalized Order-Flow Imbalance, window={W}, level={k}",
                "family": "F6",
                "raw_or_derived": "derived",
                "formula_latex": formula_g,
                "code_loc": "fast_features_batch.py:395-429 (compute_stage2_batch -> gofi block)",
                "numpy_pseudo": pseudo_g,
                "physical_meaning": physical_g,
                "nan_handling": {
                    "formula_level": (
                        "差分项可能产生 inf (size cast 异常)；np.where(isfinite, v, 0.0) 兜底；"
                        "无 clip，但典型值在 ±1e6 量级。"
                    ),
                    "nn_pipeline": (
                        "window-z 强制规整 → 不会爆炸；nan_to_num(0); clip(±10)。"
                        "无 window-z 时仅 nan_to_num(0)。"
                    ),
                    "lgb_pipeline": "不动；LGB 内置 missing routing。",
                },
                "in_ks_drop_11": False,
                "notes": "",
            })

    # --- liq_asym_top5_W (4) ---
    KS_FAIL_LA = {"liq_asym_top5_W5"}
    formula_la = (
        r"\mathrm{liq\_asym\_top5}_t^W = \log\frac{1 + \sum_{i=t-W+1}^{t}\sum_{k=1}^{5} bsize_i^{(k)}}{1 + \sum_{i=t-W+1}^{t}\sum_{k=1}^{5} asize_i^{(k)}}"
    )
    pseudo_la = (
        "bid_top5 = sum(X[:,:,bsize_k] for k=1..5)  # (N, T)\n"
        "ask_top5 = sum(X[:,:,asize_k] for k=1..5)\n"
        "sb = bid_top5[:, -W:].sum(-1); sa = ask_top5[:, -W:].sum(-1)\n"
        "v = np.clip(np.log((1.+sb)/(1.+sa)), -10, 10)"
    )
    physical_la = (
        "买卖 top-5 挂量在过去 W tick 累计的 log-比 → 流动性不对称 (log-odds)。"
        "正 = 买盘厚于卖盘 → 看多；负 = 卖盘厚 → 看空。"
        "+1 偏置保证数值稳定 (size 为 0 时也 well-defined)。"
    )
    for W in (5, 20, 50, 100):
        name = f"liq_asym_top5_W{W}"
        in_ks = name in KS_FAIL_LA
        notes = ""
        if in_ks:
            notes = (
                "KS-drop 11 之一。W=5 太短 → log-ratio 在 5 tick 内噪声极大、自相关高、跨股分布发散 → "
                "训练时被 KS 滤除。W=20/50/100 更稳，全部保留。"
            )
        recs.append({
            "name": name,
            "name_cn": f"Top-5 流动性对数不对称 W={W}",
            "name_en_full": f"Top-5 liquidity log-asymmetry, window={W}",
            "family": "F6",
            "raw_or_derived": "derived",
            "formula_latex": formula_la,
            "code_loc": "fast_features_batch.py:721-733 (compute_stage5_batch -> liq_asym block)",
            "numpy_pseudo": pseudo_la,
            "physical_meaning": physical_la,
            "nan_handling": {
                "formula_level": "log((1+sb)/(1+sa)) — 分子分母 ≥ 1，log 始终 finite；np.where + clip(±10) 兜底。",
                "nn_pipeline": "window-z 或 raw + nan_to_num(0)",
                "lgb_pipeline": "不动；几乎无 NaN。",
            },
            "in_ks_drop_11": in_ks,
            "notes": notes,
        })

    return recs


def build_all_records():
    return f5_records() + f6_records()
