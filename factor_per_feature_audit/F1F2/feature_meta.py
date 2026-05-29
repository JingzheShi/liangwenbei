"""Per-feature metadata: identity, formulas, code_loc, numpy_pseudo, physical_meaning, NaN handling.

For F1 (105) + F2 (50) = 155 features.

The full 154 raw layout (from build_schemeP_cache.py / fast_features_batch.py) at
last-tick is reproduced here as RAW_154_INDEX so we can answer "in raw 154 index = ?".

The 216 derived feature names follow the order in fast_features_batch.all_feature_names()
appended to the 154 raw, so X[:, 154+i] = derived[i].
"""
from __future__ import annotations

from typing import Dict, List

# === Raw 154 layout (from schemeP_feat_names.txt rows 0..153) ===
# This matches build_schemeP_cache.RAW_COLS.
RAW_154 = [
    "open", "high", "low", "close", "volume_delta", "amount_delta",
] + [f"bid{k}" for k in range(1, 11)] + [f"bsize{k}" for k in range(1, 11)] \
  + [f"ask{k}" for k in range(1, 11)] + [f"asize{k}" for k in range(1, 11)] \
  + ["avgbid", "avgask", "totalbsize", "totalasize"] \
  + ["lb_intst", "la_intst", "mb_intst", "ma_intst", "cb_intst", "ca_intst"] \
  + ["lb_ind", "la_ind", "mb_ind", "ma_ind", "cb_ind", "ca_ind"] \
  + ["lb_acc", "la_acc", "mb_acc", "ma_acc", "cb_acc", "ca_acc"] \
  + [f"midprice{k}" for k in range(1, 11)] \
  + [f"spread{k}" for k in range(1, 11)] \
  + [f"bid_diff{k}" for k in range(1, 11)] \
  + [f"ask_diff{k}" for k in range(1, 11)] \
  + ["bid_mean", "ask_mean", "bsize_mean", "asize_mean", "cumspread", "imbalance"] \
  + [f"bid_rate{k}" for k in range(1, 11)] + [f"ask_rate{k}" for k in range(1, 11)] \
  + [f"bsize_rate{k}" for k in range(1, 11)] + [f"asize_rate{k}" for k in range(1, 11)]

RAW_154_INDEX: Dict[str, int] = {n: i for i, n in enumerate(RAW_154)}


# =============================================================================
# F1 metadata builder
# =============================================================================

KS_FAIL = {
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
}


def _nan_handling_raw() -> Dict[str, str]:
    return {
        "formula_level": "raw 字段，无公式层兜底（直接来自主办方 parquet 的最后一 tick）",
        "nn_pipeline": "wz=True 路径: nanmean/nanstd→nan_to_num(nan=0, +inf=10, -inf=-10)→clip(±10); wz=False: nan_to_num(nan=0)",
        "lgb_pipeline": "完全不动，LightGBM 内置 missing routing",
    }


def _nan_handling_derived(extra: str = "") -> Dict[str, str]:
    base = "fast_features_batch 输出后立刻 np.where(isfinite, x, 0.0) → 整体 NaN 已被 0 替换"
    if extra:
        base = f"{base}；{extra}"
    return {
        "formula_level": base,
        "nn_pipeline": "wz=True: nanmean/nanstd→nan_to_num→clip(±10) （derived 已被 0 填充，wz 实际只是再标准化）；wz=False: nan_to_num(nan=0)",
        "lgb_pipeline": "完全不动，但 derived 已经无 NaN，所以 missing routing 不触发",
    }


def f1_meta() -> List[Dict]:
    """105 F1 features. raw_or_derived + formula_latex + numpy_pseudo + physical_meaning."""
    items: List[Dict] = []

    # ---- (1) OHLC 4 raw ----
    for n, cn, en in [
        ("open", "本 tick 开盘价", "Open price of current tick (3-second bar)"),
        ("high", "本 tick 最高价", "High price of current tick"),
        ("low", "本 tick 最低价", "Low price of current tick"),
        ("close", "本 tick 收盘价", "Close price of current tick"),
    ]:
        items.append({
            "name": n, "name_cn": cn, "name_en_full": en,
            "raw_or_derived": "raw",
            "raw_idx_in_154": RAW_154_INDEX[n],
            "formula_latex": rf"\text{{{n}}}_t \;=\; \text{{raw input (3s OHLC)}}",
            "code_loc": "build_schemeP_cache.py:107 raw_last = X3d[:, -1, :]",
            "numpy_pseudo": f"{n} = raw_input[:, {RAW_154_INDEX[n]}]",
            "physical_meaning": f"主办方给出的 3 秒级 K 线 {cn}；这 4 列共同刻画了短期价格振幅与瞬时方向。",
            "nan_handling": _nan_handling_raw(),
        })

    # ---- (2) bid1..bid10 / ask1..ask10 raw ----
    for side, side_cn, side_en in [("bid", "买盘", "bid side"), ("ask", "卖盘", "ask side")]:
        for k in range(1, 11):
            n = f"{side}{k}"
            items.append({
                "name": n, "name_cn": f"第 {k} 档{side_cn}报价",
                "name_en_full": f"Level-{k} {side_en} price",
                "raw_or_derived": "raw",
                "raw_idx_in_154": RAW_154_INDEX[n],
                "formula_latex": rf"\text{{{n}}}_t = \text{{level-{k} {side_en} quote}}",
                "code_loc": "build_schemeP_cache.py:107 raw_last (主办方原始字段)",
                "numpy_pseudo": f"{n} = raw_input[:, {RAW_154_INDEX[n]}]  # level-{k} {side_en} quote",
                "physical_meaning": f"第 {k} 档{side_cn}的报价（{side_en} L{k}）；构造 LOB 价格梯度与挂单深度的基础维度。",
                "nan_handling": _nan_handling_raw(),
            })

    # ---- (3) bsize1..bsize10 / asize1..asize10 raw ----
    for side, side_cn, side_en in [("bsize", "买盘挂单量", "bid-size"), ("asize", "卖盘挂单量", "ask-size")]:
        for k in range(1, 11):
            n = f"{side}{k}"
            items.append({
                "name": n, "name_cn": f"第 {k} 档{side_cn}",
                "name_en_full": f"Level-{k} {side_en}",
                "raw_or_derived": "raw",
                "raw_idx_in_154": RAW_154_INDEX[n],
                "formula_latex": rf"\text{{{n}}}_t = \text{{level-{k} {side_en}}}",
                "code_loc": "build_schemeP_cache.py:107 raw_last",
                "numpy_pseudo": f"{n} = raw_input[:, {RAW_154_INDEX[n]}]  # {side_en} L{k}",
                "physical_meaning": f"第 {k} 档{side_cn}（{side_en} L{k}）的挂单总量；衡量该价位被动流动性。",
                "nan_handling": _nan_handling_raw(),
            })

    # ---- (4) avgbid / avgask raw ----
    items.append({
        "name": "avgbid", "name_cn": "买盘均价",
        "name_en_full": "Volume-weighted bid quote average across 10 levels",
        "raw_or_derived": "raw",
        "raw_idx_in_154": RAW_154_INDEX["avgbid"],
        "formula_latex": r"\text{avgbid}_t \;=\; \frac{\sum_{k=1}^{10} \text{bid}_k \cdot \text{bsize}_k}{\sum_{k=1}^{10} \text{bsize}_k}\;(\text{exchange-provided})",
        "code_loc": "build_schemeP_cache.py:107 raw_last",
        "numpy_pseudo": "avgbid = raw_input[:, 46]  # exchange-provided",
        "physical_meaning": "交易所计算的 10 档买盘加权均价，等价于‘卖出全部 10 档可成交的平均价’，衡量买盘综合价位中心。",
        "nan_handling": _nan_handling_raw(),
    })
    items.append({
        "name": "avgask", "name_cn": "卖盘均价",
        "name_en_full": "Volume-weighted ask quote average across 10 levels",
        "raw_or_derived": "raw",
        "raw_idx_in_154": RAW_154_INDEX["avgask"],
        "formula_latex": r"\text{avgask}_t \;=\; \frac{\sum_{k=1}^{10} \text{ask}_k \cdot \text{asize}_k}{\sum_{k=1}^{10} \text{asize}_k}",
        "code_loc": "build_schemeP_cache.py:107 raw_last",
        "numpy_pseudo": "avgask = raw_input[:, 47]",
        "physical_meaning": "10 档卖盘加权均价，与 avgbid 配对刻画买卖双方综合价格中心。",
        "nan_handling": _nan_handling_raw(),
    })

    # ---- (5) totalbsize / totalasize ----
    items.append({
        "name": "totalbsize", "name_cn": "总买盘挂单量",
        "name_en_full": "Total bid depth across 10 levels",
        "raw_or_derived": "raw",
        "raw_idx_in_154": RAW_154_INDEX["totalbsize"],
        "formula_latex": r"\text{totalbsize}_t = \sum_{k=1}^{10}\text{bsize}_k",
        "code_loc": "build_schemeP_cache.py:107 raw_last (主办方提供，与 sum(bsize_k) 应一致)",
        "numpy_pseudo": "totalbsize = bsize1 + bsize2 + ... + bsize10",
        "physical_meaning": "10 档买方流动性合计；体现接盘力量总规模。",
        "nan_handling": _nan_handling_raw(),
    })
    items.append({
        "name": "totalasize", "name_cn": "总卖盘挂单量",
        "name_en_full": "Total ask depth across 10 levels",
        "raw_or_derived": "raw",
        "raw_idx_in_154": RAW_154_INDEX["totalasize"],
        "formula_latex": r"\text{totalasize}_t = \sum_{k=1}^{10}\text{asize}_k",
        "code_loc": "build_schemeP_cache.py:107 raw_last",
        "numpy_pseudo": "totalasize = asize1 + asize2 + ... + asize10",
        "physical_meaning": "10 档卖方流动性合计；与 totalbsize 配合得到 imbalance 信号的基础量。",
        "nan_handling": _nan_handling_raw(),
    })

    # ---- (6) midprice 1..10 ----
    for k in range(1, 11):
        n = f"midprice{k}"
        items.append({
            "name": n, "name_cn": f"第 {k} 档中间价",
            "name_en_full": f"Level-{k} midprice = (bid_k + ask_k)/2",
            "raw_or_derived": "raw",
            "raw_idx_in_154": RAW_154_INDEX[n],
            "formula_latex": rf"\text{{midprice}}_k = (\text{{bid}}_k + \text{{ask}}_k)/2",
            "code_loc": "build_schemeP_cache.py:107 raw_last (主办方给出)",
            "numpy_pseudo": f"midprice{k} = (bid{k} + ask{k}) / 2",
            "physical_meaning": f"第 {k} 档买一卖一的算术平均价；尤其 midprice1 是预测目标的载体（label 基于未来 midprice1 变动）。",
            "nan_handling": _nan_handling_raw(),
        })

    # ---- (7) spread 1..10 ----
    for k in range(1, 11):
        n = f"spread{k}"
        items.append({
            "name": n, "name_cn": f"第 {k} 档买卖价差",
            "name_en_full": f"Level-{k} bid-ask spread",
            "raw_or_derived": "raw",
            "raw_idx_in_154": RAW_154_INDEX[n],
            "formula_latex": rf"\text{{spread}}_k = \text{{ask}}_k - \text{{bid}}_k",
            "code_loc": "build_schemeP_cache.py:107 raw_last",
            "numpy_pseudo": f"spread{k} = ask{k} - bid{k}",
            "physical_meaning": f"第 {k} 档买卖价差，是流动性成本与做市意愿的直接刻画。",
            "nan_handling": _nan_handling_raw(),
        })

    # ---- (8) bid_diff k, ask_diff k ----
    for side, side_cn in [("bid", "买"), ("ask", "卖")]:
        for k in range(1, 11):
            n = f"{side}_diff{k}"
            items.append({
                "name": n, "name_cn": f"第 {k}-1 档{side_cn}价价差",
                "name_en_full": f"Level-{k} minus level-{k-1 if k>1 else 0} {side} price diff",
                "raw_or_derived": "raw",
                "raw_idx_in_154": RAW_154_INDEX[n],
                "formula_latex": rf"\text{{{side}\_diff}}_k = \text{{{side}}}_k - \text{{{side}}}_{{k-1}}",
                "code_loc": "build_schemeP_cache.py:107 raw_last",
                "numpy_pseudo": f"{side}_diff{k} = {side}{k} - {side}{k-1 if k>1 else 1}  # 内档间报价跨度",
                "physical_meaning": f"第 {k} 档与上一档{side_cn}盘报价的跨度，刻画 LOB {'买' if side=='bid' else '卖'}盘的价格台阶密度（密=深度好）。",
                "nan_handling": _nan_handling_raw(),
            })

    # ---- (9) bid_mean / ask_mean / bsize_mean / asize_mean ----
    for n, cn, en_full, formula, ps in [
        ("bid_mean", "10 档买价均值", "Mean of 10 bid prices", r"\frac{1}{10}\sum_{k=1}^{10}\text{bid}_k", "bid_mean = (bid1+...+bid10)/10"),
        ("ask_mean", "10 档卖价均值", "Mean of 10 ask prices", r"\frac{1}{10}\sum_{k=1}^{10}\text{ask}_k", "ask_mean = (ask1+...+ask10)/10"),
        ("bsize_mean", "10 档买量均值", "Mean of 10 bid-sizes", r"\frac{1}{10}\sum_{k=1}^{10}\text{bsize}_k", "bsize_mean = totalbsize/10"),
        ("asize_mean", "10 档卖量均值", "Mean of 10 ask-sizes", r"\frac{1}{10}\sum_{k=1}^{10}\text{asize}_k", "asize_mean = totalasize/10"),
    ]:
        items.append({
            "name": n, "name_cn": cn, "name_en_full": en_full,
            "raw_or_derived": "raw",
            "raw_idx_in_154": RAW_154_INDEX[n],
            "formula_latex": formula,
            "code_loc": "build_schemeP_cache.py:107 raw_last",
            "numpy_pseudo": ps,
            "physical_meaning": "对应方向的算术均值——与 totalbsize/totalasize 等价信息，但更稳健（不受单档异常挂单影响）。",
            "nan_handling": _nan_handling_raw(),
        })

    # ---- (10) cumspread, imbalance ----
    items.append({
        "name": "cumspread", "name_cn": "累计 10 档价差",
        "name_en_full": "Cumulative spread across 10 levels (ask_mean - bid_mean equivalent)",
        "raw_or_derived": "raw",
        "raw_idx_in_154": RAW_154_INDEX["cumspread"],
        "formula_latex": r"\text{cumspread}_t = \sum_{k=1}^{10}(\text{ask}_k - \text{bid}_k) = 10\,(\text{ask\_mean} - \text{bid\_mean})",
        "code_loc": "build_schemeP_cache.py:107 raw_last",
        "numpy_pseudo": "cumspread = sum(spread1..spread10) = 10*(ask_mean - bid_mean)",
        "physical_meaning": "10 档价差累积；衡量整个 LOB 的整体宽度（流动性/做市意愿宏观指标）。",
        "nan_handling": _nan_handling_raw(),
    })
    items.append({
        "name": "imbalance", "name_cn": "买卖盘力量失衡",
        "name_en_full": "Order-book imbalance (bid vs ask depth, exchange-provided)",
        "raw_or_derived": "raw",
        "raw_idx_in_154": RAW_154_INDEX["imbalance"],
        "formula_latex": r"\text{imbalance}_t = \frac{\text{totalbsize} - \text{totalasize}}{\text{totalbsize} + \text{totalasize}}",
        "code_loc": "build_schemeP_cache.py:107 raw_last (主办方提供)",
        "numpy_pseudo": "imbalance = (totalbsize - totalasize) / (totalbsize + totalasize)",
        "physical_meaning": "买卖盘挂单不平衡比例 ∈ (−1, +1)，>0 买方占优、价格倾向上行；最经典的 LOB 方向预测器之一。",
        "nan_handling": _nan_handling_raw(),
    })

    # ---- (11) wmp_lvl 1..10 (derived) ----
    # In all_feature_names() ordering: mlofi(30) + wmp_lvl(10) + wmp_balance_12(1) + rv(4) + ewma(24) + s1(54) + s2(59) + s3(14) + s5(20)
    # So wmp_lvl1 is index 30 within derived 216 → dim_in_X = 154 + 30 = 184
    for k in range(1, 11):
        n = f"wmp_lvl{k}"
        derived_idx = 30 + (k - 1)
        items.append({
            "name": n, "name_cn": f"第 {k} 档加权中价 (WMP)",
            "name_en_full": f"Level-{k} weighted mid price (size-weighted micro price)",
            "raw_or_derived": "derived",
            "raw_idx_in_154": None,
            "derived_idx_in_216": derived_idx,
            "formula_latex": (
                r"\text{wmp}_k = \frac{\text{ask}_k\cdot\text{bsize}_k + \text{bid}_k\cdot\text{asize}_k}"
                r"{\text{bsize}_k + \text{asize}_k}"
                r"\;\;\text{(fallback: }(\text{ask}_k+\text{bid}_k)/2\text{ when denom=0)}"
            ),
            "code_loc": "fast_features_batch.py:202-217",
            "numpy_pseudo": (
                "denom = bsize_k + asize_k\n"
                "wmp_k = where(denom==0, (ask_k+bid_k)/2, "
                "(ask_k*bsize_k + bid_k*asize_k) / denom)"
            ),
            "physical_meaning": (
                "Stoikov micro-price的第 k 档版本——用对侧挂单量给报价加权：买盘量大时 wmp 偏 ask、卖盘量大时偏 bid，"
                "因此 wmp 的偏离方向比 midprice 更能预测下一步价格走向；wmp_lvl1 是 NN 监督路径上 RV 的基底。"
            ),
            "nan_handling": _nan_handling_derived("denom==0 时回退到 (ask+bid)/2 (公式层兜底)"),
        })

    # ---- (12) wmp_balance_12 derived ----
    items.append({
        "name": "wmp_balance_12", "name_cn": "1-2 档加权中价之差",
        "name_en_full": "WMP balance level-1 minus level-2",
        "raw_or_derived": "derived",
        "raw_idx_in_154": None,
        "derived_idx_in_216": 40,
        "formula_latex": r"\text{wmp\_balance\_12} = \text{wmp\_lvl1} - \text{wmp\_lvl2}",
        "code_loc": "fast_features_batch.py:215-216",
        "numpy_pseudo": "wmp_balance_12 = wmp_lvl1[-1] - wmp_lvl2[-1]",
        "physical_meaning": (
            "1 档与 2 档加权中价之差，正负号反映 1 档相对 2 档的局部失衡——"
            "当 lvl1 的加权中价显著高于 lvl2 表示 ask 一侧近端挂单量更大或将吸吃，预测短期下行。"
        ),
        "nan_handling": _nan_handling_derived(),
    })

    return items


def f2_meta() -> List[Dict]:
    """50 F2 features."""
    items: List[Dict] = []

    # ---- mlofi W=5,20,60 × lvl=1..10 (30) ----
    # derived idx within 216: mlofi block占据 0..29
    cont_formula = (
        r"e_t^{(k)} \;=\; \mathbb{1}[b_t \!\geq\! b_{t-1}]\,bs_t "
        r"-\; \mathbb{1}[b_t \!\leq\! b_{t-1}]\,bs_{t-1} "
        r"-\; \mathbb{1}[a_t \!\leq\! a_{t-1}]\,as_t "
        r"+\; \mathbb{1}[a_t \!\geq\! a_{t-1}]\,as_{t-1}"
    )
    for i_W, W in enumerate([5, 20, 60]):
        for k in range(1, 11):
            n = f"mlofi_W{W}_lvl{k}"
            derived_idx = i_W * 10 + (k - 1)
            items.append({
                "name": n, "name_cn": f"多档 OFI 窗口{W}-档{k}",
                "name_en_full": f"Multi-level OFI window={W} ticks, level={k} (Cont 2014)",
                "raw_or_derived": "derived",
                "raw_idx_in_154": None,
                "derived_idx_in_216": derived_idx,
                "formula_latex": (
                    cont_formula +
                    rf"\;,\quad \text{{mlofi}}_W^{{(k)}} = \sum_{{\tau=t-W+1}}^{{t}} e_\tau^{{(k)}}"
                ),
                "code_loc": "fast_features_batch.py:156-184 (e_per_lvl + W-sum)",
                "numpy_pseudo": (
                    "# Cont 2014 OFI 单档 e_t (with prev-tick lag):\n"
                    f"# b, a, bs, as = bid{k}, ask{k}, bsize{k}, asize{k}\n"
                    "ind_b_up = (b >= b_lag);  ind_b_dn = (b <= b_lag)\n"
                    "ind_a_dn = (a <= a_lag);  ind_a_up = (a >= a_lag)\n"
                    "e_t = ind_b_up*bs - ind_b_dn*bs_lag - ind_a_dn*as_ + ind_a_up*as_lag\n"
                    f"mlofi_W{W}_lvl{k} = nansum(e_t[last_{W}_ticks])"
                ),
                "physical_meaning": (
                    "Cont (2014) 多档 Order-Flow-Imbalance 在窗口 W 上的累积：买价上行 (+bs) 视为主动买推升、"
                    "买价下行视为吃单或挂单消失 (-bs_prev)；对称地处理 ask 一侧。"
                    f"W={W}-tick (≈{W*3}s) 平滑后的本档主动方向流，正值预示短期上行。"
                ),
                "nan_handling": _nan_handling_derived("窗口内 NaN→0 后 sum (fast_features_batch:182)"),
            })

    # ---- kyle_inv W=50, 100 (2). Stage1 in derived ordering ----
    # derived order: mlofi(30) + wmp(10) + wmp_bal(1) + rv(4) + ewma_intst(24) = 69 (T3 block)
    # then stage1: dualz(37) + signed_rv(3) + kyle_inv(2) + ewma_ofi(12) = 54
    # So kyle_inv_W50 derived idx = 69 + 37 + 3 + 0 = 109
    base_s1 = 69
    for j, W in enumerate([50, 100]):
        n = f"kyle_inv_W{W}"
        derived_idx = base_s1 + 37 + 3 + j
        items.append({
            "name": n, "name_cn": f"Kyle 逆冲击 W={W}",
            "name_en_full": f"Kyle inverse-impact proxy at last tick over W={W}",
            "raw_or_derived": "derived",
            "raw_idx_in_154": None,
            "derived_idx_in_216": derived_idx,
            "formula_latex": (
                r"\sigma_W = \mathrm{std}(r_{t-W+1:t}),\;"
                r"a = (|\text{amt}_t| + \epsilon)\,\sigma_W + \epsilon\;,\;"
                rf"\text{{kyle\_inv}}_W = \text{{amt}}_t / (\sqrt[3]{{a}} + \epsilon)"
            ),
            "code_loc": "fast_features_batch.py:310-321",
            "numpy_pseudo": (
                "# r = log-return of midprice; amt = signed amount_delta\n"
                f"sigma_W = std(r[-{W}:])\n"
                f"a = |amt_last| * sigma_W + EPS\n"
                f"kyle_inv_W{W} = amt_last / (cbrt(a) + EPS)"
            ),
            "physical_meaning": (
                "Kyle (1985) λ 的逆方向变体：单位 ‘信息含量’ amount × σ 折算后归一化的金额——"
                "若 |amount| 足够大且 σ 较小（信息冲击型），信号值大；正号表示买方驱动。"
                "W=50 短期、W=100 中期。"
            ),
            "nan_handling": _nan_handling_derived("分母 +EPS、isfinite 兜底 →0"),
        })

    # ---- ewma_ofi α∈{0.05,0.1,0.3,0.5} × lvl∈{1,5,10} = 12 ----
    # ordering in code (fast_features_batch:323-328): for k in (1,5,10): for a in (0.05,0.1,0.3,0.5)
    # name in family_assignment: "ewma_ofi_a{α}_lvl{k}" — outer over k, inner over α
    base_ewma_ofi = base_s1 + 37 + 3 + 2  # 111
    alphas = (0.05, 0.1, 0.3, 0.5)
    for ki, k in enumerate([1, 5, 10]):
        for ai, alpha in enumerate(alphas):
            n = f"ewma_ofi_a{alpha}_lvl{k}"
            derived_idx = base_ewma_ofi + ki * 4 + ai
            items.append({
                "name": n, "name_cn": f"OFI EWMA α={alpha} 档{k}",
                "name_en_full": f"EWMA of rolling-W20 OFI level-{k}, alpha={alpha}",
                "raw_or_derived": "derived",
                "raw_idx_in_154": None,
                "derived_idx_in_216": derived_idx,
                "formula_latex": (
                    rf"x_t = \text{{mlofi\_W20\_lvl{k}}}_t,\;"
                    rf"\text{{ewma\_ofi}}^{{(k)}}_\alpha(t) = \alpha x_t + (1-\alpha)\,\text{{ewma\_ofi}}^{{(k)}}_\alpha(t-1)"
                    rf"\;,\;\alpha={alpha}"
                ),
                "code_loc": "fast_features_batch.py:323-328 (_ewma_last_batch on derived mlofi_W20)",
                "numpy_pseudo": (
                    f"x = mlofi_W20_lvl{k}_series  # rolling-20 OFI 完整序列\n"
                    f"y_t = {alpha}*x_t + (1-{alpha})*y_{{t-1}};  y_0={alpha}*x_0\n"
                    "out = y[-1]"
                ),
                "physical_meaning": (
                    f"对档 {k} 的 20-tick OFI 序列做 EWMA(α={alpha}) 平滑：α 大→更看重近 1-2 tick (短记忆)，"
                    f"α 小→长记忆（半衰期≈{round(0.693/alpha,1)} ticks）。同方向（买推升）OFI 的指数平均，"
                    "比 raw mlofi 更平滑也更鲁棒。"
                ),
                "nan_handling": _nan_handling_derived(),
            })

    # ---- kyle_lam W=50,100 (Stage 2) — 2 ----
    # Stage2 derived order in code: qrank(20)+rskew(3)+gofi(30)+kyle_lam(2)+vol_burst(4) = 59
    # qrank starts at derived 69+54 = 123
    base_s2 = 69 + 54  # 123
    for j, W in enumerate([50, 100]):
        n = f"kyle_lam_W{W}"
        derived_idx = base_s2 + 20 + 3 + 30 + j
        items.append({
            "name": n, "name_cn": f"Kyle λ 滚动 W={W}",
            "name_en_full": f"Rolling Kyle lambda over W={W} ticks",
            "raw_or_derived": "derived",
            "raw_idx_in_154": None,
            "derived_idx_in_216": derived_idx,
            "formula_latex": (
                r"x_t = \mathrm{sign}(\Delta m_t)\sqrt{|\text{amt}_t|+\epsilon},\;"
                r"y_t = \Delta m_t,\;"
                rf"\lambda_W = \frac{{\mathrm{{Cov}}_W(x,y)}}{{\mathrm{{Var}}_W(x)+\epsilon}}, \text{{clip}}[-1,1]"
            ),
            "code_loc": "fast_features_batch.py:431-453",
            "numpy_pseudo": (
                "dmid = diff(midprice1)\n"
                "x = sign(dmid) * sqrt(|amount_delta| + EPS)\n"
                "y = dmid\n"
                f"lam = cov(x[-{W}:], y[-{W}:]) / (var(x[-{W}:]) + EPS)\n"
                f"kyle_lam_W{W} = clip(lam, -1, 1)"
            ),
            "physical_meaning": (
                "经典 Kyle (1985) 价格冲击系数滚动估计：单位 ‘签名根成交量’ 引致的 mid-price 变化。"
                "λ 大表示 LOB 浅、价格易被推动。**KS-drop 命中**：训练集分布与 val/test 偏差过大被剔除。"
            ),
            "nan_handling": _nan_handling_derived("clip 到 [-1, 1]; isfinite 兜底 →0"),
        })

    # ---- ofi_tox lvl∈{1,5} × W∈{20,50} (Stage 5) — 4 ----
    # Stage 5 derived order (compute_stage5_batch): adapt_mom(3) + ofi_tox(4) + signed_bv(3) + spread_reg(3) + trade_pers(3) + liq_asym(4) = 20
    # Stage 5 starts at derived 69+54+59+14 = 196
    base_s5 = 196
    s5_pairs = [(1, 20), (1, 50), (5, 20), (5, 50)]
    for j, (lvl, W) in enumerate(s5_pairs):
        n = f"ofi_tox_lvl{lvl}_W{W}"
        derived_idx = base_s5 + 3 + j
        items.append({
            "name": n, "name_cn": f"OFI 毒性 (corr) 档{lvl} W={W}",
            "name_en_full": f"OFI toxicity = corr(OFI_l{lvl}, Δmid) over last W={W} ticks",
            "raw_or_derived": "derived",
            "raw_idx_in_154": None,
            "derived_idx_in_216": derived_idx,
            "formula_latex": (
                rf"\text{{ofi\_tox}}_{{l={lvl},W={W}}} = \mathrm{{corr}}_W(\text{{OFI}}^{{(lvl)}}_t,\;\Delta m_t),\;"
                r"\text{clip}[-1,1]"
            ),
            "code_loc": "stage5_features.py / fast_features_batch.py:634-669",
            "numpy_pseudo": (
                f"ofi_l{lvl} = (Cont OFI per-tick e at level {lvl})\n"
                f"dmid = diff(midprice1)\n"
                f"ofi_tox_lvl{lvl}_W{W} = pearson_corr(ofi_l{lvl}[-{W}:], dmid[-{W}:])"
            ),
            "physical_meaning": (
                f"档 {lvl} 的 OFI 与 Δmid 在最近 W={W} 上的 Pearson 相关——"
                "近期订单流方向与价格变动是否一致（即 ‘有信息’ 程度）。"
                "高正相关 = 订单流推得动价格 = 信息毒性高。"
            ),
            "nan_handling": _nan_handling_derived("denom +EPS、clip[-1,1]、isfinite 兜底"),
        })

    return items


def all_meta() -> List[Dict]:
    return f1_meta() + f2_meta()
