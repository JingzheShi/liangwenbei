"""Static metadata for F3 (订单流强度) + F4 (微结构波动) features.

Each entry: name -> dict with identity, formula, physical_meaning, nan_handling.
Stats (NaN%, PSI, KS) are filled in by audit pipeline.
"""
from __future__ import annotations


# ---------- F3 raw 18 (6 categories × 3 variants: intst / ind / acc) ----------
_F3_CN = {
    "lb": "限价买", "la": "限价卖", "mb": "市价买",
    "ma": "市价卖", "cb": "撤买", "ca": "撤卖",
}
_F3_EN = {
    "lb": "Limit-Bid", "la": "Limit-Ask", "mb": "Market-Bid",
    "ma": "Market-Ask", "cb": "Cancel-Bid", "ca": "Cancel-Ask",
}
_VARIANT_CN = {"intst": "强度", "ind": "二值指示", "acc": "二阶差分"}
_VARIANT_EN = {
    "intst": "Order-Flow Intensity (count or amount)",
    "ind": "Binary Indicator (presence)",
    "acc": "2nd-order Difference (acceleration)",
}


def _f3_raw(name: str) -> dict:
    cat, var = name.split("_")  # e.g. "ma_intst"
    return {
        "name": name,
        "name_cn": f"{_F3_CN[cat]}{_VARIANT_CN[var]}",
        "name_en_full": f"{_F3_EN[cat]} {_VARIANT_EN[var]}",
        "family": "F3",
        "raw_or_derived": "raw",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"x_t^{(\mathrm{intst})} = N^{(\cdot)}_t \quad \text{or aggregated amount of "
            "this order type at tick t}"
            if var == "intst"
            else (
                r"x_t^{(\mathrm{ind})} = \mathbf{1}\{N^{(\cdot)}_t > 0\}"
                if var == "ind"
                else r"x_t^{(\mathrm{acc})} = N^{(\cdot)}_t - 2 N^{(\cdot)}_{t-1} + N^{(\cdot)}_{t-2}"
            )
        ),
        "code_loc": "fast_features_batch.py:N/A (raw, 主办方原始 column)",
        "numpy_pseudo": f'{name} = raw_X[:, col_idx["{name}"]]  # 3s tick aggregation',
        "physical_meaning": (
            "intst = 该类委托/成交在 3s tick 内的频次或量；信息密度最高。lb/la 反映报价新增，"
            "mb/ma 反映主动成交（信息含量最强：Top-4 都是 mb_intst/ma_intst/la_intst/lb_intst），"
            "cb/ca 反映撤单博弈。intst > ind > acc：强度保留幅度信息，二值化 (ind) 损失幅度而只保留方向，"
            "二阶差分 (acc) 是噪声/突变检测，信噪比通常最低。"
            if var == "intst" else
            "ind = 是否发生此类委托的二值指示。把幅度信息压成 {0,1}，信号粒度变粗；"
            "在 LGB 树模型里 ind 几乎可被 intst 完全替代（intst==0 ↔ ind==0），所以 ind 重要性远低于 intst。"
            if var == "ind" else
            "acc = 二阶差分，捕捉强度的二阶突变（如订单流加速/减速）。"
            "由于 3s tick 强度本身波动大、二阶差分放大噪声，信噪比最低；ablation 中删除 acc 几乎不影响指标。"
        ),
        "nan_handling": {
            "formula_level": "raw 字段；主办方 cache 已保证无 NaN（log1p 后 finite）",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动，LightGBM 内置 missing routing",
        },
    }


# ---------- F3 EWMA derived 24 (4 alphas × 6 intst columns) ----------
def _f3_ewma(alpha: float, cat: str) -> dict:
    half_life = -0.6931 / (-1e-9 + (1.0 if alpha >= 1.0 else float(__import__("math").log(1 - alpha))))
    hl = abs(half_life)
    name = f"ewma_a{alpha}_{cat}_intst"
    return {
        "name": name,
        "name_cn": f"{_F3_CN[cat]}强度-EWMA(α={alpha}, HL≈{hl:.1f}tick)",
        "name_en_full": f"EWMA of {_F3_EN[cat]} Intensity (alpha={alpha}, half-life≈{hl:.1f} ticks)",
        "family": "F3",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\tilde{x}_t^{(\alpha)} = \alpha \cdot x_t + (1-\alpha)\cdot \tilde{x}_{t-1}^{(\alpha)},\;"
            r"\tilde{x}_0 = x_0;\;\;\mathrm{HalfLife}=\frac{\ln 2}{-\ln(1-\alpha)}"
        ),
        "code_loc": (
            "fast_features_batch.py:235-241 (compute_t3_no_time_batch 内 EWMA loop) — "
            "底层调用 _ewma_last_batch (line 82-100) 用 scipy.signal.lfilter 等价 pandas .ewm(adjust=False)"
        ),
        "numpy_pseudo": (
            f"x = raw_X3d[:, :, col_idx['{cat}_intst']]\n"
            f"x_safe = np.where(np.isfinite(x), x, 0.0)\n"
            f"# IIR: y_0=x_0; y_t = a*x_t + (1-a)*y_{{t-1}}\n"
            f"y = lfilter([{alpha}], [1, -(1-{alpha})], x_safe, axis=-1, zi=(1-{alpha})*x_safe[:, :1])[0]\n"
            f"{name} = y[:, -1]"
        ),
        "physical_meaning": (
            f"对 {cat}_intst 做平滑（α={alpha}, half-life≈{hl:.1f} ticks ≈ {hl*3:.0f}s）。"
            f"4 个 α 对应 4 个时间尺度 (HL ≈ 13/7/2/1 ticks 即 ~40/20/6/3 s)，"
            f"覆盖『慢趋势→快冲击』的多尺度订单流强度，对 NN 提供时间维度的平滑特征金字塔。"
            f"α 大 → 反应快但抗噪差；α 小 → 平滑稳但延迟大。Top-4 是 raw intst (α=1)，EWMA 提供互补的滞后/平滑信息。"
        ),
        "nan_handling": {
            "formula_level": "np.where(np.isfinite(x), x, 0.0) 兜底，避免 NaN 污染 IIR 递推 (line 238)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动，LightGBM 内置 missing routing",
        },
    }


# ---------- F3 cancel_imb_W (3 W values) ----------
def _f3_cancel_imb(W: int) -> dict:
    return {
        "name": f"cancel_imb_W{W}",
        "name_cn": f"撤单不平衡(窗口W={W}tick≈{W*3}s)",
        "name_en_full": f"Cancellation Imbalance over W={W} ticks",
        "family": "F3",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{cancel\_imb}_W = "
            r"\frac{\sum_{s=t-W+1}^{t} cb_s}{\sum (lb_s+mb_s+cb_s)+\epsilon}"
            r"-\frac{\sum_{s=t-W+1}^{t} ca_s}{\sum (la_s+ma_s+ca_s)+\epsilon}"
        ),
        "code_loc": "fast_features_batch.py:535-554 (compute_stage3_batch --- cancel_imb)",
        "numpy_pseudo": (
            f"cb_sum=cb[:, -{W}:].sum(-1); ca_sum=ca[:, -{W}:].sum(-1)\n"
            f"bt_sum=(lb+mb+cb)[:, -{W}:].sum(-1); at_sum=(la+ma+ca)[:, -{W}:].sum(-1)\n"
            f"imb=(cb_sum/(bt_sum+1e-8))-(ca_sum/(at_sum+1e-8))\n"
            f"cancel_imb_W{W} = np.clip(np.nan_to_num(imb), -1, 1)"
        ),
        "physical_meaning": (
            f"窗口 W={W} 内『买侧撤单占买侧总下单的比例』减去『卖侧撤单占卖侧总下单的比例』，∈[-1,1]。"
            f"+1 = 买侧撤单密集（看跌信号，参与者放弃挂单）；-1 = 卖侧撤单密集（看涨信号）。"
            f"与 ma_intst/mb_intst 互补：intst 看主动成交方向，cancel_imb 看『谁先怂』。"
            f"3 个 W 覆盖短/中/长尺度（W=20≈60s 短期博弈，W=100≈300s 中期持仓压力）。"
        ),
        "nan_handling": {
            "formula_level": "denom +EPS=1e-8 避免除零；np.where(np.isfinite, ..., 0.0) 兜底；clip(-1, 1)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 raw 2 ----------
def _f4_raw(name: str) -> dict:
    cn = "成交量增量" if name == "volume_delta" else "成交额增量"
    en = "Volume Delta (Δ shares per tick)" if name == "volume_delta" else "Amount Delta (Δ cash per tick)"
    return {
        "name": name,
        "name_cn": cn,
        "name_en_full": en,
        "family": "F4",
        "raw_or_derived": "raw",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\Delta V_t = V_t - V_{t-1}\;\;(\text{shares})"
            if name == "volume_delta" else
            r"\Delta A_t = A_t - A_{t-1}\;\;(\text{cash})"
        ),
        "code_loc": "fast_features_batch.py:N/A (raw, 主办方原始 column)",
        "numpy_pseudo": f'{name} = raw_X[:, col_idx["{name}"]]',
        "physical_meaning": (
            "3s tick 内的成交量/成交额增量。"
            "F4 微结构波动的两个 raw 输入。vol_burst_W / amt_burst_W 直接以它们为基础做窗口冲击检测；"
            "kyle_lam / signed_dvol / OFI toxicity 也用 amount_delta 衡量主动成交强度。"
        ),
        "nan_handling": {
            "formula_level": "raw 字段；cache 已 log1p 处理，保证无 NaN/inf",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 rv_w (4 W) ----------
def _f4_rv(W: int) -> dict:
    return {
        "name": f"rv_w{W}",
        "name_cn": f"实现波动率(窗口W={W}tick≈{W*3}s)",
        "name_en_full": f"Realized Volatility over W={W} ticks (wmp_lvl1-based)",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{RV}_W = \sqrt{\max\left(0, \sum_{s=t-W+1}^{t} r_s^2\right)},\;"
            r"r_s = \log\frac{P_s+1}{P_{s-1}+1}"
        ),
        "code_loc": "fast_features_batch.py:216-230 (T3 RV loop on wmp_lvl1+1)",
        "numpy_pseudo": (
            f"safe = wmp_lvl1 + 1.0  # +1 to guard zero/negative\n"
            f"log_ret = np.log(safe[:,1:]/safe[:,:-1])\n"
            f"log_ret = np.where(np.isfinite(log_ret), log_ret, 0.0)\n"
            f"rv_w{W} = np.sqrt(np.maximum((log_ret**2)[:, -{W}:].sum(-1), 0.0))"
        ),
        "physical_meaning": (
            f"T3 stage 计算的实现波动率：基于 wmp_lvl1 (1档加权中间价) 的对数收益平方和。"
            f"W={W} ticks ≈ {W*3}s 时间窗。RV 是经典 Andersen-Bollerslev 类指标，"
            f"反映该窗口内中间价的累积波动幅度；高 RV 通常对应剧烈波动 / 大额成交 / 重大新闻冲击。"
            f"NN/LGB 利用 RV 调整对 mid 走势的『风险下注幅度』。"
        ),
        "nan_handling": {
            "formula_level": "np.where(safe>0, safe, np.nan) 防 log 负数; np.where(isfinite, lr, 0.0) 填零; sqrt(max(.,0))",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 signed_rv (3 W) ----------
def _f4_signed_rv(W: int) -> dict:
    return {
        "name": f"signed_rv_W{W}",
        "name_cn": f"方向化实现波动率(W={W}tick≈{W*3}s)",
        "name_en_full": f"Signed Realized Volatility over W={W} ticks",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{signedRV}_W = \frac{\sum r_s^2\mathbf{1}\{r_s>0\}-\sum r_s^2\mathbf{1}\{r_s<0\}}"
            r"{\sum r_s^2\mathbf{1}\{r_s>0\}+\sum r_s^2\mathbf{1}\{r_s<0\}+\epsilon}\in[-1,1]"
        ),
        "code_loc": "fast_features_batch.py:293-308 (compute_stage1_batch --- signed_rv)",
        "numpy_pseudo": (
            f"r = log_ret(midprice)  # 基于 midprice1\n"
            f"r2_pos = r**2 * (r>0); r2_neg = r**2 * (r<0)\n"
            f"rp = r2_pos[:, -{W}:].sum(-1); rn = r2_neg[:, -{W}:].sum(-1)\n"
            f"signed_rv_W{W} = (rp-rn)/(rp+rn+1e-8)"
        ),
        "physical_meaning": (
            f"W={W} 窗口内『涨方差 - 跌方差』除以总方差，∈[-1,1]。"
            f"+1 = 该窗口完全由上涨贡献波动（涨主导），-1 = 完全由下跌贡献（跌主导）。"
            f"是 mid 方向预测的强信号：与 mb_intst/ma_intst 互补——后者看意图（主动买/卖），"
            f"signed_rv 看实际成交后的中间价方向。"
        ),
        "nan_handling": {
            "formula_level": "(rp-rn)/(rp+rn+EPS) 防除零；上游 r=np.where(isfinite, r, 0)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 rskew (3 W) ----------
def _f4_rskew(W: int) -> dict:
    return {
        "name": f"rskew_W{W}",
        "name_cn": f"对数收益偏度(W={W}tick≈{W*3}s)",
        "name_en_full": f"Realized Skewness of log-returns over W={W} ticks",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{rskew}_W = \mathrm{clip}\left(\frac{\mu_3}{\sigma^3+\epsilon}, -10, 10\right),\;"
            r"\mu_3 = \overline{r^3} - 3\overline{r}\,\overline{r^2}+2\overline{r}^3"
        ),
        "code_loc": "fast_features_batch.py:374-393 (compute_stage2_batch --- skew)",
        "numpy_pseudo": (
            f"r = log_ret(midprice)\n"
            f"seg = r[:, -{W}:]; mu = seg.mean(-1); sd = seg.std(-1)+1e-8\n"
            f"m3 = (seg**3).mean(-1) - 3*mu*(seg**2).mean(-1) + 2*mu**3\n"
            f"rskew_W{W} = np.clip(np.nan_to_num(m3/(sd**3+1e-8)), -10, 10)"
        ),
        "physical_meaning": (
            f"W={W} 窗口内 log 收益的偏度（3 阶矩 / 3 阶标准差）。"
            f"正偏 = 大涨少而显著、小跌频繁；负偏反之。"
            f"对 mid 方向预测有领先性：负偏暗示『下跌尾部风险积累』，往往伴随后续向下突破。"
            f"采用 sample 中心矩公式避免数值不稳，clip(±10) 防长尾爆炸。"
        ),
        "nan_handling": {
            "formula_level": "denom (sd**3+EPS) 防除零；np.where(isfinite, skew, 0)；clip(±10)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 vol_burst / amt_burst (2 W each) ----------
def _f4_vol_burst(W: int) -> dict:
    return {
        "name": f"vol_burst_W{W}",
        "name_cn": f"成交量冲击比(W={W}tick≈{W*3}s)",
        "name_en_full": f"Volume Burst Ratio (last tick vs mean over W={W} ticks)",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{vol\_burst}_W = \mathrm{clip}\left(\frac{\Delta V_t}{\overline{\Delta V}_{[t-W+1,t]}+\epsilon}, -100, 100\right)"
        ),
        "code_loc": "fast_features_batch.py:455-466 (compute_stage2_batch --- vol_burst)",
        "numpy_pseudo": (
            f"mean_v = volume_delta[:, -{W}:].mean(-1) + 1e-8\n"
            f"vol_burst_W{W} = np.clip(np.nan_to_num(volume_delta[:, -1]/mean_v), -100, 100)"
        ),
        "physical_meaning": (
            f"当前 tick 成交量相对窗口均值的『冲击倍数』；W={W} 给短期 ({W*3}s) 窗口。"
            f">>1 = 突发大单/连续放量（机构进出场信号）；<<1 = 当前 tick 异常清淡。"
            f"对短期价格突变有领先性，常与 amt_burst / kyle_lam 联合解释流动性事件。"
        ),
        "nan_handling": {
            "formula_level": "+EPS 防除零；np.where(isfinite, ., 0)；clip(-100, 100)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


def _f4_amt_burst(W: int) -> dict:
    return {
        "name": f"amt_burst_W{W}",
        "name_cn": f"成交额冲击比(W={W}tick≈{W*3}s)",
        "name_en_full": f"Amount Burst Ratio (|Δamt_t| vs mean |Δamt| over W={W} ticks)",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{amt\_burst}_W = \mathrm{clip}\left(\frac{|\Delta A_t|}{\overline{|\Delta A|}_{[t-W+1,t]}+\epsilon},\,0,\,100\right)"
        ),
        "code_loc": "fast_features_batch.py:455-468 (compute_stage2_batch --- amt_burst, 配对 vol_burst)",
        "numpy_pseudo": (
            f"mean_a = np.abs(amount_delta)[:, -{W}:].mean(-1) + 1e-8\n"
            f"amt_burst_W{W} = np.clip(np.nan_to_num(np.abs(amount_delta[:, -1])/mean_a), 0, 100)"
        ),
        "physical_meaning": (
            f"当前 tick |成交额| 相对窗口均值的冲击倍数 (∈[0,100])。"
            f"与 vol_burst 互补——vol_burst 看股数（不分价格），amt_burst 看金额（含价格信息）。"
            f"金额冲击比 = 异动检测，对识别大单/批量交易尤其敏感。"
        ),
        "nan_handling": {
            "formula_level": "+EPS 防除零；np.where(isfinite, ., 0)；clip(0, 100)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 rv_ratio (3 pairs) ----------
def _f4_rv_ratio(Wn: int, Wd: int) -> dict:
    return {
        "name": f"rv_ratio_W{Wn}_W{Wd}",
        "name_cn": f"波动率比({Wn}/{Wd}tick)",
        "name_en_full": f"Realized Variance Ratio short(W={Wn})/long(W={Wd})",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{rv\_ratio}_{W_n/W_d} = \mathrm{clip}\left(\frac{\sum_{s=t-W_n+1}^t r_s^2}{\sum_{s=t-W_d+1}^t r_s^2 + \epsilon}, 0, 100\right)"
        ),
        "code_loc": "fast_features_batch.py:506-519 (compute_stage3_batch --- rv_ratio)",
        "numpy_pseudo": (
            f"r2 = log_ret(midprice)**2\n"
            f"rv_{Wn} = r2[:, -{Wn}:].sum(-1); rv_{Wd} = r2[:, -{Wd}:].sum(-1)\n"
            f"rv_ratio_W{Wn}_W{Wd} = np.clip(np.nan_to_num(rv_{Wn}/(rv_{Wd}+1e-8)), 0, 100)"
        ),
        "physical_meaning": (
            f"短窗 RV / 长窗 RV ∈ [0, 100]。"
            f">> Wn/Wd（无量纲尺度比）= 短期波动突增（波动率冲击），"
            f"对识别『盘整→爆发』非常有效；NN 通过它做时间尺度的相对放大检测。"
        ),
        "nan_handling": {
            "formula_level": "+EPS 防除零；np.where(isfinite, ratio, 0)；clip(0, 100)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 jshare (4 W) ----------
def _f4_jshare(W: int) -> dict:
    return {
        "name": f"jshare_W{W}",
        "name_cn": f"跳跃份额(BNS, W={W}tick≈{W*3}s)",
        "name_en_full": f"Barndorff-Nielsen-Shephard Jump Share over W={W} ticks",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{jshare}_W = \mathrm{clip}\left(\frac{\mathrm{RV}_W - \mathrm{BV}_W}{\mathrm{RV}_W + \epsilon},0,1\right),\;"
            r"\mathrm{BV}_W = \frac{\pi}{2}\sum_{s=t-W+2}^{t} |r_s||r_{s-1}|"
        ),
        "code_loc": "fast_features_batch.py:521-533 (compute_stage3_batch --- jshare)",
        "numpy_pseudo": (
            f"abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1)\n"
            f"bv = (np.pi/2) * (abs_r*abs_r_lag)[:, -{W}:].sum(-1)\n"
            f"rv = (r**2)[:, -{W}:].sum(-1)\n"
            f"jshare_W{W} = np.clip(np.nan_to_num((rv-bv)/(rv+1e-8)), 0, 1)"
        ),
        "physical_meaning": (
            f"BNS (2004) 跳跃份额 ∈ [0,1]：RV 包含跳跃 + 连续波动；BV (Bipower Variation) 只对连续部分稳健。"
            f"(RV-BV)/RV 即跳跃占总方差比例。π/2 是 |r||r_lag| 在 Brownian motion 下的方差归一化因子。"
            f"高 jshare = 价格存在不连续跳跃（如撮合大单冲击），对预测后续 mean-reversion / 跟随有意义。"
        ),
        "nan_handling": {
            "formula_level": "+EPS 防除零；np.where(isfinite, j, 0)；clip(0, 1)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 roll_eff_spr (3 W) ----------
def _f4_roll(W: int) -> dict:
    is_ks_drop = (W == 100)
    return {
        "name": f"roll_eff_spr_ratio_W{W}",
        "name_cn": f"Roll有效价差比(W={W}tick≈{W*3}s)",
        "name_en_full": f"Roll (1984) Effective Spread Ratio over W={W} ticks",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": is_ks_drop,
        "formula_latex": (
            r"\mathrm{roll\_eff\_spr\_ratio}_W = \mathrm{clip}\left(\frac{2\sqrt{\max(0, -\mathrm{cov}(\Delta P_t,\Delta P_{t-1}))}}"
            r"{\mathrm{spread1}_t + 1}, 0, 10\right)"
        ),
        "code_loc": "fast_features_batch.py:556-576 (compute_stage3_batch --- roll_eff_spr)",
        "numpy_pseudo": (
            f"dc = np.diff(close, prepend=close[:,:1])  # ΔP\n"
            f"dc_lag = shift(dc, 1)\n"
            f"cov_W = (dc*dc_lag)[:, -{W}:].mean(-1) - dc[:, -{W}:].mean(-1)*dc_lag[:, -{W}:].mean(-1)\n"
            f"eff = 2 * np.sqrt(np.maximum(-cov_W, 0))\n"
            f"roll_eff_spr_ratio_W{W} = np.clip(np.nan_to_num(eff/(spread1[:,-1]+1+1e-8)), 0, 10)"
        ),
        "physical_meaning": (
            f"Roll 1984 经典：在『无信息流交易』模型下，价格变化序列的负自协方差直接编码 bid-ask 弹跳幅度。"
            f"-cov(ΔP, ΔP_lag) > 0 时，2√(-cov) 即有效价差估计；再除以名义 spread1 得『相对成本』。"
            f"高 ratio = 隐性流动性成本远高于 quote spread → 可能预示反转/参与者撤出。"
            f"⚠️ KS-drop 11 之一（W=100 版本）：train→test KS 高于阈值，被训练管线 mask；W=30/50 保留。"
        ),
        "nan_handling": {
            "formula_level": "max(-cov, 0) 防 sqrt 负值；+EPS 防除零；np.where(isfinite, ., 0)；clip(0, 10)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


# ---------- F4 signed_bv (3 W) — stage5 ----------
def _f4_signed_bv(W: int) -> dict:
    return {
        "name": f"signed_bv_W{W}",
        "name_cn": f"方向化双幂变差(W={W}tick≈{W*3}s)",
        "name_en_full": f"Signed Bipower Variation Ratio over W={W} ticks",
        "family": "F4",
        "raw_or_derived": "derived",
        "in_ks_drop_11": False,
        "formula_latex": (
            r"\mathrm{signed\_bv}_W = \mathrm{clip}\left(\frac{\frac{\pi}{2}\sum_{s=t-W+1}^{t}\mathrm{sign}(r_s)|r_s||r_{s-1}|}"
            r"{\sum r_s^2 + \epsilon},-10,10\right)"
        ),
        "code_loc": "fast_features_batch.py:671-685 (compute_stage5_batch --- C. Multi-scale signed bipower)",
        "numpy_pseudo": (
            f"abs_r = np.abs(r); abs_r_lag = shift(abs_r, 1); sign_r = np.sign(r)\n"
            f"sbv = (np.pi/2) * (sign_r*abs_r*abs_r_lag)[:, -{W}:].sum(-1)\n"
            f"rv = (r**2)[:, -{W}:].sum(-1) + 1e-8\n"
            f"signed_bv_W{W} = np.clip(np.nan_to_num(sbv/rv), -10, 10)"
        ),
        "physical_meaning": (
            f"对 jshare 的方向化版本：保留 sign(r_t)，把连续段（BV-like）的方向信号汇总后再除 RV 归一。"
            f">0 = 该窗口『连续波动』偏涨主导；<0 = 偏跌主导。"
            f"与 signed_rv 互补——signed_rv 用 r²（强调极值），signed_bv 用 |r||r_lag|（更稳健、抑制单点跳跃）。"
        ),
        "nan_handling": {
            "formula_level": "+EPS 防除零；np.where(isfinite, v, 0)；clip(-10, 10)",
            "nn_pipeline": "wz 路径: (x-μ)/σ → nan_to_num(0)/posinf=10/neginf=-10 → clip(±10)；raw 路径: nan_to_num(0)",
            "lgb_pipeline": "不动",
        },
    }


def build_all_meta() -> list[dict]:
    """Build metadata for all 45 F3 + 29 F4 = 74 features (without stat fields)."""
    meta: list[dict] = []

    # ---- F3 raw 18: order = lb,la,mb,ma,cb,ca × intst,ind,acc; matches feat_names.txt
    cats = ["lb", "la", "mb", "ma", "cb", "ca"]
    for var in ["intst", "ind", "acc"]:
        for cat in cats:
            meta.append(_f3_raw(f"{cat}_{var}"))

    # ---- F3 EWMA derived 24: order alpha then cat (matches feat_names.txt 200-223)
    for alpha in [0.05, 0.1, 0.3, 0.5]:
        for cat in cats:
            meta.append(_f3_ewma(alpha, cat))

    # ---- F3 cancel_imb (3)
    for W in [20, 50, 100]:
        meta.append(_f3_cancel_imb(W))

    # ---- F4 raw 2
    meta.append(_f4_raw("volume_delta"))
    meta.append(_f4_raw("amount_delta"))

    # ---- F4 rv_w (4)
    for W in [5, 10, 20, 50]:
        meta.append(_f4_rv(W))

    # ---- F4 signed_rv (3)
    for W in [20, 50, 100]:
        meta.append(_f4_signed_rv(W))

    # ---- F4 rskew (3)
    for W in [20, 50, 100]:
        meta.append(_f4_rskew(W))

    # ---- F4 vol_burst / amt_burst (2 W each, interleaved per W)
    for W in [20, 50]:
        meta.append(_f4_vol_burst(W))
        meta.append(_f4_amt_burst(W))

    # ---- F4 rv_ratio (3)
    for (Wn, Wd) in [(5, 50), (20, 100), (50, 100)]:
        meta.append(_f4_rv_ratio(Wn, Wd))

    # ---- F4 jshare (4)
    for W in [20, 30, 50, 100]:
        meta.append(_f4_jshare(W))

    # ---- F4 roll (3)
    for W in [30, 50, 100]:
        meta.append(_f4_roll(W))

    # ---- F4 signed_bv (3)
    for W in [20, 50, 100]:
        meta.append(_f4_signed_bv(W))

    return meta


if __name__ == "__main__":
    m = build_all_meta()
    print(f"Total metadata entries: {len(m)}")
    fams = {"F3": 0, "F4": 0}
    for x in m:
        fams[x["family"]] += 1
    print(f"  F3: {fams['F3']}, F4: {fams['F4']}")
    for x in m[:3]:
        print(x["name"], "|", x["name_cn"])
    print("...")
    for x in m[-3:]:
        print(x["name"], "|", x["name_cn"])
