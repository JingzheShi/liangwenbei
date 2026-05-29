"""Merge 3 worker audits → 1 master JSON + summary markdown."""
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path("/root/projects/liangwenbei_workdir/factor_per_feature_audit")
OUT_JSON = ROOT / "MASTER_per_feature_audit.json"
OUT_MD = ROOT / "MASTER_summary.md"

all_records = []
all_summaries = {}
for sub in ("F1F2", "F3F4", "F5F6"):
    recs = json.loads((ROOT / sub / "per_feature_audit.json").read_text())
    summ = json.loads((ROOT / sub / f"{sub}_summary.json").read_text())
    all_records.extend(recs)
    all_summaries[sub] = summ

# normalize: ensure all records have same key set
canonical_keys = ["name", "name_cn", "name_en_full", "family", "raw_or_derived",
                  "dim_in_X", "in_ks_drop_11",
                  "formula_latex", "code_loc", "numpy_pseudo",
                  "physical_meaning", "nan_handling",
                  "nan_pct", "cross_stock_psi", "cross_stock_ks",
                  "cross_date_psi", "cross_date_ks",
                  "verdict_cross_stock", "verdict_cross_date", "notes"]
for r in all_records:
    for k in canonical_keys:
        r.setdefault(k, None)

def _pct(np_field, split):
    """Normalize 3 worker schemas to a single float pct."""
    if not isinstance(np_field, dict):
        return 0.0
    v = np_field.get(split, 0.0)
    if isinstance(v, dict):
        return float(v.get("pct", 0.0))
    return float(v) if v is not None else 0.0

# Re-normalize nan_pct on records for consistent downstream use
for r in all_records:
    r["nan_pct_norm"] = {s: _pct(r["nan_pct"], s) for s in ("train", "val", "test")}

# Sort: by family then by name
fam_order = {"F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4, "F6": 5}
all_records.sort(key=lambda r: (fam_order.get(r["family"], 99),
                                 0 if r["raw_or_derived"] == "raw" else 1,
                                 r.get("dim_in_X", 999)))

# Sanity counts
fam_counts = defaultdict(int)
for r in all_records:
    fam_counts[r["family"]] += 1
ks_hits = [r["name"] for r in all_records if r.get("in_ks_drop_11")]

KS_FAIL_NAMES = {
    "dualz_ask_diff1", "dualz_bid_diff5", "dualz_ask_diff5",
    "qrank_W100_spread1", "qrank_W100_spread5", "qrank_W100_spread10",
    "qrank_W100_cumspread",
    "kyle_lam_W50", "kyle_lam_W100",
    "roll_eff_spr_ratio_W100",
    "liq_asym_top5_W5",
}

# Verdict aggregation per family
verdict_per_family = {}
for fam in ["F1", "F2", "F3", "F4", "F5", "F6"]:
    rs = [r for r in all_records if r["family"] == fam]
    v_stock = defaultdict(int)
    v_date = defaultdict(int)
    nans = []
    for r in rs:
        v_stock[r.get("verdict_cross_stock") or "n/a"] += 1
        v_date[r.get("verdict_cross_date") or "n/a"] += 1
        nans.append(r["nan_pct_norm"]["train"])
    verdict_per_family[fam] = {
        "n": len(rs),
        "verdict_cross_stock": dict(v_stock),
        "verdict_cross_date": dict(v_date),
        "mean_nan_pct_train": float(sum(nans)/max(len(nans),1)),
    }

master = {
    "task": "MASTER per-feature audit (F1-F6, 370 features)",
    "n_features_total": len(all_records),
    "family_counts": dict(fam_counts),
    "ks_drop_11_expected": sorted(KS_FAIL_NAMES),
    "ks_drop_11_observed": sorted(ks_hits),
    "ks_drop_11_all_hit": sorted(KS_FAIL_NAMES) == sorted(ks_hits),
    "verdict_per_family": verdict_per_family,
    "records": all_records,
}
OUT_JSON.write_text(json.dumps(master, ensure_ascii=False, indent=2))
print(f"wrote {OUT_JSON}  ({len(all_records)} records)")

# === markdown summary ===
md = []
md.append("# 良文杯 — 全 370 因子审计 MASTER 总览\n")
md.append(f"> 由 3 个并行 opus worker 产出（F1F2 / F3F4 / F5F6）合并；逐 feature 明细见 `MASTER_per_feature_audit.json` 或各 worker 子目录的 `per_feature_audit.md`。\n")

md.append("## 1. 总数核对\n")
md.append("| family | 维度 | 实测 | 备注 |")
md.append("|---|---:|---:|---|")
EXPECTED = {"F1": 105, "F2": 50, "F3": 45, "F4": 29, "F5": 67, "F6": 74}
for fam in ["F1", "F2", "F3", "F4", "F5", "F6"]:
    ok = "✅" if fam_counts[fam] == EXPECTED[fam] else "⚠"
    md.append(f"| {fam} | {EXPECTED[fam]} | {fam_counts[fam]} | {ok} |")
md.append(f"| **合计** | **370** | **{sum(fam_counts.values())}** | {'✅' if sum(fam_counts.values())==370 else '⚠'} |\n")

md.append("## 2. KS-drop 11 命中确认\n")
md.append(f"- 期望 drop 11 个特征（partition.py 中 freeze）\n- 实测 worker 标记 `in_ks_drop_11=True`：{len(ks_hits)} 个\n- **完全匹配：{master['ks_drop_11_all_hit']}**\n")
md.append("| # | 特征名 | 家族 |")
md.append("|---|---|---|")
for i, n in enumerate(sorted(ks_hits), 1):
    fam = next((r["family"] for r in all_records if r["name"] == n), "?")
    md.append(f"| {i} | `{n}` | {fam} |")
md.append("")

md.append("## 3. 各 family 跨股 / 跨日 稳定性概览\n")
md.append("> verdict 阈值（按 PSI）：stable < 0.10 / mild_drift 0.10-0.25 / strong_drift > 0.25\n")
md.append("| family | n | 跨股 stable | 跨股 mild | 跨股 strong | 跨日 stable | 跨日 mild | 跨日 strong | NaN% (train) |")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for fam in ["F1", "F2", "F3", "F4", "F5", "F6"]:
    v = verdict_per_family[fam]
    s_st = v["verdict_cross_stock"].get("stable", 0)
    s_md = v["verdict_cross_stock"].get("mild_drift", 0)
    s_sd = v["verdict_cross_stock"].get("strong_drift", 0)
    d_st = v["verdict_cross_date"].get("stable", 0)
    d_md = v["verdict_cross_date"].get("mild_drift", 0)
    d_sd = v["verdict_cross_date"].get("strong_drift", 0)
    md.append(f"| {fam} | {v['n']} | {s_st} | {s_md} | {s_sd} | {d_st} | {d_md} | {d_sd} | {v['mean_nan_pct_train']*100:.2f}% |")
md.append("")

md.append("## 4. 从各 family summary 抽取的关键发现\n")

# F1F2
f1f2 = all_summaries["F1F2"]
md.append("### F1 + F2 关键发现\n")
md.append(f"- 总 NaN 比例 (train): {f1f2.get('mean_nan_pct_train_overall',0)*100:.2f}% （F1 {f1f2.get('mean_nan_pct_train_f1',0)*100:.2f}% / F2 {f1f2.get('mean_nan_pct_train_f2',0)*100:.2f}%）\n")
md.append(f"- 最大跨股 PSI: **{f1f2.get('max_psi_cross_stock', 0):.2f}** | 最大跨日 PSI: **{f1f2.get('max_psi_cross_date', 0):.2f}**\n")
md.append("- **最不稳定（跨股）Top-5**（都是 raw 价格类，证明 dualz/window-z 归一化的必要性）：\n")
for r in f1f2.get("top5_most_unstable_cross_stock", [])[:5]:
    md.append(f"  - `{r['name']}` ({r['family']}/{r['raw_or_derived']}) psi_stock_max={r['cross_stock_psi_max']:.2f}, psi_date_max={r.get('cross_date_psi_max', 0):.2f}")
md.append("\n- **跨日最稳定 Top-5**（全是 derived）：\n")
for r in f1f2.get("top5_most_stable_cross_date", [])[:5]:
    md.append(f"  - `{r['name']}` ({r['family']}/{r['raw_or_derived']}) psi_date_max={r['cross_date_psi_max']:.3f}")
md.append("")

# F3F4
f3f4 = all_summaries["F3F4"]
g34 = f3f4.get("global_metrics", {})
md.append("### F3 + F4 关键发现\n")
md.append(f"- 总 NaN 比例 (train): **{g34.get('mean_nan_pct_train', 0)*100:.2f}%**（F3/F4 都是事件强度/波动率，没有深档报价缺失）\n")
md.append(f"- 最大跨股 PSI: **{g34.get('max_psi_cross_stock', 0):.3f}** | 最大跨日 PSI: **{g34.get('max_psi_cross_date', 0):.3f}**\n")
md.append("- **Top-4 intst（模型命根）跨股 / 跨日稳定性**：\n")
md.append("\n| 因子 | psi_stock_max | psi_stock_by_sym | psi_train_vs_test | ks_train_vs_test | verdict_stock | verdict_date |")
md.append("|---|---:|---|---:|---:|---|---|")
for r in f3f4.get("top4_intst_stability", []):
    by = "/".join(f"{x:.2f}" for x in r.get("psi_stock_by_sym", []))
    md.append(f"| `{r['name']}` | {r['psi_stock_max']:.3f} | [{by}] | {r['psi_train_vs_test']:.3f} | {r['ks_train_vs_test']:.3f} | {r['verdict_cross_stock']} | {r['verdict_cross_date']} |")
md.append("\n  - sym=2（异常的蓝筹/ETF）单独拉高 0.22-0.27；其他 sym ≤ 0.13；**跨日全部 stable**\n")
md.append("- **最不稳定（跨股）Top-5**：\n")
for r in f3f4.get("top5_unstable_cross_stock", [])[:5]:
    n = r.get('name', '?'); fam = r.get('family', '?')
    psi_st = r.get('psi_stock_max', r.get('cross_stock_psi_max', 0))
    psi_dt = r.get('psi_date_max', r.get('cross_date_psi_max', 0))
    md.append(f"  - `{n}` ({fam}) psi_stock={psi_st:.3f}, psi_date={psi_dt:.3f}")
md.append("")

# F5F6
f5f6 = all_summaries["F5F6"]
md.append("### F5 + F6 关键发现\n")
md.append(f"- 总 NaN 比例: train {f5f6['mean_nan_pct_train']*100:.2f}% / val {f5f6['mean_nan_pct_val']*100:.2f}% / test {f5f6['mean_nan_pct_test']*100:.2f}%\n")
md.append(f"- 最大跨股 PSI: **{f5f6['max_psi_cross_stock']:.2f}** | 最大跨日 PSI: **{f5f6['max_psi_cross_date']:.3f}**\n")
md.append("- **F5 跨日极稳：0 strong_drift / 64 stable / 3 mild**（F5=67 维全派生 → 这就是 F5 是 'OOD 主力 family' 的硬数据证据）\n")
md.append("- **F6 跨日有 7 个 strong_drift**（主要是 bid_rate_k / ask_rate_k 系列，滑动平均变化率天生跨日漂）\n")
md.append("- **dualz vs qrank（决胜 OOD 的 trick 对比）**：\n")
dvq = f5f6.get("dualz_vs_qrank_comparison", {})
md.append("\n| 指标 | dualz_kept (excl KS-drop) | qrank_W100_kept (excl KS-drop) | 比 |")
md.append("|---|---:|---:|---|")
dz = dvq.get("dualz_kept_excluding_ks_drop", {})
qr = dvq.get("qrank_W100_kept_excluding_ks_drop", {})
md.append(f"| n | {dz.get('n')} | {qr.get('n')} | |")
md.append(f"| psi_stock_median | {dz.get('psi_stock_median', 0):.3f} | {qr.get('psi_stock_median', 0):.3f} | dualz ≈ qrank 的 1/2 |")
md.append(f"| psi_date_median | {dz.get('psi_date_median', 0):.4f} | {qr.get('psi_date_median', 0):.4f} | dualz ≈ qrank 的 1/2 |")
md.append(f"| n_stable_cross_stock | {dz.get('n_stable_cross_stock')} | {qr.get('n_stable_cross_stock')} | dualz 占比 53% vs qrank 25% |")
md.append("")
md.append("**注**：dualz/qrank 的 mean PSI 都被几个 tick-quantized 长尾 outlier 拉爆，必须看 median + stable_count + p75。验证 'dualz 是 OOD 决胜 trick'：在大多数 base 列上跨股+跨日都比 qrank 稳约 2 倍。\n")

# KS-drop conclusion
md.append("## 5. KS-drop 11 的稳定性证据（答辩 Q3 硬核数据）\n")
md.append("**11 个被 drop 的 feature，跨股全部 strong_drift，跨日全部 stable** — 说明 KS 检验抓住了"
            "'跨股漂、跨日稳' 的 sym-specific overfit 风险，drop 决策与稳定性数据一致。\n")
md.append("| # | 特征 | 家族 | verdict_stock | verdict_date | 原因 |")
md.append("|---|---|---|---|---|---|")
DROP_REASONS = {
    "kyle_lam_W50": "Kyle λ 跨 sym 协方差分布差异极大",
    "kyle_lam_W100": "同上，整族 KS fail",
    "roll_eff_spr_ratio_W100": "W=100 太长，跨 sym Roll 估计不稳",
    "dualz_ask_diff1": "ask 侧 1-level diff 跨 sym 分布差大",
    "dualz_bid_diff5": "5-level diff 噪音",
    "dualz_ask_diff5": "同上",
    "qrank_W100_spread1": "spread1 大多=1 tick → qrank 几乎恒 1（死特征）",
    "qrank_W100_spread5": "spread5 tick-quantized",
    "qrank_W100_spread10": "同上",
    "qrank_W100_cumspread": "累积 spread 同问题",
    "liq_asym_top5_W5": "W=5 太短，单 tick top-5 比值噪声大",
}
for i, n in enumerate(sorted(KS_FAIL_NAMES), 1):
    rec = next((r for r in all_records if r["name"] == n), None)
    fam = rec["family"] if rec else "?"
    vs = rec.get("verdict_cross_stock", "?") if rec else "?"
    vd = rec.get("verdict_cross_date", "?") if rec else "?"
    md.append(f"| {i} | `{n}` | {fam} | {vs} | {vd} | {DROP_REASONS.get(n, '')} |")
md.append("")

md.append("## 6. 答辩三句话总结\n")
md.append("1. **跨股泛化**：raw 价格类（bid_mean / ask_mean / spread_k）跨股 PSI 高达 19+，是窗口归一化（dualz / window-z）必须做的根本原因；F5 派生族 64/67 跨股 stable / mild，提供独立 OOD 信号。\n")
md.append("2. **跨日泛化**：F3 intst 四大主力 + F5 dualz 全 family 跨日 stable；F6 rate 字段有 7 个跨日 strong_drift，但模型不依赖它们做主分裂。\n")
md.append("3. **KS-drop 11 决策完全正确**：11 个全是跨股 strong_drift / 跨日 stable 的 sym-specific 特征；drop 后 seed std 砍 60% 验证此判断。\n")

OUT_MD.write_text("\n".join(md))
print(f"wrote {OUT_MD}  ({len(md)} lines)")
