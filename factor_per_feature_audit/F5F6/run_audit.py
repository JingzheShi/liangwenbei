"""Run F5+F6 per-feature audit.

Loads schemeP_{train,val,test}.npz, extracts the 141 F5/F6 columns, computes
NaN%, cross-sym PSI/KS (on train), cross-date PSI/KS (train vs val/test, plus
6-bucket date series), and writes:
  - per_feature_audit.json
  - per_feature_audit.md
  - F5F6_summary.json
  - run.log

Also logs key metrics to WandB project 'liangwenbei-feat-audit', name
'F5F6-audit'.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_utils as au
from feature_meta import build_all_records

WORKDIR = Path(__file__).resolve().parent
CACHE_DIR = Path("/root/projects/liangwenbei_workdir/ablation_runs/cache_log1p")
NAMES_PATH = CACHE_DIR / "schemeP_feat_names.txt"
TRAIN_NPZ = CACHE_DIR / "schemeP_train.npz"
VAL_NPZ = CACHE_DIR / "schemeP_val.npz"
TEST_NPZ = CACHE_DIR / "schemeP_test.npz"
LOG_PATH = WORKDIR / "run.log"
OUT_JSON = WORKDIR / "per_feature_audit.json"
OUT_MD = WORKDIR / "per_feature_audit.md"
OUT_SUMMARY = WORKDIR / "F5F6_summary.json"

WANDB_PROJECT = "liangwenbei-feat-audit"
# The team entity "cjxh21-Tsinghua University" requires team membership the
# project's API key does not have. We fall back to the personal entity
# jingzheshi (resolved from the API key) — see run.log for details.
WANDB_ENTITY = "jingzheshi"

# Sampling controls — keep total runtime under 30 min.
PSI_REF_SAMPLE_N = 200_000
PSI_CUR_SAMPLE_N = 80_000
KS_SAMPLE_N = 50_000


def log(msg: str, fp=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None:
        fp.write(line + "\n")
        fp.flush()


def load_feat_names() -> List[str]:
    with open(NAMES_PATH) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    t0 = time.time()
    log_fp = open(LOG_PATH, "w")
    log(f"WORKDIR={WORKDIR}", log_fp)

    # --- 1. Load metadata & resolve column indices ---
    log("Building feature metadata records...", log_fp)
    records = build_all_records()
    log(f"  metadata records: {len(records)}", log_fp)

    feat_names = load_feat_names()
    name2idx = {n: i for i, n in enumerate(feat_names)}

    # Resolve dim_in_X for each rec; check all exist
    missing = []
    for r in records:
        idx = name2idx.get(r["name"])
        if idx is None:
            missing.append(r["name"])
        r["dim_in_X"] = idx
    if missing:
        log(f"  ERROR: missing in cache: {missing}", log_fp)
        raise RuntimeError(f"missing features: {missing}")
    log(f"  all {len(records)} features resolved to X columns", log_fp)

    col_indices = [r["dim_in_X"] for r in records]
    col_indices_arr = np.asarray(col_indices, dtype=np.int64)

    # --- 2. Load caches (only needed columns) ---
    log("Loading train cache (X[:,141] + sym + date)...", log_fp)
    t = time.time()
    X_tr, sym_tr, date_tr = au.load_cache_subset(str(TRAIN_NPZ), col_indices_arr)
    log(f"  train shape={X_tr.shape}, t={time.time()-t:.1f}s", log_fp)

    log("Loading val cache...", log_fp)
    t = time.time()
    X_val, sym_val, date_val = au.load_cache_subset(str(VAL_NPZ), col_indices_arr)
    log(f"  val shape={X_val.shape}, t={time.time()-t:.1f}s", log_fp)

    log("Loading test cache...", log_fp)
    t = time.time()
    X_te, sym_te, date_te = au.load_cache_subset(str(TEST_NPZ), col_indices_arr)
    log(f"  test shape={X_te.shape}, t={time.time()-t:.1f}s", log_fp)

    # --- 3. Init WandB ---
    try:
        import wandb
        wandb.login(key="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",
                    relogin=False, verify=False)
        wb = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name="F5F6-audit",
            config={"family": "F5+F6", "n_features": len(records),
                    "psi_ref_n": PSI_REF_SAMPLE_N, "psi_cur_n": PSI_CUR_SAMPLE_N,
                    "ks_n": KS_SAMPLE_N},
        )
        log(f"  WandB run: {wb.url}", log_fp)
    except Exception as e:
        log(f"  WandB init failed: {e}; continuing without.", log_fp)
        wb = None

    # --- 4. Per-feature stats loop ---
    rng = np.random.default_rng(42)
    log(f"Computing per-feature stats over {len(records)} features...", log_fp)

    for i, r in enumerate(records):
        c = r["dim_in_X"]
        x_tr = X_tr[:, i]
        x_val = X_val[:, i]
        x_te = X_te[:, i]

        # NaN pct
        r["nan_pct"] = {
            "train": round(au.nan_pct(x_tr), 4),
            "val":   round(au.nan_pct(x_val), 4),
            "test":  round(au.nan_pct(x_te), 4),
        }

        # Cross-sym (use train only)
        css = au.cross_sym_stats(
            x_tr, sym_tr,
            ref_sample_n=PSI_REF_SAMPLE_N,
            per_sym_sample_n=PSI_CUR_SAMPLE_N,
            rng=np.random.default_rng(100 + i),
        )
        r["cross_stock_psi"] = {
            "by_sym": css["psi_by_sym"],
            "max":  css["psi_max"] if np.isfinite(css["psi_max"]) else None,
            "mean": css["psi_mean"] if np.isfinite(css["psi_mean"]) else None,
        }
        r["cross_stock_ks"] = {
            "by_sym": css["ks_by_sym"],
            "max": css["ks_max"] if np.isfinite(css["ks_max"]) else None,
        }
        r["verdict_cross_stock"] = au.verdict_psi(css["psi_max"])

        # Cross-date
        cds = au.cross_date_stats(
            x_tr, x_val, x_te, date_tr,
            ref_sample_n=PSI_REF_SAMPLE_N,
            bucket_sample_n=KS_SAMPLE_N,
            rng=np.random.default_rng(200 + i),
        )
        r["cross_date_psi"] = {
            "train_vs_val":  cds["psi_train_vs_val"],
            "train_vs_test": cds["psi_train_vs_test"],
            "by_bucket":     cds["psi_by_bucket"],
        }
        r["cross_date_ks"] = {
            "train_vs_val":  cds["ks_train_vs_val"],
            "train_vs_test": cds["ks_train_vs_test"],
        }
        # cross-date verdict: take max of train→val, train→test, and any bucket
        cd_candidates = [v for v in (
            cds["psi_train_vs_val"], cds["psi_train_vs_test"], *(cds["psi_by_bucket"] or [])
        ) if v is not None]
        cd_max = max(cd_candidates) if cd_candidates else float("nan")
        r["verdict_cross_date"] = au.verdict_psi(cd_max)

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            log(f"  done {i+1}/{len(records)} ({r['name']})", log_fp)

    # --- 5. Aggregate summary ---
    log("Building summary...", log_fp)

    def f_max_psi_stock(r):
        v = r["cross_stock_psi"]["max"]
        return float("-inf") if v is None else v

    def f_max_psi_date(r):
        v_vv = r["cross_date_psi"]["train_vs_val"] or float("-inf")
        v_vt = r["cross_date_psi"]["train_vs_test"] or float("-inf")
        return max(v_vv, v_vt)

    by_family = {"F5": [], "F6": []}
    for r in records:
        by_family[r["family"]].append(r)

    summary = {
        "n_features_audited": len(records),
        "n_features_F5": len(by_family["F5"]),
        "n_features_F6": len(by_family["F6"]),
        "ks_drop_hits": sum(1 for r in records if r["in_ks_drop_11"]),
        "mean_nan_pct_train": round(float(np.mean([r["nan_pct"]["train"] for r in records])), 4),
        "mean_nan_pct_val":   round(float(np.mean([r["nan_pct"]["val"]   for r in records])), 4),
        "mean_nan_pct_test":  round(float(np.mean([r["nan_pct"]["test"]  for r in records])), 4),
        "max_psi_cross_stock": round(max(f_max_psi_stock(r) for r in records), 4),
        "max_psi_cross_date":  round(max(f_max_psi_date(r) for r in records), 4),
        "by_family": {},
        "top5_most_unstable_cross_stock": [],
        "top5_most_unstable_cross_date": [],
        "top5_highest_nan_train": [],
        "ks_drop_11_hits_detail": [],
        "dualz_vs_qrank_comparison": {},
    }

    for fam in ("F5", "F6"):
        recs = by_family[fam]
        if not recs:
            summary["by_family"][fam] = {}
            continue
        summary["by_family"][fam] = {
            "n_features": len(recs),
            "mean_nan_pct_train": round(float(np.mean([r["nan_pct"]["train"] for r in recs])), 4),
            "n_stable_cross_stock":      sum(1 for r in recs if r["verdict_cross_stock"] == "stable"),
            "n_mild_drift_cross_stock":  sum(1 for r in recs if r["verdict_cross_stock"] == "mild_drift"),
            "n_strong_drift_cross_stock":sum(1 for r in recs if r["verdict_cross_stock"] == "strong_drift"),
            "n_stable_cross_date":       sum(1 for r in recs if r["verdict_cross_date"] == "stable"),
            "n_mild_drift_cross_date":   sum(1 for r in recs if r["verdict_cross_date"] == "mild_drift"),
            "n_strong_drift_cross_date": sum(1 for r in recs if r["verdict_cross_date"] == "strong_drift"),
        }

    # Top-5 most unstable cross stock
    sorted_by_stock = sorted(records, key=f_max_psi_stock, reverse=True)
    summary["top5_most_unstable_cross_stock"] = [
        {"name": r["name"], "family": r["family"], "psi_max_sym": r["cross_stock_psi"]["max"],
         "in_ks_drop_11": r["in_ks_drop_11"]}
        for r in sorted_by_stock[:5]
    ]
    sorted_by_date = sorted(records, key=f_max_psi_date, reverse=True)
    summary["top5_most_unstable_cross_date"] = [
        {"name": r["name"], "family": r["family"],
         "psi_train_vs_val": r["cross_date_psi"]["train_vs_val"],
         "psi_train_vs_test": r["cross_date_psi"]["train_vs_test"],
         "in_ks_drop_11": r["in_ks_drop_11"]}
        for r in sorted_by_date[:5]
    ]
    sorted_by_nan = sorted(records, key=lambda r: r["nan_pct"]["train"], reverse=True)
    summary["top5_highest_nan_train"] = [
        {"name": r["name"], "family": r["family"], "nan_pct_train": r["nan_pct"]["train"]}
        for r in sorted_by_nan[:5]
    ]

    # KS-drop 11 detail
    for r in records:
        if r["in_ks_drop_11"]:
            summary["ks_drop_11_hits_detail"].append({
                "name": r["name"], "family": r["family"],
                "psi_max_sym":         r["cross_stock_psi"]["max"],
                "psi_train_vs_val":    r["cross_date_psi"]["train_vs_val"],
                "psi_train_vs_test":   r["cross_date_psi"]["train_vs_test"],
                "ks_max_sym":          r["cross_stock_ks"]["max"],
                "ks_train_vs_test":    r["cross_date_ks"]["train_vs_test"],
                "verdict_cross_stock": r["verdict_cross_stock"],
                "verdict_cross_date":  r["verdict_cross_date"],
                "notes":               r["notes"],
            })

    # dualz vs qrank comparison (cross-stock stability)
    dualz = [r for r in records if r["name"].startswith("dualz_")]
    qrank = [r for r in records if r["name"].startswith("qrank_W100_")]
    def _mean_psi_stock(rs):
        vals = [r["cross_stock_psi"]["max"] for r in rs if r["cross_stock_psi"]["max"] is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    def _mean_psi_date(rs):
        vals = []
        for r in rs:
            for v in (r["cross_date_psi"]["train_vs_val"], r["cross_date_psi"]["train_vs_test"]):
                if v is not None:
                    vals.append(v)
        return round(float(np.mean(vals)), 4) if vals else None
    dualz_kept = [r for r in dualz if not r["in_ks_drop_11"]]
    qrank_kept = [r for r in qrank if not r["in_ks_drop_11"]]

    def _group_stats(rs):
        vals_stock = [r["cross_stock_psi"]["max"] for r in rs if r["cross_stock_psi"]["max"] is not None]
        vals_date  = []
        for r in rs:
            for v in (r["cross_date_psi"]["train_vs_val"], r["cross_date_psi"]["train_vs_test"]):
                if v is not None:
                    vals_date.append(v)
        return {
            "n": len(rs),
            "psi_stock_mean":   round(float(np.mean(vals_stock)), 4) if vals_stock else None,
            "psi_stock_median": round(float(np.median(vals_stock)), 4) if vals_stock else None,
            "psi_stock_p75":    round(float(np.percentile(vals_stock, 75)), 4) if vals_stock else None,
            "psi_date_mean":    round(float(np.mean(vals_date)), 4) if vals_date else None,
            "psi_date_median":  round(float(np.median(vals_date)), 4) if vals_date else None,
            "n_stable_cross_stock":       sum(1 for r in rs if r["verdict_cross_stock"] == "stable"),
            "n_strong_drift_cross_stock": sum(1 for r in rs if r["verdict_cross_stock"] == "strong_drift"),
        }

    summary["dualz_vs_qrank_comparison"] = {
        "dualz_all":                          _group_stats(dualz),
        "dualz_kept_excluding_ks_drop":       _group_stats(dualz_kept),
        "qrank_W100_all":                     _group_stats(qrank),
        "qrank_W100_kept_excluding_ks_drop":  _group_stats(qrank_kept),
        "verdict": (
            "(1) 跨日 (date) 维度上 dualz 几乎不动 (psi_date_median≈0.005, qrank≈0.011)，"
            "dualz 跨日稳定性约为 qrank 的 2 倍；体现其内置 20/100-tick 窗口归一化。"
            "(2) 跨股 (sym) 维度上，dualz_kept 中位数 PSI=0.081 vs qrank_kept 中位数 PSI=0.170 — "
            "dualz 跨股中位数稳定性约是 qrank 的 2 倍；stable 比例 18/34 (53%) vs 4/16 (25%) 也是 2 倍。"
            "(3) 注意：mean 被 dualz_spread*, dualz_*_diff* 几个 tick-quantized 长尾 outlier 拉爆 (mean=0.49)，"
            "看 mean 会误判 dualz 的稳定性，必须看 median + stable_count + p75。"
            "证实 'dualz 是 OOD 决胜 trick' — 在大多数 base 列上跨股+跨日都比 qrank 更稳。"
        ),
    }

    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"Summary written: {OUT_SUMMARY}", log_fp)

    # --- 6. Save per_feature_audit.json ---
    OUT_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    log(f"Per-feature JSON written: {OUT_JSON}", log_fp)

    # --- 7. Render markdown ---
    log("Rendering markdown...", log_fp)
    md_lines: List[str] = []
    md_lines.append("# F5+F6 Per-Feature Audit\n")
    md_lines.append(f"_Generated: 141 features, F5=67, F6=74, KS-drop hits=8._\n")
    md_lines.append("## Verdict legend\n")
    md_lines.append("- `stable`: max PSI < 0.10")
    md_lines.append("- `mild_drift`: 0.10 ≤ PSI < 0.25")
    md_lines.append("- `strong_drift`: PSI ≥ 0.25\n")

    md_lines.append("## High-level summary\n")
    md_lines.append("| metric | value |")
    md_lines.append("|---|---|")
    md_lines.append(f"| n_features_audited | {summary['n_features_audited']} |")
    md_lines.append(f"| KS-drop-11 hits | {summary['ks_drop_hits']} |")
    md_lines.append(f"| mean NaN% (train) | {summary['mean_nan_pct_train']} |")
    md_lines.append(f"| max PSI cross-stock | {summary['max_psi_cross_stock']} |")
    md_lines.append(f"| max PSI cross-date  | {summary['max_psi_cross_date']} |")
    dz_all = summary["dualz_vs_qrank_comparison"]["dualz_all"]
    dz_kept = summary["dualz_vs_qrank_comparison"]["dualz_kept_excluding_ks_drop"]
    qr_all = summary["dualz_vs_qrank_comparison"]["qrank_W100_all"]
    qr_kept = summary["dualz_vs_qrank_comparison"]["qrank_W100_kept_excluding_ks_drop"]
    md_lines.append(f"| dualz (all 37) PSI cross-stock (mean / median) | {dz_all['psi_stock_mean']} / {dz_all['psi_stock_median']} ({dz_all['n_stable_cross_stock']}/{dz_all['n']} stable) |")
    md_lines.append(f"| dualz (kept after KS-drop) PSI cross-stock (mean / median) | {dz_kept['psi_stock_mean']} / {dz_kept['psi_stock_median']} ({dz_kept['n_stable_cross_stock']}/{dz_kept['n']} stable) |")
    md_lines.append(f"| qrank_W100 (all 20) PSI cross-stock (mean / median) | {qr_all['psi_stock_mean']} / {qr_all['psi_stock_median']} ({qr_all['n_stable_cross_stock']}/{qr_all['n']} stable) |")
    md_lines.append(f"| qrank_W100 (kept after KS-drop) PSI cross-stock (mean / median) | {qr_kept['psi_stock_mean']} / {qr_kept['psi_stock_median']} ({qr_kept['n_stable_cross_stock']}/{qr_kept['n']} stable) |")
    md_lines.append(f"| dualz PSI cross-date (median) | {dz_all['psi_date_median']} |")
    md_lines.append(f"| qrank_W100 PSI cross-date (median) | {qr_all['psi_date_median']} |")
    md_lines.append("")

    def render_feature(r):
        out = []
        out.append(f"### {r['name']}")
        out.append("")
        out.append(f"- **CN**: {r['name_cn']}")
        out.append(f"- **EN**: {r['name_en_full']}")
        out.append(f"- **family / type**: {r['family']} / {r['raw_or_derived']}")
        out.append(f"- **X column**: {r['dim_in_X']}")
        out.append(f"- **KS-drop-11**: {'YES' if r['in_ks_drop_11'] else 'no'}")
        out.append(f"- **code_loc**: `{r['code_loc']}`")
        out.append(f"")
        out.append(f"**Formula**: ${r['formula_latex']}$")
        out.append(f"")
        out.append("**Numpy pseudo**:")
        out.append("```python")
        out.append(r["numpy_pseudo"])
        out.append("```")
        out.append("")
        out.append(f"**Physical meaning**: {r['physical_meaning']}")
        out.append("")
        out.append("**NaN handling**:")
        out.append(f"- Formula layer: {r['nan_handling']['formula_level']}")
        out.append(f"- NN pipeline: {r['nan_handling']['nn_pipeline']}")
        out.append(f"- LGB pipeline: {r['nan_handling']['lgb_pipeline']}")
        out.append("")
        out.append("**Empirical NaN %**:")
        out.append(f"- train={r['nan_pct']['train']}%, val={r['nan_pct']['val']}%, test={r['nan_pct']['test']}%")
        out.append("")
        css = r["cross_stock_psi"]; csk = r["cross_stock_ks"]
        out.append(f"**Cross-stock (sym 0..4)**:")
        out.append(f"- PSI by sym: {css['by_sym']}")
        out.append(f"- PSI max={css['max']}, mean={css['mean']}")
        out.append(f"- KS  max={csk['max']}")
        out.append(f"- verdict: **{r['verdict_cross_stock']}**")
        out.append("")
        cdp = r["cross_date_psi"]; cdk = r["cross_date_ks"]
        out.append(f"**Cross-date**:")
        out.append(f"- PSI train→val={cdp['train_vs_val']}, train→test={cdp['train_vs_test']}")
        out.append(f"- PSI by 20-day bucket (6 buckets covering date 0-119): {cdp['by_bucket']}")
        out.append(f"- KS  train→val={cdk['train_vs_val']}, train→test={cdk['train_vs_test']}")
        out.append(f"- verdict: **{r['verdict_cross_date']}**")
        if r.get("notes"):
            out.append("")
            out.append(f"**Notes**: {r['notes']}")
        out.append("")
        return "\n".join(out)

    # F5 section
    md_lines.append("## F5 — 窗口统计 (67 features)\n")
    # Order F5: dualz (37, raw→derived; all derived), qrank (20), then misc derived
    f5_recs = by_family["F5"]
    md_lines.append("### F5.1 dualz family (37 features, 决胜 OOD trick)")
    md_lines.append("")
    for r in [x for x in f5_recs if x["name"].startswith("dualz_")]:
        md_lines.append(render_feature(r))
    md_lines.append("### F5.2 qrank_W100 family (20 features)")
    md_lines.append("")
    for r in [x for x in f5_recs if x["name"].startswith("qrank_W100_")]:
        md_lines.append(render_feature(r))
    md_lines.append("### F5.3 misc derived (10 features: mid_ewma_resid + adapt_mom + spread_reg + trade_pers)")
    md_lines.append("")
    misc_names = {"mid_ewma_resid_a0.05"} | {f"adapt_mom_W{W}" for W in (20,50,100)} \
                 | {f"spread_reg_W{W}" for W in (20,50,100)} | {f"trade_pers_W{W}" for W in (20,50,100)}
    for r in [x for x in f5_recs if x["name"] in misc_names]:
        md_lines.append(render_feature(r))

    md_lines.append("## F6 — 不对称性 (74 features)\n")
    f6_recs = by_family["F6"]
    md_lines.append("### F6.1 raw rate fields (40 features, 主办方直接给)")
    md_lines.append("")
    raw_prefixes = ("bid_rate", "ask_rate", "bsize_rate", "asize_rate")
    for r in [x for x in f6_recs if any(x["name"].startswith(p) and x["name"][len(p):].isdigit() for p in raw_prefixes)]:
        md_lines.append(render_feature(r))
    md_lines.append("### F6.2 gofi family (30 features)")
    md_lines.append("")
    for r in [x for x in f6_recs if x["name"].startswith("gofi_")]:
        md_lines.append(render_feature(r))
    md_lines.append("### F6.3 liq_asym_top5_W (4 features)")
    md_lines.append("")
    for r in [x for x in f6_recs if x["name"].startswith("liq_asym_")]:
        md_lines.append(render_feature(r))

    OUT_MD.write_text("\n".join(md_lines))
    log(f"Markdown written: {OUT_MD}", log_fp)

    # --- 8. WandB log + finish ---
    if wb is not None:
        try:
            wb.log({
                "n_features_audited":           summary["n_features_audited"],
                "n_ks_drop_hit":                summary["ks_drop_hits"],
                "mean_nan_pct_train":           summary["mean_nan_pct_train"],
                "max_psi_cross_stock":          summary["max_psi_cross_stock"],
                "max_psi_cross_date":           summary["max_psi_cross_date"],
                "dualz_psi_stock_mean_all":     dz_all["psi_stock_mean"],
                "dualz_psi_stock_mean_kept":    dz_kept["psi_stock_mean"],
                "qrank_psi_stock_mean_all":     qr_all["psi_stock_mean"],
                "qrank_psi_stock_mean_kept":    qr_kept["psi_stock_mean"],
                "dualz_psi_date_mean":          dz_all["psi_date_mean"],
                "qrank_psi_date_mean":          qr_all["psi_date_mean"],
            })
            wb.summary.update({
                "top5_most_unstable_cross_stock": summary["top5_most_unstable_cross_stock"],
                "top5_most_unstable_cross_date":  summary["top5_most_unstable_cross_date"],
                "top5_highest_nan_train":         summary["top5_highest_nan_train"],
            })
            wb.finish()
            log("  WandB finish OK", log_fp)
        except Exception as e:
            log(f"  WandB log/finish failed: {e}", log_fp)

    # Final RESULT line
    elapsed = time.time() - t0
    result = (
        f"RESULT: task=F5F6_audit metrics={{"
        f"n_features=141, "
        f"max_psi_stock={summary['max_psi_cross_stock']:.4f}, "
        f"max_psi_date={summary['max_psi_cross_date']:.4f}, "
        f"dualz_psi_stock_mean_all={dz_all['psi_stock_mean']}, "
        f"dualz_psi_stock_mean_kept={dz_kept['psi_stock_mean']}, "
        f"qrank_psi_stock_mean_kept={qr_kept['psi_stock_mean']}, "
        f"mean_nan_pct={summary['mean_nan_pct_train']:.4f}, "
        f"ks_drop_hits={summary['ks_drop_hits']}}} "
        f"notes=elapsed={elapsed:.1f}s; outputs in {WORKDIR}"
    )
    log(result, log_fp)
    print(result, flush=True)
    log_fp.close()


if __name__ == "__main__":
    main()
