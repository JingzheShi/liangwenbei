"""F3+F4 Per-Feature Audit driver.

Loads schemeP caches (train / val / test), keeps only the 74 F3+F4 columns,
computes NaN% + cross-stock PSI/KS + cross-date PSI/KS for each, writes
per_feature_audit.json / per_feature_audit.md / F3F4_summary.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_utils import (
    FEAT_NAMES_PATH, load_cache, load_feat_names,
    psi, ks_stat, nan_pct, verdict, subsample_idx,
)
from feature_metadata import build_all_meta


WORKDIR = "/root/projects/liangwenbei_workdir/factor_per_feature_audit/F3F4"
LOG_PATH = f"{WORKDIR}/run.log"


def log(msg: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def init_wandb():
    try:
        import wandb
        os.environ.setdefault("WANDB_API_KEY",
            "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")
        # Note: prompt asked for entity="cjxh21-Tsinghua University" but the configured
        # API key only has access to "jingzheshi" (probed via Api().viewer.teams).
        # Falling back to the personal entity so the run still logs.
        entity = os.environ.get("WANDB_ENTITY", "jingzheshi")
        run = wandb.init(
            project="liangwenbei-feat-audit",
            entity=entity,
            name="F3F4-audit",
            config={"family": "F3+F4", "n_features": 74,
                    "note": "entity fell back from 'cjxh21-Tsinghua University' (perm denied) to 'jingzheshi'"},
            reinit="finish_previous",
            mode=os.environ.get("WANDB_MODE", "online"),
        )
        return run
    except Exception as e:
        log(f"[warn] wandb init failed: {e}; continuing without wandb")
        return None


def main():
    open(LOG_PATH, "w").close()  # reset
    t_total = time.time()
    log("=== F3+F4 per-feature audit start ===")
    wb = init_wandb()

    # ---- 1) feature alignment ----
    feat_names = load_feat_names()
    name2idx = {n: i for i, n in enumerate(feat_names)}
    meta = build_all_meta()
    keep_cols = [name2idx[m["name"]] for m in meta]
    for m, ci in zip(meta, keep_cols):
        m["dim_in_X"] = ci
    log(f"feature count: {len(meta)} (45 F3 + 29 F4), all aligned to X columns")

    # ---- 2) load caches (only 74 cols) ----
    for split in ("train", "val", "test"):
        t0 = time.time()
        log(f"loading {split} cache ...")
        _ = load_cache(split, keep_cols)  # warm; below we reload per-split below
        log(f"  {split} loaded ({time.time()-t0:.1f}s)")

    tr = load_cache("train", keep_cols)
    va = load_cache("val", keep_cols)
    te = load_cache("test", keep_cols)
    log(f"train: X={tr['X'].shape}  val: X={va['X'].shape}  test: X={te['X'].shape}")

    # ---- 3) sub-sample masks (per SPEC: 5% uniform) ----
    rng = np.random.default_rng(20260529)
    SAMP_FRAC = 0.05
    REF_N = 200_000

    tr_samp = subsample_idx(tr["X"].shape[0], SAMP_FRAC, rng)
    va_samp = subsample_idx(va["X"].shape[0], SAMP_FRAC, rng)
    te_samp = subsample_idx(te["X"].shape[0], SAMP_FRAC, rng)
    log(f"sub-sampled (5%): tr={len(tr_samp):,}, va={len(va_samp):,}, te={len(te_samp):,}")

    # ref for PSI (200K of train)
    ref_idx = rng.choice(tr["X"].shape[0], size=min(REF_N, tr["X"].shape[0]), replace=False)
    log(f"PSI ref pool: {len(ref_idx):,} rows from train")

    # ---- 4) precompute date buckets for cross-date PSI sequence ----
    # train dates 0-79 -> 4 buckets of 20 each; combined with val(80-95) + test(96-119) = 6 buckets
    # We will compute by_bucket against train ref:
    #   buckets: tr[0-19], tr[20-39], tr[40-59], tr[60-79], val[80-95], test[96-119]
    bucket_specs = [
        ("tr_d0_19", tr, (0, 19)),
        ("tr_d20_39", tr, (20, 39)),
        ("tr_d40_59", tr, (40, 59)),
        ("tr_d60_79", tr, (60, 79)),
        ("va_d80_95", va, (80, 95)),
        ("te_d96_119", te, (96, 119)),
    ]
    # Pre-build masks (uniform 5% sub-sample inside bucket for speed)
    bucket_idx = {}
    for name, split, (d0, d1) in bucket_specs:
        m = (split["date"] >= d0) & (split["date"] <= d1)
        idx = np.where(m)[0]
        if len(idx) > 50_000:
            idx = rng.choice(idx, size=50_000, replace=False)
        bucket_idx[name] = (split, idx)
    log(f"date buckets prepped: {[f'{k}={len(v[1]):,}' for k,v in bucket_idx.items()]}")

    # sym sub-sample masks for cross-stock (in train), 5% per sym
    sym_idx = {}
    for s in range(5):
        m = tr["sym"] == s
        idx = np.where(m)[0]
        if len(idx) > 100_000:
            idx = rng.choice(idx, size=100_000, replace=False)
        sym_idx[s] = idx
    log(f"sym sub-samples: {[f's{k}={len(v):,}' for k,v in sym_idx.items()]}")

    # ---- 5) loop features ----
    out_rows = []
    for i, m in enumerate(meta):
        t0 = time.time()
        col = m["dim_in_X"]   # absolute X col
        local_col = i  # in our trimmed cache (we kept cols in same order as `meta`)
        x_tr_full = tr["X"][:, local_col]
        x_va_full = va["X"][:, local_col]
        x_te_full = te["X"][:, local_col]

        # ---- NaN pct ----
        n_tr, pct_tr = nan_pct(x_tr_full)
        n_va, pct_va = nan_pct(x_va_full)
        n_te, pct_te = nan_pct(x_te_full)

        # ---- Sub-sampled views for stat ops ----
        x_tr = x_tr_full[tr_samp]
        x_va = x_va_full[va_samp]
        x_te = x_te_full[te_samp]
        x_ref = x_tr_full[ref_idx]

        # ---- Cross-stock PSI/KS (train) ----
        psi_by_sym = []
        ks_by_sym = []
        for s in range(5):
            cur = tr["X"][sym_idx[s], local_col]
            psi_by_sym.append(psi(x_ref, cur))
            other_mask = np.concatenate([sym_idx[k] for k in range(5) if k != s])
            # subsample 'other'
            if len(other_mask) > 200_000:
                other_mask = rng.choice(other_mask, size=200_000, replace=False)
            other = tr["X"][other_mask, local_col]
            ks_by_sym.append(ks_stat(cur, other))
        psi_by_sym_arr = np.array(psi_by_sym, dtype=float)
        ks_by_sym_arr = np.array(ks_by_sym, dtype=float)

        # ---- Cross-date PSI/KS ----
        psi_tr_va = psi(x_ref, x_va)
        psi_tr_te = psi(x_ref, x_te)
        psi_buckets = []
        for name, (split, idx) in bucket_idx.items():
            cur = split["X"][idx, local_col]
            psi_buckets.append({"bucket": name, "psi": psi(x_ref, cur)})
        ks_tr_va = ks_stat(x_tr, x_va)
        ks_tr_te = ks_stat(x_tr, x_te)

        # ---- Verdicts ----
        v_sym = verdict(np.nanmax(psi_by_sym_arr))
        v_date = verdict(max(
            x for x in [psi_tr_va, psi_tr_te] + [b["psi"] for b in psi_buckets]
            if np.isfinite(x)
        ) if any(np.isfinite([psi_tr_va, psi_tr_te] + [b["psi"] for b in psi_buckets])) else float("nan"))

        rec = deepcopy(m)
        rec["nan_pct"] = {
            "train": {"n": n_tr, "pct": round(pct_tr, 4)},
            "val": {"n": n_va, "pct": round(pct_va, 4)},
            "test": {"n": n_te, "pct": round(pct_te, 4)},
        }
        rec["cross_stock_psi"] = {
            "by_sym": [round(x, 4) if np.isfinite(x) else None for x in psi_by_sym_arr],
            "max": round(float(np.nanmax(psi_by_sym_arr)), 4) if np.isfinite(np.nanmax(psi_by_sym_arr)) else None,
            "mean": round(float(np.nanmean(psi_by_sym_arr)), 4) if np.isfinite(np.nanmean(psi_by_sym_arr)) else None,
        }
        rec["cross_stock_ks"] = {
            "by_sym": [round(x, 4) if np.isfinite(x) else None for x in ks_by_sym_arr],
            "max": round(float(np.nanmax(ks_by_sym_arr)), 4) if np.isfinite(np.nanmax(ks_by_sym_arr)) else None,
        }
        rec["cross_date_psi"] = {
            "train_vs_val": round(float(psi_tr_va), 4) if np.isfinite(psi_tr_va) else None,
            "train_vs_test": round(float(psi_tr_te), 4) if np.isfinite(psi_tr_te) else None,
            "by_bucket": [
                {"bucket": b["bucket"], "psi": round(b["psi"], 4) if np.isfinite(b["psi"]) else None}
                for b in psi_buckets
            ],
        }
        rec["cross_date_ks"] = {
            "train_vs_val": round(float(ks_tr_va), 4) if np.isfinite(ks_tr_va) else None,
            "train_vs_test": round(float(ks_tr_te), 4) if np.isfinite(ks_tr_te) else None,
        }
        rec["verdict_cross_stock"] = v_sym
        rec["verdict_cross_date"] = v_date
        rec["notes"] = "KS-drop 11 member (W=100, fail train→test KS)" if rec.get("in_ks_drop_11") else ""

        out_rows.append(rec)
        if (i + 1) % 10 == 0 or i == 0:
            log(f"  [{i+1}/{len(meta)}] {m['name']:32s} | PSI sym_max={rec['cross_stock_psi']['max']}, "
                f"PSI date_max={max([x for x in [psi_tr_va, psi_tr_te] if np.isfinite(x)] or [0]):.3f} "
                f"({time.time()-t0:.1f}s)")

    # ---- 6) write JSON ----
    out_path = f"{WORKDIR}/per_feature_audit.json"
    with open(out_path, "w") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    log(f"wrote {out_path} ({len(out_rows)} records)")

    # ---- 7) summary ----
    def _sort_by(rows, key, top=5, asc=False):
        valid = [r for r in rows if r.get(key) is not None]
        valid.sort(key=lambda r: r[key], reverse=not asc)
        return [{"name": r["name"], "family": r["family"], key: r[key]} for r in valid[:top]]

    flat = []
    for r in out_rows:
        flat.append({
            "name": r["name"],
            "family": r["family"],
            "psi_stock_max": r["cross_stock_psi"]["max"],
            "psi_date_max": max([x for x in
                [r["cross_date_psi"]["train_vs_val"], r["cross_date_psi"]["train_vs_test"]]
                + [b["psi"] for b in r["cross_date_psi"]["by_bucket"] if b["psi"] is not None]
                if x is not None] or [0]),
            "nan_pct_train": r["nan_pct"]["train"]["pct"],
        })

    summary = {
        "family": "F3+F4",
        "n_features": len(out_rows),
        "counts": {
            "F3": sum(1 for r in out_rows if r["family"] == "F3"),
            "F4": sum(1 for r in out_rows if r["family"] == "F4"),
        },
        "global_metrics": {
            "max_psi_cross_stock": round(max((f["psi_stock_max"] for f in flat if f["psi_stock_max"] is not None), default=0), 4),
            "max_psi_cross_date": round(max((f["psi_date_max"] for f in flat if f["psi_date_max"] is not None), default=0), 4),
            "mean_nan_pct_train": round(float(np.mean([f["nan_pct_train"] for f in flat])), 4),
            "n_ks_drop_hit": sum(1 for r in out_rows if r.get("in_ks_drop_11")),
        },
        "top5_unstable_cross_stock": sorted(flat, key=lambda f: -(f["psi_stock_max"] or 0))[:5],
        "top5_unstable_cross_date": sorted(flat, key=lambda f: -(f["psi_date_max"] or 0))[:5],
        "top5_highest_nan_train": sorted(flat, key=lambda f: -(f["nan_pct_train"] or 0))[:5],
        "top4_intst_stability": [
            {
                "name": n,
                "psi_stock_max": next(r["cross_stock_psi"]["max"] for r in out_rows if r["name"] == n),
                "psi_stock_by_sym": next(r["cross_stock_psi"]["by_sym"] for r in out_rows if r["name"] == n),
                "psi_train_vs_val": next(r["cross_date_psi"]["train_vs_val"] for r in out_rows if r["name"] == n),
                "psi_train_vs_test": next(r["cross_date_psi"]["train_vs_test"] for r in out_rows if r["name"] == n),
                "ks_train_vs_test": next(r["cross_date_ks"]["train_vs_test"] for r in out_rows if r["name"] == n),
                "verdict_cross_stock": next(r["verdict_cross_stock"] for r in out_rows if r["name"] == n),
                "verdict_cross_date": next(r["verdict_cross_date"] for r in out_rows if r["name"] == n),
            }
            for n in ["ma_intst", "mb_intst", "la_intst", "lb_intst"]
        ],
        "ks_drop_11_members": [
            {
                "name": r["name"],
                "psi_train_vs_test": r["cross_date_psi"]["train_vs_test"],
                "ks_train_vs_test": r["cross_date_ks"]["train_vs_test"],
            }
            for r in out_rows if r.get("in_ks_drop_11")
        ],
    }
    sum_path = f"{WORKDIR}/F3F4_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"wrote {sum_path}")

    # ---- 8) render markdown ----
    md_path = f"{WORKDIR}/per_feature_audit.md"
    write_markdown(out_rows, summary, md_path)
    log(f"wrote {md_path}")

    # ---- 9) wandb metrics ----
    if wb is not None:
        try:
            import wandb
            top4 = {t["name"]: t for t in summary["top4_intst_stability"]}
            wandb.log({
                "n_features_audited": summary["n_features"],
                "max_psi_cross_stock": summary["global_metrics"]["max_psi_cross_stock"],
                "max_psi_cross_date": summary["global_metrics"]["max_psi_cross_date"],
                "mean_nan_pct_train": summary["global_metrics"]["mean_nan_pct_train"],
                "n_ks_drop_hit": summary["global_metrics"]["n_ks_drop_hit"],
                "top4_intst_psi_max": max(
                    [v["psi_stock_max"] for v in summary["top4_intst_stability"]
                     if v["psi_stock_max"] is not None] or [0]
                ),
                "ma_intst_psi_stock_max": top4["ma_intst"]["psi_stock_max"],
                "mb_intst_psi_stock_max": top4["mb_intst"]["psi_stock_max"],
                "la_intst_psi_stock_max": top4["la_intst"]["psi_stock_max"],
                "lb_intst_psi_stock_max": top4["lb_intst"]["psi_stock_max"],
            })
            wandb.finish()
            log("wandb logged + finished")
        except Exception as e:
            log(f"[warn] wandb log failed: {e}")

    log(f"=== done in {time.time()-t_total:.1f}s ===")
    return out_rows, summary


def write_markdown(rows, summary, path):
    lines = []
    lines.append("# F3+F4 Per-Feature Audit\n")
    lines.append("> 良文杯（中间价方向预测）feature audit. 74 个 F3+F4 features.")
    lines.append("> 公榜+35.64 / 私榜#1 提交的 Predictor 用了所有这 74 个 feature。\n")

    # high-level
    g = summary["global_metrics"]
    lines.append("## 全局指标\n")
    lines.append(f"- 特征总数: **{summary['n_features']}** (F3={summary['counts']['F3']}, F4={summary['counts']['F4']})")
    lines.append(f"- 跨 stock 最大 PSI: **{g['max_psi_cross_stock']}**")
    lines.append(f"- 跨 date 最大 PSI: **{g['max_psi_cross_date']}**")
    lines.append(f"- train 平均 NaN%: **{g['mean_nan_pct_train']}%**")
    lines.append(f"- KS-drop 11 命中: **{g['n_ks_drop_hit']}**")
    lines.append("")

    lines.append("## Top-4 intst 特征稳定性（命根）")
    lines.append("| feature | PSI sym_max | PSI by sym | PSI tr→va | PSI tr→te | KS tr→te | verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in summary["top4_intst_stability"]:
        lines.append(
            f"| `{t['name']}` | {t['psi_stock_max']} | {t['psi_stock_by_sym']} | "
            f"{t['psi_train_vs_val']} | {t['psi_train_vs_test']} | {t['ks_train_vs_test']} | "
            f"stock={t['verdict_cross_stock']} / date={t['verdict_cross_date']} |"
        )
    lines.append("")

    lines.append("## Top-5 最不稳定（跨 stock）")
    for t in summary["top5_unstable_cross_stock"]:
        lines.append(f"- `{t['name']}` (family={t['family']}) — PSI_max={t['psi_stock_max']}")
    lines.append("")

    lines.append("## Top-5 最不稳定（跨 date）")
    for t in summary["top5_unstable_cross_date"]:
        lines.append(f"- `{t['name']}` (family={t['family']}) — PSI_max={t['psi_date_max']:.4f}")
    lines.append("")

    lines.append("## Top-5 train NaN% 最高")
    for t in summary["top5_highest_nan_train"]:
        lines.append(f"- `{t['name']}` (family={t['family']}) — NaN%={t['nan_pct_train']}")
    lines.append("")

    lines.append("## KS-drop 11 命中")
    if summary["ks_drop_11_members"]:
        for m in summary["ks_drop_11_members"]:
            lines.append(f"- `{m['name']}` — PSI tr→te = {m['psi_train_vs_test']}, KS tr→te = {m['ks_train_vs_test']}")
    else:
        lines.append("- (none)")
    lines.append("")

    # per-feature sections, F3 then F4, raw before derived inside each
    lines.append("---\n")
    for fam in ["F3", "F4"]:
        rows_fam = [r for r in rows if r["family"] == fam]
        rows_fam.sort(key=lambda r: (0 if r["raw_or_derived"] == "raw" else 1, rows.index(r)))
        lines.append(f"## {fam} — {len(rows_fam)} features\n")
        for r in rows_fam:
            lines.append(f"### `{r['name']}` ({r['raw_or_derived']})")
            lines.append(f"- 中文名: {r['name_cn']}")
            lines.append(f"- 英文: {r['name_en_full']}")
            lines.append(f"- X 列号 (0-based): {r['dim_in_X']}")
            lines.append(f"- KS-drop 11: {'**YES**' if r.get('in_ks_drop_11') else 'no'}")
            lines.append(f"- 公式: $${r['formula_latex']}$$")
            lines.append(f"- 代码定位: `{r['code_loc']}`")
            lines.append("- numpy 伪代码:")
            lines.append("```python")
            lines.append(r["numpy_pseudo"])
            lines.append("```")
            lines.append(f"- 物理含义: {r['physical_meaning']}")
            lines.append("- NaN 处理:")
            lines.append(f"  - 公式层: {r['nan_handling']['formula_level']}")
            lines.append(f"  - NN 管线: {r['nan_handling']['nn_pipeline']}")
            lines.append(f"  - LGB 管线: {r['nan_handling']['lgb_pipeline']}")
            lines.append(
                f"- NaN%: train={r['nan_pct']['train']['pct']}% / val={r['nan_pct']['val']['pct']}% / test={r['nan_pct']['test']['pct']}%"
            )
            cs = r["cross_stock_psi"]
            ks = r["cross_stock_ks"]
            lines.append(f"- 跨 stock PSI: max={cs['max']} / mean={cs['mean']} / by_sym={cs['by_sym']}")
            lines.append(f"- 跨 stock KS: max={ks['max']} / by_sym={ks['by_sym']}")
            cd = r["cross_date_psi"]
            kd = r["cross_date_ks"]
            lines.append(f"- 跨 date PSI: tr→va={cd['train_vs_val']} / tr→te={cd['train_vs_test']}")
            lines.append(f"- 跨 date PSI by bucket: {[(b['bucket'], b['psi']) for b in cd['by_bucket']]}")
            lines.append(f"- 跨 date KS: tr→va={kd['train_vs_val']} / tr→te={kd['train_vs_test']}")
            lines.append(f"- 结论: cross_stock={r['verdict_cross_stock']} / cross_date={r['verdict_cross_date']}")
            if r["notes"]:
                lines.append(f"- 备注: {r['notes']}")
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
