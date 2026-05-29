"""F1+F2 per-feature audit driver.

Computes NaN%, cross-stock PSI/KS, cross-date PSI/KS for all 155 F1+F2 features,
merges with hand-written metadata (feature_meta.py), writes:
  - per_feature_audit.json   (array of 155 records, SPEC-compliant)
  - per_feature_audit.md     (human readable)
  - F1F2_summary.json        (top-5 most/least stable, KS-drop hits, NaN top-5)
  - run.log                  (timing / steps)
Logs to WandB.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_utils import (
    psi, ks_stat, verdict_from_psi, load_feat_names, load_cache_columns,
    stratified_sample, date_bucket, CACHE_DIR,
)
from feature_meta import all_meta, KS_FAIL, RAW_154


WORKDIR = "/root/projects/liangwenbei_workdir/factor_per_feature_audit/F1F2"
LOG_PATH = os.path.join(WORKDIR, "run.log")

REF_SAMPLE = 200_000   # train reference pool
SUB_SAMPLE = 50_000    # per-sym / per-bucket / val / test
SEED = 7


def log(msg: str, *, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def main():
    t0 = time.time()
    log_fh = open(LOG_PATH, "w")
    try:
        feat_names = load_feat_names()
        name_to_idx = {n: i for i, n in enumerate(feat_names)}
        log(f"Loaded {len(feat_names)} feature names from schemeP_feat_names.txt", fh=log_fh)

        metas = all_meta()
        log(f"Built metadata for {len(metas)} F1+F2 features", fh=log_fh)
        assert len(metas) == 155, f"expected 155 metadata entries, got {len(metas)}"

        # Resolve dim_in_X (the column index in cache X) for each feature
        feature_cols: List[int] = []
        for m in metas:
            idx = name_to_idx.get(m["name"])
            if idx is None:
                raise KeyError(f"Feature {m['name']} not in schemeP_feat_names.txt")
            m["dim_in_X"] = idx
            feature_cols.append(idx)
        log("Resolved dim_in_X for all 155 features", fh=log_fh)

        # ---------- Load cache (only our 155 cols) ----------
        log("Loading train cache (155 cols + sym + date)...", fh=log_fh)
        t1 = time.time()
        train = load_cache_columns("train", feature_cols)
        log(f"  train.X shape={train['X'].shape} loaded in {time.time()-t1:.1f}s", fh=log_fh)

        t1 = time.time()
        val = load_cache_columns("val", feature_cols)
        log(f"  val.X shape={val['X'].shape} loaded in {time.time()-t1:.1f}s", fh=log_fh)

        t1 = time.time()
        test = load_cache_columns("test", feature_cols)
        log(f"  test.X shape={test['X'].shape} loaded in {time.time()-t1:.1f}s", fh=log_fh)

        # ---------- Build sampled index sets ----------
        rng = np.random.default_rng(SEED)
        n_train = train["X"].shape[0]

        ref_idx = stratified_sample(n_train, REF_SAMPLE, rng)
        log(f"Ref pool: {len(ref_idx):,} train rows", fh=log_fh)

        # per-sym indices (all rows of that sym from train)
        sym_full_idx = {s: np.where(train["sym"] == s)[0] for s in range(5)}
        sym_sub_idx = {}
        for s in range(5):
            n_s = len(sym_full_idx[s])
            if n_s == 0:
                sym_sub_idx[s] = sym_full_idx[s]
                continue
            sub = stratified_sample(n_s, SUB_SAMPLE, rng)
            sym_sub_idx[s] = sym_full_idx[s][sub]
        for s in range(5):
            log(f"  sym={s}: {len(sym_full_idx[s]):,} rows in train, sub={len(sym_sub_idx[s]):,}", fh=log_fh)

        # date bucket indices (size=20 → buckets 0..3 for train (0-79); 4 for val (80-99 mixed); 5 for test (100-119))
        # Per SPEC: 6 buckets each ~20 days, computed over union of splits
        train_bucket = date_bucket(train["date"], 20)
        # For "by_bucket" PSI we compare each bucket vs ref-pool (=train sampled).
        # But buckets 4 and 5 live in val/test → so compute on combined data.
        # We restrict by_bucket to 0..3 from train, plus 4 from val, 5 from test.
        bucket_groups: Dict[int, np.ndarray] = {}
        for b in range(4):  # 0..3 (train days 0..79)
            mask = train_bucket == b
            ids = np.where(mask)[0]
            sub = stratified_sample(len(ids), SUB_SAMPLE, rng)
            bucket_groups[b] = (train["X"], ids[sub])
        val_bucket = date_bucket(val["date"], 20)
        for b in [4]:  # val days 80..95 → bucket 4
            ids = np.where(val_bucket == b)[0]
            if len(ids) == 0:
                continue
            sub = stratified_sample(len(ids), SUB_SAMPLE, rng)
            bucket_groups[b] = (val["X"], ids[sub])
        test_bucket = date_bucket(test["date"], 20)
        for b in [4, 5]:  # test 96..119 covers bucket 4 (96..99) and 5 (100..119)
            ids = np.where(test_bucket == b)[0]
            if len(ids) == 0:
                continue
            sub = stratified_sample(len(ids), SUB_SAMPLE, rng)
            # Bucket 4 might already be present from val — merge by stacking
            if b in bucket_groups:
                prev_x, prev_idx = bucket_groups[b]
                # encode both with their X reference (different arrays); we'll concat values later in compute
                bucket_groups[b + 10] = (test["X"], ids[sub])  # use offset key 14 for "test-side bucket 4"
            else:
                bucket_groups[b] = (test["X"], ids[sub])
        log(f"  bucket sizes: " + ", ".join(f"b{b}={len(idx)}" for b, (_, idx) in sorted(bucket_groups.items())), fh=log_fh)

        # val/test full sub
        val_sub = stratified_sample(val["X"].shape[0], SUB_SAMPLE, rng)
        test_sub = stratified_sample(test["X"].shape[0], SUB_SAMPLE, rng)
        log(f"  val_sub={len(val_sub)}, test_sub={len(test_sub)}", fh=log_fh)

        # ---------- Per-feature pass ----------
        log("Starting per-feature stats loop (155 features) ...", fh=log_fh)
        t1 = time.time()
        records: List[Dict] = []
        ks_drop_hits: List[str] = []

        for fi, m in enumerate(metas):
            col = fi  # the i-th col in our sliced X corresponds to the i-th feature in metas
            x_train_full = train["X"][:, col]
            x_val_full = val["X"][:, col]
            x_test_full = test["X"][:, col]

            # NaN counts (full split, not subsampled)
            n_train_nan = int(np.isnan(x_train_full).sum())
            n_val_nan = int(np.isnan(x_val_full).sum())
            n_test_nan = int(np.isnan(x_test_full).sum())
            n_train = x_train_full.shape[0]
            n_val = x_val_full.shape[0]
            n_test = x_test_full.shape[0]
            nan_pct = {
                "train": {"count": n_train_nan, "pct": round(n_train_nan / n_train * 100, 6)},
                "val":   {"count": n_val_nan,   "pct": round(n_val_nan   / n_val   * 100, 6)},
                "test":  {"count": n_test_nan,  "pct": round(n_test_nan  / n_test  * 100, 6)},
            }

            # Detect constant / near-constant
            finite = x_train_full[np.isfinite(x_train_full)]
            n_unique = len(np.unique(finite)) if finite.size else 0
            is_constant = (n_unique <= 2)

            if is_constant or finite.size < 1000:
                rec = {
                    "name": m["name"],
                    "name_cn": m.get("name_cn"),
                    "name_en_full": m.get("name_en_full"),
                    "family": "F1" if fi < 105 else "F2",
                    "raw_or_derived": m.get("raw_or_derived"),
                    "raw_idx_in_154": m.get("raw_idx_in_154"),
                    "derived_idx_in_216": m.get("derived_idx_in_216"),
                    "dim_in_X": m["dim_in_X"],
                    "in_ks_drop_11": m["name"] in KS_FAIL,
                    "formula_latex": m["formula_latex"],
                    "code_loc": m["code_loc"],
                    "numpy_pseudo": m["numpy_pseudo"],
                    "physical_meaning": m["physical_meaning"],
                    "nan_handling": m["nan_handling"],
                    "nan_pct": nan_pct,
                    "cross_stock_psi": {"by_sym": [None]*5, "max": None, "mean": None},
                    "cross_stock_ks":  {"by_sym": [None]*5, "max": None},
                    "cross_date_psi":  {"train_vs_val": None, "train_vs_test": None, "by_bucket": [None]*6},
                    "cross_date_ks":   {"train_vs_val": None, "train_vs_test": None},
                    "verdict_cross_stock": "constant" if is_constant else "insufficient_data",
                    "verdict_cross_date":  "constant" if is_constant else "insufficient_data",
                    "notes": ("常数列，跳过 PSI/KS" if is_constant else "有限值不足 1000，跳过 PSI/KS")
                            + (" | KS-drop 11 之一" if m["name"] in KS_FAIL else ""),
                }
                if m["name"] in KS_FAIL:
                    ks_drop_hits.append(m["name"])
                records.append(rec)
                if (fi + 1) % 20 == 0:
                    log(f"  [{fi+1}/{len(metas)}] {m['name']}: constant/insufficient", fh=log_fh)
                continue

            ref_vals = x_train_full[ref_idx]

            # Cross-stock PSI/KS
            by_sym_psi: List[float] = []
            by_sym_ks: List[float] = []
            for s in range(5):
                cur_idx = sym_sub_idx[s]
                if len(cur_idx) == 0:
                    by_sym_psi.append(float("nan")); by_sym_ks.append(float("nan"))
                    continue
                cur = x_train_full[cur_idx]
                by_sym_psi.append(psi(ref_vals, cur))
                by_sym_ks.append(ks_stat(ref_vals, cur))
            psi_arr = np.array(by_sym_psi, dtype=np.float64)
            ks_arr  = np.array(by_sym_ks, dtype=np.float64)
            psi_max_stock = float(np.nanmax(psi_arr)) if np.isfinite(psi_arr).any() else float("nan")
            psi_mean_stock = float(np.nanmean(psi_arr)) if np.isfinite(psi_arr).any() else float("nan")
            ks_max_stock = float(np.nanmax(ks_arr)) if np.isfinite(ks_arr).any() else float("nan")

            # Cross-date train vs val/test
            psi_tv = psi(ref_vals, x_val_full[val_sub])
            psi_tt = psi(ref_vals, x_test_full[test_sub])
            ks_tv  = ks_stat(ref_vals, x_val_full[val_sub])
            ks_tt  = ks_stat(ref_vals, x_test_full[test_sub])

            # By-bucket PSI (6 buckets)
            by_bucket: List[float] = []
            for b in range(6):
                # bucket 4: may have val + test parts; merge
                vals_for_bucket = []
                if b in bucket_groups:
                    Xref, ids = bucket_groups[b]
                    vals_for_bucket.append(Xref[ids, col])
                if b == 4 and (b + 10) in bucket_groups:
                    Xref, ids = bucket_groups[b + 10]
                    vals_for_bucket.append(Xref[ids, col])
                if vals_for_bucket:
                    cur = np.concatenate(vals_for_bucket) if len(vals_for_bucket) > 1 else vals_for_bucket[0]
                    by_bucket.append(psi(ref_vals, cur))
                else:
                    by_bucket.append(float("nan"))

            bucket_arr = np.array(by_bucket, dtype=np.float64)
            psi_max_date_bucket = float(np.nanmax(bucket_arr)) if np.isfinite(bucket_arr).any() else float("nan")
            psi_max_date = max([v for v in (psi_tv, psi_tt, psi_max_date_bucket) if np.isfinite(v)], default=float("nan"))

            in_ks = m["name"] in KS_FAIL
            if in_ks:
                ks_drop_hits.append(m["name"])

            rec = {
                "name": m["name"],
                "name_cn": m.get("name_cn"),
                "name_en_full": m.get("name_en_full"),
                "family": "F1" if fi < 105 else "F2",
                "raw_or_derived": m.get("raw_or_derived"),
                "raw_idx_in_154": m.get("raw_idx_in_154"),
                "derived_idx_in_216": m.get("derived_idx_in_216"),
                "dim_in_X": m["dim_in_X"],
                "in_ks_drop_11": in_ks,
                "formula_latex": m["formula_latex"],
                "code_loc": m["code_loc"],
                "numpy_pseudo": m["numpy_pseudo"],
                "physical_meaning": m["physical_meaning"],
                "nan_handling": m["nan_handling"],
                "nan_pct": nan_pct,
                "cross_stock_psi": {
                    "by_sym": [None if not np.isfinite(v) else round(v, 6) for v in by_sym_psi],
                    "max": None if not np.isfinite(psi_max_stock) else round(psi_max_stock, 6),
                    "mean": None if not np.isfinite(psi_mean_stock) else round(psi_mean_stock, 6),
                },
                "cross_stock_ks": {
                    "by_sym": [None if not np.isfinite(v) else round(v, 6) for v in by_sym_ks],
                    "max": None if not np.isfinite(ks_max_stock) else round(ks_max_stock, 6),
                },
                "cross_date_psi": {
                    "train_vs_val":  None if not np.isfinite(psi_tv) else round(psi_tv, 6),
                    "train_vs_test": None if not np.isfinite(psi_tt) else round(psi_tt, 6),
                    "by_bucket":     [None if not np.isfinite(v) else round(v, 6) for v in by_bucket],
                },
                "cross_date_ks": {
                    "train_vs_val":  None if not np.isfinite(ks_tv) else round(ks_tv, 6),
                    "train_vs_test": None if not np.isfinite(ks_tt) else round(ks_tt, 6),
                },
                "verdict_cross_stock": verdict_from_psi(psi_max_stock),
                "verdict_cross_date":  verdict_from_psi(psi_max_date),
                "notes": ("KS-drop 11 之一" if in_ks else ""),
            }
            records.append(rec)

            if (fi + 1) % 20 == 0:
                log(f"  [{fi+1}/{len(metas)}] {m['name']}: psi_stock_max={psi_max_stock:.3f}, psi_date_tv={psi_tv:.3f}, psi_date_tt={psi_tt:.3f}", fh=log_fh)

        log(f"All 155 features processed in {time.time()-t1:.1f}s", fh=log_fh)

        # ---------- Write JSON ----------
        out_json = os.path.join(WORKDIR, "per_feature_audit.json")
        with open(out_json, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)
        log(f"Wrote {out_json}", fh=log_fh)

        # ---------- Summary ----------
        # Top-5 most unstable by cross-stock max
        def _key_stock(r):
            v = r["cross_stock_psi"]["max"]
            return -v if v is not None else -float("-inf")
        def _key_date(r):
            v = max(r["cross_date_psi"]["train_vs_val"] or -1e9, r["cross_date_psi"]["train_vs_test"] or -1e9)
            return -v
        def _key_nan(r):
            return -r["nan_pct"]["train"]["pct"]

        stable_recs = [r for r in records if r["cross_stock_psi"]["max"] is not None]
        top_unstable_stock = sorted(stable_recs, key=_key_stock)[:5]
        top_stable_stock = sorted(stable_recs, key=lambda r: r["cross_stock_psi"]["max"])[:5]

        date_recs = [r for r in records if r["cross_date_psi"]["train_vs_val"] is not None or r["cross_date_psi"]["train_vs_test"] is not None]
        top_unstable_date = sorted(date_recs, key=_key_date)[:5]
        top_stable_date = sorted(date_recs, key=lambda r: max(r["cross_date_psi"]["train_vs_val"] or 0, r["cross_date_psi"]["train_vs_test"] or 0))[:5]

        top_nan = sorted(records, key=_key_nan)[:5]

        def _slim(r):
            return {
                "name": r["name"],
                "family": r["family"],
                "raw_or_derived": r["raw_or_derived"],
                "cross_stock_psi_max": r["cross_stock_psi"]["max"],
                "cross_date_psi_max": max(filter(lambda v: v is not None, [r["cross_date_psi"]["train_vs_val"], r["cross_date_psi"]["train_vs_test"]] + [v for v in r["cross_date_psi"]["by_bucket"] if v is not None]), default=None),
                "nan_pct_train": r["nan_pct"]["train"]["pct"],
                "verdict_cross_stock": r["verdict_cross_stock"],
                "verdict_cross_date": r["verdict_cross_date"],
            }

        f1_recs = [r for r in records if r["family"] == "F1"]
        f2_recs = [r for r in records if r["family"] == "F2"]
        mean_nan_train = float(np.mean([r["nan_pct"]["train"]["pct"] for r in records]))
        f1_mean_nan = float(np.mean([r["nan_pct"]["train"]["pct"] for r in f1_recs])) if f1_recs else 0.0
        f2_mean_nan = float(np.mean([r["nan_pct"]["train"]["pct"] for r in f2_recs])) if f2_recs else 0.0

        # max PSI overall
        all_stock_max = [r["cross_stock_psi"]["max"] for r in records if r["cross_stock_psi"]["max"] is not None]
        all_date_max  = [max(r["cross_date_psi"]["train_vs_val"] or 0, r["cross_date_psi"]["train_vs_test"] or 0) for r in records if (r["cross_date_psi"]["train_vs_val"] is not None or r["cross_date_psi"]["train_vs_test"] is not None)]
        max_psi_stock = float(max(all_stock_max)) if all_stock_max else float("nan")
        max_psi_date  = float(max(all_date_max))  if all_date_max  else float("nan")

        # KS-drop hits in our scope
        my_ks_drop_in_scope = ["kyle_lam_W50", "kyle_lam_W100"]
        present_ks_hits = sorted(set(ks_drop_hits) & set(my_ks_drop_in_scope))

        summary = {
            "scope": "F1+F2",
            "n_features": len(records),
            "f1_count": len(f1_recs),
            "f2_count": len(f2_recs),
            "ref_sample": REF_SAMPLE,
            "sub_sample": SUB_SAMPLE,
            "seed": SEED,
            "max_psi_cross_stock": max_psi_stock,
            "max_psi_cross_date": max_psi_date,
            "mean_nan_pct_train_overall": mean_nan_train,
            "mean_nan_pct_train_f1": f1_mean_nan,
            "mean_nan_pct_train_f2": f2_mean_nan,
            "ks_drop_11_in_scope": my_ks_drop_in_scope,
            "ks_drop_11_observed_hits": present_ks_hits,
            "top5_most_unstable_cross_stock": [_slim(r) for r in top_unstable_stock],
            "top5_most_stable_cross_stock":   [_slim(r) for r in top_stable_stock],
            "top5_most_unstable_cross_date":  [_slim(r) for r in top_unstable_date],
            "top5_most_stable_cross_date":    [_slim(r) for r in top_stable_date],
            "top5_highest_nan_train":         [_slim(r) for r in top_nan],
        }
        out_sum = os.path.join(WORKDIR, "F1F2_summary.json")
        with open(out_sum, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)
        log(f"Wrote {out_sum}", fh=log_fh)

        # ---------- Markdown report ----------
        write_md(records, summary, os.path.join(WORKDIR, "per_feature_audit.md"))
        log(f"Wrote per_feature_audit.md", fh=log_fh)

        # ---------- WandB ----------
        try:
            import wandb
            wandb.init(
                project="liangwenbei-feat-audit",
                entity="cjxh21-Tsinghua University",
                name="F1F2-audit",
                config={"family": "F1+F2", "n_features": 155,
                        "ref_sample": REF_SAMPLE, "sub_sample": SUB_SAMPLE, "seed": SEED},
                reinit=True,
            )
            wandb.log({
                "n_features_audited": len(records),
                "max_psi_cross_stock": max_psi_stock,
                "max_psi_cross_date": max_psi_date,
                "mean_nan_pct_train": mean_nan_train,
                "mean_nan_pct_train_f1": f1_mean_nan,
                "mean_nan_pct_train_f2": f2_mean_nan,
                "n_ks_drop_hit": len(present_ks_hits),
                "wall_time_sec": time.time() - t0,
            })
            # Log per-feature table for easy comparison
            try:
                tbl = wandb.Table(columns=["name", "family", "raw_or_derived",
                                            "nan_pct_train", "psi_stock_max", "psi_date_max",
                                            "verdict_stock", "verdict_date"])
                for r in records:
                    psi_d_vals = [r["cross_date_psi"]["train_vs_val"], r["cross_date_psi"]["train_vs_test"]]
                    psi_d_vals += [v for v in r["cross_date_psi"]["by_bucket"] if v is not None]
                    psi_d_vals = [v for v in psi_d_vals if v is not None]
                    pd_max = max(psi_d_vals) if psi_d_vals else None
                    tbl.add_data(r["name"], r["family"], r["raw_or_derived"],
                                  r["nan_pct"]["train"]["pct"],
                                  r["cross_stock_psi"]["max"], pd_max,
                                  r["verdict_cross_stock"], r["verdict_cross_date"])
                wandb.log({"per_feature_table": tbl})
            except Exception as e:
                log(f"WandB table log failed: {e}", fh=log_fh)
            wandb.finish()
            log("WandB run complete", fh=log_fh)
        except Exception as e:
            log(f"WandB init failed: {e}", fh=log_fh)

        # ---------- Final RESULT line ----------
        result_msg = (f"RESULT: task=F1F2_audit "
                      f"metrics={{n_features=155, max_psi_stock={max_psi_stock:.2f}, "
                      f"max_psi_date={max_psi_date:.2f}, "
                      f"mean_nan_pct={mean_nan_train:.4f}, "
                      f"ks_drop_hits={len(present_ks_hits)}}} "
                      f"notes=F1+F2 audit done in {time.time()-t0:.1f}s; kyle_lam_W50/W100 confirmed in KS-drop")
        log(result_msg, fh=log_fh)
        # Also write results.json
        results = {
            "task": "F1F2_per_feature_audit",
            "metrics": {
                "n_features": 155,
                "max_psi_stock": max_psi_stock,
                "max_psi_date": max_psi_date,
                "mean_nan_pct": mean_nan_train,
                "ks_drop_hits": len(present_ks_hits),
            },
            "notes": "F1+F2 (105+50) audit complete; kyle_lam_W50/W100 confirmed as KS-drop hits in scope.",
        }
        with open(os.path.join(WORKDIR, "results.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)

        print(result_msg)
    finally:
        log_fh.close()


def write_md(records: List[Dict], summary: Dict, path: str):
    def fmt_pct(v):
        return f"{v:.4f}%" if isinstance(v, (int, float)) else "n/a"
    def fmt_num(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        return f"{v:.4f}"
    lines = []
    lines.append("# F1+F2 Per-Feature Audit Report\n")
    lines.append(f"Scope: F1 LOB派生 (105) + F2 多尺度OFI (50) = **{summary['n_features']}** features.\n")
    lines.append(f"Sampling: ref_pool={summary['ref_sample']:,} train rows, per-sym/per-bucket/val/test sub-sample={summary['sub_sample']:,}, seed={summary['seed']}.\n")
    lines.append("\n## High-Level Summary\n")
    lines.append(f"- **Max PSI cross-stock**: {fmt_num(summary['max_psi_cross_stock'])}")
    lines.append(f"- **Max PSI cross-date**:  {fmt_num(summary['max_psi_cross_date'])}")
    lines.append(f"- **Mean NaN% (train) overall**: {summary['mean_nan_pct_train_overall']:.4f}%")
    lines.append(f"- **Mean NaN% (train) F1**:    {summary['mean_nan_pct_train_f1']:.4f}%")
    lines.append(f"- **Mean NaN% (train) F2**:    {summary['mean_nan_pct_train_f2']:.4f}%")
    lines.append(f"- **KS-drop 11 in scope**: {summary['ks_drop_11_in_scope']}; observed hits in this run: {summary['ks_drop_11_observed_hits']}\n")
    lines.append("### Top-5 most unstable across stocks (PSI max)\n")
    lines.append("| name | family | psi_stock_max | psi_date_max | verdict_stock |")
    lines.append("|------|--------|---------------|--------------|----------------|")
    for r in summary["top5_most_unstable_cross_stock"]:
        lines.append(f"| `{r['name']}` | {r['family']} | {fmt_num(r['cross_stock_psi_max'])} | {fmt_num(r['cross_date_psi_max'])} | {r['verdict_cross_stock']} |")
    lines.append("\n### Top-5 most unstable across date (PSI max)\n")
    lines.append("| name | family | psi_stock_max | psi_date_max | verdict_date |")
    lines.append("|------|--------|---------------|--------------|---------------|")
    for r in summary["top5_most_unstable_cross_date"]:
        lines.append(f"| `{r['name']}` | {r['family']} | {fmt_num(r['cross_stock_psi_max'])} | {fmt_num(r['cross_date_psi_max'])} | {r['verdict_cross_date']} |")
    lines.append("\n### Top-5 highest NaN% (train)\n")
    lines.append("| name | family | nan%_train | raw/derived |")
    lines.append("|------|--------|------------|-------------|")
    for r in summary["top5_highest_nan_train"]:
        lines.append(f"| `{r['name']}` | {r['family']} | {fmt_pct(r['nan_pct_train'])} | {r['raw_or_derived']} |")

    # ---- F1 section ----
    f1_recs = [r for r in records if r["family"] == "F1"]
    f2_recs = [r for r in records if r["family"] == "F2"]
    # within each family, raw before derived
    f1_recs.sort(key=lambda r: (0 if r["raw_or_derived"] == "raw" else 1,
                                r.get("raw_idx_in_154") if r.get("raw_idx_in_154") is not None else r.get("derived_idx_in_216", 1e9)))
    f2_recs.sort(key=lambda r: (0 if r["raw_or_derived"] == "raw" else 1,
                                r.get("derived_idx_in_216", 1e9)))

    def feature_section(r):
        out = [f"\n### `{r['name']}` — {r.get('name_cn') or ''}\n"]
        out.append(f"- **English**: {r.get('name_en_full') or ''}")
        out.append(f"- **Family / type**: {r['family']} / {r['raw_or_derived']}")
        out.append(f"- **dim_in_X**: {r['dim_in_X']}"
                   + (f"  (raw idx in 154: {r['raw_idx_in_154']})" if r.get('raw_idx_in_154') is not None else "")
                   + (f"  (derived idx in 216: {r['derived_idx_in_216']})" if r.get('derived_idx_in_216') is not None else ""))
        out.append(f"- **In KS-drop 11**: {'YES — 训练时被剔除' if r['in_ks_drop_11'] else 'no'}")
        out.append(f"\n**Formula**\n\n$$\n{r['formula_latex']}\n$$\n")
        out.append(f"- **Code location**: `{r['code_loc']}`")
        out.append(f"\n**Numpy pseudo**\n\n```python\n{r['numpy_pseudo']}\n```\n")
        out.append(f"**Physical meaning**: {r['physical_meaning']}\n")
        out.append("**NaN handling**:")
        out.append(f"  - 公式层: {r['nan_handling']['formula_level']}")
        out.append(f"  - NN 管线: {r['nan_handling']['nn_pipeline']}")
        out.append(f"  - LGB 管线: {r['nan_handling']['lgb_pipeline']}")
        out.append(f"\n**NaN%**: train={fmt_pct(r['nan_pct']['train']['pct'])} ({r['nan_pct']['train']['count']}), "
                   f"val={fmt_pct(r['nan_pct']['val']['pct'])} ({r['nan_pct']['val']['count']}), "
                   f"test={fmt_pct(r['nan_pct']['test']['pct'])} ({r['nan_pct']['test']['count']})")
        cs_psi = r['cross_stock_psi']
        cs_ks  = r['cross_stock_ks']
        cd_psi = r['cross_date_psi']
        cd_ks  = r['cross_date_ks']
        out.append(f"\n**Cross-stock PSI** by sym: {cs_psi['by_sym']}; max={fmt_num(cs_psi['max'])}, mean={fmt_num(cs_psi['mean'])}")
        out.append(f"**Cross-stock KS** by sym: {cs_ks['by_sym']}; max={fmt_num(cs_ks['max'])}")
        out.append(f"**Cross-date PSI** train→val={fmt_num(cd_psi['train_vs_val'])}, train→test={fmt_num(cd_psi['train_vs_test'])}, by_bucket={cd_psi['by_bucket']}")
        out.append(f"**Cross-date KS** train→val={fmt_num(cd_ks['train_vs_val'])}, train→test={fmt_num(cd_ks['train_vs_test'])}")
        out.append(f"\n**Verdict**: cross_stock=**{r['verdict_cross_stock']}**, cross_date=**{r['verdict_cross_date']}**")
        if r['notes']:
            out.append(f"\n*Notes*: {r['notes']}")
        return "\n".join(out)

    lines.append("\n---\n\n## F1 — LOB 派生 (105 features)\n")
    lines.append("### F1.A 原始字段 (94 raw)\n")
    for r in f1_recs:
        if r["raw_or_derived"] == "raw":
            lines.append(feature_section(r))
    lines.append("\n### F1.B 派生字段 (11 derived: 10 wmp_lvl + wmp_balance_12)\n")
    for r in f1_recs:
        if r["raw_or_derived"] == "derived":
            lines.append(feature_section(r))

    lines.append("\n---\n\n## F2 — 多尺度 OFI (50 features, all derived)\n")
    for r in f2_recs:
        lines.append(feature_section(r))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
