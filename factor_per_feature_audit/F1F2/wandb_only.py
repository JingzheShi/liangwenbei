"""Standalone WandB upload from existing per_feature_audit.json + F1F2_summary.json.

Tries entity='cjxh21-Tsinghua University' first; on permission denial falls back to the
API key's default entity to ensure metrics get logged.
"""
import json
import os
import sys
import time
import numpy as np

WORKDIR = "/root/projects/liangwenbei_workdir/factor_per_feature_audit/F1F2"

with open(os.path.join(WORKDIR, "per_feature_audit.json")) as f:
    records = json.load(f)
with open(os.path.join(WORKDIR, "F1F2_summary.json")) as f:
    summary = json.load(f)


def _build_table(wandb):
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
    return tbl


def run_wandb(entity):
    import wandb
    label = entity or "(default)"
    print(f"[wandb] init project=liangwenbei-feat-audit entity={label}", flush=True)
    kwargs = dict(project="liangwenbei-feat-audit", name="F1F2-audit",
                  config={"family": "F1+F2", "n_features": 155,
                          "ref_sample": summary["ref_sample"],
                          "sub_sample": summary["sub_sample"],
                          "seed": summary["seed"]})
    if entity:
        kwargs["entity"] = entity
    run = wandb.init(**kwargs)
    wandb.log({
        "n_features_audited": summary["n_features"],
        "max_psi_cross_stock": summary["max_psi_cross_stock"],
        "max_psi_cross_date": summary["max_psi_cross_date"],
        "mean_nan_pct_train": summary["mean_nan_pct_train_overall"],
        "mean_nan_pct_train_f1": summary["mean_nan_pct_train_f1"],
        "mean_nan_pct_train_f2": summary["mean_nan_pct_train_f2"],
        "n_ks_drop_hit": len(summary["ks_drop_11_observed_hits"]),
    })
    wandb.log({"per_feature_table": _build_table(wandb)})
    url = run.url
    wandb.finish()
    return url


try:
    url = run_wandb("cjxh21-Tsinghua University")
    print(f"[wandb] team entity OK, run URL: {url}")
except Exception as e:
    print(f"[wandb] team entity failed: {e}; falling back to default entity")
    try:
        url = run_wandb(None)
        print(f"[wandb] fallback OK, run URL: {url}")
    except Exception as e2:
        print(f"[wandb] fallback also failed: {e2}")
        sys.exit(1)
