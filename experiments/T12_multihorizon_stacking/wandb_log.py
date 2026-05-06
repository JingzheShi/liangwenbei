"""Log T12 results to WandB."""
import json
import os
import wandb

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "results.json")) as f:
    res = json.load(f)

run = wandb.init(
    project="liangwenbei",
    name="T12-multihorizon-stacking",
    config={
        "experiment": "T12",
        "method": "multi-horizon stacking (Ridge/LR/LightGBM meta)",
        "base": "iter_002 Scheme C 223-d LightGBM, 5 horizons",
        "target": "label_10",
        "fold_strategy": "LOSO 5-fold (sym held-out)",
        "feat_dim": 15,
        "outcome": "negative",
    },
    notes=("Stacking does NOT improve over iter_002 h_10 base "
           "(+21.86 → +17.97 LR / +17.46 LGBM). Meta is too conservative."),
)

m = res["metrics"]
wandb.log({
    "iter002_h10_loso_sum": m["iter002_h10_loso_sum"],
    "meta_lr_h10_loso_sum_best": m["meta_lr_h10_loso_sum_best"],
    "meta_lgbm_h10_loso_sum_best": m["meta_lgbm_h10_loso_sum_best"],
    "delta_vs_iter002_lr": m["delta_vs_iter002_lr"],
    "delta_vs_iter002_lgbm": m["delta_vs_iter002_lgbm"],
})

# Per-fold table
import pandas as pd
table = wandb.Table(
    columns=["fold", "iter002", "meta_lr_best", "meta_lgbm_best"],
    data=[
        [k, res["per_fold_pnl"]["iter002"][k],
         res["per_fold_pnl"]["meta_lr_best"][k],
         res["per_fold_pnl"]["meta_lgbm_best"][k]]
        for k in range(5)
    ],
)
wandb.log({"per_fold_pnl_table": table})

# Diagnosis
wandb.log({
    "diag/base_take_rate_T0.55_d0.10": res["diagnosis"][
        "base_h10_take_rate_at_T0.55_d0.10"],
    "diag/meta_lr_take_rate_T0.55_d0.10": res["diagnosis"][
        "meta_lr_take_rate_at_T0.55_d0.10"],
    "diag/meta_lgbm_take_rate_T0.55_d0.10": res["diagnosis"][
        "meta_lgbm_take_rate_at_T0.55_d0.10"],
    "diag/base_mean_p1": res["diagnosis"]["base_h10_mean_p1"],
    "diag/meta_lr_mean_p1": res["diagnosis"]["meta_lr_mean_p1"],
    "diag/meta_lgbm_mean_p1": res["diagnosis"]["meta_lgbm_mean_p1"],
})

# Upload report.md as artifact
art = wandb.Artifact("T12_report", type="report")
art.add_file(os.path.join(HERE, "report.md"))
art.add_file(os.path.join(HERE, "results.json"))
art.add_file(os.path.join(HERE, "fold_summaries.json"))
art.add_file(os.path.join(HERE, "fold_summaries_lgbm.json"))
art.add_file(os.path.join(HERE, "threshold_results.json"))
art.add_file(os.path.join(HERE, "threshold_results_lgbm.json"))
run.log_artifact(art)

run.finish()
print("WandB run finished")
