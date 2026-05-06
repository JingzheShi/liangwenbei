"""Build iter_005 submission package: h_60 5-seed ensemble (T25) + h_40 from iter_002.

Strategy:
  h_5/10/20: active=False (platform showed -7.68 / -8.64 / -5.09 — over-fit, no edge)
  h_40: reuse iter_002 single-seed model + iter_002 thresholds (platform +2.02)
        OR if h_40 5-seed LOSO > h_40 single LOSO (+11.71), use 5-seed ensemble
  h_60: 5-seed ensemble + new tuned threshold

Reads:
    experiments/T25_h60_specialized/threshold_results.json   per-horizon best (T,delta)
    experiments/T25_h60_specialized/final_223_model_h{H}_seed{S}.txt   for each (S, H)
    submission/iter_002_lgbm_schemeC/model_h40.txt           (fallback h_40)

Writes:
    submission/iter_005_h60_5seed/
        Predictor.py
        compute.py
        config.json
        requirements.txt
        thresholds.json
        model_h60_seed{S}.txt   for each seed
        model_h40.txt           (if h_40 single) OR model_h40_seed{S}.txt (if h_40 ensemble)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SUBMIT_DIR = os.path.join(ROOT, "submission", "iter_005_h60_5seed")
ITER002_DIR = os.path.join(ROOT, "submission", "iter_002_lgbm_schemeC")


def _load_iter002_h40_threshold():
    p = os.path.join(ITER002_DIR, "thresholds.json")
    with open(p) as f:
        cfg = json.load(f)
    for h in cfg["horizons"]:
        if int(h["h"]) == 40:
            return float(h["T"]), float(h["delta"]), float(h.get("loso_best_sum", 0.0))
    sys.exit("h_40 not found in iter_002 thresholds.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,1,7,13,100")
    ap.add_argument("--h40-mode", choices=["single", "ensemble", "off"], default="single",
                    help="single: use iter_002 model_h40.txt; ensemble: use T25 5-seed h_40 final models; off: disable h_40")
    ap.add_argument("--h60-min-sum", type=float, default=7.0,
                    help="min LOSO ensemble best sum_cum_pnl to keep h_60 active")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"Building iter_005 — seeds={seeds} h40_mode={args.h40_mode}", flush=True)

    sweep_path = os.path.join(HERE, "threshold_results.json")
    if not os.path.isfile(sweep_path):
        sys.exit(f"missing {sweep_path}")
    with open(sweep_path) as f:
        sweep = json.load(f)

    if os.path.isdir(SUBMIT_DIR):
        # Clean any stale model files from previous build
        for fn in os.listdir(SUBMIT_DIR):
            if fn.startswith("model_") and fn.endswith(".txt"):
                os.remove(os.path.join(SUBMIT_DIR, fn))
                print(f"  removed stale {fn}", flush=True)
    os.makedirs(SUBMIT_DIR, exist_ok=True)

    for fn in ("compute.py", "config.json", "requirements.txt"):
        src = os.path.join(ITER002_DIR, fn)
        dst = os.path.join(SUBMIT_DIR, fn)
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}", flush=True)

    predictor_src = os.path.join(HERE, "Predictor.py")
    predictor_dst = os.path.join(SUBMIT_DIR, "Predictor.py")
    if not os.path.isfile(predictor_src):
        sys.exit(f"missing {predictor_src}")
    shutil.copy(predictor_src, predictor_dst)
    print(f"copied {predictor_src} -> {predictor_dst}", flush=True)

    horizons_in = []
    valid_hs = set(int(h.lstrip("h")) for h in sweep["per_horizon"].keys())

    # Short horizons: active=False (no h_5/10/20 model files in package).
    for H in (5, 10, 20):
        horizons_in.append({"h": H, "T": 0.5, "delta": 0.0, "active": False,
                            "note": "deactivated — platform over-fit",
                            "model_files": []})

    # h_40
    if args.h40_mode == "single":
        T40, d40, h40_loso = _load_iter002_h40_threshold()
        src = os.path.join(ITER002_DIR, "model_h40.txt")
        dst = os.path.join(SUBMIT_DIR, "model_h40.txt")
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}", flush=True)
        horizons_in.append({
            "h": 40, "T": T40, "delta": d40, "active": True,
            "model_files": ["model_h40.txt"],
            "source": "iter_002 single seed",
            "iter002_loso_best_sum": h40_loso,
        })
    elif args.h40_mode == "ensemble":
        if 40 not in valid_hs:
            sys.exit("h40_mode=ensemble but threshold_results has no h40")
        ens40 = sweep["per_horizon"]["h40"]["ensemble"]
        T40 = float(ens40["best"]["T"])
        d40 = float(ens40["best"]["delta"])
        loso_sum = float(ens40["best"]["sum_cum_pnl"])
        files = []
        for s in seeds:
            src = os.path.join(HERE, f"final_223_model_h40_seed{s}.txt")
            dst = os.path.join(SUBMIT_DIR, f"model_h40_seed{s}.txt")
            if not os.path.isfile(src):
                sys.exit(f"missing final h40 model: {src}")
            shutil.copy(src, dst)
            files.append(f"model_h40_seed{s}.txt")
        horizons_in.append({
            "h": 40, "T": T40, "delta": d40, "active": True,
            "model_files": files,
            "source": "T25 5-seed ensemble",
            "loso_ensemble_best_sum": loso_sum,
            "loso_ensemble_n_pos_folds": int(ens40["best"]["n_pos_folds"]),
            "loso_ensemble_argmax_sum": float(ens40["baseline"]["raw_argmax_sum_cum_pnl"]),
        })
    else:
        horizons_in.append({"h": 40, "T": 0.5, "delta": 0.0, "active": False,
                            "model_files": []})

    # h_60
    if 60 not in valid_hs:
        sys.exit("h60 missing from threshold_results.json")
    ens60 = sweep["per_horizon"]["h60"]["ensemble"]
    T60 = float(ens60["best"]["T"])
    d60 = float(ens60["best"]["delta"])
    loso_sum60 = float(ens60["best"]["sum_cum_pnl"])
    h60_active = loso_sum60 >= args.h60_min_sum
    files60 = []
    if h60_active:
        for s in seeds:
            src = os.path.join(HERE, f"final_223_model_h60_seed{s}.txt")
            dst = os.path.join(SUBMIT_DIR, f"model_h60_seed{s}.txt")
            if not os.path.isfile(src):
                sys.exit(f"missing final h60 model: {src}")
            shutil.copy(src, dst)
            files60.append(f"model_h60_seed{s}.txt")
    horizons_in.append({
        "h": 60, "T": T60, "delta": d60, "active": bool(h60_active),
        "model_files": files60,
        "source": "T25 5-seed ensemble" if h60_active else "deactivated (LOSO too low)",
        "loso_ensemble_best_sum": loso_sum60,
        "loso_ensemble_n_pos_folds": int(ens60["best"]["n_pos_folds"]),
        "loso_ensemble_argmax_sum": float(ens60["baseline"]["raw_argmax_sum_cum_pnl"]),
    })

    horizons_in.sort(key=lambda h: h["h"])

    out_th = os.path.join(SUBMIT_DIR, "thresholds.json")
    with open(out_th, "w") as f:
        json.dump({
            "_doc": "iter_005 h_60 specialized 5-seed ensemble + h_40 from iter_002. Generated by build_iter_005.py.",
            "seeds": seeds,
            "horizons": horizons_in,
            "h40_mode": args.h40_mode,
            "h60_min_sum_threshold": args.h60_min_sum,
        }, f, indent=2)
    print(f"thresholds -> {out_th}", flush=True)

    print("\n=== Summary ===")
    for h in horizons_in:
        flag = "ACTIVE" if h["active"] else "flat"
        loso = h.get("loso_ensemble_best_sum", h.get("iter002_loso_best_sum", "n/a"))
        print(f"  h={h['h']:2d}: {flag:6s} (T={h['T']:.2f} d={h['delta']:.2f})  "
              f"LOSO={loso}  files={h['model_files']}")


if __name__ == "__main__":
    main()
