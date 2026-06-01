"""Ensemble NN+SPO + LGB (1+1), weights 1.0 + 1.5, symmetric or asymmetric EV gate.

Loads preds.npz from two ablation_runs/big_table/<row>/ directories and computes
the weighted-mean predictor. Searches threshold on val, evaluates on test.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np


def compute_pnl(y_pred, mp_t, mp_th, thr_up, thr_dn, fee=1e-4):
    a = np.zeros_like(y_pred)
    a[y_pred > thr_up] = 1.0
    a[y_pred < -thr_dn] = -1.0
    raw = a * (mp_th - mp_t)
    cost = fee * np.abs(a) * ((mp_th + 1) + (mp_t + 1))
    pnl = (raw - cost) / (mp_t + 1.0)
    return pnl.sum(), int((a != 0).sum())


def search_threshold_sym(y_pred, mp_t, mp_th, fee=1e-4):
    mean_abs = max(float(np.mean(np.abs(y_pred))), 1e-8)
    best = (-1e18, None, None)
    for k in np.linspace(0.5, 3.0, 26):
        thr = k * mean_abs
        s, _ = compute_pnl(y_pred, mp_t, mp_th, thr, thr, fee)
        if s > best[0]:
            best = (s, thr, k)
    return best


def search_threshold_asym_de(y_pred, mp_t, mp_th, fee=1e-4):
    """DE for 2D asym EV gate, bounded around the symmetric optimum to avoid val overfit."""
    from scipy.optimize import differential_evolution
    # First find the symmetric optimum.
    _, sym_thr, _ = search_threshold_sym(y_pred, mp_t, mp_th, fee)
    lo = max(0.7 * sym_thr, 1e-8)
    hi = 1.5 * sym_thr

    def neg(x):
        s, _ = compute_pnl(y_pred, mp_t, mp_th, x[0], x[1], fee)
        return -s

    res = differential_evolution(neg, bounds=[(lo, hi), (lo, hi)],
                                  maxiter=80, popsize=25, tol=1e-9,
                                  seed=1, polish=True, mutation=(0.5, 1.0))
    s, _ = compute_pnl(y_pred, mp_t, mp_th, res.x[0], res.x[1], fee)
    return float(s), float(res.x[0]), float(res.x[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-dir", required=True)
    ap.add_argument("--lgb-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-nn", type=float, default=1.0)
    ap.add_argument("--w-lgb", type=float, default=1.5)
    ap.add_argument("--gate", choices=["sym", "asym"], default="sym")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-project", default="liangwenbei-ablation-rerun")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nn = np.load(os.path.join(args.nn_dir, "preds.npz"))
    lgb = np.load(os.path.join(args.lgb_dir, "preds.npz"))

    assert nn["yp_val"].shape == lgb["yp_val"].shape, "val preds mismatch"
    assert nn["yp_test"].shape == lgb["yp_test"].shape, "test preds mismatch"
    assert np.allclose(nn["mp_t_val"], lgb["mp_t_val"]), "val mp_t mismatch"
    assert np.allclose(nn["mp_t_test"], lgb["mp_t_test"]), "test mp_t mismatch"

    W = args.w_nn + args.w_lgb
    yv = (args.w_nn * nn["yp_val"] + args.w_lgb * lgb["yp_val"]) / W
    yt = (args.w_nn * nn["yp_test"] + args.w_lgb * lgb["yp_test"]) / W
    mp_t_v = nn["mp_t_val"]; mp_th_v = nn["mp_th_val"]
    mp_t_t = nn["mp_t_test"]; mp_th_t = nn["mp_th_test"]

    if args.gate == "sym":
        val_pnl, thr, k = search_threshold_sym(yv, mp_t_v, mp_th_v)
        test_pnl, n_act = compute_pnl(yt, mp_t_t, mp_th_t, thr, thr)
        decision = {"gate": "sym", "thr_sym": float(thr), "k_sym": float(k),
                    "val_pnl_h60": float(val_pnl), "test_pnl_h60": float(test_pnl)}
    else:
        val_pnl, up, dn = search_threshold_asym_de(yv, mp_t_v, mp_th_v)
        test_pnl, n_act = compute_pnl(yt, mp_t_t, mp_th_t, up, dn)
        decision = {"gate": "asym_de", "thr_up": float(up), "thr_dn": float(dn),
                    "val_pnl_h60": float(val_pnl), "test_pnl_h60": float(test_pnl)}

    results = {"task_args": vars(args), "decision": decision,
                "nn_dir": args.nn_dir, "lgb_dir": args.lgb_dir}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)

    if not args.no_wandb:
        try:
            import wandb
            os.environ.setdefault(
                "WANDB_API_KEY",
                "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",
            )
            wandb.init(project=args.wandb_project, entity="cjxh21-Tsinghua University",
                       name=args.wandb_name or os.path.basename(args.out),
                       config={**vars(args)}, reinit=True,
                       settings=wandb.Settings(start_method="thread"))
            wandb.log({"val/h60_pnl": float(decision["val_pnl_h60"]),
                       "test/h60_pnl": float(decision["test_pnl_h60"])})
            wandb.finish()
        except Exception as e:
            print(f"wandb fail: {e}", flush=True)

    print(f"RESULT_LINE: gate={args.gate} val={decision['val_pnl_h60']:.4f} "
          f"test={decision['test_pnl_h60']:.4f}", flush=True)


if __name__ == "__main__":
    main()
