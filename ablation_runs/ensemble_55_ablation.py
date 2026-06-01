"""5+5 multi-seed ensemble: 5 NN (simple mean) + 5 LGB (simple mean), cross-family
weighted mean (w_NN=1.0, w_LGB=1.5), then sym or asym EV threshold search.

Usage:
  python ensemble_55_ablation.py \
      --nn-dirs <dir1> <dir2> ... \
      --lgb-dirs <dir1> <dir2> ... \
      --out <out> --gate {sym|asym} \
      [--wandb-name ...]
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
    """DE 2D bounded around the symmetric optimum (0.7×..1.5×)."""
    from scipy.optimize import differential_evolution
    _, sym_thr, _ = search_threshold_sym(y_pred, mp_t, mp_th, fee)
    lo = max(0.7 * sym_thr, 1e-9)
    hi = 1.5 * sym_thr

    def neg(x):
        s, _ = compute_pnl(y_pred, mp_t, mp_th, x[0], x[1], fee)
        return -s

    res = differential_evolution(
        neg, bounds=[(lo, hi), (lo, hi)],
        maxiter=80, popsize=25, tol=1e-9, seed=1, polish=True,
        mutation=(0.5, 1.0),
    )
    s, _ = compute_pnl(y_pred, mp_t, mp_th, res.x[0], res.x[1], fee)
    return float(s), float(res.x[0]), float(res.x[1]), float(sym_thr)


def load_preds(dirs):
    yv_all, yt_all = [], []
    mp_t_v_ref = mp_th_v_ref = mp_t_t_ref = mp_th_t_ref = None
    for d in dirs:
        z = np.load(os.path.join(d, "preds.npz"))
        if mp_t_v_ref is None:
            mp_t_v_ref = z["mp_t_val"]; mp_th_v_ref = z["mp_th_val"]
            mp_t_t_ref = z["mp_t_test"]; mp_th_t_ref = z["mp_th_test"]
        else:
            assert np.allclose(z["mp_t_val"], mp_t_v_ref), f"val mp_t mismatch in {d}"
            assert np.allclose(z["mp_t_test"], mp_t_t_ref), f"test mp_t mismatch in {d}"
        yv_all.append(z["yp_val"].astype(np.float64))
        yt_all.append(z["yp_test"].astype(np.float64))
    return (np.mean(yv_all, axis=0), np.mean(yt_all, axis=0),
            mp_t_v_ref, mp_th_v_ref, mp_t_t_ref, mp_th_t_ref,
            yv_all, yt_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-dirs", nargs="+", required=True)
    ap.add_argument("--lgb-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-nn", type=float, default=1.0)
    ap.add_argument("--w-lgb", type=float, default=1.5)
    ap.add_argument("--gate", choices=["sym", "asym"], default="sym")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-project", default="liangwenbei-ablation-bigger")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    yv_nn, yt_nn, mp_t_v, mp_th_v, mp_t_t, mp_th_t, yv_nn_list, yt_nn_list = load_preds(args.nn_dirs)
    yv_lgb, yt_lgb, _, _, _, _, yv_lgb_list, yt_lgb_list = load_preds(args.lgb_dirs)
    assert yv_nn.shape == yv_lgb.shape, "val shape mismatch"
    assert yt_nn.shape == yt_lgb.shape, "test shape mismatch"
    # rescale to comparable magnitudes (within-family avg already on same scale; cross-family
    # uses the prescribed weights as-is, matching project ensemble logic).
    W = args.w_nn + args.w_lgb
    yv_ens = (args.w_nn * yv_nn + args.w_lgb * yv_lgb) / W
    yt_ens = (args.w_nn * yt_nn + args.w_lgb * yt_lgb) / W

    if args.gate == "sym":
        val_pnl, thr, k = search_threshold_sym(yv_ens, mp_t_v, mp_th_v)
        test_pnl, n_act = compute_pnl(yt_ens, mp_t_t, mp_th_t, thr, thr)
        decision = {"gate": "sym", "thr_sym": float(thr), "k_sym": float(k),
                    "val_pnl_h60": float(val_pnl), "test_pnl_h60": float(test_pnl),
                    "n_act_test": int(n_act)}
    else:
        val_pnl, up, dn, sym_thr = search_threshold_asym_de(yv_ens, mp_t_v, mp_th_v)
        test_pnl, n_act = compute_pnl(yt_ens, mp_t_t, mp_th_t, up, dn)
        decision = {"gate": "asym_de_bounded", "thr_up": float(up), "thr_dn": float(dn),
                    "sym_thr_ref": float(sym_thr),
                    "val_pnl_h60": float(val_pnl), "test_pnl_h60": float(test_pnl),
                    "n_act_test": int(n_act)}

    # also report per-family within-ensemble PnL for diagnostic
    val_pnl_nn_only, *_ = search_threshold_sym(yv_nn, mp_t_v, mp_th_v)
    val_pnl_lgb_only, *_ = search_threshold_sym(yv_lgb, mp_t_v, mp_th_v)

    results = {"task_args": vars(args), "decision": decision,
               "n_nn_dirs": len(args.nn_dirs), "n_lgb_dirs": len(args.lgb_dirs),
               "diagnostic": {
                   "val_pnl_nn_only_avg_sym": float(val_pnl_nn_only),
                   "val_pnl_lgb_only_avg_sym": float(val_pnl_lgb_only),
               }}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2), flush=True)

    np.savez(os.path.join(args.out, "preds_ens.npz"),
             yp_val=yv_ens.astype(np.float32), yp_test=yt_ens.astype(np.float32),
             mp_t_val=mp_t_v, mp_th_val=mp_th_v,
             mp_t_test=mp_t_t, mp_th_test=mp_th_t)

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
