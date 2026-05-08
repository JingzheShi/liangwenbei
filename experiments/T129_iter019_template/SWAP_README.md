# T129 — iter_019 packaging template (Roll+TSRV plug-in)

This dir is a **byte-identical copy** of `T127_iter018_v1_pkg` (the iter_018 v1 SOTA stack:
T87 NN 5-seed + T75 LGB 5-seed + per-sym β-conformal abstain wrapper).

It is the staging template for **iter_019 = iter_018 + Roll+TSRV winner LGBs**, if/when the
`R_roll_tsrv` experiment produces a winning variant.

## State as of T129 worker run (2026-05-08)

R_roll_tsrv worker [`ad0d915b`] completed but its remote work dir
(`/root/lwb_work_roll_tsrv/`) is **empty / never populated** — no V1 (clean-target)
or V2 (meta-feature) preds or models exist. R_roll_tsrv produced no winner this round.

**iter_019 fallback** (per PM plan): ship same as iter_018. No new submission zip is
created in the fallback path; this template stays as a ready-to-swap scaffold.

## How to swap in Roll+TSRV LGB winner (if/when available)

Two trick branches:

### Trick A — clean-target LGBs (V1: OLS slope through 6 forward mid points)

Same schemeP 359-d features, same NN, same conformal wrapper. Only the 5 T75 LGB
boosters change. Predictor.py and config.json **do not need to change**.

```bash
TPL=/root/projects/liangwenbei_workdir/experiments/T129_iter019_template/pkg
SRC=/root/lwb_work_roll_tsrv  # adjust if winner artifacts live elsewhere

# 1. Drop in the 5 retrained LGB boosters (V1)
for S in 1 7 13 42 100; do
  cp ${SRC}/V1/model_h60_seed${S}.txt ${TPL}/model_h60_seed${S}.txt
done

# 2. Recompute per-sym sigma on local val 442k (combined_pred dist changes!)
cd /root/projects/liangwenbei_workdir/experiments/T129_iter019_template
python3 compute_sigma_and_verify.py --pkg-dir ./pkg

# 3. Re-search DE-optimal (thr_up, thr_dn, w_nn, w_lgb) on the new combined preds
#    (β stays from R_conformal_select V8 fit; β is robust to LGB target swap.
#     If verify_cum_pnl drops by >0.5 vs iter_018 baseline 41.49, redo β fit too.)
python3 ../../R_conformal_select/refit_thresholds.py --pkg-dir ./pkg

# 4. End-to-end verify on 442k
python3 e2e_spot_check.py --pkg-dir ./pkg

# 5. Pack
( cd ./pkg && zip -r /root/projects/liangwenbei_workdir/submission_050818_iter019_v1_conformal_plus_rolltsrv_A.zip . )
```

### Trick B — meta-feature LGBs (V2: + 4 noise meta-features in schemeP)

Same NN, same conformal wrapper. The schemeP feature dim grows from 359 → 363.

**Predictor.py and config.json need updating**:
- Add 4 meta features to `config["feature"]`: `ep_var_proxy`, `signal_var_proxy`,
  `noise_share`, `log_snr`
- Update `fast_features.py` and `fast_features_batch.py` to compute the 4 meta features
  per tick (vectorized; depends on `roll_eff_spr_ratio_W50`, `rv_w50` already in pipeline)
- Re-export with the V2 LGB boosters
- Same DE re-search + verify + zip flow as Trick A

The four meta features are defined in `R_roll_tsrv/train_lgb_roll.py` (`clean_target` /
V2 prep block).

## What stays unchanged across both tricks

- `nn_h60_seed{1,7,13,42,100}.npz` (T87 NN, byte-identical iter_015 v1)
- `pkg/Predictor.py` core gate logic (per-sym β-conformal abstain band)
- `thresholds.json` `conformal_wrapper.per_sym_beta` (β robust to small target/feature
  changes; only re-fit if verify_cum_pnl drops noticeably)
- The `sym` config injection (used as lookup key only, **not** fed to forward)

## Hard constraints (CRITICAL_CONSTRAINTS.md, do not violate)

1. `date` is zeroed at eval time — never use as feature.
2. Test points are shuffled — Predictor must hold no cross-call state.
3. sym 0–4 may include unseen-stocks-at-eval — model is sym-agnostic in forward;
   sym only used as lookup key for per-sym β/σ (with mean fallback for OOD IDs).
