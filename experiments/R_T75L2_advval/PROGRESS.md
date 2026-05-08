# R_T75L2_advval — Adversarial-validation reweighted T75 LGB L2

## Hypothesis
Train binary classifier `is_test_distribution` (label=1 if date>=80 future
[val+test], label=0 if date<=79 past). Use OOF proba on past samples as a
multiplicative sample weight to upweight training samples that "look like"
the test distribution. Compare 3-seed avg of T75 LGB L2 with this trick vs
canonical 5-seed T75 baseline (averaged over the same 3 seeds for fairness).

vs T101 R1 date-decay (linear w=1+1.5·date/79, +0.72 LOSO confirmed): adv val
is **data-driven** rather than prior — model identifies which training samples
are most distributionally aligned with future, instead of assuming time-linear.

## Step 1: Adversarial classifier (DONE)
- 5-fold date-block CV on past dates (0-79), 16 dates per fold
- Each fold: train on (past minus held) ∪ ALL future, eval on held past + sample of future
- Drop date/sym/time (forbidden) — schemeP 359 dim same as T75
- LGB binary, num_leaves=127, GPU
- **Mean fold AUC = 0.9955** (very high — past/future distributionally separable)
- Top discriminative features: ask_mean, bid_mean, totalasize, open, totalbsize, ask_diff10
  (price levels and order book size — drift over time)

## Step 2: Calibrate to sample weight
- p_clip = clip(proba, 0.01, 0.99)
- sw_raw = p_clip / (1-p_clip)  (logit-style)
- sw_clip = clip(sw_raw, 0.25, 4.0)
- isotonic regression over sorted (p_clip → sw_clip), normalize mean=1
- Result distribution: p10=0.42, p50=0.42, p90=1.87, max=6.77, std=1.74
  (bimodal: most samples deweighted to 0.42, ~10% upweighted strongly)

## Step 3: 3-seed × T75 LGB L2 trick training (in progress)
Seeds {42, 7, 13}, hyperparams identical to T75 baseline, sw = sw_class_balanced × sw_advval (renormalized).
GPU on RTX 3080, ~70s per seed.

Seed 42 result: best_iter=62, val_mse=5.04e-6, test_corr=0.1315

## Step 4: DE asymmetric thresh + LOSO-equiv eval (DONE)
3-seed avg pred → DE search (5 DE seeds) → best per-sym PnL sum = LOSO-equiv.
Compare to 3-seed avg of canonical T75 baseline (same 3 seeds).

## Result (NEGATIVE)

| Variant                    | LOSO-equiv | Δ vs baseline |
|----------------------------|-----------:|--------------:|
| Baseline T75 LGB L2        | +35.4487   | (anchor)      |
| Trick aggressive [0.25,4]  | +25.2739   | **−10.1748**  |
| Trick mild [0.5,2.0]       | +35.1879   | **−0.2608**   |

Hypothesis falsified. Adv-val classifier AUC was too high (0.9955), driven by
price-level features that drift over time. Aggressive reweight concentrates
training mass on outlier past samples → loses signal. Mild cap merely adds
noise.

See `results_summary.md` for full analysis. NOT iter_017 candidate.
R1 date-decay (+0.72) remains the active L2 reweighting trick.
