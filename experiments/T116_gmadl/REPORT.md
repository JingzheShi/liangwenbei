# T116 — GMADL (Generalized Mean Absolute Directional Loss) for LightGBM

**Hypothesis (T114 lit-scan-derived):** GMADL is a PnL-aligned regression loss
for GBDT, expected to give +0.5~+2.0 LOSO over T99 LGB Huber baseline. Source:
Michańków et al. arxiv 2412.18405 (2024).

**Loss form**

  L(R, R̂) = -(σ(a · R · R̂) - 0.5) · |R|^b

Gradient/Hessian (for LGB custom obj):

  ∂L/∂R̂ = -a · R · σ · (1-σ) · |R|^b
  ∂²L/∂R̂² = -a² · R² · |R|^b · σ · (1-σ) · (1-2σ)   (true)
  hess_pos = a² · R² · |R|^b · σ · (1-σ) + 1e-8     (positive-bound used)

We use the positive-majorant hessian to keep LGB's Newton step stable. The
true Hessian flips sign at σ=½ (init state), making `1e-8` floor → catastrophic
Newton blow-ups. Dropping the (1-2σ) factor preserves convexity locally.

## Critical scale calibration

The literature value `a ∈ {500, 1000, 2000}` (and the prompt's recommendation)
is calibrated for **unscaled returns** at scale ~1e-2 (daily crypto returns).

Our target Δmid_norm has **std ≈ 2e-3** (50× smaller). For sigmoid arg z = a·R·R̂
to lie in the informative range, a must scale as 1/(R·R̂) ≈ 1/(R²) ≈ 50² × paper_a.

Empirical scan (seed=42, b=2.0 unless noted):

| a       | b   | best_iter | val dirpnl  | test cum_pnl (k=1) | notes |
|---------|-----|-----------|-------------|--------------------|-------|
| 2,000   | 2.0 | 1         | -3.5e-5     | -48.47             | dead — sigmoid nearly flat |
| 10,000  | 2.0 | 1         | -2.5e-5     | -49.68             | dead |
| 50,000  | 2.0 | 3         | +3.6e-5     | -0.25              | barely alive |
| 100,000 | 2.0 | 7         | +3.2e-5     | -7.07              |  |
| 200,000 | 2.0 | 14        | +3.7e-5     | -4.33              |  |
| 500,000 | 2.0 | 38        | +4.2e-5     | -2.42              |  |
| 500,000 | 1.5 | 49        | +7.5e-5     | +3.60              |  |
| 500,000 | 1.0 | 55        | +1.05e-4    | **+13.27**         | **best b sweep** |
| 200,000 | 1.0 | 23        | +9.9e-5     | +9.67              |  |
| 1,000,000 | 1.0 | 300+    | +1.53e-4    | **+23.74**         | still climbing → re-train at 600 rds |
| 2,000,000 | 1.0 | 1       | 0           | 0.00               | sigmoid saturated — dead |

The prompt's `a ∈ {500, 1000, 2000}` configs do not train at all on our data.
We instead run **3 calibrated configs** that actually learn:

- (a=500000, b=1.0)
- (a=500000, b=1.5)
- (a=1000000, b=1.0)

× 3 seeds (1, 7, 42) = 9 trains.

## Training (in progress)

Pipeline matches T99 LGB Huber: schemeP cache 359 features, V4 split (train
date 0-75, val 76-79, test 96-119 5-sym), aug_a [0.8,1.2] concat, class-balanced
sample weights, num_boost_round=600, early_stopping=40 on custom feval (dirpnl
= mean(sign(p)·R - fee·|sign(p)|) at fee_thr=2e-4).

LGB CPU (custom obj + GPU is brittle in 4.x). 18 threads.

## Results (TBD — populated after eval_gmadl.py runs)

See `eval_gmadl.json` and `results.json`.

## Eval protocol

DE 2D asymmetric thresh search (sym-shared) per ensemble, matching
T99/T87/T89/T95 framework. 5 DE seeds × 80 maxiter × popsize=24 each.

Compares:
- T99 Huber 3-seed (baseline) vs each GMADL config 3-seed (single)
- Huber + GMADL stacks at w ∈ {0.5, 1.0, 1.5}
- Cross-correlation Huber↔GMADL

## Conclusion (TBD)
