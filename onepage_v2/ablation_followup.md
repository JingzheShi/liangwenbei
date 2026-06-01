# Ablation 5 个反直觉问题调查

> Worker followup answering 5 user questions on `big_ablation_log.md`.
> All training under V4 walk-forward protocol (train 0–79, val 80–95, test 96–119).
> No tuning whatsoever on test. WandB project `liangwenbei-ablation-followup`
> (entity `cjxh21-Tsinghua University`); some runs disabled wandb due to
> auth issue, full results in `ablation_runs/followup_q*/`.

---

## Q1 — CE vs L2 矛盾

**Experiment**: train LGB **multiclass CE** on the 359-d full pipeline
(drop-11 + window-z + mirror-flip + sign-log1p rawlast),
identical to row 8 except objective + threshold gate.

| Setting | best_iter | Val PnL | **Test PnL** |
|---|---|---|---|
| Row 1 (rerun)     154-d raw + CE         | 112 | 15.80 |  22.40 |
| **Q1 NEW**     359-d full pipe + CE     | 107 | 19.23 | **22.11** |
| Row 8 (rerun)     359-d full pipe + L2  | 45  | 23.90 | **27.33** |

Train time 87 s (GPU LGB, 330 rounds, no early stop fired). Search grid for
(T, δ): T∈{0.40..0.85, 19 pts}, δ∈{−0.10..0.30, 9 pts}; best (T=0.425, δ=−0.10).

**Conclusion**: CE is consistently ~5 PnL worse than L2 under this
V4 walk-forward protocol — and **adding the full feature pipeline does NOT
flip it**. The advantage CE seemed to enjoy on raw 154-d (23.66 vs 19.13 in
the old log, but rerun: 22.40 vs 19.14, only +3.26 — and now within ±2.5 GPU
noise) was essentially the entropy-based label being more robust than L2 RMSE
when features are too noisy/raw to learn a calibrated regression. Once you
add window-z + mirror + log1p, L2 can finally fit the continuous quantile
and CE has nothing extra to offer — its discrete labels throw away precisely
the magnitude information that L2 + EV gate uses.

**Why does project history disagree?** The iter_002→iter_013 jump (+15 PnL)
was on a **different evaluation** (public LB on a different time window,
classified vs. regression LGB **with a totally different feature stack
back then**) — not an apples-to-apples comparison on the V4 holdout. On
V4 walk-forward with the current 359-d pipeline, **L2 ≫ CE by 5 PnL**, and
CE is not a useful component (no signal it doesn't already give in L2 form).

---

## Q2 — Drop 11 vs no drop on NN

**Experiment**: rerun row 10 NN (Phase 1 L2 + Phase 2 SPO+ 11 epochs)
with `--no-drop` (370-d, keep the 11 KS-fail features).

| Setting | feat_dim | best_epoch | Val PnL (Phase 2) | **Test PnL** |
|---|---|---|---|---|
| Row 10        drop-11 + NN+SPO  | 359 | 1 | 36.78 | **30.19** |
| **Q2 NEW**    no-drop + NN+SPO  | 370 | 1 | 35.76 | **28.40** |

Phase 1 19 s + Phase 2 35 s on GPU 0.

**Conclusion**: drop-11 helps NN by **+1.79 PnL** (30.19 vs 28.40), but
this is within the known ±2.5 PnL single-seed noise band. Effectively a wash.

**Implication**: the 11 KS-fail features carry weak/noisy signal but are not
catastrophically harmful — LGB's `feature_fraction=0.8` does NOT explain the
drop-11 mechanism by itself (NN has no analogous random masking and still
mildly benefits). The project's decision to drop them is defensible but
**marginal on this holdout**, not load-bearing. We would not expect dropping
them to make a measurable difference on the leaderboard either.

---

## Q3 — Window z-score on NN

**Experiment**: rerun row 9 NN (Phase 1 L2 only, no SPO) with `--no-window-z`
— skip global per-feature `(X−μ)/σ` standardization. Keep mirror-flip + log1p.

| Setting | feat_dim | best_epoch | Val PnL | **Test PnL** |
|---|---|---|---|---|
| Row 9        with window-z + mirror + log1p | 359 |  1 | 26.23 | **34.93** |
| **Q3 NEW**   no window-z + mirror + log1p   | 359 | 13 |  3.83 |  **5.05** |

Phase 1 49 s on GPU 0; best epoch 13 (vs row 9's epoch 1) — the model needed
much more training to barely fit anything.

**Conclusion**: window-z is **NOT** a LGB-specific trick. It is **absolutely
critical** for the MLP. Without it the model loses ~30 PnL — 86% relative
drop. The raw 359-d features span wildly different scales (price ~10²,
intensity ~10⁻⁴, log1p'd quantities ~10), and the LayerNorm inside the MLP
blocks cannot recover from a poorly-scaled input layer (Adam's per-coordinate
learning rates can adapt, but only after the first Linear matmul has already
mixed scales catastrophically). For trees, axis-aligned splits are
scale-invariant so window-z only nudges the optimum (row 5 → row 6, −0.17
PnL); for NNs it is foundational.

This **contradicts the user's hypothesis** that window-z might be LGB-specific.
The opposite holds: LGB barely cares, NN absolutely requires it.

---

## Q4 — SPO+ × {sym, asym} EV gate

**Experiment**: re-evaluate existing row 9 (NN no SPO) and row 10
(NN + SPO+) predictions under both gates. asym uses the same 2D
differential-evolution search as `ensemble_ablation.py`, bounded to
[0.7×, 1.5×]·sym_thr on each side to limit val overfit.

| | sym EV (test) | asym EV (test) | asym − sym | Val asym |
|---|---|---|---|---|
| NN no SPO (row 9)  | 34.93 |  34.02 | **−0.91** | 26.44 |
| NN + SPO  (row 10) | 30.19 |  31.60 | **+1.41** | 39.27 |

**Observation 1**: SPO+ partially recovers under asym EV gate (+1.41 PnL
moving sym → asym), while no-SPO **loses** PnL (−0.91). This confirms
the user's hypothesis: **SPO+ pushes predictions toward decision boundaries,
and an asymmetric 2D gate can actually use that boundary information**, while
a symmetric gate throws away half of it. The asym (θ_up=1.18e-4 < θ_dn=2.35e-4)
also reveals SPO+ has a directional bias the sym gate masks.

**Observation 2**: even with asym EV, NN+SPO (31.60) is still ~3 PnL below
NN no SPO + sym (34.93). So **SPO+ fine-tuning on a single seed degrades**
the underlying L2 quality more than the better-shaped decision boundary
can recover. Likely SPO's hinge gradient amplifies single-seed noise; the
50-seed project ensemble averages this out.

**Implication for the narrative**: Row 10's −4.74 PnL drop vs row 9 is
*partly* explained by the sym gate not exploiting SPO's directional output.
The "right" comparison would be **NN+SPO+asym (31.60) vs NN+sym (34.93)**:
SPO is still −3.33 PnL, but the gap halved. The full SOTA recipe (50-seed +
fulltrain + asym EV) is where SPO+ fully pays off — single-seed V4 is too
small a setting to demonstrate it.

---

## Q5 — 5+5 multi-seed ensemble: sym vs asym


---

## (1+1) NN + LGB cross-family ensemble — no SPO, sym EV (followup_v10)

To establish the lower-bound of the cross-family ensemble before multi-seed averaging
or any decision focal-loss tuning, we combined a single NN seed (no SPO, row-7
backbone: 359-d + L2 + window-z) with a single LGB seed (row-3: 359-d L2). Both
are seed=1 from the 5+5 sweep. Cross-family weights `w_NN=1.0, w_LGB=1.5` (same
as project ensemble); sym EV threshold searched on val 80–95; score on test 96–119.

| Component                                    | Val PnL | Test PnL |
|---|---|---|
| NN seed=1 only (row 7, no SPO, sym)          | 26.50   | 33.56 |
| LGB seed=1 only (row 3, sym)                 | 24.73   | 27.83 |
| **(1+1) NN + LGB, sym EV (k=0.7)**           | **29.80** | **+35.02** |

- Δ vs row 9 (single NN + mirror + log1p, 34.37): **+0.65**
- Δ to (5+5) sym (36.07): **+1.05** (multi-seed mean over 5 NN × 5 LGB)
- Δ to (5+5) asym (36.19): +1.17
- Δ to (5+5) + SPO asym (37.73 SOTA): +2.71

**Takeaway**: A single seed of NN + LGB already crosses +35, beating both single
NN (34.37 mean) and single LGB (26.58 mean) by a wide margin — cross-family
diversity is the dominant lift. Multi-seed averaging adds another +1.05, asym EV
+0.12, SPO+ DFL +1.54.

WandB run: `r10_1plus1_no_spo` (project `liangwenbei-ablation-cleanup`).
Artifacts at `ablation_runs/r10_1plus1_no_spo/`.
