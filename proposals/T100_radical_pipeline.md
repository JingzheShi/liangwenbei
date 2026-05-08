# T100 — Radical Pipeline Rethink (data → features → model → decision)

> **Mode**: Synthesis after reading PROGRESS, T75/T87/T99/T98/iter_015 packages, and brainstorm rounds R36–R44.
> **Date**: 2026-05-08
> **Goal**: Identify the pipeline-level *fundamental* weaknesses that no T-experiment has yet hit, and propose 5–10 "basic but disruptive" tricks (in the spirit of T75: target=Δmid, +9.79 LOSO single-trick).

---

## Part 1 — Diagnosis: 5 pipeline-level weaknesses

These are things we have **structurally** built into our pipeline that are independently dubious. None is a hyperparameter or a feature; each is a load-bearing design choice we never re-questioned after iter_001.

### W1. **One-horizon monomania** — we throw away 4 of 5 leaderboard slots

- **What we do**: `iter_015/thresholds.json` ships h=60 active=true and h={5,10,20,40} active=false. Every test point we predict `flat` for 4 horizons. Score = max over 5 horizons.
- **Why it's a fundamental hole**: the leaderboard rule is `max(score_h5, score_h10, score_h20, score_h40, score_h60)`. The 4 inactive horizons add **exactly 0** — they cannot lower the max. **Any** of them giving a positive score is a *free improvement*. T98 already trained 5-horizon LGB+CB regressors (40 boosters × 5 horizons sit on disk) with per-horizon DE: h=20 LOSO ≈ +25 single-base, h=40 ≈ +28. We *have* the models; we *aren't shipping them*. Even if h=60 still wins on the public test, **paying the multi-horizon insurance costs zero score** and saves us if h=60 specifically degrades on a quirk of the private test set.
- **Why it never got fixed**: each iter we optimized one horizon, declared "h=60 wins LOSO best of 5", and shipped only that horizon. We forgot that `max(a, 0, 0, 0, 0)` ≤ `max(a, b, c, d, e)`. The trap is "don't add complexity unless it improves" — but here adding 4 horizons that even *break even* improves max-rule expected value (variance reduction).
- **Connected lever (compounding)**: 5 per-row predictions are an *uncertainty oracle*. If h=5/10/20 disagree on sign with h=60, we should *not* trade h=60. We've never used inter-horizon coherence as a confidence gate.

### W2. **A 16-day blackout in the data** — train ends at day 79, test starts at day 96

- **What we do**: `train_regr.py` builds V4 walk-forward as `train = date 0–75`, `val = date 76–79`, `test = date 96–119`. Date **80–95** is *never* used. That's a 16-day chunk of fully-labeled data discarded for no methodological reason.
- **Why it's fundamental**: the platform test is **time-OOD** (T80 calibration shows -17 LOSO→platform gap, almost certainly because dist shifts over 30+ ticks of dates). T66 adversarial validation confirms train-vs-test is trivially separable. **Date 80–95 is 16 days closer to the test distribution than train ends.** Adding it to train at full weight is a (free) "domain adaptation by data" — no fancy DRO needed.
- **Why it never got fixed**: T69 once tried "V4 + full data 0–95" and got +0.02 LOSO-equiv, called it "saturated", and moved on. But T69 was under iter_011's CE objective (3-class + DE-4D thresh) — *not* under the regression+EV-gate regime. The +0.02 was a *no-op test under a saturated CE pipeline*; we never re-ran it under regression where the magnitude signal is preserved. Plus T69 didn't *time-weight* the recent days.

### W3. **The model sees only the LAST tick of raw features** — temporal shape of LOB is thrown away

- **What we do**: `Predictor._compute_batch_features` does `raw_last = X3d[:, -1, :].astype(np.float32)` — the model receives **154 raw features at t=−1 only**, plus 216 *summary statistics* over the 100-tick window (z-scores, EWMAs, RV, OFI). That's it. 359 dims.
- **Why it's a fundamental information loss**: a monotonically rising bid sequence and a noisy bid sequence with the same mean produce **identical summary statistics** but reflect *radically different* short-term futures. The summary-stat encoding throws away any path-dependent structure beyond what the hand-crafted summaries capture. CNN/RNN baselines failed under CE not because temporal info is useless — they failed because of **target compression**, not arch. Under regression + Δmid target the temporal shape becomes informative again, and we are denying it to the model.
- **Why it never got fixed**: GBDT can't natively eat (100, 154) tensors, so we projected to (1, 154) + summaries. The *cheap fix* is **multi-tick raw snapshots** — instead of just t=−1, feed t ∈ {−1, −5, −20, −50}. That's 4×154 = 616 raw features (vs. current 154); still tabular, still GBDT-friendly. We've never tried it. T33 ("long window") tried > 100-tick window which is a different (and infrastructure-blocked) lever; nobody tried "use multiple ticks within the existing 100-tick window".

### W4. **Target = mean Δmid; the model learns to fit gradient on rows where alpha cannot exist**

- **What we do**: `train_regr.py:88` defines `y_regr = (mp_t60 − mp_t)/(mp_t + 1)`. L2 loss penalizes residuals on **every** row — including the ~53% of rows where `|Δmid| < fee` and the optimal action is "do nothing" regardless of what we predict. The gradient pulls the model toward fitting *noise* on those rows because L2 doesn't know about the fee threshold.
- **Why it's fundamental**: T75 fixed the *upstream* compression (3-class label → continuous Δmid), but did not fix the *downstream* misalignment (the target still doesn't know the decision rule). The model spends ≥50% of its capacity learning to predict tiny Δmid values that the EV gate will *always* reject. The cleanest fix: make the target **the actual PnL of the optimal action**, which is exactly zero on sub-fee rows:
  - `y_NetPnL = sign(Δmid) · max(0, |Δmid_norm| − 2·fee_norm)`
  - Target = 0 ⇒ zero gradient ⇒ no capacity wasted ⇒ all capacity flows to alpha rows.
- **Why it never got fixed**: T57 tried *sample-weight* magnitude weighting (CE) → −3 to −11. T92 tried magnitude weighting under regression → also failed (R44 hypothesis falsified). But neither tried a **target-level transformation**. R38 #3 proposed exactly this signed-net-PnL target, and it has *never been built*. It is a different mathematical intervention from sample-weight (target-level vs. weight-level).

### W5. **Decision = single global threshold; ensemble disagreement is thrown away**

- **What we do**: `Predictor._ev_gate_predict` is `out[pred > thr_up] = 2; out[pred < -thr_dn] = 0`. The threshold is a *single pair* tuned on the test set (`thr_up=3.58e-4, thr_dn=2.16e-4`). The ensemble of 5 LGB + 5 NN gives us **10 predictions per row** — but we collapse them to 1 mean before the gate. The standard deviation across 10 base-model predictions is a ready-made uncertainty proxy that we discard.
- **Why it's fundamental**: a trade where 5/5 LGB seeds and 5/5 NN seeds *all* agree on +0.0005 is qualitatively different from a trade where the mean is +0.0005 but seeds range from −0.0003 to +0.0013. The first is a high-conviction signal; the second is noise-dominated and almost certainly a bad trade after fee. R37 diagnosed the public team trades **2× more, half per-trade alpha** — they're harvesting the right *marginal* signals while we conservatively gate them all. The tool we need to discriminate marginal-and-real from marginal-and-noise is **already in our model output (10 predictions per row)** and we are silently averaging it away.
- **Why it never got fixed**: ensemble disagreement as gate is a basic uncertainty trick from Bayesian ML, but our DE thresh sweep optimizes only `(thr_up, thr_dn)` and never thought to add a `pred_std` axis. R13's "calibration + EV gating" focused on temperature scaling and z-gate, not ensemble variance.

---

### How we got into each trap

A common pathology unifies all five:

> **We optimized each iteration "vs. the previous LOSO number"** instead of asking "is this pipeline the right pipeline?". Once T75 unlocked +9.79 by changing target, we instinctively treated the rest of the pipeline as fixed and only varied feature sets / loss flavors / ensemble weights. The 4 inactive horizons, the 16-day data hole, the last-tick-only raw input, the gate-on-mean — these were all there in iter_001 and survived 14 iterations because **no iteration explicitly asked "should this be different?"**.

---

## Part 2 — Disruptive trick proposals

For each trick: (1) one-line description, (2) why "basic" (well-known concept), (3) why "disruptive" (different from current and >+1 LOSO plausible), (4) implementation difficulty, (5) expected LOSO lift, (6) failure risk.

### Trick T100-A: **Multi-horizon active inference + horizon-coherence gating**

1. Activate h=5/10/20/40 in `thresholds.json` using T98's already-trained 5-h × 5-seed LGB+CB regressors. Per-horizon DE thresholds tuned independently. THEN gate the headline h=60 trade by *coherence* across the 5 horizons: trade only if ≥3 of 5 horizons agree on sign and `|pred_h60| > thr_h60`.
2. **Basic**: Multi-output regression + ensemble agreement gating — both 100-level ML.
3. **Disruptive**: We have been *single-horizon* the entire competition, throwing away free max-rule shots and free uncertainty signal. The 5-horizon ensemble is **already trained** (T98) — there is literally a packaging change separating us from this.
4. **Difficulty**: 4–6 h. Repackage `iter_015_pkg/Predictor.py` to load 5-horizon ensembles (already on disk in T98); add coherence rule to h=60 path; DE-tune 5 per-horizon thresholds on LOSO OOF (purged split, not on test).
5. **Expected lift**: +1.0 ~ +3.0 LOSO. Mechanism is *(a)* max-rule pickup if any auxiliary horizon edges out h=60 on a fold, *(b)* ~10–15% reduction in h=60 false positives via coherence gate.
6. **Risk**: low. Failure mode = coherence gate too strict ⇒ active count drops ⇒ score drops. Mitigated by DE-tuning the coherence cutoff (3-of-5 vs 2-of-5) on OOF.

### Trick T100-B: **Net-PnL target — `y = sign(Δmid)·max(0, |Δmid_norm| − 2·fee_norm)`**

1. Replace `y_regr = (mp_t60 − mp_t)/(mp_t + 1)` with the *signed clipped fee-aware* target. 53% of rows get y=0 ⇒ zero gradient ⇒ no capacity on noise. Decision rule trivially `pred > 0 → long, pred < 0 → short, else flat` (single-threshold or no-threshold variant).
2. **Basic**: target shaping / decision-aware regression — Lopez de Prado AFML Ch. 3, R38 #3, R10 S6.
3. **Disruptive**: T75's win was "use continuous target instead of 3-class". This is the **next link in that chain**: "use the *decision-aligned* continuous target instead of the *prediction-aligned* one". Different from T57/T92 magnitude-weight failures because the intervention is at the **target** level, not the **loss-weight** level.
4. **Difficulty**: 2 h. Single line change in `train_regr.py:88` to redefine `y_regr`; same model architecture, same features, same 5 seeds. EV-gate becomes `pred > 0` (no fee subtraction).
5. **Expected lift**: +1.0 ~ +2.5 LOSO. The model's effective sample size doubles (no gradient on 53% sub-fee rows means the remaining 47% drive *all* learning, with cleaner labels).
6. **Risk**: medium-high. The bimodal target distribution (massive spike at 0, thin tails at ±) may confuse LightGBM's split selection. Mitigation: pilot single-seed first; if test_corr drops vs. T75 by > 30%, revert.

### Trick T100-C: **Ensemble-uncertainty gate — `trade iff |pred| > τ AND pred_std < β·|pred|`**

1. Replace single global threshold with: trade only if `|pred_mean| > τ` AND ensemble-std `pred_std < β · |pred_mean|` (signal-to-noise > 1/β). `pred_std` is the std across the 10 base predictions (5 LGB + 5 NN) we already compute. β is one new hyperparameter.
2. **Basic**: Bayesian uncertainty quantification via deep ensembles (Lakshminarayanan 2017); Kelly criterion with variance term.
3. **Disruptive**: We presently *average away* the 10 predictions before gating, throwing away the disagreement signal. R37 says the public team trades 2× more at half per-trade alpha — they're harvesting the marginal trades that pass agreement filter. The agreement filter is *free* (no retraining).
4. **Difficulty**: 1–2 h. Modify `_ev_gate_predict` to take an array of 10 preds, compute mean+std, apply both filters.
5. **Expected lift**: +0.7 ~ +2.0 LOSO. Net effect: shifts trades from low-confidence (lossy) to high-confidence (profitable). May *also* enable lowering `thr_up`, increasing active count — recovering the public team's 2× volume.
6. **Risk**: low. Worst case β=∞ degenerates to current behavior. DE-tunable.

### Trick T100-D: **Multi-tick raw snapshots — feed t ∈ {−1, −5, −20, −50} not just t=−1**

1. In `_compute_batch_features`, replace `raw_last = X3d[:, -1, :]` with `raw_panel = X3d[:, [-1,-5,-20,-50], :]` flattened to (N, 4·154 = 616). Append to derived features ⇒ ~975-d input. Same target, same model.
2. **Basic**: panel-data tabular features — Optiver Realized Vol 1st-place built 600+ features from 3s snapshots by exactly this method.
3. **Disruptive**: GBDT in our pipeline currently sees only one row of raw LOB plus summary stats. Hand-crafted summaries (means, EWMAs, RV) cannot recover phase / monotonicity / step-pattern information. Multi-tick snapshots restore the **path** — at near-zero cost (4× the feature dim, but still tabular, still well within 2 GB model and 3 h inference budget). Different from T33 "long window" (which extended the window beyond 100 ticks; infrastructure-blocked) — this lever stays inside the existing 100-tick window.
4. **Difficulty**: 4–6 h. Cache rebuild for new 970-d feature vector, retrain 5-seed LGB and 5-seed NN, repackage Predictor, sanity tests.
5. **Expected lift**: +0.5 ~ +2.5 LOSO. Plausible mechanism: GBDT picks up "is bid1[t=-50] < bid1[t=-1]?" splits that no derived feature currently encodes. Confidence is moderate — could be that summaries already capture all signal.
6. **Risk**: medium. 970-d feature space is large; needs strong feature_fraction reg. May overfit train if poorly regularized.

### Trick T100-E: **Train-on-0–95 with time-decay reweighting**

1. Drop the gap: train = date 0–95 (concatenated train + val + previously-unused 80–95). Apply linear time-decay sample weight `w_t = 1 + α·(date_t / 95)` to upweight recent data (test is date 96+, so date 95 is closest in time). Use a small purged sub-slice for early-stopping (e.g., date 0–10 as a "very-old" holdout).
2. **Basic**: domain adaptation by time-recent reweighting — standard in financial ML (Lopez de Prado AFML Ch. 4, "fractional differentiation" + time-decay).
3. **Disruptive**: We have a 16-day chunk of *fully-labeled* data sitting unused between train and test. T69 tried this under the *CE* regime and got +0.02 (saturated) — but under the regression target the magnitude information is preserved, and 16 extra days × 5 syms × ~2400 windows/sym/day = ~190k extra training rows is **non-trivial new data**, *closer to the test distribution than train ever was*.
4. **Difficulty**: 3–5 h. Modify `build_v4_split` to use date 0–95, choose a sound early-stop scheme (e.g., random 5% holdout from 0–95, OR use date 0–10 as "purged early holdout"), re-tune `num_boost_round` upper bound.
5. **Expected lift**: +1.0 ~ +3.0 LOSO and *crucially* **reduces the LOSO→platform gap** (T80's −17 gap is largely time-OOD).
6. **Risk**: low-medium. Risk of "more data hurts because labels at recent dates are differently distributed" is real but small; the time-decay weight cushions it.

### Trick T100-F: **Bootstrap-CV-stable thresh selection (anti-test-fit)**

1. Replace single DE on full 442k test with: K=5 CV-DE — split test 5 ways, DE on 4 folds, take median over 5 thresh estimates. Or: 20-bootstrap DE on test 80% subsamples + median threshold.
2. **Basic**: cross-validation for hyperparameter selection — most basic ML 101.
3. **Disruptive**: We presently fit (`thr_up`, `thr_dn`) on the same 442k test we report on — this is the dominant source of T80's −17 LOSO→platform gap. CV-DE *lowers* LOSO but *raises* platform transmission ratio from 0.70 → ~0.85. **Pure transmission gain**: +1 LOSO (from ↓) → −0.7 platform; transmission lift +0.15 → +5–6 platform. Net: **+5 platform from neutral LOSO**. R39 #1.
4. **Difficulty**: 2–3 h. Wrapper around the existing DE; ~5× the DE compute (still < 1 h on this hardware).
5. **Expected lift**: 0 (or slightly negative) on **LOSO**, +3 ~ +6 on **platform** (via transmission ratio). Verify with held-out slice (date 96–105 fit, 106–119 eval).
6. **Risk**: low. Hardest part is *believing* the LOSO drop is OK — must use T80 transmission formula, not raw LOSO, to compare.

### Trick T100-G: **Inference-time augmentation (TTA) — average over 5 input perturbations**

1. At inference, for each 100×154 window, compute 5 augmented copies via `aug_a` per-feature scale [0.95, 1.05] noise, predict each, average. Apply to *both* LGB and NN.
2. **Basic**: TTA is standard in vision; equivalent to free ensemble at inference.
3. **Disruptive**: We currently train with `aug_a` augmentation but apply only the original at inference. The `aug_a` augmentation explicitly creates an invariance the model is asked to respect — TTA at inference *uses* that invariance to denoise. Shuffle-invariant per-window ⇒ no constraint violation.
4. **Difficulty**: 2 h. Wrap `_compute_batch_features` to produce 5 perturbed copies; predict; average.
5. **Expected lift**: +0.3 ~ +1.0 LOSO. R39 #21.
6. **Risk**: low. Cost = 5× inference time = ~5 min on full 442k (was 0.6 s × ~430 = ~4 min for 1×). Well within 3-h budget.

### Trick T100-H: **Window-self-normalization for OOD robustness**

1. Inside each 100-tick window, compute per-feature window-median + window-MAD; replace each feature value with `(x - median) / (MAD + ε)`. *Or* concat to existing features (preserve absolute info too). Each window normalizes itself ⇒ guaranteed sym-agnostic ⇒ test-stock OOD robust.
2. **Basic**: per-sample standardization (BatchNorm-style without batch state); robust statistics.
3. **Disruptive**: Our normalization (NN's `feat_mean`/`feat_std`) is **global across all train data** — so a test sym whose distribution differs from train pays the OOD cost. Window-self-norm is *automatic OOD adaptation*: each window's own context defines its scale. Different from T7 z-score (which used raw rolling z; this proposes per-window robust median/MAD as an additive feature view).
4. **Difficulty**: 5 h. New cache, full retrain.
5. **Expected lift**: +0.5 ~ +2 LOSO, with disproportionately better LOSO→platform transmission (the new sym OOD that the platform may include).
6. **Risk**: medium. Loses absolute level information; mitigation = concat to original features rather than replace.

### Trick T100-I: **Per-horizon decision *coherence* exploitation, not vote**

1. Currently the DE threshold operates on a single horizon's prediction. **Replace** with: trade h=60 only if *all* horizons agree on sign of Δmid (or weighted vote with weights ∝ horizon's standalone LOSO PnL). Equivalent to "5-of-5 horizon vote required for h=60 trade", but use **continuous coherence** = `mean over h of sign(pred_h)` ∈ [−1,+1] as the coherence score; trade if `|coherence| > 0.6`.
2. **Basic**: ensemble vote, signal coherence (signal processing).
3. **Disruptive**: Different from T77 (which voted on 3-class probs of multi-horizon classifiers). Here we use the regression signs of multi-horizon *regressors* — produces continuous coherence and is naturally compatible with the EV gate.
4. **Difficulty**: 3 h. Stack on top of T100-A (which provides the multi-horizon predictions).
5. **Expected lift**: +0.5 ~ +1.5 LOSO. Stacks with T100-A.
6. **Risk**: low.

### Trick T100-J: **Direct PnL-as-EV regression — predict *signed expected NetPnL post-fee* per row**

1. Like T100-B, but also subtract a regression on `|Δspread|·fee` from the target. The actual PnL formula is `sign·Δmid − fee·|sign|·((mp_th+1)+(mp_t+1))/(mp_t+1) ≈ sign·Δmid − 2·fee·(1 + Δmid/(2·(mp_t+1)))`. The fee is *not* a constant in the normalized return units — it scales by 1 + Δmid/(2(mp_t+1)). Currently our `fee_norm = 1e-4` constant. The correction is small (Δmid is tiny vs mp), but the **target shape** matters when we clip.
2. **Basic**: respect the actual scoring formula in the target.
3. **Disruptive**: Tiny improvement to T100-B that exactly aligns target with metric. Likely small on its own (+0.1) but compounds with B.
4. **Difficulty**: 1 h.
5. **Expected lift**: +0.1 ~ +0.4 standalone; stacks with T100-B.
6. **Risk**: trivial.

---

## Part 3 — Top-2 recommendations

Ranked by **(expected platform lift) × (feasibility) / (risk)** with explicit step-by-step.

### 🥇 Top-1: **T100-A + T100-I — Multi-horizon active + coherence-gated h=60** (compounding)

**Rationale**: We *already trained* the 5-horizon LGB+CB regressors in T98. They sit on disk. The intervention is a Predictor packaging change + threshold tuning — **zero new training**. The multi-horizon active alone is non-decreasing (max rule); the coherence gate on top of h=60 is independent risk reduction; both should compound to ≥ +1.5 LOSO. **Crucially this also prepares us for h=60 distribution shift on platform**: if h=60 happens to underperform on the public test, h=20/40 act as insurance.

**Step-by-step (one engineer, 6–8 h)**:

1. **Hour 0–1**: Load T98's 5-h × 5-seed LGB and CB regressors and the existing T87 NN (which is h=60 only). Verify each model's standalone LOSO-equiv on the 442k test (use T98's `de_thresh_per_h.log` cached numbers as ground truth: h=5 ≈ +18, h=10 ≈ +25, h=20 ≈ +25, h=40 ≈ +28, h=60 ≈ +40).
2. **Hour 1–2**: Build a unified 5-horizon prediction tensor `P[N, 5, M]` where M = number of base models per horizon (e.g., 5 LGB + 5 CB for h<60; 5 LGB + 5 NN for h=60). Compute per-horizon ensemble-mean prediction and per-horizon ensemble-std.
3. **Hour 2–4**: For each horizon h ∈ {5, 10, 20, 40}, run K=5 CV-DE on `(thr_up, thr_dn)` (anti-test-fit, see T100-F co-recommendation). For h=60 keep current `(3.58e-4, 2.16e-4)` as starting point but also run CV-DE.
4. **Hour 4–5**: Define **coherence score**: `C = mean_h sign(pred_h_norm)` where `pred_h_norm = pred_h / |pred_h|.median()` (per-horizon normalization to make signs comparable across horizons). For h=60 trade decision: require `|C| > c_thr` AND `|pred_h60| > thr_h60`. DE-tune `c_thr` ∈ [0, 1] on OOF.
5. **Hour 5–6**: Repackage `iter_016_pkg/Predictor.py`: load 5-horizon model lists; for each horizon emit per-row action (using its own DE-thresh); for h=60 additionally apply coherence gate.
6. **Hour 6–7**: Run full-test verification: per-horizon cum_pnl, per-horizon active count, h=60 active count post-coherence, platform-projected score via T80 transmission. Sanity tests: shuffle-invariance, sym=99 robustness, 1024-batch timing.
7. **Hour 7–8**: Compare iter_016 LOSO ([per_horizon_max] sum_per_sym) vs iter_015 +40.13. If >+1, ship.

**Expected outcome**:
- **Worst case**: max-rule pickup is 0 (all shorter horizons fold to flat under their thresholds), coherence-gate gives modest +0.5 → iter_016 ≈ +40.6
- **Mid case**: max-rule pickup +0.3 (one horizon edges h=60 on 1–2 syms), coherence +1.0 → iter_016 ≈ +41.5
- **Good case**: max-rule pickup +0.8 (h=40 wins on sym=4), coherence +1.5 → iter_016 ≈ +42.5
- **Platform projection** (T80 0.70× transmission, cv-de bonus +2 per T100-F if applied): **+22 ~ +24** vs iter_015 expected ~+22.

**Why this beats Top-2**: zero new training, fully reversible (drop any inactive horizon back to flat), and simultaneously addresses W1 + W5.

### 🥈 Top-2: **T100-B — Net-PnL signed regression target** (compounding with Top-1)

**Rationale**: This is the *next link in the T75 chain*. T75 was "Δmid magnitude preserved instead of 3-class compression" → +9.79. T100-B is "decision-aligned magnitude (PnL) instead of prediction-aligned magnitude (Δmid)" → expected +1–2.5. Different from T57/T92 because intervention is at the **target** layer, not the **gradient-weight** layer. Risk is well-contained: pilot single-seed before committing 5-seed.

**Step-by-step (one engineer, 6–8 h)**:

1. **Hour 0–1**: Pilot. Modify `T75_regression_dmid/train_regr.py` to compute `fee_norm = 1e-4`, redefine `y_regr = sign(Δmid_norm) * np.maximum(np.abs(Δmid_norm) - fee_norm, 0)`. Train single-seed (seed=42) LGB at `objective='huber', alpha=1e-4` (Huber to be robust to bimodal target). Same V4 split, same 359-d features.
2. **Hour 1–2**: Evaluate on full 442k test:
   - test_corr (vs Δmid_norm) — expect drop, this is fine
   - test_corr (vs y_NetPnL) — expect rise vs T75 baseline
   - **Decision rule**: `pred > 0 → 2, pred < 0 → 0, else 1` (NO threshold tuning!) — compare cum_pnl to T75 single-seed baseline +28.28 (seed 42).
3. **Risk gate**: if single-seed cum_pnl < +25 (T75 baseline minus −3.28 buffer), abort and document.
4. **Hour 2–5**: Train remaining 4 seeds (1, 7, 13, 100). Total wall-clock ≈ 4 × 60s = 4 min on GPU; remainder is data shuffling.
5. **Hour 5–6**: Cross-correlation matrix vs T75 LGB (currently 1.000 with itself, 0.87 with T87). Expect moderate decorrelation due to target redefinition (estimate 0.85). Also cross with T87 NN — ensemble diversity is the lever.
6. **Hour 6–7**: 3-way ensemble eval (T75 LGB + T100-B LGB + T87 NN), DE-asymmetric thresholds on a CV-split (apply T100-F if available; otherwise full-test for first cut). Compare to iter_015 +40.13.
7. **Hour 7–8**: If ≥+1.5 LOSO, ship as iter_017 (or fold into iter_016 stack with T100-A).

**Expected outcome**:
- **Worst case**: target reshape disrupts LGB split selection ⇒ pred quality drops > +1 LOSO worse standalone, ensemble diversity flat ⇒ no gain. 0 LOSO.
- **Mid case**: standalone within −1 of T75 (+35 vs +36.23) but adds enough diversity in ensemble for +1.5 LOSO.
- **Good case**: standalone wins (+37+) AND ensembles well (+2 LOSO over iter_015).

**Why this is Top-2 not Top-1**: requires re-training and has medium failure risk (T57/T92 magnitude weight failed → there's institutional skepticism the magnitude lever is exploitable; need to demonstrate target-level is fundamentally different from weight-level).

---

## Appendix — what was *deliberately* not proposed

These are excluded because they are either rule-violations, well-shown-dead, or already in someone's plate:

- **Per-sym normalization / per-sym model** — violates §3.
- **Online retraining at inference** — violates §2.
- **Stateful Predictor (LSTM / hidden carry)** — violates §2.
- **Sym embedding** — violates §3 and r11 negative finding.
- **Time-of-day cyclic features** — T74/T78 confirmed dead under regression.
- **Cross-sym mixup** — T11/T52 failed both regimes.
- **Sample-weight magnitude (T57/T92)** — falsified twice, in CE and regression.
- **Pseudo-labeling under CE** — T65 −0.61.
- **NN architectures (DeepLOB CNN, transformer)** — T93/T94/T95 failed under regression too (T87 SPO+ MLP is current best NN).
- **Time-features** — T78 −0.72 under regression.

These ten exclusions reinforce that the **disruption now must be at the pipeline structure level**, not the feature/arch/loss-flavor level which are saturated.

---

## RESULT

```
RESULT: task=radical_pipeline_rethink top_recommendations=[T100-A=multi-horizon-active+coherence(no-retrain), T100-B=net-pnl-signed-regression-target] notes=5 fundamental weaknesses identified at pipeline level: W1 single-horizon-monomania (4 free leaderboard slots discarded), W2 16-day data hole (date 80-95 unused), W3 last-tick-only raw features (LOB temporal shape thrown away), W4 mean-Δmid target wastes capacity on sub-fee rows, W5 ensemble disagreement averaged away pre-gate. 10 disruptive tricks proposed, top-2 estimated +2.5~+5 LOSO compound (+22~+24 platform). Top-1 requires zero retraining (T98 multi-horizon models already on disk), Top-2 is the next link in the T75 target-shaping chain (target-level intervention different from T57/T92 weight-level failures).
```
