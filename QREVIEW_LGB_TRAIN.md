# QREVIEW — LGB Training Script (`train_T188v2_lgb_seed.py` + `run_all_lgb_seeds.sh`)

Whiteboard review. No README/PROGRESS context. Pure "why?" questions on every design point.

Files reviewed:
- `final_submission_code/02_train_lgb/train_T188v2_lgb_seed.py` (209 lines)
- `final_submission_code/02_train_lgb/run_all_lgb_seeds.sh` (32 lines)

---

## A. HP grid / cycling design

### Q1 — Why exactly 5 HP configs?
Lines 25–31: `HP_CONFIGS` has 5 dicts. Why 5 and not 3, 8, 10? Did you sweep more and prune, or pick 5 a priori? What's the basis for "5 is enough diversity for an ensemble"?

### Q2 — Why (seed-1)%5 cycling instead of random assignment?
Line 83: `hp = HP_CONFIGS[(S - 1) % len(HP_CONFIGS)]`. Why deterministic cycling? Why not random pairing of seed↔config so the ensemble doesn't have correlated (seed, hp) pairs? Why exactly 10 seeds per config (50/5)?

### Q3 — `feature_fraction` values 0.8 / 0.6 / 0.7 / 0.5 / 0.4 — why these?
Lines 26–30. The spread is 0.4–0.8 with no obvious pattern (not a uniform grid, not halving). How were these chosen? Any prior CV showing 0.4 is useful?

### Q4 — `bagging_fraction` values 0.8 / 0.7 / 0.85 / 0.6 / 0.5 — why these?
Same lines. Also not a clean grid. Why was config 3 the only one with 0.85 (higher than baseline 0.8)?

### Q5 — `num_leaves` values 127 / 127 / 63 / 255 / 127 — why this distribution?
Three of five configs use 127. Why does only one config explore 63 (shallower) and one 255 (deeper)? Is 127 a known sweet spot from earlier sweeps?

### Q6 — `lambda_l2` values 1.0 / 1.0 / 2.0 / 0.5 / 3.0 — why?
Why is the wide-tree config (num_leaves=255) given *less* L2 (0.5) and the narrow-tree config (num_leaves=63) given *more* L2 (2.0)? Intuition says the opposite.

### Q7 — Why no HP variation in `learning_rate`?
Line 73: LR is a CLI arg defaulting to 0.05 and never varied across configs. LR diversity is one of the highest-value ensemble axes — why fix it?

### Q8 — Why no HP variation in `num_boost_round`?
Line 72: also fixed at 330. With wildly different `num_leaves` (63 vs 255), effective capacity is very different; same boost rounds means very different convergence states.

### Q9 — Why no HP variation in `min_data_in_leaf`?
Line 74: fixed at 100. With `num_leaves=255` and ~3M rows, leaves can be much smaller than with 63 — `min_data_in_leaf` should arguably scale with tree complexity.

### Q10 — Why no HP variation in `bagging_freq`?
Fixed at 5 for all configs.

### Q11 — Why no `lambda_l1` anywhere?
Only `lambda_l2` is tuned. L1 is omitted entirely from both defaults and HP grid. Was L1 tested and dropped, or never considered?

### Q12 — Why no `min_gain_to_split` set?
Defaulted to 0 (LightGBM default). With L2=0.5 and 255 leaves you may want a positive gain floor.

### Q13 — Why no `max_depth` cap?
Only `num_leaves` controls tree shape. With 255 leaves the implicit depth ≈ log2(255) ≈ 8 but unbalanced trees could go much deeper. Intentional?

### Q14 — Why `max_bin` left at default (255)?
On 226 features with float32 values, 255 bins may or may not be enough. Did you test 511 / 1023?

### Q15 — Why no `min_data_in_bin`, `min_sum_hessian_in_leaf`, `cat_smooth`, etc.?
All defaulted. With class-balanced weights, `min_sum_hessian_in_leaf` (default 1e-3) interacts with weights — was this checked?

---

## B. Objective / metric

### Q16 — Why `objective="regression_l2"` and not L1/Huber/Quantile?
Line 153. L2 is sensitive to tail returns. Were L1 / Huber / quantile tested? Financial returns are heavy-tailed.

### Q17 — Why `metric="rmse"`?
Line 154. Identical to objective. No tracking of MAE, correlation, or task-aligned ranking metric. RMSE on weighted regression is hard to interpret.

### Q18 — Why train regression at all when `y_cls` is loaded?
Lines 123, 135: `y_cls_tr` is loaded and used to compute weights but never as a target. Why not train multiclass directly? If the downstream uses class probabilities, the regression head is doing a different job.

### Q19 — Why `NUM_CLASS = 3` but training a regression?
Line 22, 47–51. The constant only flows into weight computation. Why hardcode 3 here if it never reaches the objective?

### Q20 — Why is `FEE = 0.0001` declared but unused?
Line 23. Defined globally, referenced nowhere in this file. Dead code or leaked from a sibling?

### Q21 — Why no custom objective / fobj that aligns with the eval metric?
A trading task usually scores via PnL/Sharpe/IC. L2 regression on a return ratio is a proxy — was the proxy gap measured?

---

## C. Sample weights (class-balanced weights on a regression task)

### Q22 — Why class-balance weights on a *regression* objective?
Lines 47–51, 148. Computing CB weights from a 3-class discretization and applying them to L2 regression is unusual. The gradient on L2 is `2w·(ŷ−y)` — does up-weighting tails actually shift the learned regressor in the intended direction?

### Q23 — What discretization produced `y_cls`? Not visible in this file.
The 3 classes are computed upstream (in cache). What thresholds? Are they symmetric? Are they horizon-specific?

### Q24 — Why `len(y) / (num_class * counts)` (inverse-freq scaled)?
Line 50. Why this exact formula vs simple inverse-freq, or median-balanced, or sqrt-balanced?

### Q25 — Class counts include augmented rows (line 148 runs after concat). Does that change the weight distribution?
Augmented samples have identical labels to originals, so class distribution is preserved — but the magnitude `len(y)` doubles. Effect on regularization (since hessian sums change)?

### Q26 — Weights cast to float32 (line 51). Precision concerns?
LightGBM hessian sums use these — small rare-class weights × large counts can lose precision.

### Q27 — Empty-class guard `counts = np.where(counts == 0, 1.0, counts)` (line 49) — why silent fix instead of assert?
If a class is empty, that's likely a data bug. Silently replacing avoids crashing but masks it.

---

## D. Augmentation

### Q28 — Why `AUG_LO=0.80, AUG_HI=1.20` (±20% per-feature uniform scaling)?
Line 44, 54. Why exactly ±20%? Why uniform instead of lognormal (multiplicative noise is typically lognormal)?

### Q29 — Why scale every feature *independently per row*?
Line 59: `rng.uniform(lo, hi, size=(sz, X_src.shape[1]))`. This destroys cross-feature correlations. For orderbook / momentum features computed from related signals, that's destructive.

### Q30 — Why is the augmentation 1× (single duplicate copy) and not 2×, 5×?
Line 117: `n_train_used = n_total * 2`. Why exactly one augmented copy? Did you sweep multiplicity?

### Q31 — Why does the augmentation leave `y_regr` unchanged?
Lines 144–145. If `y_regr = (mp_th − mp_t) / (mp_t + 1.0)` is a function of price levels, and you scale features (some of which may be price-related) by random factors, the *correct* label may also change. Has the y/X consistency been checked?

### Q32 — Some features are dimensionless (e.g. spreads-in-ticks, ranks). Scaling these by 0.8–1.2 is non-physical. Was per-feature aug applicability considered?
A 0.8× on a rank feature is meaningless. All features are scaled the same way.

### Q33 — `CHUNK = 200_000` — why this magic number?
Line 43. Memory? Cache? Set empirically?

### Q34 — `del scales` after the loop (line 61) — but `scales` only exists inside the loop scope after the last iteration. Why this defensive cleanup?
It hints that augmentation has been a memory pain point. Why not preallocate `scales` once?

### Q35 — Augmented copy placed in *second half* of `X_tr_full`. With LightGBM bagging (random row sampling) is this fine, but is row order otherwise relied upon? No shuffle is applied before training.

### Q36 — Same RNG `np.random.default_rng(S * 7919 + 42)` for augmentation (line 142). Why magic numbers 7919 (prime) and 42?

### Q37 — Why is `aug_a_scale_chunked` a separate function with a chunked loop instead of a single broadcast multiply?
Line 54–61. `X_src * rng.uniform(lo, hi, X_src.shape).astype(np.float32)` does it in one shot, with peak memory of ~X_src size. Chunking saves peak memory but adds Python overhead — was the tradeoff measured?

---

## E. Validation / training-loop config

### Q38 — `valid_sets=[dtrain]` — using *train* as validation (line 181)
There's no held-out val set passed to `lgb.train`. The "metric" prints will be training error only.

### Q39 — No `early_stopping_rounds` / `early_stopping` callback. Why?
Without early stopping you can't tell if 330 rounds over/underfits. Was 330 chosen by an earlier sweep?

### Q40 — Why `num_boost_round=330` (not 300, 500)?
Line 72. Suspiciously specific. What was the basis?

### Q41 — `log_evaluation(period=50)` — why every 50?
Line 182. With 330 rounds you get only ~6 log lines per training run. Hard to diagnose anything.

### Q42 — No `record_evaluation` callback — training curve is thrown away.
You lose the per-round loss trajectory after training finishes. No way to retrospectively pick optimal iteration.

### Q43 — Why concatenate train + val + test all into training data (line 127 loop)?
"All dates 0-119" (file docstring) suggests intentional full-data training, but then you have *zero held-out data* in this script. How was the choice of 330 rounds validated externally?

### Q44 — `lgb.train` doesn't get `init_model` — every run starts from scratch.
Fine for an ensemble of independent learners, but for resumability or continuation training there's no hook.

### Q45 — No `feval` / custom eval. Why?
Only built-in RMSE. A trading-aligned eval (e.g. Spearman rank corr) per round would be more informative than RMSE.

---

## F. Data loading / memory

### Q46 — `np.empty((n_train_used, feat_dim), dtype=np.float32)` (line 121) — pre-allocation strategy
With float32 × ~3M × ~226 = ~2.7 GB. What if the host has <8GB free? Any OOM guard?

### Q47 — `d["X"][:, keep_idx]` (line 131) — fancy indexing creates a copy and a transient peak
For a moment you hold (full X, X[:,keep_idx], X_tr_full slice). On larger splits this peaks. Why not column-filter on disk write?

### Q48 — `.astype(np.float32)` on line 131 — what is the cache stored as?
If already float32 this is a no-op + copy. If float64, why not store cache as float32 directly?

### Q49 — `np.load(...)` without `mmap_mode='r'`. Full materialization.
On big splits this is wasteful — you only need it once for one big copy. Why not mmap?

### Q50 — `d.close()` on NpzFile — relies on garbage collection of internals; you also `gc.collect()` repeatedly (lines 124, 138, 146, 175, 186). Why so much defensive GC? Hints at a leak.

### Q51 — `mp_t` and `mp_th` cast to `float64` (lines 132–133), then the ratio is cast back to `float32` (line 134). Why upcast intermediate?
If precision is needed, why discard it on the final cast?

### Q52 — `(mp_th - mp_t) / (mp_t + 1.0)` — what is `+ 1.0` for?
Line 134. A stabilization to avoid division by zero? But `mp_t + 1` shifts the denominator non-trivially when mp_t is small. Why this specific constant and not e.g. `+ 1e-9`?

### Q53 — `y_regr` not winsorized / clipped.
Line 134. Large outlier returns under L2 dominate gradient. Why not clip to a sane percentile range?

### Q54 — No NaN / Inf check on X or y before training.
LightGBM tolerates NaN in features (treats as missing) but Inf in y will silently break training. Was data integrity verified upstream?

### Q55 — `y_regr stats: mean / std` printed (line 150) but no skew/kurtosis or NaN count.
You log mean/std but no sanity bounds (e.g. `assert np.isfinite(y_regr).all()`).

### Q56 — Feature names file has no schema validation — assumes one name per line, no header.
Line 96–97. What if the file has trailing blank line or BOM?

### Q57 — `DROP_NAMES` is a hardcoded list of "failed" features. Provenance?
Lines 33–41. Two groups: `T59_FAIL_NAMES` and `STAGE5_FAIL_NAMES`. Why two groups? Who decided "fail"? What was the criterion?

### Q58 — `drop_idx = set(name_to_idx[n] for n in DROP_NAMES if n in name_to_idx)` — silently skips missing names.
Line 100. If a `DROP_NAMES` entry isn't in the feature list (typo / renamed), it's dropped from drop set silently. No warning.

### Q59 — `forbidden = {"date", "sym", "time"}` (line 104) — only assert *after* DROP_NAMES filter.
If `date`/`sym`/`time` are not in `DROP_NAMES`, they would survive into features and only then be caught by assert. Why not validate before any filtering?

### Q60 — `keep_idx` is `int64`. Are columns guaranteed to be in the same order as `all_feat_names`?
Line 101. Implicit assumption: cache columns match the names file index-for-index. Any cross-check?

---

## G. LightGBM Dataset construction

### Q61 — `free_raw_data=True` and *also* `del X_tr_full` (lines 172–174). Belt-and-suspenders or risk?
`free_raw_data=True` frees on `construct()`; explicit del also drops the Python reference. With `valid_sets=[dtrain]` the dataset will be constructed for validation too — does free + del actually release before `lgb.train`?

### Q62 — `feature_name=feat_names` but no `categorical_feature` argument.
Line 172. Are there categorical features in the data? If `sym` were a feature this would matter. (`sym` is in forbidden set so probably not, but worth confirming none of the kept 226 features are categorical.)

### Q63 — No `init_score` passed. Why no warm-start from a baseline (e.g. mean)?
For heavy-tailed targets, initializing with the global mean of y helps. LightGBM's default boost-from-average is implicit.

### Q64 — No `boost_from_average` explicitly set. Default is True for regression — verified?

### Q65 — Dataset constructed with weight=sw but no `init_score`. Combined with class-balanced weights, does L2 regression converge to weighted mean? Has the bias term been inspected?

---

## H. Seeds / reproducibility

### Q66 — Only `params["seed"]=S` set (line 163). What about `feature_fraction_seed`, `bagging_seed`, `data_random_seed`, `objective_seed`?
LightGBM derives these from `seed` if unset, but explicit control matters for true reproducibility. Were they intentionally left to derive?

### Q67 — `deterministic` not set. With `force_col_wise/force_row_wise` also unset and 18 threads, results are not bit-reproducible. Acceptable?

### Q68 — RNG for augmentation uses `S * 7919 + 42`; LightGBM uses just `S`. Why decoupled?
Augmentation RNG and LightGBM RNG seeded differently — is the offset (`*7919 + 42`) intentional decorrelation or just superstition?

### Q69 — With 50 seeds (1–50) and HP cycle of 5, seeds {1,6,11,…,46} all get config 0. Their LightGBM `seed=1,6,11,…` differs but augmentation RNG differs by `*7919+42`. Is the per-config diversity actually large enough?

---

## I. Argparse / CLI design

### Q70 — `--num-boost-round`, `--learning-rate`, `--min-data-in-leaf`, `--bagging-freq`, `--num-threads`, `--gpu` are CLI args but the shell script never passes them.
Lines 72–78 vs `run_all_lgb_seeds.sh` line 22–26. CLI surface implies tunability but the actual invocation hardcodes defaults. Dead surface area or deliberate?

### Q71 — `--no-wandb` is declared (line 77) but `wandb` is *never* imported or used in the file.
Why declare a wandb flag if no wandb code? Was wandb intended? The user's project rules say "all experiments must use wandb."

### Q72 — `--horizon` default is 60 (line 67). What other horizons exist? Why is "T188v2" tied to 60?

### Q73 — `--num-threads` default 18. Why 18, not 16 or `os.cpu_count()`? Was this tuned for a specific host?

### Q74 — No `--quiet` / log-level flag, but every print uses `flush=True`. Inconsistent — pick one logging approach.

### Q75 — `os.makedirs(out_dir, exist_ok=True)` but no `cache_dir` existence check.
Line 84. Fails late at `np.load` if cache is wrong.

### Q76 — `--seed` help says "1–50" (line 66) but no `choices` constraint or range check.
You can pass `--seed 99999` and it'll wrap via `(S-1) % 5` for HP selection. Quietly works but probably unintended.

---

## J. Output / model save

### Q77 — `booster.save_model(out_path)` saves text format (line 188). Why text and not binary?
Text is larger and slower to load. The CLAUDE.md says inference uses `.txt`, but why? Was binary tested?

### Q78 — No `num_iteration` passed to `save_model`. Saves full model.
Without early stopping there's no "best iteration" anyway, but it's worth verifying you save all 330 trees.

### Q79 — Summary JSON (lines 191–202) doesn't record final training loss.
Why save HP and timing but not the actual final RMSE? You're throwing away ground truth.

### Q80 — Summary JSON doesn't record `feat_dim`, `n_train_used`, `n_total`, augmentation params, or feature-drop list.
For reproducibility, you'd want every binding to be reproducible from the JSON alone.

### Q81 — Summary JSON doesn't include LightGBM version, NumPy version, GPU vs CPU, host info.
Hard to debug differences later.

### Q82 — `out_path` naming `model_h{H}_seed{S}.txt`. No HP-group in the name.
You can recover HP from the seed via modulo, but if the cycle order ever changes the names break silently.

### Q83 — `SKIP` check on existing file (lines 91–93) — no integrity / size / hash check.
A half-written model file from a crash would be skipped as "already exists."

---

## K. Shell script (`run_all_lgb_seeds.sh`)

### Q84 — `set -e` (line 5) — any python failure aborts the whole loop.
On a 50-seed run, a single seed failure stops all subsequent ones. Why not `set +e` per-seed with explicit error logging?

### Q85 — Loop is strictly sequential (line 20).
With independent seeds and potentially multi-GPU hosts, why no parallelism (`xargs -P`, GNU parallel, background `&` + `wait`)?

### Q86 — `--num-threads 18` would mean 50 sequential trainings × ~minutes each = many hours.
Was wall-clock measured? Why not parallelize across CPU cores or multiple hosts?

### Q87 — No retry on failure.
If seed 27 OOMs transiently, the whole run dies and the user must resume manually.

### Q88 — No per-seed log file.
All output goes to stdout/stderr of the calling shell. With 50 seeds this is hard to grep through.

### Q89 — End-of-script `ls | wc -l` (line 31) just prints a number; no assertion `== 50`.
A partial run shows e.g. `42` and the script exits 0.

### Q90 — `GPU_FLAG=${3:-""}` — positional CLI is fragile.
Why not `bash run_all.sh --cache=... --out=... --gpu`?

### Q91 — `--gpu` flag is passed verbatim to *every* seed. No `CUDA_VISIBLE_DEVICES` pinning.
If the host has 4 GPUs, all 50 seeds will land on GPU 0. Was multi-GPU dispatch considered?

### Q92 — `$GPU_FLAG` unquoted (line 26). If `GPU_FLAG` contained spaces it would split.
Currently safe (only `--gpu`) but brittle.

### Q93 — Hardcoded `seq 1 50` (line 20). No way to run a subset for debugging.
`bash run_all.sh ... 1 10` to run seeds 1–10 would be nice.

### Q94 — `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"` is portable, but doesn't protect against being sourced.

### Q95 — No venv / conda activation. Assumes `python3` on PATH has lightgbm installed.

### Q96 — No check that `CACHE_DIR` actually contains `schemeP_*.npz` and `schemeP_feat_names.txt`.

### Q97 — No write-permission check on `OUT_DIR` before starting.
A long run can fail at save time.

### Q98 — No `time` / wall-clock reporting per seed at the shell level.

### Q99 — `echo` headers but no machine-readable progress (JSON / TSV).

### Q100 — Shell uses `bash` features (set -e behavior, `${var:-default}`) but shebang is `#!/bin/bash` — fine, but the run example in the comment uses `bash run_all_lgb_seeds.sh ...` which would work even without exec bit.

---

## L. Higher-level / cross-cutting

### Q101 — "T188v2" name suggests a v1. What changed?
Filename: `train_T188v2_lgb_seed.py`. The "v2" implies prior iteration but nothing in this script documents the delta.

### Q102 — Why is this called "schemeP" in the cache (lines 95, 112, 129)?
The naming `schemeP_train.npz` is opaque. Where does P come from?

### Q103 — The training file imports `datetime, timezone` (line 17) but uses neither.
Dead imports.

### Q104 — `sys` imported (line 15), unused.

### Q105 — The model is a single regression head per seed; downstream ensembling assumes simple averaging of point predictions. Was that assumption tested against e.g. median or rank-averaging?

### Q106 — Same `y_regr` used regardless of `hp_group`. Why no target-side diversity (e.g. some configs predict {h=20, h=60, h=120})?

### Q107 — The CLAUDE.md says "LightGBM training default GPU=on." This script defaults to CPU (`args.gpu` defaults False). Conflict?

### Q108 — Class-balanced weights are computed *after* augmentation (line 148). Since augmentation duplicates labels, weights are unaffected in relative terms but `len(y)` doubles. Was the weight magnitude verified against the non-augmented case?

### Q109 — `bagging_fraction` ranges down to 0.5 (config 4). With `bagging_freq=5` and 330 rounds you get ~66 bagging draws. With 0.5 sampling, variance in tree training is high — was this stability checked across seeds within the same HP group?

### Q110 — No deterministic check or unit test that two runs with same seed produce identical models.
Reproducibility is asserted by construction (single seed param) but not verified.

---

## M. Critical-constraint angle (per CRITICAL_CONSTRAINTS.md)

### Q111 — Forbidden set is `{"date","sym","time"}`. Are *all* sym-related features (sym embedding, sym-specific normalization residuals) absent from `feat_names`? The assert only covers literal names.

### Q112 — `class_balanced_weight` discretizes y_cls per-row but doesn't condition on `sym`. Fine for a sym-agnostic model. But the *augmentation* multiplies features per-row independently — could that introduce a sym-correlated bias if some features are sym-specific?

### Q113 — Training data spans all 120 dates ("all dates 0-119"). If `date` is in the X matrix at all (even ignored by `DROP_NAMES`), models could learn date drift. The forbidden-set assert only triggers if `date` is a literal feature name in the kept list — what guarantees it's not buried under another name (e.g. `time_of_day_norm`)?

---

## N. Style / minor

### Q114 — Mix of `int(hp["num_leaves"])` and `float(hp["feature_fraction"])` (lines 156–160). Cast is defensive — what is `HP_CONFIGS` typed as that necessitates cast at use site?

### Q115 — `verbose=-1` (line 164) silences LightGBM but you also use `log_evaluation(period=50)` (line 182). Why mix silencing with explicit logging callback?

### Q116 — `bagging_fraction` printed via `hp` dict only — no echo of effective LightGBM `params` dict for audit.

### Q117 — No `assert n_total > 0` after loading sizes.

### Q118 — `print(f"  ...")` indent style — two-space indent in log lines, no structured logger.

### Q119 — Magic constant `2` in `n_train_used = n_total * 2` (line 117) — should be derived from "augmentation multiplicity = 1, so 1 original + 1 aug = 2x." Worth a named constant.

### Q120 — `for split in ["train","val","test"]` repeated twice (lines 111, 127). Could be DRYed up — minor but the duplication risks divergence.

---

## Summary

Total: **120 questions** across HP grid (Q1–Q15), objective (Q16–Q21), weights (Q22–Q27), augmentation (Q28–Q37), training loop (Q38–Q45), data IO (Q46–Q60), Dataset construction (Q61–Q65), seeds/reproducibility (Q66–Q69), CLI (Q70–Q76), output (Q77–Q83), shell script (Q84–Q100), cross-cutting (Q101–Q110), constraint compliance (Q111–Q113), style (Q114–Q120).

Highest-priority "wat?" items in my view (not for answering, just flagging):
- **Q18/Q22**: training L2 regression with class-balanced weights from a discretization of the same target — unusual setup.
- **Q31/Q32**: per-feature uniform ±20% scaling without adjusting y — assumes y is augmentation-invariant, which is non-obvious for return-ratio targets and price-dimensioned features.
- **Q38/Q43**: no held-out validation in the training script — `valid_sets=[dtrain]` is training error.
- **Q43**: train+val+test all concatenated into training data — implies upstream did model selection, but where?
- **Q52**: `mp_t + 1.0` denominator stabilization — arbitrary constant.
- **Q70/Q71**: declared CLI flags (LR, boost rounds, wandb) that are never wired up.
- **Q84/Q85/Q91**: shell is strictly sequential, no GPU pinning, fails-fast on first error.
