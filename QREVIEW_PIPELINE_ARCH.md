# QREVIEW: Pipeline Architecture — final_submission_code/

Reviewer perspective: senior systems / architecture reviewer, white-board view.
Scope: end-to-end pipeline organization, data contracts, packaging, reproducibility, ops, failure modes. **Not** model performance, not magic-number choices.

Questions are clustered by topic. ~90 questions total.

---

## A. Overall pipeline shape (01 → 02 → 03 → 04)

### Q1 — Stage granularity: why 4 stages?
**Location**: `run_pipeline.sh`, top-level directory layout.
**Issue**: The split is feature → LGB → NN → packaging. Why is feature build a single monolithic stage rather than per-split (train/val/test)? Why are LGB and NN sibling stages rather than independent pipelines? Could 02 and 03 be merged under a "models" stage (a single orchestrator that fans out), so adding/removing a model family does not require a new top-level directory?

### Q2 — Are stages independently re-runnable?
**Location**: `run_pipeline.sh:38-75` — sequential `python3` calls with no per-stage outputs/inputs declared.
**Issue**: If Step 2 partly completes (e.g., seeds 1-30 succeed, seed 31 OOMs), can the user resume from seed 31 without rebuilding the cache? `train_T188v2_lgb_seed.py:91` has a `if os.path.isfile(out_path): SKIP`, but `run_all_lgb_seeds.sh` does not surface this granularity in `run_pipeline.sh` — does a single `set -e` failure simply re-trigger Step 1 too?

### Q3 — Numeric prefixes encode strict ordering — what if reality is a DAG?
**Location**: `01_build_features/`, `02_train_lgb/`, `03_train_nn/`, `04_build_pkg/`.
**Issue**: LGB and NN are *independent* given the cache, but the numbering implies they run serially. The current `run_pipeline.sh` does run them serially (52 → 62). Why not parallelize? Is there a resource conflict (RAM contention reading the same cache files)?

### Q4 — No pipeline orchestrator
**Issue**: There is no Make / Snakemake / Airflow / Nextflow. The pipeline is a single bash file. How do you express "rerun only what's stale" semantics (e.g., cache changed → invalidate models)? Today, if `schemeP_train.npz` is regenerated with a different feature implementation, old `model_h60_seed*.txt` files remain valid by filename only.

### Q5 — What is the data contract between 01 and {02,03}?
**Location**: `build_schemeP_cache.py:183-194` writes a dict to npz; `train_T188v2_lgb_seed.py:127-137` and `train_T188v2_nn_seed.py:215-219` read it.
**Issue**: The contract is implicit (string keys `X`, `mp_t`, `y{H}`, `mp_t{H}`, etc.). Is there a schema validation step? A version stamp? What if a future build of 01 changes a key name — would 02/03 silently fall back or crash with a `KeyError`?

### Q6 — schemeP_feat_names.txt is a critical contract — where is its integrity guarded?
**Location**: `build_schemeP_cache.py:215-220` writes it; `train_T188v2_lgb_seed.py:95-103` and `train_T188v2_nn_seed.py:198-204` read it (text file, line-per-feature).
**Issue**: Plain-text, no checksum, no version. If a user edits it manually (or re-runs 01 with different feature set), there is no guarantee the model dimensions still match. Is the feature order stable across re-runs? Stage5 ordering inside `compute_batch_features` — what enforces stability?

### Q7 — The cache npz is not versioned / fingerprinted
**Location**: `build_schemeP_cache.py:232-234`.
**Issue**: No git SHA, code version, hash of `fast_features_batch.py`, or schema version is embedded in the npz. If you upgrade `fast_features_batch.py` and forget to clear cache, models trained against old features but using new package code will silently misalign at inference.

### Q8 — How does the data contract between 03 (training) and 04 (Predictor) stay aligned?
**Location**: `train_T188v2_nn_seed.py:130-159` (`extract_npz`) and `Predictor.py:78-178` (`_BatchedMLPEnsemble`).
**Issue**: `extract_npz` writes keys `L0_W`, `LN0_W`, `LF_W`, etc. The Predictor parses them by string match. No version field, no architecture descriptor (apart from `hidden` array). If you add a new layer in training, you must remember to update the Predictor; no test catches it.

### Q9 — `04_build_pkg/Predictor.py` is committed as the source of truth, but it has no link to the training code
**Location**: `04_build_pkg/Predictor.py:78` (`_BatchedMLPEnsemble`) re-implements the MLP forward pass manually with `torch.bmm`. The training code uses `nn.Sequential` (`train_T188v2_nn_seed.py:80-104`).
**Issue**: Two independent implementations of the same MLP. What enforces equivalence? Is there a numerical-parity unit test (e.g., load one .pt, run through PyTorch nn.Module, compare to `_BatchedMLPEnsemble` on same input)? Comment claims `~1e-6 typical, 1e-4 max` — where is the test that verifies this?

### Q10 — Pipeline checkpoint / resume protocol is per-script only
**Issue**: Each train script (`train_T188v2_lgb_seed.py:91` and the NN equivalent `--skip-phase1`) has its own ad-hoc skip logic. There is no top-level `pipeline_state.json` that says "step 1 done at 12:00, step 2 done at 13:00". No way to ask "are we at a known-good state?" without re-running.

### Q11 — Bash `set -e` only; no logging discipline
**Location**: `run_pipeline.sh:9` uses `set -e`. No `set -u`, no `set -o pipefail`, no `trap` for cleanup.
**Issue**: `set -u` would catch typos like `$LBG_GPU_FLAG`. `pipefail` would catch silent upstream failures in `cd ... && zip ...`. No `trap ERR` to print which line failed.

### Q12 — No log file
**Issue**: `run_pipeline.sh` prints to stdout. A user running `./run_pipeline.sh > log.txt 2>&1` could capture it, but the script itself does not enforce a log location. Per-seed `train.log` files are not written either (despite CLAUDE.md mandating it).

### Q13 — No monitoring / progress beyond `print(..., flush=True)`
**Issue**: For a 90-minute single-GPU run, the user only sees stdout. No JSON progress file (`worker-progress.json` is mandated in CLAUDE.md but absent). How does an operator monitor liveness?

### Q14 — What if a stage takes longer than expected?
**Issue**: No `timeout`, no kill switch. If Step 1 reads a corrupt parquet that triggers a pandas hang, the entire pipeline silently hangs.

---

## B. Train ↔ submit isolation (CUDA vs CPU torch, etc.)

### Q15 — Two requirements environments, only one is documented
**Location**: `README.md:34-41` mentions `pip install numpy pandas lightgbm scipy torch` for training and `pip install -r 04_build_pkg/requirements.txt` for submission. `04_build_pkg/requirements.txt` pins CPU-only torch.
**Issue**: There is no `requirements-train.txt` checked in. The training environment is implicit / unpinned. If a user installs `lightgbm==3.x` from PyPI but the submission expects 4.x model format, models won't load at inference.

### Q16 — How are torch versions kept consistent between training & inference?
**Location**: `04_build_pkg/requirements.txt:6` pins `torch==2.5.1+cpu`. Training script `train_T188v2_nn_seed.py` uses whatever torch is installed.
**Issue**: state_dict format between torch versions is usually stable, but `weights_only=False` (line 231) is becoming default `True` in newer torch versions. Cross-version compatibility is not documented.

### Q17 — The Predictor loads with `torch.cuda.is_available()` — what does the platform sandbox have?
**Location**: `Predictor.py:193`.
**Issue**: README claims "CPU-only torch" but the Predictor still calls `torch.cuda.is_available()`. With CPU-only torch this returns False, so it works — but if a user accidentally bundles a CUDA torch wheel, the Predictor would try CUDA on a sandbox without GPU and fail.

### Q18 — Is there a check that the package wheels actually install in the platform image?
**Issue**: `requirements.txt` uses `--extra-index-url https://download.pytorch.org/whl/cpu`. If the platform's sandbox has no outbound network, this fails. Is this assumption documented? Was it ever tested in an offline `pip install` against the official platform image?

### Q19 — `numpy==2.4.4` is pinned
**Location**: `04_build_pkg/requirements.txt:2`.
**Issue**: numpy 2.x is a major version bump from 1.x. Will the LightGBM 4.6.0 wheel link correctly? Are the saved LGB model.txt files numpy-version-portable? Most are — but this is a fragile assumption.

### Q20 — Training artifacts (.pt files) are not in the submission package, but they live in `outputs/models/` next to the .npz files
**Location**: `train_T188v2_nn_seed.py:502` saves `nn_h60_seed{S}.pt`; `extract_npz` (line 505) then writes `.npz`. `build_pkg.py:77-83` copies only `.npz`.
**Issue**: Good — but the `.pt` files persist in `outputs/models/`. If someone manually zips that directory, they'll get 2× the model files. `.gitignore:9-10` lists `*.pt` and `*.zip` (good), but the package builder does not actively exclude `.pt` from `outputs/pkg/`. Defense in depth?

### Q21 — `T81_pretrained/anchor_seed{S}.pt` lives inside `outputs/models/`
**Location**: `train_T188v2_nn_seed.py:180` makes `T81_pretrained/` inside `--out-dir`.
**Issue**: This mingles "intermediate pretraining checkpoints" with "final inference weights". If a user does `cp -r outputs/models /pkg/`, they ship pretraining anchors too. Why not separate `outputs/pretrain/` and `outputs/final_models/`?

### Q22 — `build_pkg.py` does NOT validate that the .npz files are loadable
**Location**: `build_pkg.py:79-93`.
**Issue**: It just `shutil.copy2`. If an .npz is corrupt (truncated write), the package builder will happily ship it. Should at least try `np.load(p)` per file.

### Q23 — Architecture/OS assumptions
**Issue**: `requirements.txt` pins manylinux-style wheels (implicit). What if the platform runs on aarch64? On Alpine (musl libc)? On Windows? No documentation of platform OS / arch.

### Q24 — Python version drift
**Location**: `README.md:25` says "Python 3.10+ (推荐 3.11)". `config.json:2` declares `python_version: "3.11"`. Training requirements unpinned.
**Issue**: numpy 2.4.4 requires Python 3.10+. If platform runs Python 3.9, the package install fails. Is the platform Python version contractually known? Where is it asserted?

### Q25 — `manifest.json` is written but never read
**Location**: `build_pkg.py:107-116`.
**Issue**: Good documentation, but if the user re-runs the pipeline 3 times, no version embeds. Should include git SHA, build timestamp, file checksums.

### Q26 — `md5_file()` is defined but never called
**Location**: `build_pkg.py:27-32`.
**Issue**: Dead code, OR forgotten checksum recording. Either delete it or use it (compute md5 of each model file in the manifest).

---

## C. Reproducibility

### Q27 — Two runs of the same seed: bit-identical?
**Location**: `train_T188v2_nn_seed.py:194-196` sets `torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`. No `torch.backends.cudnn.deterministic = True`. No `torch.use_deterministic_algorithms(True)`. No `os.environ['CUBLAS_WORKSPACE_CONFIG']`.
**Issue**: Two runs on the same GPU will likely differ in low-order bits, possibly more for SPO+ DFL with gradients near hinge. Has anyone verified this experimentally? Is bit-reproducibility a requirement?

### Q28 — LightGBM determinism
**Location**: `train_T188v2_lgb_seed.py:152-169` sets `params['seed'] = S`. `params['num_threads'] = 18`.
**Issue**: LightGBM with `bagging_freq>0` is reproducible only if `num_threads=1` or with `deterministic=true` param. Not set. If a user runs with different `--num-threads`, model files differ. Why is this not flagged?

### Q29 — GPU LightGBM ≠ CPU LightGBM (different model bytes)
**Location**: `train_T188v2_lgb_seed.py:166-168` adds `device='gpu'`. `run_pipeline.sh:13` defaults to `--gpu`.
**Issue**: README claims "(matches original training; pass '' to use CPU LightGBM)" implying the original was GPU. If a downstream consumer rebuilds without GPU, models differ. This is a hard-to-reproduce gotcha and the documentation does not call it out beyond a parenthetical.

### Q30 — Data shuffle order
**Location**: `train_T188v2_lgb_seed.py:127-145` concatenates `train + val + test` in fixed order. `train_T188v2_nn_seed.py:390-394` same.
**Issue**: LGB bagging uses `seed` but data row order matters for `bagging_fraction`. NN training uses `torch.randperm(n_train, generator?)` — no explicit generator (line 306, 446), so it uses global state. Are these tracked deterministically?

### Q31 — NumPy global generator state vs explicit
**Location**: `train_T188v2_nn_seed.py:196` does `np.random.seed(S)`, but augmentation uses `np.random.default_rng(S * 7919 + 137)` (line 406).
**Issue**: Two coexisting RNG schemes (global + explicit Generator). If the augmentation code path changes RNG type, the seed stream changes. Why not exclusively use explicit Generators?

### Q32 — `aug_a_scale_chunked` consumes RNG in chunks of CHUNK=200_000
**Location**: `train_T188v2_lgb_seed.py:43-61`.
**Issue**: The output sequence depends on `CHUNK`. If a future revisit changes CHUNK to 100K (e.g., RAM concerns), the augmentation differs and the model differs. CHUNK should be a documented seed-sensitive constant.

### Q33 — Phase 1 early-stopping epoch depends on val_mse trajectory which can be noisy
**Location**: `train_T188v2_nn_seed.py:331-340`.
**Issue**: A 1e-10 tolerance and 10-epoch patience means a small float perturbation can shift best_epoch by 1-2 epochs. The downstream Phase 2 starts from this checkpoint. Two re-runs with bit-equivalent setup *might* still produce different Phase 1 anchor checkpoints. Has this been measured?

### Q34 — `--skip-phase1` semantics: trusts whatever anchor is on disk
**Location**: `train_T188v2_nn_seed.py:229-237`, `run_all_nn_seeds.sh:28` always passes `--skip-phase1`.
**Issue**: If anchor was trained with a different feature_dim / dropout / lr, Phase 2 loads it silently and the resulting model is invalid. There is *no* compatibility check between anchor and current run args. The pipeline depends on the user not re-running pieces with different args.

### Q35 — `run_all_nn_seeds.sh` passes `--skip-phase1` unconditionally — first-run behavior?
**Location**: `run_all_nn_seeds.sh:27`.
**Issue**: On a fresh checkout, the anchors don't exist, so `if args.skip_phase1 and os.path.isfile(anchor_path)` is False and Phase 1 runs. Good. But the README (lines 152-160) discusses Phase 1 as a first-class step without noting that `--skip-phase1` is the default in the orchestrator. Reviewers may be confused.

### Q36 — `target_scale` is computed from training data and saved per-seed
**Location**: `train_T188v2_nn_seed.py:266`.
**Issue**: `target_scale = 1.0 / max(y_regr_tr.std(), 1e-8)`. If two re-runs see the same data in the same order, same value — but the aug scales modify nothing because target_scale is computed *before* augmentation. OK. Just verify this is intended.

### Q37 — Phase 2 uses the *same* `target_scale` written by Phase 1
**Location**: `train_T188v2_nn_seed.py:377`.
**Issue**: But Phase 2 retrains on M7 (all dates 0-119), where the y distribution may differ from Phase 1's train (0-79). Is the carry-over of target_scale principled or accidental? At inference, predictions are divided by this scale. If Phase 2 sees broader y, predictions might be shrunk too much.

### Q38 — Conformal `per_sym_beta` and `per_sym_sigma` are baked into `thresholds.json` — where are they computed?
**Location**: `04_build_pkg/thresholds.json:54-59`.
**Issue**: These were calibrated externally. The pipeline does not regenerate them after retraining. If you retrain the 50+50 ensemble (different seeds, different convergence), the abstain band is calibrated against the *old* model's residuals. Is this consistent?

---

## D. Performance / resource budgets

### Q39 — Wall-clock estimate is anecdotal
**Location**: README's "5 GPU 并行 ≈ 18 min", run_pipeline.sh's "~30 min on CPU / ~12 min with GPU LightGBM".
**Issue**: No code path enforces or verifies wall-clock. If Step 3 takes 6h on a small GPU (e.g., a T4), there's no warning to the user.

### Q40 — `run_pipeline.sh` runs NN serial, not parallel
**Location**: `run_all_nn_seeds.sh:20-28` is a sequential `for SEED in $(seq 1 50)`.
**Issue**: README says 5 GPU parallel is possible — but the script does not support it. To use 5 GPUs the user must manually shard. No orchestrator (e.g., `--seeds 1-10`).

### Q41 — Memory peak in NN Phase 2
**Location**: `train_T188v2_nn_seed.py:390-414`. Concatenates all splits' X into `X_all`, then duplicates into `X_full` (2× size), then `X_tr_raw = torch.from_numpy(...)`.
**Issue**: For a 226-feat dataset of 1.47M rows: 226 × 1.47M × 4 bytes × 2 (aug) × 2 (numpy + torch ref) ≈ 5.3 GB. On 16GB RAM with concurrent processes (LGB), this could OOM. Documented anywhere?

### Q42 — Disk space for cache
**Location**: README "~20 GB cache".
**Issue**: No precondition check in `run_pipeline.sh` (e.g., `df -h` to ensure target dir has enough free space).

### Q43 — `run_pipeline.sh` does not check preconditions
**Issue**: Does not verify `nvidia-smi` exists if `--gpu`. Does not verify Python version. Does not verify `lightgbm` is installed. Does not verify input parquets exist before starting Step 1.

### Q44 — Inference time budget
**Issue**: README mentions ~500ms per batch on CPU for NN ensemble. Platform limit is 3h. What is the worst-case test set size, and what is the worst-case predict() latency × number of calls? No estimate is in the code or docs.

### Q45 — `_BatchedMLPEnsemble.predict_mean` materializes `(N=50, B, in_dim)` tensor
**Location**: `Predictor.py:155`. `h.unsqueeze(0).expand(50, -1, -1).contiguous()`.
**Issue**: With batch=1024 (from `config.json:3`) and in_dim=359: 50 × 1024 × 359 × 4 B ≈ 73 MB. Then `bmm(h, W^T)` allocates 50 × 1024 × 256 × 4 B = 52 MB for layer 1. Peak inference memory is ~150 MB — fine, but `.contiguous()` after `.expand()` defeats the broadcast trick. Why not let bmm broadcast `feat_mean/std` directly?

### Q46 — LightGBM inference is sequential (50 boosters in a Python loop)
**Location**: `Predictor.py:310-314`.
**Issue**: 50 × N rows × ~330 trees = sizable. README claims 84 ms is acceptable. But each `b.predict(X)` allocates a new array. No reuse. Acceptable, but is the inference latency budgeted?

---

## E. Submission package

### Q47 — 148 MB package, what's the platform limit?
**Issue**: README says "约 148 MB". Platform limit (per task brief "2GB") is fine. But there's no `assert pkg_size < limit` in `build_pkg.py`. If 4 weights doubled in size for some reason, would the package still build silently?

### Q48 — fp16 / quantize?
**Issue**: 50 NN × ~270 KB each = ~13 MB. 50 LGB × ~2 MB = 100 MB. The LGB text format is verbose. Could be GZIP'd in the npz, but probably not worth optimizing — though no analysis is presented.

### Q49 — Package format requirements (from platform)
**Issue**: What does the platform expect at the root of the unzipped package? A `Predictor.py` with class `Predictor`? Does it call `Predictor()` and then `.predict()`? Where is this contract documented in this repo? No `PLATFORM_API.md`.

### Q50 — `predict()` signature: `List[pd.DataFrame] -> List[List[int]]`
**Location**: `Predictor.py:316`.
**Issue**: Is this exactly what the platform expects? What if the platform passes a single DataFrame? A list of dicts? No test stub exercises the real platform input.

### Q51 — `__main__` smoke test in Predictor.py uses random data
**Location**: `Predictor.py:368-388`.
**Issue**: Smoke test creates random normal data, not a real 100-tick LOB window. It won't exercise edge cases like zero spread, negative midprice, NaN raw features. The test verifies "code runs", not "outputs are sane".

### Q52 — Versioning of the package
**Issue**: Filename is `submission_050911_iter019_v2N_50plus50_optimized_fullhorizon.zip` — hardcoded in `run_pipeline.sh:19`. If user re-runs with tweaks, the filename collision overwrites the previous submission. No semver, no timestamp suffix.

### Q53 — `outputs/submission_*.zip` is excluded by gitignore — good, but no archive policy
**Issue**: If a user makes 10 submissions, all 10 zips live in `outputs/` until manually deleted. Pipeline doesn't move old ones to `outputs/archive/`.

### Q54 — Package doesn't include training metadata
**Issue**: At inference time, the platform receives Predictor + weights + thresholds. There's no `MODEL_CARD.md` describing what dates were used, what seeds, what code version. If the platform asks "explain why this submission scored X", you can't answer from the zip alone.

### Q55 — No `BUILD_INFO.json` or git SHA in package
**Issue**: Combined with Q25, if the source tree changes after building, the package has no provenance.

---

## F. Shell scripts hygiene

### Q56 — `set -e` only, no `-u -o pipefail`
**Location**: All 3 shell scripts.
**Issue**: See Q11. `${LGB_GPU_FLAG}` (run_pipeline.sh:51) is unquoted — if it contains spaces, breaks. With `set -u` you'd catch typos early.

### Q57 — Positional args, no `--data-dir` / `--cuda`
**Location**: `run_pipeline.sh:11-13`.
**Issue**: Positional bash args are fragile. User must remember "first is data, second is cuda, third is GPU flag". A typo in arg order (e.g., `./run_pipeline.sh 0 ./data`) silently passes wrong data dir. No `getopts` / `--help`.

### Q58 — `WORKDIR=$(cd "$(dirname "$0")" && pwd)` assumes script is callable directly
**Location**: `run_pipeline.sh:14`.
**Issue**: Fine if `bash run_pipeline.sh` from project root. If sourced or invoked with `bash -c "cat $0 | bash"`, breaks. Edge case but worth noting.

### Q59 — Paths hardcoded relative to `WORKDIR`
**Location**: `run_pipeline.sh:15-19`.
**Issue**: `$OUT` is `$WORKDIR/outputs` — cannot be overridden via env var. What if user has `outputs/` symlinked to a fast SSD? They have to symlink, can't pass `--out`.

### Q60 — `cd "$PKG_DIR" && zip -qr "$ZIP_OUT" .`
**Location**: `run_pipeline.sh:75`.
**Issue**: `cd` then `zip .` includes the working directory's `.gitkeep`, hidden files, etc. The grep is `.` which catches dotfiles. Safer: explicit file list.

### Q61 — `cd "$WORKDIR"` after zip — but the script ends right after
**Location**: `run_pipeline.sh:76`.
**Issue**: Harmless, but suggests this used to be sourced. Dead code.

### Q62 — `run_all_lgb_seeds.sh:9` `GPU_FLAG=${3:-""}`
**Location**: `run_all_lgb_seeds.sh:9`.
**Issue**: Empty string default; line 26 `$GPU_FLAG` expands to nothing. Quoting is correct (unquoted, intentionally word-split). With `set -u` this would still work because `${3:-""}` provides default. But there's no validation that user passed `--gpu` literal (not `gpu` or `-g`).

### Q63 — `run_all_nn_seeds.sh:27` `--skip-phase1` hardcoded
**Location**: `run_all_nn_seeds.sh:27`.
**Issue**: As Q34: no way for the orchestrator to force re-run of Phase 1. If user re-runs all NNs intentionally, they have to manually delete `outputs/models/T81_pretrained/`.

### Q64 — All bash scripts use 4-space indentation but the actual exec mode bit is set
**Location**: `run_pipeline.sh:1` shebang `#!/bin/bash`. Some files (`run_all_*.sh`) are executable. Stylistic.
**Issue**: `run_pipeline.sh` and `run_all_lgb_seeds.sh` start with `#!/bin/bash`. Mixed quoting style.

### Q65 — No `dry-run` mode
**Issue**: Can't test "what would this do" without actually running 90 minutes of training.

---

## G. README documentation accuracy

### Q66 — README claims feature dim 370; `fast_features.py:10-11` docstring says 350
**Location**: `README.md:18-19` (370 = 154+196+20) vs `fast_features.py:10-11` ("350-feature vector ... = 350").
**Issue**: `fast_features.py` docstring says "350 (T3 69 + Stage1 54 + Stage2 59 + Stage3 14 + raw last 154 = 350)" — but README's 370 = 154 + 196 + 20 Stage5. Where exactly is Stage5 integrated? `build_schemeP_cache.py:19` says "fast_features_batch.py already integrates Stage5". So Stage5 is in `fast_features_batch.py`. But `fast_features.py` (the single-window version used at inference) — does it include Stage5? Its docstring says no. Does `Predictor.py` use `fast_features.py` or `fast_features_batch.py`? **Predictor.py:191** uses `fast_features_batch.py`. So `fast_features.py` may be dead code in the pkg.

### Q67 — `fast_features.py` may be dead code in the package
**Location**: `04_build_pkg/fast_features.py` is shipped (568 lines), `Predictor.py:191` loads `fast_features_batch.py` only.
**Issue**: Confirm. If dead, why ship 568 lines? If conditional, document the trigger.

### Q68 — README "MLP [359→256→128→64→1]" — derive from code
**Location**: `train_T188v2_nn_seed.py:34` `HIDDEN = (256, 128, 64)`; feat_dim = 154 + 216 - 11 = 359.
**Issue**: Matches. But the magic 11 is the count of `T59_FAIL_NAMES + STAGE5_FAIL_NAMES` (lines 52-60). If a feature is renamed and falls out of `DROP_NAMES`, feat_dim changes silently. No assert against an expected 359 anywhere.

### Q69 — README lists "5 HP configs × 10 seeds = 50" — but seeds cycle 1..50 with hp_group = (seed-1)%5
**Location**: `train_T188v2_lgb_seed.py:25-31, 83`.
**Issue**: Matches. But: HP combination diversity (NN) has 3×4×3=36 combinations cycled across 50 seeds; not all 36 combinations are visited. Is this intentional? Bias from missing combinations is not analyzed in README.

### Q70 — README's "phase1_val_split: dates 80-95 (15 days)" — actually 16 days
**Location**: `build_schemeP_cache.py:82` (`80 <= d < 96` = dates 80..95 = 16 days). README:156 says "15 天".
**Issue**: Off-by-one in documentation.

### Q71 — "Conformal abstain" formula in README differs from code
**Location**: README line 217: "eff_thr_up = thr_up + β_sym * σ_sym". `Predictor.py:260`: `pred_dmid > (thr_up + band_per_row)` where `band_per_row` is `β * σ` precomputed.
**Issue**: Matches. Good. But: the README also says "若 |pred| 在带内则 action=1" — that's the *symmetric* abstain; the code uses *asymmetric* (thr_up and thr_dn are independent). The formula in README is approximate.

### Q72 — `share_with` semantics not formalized
**Location**: `Predictor.py:336-340`. `thresholds.json:6` `"share_with": 60`.
**Issue**: If `share_with` points to a horizon that's `active: false`, what happens? `self._lgb_lists.get(src_H)` returns None and the loop `continue`s — but the actions stay as default 1 for that horizon. README doesn't cover this edge case.

### Q73 — Missing TROUBLESHOOTING / FAQ
**Issue**: No "common errors" doc. What if Step 1 raises `KeyError: label_60`? What if Step 3 OOMs? What if zip fails because of large file?

### Q74 — Missing CHANGELOG
**Issue**: README mentions "iter_018", "T188v2", "T188v3", "T75", "T81", "T87", "T170" — these are internal experiment IDs not explained anywhere in the repo. A new reader cannot map them to features.

### Q75 — Missing ARCHITECTURE.md
**Issue**: Why M7 fullretrain? Why SPO+? Why 50+50? README has a table at the bottom but no causal narrative.

---

## H. Implicit assumptions / schema

### Q76 — Parquet schema is not validated
**Location**: `build_schemeP_cache.py:155` `pd.read_parquet(path)`. Line 156 calls `build_one_session(df)`. Line 94: `df[RAW_COLS].to_numpy(...)`.
**Issue**: If a column is missing from a parquet (e.g., `volume_delta` typo'd), pandas raises `KeyError` deep in the loop. No upfront schema check. With 1200 parquet files, late-discovery is painful.

### Q77 — Missing parquets silently skipped
**Location**: `build_schemeP_cache.py:152-154, 178`.
**Issue**: `if not os.path.isfile(path): skipped += 1; continue`. Prints a warning at the end. But the pipeline still produces a smaller `schemeP_train.npz`. Downstream training treats it as valid. The warning doesn't fail the pipeline. What if 50% of files are missing?

### Q78 — Per-session window assumption: 100 ticks contiguous
**Location**: `build_schemeP_cache.py:88-97`.
**Issue**: `sliding_window_view` assumes consecutive rows in the parquet are consecutive ticks. If the parquet has gaps (e.g., a missing minute), the window blends across the gap. Is the parquet guaranteed contiguous?

### Q79 — AM vs PM session boundary not crossed (good) — but documented?
**Location**: `build_schemeP_cache.py:73-84` splits by `(sym, date, sess)`.
**Issue**: Correct: each session is processed independently, no cross-session window. Good. But this is not explicit in README; a reader might worry about overnight gaps.

### Q80 — sym is read from input DataFrame's *last row only*
**Location**: `Predictor.py:298-303` `s = int(df["sym"].iloc[-1])`.
**Issue**: Assumes sym is constant within a 100-tick window. Reasonable. But: what if `sym` column is missing entirely? Falls back to `_cw_default_band`. What if sym is a string? `int(...)` raises ValueError, caught at line 300. What if sym is, say, 7 (training was 0-4)? OOD branch fires (default beta). README says this is intentional. Good.

### Q81 — `keep_idx` assumption: all 50 NN models share the same one
**Location**: `Predictor.py:98` (loads from first npz only).
**Issue**: If seed 1's npz has different keep_idx than seed 25's, the Predictor uses seed 1's silently. Should assert equality across all npz files.

### Q82 — `in_dim` assumption: all 50 NN share the same in_dim
**Location**: `Predictor.py:95, 110-123`.
**Issue**: Same as Q81. Loop assumes same `hidden`, `use_layernorm`, `in_dim`. No assert.

### Q83 — `target_scale` per-NN, but no per-NN `clip`
**Location**: `Predictor.py:114, 130` `self.clip = max(clips)`.
**Issue**: Uses the *maximum* clip across all 50 — that's a conservative choice but it differs from per-NN behavior. Why max? If one NN was trained with CLIP=5, others with CLIP=10, the inference clips at 10 always. This biases ensemble vs training. Documented?

### Q84 — Inference assumes batches are List[pd.DataFrame]
**Location**: `Predictor.py:316-319`. `predict(self, batches: List[pd.DataFrame])`.
**Issue**: Each batch must be exactly 100 rows (`WINDOW`). What happens if not? `df[self._raw_feat_cols].to_numpy(...)` at line 278 will return rows≠100 and the subsequent `X3d = np.empty((N, WINDOW, K))` slice at line 277 will mismatch. Silent shape error or crash?

### Q85 — `fast_features_batch.compute_batch_features` returns 216-d (or 196? Q66 ambiguity)
**Issue**: If the count differs from `_extra_keep_idx` size, the concat at `_compute_batch_features:289` produces wrong-dim features. No assert.

### Q86 — Mp_t = midprice1 of the LAST tick
**Location**: `build_schemeP_cache.py:115`. `mp_t = raw_last[:, mid_idx]`.
**Issue**: Implicit. If the platform's inference passes a window whose last tick differs from training convention (e.g., off by one), predictions skew. Document the "what is t?" convention.

### Q87 — Stage5 is in `01_build_features/stage5_features.py` AND in `fast_features_batch.py`?
**Location**: `build_schemeP_cache.py:42` comment "Note: stage5_features.py is kept in this directory for reference but not called here."
**Issue**: Dead file. Why ship it? Confusing for readers.

---

## I. Risk / fragility hot-spots

### Q88 — "If this breaks, evaluation gets 0/NaN" — the NaN guard
**Location**: `_BatchedMLPEnsemble.predict_mean:159` `torch.where(torch.isnan(h), 0, h)`. `build_schemeP_cache.py:105` `np.where(np.isfinite(extras), extras, 0.0)`.
**Issue**: NaN-to-zero strategy at *two* layers. If a new Stage5 feature produces Inf, the build cache replaces with 0; if inference produces NaN after normalization, replaced with 0. But what if a *LightGBM* prediction is NaN (e.g., feature has Inf at inference but not training)? `_ensemble_predict_lgb` doesn't sanitize.

### Q89 — `pred_dmid` NaN propagates to `out = np.full(B, 1)` then `pred > thr` → False
**Location**: `Predictor.py:259-262`.
**Issue**: If LGB returns NaN, the comparison `pred_dmid > (thr_up + band)` is False, action stays 1 (flat). Safe degradation. But there's no telemetry.

### Q90 — Single-source-of-truth for FAIL_NAMES is duplicated 3× (training × 2, inference × 1)
**Location**: `train_T188v2_lgb_seed.py:33-41`, `train_T188v2_nn_seed.py:52-60`, `Predictor.py:68-75`.
**Issue**: Three copies of the same list. If a feature is added to or removed from the FAIL list, all three must be updated. No central constants file.

### Q91 — Feature ordering in `fast_features_batch.all_feature_names()` is the source of truth
**Location**: `build_schemeP_cache.py:210`, `Predictor.py:199`.
**Issue**: If `all_feature_names()` returns features in a slightly different order in 04 vs 01 (e.g., due to a Python dict ordering change in 3.x patch), models break silently. We saw that 01 and 04 copies are byte-identical (diff is empty) — good. But the duplication exists; one shared module would be safer.

### Q92 — Predictor uses `importlib.util.spec_from_file_location` to load `fast_features_batch.py`
**Location**: `Predictor.py:181-185, 191`.
**Issue**: Why not a regular `import`? Because the package may not be a Python package (no `__init__.py`). OK — but this hides ImportErrors until runtime. If `scipy` is missing at the platform, the import fails *inside* `Predictor.__init__`, deep stack.

### Q93 — Conformal default_beta_for_ood = 0.16, default_sigma_for_ood = 4e-4
**Location**: `thresholds.json:58-59`.
**Issue**: A new sym (e.g., sym=7) silently uses these defaults. If platform introduces 5 new syms with very different volatility, the abstain logic is miscalibrated. Documented as "best effort" but not tested.

### Q94 — `_extract_band_per_row` reads `df["sym"].iloc[-1]` — sym must be int-castable
**Location**: `Predictor.py:296-303`.
**Issue**: Exception caught. Good. But if sym is a numpy scalar with a weird dtype, `int(...)` may succeed but with truncation. Edge case.

### Q95 — `_BatchedMLPEnsemble.__init__` keeps `feat_mean`, `feat_std`, etc. on the chosen device
**Location**: `Predictor.py:128`.
**Issue**: If CUDA is selected but the platform has limited VRAM (e.g., shared with another process), allocation may fail. No fallback to CPU.

### Q96 — `torch.bmm` numerical drift vs sequential PyTorch nn.Sequential
**Location**: `Predictor.py:164` claims `~1e-6` typical, `~1e-4` max.
**Issue**: 1e-4 absolute is *large* if `pred_dmid` is on the order of 1e-3 (threshold 0.0003). A 1e-4 drift could flip an action gate. Has the cross-check been run on real platform-like data, not random?

### Q97 — `thresholds.json` precision in `_doc` comments vs actual values
**Location**: `thresholds.json:12, 22, 32, 42`.
**Issue**: `_note` says "* sqrt(5/60) = * 0.2887" applied to h=60 `thr_up=0.0003` → 8.66e-5. JSON has 8.66e-5. Good. But the `thr_dn` ratio is computed differently (h=60 `thr_dn` = 2.16e-4 → 0.2887 * 2.16e-4 = 6.24e-5). Matches. Brittle; if someone tweaks h=60 thresholds, the share_with values become stale.

### Q98 — No automated test for `thresholds.json` schema
**Location**: `Predictor.py:208-223` parses it loosely.
**Issue**: A typo in JSON (`enabled: true` as string) breaks at runtime. No schema validation.

### Q99 — No CI/CD
**Issue**: Repo has no `.github/workflows/`, no smoke test pipeline, no pre-submit lint. A user can commit a broken Predictor.py and only discover at platform submission time.

### Q100 — No end-to-end integration test
**Issue**: Even locally, there is no `test_pipeline.sh` that runs on a tiny synthetic dataset and asserts the final zip is valid and the Predictor produces plausible outputs. The smoke test in `Predictor.py.__main__` is too narrow.

---

**Total: 100 questions** (target 70+, exceeded).

Key architecture concerns (in rough priority):
1. **Q5, Q6, Q7, Q8** — Data contracts between stages are implicit; no schema/version/checksum guards.
2. **Q9, Q22, Q81-83, Q100** — No equivalence tests between training-time and inference-time MLP forward; no integration smoke test.
3. **Q27-Q34, Q37** — Reproducibility is best-effort; bit-equivalence not enforced; multiple RNG streams; `target_scale` carry-over needs justification.
4. **Q15, Q23, Q24** — Two requirement environments, only one pinned; platform OS/arch/python assumptions undocumented.
5. **Q43, Q56-Q57** — `run_pipeline.sh` lacks preflight checks, strict mode, named args.
6. **Q66-Q70, Q74-Q75** — README accuracy issues; missing CHANGELOG / ARCHITECTURE / FAQ.
7. **Q34, Q35** — `--skip-phase1` is the orchestrator default; silent reuse of stale anchors with mismatched args.
8. **Q40** — Multi-GPU parallel training is documented but not implemented.
9. **Q49, Q50** — Platform API contract not documented in repo.
10. **Q88, Q96** — NaN safety + bmm-vs-sequential drift could individually flip an action.

RESULT: task=[pipeline arch review] metrics={n_questions=100} notes=[Wrote 100 architecture-level questions to QREVIEW_PIPELINE_ARCH.md across stage organization, data contracts, train/submit isolation, reproducibility, performance budgets, packaging, shell hygiene, README accuracy, implicit assumptions, and risk hot-spots. Headline issues: missing schema/version/checksum guards on inter-stage artifacts (Q5-Q8), no equivalence test between training MLP and bmm-ensemble Predictor (Q9, Q96), --skip-phase1 default could silently reuse mismatched anchors (Q34-Q35), no preflight/strict-mode in shell (Q43, Q56), README/code drift on feature counts (Q66-Q70).]
