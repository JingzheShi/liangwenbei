# Repository Guidelines

## Project Structure & Module Organization

This repository reproduces a staged model-training and submission pipeline. `run_pipeline.sh` is the end-to-end entry point. `01_build_features/` builds the schemeP feature cache from raw parquet data. `02_train_lgb/` trains the 50 LightGBM seed ensemble. `03_train_nn/` trains the 50 neural-network seed ensemble. `04_build_pkg/` assembles the final platform package and contains runtime inference code such as `Predictor.py`, `fast_features.py`, `config.json`, `thresholds.json`, and `requirements.txt`. Generated artifacts belong under `outputs/`; raw data should be placed in `data/` and is not tracked.

## Build, Test, and Development Commands

Install training dependencies with:

```bash
pip install numpy pandas lightgbm scipy torch
```

Install submission-package runtime dependencies with:

```bash
pip install -r 04_build_pkg/requirements.txt
```

Run the full reproduction pipeline:

```bash
./run_pipeline.sh ./data 0
```

Build only features:

```bash
python3 01_build_features/build_schemeP_cache.py --data_dir ./data --out ./outputs/cache
```

Train individual model families with `bash 02_train_lgb/run_all_lgb_seeds.sh ./outputs/cache ./outputs/models --gpu` and `bash 03_train_nn/run_all_nn_seeds.sh ./outputs/cache ./outputs/models 0`. Assemble the package with `python3 04_build_pkg/build_pkg.py --models ./outputs/models --pkg ./outputs/pkg`.

## Coding Style & Naming Conventions

Use Python 3.10+ and keep scripts CLI-friendly with `argparse`. Follow the existing style: 4-space indentation, module constants in `UPPER_CASE`, snake_case functions and variables, and seed/model filenames matching existing patterns such as `model_h60_seed{S}.txt` and `nn_h60_seed{S}.npz`. Keep feature order and dimensionality assertions explicit; inference code must not use `date` or cross-call mutable state.

## Testing Guidelines

There is no standalone unit-test suite. Validate changes by running the smallest affected stage first, then package assembly. For feature changes, rebuild `outputs/cache` and confirm expected `.npz` files and feature-name counts. For inference/package changes, rebuild `outputs/pkg` and smoke-test importability of `Predictor.py` in an environment using `04_build_pkg/requirements.txt`.

## Commit & Pull Request Guidelines

Recent history uses concise, imperative commit subjects with context, for example `Fix gitignore: scope *.txt to LGB model files` or `Add reproducibility bundle: raw data → 50+50 fullhorizon submission pipeline`. Keep commits focused and avoid committing generated caches, model artifacts, raw parquet data, or submission zips unless explicitly required. Pull requests should describe the changed pipeline stage, commands run, expected artifact changes, and any score, speed, or reproducibility impact.

## Security & Configuration Tips

Do not hardcode local data paths, credentials, or machine-specific CUDA settings. Prefer command-line parameters and keep large generated outputs in `outputs/`, which can be recreated from source and data.
