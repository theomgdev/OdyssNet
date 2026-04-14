# Changelog

All notable changes to OdyssNet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.5.0] — 2026-04-14

### Added
- **ChaosGrad v3** (`odyssnet/training/chaos_optimizer.py`): Zero-hyperparameter optimizer re-introduced as a fully optional, drop-in custom optimizer. Pass it via `OdyssNetTrainer(model, optimizer=ChaosGrad(...))`. Default optimizer selection (Prodigy / AdamW) is unchanged.
  - **Second-moment adaptive normalisation** (`v2` EMA + bias correction + `denom`) — closes the AdamW performance gap by continuously re-calibrating gradient scale.
  - **Bias-corrected momentum** (`v_hat = v / (1 - β^t)`) — eliminates cold-start understepping.
  - **Grad EMA signal reference** — replaces single-step `prev_grad` with a slow EMA (`α = 0.6`) for more stable hypergradient signals in recurrent regimes.
  - **Group-aware frustration bursts** — Hebbian logits (`hebb_factor`, `hebb_decay`) are unconditionally excluded from burst noise. `chaos_core`/`memory`/`projections` receive full bursts; all other groups receive half-scale noise with no meta-reset.
  - **9-group parameter classification** (`classify_params`) — `bias`, `norm`, and `scales` promoted from `lightweight` into dedicated groups with appropriate beta equilibria (0.95 for `chaos_core`/`memory`, 0.85 for `gates`).
- `OdyssNetTrainer.trigger_plateau_escape()` re-introduced (no-op when non-ChaosGrad optimizer is active).
- `OdyssNetTrainer.get_diagnostics()` automatically includes `'optimizer'` key with ChaosGrad diagnostics when ChaosGrad is detected.
- `ChaosGrad` exported from `odyssnet` public API.
- Neurogenesis (`trainer.expand()`) handles ChaosGrad migration natively: classified param groups are rebuilt for the grown model and global frustration state is preserved.

## [2.4.0] — 2026-04-10

### Added
- **Prodigy optimizer** is now the default when `lr=None` (the new default). Prodigy auto-calibrates the learning rate continuously — no manual LR tuning required. Requires the `prodigyopt` package (`pip install prodigyopt`), now listed as a core dependency.

### Changed
- `OdyssNetTrainer` default `lr` changed from `1e-4` to `None`. Passing `lr=None` (default) activates Prodigy; passing an explicit float (e.g. `lr=1e-4`) still selects AdamW with `weight_decay=0.01`.

## [2.3.1] — 2026-04-09

### Added
- Added `ODYSSNET_DISABLE_PLOT` environment variable support to `TrainingHistory.plot()` to bypass interactive plotting during automated runs.
- Updated `examples/test_all.py` to automatically set `ODYSSNET_DISABLE_PLOT=1` before spawning sub-processes.

### Fixed
- Fixed bug in `save_checkpoint` where `os.makedirs` crashes if a bare filename is provided (e.g. `"model.pt"`) due to an empty directory string.
- Fixed 5/6/7-space indentations across codebase to comply with strict 4-space PEP 8 guidelines.
- Cleaned up several unused imports (`torch.nn`, `Dataset`, `math`) in advanced examples.

### Changed
- Default learning rate in `OdyssNetTrainer` changed to `1e-4` (previously `1e-3`).
- Centralized repetitive output-extraction and autocast resolution logic in `OdyssNetTrainer` into private helper methods (`_extract_outputs`, `_get_autocast_ctx`), standardizing logic.
- Optimized optimizer state transferring logic in `neurogenesis.expand()`, collapsing multiple loops.

## [2.3.0] — 2026-04-06

### Removed
- Removed `ChaosGrad` optimizer — replaced with standard `AdamW` as default.
- Removed `bitsandbytes` dependency and all `NO_BNB` environment variable usage.
- Removed `trigger_plateau_escape()` from trainer (was ChaosGrad-specific).
- Renamed `micro_quiet_8bit` init strategy to `micro_quiet_warm`.

### Changed
- Default optimizer is now `torch.optim.AdamW(lr=1e-3, weight_decay=0.01)`.
- Diagonal zeroing of chaos core `W` matrix is now enforced by the trainer.
- `get_diagnostics()` simplified — removed ChaosGrad-specific metrics.

## [2.2.0] - 2026-04-06

### Added
- **ChaosGrad v2.2 "The Learning Teacher"**: Zero-hyperparameter optimizer with Analytic Hypergradient Descent. All meta-parameters (LR, momentum, weight decay, centralization) are autonomously adapted per-parameter at each step.
- **Heterogeneous Synaptic Plasticity** (`hebb_type`): Three resolution levels (`global`, `neuron`, `synapse`) for online Hebbian learning with fully differentiable logit parameters.
- **Parametric Gating**: Configurable per-branch gates (`encoder_decoder`, `core`, `memory`) with `identity` and `sigmoid` modes.
- **Label Smoothing**: Integrated into trainer for classification tasks.
- **Debug Mode** (`debug=True`): NaN/Inf diagnosis with per-operation forward-pass checks and automatic `detect_anomaly`.
- **Enhanced Diagnostics**: Both `ChaosGrad.get_diagnostics()` and `OdyssNetTrainer.get_diagnostics()` now support a `debug` parameter.
  - **ChaosGrad debug mode** includes per-parameter statistics (min/max/std) for learning rate, beta, alpha, decay, per-group breakdowns, and step count statistics.
  - **Trainer debug mode** includes gradient persistence tracking, anomaly detection state, loss tracking buffer info, AMP scaler state, and gradient statistics (norms/means).
- **Training history plotting** (`plot_history`): Utility to visualize loss, learning rate, and custom metrics over training.
- `pyproject.toml` for standard Python packaging (`pip install -e .`).
- `CONTRIBUTING.md` with example standards, initialization protocols, and contributor checklist.
- `LICENSE` file (MIT).
- `CHANGELOG.md` (this file).

### Changed
- Removed legacy ChaosScheduler — ChaosGrad now handles scheduling at granular synaptic level.
- Renamed `PoC/` to `examples/`, `PoC/experiments/` to `examples/advanced/` for open-source clarity.
- Removed `ChaosGradConfig` — ChaosGrad requires only a genesis `lr`.
- Removed `sys.path.append` hacks from all example scripts (use `pip install -e .` instead).
