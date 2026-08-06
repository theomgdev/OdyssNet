# Changelog

All notable changes to OdyssNet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.6.0] — 2026-08-06

### Added
- **ChaosGrad** — OdyssNet's bespoke zero-config optimizer, rebuilt from first principles and now the default. Combines Adam-style per-synapse preconditioning with online distance adaptation (D-adaptation class estimator): no learning rate is required. Exported as `odyssnet.ChaosGrad`.
- **Architecture-aware family policy**: parameters are auto-classified into `chaos_core`, `memory_feedback`, `projections`, `plasticity`, and `modulation` families. Weight decay applies only to connective structure; Hebbian logits and gates are never decayed. The chaos core's zero-diagonal constraint is enforced inside the optimizer step.
- **Anchored traction limit** (`trust_ratio`): the applied step scale is capped at a fraction of the network's initial weight scale, shielding tiny chaotic networks from distance-estimator overshoot (stock Prodigy could not solve the 9-parameter XOR example; ChaosGrad solves it zero-config with the step scale settling at the previously hand-tuned value).
- **Loss-spike brake** (`brake_factor`): on a statistical loss spike the distance estimate is scaled down and re-grows only if the landscape supports it, fixing late-training divergence on sharpening temporal tasks (delayed adder). The trainer feeds the loss stream automatically; custom loops can call `optimizer.report_loss(loss)`.
- `trainer.get_diagnostics()` now includes ChaosGrad health metrics under the `optimizer` key, and `current_lr` reports the live step-scale estimate.

### Changed
- **`OdyssNetTrainer` default optimizer is now ChaosGrad** (`lr=None` → automatic estimation). Passing an explicit `lr` selects ChaosGrad's fixed-rate mode (AdamW-equivalent updates under the family policy) instead of AdamW.
- All convergence examples now run zero-config (no `lr` argument). Precision-record examples (embed/record/reverse-record/LLM) keep their tuned rates via fixed-rate mode.
- `Neurogenesis.expand()` migrates ChaosGrad with fresh family grouping while preserving the step-scale estimate.

### Fixed
- `Neurogenesis.expand()` silently copied shape-tracking optimizer state tensors (`exp_avg`, `exp_avg_sq`, ...) without resizing them, crashing the first optimizer step after expansion.
- Examples with emoji output crashed on legacy Windows code pages (cp1254); affected scripts now reconfigure stdout to UTF-8.

### Removed
- `prodigyopt` dependency (Prodigy is superseded by ChaosGrad).

## [2.5.0] — 2026-04-30

### Added
- **Spatial Hebbian Plasticity**: Introduced a co-activation learning mechanism (classic Hebbian) alongside the existing STDP-style learning.
- **`hebb_mode` functionality (`hebb_type`)**: `hebb_type` is now repurposed to act as the mechanism toggle: `None` (disabled), `"temporal"`, `"spatial"`, or `"both"`.
- **`hebb_res`**: Controls the structural resolution (`"global"`, `"neuron"`, `"synapse"`). Defaults to `"neuron"`.

### Changed
- **BREAKING**: Replaced single-path Hebbian parameters with path-specific prefixes (`t_hebb_factor` and `s_hebb_factor`, etc.). Existing checkpoints utilizing `hebb_factor` will need to be loaded with `strict=False` and re-trained, or manually patched, as we have prioritized a clean architecture over legacy support.

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
