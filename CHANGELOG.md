# Changelog

All notable changes to OdyssNet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
