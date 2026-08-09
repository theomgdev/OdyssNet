# Changelog

All notable changes to OdyssNet will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.6.2] — 2026-08-09

### Added
- **`cold_start_every`** (default 32) in the LLM example: during training each batch row's hidden state is independently zeroed with probability `1/cold_start_every`, giving an exponential spread of context lengths and keeping cold and warm contexts mixed in every batch. Without it the carried state is never reset, so gradient descent puts no pressure on cold-start behaviour at all. Measured at 1500 steps / batch 128 / same seed: `cold_start_every=0` collapsed 15.6% of held-out shards (pooled val loss 39.55, ppl 1e13) while its training loss read a healthy 2.689; `cold_start_every=32` collapsed 0.0% (pooled val loss 2.68, ppl 14.60) at training loss 2.706. The failure is removed at no measurable cost in fit quality.
- Reference run for the rewritten LLM example (`--mode train --minutes 25 --batch 256`, 2,625,280 params, RTX 3060 Ti): held-out loss 2.2009, ppl 9.03, bits/byte 0.8764, median cold start 2.3154, 0.0% collapsed cold starts, 226.15M tokens at 150,761 tok/s. Not a README-advertised metric — a reproducible baseline for the harness, quoted with the exact command that produces it.
- **Cold-start collapse is now a measured quantity.** `examples/advanced/experiment_llm.py` reports `val_p50` (median per-shard loss — what the model does on a *typical* cold start) and `val_coldfail` (fraction of shards that collapsed) alongside pooled loss/PPL/bits-per-byte. The pooled mean alone is unreadable when this failure mode is present: a handful of collapsed shards at ~240 nats swamp it and the run looks like a global divergence when most shards are healthy.

### Changed
- **`examples/advanced/experiment_llm.py` rewritten.** Was an unconditional infinite training loop with a blocking `input()` prompt on architecture mismatch, HuggingFace streaming (network-dependent, non-reproducible ordering, `MAX_START_SKIP` randomness), `exp(label_smoothed_loss)` reported as "perplexity", and best-checkpoint selection on *training* loss over a streaming corpus. Now `--mode {smoke,sweep,train,eval,gen}` over a local uint16 memmap token cache, with a held-out split, unsmoothed-CE perplexity, and compute-matched sweep presets (`gap`, `arch`, `batch`, `coldstart`, `optim`, `window`). TBPTT chunk 5 → 48 tokens: at chunk 5 a 512-token batch spent 103 full ChaosGrad steps over the 4.2M-element core, so the optimizer cost more than the model did.
- **The LLM example is now zero-config** (`lr=None`, ChaosGrad's estimator) instead of the pinned `1e-4` it carried since 2.6.0; `--lr <float>` still selects fixed-rate mode. The reference run above was measured under the automatic estimate. CLAUDE.md's convention list, which named the LLM example among those keeping an explicit rate, is updated to match. Note the estimator pins at its traction cap (~3.9e-3) within ~300 steps and stays there for the whole run — the `optim` sweep preset exists to compare that against pinned rates, but has not been run at a meaningful budget, so no claim is made that auto beats fixed here.
- `experiment_llm.py` validation runs in fp32 with TF32 disabled — not because precision was ever implicated (AMP on/off and TF32 on/off were measured to agree to four decimals) but because eval throughput is irrelevant and a metric used to rank sweep arms should not move with kernel choice. What *did* make the score batch-size dependent — same weights scoring 2.6 at eval batch 8 and 39 at batch 32 — was shard layout: eval batch size determines how the val split is cut and therefore which corpus positions each shard starts from, and only some starting positions fall into the absorbing state described under Known issues. `val_coldfail` now reports that directly instead of letting it distort the pooled mean.

### Fixed
- `OdyssNet.forward` allocated a throwaway CPU tensor per timestep (`torch.tensor(t)`) to pass the step index into `checkpoint.checkpoint`. With `use_reentrant=False` plain ints are accepted; on an LLM-scale sequence that was one allocation per `tokens x thinking_steps` iteration. Behaviour-identical.
- `OdyssNet.forward` scaled the full `(B, T, N)` activity tensor by `output_scale_vec` before slicing the output neurons out of it, in vocab-projection mode. Now slices first and scales the slice — mathematically identical (`output_scale_vec` is 1.0 outside `output_pos`), but on long sequences that tensor is the largest allocation in the forward pass and only `len(output_ids)` of its columns are ever read.

### Known issues
- **The chaos core can develop a degenerate absorbing state that training loss cannot see.** Measured on a 1024-neuron LM: after ~1500 steps with ChaosGrad's estimate pinned at its traction cap (3.93e-3), certain cold starts drive the hidden state into a regime producing ~240 nats/token which it never leaves, while training loss continued improving smoothly (2.67 nats). Per-row and single-row-batch replays confirm the rows are fully independent and the effect is reproducible per starting position, so this is dynamics, not numerics. Two mechanisms hide it: stateful TBPTT never re-enters the bad basin once training is on a benign trajectory, and ChaosGrad's spike brake watches the *training* loss stream, which stays smooth throughout — a blind spot in the brake's coverage worth noting for the next optimizer pass. `cold_start_every` is the example-level mitigation and is sufficient in every configuration measured so far, but it treats the symptom; the underlying dynamics are unchanged.
- The LLM example's `--sweep gap` verdict (`gap=1` best at equal wall-clock, `gap=3`/`gap=5` far behind) is a wall-clock result at a 1.3 min/arm budget, where the deeper arms only consumed 1.5-2.3M tokens against `gap=1`'s 5.68M. It does not establish that deeper thinking is worse *per token* — that needs a token-matched run, which the harness supports but which has not been done.

## [2.6.1] — 2026-08-07

### Fixed
- **ChaosGrad's loss-spike brake permanently scarred the estimator on ordinary noise.** Diagnosed via instrumented probes on `convergence_mnist_record.py`: the brake was firing every ~150-350 batches on nothing worse than normal per-batch classification variance (confirmed unrelated to Hebbian plasticity — fires *more* often with `hebb_type=None`), and each trigger permanently multiplied `d_numerator`/`d_max` by `brake_factor` with no way back. By epoch 25 of a 3k-subset probe, `effective_lr` had collapsed 500x and training had silently frozen — matching the ~83-87% accuracy plateau seen in full 100-epoch runs. Replaced the permanent mutation with a transient `brake_ceiling` multiplier applied only at the point of the actual parameter update, leaving the estimator's own bookkeeping untouched so it keeps learning the true scale while suppressed. The ceiling relaxes geometrically back toward 1.0 every `report_loss` call rather than on a fixed window: an isolated spike heals in tens of steps, a fast cascade (genuine divergence — the delayed-adder case this brake exists for) still compounds down. Also fixed a secondary bug where the post-spike variance reseed collapsed to 0.0, making the sigma test degenerate for ~20 calls after every fire.
- **The traction limit's `trust_ratio` anchor selection was a hard cutoff sitting exactly on the library's own init scale.** `_trust_cap` excluded groups with `rms0 < 1e-3` or `0.9 ≤ rms0 ≤ 1.1` outright — but `micro_quiet_warm` (one of OdyssNet's own bundled init strategies) initializes at `std=1e-3`, the exact value of the floor. Measured on `convergence_mnist_record.py` (which uses `micro_quiet_warm`): the same seed produced a traction cap of 0.000254 on CPU vs 0.004266 on CUDA — a 16.8x difference from nothing but which side of the 1e-3 cutoff a group's RNG-drawn `rms0` happened to land on. In practice this washed out on our examples (both land below the floor on the CUDA runs the READMEs are measured on), but it meant the cap a *library user* got was silently discontinuous in a region their own init could easily sit in. Replaced the hard include/exclude with `_anchor_weight` (a smooth ramp over the same two boundaries) plus `_trust_cap` blending each group toward the smallest already-fully-trusted group's own rms0 (not a fixed constant — an earlier version that blended toward a fixed reference passed every test but silently *moved* the cap on record.py itself, from 0.006 to 0.0025, by letting an excluded near-zero group's fallback undercut a real, smaller anchor elsewhere in the same model; caught by re-measuring the actual example configs before merging, not by the test suite). Same cap on every example already validated (confirmed by direct measurement on record.py's and the adder's actual param groups), continuous everywhere else. If literally every group is excluded (e.g. a lone all-zero-initialized parameter with no other family to anchor against), the cap is disabled rather than pinned to an arbitrary value — same as before.
- `Neurogenesis.expand()` rebuilt ChaosGrad via `from_model` without forwarding `brake_factor` (or the newly-exposed `brake_sigma`/`brake_ratio`/`brake_ema_alpha`) — a user who customized the brake would silently get the defaults back the moment their network grew. Now forwarded from the pre-expansion param group.
- `brake_sigma`/`brake_ratio`/`brake_ema_alpha` were first exposed as plain instance attributes, which `torch.optim.Optimizer.state_dict()` doesn't serialize (only `param_groups` and `state` are) — the same class of bug as the `Neurogenesis` one above, one mechanism over: a save/load round trip would have silently reverted a customized brake back to defaults. Moved into `defaults` so they ride the param groups like `brake_factor` already does.
- `examples/advanced/convergence_skill_transfer.py`'s "Claim check" required the transplanted model's *average* loss across the whole run to beat scratch's — but the transplanted model's epoch-0 loss is inflated by the 93%-freshly-initialized new region, dragging the average up regardless of how well it ultimately converges. The check printed "No clear transfer win" on every run even when transplanted beat scratch on final loss and time-to-threshold by a wide margin. Now judges the claim by final loss and first-epoch-below-threshold instead.

### Changed
- **`brake_sigma`, `brake_ratio`, and `brake_ema_alpha` are now constructor parameters** (previously hidden class constants a library user could not reach at all), defaults unchanged (3.0, 1.2, 0.05). The EWMA memory (`brake_ema_alpha`) in particular was tuned against OdyssNet's own ~16-32 batch-size examples; a user training at a very different batch size sees a differently-scaled loss-noise floor and can now retune it instead of silently inheriting ours.
- The brake's loss-EWMA/variance now use a bias-correction-style warmup (`alpha = max(brake_ema_alpha, 1/(brake_step+1))`, the same technique already used for Adam's own bias correction elsewhere in this file): the first ~20 calls after construction (or after a checkpoint reload) converge the estimate from every sample seen so far rather than committing to the steady-state window immediately. Verified this doesn't introduce spurious early spikes by logging `(loss, ema, std, is_spike)` for the delayed adder's first 50 `report_loss` calls — one legitimate-looking fire, no cluster near the start.
- `convergence_skill_transfer.py`: retuned `add_epochs` 500→250, `mul_epochs` 1500→500 — the shorter add-phase avoids overfitting the small model before transplant, giving a consistent, clear transfer win instead of a partial one.
- Re-validated numbers across README.md/README_TR.md under the fixed brake: MNIST 98.62%→98.71%, MNIST Revive 98.54%→98.70%, MNIST Tiny 95.15%→95.58%, MNIST Scaled 97.38%→98.01%, MNIST (8k) Embed 94.08%→93.71% (within run-to-run noise), Skill Transfer speedup 3.0x→3.6x. Sine Wave/Latch/Stopwatch log excerpts refreshed to match current runs.
- `convergence_mnist_record.py`: confirmed at full scale — previously froze around epoch 15-20 (~83-87% plateau, the exact symptom the brake fix targets), now trains cleanly through all 100 epochs, landing at 87.98% (peak 88.46%, epoch 86) zero-config. The README's 90.14% "WORLD RECORD" banner predates this optimizer entirely (different scheduler/preset pipeline, since removed) and is left as-is with an added status note — not a regression target, since the script itself changed too much for a like-for-like comparison. `LR` set to `None` (zero-config).
- Re-validated after the trust-cap and brake-warmup changes above: delayed adder (2000 epochs, no divergence, final loss ~0.00017), `convergence_mnist_record.py` 25-epoch diagnostic probe (`d_max` flat at 0.060141, `effective_lr` steady at 0.005958 with brake recovering after each dip — unchanged from before this fix), XOR seeds 42/123 (both solve zero-config), MNIST-3k/6-epoch probe (89.20%, byte-identical to the pre-change run), full `pytest tests/` (297/297).

### Known issues
- `convergence_sine_wave.py` shows a late-training instability under the fixed brake: loss explodes from ~0.001 to 0.01-0.06 starting around epoch 7900 of a 10000-epoch run. Not yet root-caused. `EPOCHS` reduced 10000→6800 as a stopgap (avoids the window entirely) rather than a fix; worth the same instrumentation approach used to diagnose the record.py brake issue.

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
- `convergence_mnist_embed.py` moved to zero-config (`lr` removed) — matches its previously tuned fixed-rate result under ChaosGrad.
- `convergence_adder.py` epochs raised 500 → 2000 to ride out the mid-training loss spike before the loss-spike brake settles the estimate back down.
- Re-validated and refreshed the advertised numbers in README.md/README_TR.md for MNIST, MNIST Revive, MNIST Tiny/Scaled, MNIST (8k) Embed, Sine Wave, Adder, Latch, Stopwatch, Detective, and Skill Transfer under ChaosGrad v2.6 — most improved (e.g. MNIST Tiny 90.2% → 95.15%, MNIST Revive 97.8% → 98.54%). MNIST (Record) and MNIST Reverse (Generation) numbers are intentionally left untouched pending further optimizer work.

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
