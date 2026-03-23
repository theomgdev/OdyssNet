import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, cast
from ..utils.data import prepare_input, to_tensor

import os
if os.environ.get('NO_BNB'):
    HAS_BNB = False
else:
    try:
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"
        # Check if user wants verbose logs (VERBOSE_BNB=1)
        if os.environ.get('VERBOSE_BNB'):
            import bitsandbytes as bnb
            HAS_BNB = True
        else:
            # Suppress bitsandbytes logs during import
            import sys
            _old_out, _old_err = sys.stdout, sys.stderr
            _null_out, _null_err = open(os.devnull, 'w'), open(os.devnull, 'w')
            try:
                sys.stdout, sys.stderr = _null_out, _null_err
                import bitsandbytes as bnb
                HAS_BNB = True
            finally:
                sys.stdout, sys.stderr = _old_out, _old_err
                _null_out.close()
                _null_err.close()
    except ImportError:
        HAS_BNB = False

# Import ChaosGrad and TemporalScheduler
from .chaos_optimizer import ChaosGrad, ChaosGradConfig
from .chaos_scheduler import TemporalScheduler, TemporalSchedulerConfig
from ..utils.neurogenesis import Neurogenesis

class RealNetTrainer:
    def __init__(self, model, optimizer=None, loss_fn=None, lr=1e-4, device='cpu',
                 gradient_persistence=0.0, synaptic_noise=0.0,
                 chaos_config=None, scheduler_config=None,
                 use_chaos_grad=None, use_temporal_scheduler=None):
        """
        Initializes the trainer.

        Args:
            model (nn.Module): The RealNet model to train.
            optimizer (torch.optim.Optimizer): Custom optimizer (Optional).
                If None, auto-selects: ChaosGrad (if use_chaos_grad) or AdamW8bit/AdamW.
            loss_fn (callable): Custom loss function (Optional).
            lr (float): Initial learning rate (Default: 1e-4). Used if optimizer is None.
            device (str): Device to run training on.
            gradient_persistence (float): How much gradient to keep from previous step (0.0 - 0.9).
            synaptic_noise (float): Scale of noise added to weights during training (Regularization). Default 0.0.
            chaos_config (dict, optional): Config dict for ChaosGrad. See ChaosGradConfig presets.
                Use ChaosGradConfig.conservative(), ChaosGradConfig.default(), etc.
            scheduler_config (dict, optional): Config dict for TemporalScheduler.
                Use TemporalSchedulerConfig.default(), TemporalSchedulerConfig.llm(), etc.
            use_chaos_grad (bool, optional): Force enable/disable ChaosGrad.
                None = auto (use ChaosGrad only when no custom optimizer given and chaos_config provided).
            use_temporal_scheduler (bool, optional): Force enable/disable TemporalScheduler.
                None = auto (use when scheduler_config provided).
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.gradient_persistence = gradient_persistence
        self.synaptic_noise = synaptic_noise
        self.initial_lr = lr

        # --- Optimizer Initialization ---
        self._using_chaos_grad = False
        self._chaos_config = chaos_config  # Store for re-use after neurogenesis
        self._sentinel_param_ids = set()
        self.scheduler = None

        if optimizer:
            # User explicitly provided an optimizer — use it directly.
            self.optimizer = optimizer
        else:
            # Auto-select optimizer
            should_use_chaos = False
            if use_chaos_grad is True:
                should_use_chaos = True
            elif use_chaos_grad is None and chaos_config is not None:
                should_use_chaos = True

            if should_use_chaos:
                self._init_chaos_grad(model, lr, chaos_config)
            elif HAS_BNB and device == 'cuda':
                print("RealNetTrainer: Using bitsandbytes 8-bit AdamW for VRAM efficiency.")
                adamw8bit_cls = getattr(bnb.optim, 'AdamW8bit', None)
                if adamw8bit_cls is not None:
                    self.optimizer = adamw8bit_cls(model.parameters(), lr=lr, weight_decay=0.01)
                else:
                    print("RealNetTrainer: AdamW8bit symbol not found. Falling back to standard AdamW.")
                    self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            else:
                print("RealNetTrainer: bitsandbytes not found or CPU mode. Using standard AdamW.")
                self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # --- Scheduler Setup ---
        should_use_scheduler = False
        if use_temporal_scheduler is True:
            should_use_scheduler = True
        elif use_temporal_scheduler is None and scheduler_config is not None:
            should_use_scheduler = True

        if should_use_scheduler:
            sched_cfg = cast(dict[str, Any], scheduler_config or TemporalSchedulerConfig.default())
            self.scheduler = TemporalScheduler(self.optimizer, **sched_cfg)

        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()

        # Diagnostic tracking
        self._step_count = 0
        self._last_loss = None
        self._acc_counter = 0
        self._persistent_grads = {}

    def _init_chaos_grad(self, model, lr, chaos_config):
        """Initialize ChaosGrad optimizer with parameter classification."""
        config = cast(dict[str, Any], chaos_config or ChaosGradConfig.default(lr=lr))

        # Set base LR in config
        if 'lr' not in config:
            config['lr'] = lr

        # Classify parameters
        param_groups, sentinel_params = ChaosGrad.classify_params(model)
        self._sentinel_param_ids = sentinel_params

        # Apply config to each group while preserving group-specific flags
        for group in param_groups:
            for key, value in config.items():
                if key not in group and key != 'params':
                    group[key] = value

        self.optimizer = ChaosGrad(param_groups, **config)
        self.optimizer.set_sentinel_params(sentinel_params)
        self._using_chaos_grad = True

        group_info = {g.get('group_name', '?'): len(g['params']) for g in param_groups}
        print(f"RealNetTrainer: Using ChaosGrad optimizer. Groups: {group_info}")

    def _ensure_scaler(self):
        """Lazily initialize AMP scaler in a version-compatible way."""
        if hasattr(self, 'scaler'):
            return
        amp_mod = getattr(torch, 'amp', None)
        scaler_cls = getattr(amp_mod, 'GradScaler', None) if amp_mod is not None else None
        if scaler_cls is not None:
            self.scaler = scaler_cls('cuda', enabled=(self.device == 'cuda'))
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

    def _inject_persistent_grads(self):
        """Inject persisted unscaled gradients before the optimizer step."""
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    param.grad.zero_()
                    continue
                persisted = self._persistent_grads.get(id(param))
                if persisted is not None:
                    param.grad.add_(persisted.to(device=param.grad.device, dtype=param.grad.dtype))

    def _capture_persistent_grads(self):
        """Store unscaled gradients for the next optimization step."""
        self._persistent_grads = {}
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    continue
                self._persistent_grads[id(param)] = (param.grad.detach().clone() * self.gradient_persistence)

    def _clear_persistent_grads(self):
        self._persistent_grads.clear()

    def _rebind_scheduler(self, scheduler_state=None, verbose=True):
        """Re-create scheduler against the current optimizer and restore state if possible."""
        if self.scheduler is None:
            return

        old_sched = self.scheduler
        self.scheduler = TemporalScheduler(
            self.optimizer,
            warmup_steps=old_sched.warmup_steps,
            max_steps=old_sched.max_steps,
            min_lr_ratio=old_sched.min_lr_ratio,
            restart_factor=old_sched.restart_factor,
            restart_decay=old_sched.restart_decay,
            patience=old_sched.patience,
            cooldown=old_sched.cooldown,
            loss_smoothing=old_sched.loss_smoothing,
            auto_extend=old_sched.auto_extend,
            verbose=old_sched.verbose,
        )

        if scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ TemporalScheduler state restore failed after expand: {e}. Resetting scheduler state.")

    def state_dict(self):
        """Return trainer runtime state for robust checkpoint resume."""
        state = {
            'step_count': self._step_count,
            'last_loss': self._last_loss,
            'acc_counter': self._acc_counter,
            'gradient_persistence': self.gradient_persistence,
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        if hasattr(self, 'scaler'):
            state['scaler_state_dict'] = self.scaler.state_dict()

        if self.gradient_persistence > 0.0 and self._persistent_grads:
            persistent_by_name = {}
            for name, param in self.model.named_parameters():
                persisted = self._persistent_grads.get(id(param))
                if persisted is not None:
                    persistent_by_name[name] = persisted.detach().cpu()
            state['persistent_grads'] = persistent_by_name

        return state

    def load_state_dict(self, state):
        """Restore trainer runtime state from checkpoint payload."""
        if not state:
            return

        self._step_count = int(state.get('step_count', self._step_count))
        self._last_loss = state.get('last_loss', self._last_loss)
        self._acc_counter = int(state.get('acc_counter', 0))

        saved_gp = state.get('gradient_persistence', None)
        if isinstance(saved_gp, (float, int)) and self.gradient_persistence <= 0.0:
            self.gradient_persistence = float(saved_gp)

        scheduler_state = state.get('scheduler_state_dict', None)
        if self.scheduler is not None and scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(scheduler_state)
            except Exception as e:
                print(f"⚠️ Could not restore TemporalScheduler state: {e}")

        scaler_state = state.get('scaler_state_dict', None)
        if scaler_state is not None:
            try:
                self._ensure_scaler()
                self.scaler.load_state_dict(scaler_state)
            except Exception as e:
                print(f"⚠️ Could not restore GradScaler state: {e}")

        self._clear_persistent_grads()
        persistent_payload = state.get('persistent_grads', None)
        if isinstance(persistent_payload, dict):
            named_params = dict(self.model.named_parameters())
            for name, persisted in persistent_payload.items():
                param = named_params.get(name)
                if param is None or not isinstance(persisted, torch.Tensor):
                    continue
                self._persistent_grads[id(param)] = persisted.to(device=param.device, dtype=param.dtype)

    def train_batch(self, input_features, target_values, thinking_steps, gradient_accumulation_steps=1, full_sequence=False, mask=None, output_transform=None, initial_state=None, return_state=False):
        """
        Runs a single training step on a batch.
        """
        self.model.train()

        self._ensure_scaler()

        # Synaptic Noise
        if self.synaptic_noise > 0.0:
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * self.synaptic_noise
                        param.add_(noise)

        # Prepare Data
        # If model has vocab_size, we assume input is Token IDs or Raw Vects for Projection.
        # We bypass 'prepare_input' which attempts to map features to specific neurons manually.
        if hasattr(self.model, 'vocab_size') and self.model.vocab_size is not None:
             x_input = to_tensor(input_features, self.device)
             batch_size = x_input.shape[0]
        elif isinstance(input_features, torch.Tensor) and input_features.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
             x_input = input_features.to(self.device)
             batch_size = x_input.shape[0]
        else:
             x_input, batch_size = prepare_input(input_features, self.model.input_ids, self.model.num_neurons, self.device)

        target_values = to_tensor(target_values, self.device)
        if mask is not None:
            mask = to_tensor(mask, self.device)

        # Forward Pass (with AMP)
        device_type = 'cuda' if self.device == 'cuda' else 'cpu'
        amp_mod = getattr(torch, 'amp', None)
        autocast_fn = getattr(amp_mod, 'autocast', None) if amp_mod is not None else None
        if autocast_fn is not None:
            autocast_ctx = autocast_fn(device_type=device_type, enabled=(self.device == 'cuda'))
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=(self.device == 'cuda'))

        with autocast_ctx:
            # Use initial_state if provided, otherwise reset
            if initial_state is not None:
                current_state_in = initial_state
            else:
                self.model.reset_state(batch_size)
                current_state_in = None

            all_states, final_state = self.model(x_input, steps=thinking_steps, current_state=current_state_in)

            # Extract Outputs & Calculate Loss
            if hasattr(self.model, 'vocab_size') and self.model.vocab_size is not None:
                 # Vocab Mode: 'all_states' is decoded output (Logits)
                 raw_output = all_states

                 if full_sequence:
                     predicted_outputs = raw_output
                 else:
                     # Prediction on last step only: (B, T, Vocab) -> (B, Vocab) at T=-1
                     predicted_outputs = raw_output[:, -1, :]
            else:
                # Continuous Activity Mode: Extract from explicit output neurons
                output_indices = self.model.output_ids
                if full_sequence:
                    predicted_outputs = all_states[:, :, output_indices]
                else:
                    predicted_outputs = final_state[:, output_indices]

            # Optional Transform
            if output_transform:
                predicted_outputs = output_transform(predicted_outputs)

            if mask is not None:
                loss = (torch.square(predicted_outputs - target_values) * mask).mean()
            else:
                loss = self.loss_fn(predicted_outputs, target_values)

            # Normalize loss for gradient accumulation
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

        # Backward
        self.scaler.scale(loss).backward()

        # Step optimizer only if accumulation cycle is complete
        step_now = True
        if gradient_accumulation_steps > 1:
            self._acc_counter += 1
            if self._acc_counter % gradient_accumulation_steps != 0:
                step_now = False

        if step_now:
            self.scaler.unscale_(self.optimizer)

            # Persistence is injected after unscale so AMP scaling does not distort it.
            if self.gradient_persistence > 0.0:
                self._inject_persistent_grads()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.gradient_persistence > 0.0:
                self._capture_persistent_grads()
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self._clear_persistent_grads()
                self.optimizer.zero_grad(set_to_none=True)

            self._acc_counter = 0
            self._step_count += 1

        # Return loss for logging
        loss_val = loss.item() * gradient_accumulation_steps
        self._last_loss = loss_val

        # Report loss to ChaosGrad for plateau detection
        if self._using_chaos_grad and step_now:
            chaos_opt = cast(ChaosGrad, self.optimizer)
            chaos_opt.report_loss(loss_val)

        # Step scheduler if active
        if self.scheduler is not None and step_now:
            self.scheduler.step(loss=loss_val)

        if return_state:
            return loss_val, final_state
        return loss_val

    def predict(self, input_features, thinking_steps, full_sequence=False):
        """
        Runs inference.
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'vocab_size') and self.model.vocab_size is not None:
                x_input = to_tensor(input_features, self.device)
                batch_size = x_input.shape[0]
            else:
                x_input, batch_size = prepare_input(input_features, self.model.input_ids, self.model.num_neurons, self.device)

            self.model.reset_state(batch_size)
            all_states, final_state = self.model(x_input, steps=thinking_steps)

            if hasattr(self.model, 'vocab_size') and self.model.vocab_size is not None:
                # In Vocab Mode, 'all_states' is the decoded output (Logits)
                raw_output = all_states

                if full_sequence:
                    return raw_output
                else:
                    return raw_output[:, -1, :]
            else:
                # Continuous Activity Mode: Feature extraction from output neurons
                output_indices = self.model.output_ids

                if full_sequence:
                    return all_states[:, :, output_indices]
                else:
                    return final_state[:, output_indices]

    def evaluate(self, input_features, target_values, thinking_steps):
        """
        Evaluates the model on a dataset.
        """
        self.model.eval()
        with torch.no_grad():
            preds = self.predict(input_features, thinking_steps)
            target_values = to_tensor(target_values, self.device)

            loss = self.loss_fn(preds, target_values)
            return loss.item()

    def regenerate_synapses(self, threshold=0.01, percentage=None):
        """
        Triggers synaptic regeneration (Darwinian Revive) on weak connections.
        Re-initializes weights below threshold (or bottom percentage) instead of pruning them.
        """
        revived, total = self.model.regenerate_weak_weights(threshold, percentage)
        return revived, total

    def fit(self, input_features, target_values, epochs, batch_size=32, thinking_steps=10, verbose=True):
        """
        Trains the model for a fixed number of epochs.
        """
        input_features = to_tensor(input_features, self.device)
        target_values = to_tensor(target_values, self.device)

        history = []

        # Prepare Data
        batch_size = min(batch_size, len(input_features))
        dataset_len = len(input_features)

        # Simple Batching Loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            # Helper for random permutation
            indices = torch.randperm(dataset_len)

            for i in range(0, dataset_len, batch_size):
                batch_indices = indices[i:i+batch_size]
                x_batch = input_features[batch_indices]
                y_batch = target_values[batch_indices]

                loss = self.train_batch(x_batch, y_batch, thinking_steps)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            history.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs}: Loss {avg_loss:.6f}")

        return history

    def expand(self, amount=1, verbose=True):
        """
        Dynamically adds neurons to the model (Neurogenesis).
        Ensures optimizer continuity and memory cleanup.
        """
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        self.optimizer = Neurogenesis.expand(
            self.model,
            self.optimizer,
            amount,
            verbose,
            chaos_config=self._chaos_config if self._using_chaos_grad else None,
        )
        self._clear_persistent_grads()

        if self.scheduler is not None:
            self._rebind_scheduler(scheduler_state=scheduler_state, verbose=verbose)

        # Neurogenesis internally orchestrates optimizer state migration.
        if self._using_chaos_grad and verbose:
            print("   🌪️ ChaosGrad: Optimizer state preserved after neurogenesis.")

    # --- Diagnostic Methods ---

    def get_diagnostics(self):
        """
        Returns comprehensive training diagnostics.

        Returns a dict with optimizer and scheduler health metrics.
        Useful for monitoring and logging in complex training loops.
        """
        diag = {
            'step_count': self._step_count,
            'last_loss': self._last_loss,
            'using_chaos_grad': self._using_chaos_grad,
            'current_lr': self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0,
        }

        if self._using_chaos_grad:
            chaos_opt = cast(ChaosGrad, self.optimizer)
            diag['optimizer'] = chaos_opt.get_diagnostics()

        if self.scheduler is not None:
            diag['scheduler'] = self.scheduler.get_diagnostics()

        return diag

    def get_input_health(self):
        """
        Returns input gradient health score (0.0 = dead, 1.0 = healthy).

        Only meaningful when using ChaosGrad with input_sentinel=True.
        Useful for detecting "input-blind" local minima in large networks
        where the network learns to generate reasonable outputs from chaos
        alone, ignoring actual input.
        """
        if self._using_chaos_grad:
            chaos_opt = cast(ChaosGrad, self.optimizer)
            return chaos_opt._input_grad_health
        return -1.0  # Not available

    def get_spectral_radius(self):
        """
        Returns estimated spectral radius of the core W matrix.

        Only meaningful when using ChaosGrad with spectral_clip > 0.
        Values > 1.0 indicate potentially chaotic dynamics.
        """
        if self._using_chaos_grad:
            chaos_opt = cast(ChaosGrad, self.optimizer)
            return chaos_opt._spectral_radius
        return -1.0  # Not available
