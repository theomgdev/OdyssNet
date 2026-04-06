import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from typing import Callable, cast
from ..utils.data import prepare_input, to_tensor

import os

from .chaos_optimizer import ChaosGrad
from ..utils.neurogenesis import Neurogenesis

class OdyssNetTrainer:
    def __init__(self, model, optimizer=None, loss_fn=None, lr=1e-3, device='cpu',
                 gradient_persistence=0.0, synaptic_noise=0.0,
                 anomaly_hook=None):
        """
        Initializes the trainer.

        Args:
            model (nn.Module): The OdyssNet model to train.
            optimizer (torch.optim.Optimizer): Custom optimizer (Optional).
                If None, ChaosGrad is used automatically with lr as genesis_lr.
            loss_fn (callable): Custom loss function (Optional).
            lr (float): Genesis learning rate for ChaosGrad. Default: 1e-3.
            device (str): Device to run training on.
            gradient_persistence (float): How much gradient to keep from previous step (0.0-0.9).
            synaptic_noise (float): Scale of noise added to weights during training. Default 0.0.
            anomaly_hook (callable, optional): Called as hook(event_type, loss_value) on
                anomalies ('spike', 'plateau', 'increase').
        """
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.gradient_persistence = gradient_persistence
        self.synaptic_noise = synaptic_noise
        self.initial_lr = lr

        # --- Optimizer Initialization ---
        self._using_chaos_grad = False
        self.anomaly_hook: Callable[[str, float], None] | None = anomaly_hook
        self._start_time_pred = None
        self._loss_time_buffer = []

        if optimizer:
            # User explicitly provided an optimizer — use it directly.
            self.optimizer = optimizer
            if isinstance(optimizer, ChaosGrad):
                self._using_chaos_grad = True
        else:
            # ChaosGrad is the native OdyssNet optimizer.
            self._init_chaos_grad(model, lr)

        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()

        # Diagnostic tracking
        self._step_count = 0
        self._last_loss = None
        self._acc_counter = 0
        self._persistent_grads = {}

    def _init_chaos_grad(self, model, lr):
        """Initialize ChaosGrad with classified parameter groups."""
        param_groups = ChaosGrad.classify_params(model)
        self.optimizer = ChaosGrad(param_groups, lr=lr)
        self._using_chaos_grad = True

        group_info = {g.get('group_name', '?'): len(g['params']) for g in param_groups}
        print(f"OdyssNetTrainer: ChaosGrad initialized. Groups: {group_info}")

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

    def state_dict(self):
        """Return trainer runtime state for robust checkpoint resume."""
        state = {
            'step_count': self._step_count,
            'last_loss': self._last_loss,
            'acc_counter': self._acc_counter,
            'gradient_persistence': self.gradient_persistence,
        }

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

        scaler_state = state.get('scaler_state_dict', None)
        if scaler_state is not None:
            try:
                self._ensure_scaler()
                self.scaler.load_state_dict(scaler_state)
            except Exception as e:
                print(f"WARNING: Could not restore GradScaler state: {e}")

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
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        noise = torch.randn_like(param) * self.synaptic_noise
                        param.add_(noise)
                        # Protect the core matrix diagonal from noise injection
                        if 'W' in name and param.dim() == 2 and param.shape[0] == param.shape[1]:
                            param.fill_diagonal_(0.0)

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

            all_states, final_state = self.model(x_input, steps=thinking_steps, current_state=current_state_in, return_sequence=full_sequence)

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

        # Predictive tracking formulation natively without blocking performance
        current_time = time.time()
        if self._start_time_pred is None:
            self._start_time_pred = current_time
            self._loss_time_buffer = []

        t_elapsed = current_time - self._start_time_pred
        # Buffer update rate throttling (max 1 point per second to ensure clear time signals and minimal overhead)
        if not self._loss_time_buffer or (t_elapsed - self._loss_time_buffer[-1][0] >= 1.0):
            self._loss_time_buffer.append((t_elapsed, loss_val))
            if len(self._loss_time_buffer) > 200:
                self._loss_time_buffer.pop(0)

        # Anomaly Hook (Spikes, Plateaus, Increases) evaluation
        hook = self.anomaly_hook
        if hook is not None:
            # 1. Step-to-step absolute increase
            if hasattr(self, '_prev_step_loss'):
                if loss_val > self._prev_step_loss:
                    hook("increase", loss_val)
            self._prev_step_loss = loss_val

            # 2. Spike Detection
            if not hasattr(self, '_anomaly_ewma') or self._anomaly_ewma is None:
                self._anomaly_ewma = loss_val
                self._anomaly_var = 0.0
            else:
                alpha = 0.05
                diff = loss_val - self._anomaly_ewma
                self._anomaly_ewma += alpha * diff
                self._anomaly_var = (1 - alpha) * (self._anomaly_var + alpha * diff ** 2)
                std = math.sqrt(self._anomaly_var) + 1e-8
                
                if diff > 3 * std and loss_val > 1.2 * self._anomaly_ewma:
                    hook("spike", loss_val)
                    self._anomaly_ewma = loss_val 
                    self._anomaly_var = 0.0

            # Plateau logic on time buffer
            if len(self._loss_time_buffer) >= 20:
                recent = [l for t, l in self._loss_time_buffer[-10:]]
                older = [l for t, l in self._loss_time_buffer[-20:-10]]
                if sum(recent) / 10 >= (sum(older) / 10) * 0.999: # Stuck or worsening
                    if not getattr(self, '_plateau_hook_triggered', False):
                        hook("plateau", loss_val)
                        self._plateau_hook_triggered = True
                else:
                    self._plateau_hook_triggered = False

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
            all_states, final_state = self.model(x_input, steps=thinking_steps, return_sequence=full_sequence)

            if hasattr(self.model, 'vocab_size') and self.model.vocab_size is not None:
                # Vocab Mode: 'all_states' is the decoded output (Logits)
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
        self.optimizer = Neurogenesis.expand(
            self.model,
            self.optimizer,
            amount,
            verbose,
        )
        self._clear_persistent_grads()

        if self._using_chaos_grad and verbose:
            print("   ChaosGrad: Optimizer state preserved after neurogenesis.")

    # --- Diagnostic Methods ---

    def trigger_plateau_escape(self):
        """Manually triggers plateau escape in ChaosGrad."""
        if self._using_chaos_grad:
            chaos_opt = cast(ChaosGrad, self.optimizer)
            chaos_opt.trigger_plateau_escape()
            print("Manual plateau escape triggered!")

    def get_diagnostics(self, debug=False):
        """
        Returns comprehensive training diagnostics.

        Args:
            debug (bool): If True, includes computationally intensive diagnostics
                such as gradient statistics, persistent gradient info, and detailed
                optimizer metrics. Default: False.

        Returns:
            dict: Diagnostic information including:
                - step_count: Number of optimization steps taken
                - last_loss: Most recent loss value
                - using_chaos_grad: Whether ChaosGrad optimizer is being used
                - current_lr: Current learning rate
                - gradient_persistence: Gradient persistence coefficient
                - persistent_grads_active: Number of active persistent gradients (if debug=True)
                - anomaly_tracking: Anomaly detection state (if debug=True)
                - scaler_state: AMP scaler information (if debug=True)
                - optimizer: Nested optimizer diagnostics (if using ChaosGrad)
        """
        diag = {
            'step_count': self._step_count,
            'last_loss': self._last_loss,
            'using_chaos_grad': self._using_chaos_grad,
            'current_lr': self.optimizer.param_groups[0]['lr'] if getattr(self, 'optimizer', None) and getattr(self.optimizer, 'param_groups', None) else 0,
            'gradient_persistence': self.gradient_persistence,
        }

        if debug:
            # Persistent gradients info
            diag['persistent_grads_active'] = len(self._persistent_grads)

            # Anomaly tracking metrics
            anomaly_info = {}
            if hasattr(self, '_anomaly_ewma') and self._anomaly_ewma is not None:
                anomaly_info['ewma'] = self._anomaly_ewma
                anomaly_info['variance'] = self._anomaly_var
                anomaly_info['std'] = math.sqrt(self._anomaly_var) + 1e-8
            if hasattr(self, '_prev_step_loss'):
                anomaly_info['prev_step_loss'] = self._prev_step_loss
            if hasattr(self, '_plateau_hook_triggered'):
                anomaly_info['plateau_hook_triggered'] = self._plateau_hook_triggered

            if anomaly_info:
                diag['anomaly_tracking'] = anomaly_info

            # Loss time buffer stats
            if self._loss_time_buffer:
                diag['loss_tracking'] = {
                    'buffer_size': len(self._loss_time_buffer),
                    'time_elapsed': self._loss_time_buffer[-1][0] if self._loss_time_buffer else 0.0,
                    'recent_losses': [l for t, l in self._loss_time_buffer[-5:]] if len(self._loss_time_buffer) >= 5 else [],
                }

            # AMP scaler state
            if hasattr(self, 'scaler'):
                diag['scaler_state'] = {
                    'enabled': self.scaler.is_enabled() if hasattr(self.scaler, 'is_enabled') else True,
                    'scale': self.scaler.get_scale() if hasattr(self.scaler, 'get_scale') else None,
                }

            # Model gradient statistics
            grad_stats = self._compute_gradient_stats()
            if grad_stats:
                diag['gradient_stats'] = grad_stats

        if self._using_chaos_grad:
            chaos_opt = cast(ChaosGrad, self.optimizer)
            diag['optimizer'] = chaos_opt.get_diagnostics(debug=debug)

        return diag

    def _compute_gradient_stats(self):
        """Compute gradient statistics across all parameters."""
        grad_norms = []
        grad_means = []
        params_with_grad = 0
        params_without_grad = 0

        for param in self.model.parameters():
            if param.grad is not None:
                params_with_grad += 1
                grad_norms.append(param.grad.norm().item())
                grad_means.append(param.grad.mean().item())
            else:
                params_without_grad += 1

        if not grad_norms:
            return None

        import torch
        grad_norm_tensor = torch.tensor(grad_norms)
        grad_mean_tensor = torch.tensor(grad_means)

        return {
            'params_with_grad': params_with_grad,
            'params_without_grad': params_without_grad,
            'norm': {
                'min': grad_norm_tensor.min().item(),
                'max': grad_norm_tensor.max().item(),
                'mean': grad_norm_tensor.mean().item(),
                'std': grad_norm_tensor.std().item() if len(grad_norms) > 1 else 0.0,
            },
            'mean': {
                'min': grad_mean_tensor.min().item(),
                'max': grad_mean_tensor.max().item(),
                'mean': grad_mean_tensor.mean().item(),
                'std': grad_mean_tensor.std().item() if len(grad_means) > 1 else 0.0,
            },
        }
