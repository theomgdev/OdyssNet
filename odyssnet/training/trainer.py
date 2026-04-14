import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import numbers
from typing import Callable
from ..utils.data import prepare_input, to_tensor
from ..utils.neurogenesis import Neurogenesis
from .chaos_optimizer import ChaosGrad

class OdyssNetTrainer:
    def __init__(self, model, optimizer=None, loss_fn=None, lr=None, device='cpu',
                 gradient_persistence=0.0, synaptic_noise=0.0,
                 anomaly_hook=None):
        """
        Initializes the trainer.

        Args:
            model (nn.Module): The OdyssNet model to train.
            optimizer (torch.optim.Optimizer): Custom optimizer (Optional).
                If None, the default optimizer is selected based on `lr`.
            loss_fn (callable): Custom loss function (Optional).
            lr (float or None): Learning rate. Default: None.
                - None: Prodigy optimizer is used. It auto-calibrates the learning
                  rate continuously and requires no manual tuning. Requires the
                  'prodigyopt' package (`pip install prodigyopt`).
                - float (e.g. 1e-4): AdamW optimizer is used with weight_decay=0.01.
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
        self.anomaly_hook: Callable[[str, float], None] | None = anomaly_hook
        self._start_time_pred = None
        self._loss_time_buffer = []

        if optimizer:
            # User explicitly provided an optimizer — use it directly.
            self.optimizer = optimizer
            self._using_chaos_grad = isinstance(optimizer, ChaosGrad)
        elif lr is None:
            # Default: Prodigy — auto-calibrating optimizer, no LR tuning required.
            try:
                from prodigyopt import Prodigy
            except ImportError as e:
                raise ImportError(
                    "The default optimizer requires the 'prodigyopt' package. "
                    "Install it with: pip install prodigyopt\n"
                    "Alternatively, pass an explicit lr (e.g. lr=1e-4) to use AdamW, "
                    "or pass optimizer=ChaosGrad(ChaosGrad.classify_params(model)) to use ChaosGrad."
                ) from e
            self.optimizer = Prodigy(model.parameters())
            self._using_chaos_grad = False
        else:
            # Explicit lr provided: use AdamW.
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            self._using_chaos_grad = False

        self.loss_fn = loss_fn if loss_fn else nn.MSELoss()

        # Diagnostic tracking
        self._step_count = 0
        self._last_loss = None
        self._acc_counter = 0
        self._persistent_grads = {}
        self._core_weight = getattr(self.model, 'W', None)

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

    def _get_autocast_ctx(self):
        """Return a version-compatible AMP autocast context for the current device."""
        device_type = 'cuda' if self.device == 'cuda' else 'cpu'
        amp_mod = getattr(torch, 'amp', None)
        autocast_fn = getattr(amp_mod, 'autocast', None) if amp_mod is not None else None
        if autocast_fn is not None:
            return autocast_fn(device_type=device_type, enabled=(self.device == 'cuda'))
        return torch.cuda.amp.autocast(enabled=(self.device == 'cuda'))

    def _extract_outputs(self, all_states, final_state, full_sequence):
        """Extract the prediction tensor from forward-pass outputs.

        Handles both vocab-projection mode (decoded logits) and continuous
        activity mode (explicit output neuron indices).
        """
        if hasattr(self.model, 'vocab_size') and self.model.vocab_size is not None:
            # Vocab mode: all_states already holds decoded logits.
            if full_sequence:
                return all_states
            # Prediction from the last timestep only: (B, T, Vocab) -> (B, Vocab)
            return all_states[:, -1, :]

        # Continuous activity mode: slice from explicit output neuron indices.
        output_indices = self.model.output_ids
        if full_sequence:
            return all_states[:, :, output_indices]
        return final_state[:, output_indices]


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

    def train_batch(self, input_features, target_values, thinking_steps, gradient_accumulation_steps=1, full_sequence=False, mask=None, output_transform=None, keep_state=False):
        """
        Runs a single training step on a batch.
        """
        if isinstance(gradient_accumulation_steps, bool) or not isinstance(gradient_accumulation_steps, numbers.Integral):
            raise ValueError("gradient_accumulation_steps must be an integer >= 1")
        gradient_accumulation_steps = int(gradient_accumulation_steps)
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be an integer >= 1")

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
        with self._get_autocast_ctx():
            if not keep_state:
                self.model.reset_state(batch_size)

            all_states, h_t = self.model(x_input, steps=thinking_steps, return_sequence=full_sequence)

            predicted_outputs = self._extract_outputs(all_states, h_t, full_sequence)

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

            # ChaosGrad: report loss for frustration tracking
            if self._using_chaos_grad:
                self.optimizer.report_loss(loss.item() * gradient_accumulation_steps)

            # Enforce zero diagonal on chaos core weight matrix.
            # Self-connections are handled by memory_feedback, not W.
            with torch.no_grad():
                core_weight = self._core_weight
                current_weight = getattr(self.model, 'W', None)
                if core_weight is not current_weight:
                    core_weight = current_weight
                    self._core_weight = current_weight
                if isinstance(core_weight, torch.Tensor) and core_weight.dim() == 2 and core_weight.shape[0] == core_weight.shape[1]:
                    core_weight.fill_diagonal_(0.0)

        # Return loss for logging
        loss_val = loss.item() * gradient_accumulation_steps
        self._last_loss = loss_val

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
            return self._extract_outputs(all_states, final_state, full_sequence)

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

        When ChaosGrad is the active optimizer, the per-parameter state for
        ``W`` is automatically cleared after regeneration so that stale
        momentum accumulated on the old weak connections does not push the
        freshly re-initialised weights in the wrong direction. ChaosGrad will
        cold-start recalibrate ``W`` on the next step.
        """
        revived, total = self.model.regenerate_weak_weights(threshold, percentage)
        if self._using_chaos_grad and revived > 0:
            w_param = getattr(self.model, 'W', None)
            if w_param is not None:
                self.optimizer.reset_param_state(w_param)
        return revived, total

    def fit(self, input_features, target_values, epochs, batch_size=32, thinking_steps=10, verbose=True):
        """
        Trains the model for a fixed number of epochs.
        """
        input_features = to_tensor(input_features, self.device)
        target_values = to_tensor(target_values, self.device)

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if len(input_features) != len(target_values):
            raise ValueError("input_features and target_values must have the same length")
        if len(input_features) == 0:
            raise ValueError("input_features and target_values must be non-empty")

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

    def trigger_plateau_escape(self) -> None:
        """
        Manually trigger a ChaosGrad frustration burst on the next optimizer step.

        Has no effect when a non-ChaosGrad optimizer is in use.
        """
        if self._using_chaos_grad:
            self.optimizer.trigger_plateau_escape()

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
        self._core_weight = getattr(self.model, 'W', None)
        self._clear_persistent_grads()

    # --- Diagnostic Methods ---

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
                - current_lr: Current learning rate
                - gradient_persistence: Gradient persistence coefficient
                - persistent_grads_active: Number of active persistent gradients (if debug=True)
                - anomaly_tracking: Anomaly detection state (if debug=True)
                - scaler_state: AMP scaler information (if debug=True)
        """
        diag = {
            'step_count': self._step_count,
            'last_loss': self._last_loss,
            'current_lr': self.optimizer.param_groups[0]['lr'] if getattr(self, 'optimizer', None) and getattr(self.optimizer, 'param_groups', None) else 0,
            'gradient_persistence': self.gradient_persistence,
        }

        # ChaosGrad-specific diagnostics (always included when active)
        if self._using_chaos_grad:
            diag['optimizer'] = self.optimizer.get_diagnostics(debug=debug)

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
