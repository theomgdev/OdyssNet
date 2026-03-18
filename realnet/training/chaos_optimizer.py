"""
ChaosGrad: A RealNet-Native Optimizer

Built specifically for RealNet's chaotic recurrent dynamics. Unlike AdamW which 
treats all parameters identically, ChaosGrad understands the different roles of 
parameters in a chaos chamber:

1. W (Core Matrix): The chaos engine. Needs careful spectral management.
2. Embeddings/Projections: Standard LLM-style parameters.
3. Bias/Scale/Norm: Lightweight params that should train faster.

Key Features:
- Per-parameter adaptive learning rates based on gradient health
- Input-gradient sentinel to detect "input-blind" local minima
- Spectral radius awareness for edge-of-chaos control
- Plateau escape via controlled gradient perturbation
- Decoupled weight decay for different parameter groups
"""

import torch
import math


class ChaosGrad(torch.optim.Optimizer):
    """
    ChaosGrad: RealNet-native optimizer.
    
    Extends AdamW with chaos-aware features:
    - Separate momentum/LR for chaos core (W) vs projections vs lightweight params
    - Gradient health monitoring with per-param adaptive LR
    - Plateau detection and escape via controlled perturbation
    - Input gradient sentinel for detecting vanishing input gradients
    
    Args:
        params: Iterable of parameters or parameter groups.
        lr (float): Base learning rate. Default: 1e-4.
        betas (tuple): AdamW momentum coefficients. Default: (0.9, 0.999).
        eps (float): Numerical stability. Default: 1e-8.
        weight_decay (float): Weight decay for chaos core. Default: 0.01.
        projection_decay (float): Weight decay for projections/embeddings. Default: 0.01.
        lightweight_lr_mult (float): LR multiplier for bias/scale/norm params. Default: 2.0.
        chaos_core_lr_mult (float): LR multiplier for core W matrix. Default: 1.0.
        projection_lr_mult (float): LR multiplier for projection layers. Default: 1.0.
        plateau_patience (int): Steps of no improvement before perturbation. Default: 0 (disabled).
        plateau_noise_scale (float): Scale of perturbation noise on plateau escape. Default: 0.01.
        spectral_clip (float): Max spectral radius for W matrix. Default: 0 (disabled).
        input_sentinel (bool): Track input gradient health. Default: False.
        adaptive_lr (bool): Enable per-param adaptive LR scaling. Default: True.
        adaptive_lr_clip (tuple): (min, max) multiplier for adaptive LR. Default: (0.1, 10.0).
        adaptive_ema (float): Smoothing factor for adaptive LR variance. Default: 0.99.
        grad_centralization (bool): Center gradients by removing mean. Default: True.
        plateau_noise_intensity (float): Internal multiplier for plateau noise. Default: 0.1.
        loss_history_min (int): Minimum number of recent loss values to retain in the history buffer. Default: 200.
        sentinel_threshold (float): Relative threshold for input health detection. Default: 0.1.
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, projection_decay=0.01,
                 lightweight_lr_mult=2.0, chaos_core_lr_mult=1.0,
                 projection_lr_mult=1.0,
                 plateau_patience=0, plateau_noise_scale=0.01,
                 spectral_clip=0.0, input_sentinel=False,
                 adaptive_lr=True, adaptive_lr_clip=(0.1, 10.0),
                 grad_centralization=True, adaptive_ema=0.99,
                 plateau_noise_intensity=0.1,
                 loss_history_min=200, sentinel_threshold=0.1):
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
            projection_decay=projection_decay,
            lightweight_lr_mult=lightweight_lr_mult,
            chaos_core_lr_mult=chaos_core_lr_mult,
            projection_lr_mult=projection_lr_mult,
            plateau_patience=plateau_patience,
            plateau_noise_scale=plateau_noise_scale,
            spectral_clip=spectral_clip,
            input_sentinel=input_sentinel,
            adaptive_lr=adaptive_lr,
            adaptive_lr_clip=adaptive_lr_clip,
            adaptive_ema=adaptive_ema,
            grad_centralization=grad_centralization,
            plateau_noise_intensity=plateau_noise_intensity,
            loss_history_min=loss_history_min,
            sentinel_threshold=sentinel_threshold,
        )
        super().__init__(params, defaults)
        
        # Global tracking
        self._global_step = 0
        self._loss_history = []
        self._plateau_counter = 0
        self._best_loss = float('inf')
        self._input_grad_health = 1.0  # 0.0 = dead, 1.0 = healthy
        self._spectral_radius = 0.0
        self._last_perturbation_step = -1
        self._diagnostics = {}
    
    @staticmethod
    def classify_params(model):
        """
        Classifies RealNet parameters into groups with appropriate settings.
        Returns a list of param groups suitable for the optimizer.
        
        Groups:
        1. chaos_core: W matrix (the NxN core)
        2. projections: Embedding, Linear projection, Output decoder
        3. lightweight: Bias, Scale, Norm parameters
        """
        chaos_core = []      # W matrix
        projections = []     # Embeddings, projections, decoders
        lightweight = []     # Bias, scale, norm
        
        # Track for sentinel
        input_sentinel_params = set()
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if name == 'W' or name.endswith('.W'):
                chaos_core.append(param)
            elif any(k in name for k in ['embed', 'proj', 'decoder', 'output_decoder']):
                projections.append(param)
            else:
                # B, input_scale, output_scale, norm.weight, norm.bias
                lightweight.append(param)
                
            # Input-related params for sentinel
            if any(k in name for k in ['input_scale', 'proj', 'embed']):
                input_sentinel_params.add(id(param))
        
        groups = []
        
        if chaos_core:
            groups.append({
                'params': chaos_core,
                'group_name': 'chaos_core',
                '_is_chaos_core': True,
                '_is_projection': False,
                '_is_lightweight': False,
            })
            
        if projections:
            groups.append({
                'params': projections,
                'group_name': 'projections',
                '_is_chaos_core': False,
                '_is_projection': True,
                '_is_lightweight': False,
            })
            
        if lightweight:
            groups.append({
                'params': lightweight,
                'group_name': 'lightweight',
                '_is_chaos_core': False,
                '_is_projection': False,
                '_is_lightweight': True,
            })
        
        return groups, input_sentinel_params
    
    def report_loss(self, loss_value):
        """Report current loss for plateau detection and adaptive behavior."""
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        self._loss_history.append(loss_value)
        
        # Keep only recent history
        hist_min = self.defaults.get('loss_history_min', 200)
        max_history = max(hist_min, self.defaults.get('plateau_patience', 0) * 2)
        if len(self._loss_history) > max_history:
            self._loss_history = self._loss_history[-max_history:]
            
        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._plateau_counter = 0
        else:
            self._plateau_counter += 1
    
    def get_diagnostics(self):
        """Returns a dict of current optimizer diagnostics."""
        return {
            'global_step': self._global_step,
            'plateau_counter': self._plateau_counter,
            'best_loss': self._best_loss,
            'input_grad_health': self._input_grad_health,
            'spectral_radius': self._spectral_radius,
            **self._diagnostics,
        }

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._global_step += 1
        
        # Plateau check
        plateau_patience = self.defaults.get('plateau_patience', 0)
        plateau_noise = self.defaults.get('plateau_noise_scale', 0.01)
        is_plateau = (plateau_patience > 0 and 
                     self._plateau_counter >= plateau_patience and
                     self._global_step - self._last_perturbation_step > plateau_patience)
        
        if is_plateau:
            self._last_perturbation_step = self._global_step
            self._plateau_counter = 0
        
        # Adaptive LR settings
        use_adaptive = self.defaults.get('adaptive_lr', True)
        adaptive_clip = self.defaults.get('adaptive_lr_clip', (0.1, 10.0))
        use_centralization = self.defaults.get('grad_centralization', True)
        spectral_clip = self.defaults.get('spectral_clip', 0.0)
        
        # Per-group processing
        total_grad_norm = 0.0
        input_grad_norm = 0.0
        total_param_norm = 0.0
        
        for group in self.param_groups:
            is_core = group.get('_is_chaos_core', False)
            is_proj = group.get('_is_projection', False)
            is_light = group.get('_is_lightweight', False)
            
            # Select appropriate LR multiplier
            if is_core:
                lr_mult = self.defaults['chaos_core_lr_mult']
                wd = self.defaults['weight_decay']
            elif is_proj:
                lr_mult = self.defaults['projection_lr_mult']
                wd = self.defaults['projection_decay']
            elif is_light:
                lr_mult = self.defaults['lightweight_lr_mult']
                wd = 0.0  # No decay for bias/scale/norm
            else:
                lr_mult = 1.0
                wd = self.defaults['weight_decay']
            
            beta1, beta2 = group.get('betas', self.defaults['betas'])
            eps = group.get('eps', self.defaults['eps'])
            base_lr = group.get('lr', self.defaults['lr'])
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                
                if grad.is_sparse:
                    raise RuntimeError('ChaosGrad does not support sparse gradients.')
                
                # Get or create state
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # For adaptive LR: track gradient variance
                    if use_adaptive:
                        state['grad_variance'] = torch.ones(1, device=p.device)
                        state['prev_grad_norm'] = torch.ones(1, device=p.device)
                
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # --- Gradient Centralization ---
                # Remove mean of gradient (proven to improve convergence)
                if use_centralization and grad.dim() >= 2:
                    grad = grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
                
                # --- Plateau Escape (Controlled Perturbation) ---
                if is_plateau and is_core:
                    intensity = group.get('plateau_noise_intensity', 0.1)
                    
                    noise_scale = grad.abs().mean()
                    if (not torch.isfinite(noise_scale)) or noise_scale < 1e-8:
                        noise_scale = p.abs().mean()
                    if (not torch.isfinite(noise_scale)) or noise_scale < 1e-8:
                        noise_scale = grad.new_tensor(1e-3)
                    noise = torch.randn_like(grad) * plateau_noise * noise_scale * intensity
                    grad = grad + noise
                
                # --- Decoupled Weight Decay ---
                if wd > 0:
                    p.data.mul_(1 - base_lr * lr_mult * wd)
                
                # --- AdamW Core Update ---
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                corrected_avg = exp_avg / bias_correction1
                corrected_avg_sq = exp_avg_sq / bias_correction2
                
                # Step size
                step_size = base_lr * lr_mult
                
                # --- Adaptive LR Scaling ---
                if use_adaptive:
                    # Calculate scalar gradient norm for adaptive scaling.
                    current_grad_norm = grad.norm().item()

                    # Retrieve previous gradient norm, converting to scalar if necessary.
                    prev_grad_norm = state.get('prev_grad_norm', 0.0)
                    if not isinstance(prev_grad_norm, float):
                        prev_grad_norm = prev_grad_norm.item()

                    # Calculate gradient consistency ratio
                    # If gradients oscillate wildly → reduce LR
                    # If gradients are consistent → maintain/increase LR
                    if prev_grad_norm > 0.0 and current_grad_norm > 0.0:
                        ratio = current_grad_norm / (prev_grad_norm + eps)

                        # Retrieve smoothed gradient variance, converting to scalar if necessary.
                        grad_var = state.get('grad_variance', 1.0)
                        if not isinstance(grad_var, float):
                            grad_var = grad_var.item()

                        # Smooth the ratio with exponential moving average
                        ema = group.get('adaptive_ema', 0.99)
                        grad_var = grad_var * ema + ratio * (1 - ema)

                        # Apply adaptive multiplier with epsilon guard
                        adaptive_mult = 1.0 / (grad_var + eps)
                        adaptive_mult = max(adaptive_clip[0], min(adaptive_clip[1], adaptive_mult))
                        step_size *= adaptive_mult

                        # Update variance tracked state
                        state['grad_variance'] = grad_var
                    
                    # Update previous norm tracked state
                    state['prev_grad_norm'] = current_grad_norm
                
                # --- Parameter Update ---
                denom = corrected_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(corrected_avg, denom, value=-step_size)
                
                # --- Spectral Clip for Core Matrix ---
                if is_core and spectral_clip > 0 and p.dim() == 2:
                    try:
                        # Efficient spectral norm estimation using power iteration
                        # Full SVD is too expensive, we estimate the largest singular value
                        u = torch.randn(p.shape[0], 1, device=p.device)
                        for _ in range(3):  # 3 power iterations
                            v = p.t() @ u
                            v = v / (v.norm() + 1e-12)
                            u = p @ v
                            u = u / (u.norm() + 1e-12)
                        sigma_max = (u.t() @ p @ v).item()
                        self._spectral_radius = abs(sigma_max)
                        
                        if self._spectral_radius > spectral_clip:
                            ratio = spectral_clip / (self._spectral_radius + eps)
                            p.data.mul_(ratio)
                            
                            # Update momentum buffers to maintain topological alignment
                            state['exp_avg'].mul_(ratio)
                            state['exp_avg_sq'].mul_(ratio ** 2)
                    except Exception:
                        pass
                
                # Track gradient norms for diagnostics
                g_norm = grad.norm().item()
                total_grad_norm += g_norm
                total_param_norm += p.norm().item()
                
                # Input sentinel tracking
                if hasattr(self, '_sentinel_params') and id(p) in self._sentinel_params:
                    input_grad_norm += g_norm
        
        # Update diagnostics
        self._diagnostics['total_grad_norm'] = total_grad_norm
        self._diagnostics['total_param_norm'] = total_param_norm
        
        # Input gradient health calculation (detects vanishing inputs)
        if total_grad_norm > 0:
            # Compared against a scaled threshold of total activity
            threshold = self.defaults.get('sentinel_threshold', 0.1)
            self._input_grad_health = min(1.0, input_grad_norm / (total_grad_norm * threshold + 1e-12))
        
        if is_plateau:
            self._diagnostics['plateau_escape_triggered'] = self._global_step
        
        return loss
    
    def set_sentinel_params(self, param_ids):
        """Set parameter IDs to monitor for input gradient health."""
        self._sentinel_params = param_ids


class ChaosGradConfig:
    """
    Pre-built configurations for ChaosGrad optimizer.
    """
    
    @staticmethod
    def default(lr=1e-4):
        """Default balanced configuration."""
        return dict(
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            projection_decay=0.01,
            lightweight_lr_mult=2.0,
            chaos_core_lr_mult=1.0,
            projection_lr_mult=1.0,
            adaptive_lr=True,
            adaptive_ema=0.99,
            grad_centralization=True,
            plateau_noise_intensity=0.1,
            sentinel_threshold=0.1,
            loss_history_min=200,
        )
    
    @staticmethod
    def aggressive(lr=3e-4):
        """Aggressive exploration for fresh/small networks."""
        return dict(
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=0.001,
            projection_decay=0.001,
            lightweight_lr_mult=3.0,
            chaos_core_lr_mult=1.5,
            projection_lr_mult=1.2,
            plateau_patience=50,
            plateau_noise_scale=0.02,
            adaptive_lr=True,
            adaptive_lr_clip=(0.2, 5.0),
            adaptive_ema=0.98,
            grad_centralization=True,
            plateau_noise_intensity=0.2,
            sentinel_threshold=0.1,
            loss_history_min=100,
        )
    
    @staticmethod
    def finetune(lr=1e-5):
        """Conservative fine-tuning configuration."""
        return dict(
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            projection_decay=0.005,
            lightweight_lr_mult=1.0,
            chaos_core_lr_mult=0.5,
            projection_lr_mult=0.8,
            adaptive_lr=True,
            adaptive_lr_clip=(0.5, 2.0),
            adaptive_ema=0.995,
            grad_centralization=False,
            plateau_noise_intensity=0.05,
            sentinel_threshold=0.05,
            loss_history_min=500,
        )
    
    @staticmethod 
    def large_network(lr=1e-4):
        """
        For big networks (1000+ neurons) that tend to develop input-blind local minima.
        Extra monitoring and plateau escape.
        """
        return dict(
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=0.01,
            projection_decay=0.01,
            lightweight_lr_mult=3.0,
            chaos_core_lr_mult=0.8,
            projection_lr_mult=1.5,
            plateau_patience=100,
            plateau_noise_scale=0.01,
            spectral_clip=3.0,
            input_sentinel=True,
            adaptive_lr=True,
            adaptive_lr_clip=(0.1, 5.0),
            adaptive_ema=0.99,
            grad_centralization=True,
            plateau_noise_intensity=0.1,
            sentinel_threshold=0.2,
            loss_history_min=200,
        )
    
    @staticmethod
    def tiny_network(lr=0.01):
        """For tiny networks (<10 neurons) like XOR, Identity."""
        return dict(
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            projection_decay=0.0,
            lightweight_lr_mult=1.0,
            chaos_core_lr_mult=1.0,
            projection_lr_mult=1.0,
            adaptive_lr=False,
            adaptive_ema=0.99,
            grad_centralization=False,
            plateau_noise_intensity=0.01,
            sentinel_threshold=0.1,
            loss_history_min=50,
        )
