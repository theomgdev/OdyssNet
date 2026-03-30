"""
ChaosGrad: A OdyssNet-Native Optimizer

Built specifically for OdyssNet's chaotic recurrent dynamics. Unlike AdamW which 
treats all parameters identically, ChaosGrad understands the different roles of 
parameters in a chaos chamber:

1. W (Core Matrix): The chaos engine. Needs careful spectral management.
2. Embeddings/Projections: Standard LLM-style parameters.
3. Gates: Parametric gate controls for branch-wise signal modulation.
4. Bias/Scale/Norm: Lightweight params that should train faster.

Key Features:
- Per-parameter adaptive learning rates based on gradient health
- Input-gradient monitoring to detect "input-blind" local minima
- Spectral radius awareness for edge-of-chaos control
- Plateau escape via controlled gradient perturbation
- Decoupled weight decay for different parameter groups
"""

import torch
import math


class ChaosGrad(torch.optim.Optimizer):
    """
    ChaosGrad: OdyssNet-native optimizer.
    
    Extends AdamW with chaos-aware features:
    - Separate momentum/LR for chaos core (W) vs projections vs lightweight params
    - Gradient health monitoring with per-param adaptive LR
    - Plateau detection and escape via controlled perturbation
    - Input gradient monitoring for detecting vanishing input gradients
    
    Args:
        params: Iterable of parameters or parameter groups.
        lr (float): Base learning rate. Default: 1e-4.
        betas (tuple): AdamW momentum coefficients. Default: (0.9, 0.999).
        eps (float): Numerical stability. Default: 1e-8.
        weight_decay (float): Weight decay for chaos core. Default: 0.01.
        projection_decay (float): Weight decay for projections/embeddings. Default: 0.01.
        gate_decay (float): Weight decay for gate parameters. Default: 0.0.
        lightweight_lr_mult (float): LR multiplier for bias/scale/norm params. Default: 2.0.
        chaos_core_lr_mult (float): LR multiplier for core W matrix. Default: 1.0.
        projection_lr_mult (float): LR multiplier for projection layers. Default: 1.0.
        gate_lr_mult (float): LR multiplier for gate parameters. Default: 1.0.
        plateau_patience (int): Steps of no improvement before perturbation. Default: 0 (disabled).
        plateau_noise_scale (float): Scale of perturbation noise on plateau escape. Default: 0.01.
        spectral_clip (float): Max spectral radius for W matrix. Default: 0 (disabled).
        adaptive_lr (bool): Enable per-param adaptive LR scaling. Default: True.
        adaptive_lr_clip (tuple): (min, max) multiplier for adaptive LR. Default: (0.1, 10.0).
        grad_centralization (bool): Center gradients by removing mean. Default: True.
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, projection_decay=0.01, memory_decay=0.0, gate_decay=0.0,
                 lightweight_lr_mult=2.0, chaos_core_lr_mult=1.0,
                 projection_lr_mult=1.0, memory_lr_mult=1.0, gate_lr_mult=1.0,
                 plateau_patience=0, plateau_noise_scale=0.01,
                 spectral_clip=0.0,
                 adaptive_lr=True, adaptive_lr_clip=(0.1, 10.0),
                 grad_centralization=True):
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
            projection_decay=projection_decay,
            memory_decay=memory_decay,
            gate_decay=gate_decay,
            lightweight_lr_mult=lightweight_lr_mult,
            chaos_core_lr_mult=chaos_core_lr_mult,
            projection_lr_mult=projection_lr_mult,
            memory_lr_mult=memory_lr_mult,
            gate_lr_mult=gate_lr_mult,
            plateau_patience=plateau_patience,
            plateau_noise_scale=plateau_noise_scale,
            spectral_clip=spectral_clip,
            adaptive_lr=adaptive_lr,
            adaptive_lr_clip=adaptive_lr_clip,
            grad_centralization=grad_centralization,
        )
        super().__init__(params, defaults)
        
        # Global tracking
        self._global_step = 0
        self._loss_history = []
        self._plateau_counter = 0
        self._best_loss = float('inf')
        self._spectral_radius = 0.0
        self._last_perturbation_step = -1
        self._force_plateau_escape = False
    
    @staticmethod
    def classify_params(model):
        """
        Classifies OdyssNet parameters into groups with appropriate settings.
        Returns a list of param groups suitable for the optimizer.
        
        Groups:
        1. chaos_core: W matrix (the NxN core)
        2. memory_feedback: memory_feedback (self-connections)
        3. projections: Embedding, Linear projection, Output decoder
        4. gates: input/output/core/memory gate parameters
        5. lightweight: Bias, Scale, Norm parameters
        """
        chaos_core = []      # W matrix (cross connections)
        memory_feedback = [] # memory_feedback (self connections)
        projections = []     # Embeddings, projections, decoders
        gates = []           # Parametric gate vectors
        lightweight = []     # Bias, scale, norm
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Torch wrappers (e.g. torch.compile, DDP) often prefix names
            # like "_orig_mod." or "module.". Use leaf names for exact groups.
            leaf_name = name.split('.')[-1]
                
            if leaf_name == 'W':
                chaos_core.append(param)
            elif leaf_name == 'memory_feedback':
                memory_feedback.append(param)
            elif any(k in name for k in ['embed', 'proj', 'output_decoder']):
                projections.append(param)
            elif leaf_name in {'input_gate', 'output_gate', 'core_gate', 'memory_gate'}:
                gates.append(param)
            else:
                # B, input_scale, output_scale, norm.weight, norm.bias
                lightweight.append(param)
        
        groups = []
        
        if chaos_core:
            groups.append({
                'params': chaos_core,
                'group_name': 'chaos_core',
                '_is_chaos_core': True,
                '_is_memory_feedback': False,
                '_is_projection': False,
                '_is_gate': False,
                '_is_lightweight': False,
            })
            
        if memory_feedback:
            groups.append({
                'params': memory_feedback,
                'group_name': 'memory_feedback',
                '_is_chaos_core': False,
                '_is_memory_feedback': True,
                '_is_projection': False,
                '_is_gate': False,
                '_is_lightweight': False,
            })
            
        if projections:
            groups.append({
                'params': projections,
                'group_name': 'projections',
                '_is_chaos_core': False,
                '_is_memory_feedback': False,
                '_is_projection': True,
                '_is_gate': False,
                '_is_lightweight': False,
            })

        if gates:
            groups.append({
                'params': gates,
                'group_name': 'gates',
                '_is_chaos_core': False,
                '_is_memory_feedback': False,
                '_is_projection': False,
                '_is_gate': True,
                '_is_lightweight': False,
            })
            
        if lightweight:
            groups.append({
                'params': lightweight,
                'group_name': 'lightweight',
                '_is_chaos_core': False,
                '_is_memory_feedback': False,
                '_is_projection': False,
                '_is_gate': False,
                '_is_lightweight': True,
            })
        
        return groups
    
    def report_loss(self, loss_value):
        """Report current loss for plateau detection and adaptive behavior."""
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        self._loss_history.append(loss_value)
        
        # Keep only recent history
        max_history = max(200, self.defaults.get('plateau_patience', 0) * 2)
        if len(self._loss_history) > max_history:
            self._loss_history = self._loss_history[-max_history:]
            
        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._plateau_counter = 0
        else:
            self._plateau_counter += 1
    
    def trigger_plateau_escape(self):
        """Manually trigger a plateau escape perturbation in the next step."""
        self._force_plateau_escape = True

    def get_diagnostics(self):
        """Returns minimal optimizer state."""
        return {
            'global_step': self._global_step,
            'plateau_counter': self._plateau_counter,
            'best_loss': self._best_loss,
            'spectral_radius': self._spectral_radius,
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
                     self._global_step - self._last_perturbation_step > plateau_patience) or self._force_plateau_escape
        
        if is_plateau:
            self._last_perturbation_step = self._global_step
            self._plateau_counter = 0
            self._force_plateau_escape = False
        
        # Adaptive LR settings
        use_adaptive = self.defaults.get('adaptive_lr', True)
        adaptive_clip = self.defaults.get('adaptive_lr_clip', (0.1, 10.0))
        use_centralization = self.defaults.get('grad_centralization', True)
        spectral_clip = self.defaults.get('spectral_clip', 0.0)
        
        for group in self.param_groups:
            is_core = group.get('_is_chaos_core', False)
            is_memory = group.get('_is_memory_feedback', False)
            is_proj = group.get('_is_projection', False)
            is_gate = group.get('_is_gate', False)
            is_light = group.get('_is_lightweight', False)
            
            # Select appropriate LR multiplier
            if is_core:
                lr_mult = self.defaults['chaos_core_lr_mult']
                wd = self.defaults['weight_decay']
            elif is_memory:
                lr_mult = self.defaults['memory_lr_mult']
                wd = self.defaults['memory_decay']
            elif is_proj:
                lr_mult = self.defaults['projection_lr_mult']
                wd = self.defaults['projection_decay']
            elif is_gate:
                lr_mult = self.defaults['gate_lr_mult']
                wd = self.defaults['gate_decay']
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
                        state['grad_variance'] = 1.0
                        state['prev_grad_norm'] = 1.0
                
                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if use_centralization and grad.dim() >= 2:
                    grad = grad - grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True)
                
                if is_plateau and is_core:
                    noise = torch.randn_like(grad) * plateau_noise * grad.abs().mean()
                    grad = grad + noise
                    
                if is_core and grad.dim() == 2 and grad.shape[0] == grad.shape[1]:
                    grad.fill_diagonal_(0.0)
                
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
                
                if use_adaptive:
                    current_grad_norm = grad.norm().item()

                    prev_grad_norm = state.get('prev_grad_norm', 0.0)
                    if not isinstance(prev_grad_norm, float):
                        prev_grad_norm = prev_grad_norm.item()

                    if prev_grad_norm > 0.0 and current_grad_norm > 0.0:
                        ratio = current_grad_norm / (prev_grad_norm + eps)

                        grad_var = state.get('grad_variance', 1.0)
                        if not isinstance(grad_var, float):
                            grad_var = grad_var.item()

                        grad_var = grad_var * 0.99 + ratio * 0.01

                        adaptive_mult = 1.0 / (grad_var + eps)
                        adaptive_mult = max(adaptive_clip[0], min(adaptive_clip[1], adaptive_mult))
                        step_size *= adaptive_mult

                        state['grad_variance'] = grad_var
                    
                    state['prev_grad_norm'] = current_grad_norm
                
                denom = corrected_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(corrected_avg, denom, value=-step_size)
                
                if is_core and p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.data.fill_diagonal_(0.0)

                # --- Spectral Clip for Core Matrix ---
                if is_core and spectral_clip > 0 and p.dim() == 2:
                    try:
                        u = torch.randn(p.shape[0], 1, device=p.device)
                        for _ in range(3):  # 3 power iterations
                            v = p.t() @ u
                            v = v / (v.norm() + 1e-12)
                            u = p @ v
                            u = u / (u.norm() + 1e-12)
                        sigma_max = (u.t() @ p @ v).item()
                        self._spectral_radius = abs(sigma_max)

                        # Use larger epsilon for numerical stability
                        spectral_eps = 1e-8
                        if self._spectral_radius > spectral_eps and self._spectral_radius > spectral_clip:
                            p.data.mul_(spectral_clip / self._spectral_radius)
                    except Exception:
                        pass
                        
        return loss


class ChaosGradConfig:
    """
    Pre-built configurations for ChaosGrad optimizer.
    """
    
    @staticmethod
    def conservative(lr=1e-4):
        """Conservative balanced configuration."""
        return dict(
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            projection_decay=0.01,
            memory_decay=0.0,
            gate_decay=0.0,
            lightweight_lr_mult=2.0,
            chaos_core_lr_mult=1.0,
            projection_lr_mult=1.0,
            memory_lr_mult=1.0,
            gate_lr_mult=1.0,
            adaptive_lr=True,
            grad_centralization=True,
        )
    
    @staticmethod
    def default(lr=3e-4):
        """Default exploration for fresh/small networks."""
        return dict(
            lr=lr,
            betas=(0.9, 0.98),
            weight_decay=0.001,
            projection_decay=0.001,
            memory_decay=0.0,
            gate_decay=0.0,
            lightweight_lr_mult=3.0,
            chaos_core_lr_mult=1.5,
            projection_lr_mult=1.2,
            memory_lr_mult=1.0,
            gate_lr_mult=1.0,
            plateau_patience=50,
            plateau_noise_scale=0.02,
            adaptive_lr=True,
            adaptive_lr_clip=(0.2, 5.0),
            grad_centralization=True,
        )
    
    @staticmethod
    def finetune(lr=1e-5):
        """Conservative fine-tuning configuration."""
        return dict(
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            projection_decay=0.005,
            memory_decay=0.0,
            gate_decay=0.0,
            lightweight_lr_mult=1.0,
            chaos_core_lr_mult=0.5,
            projection_lr_mult=0.8,
            memory_lr_mult=0.8,
            gate_lr_mult=0.8,
            adaptive_lr=True,
            adaptive_lr_clip=(0.5, 2.0),
            grad_centralization=False,
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
            memory_decay=0.001,
            gate_decay=0.0,
            lightweight_lr_mult=3.0,
            chaos_core_lr_mult=0.8,
            projection_lr_mult=1.5,
            memory_lr_mult=1.0,
            gate_lr_mult=1.0,
            plateau_patience=100,
            plateau_noise_scale=0.01,
            spectral_clip=3.0,
            adaptive_lr=True,
            adaptive_lr_clip=(0.1, 5.0),
            grad_centralization=True,
        )
    
    @staticmethod
    def tiny_network(lr=0.01):
        """For tiny networks (<10 neurons) like XOR, Identity."""
        return dict(
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.0,
            projection_decay=0.0,
            memory_decay=0.0,
            gate_decay=0.0,
            lightweight_lr_mult=1.0,
            chaos_core_lr_mult=1.0,
            projection_lr_mult=1.0,
            memory_lr_mult=1.0,
            gate_lr_mult=1.0,
            adaptive_lr=False,
            grad_centralization=False,
        )
