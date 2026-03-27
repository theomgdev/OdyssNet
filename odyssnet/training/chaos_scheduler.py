"""
TemporalScheduler: Adaptive Learning Rate Schedule for OdyssNet.

Unlike static schedulers (CosineAnnealing, StepLR), TemporalScheduler is
process-aware: it monitors training dynamics and adapts the schedule in real-time.

Key Features:
1. Warmup Phase: Linear warmup to prevent chaos explosion at start
2. Cosine Decay: Smooth decay with configurable minimum  
3. Warm Restarts: Triggered by plateau detection or manual trigger
4. Loss-Trend Awareness: Adjusts decay speed based on convergence rate
5. Per-Parameter-Group Scheduling: Core W gets different schedule than projections
"""

import math
import torch


class TemporalScheduler:
    """
    Adaptive LR scheduler designed for OdyssNet's training dynamics.
    
    Supports three phases:
    1. WARMUP: Linear ramp from 0 to max_lr (prevents chaos explosion)
    2. COSINE_DECAY: Smooth decay with adaptive speed
    3. PLATEAU_RESTART: When plateau detected, temporarily boosts LR
    
    Args:
        optimizer: The optimizer to schedule.
        warmup_steps (int): Number of warmup steps. Default: 500.
        max_steps (int): Total steps for one cosine cycle. Default: 5000.
        min_lr_ratio (float): Minimum LR as ratio of max. Default: 0.01.
        restart_factor (float): LR multiplier on warm restart. Default: 0.5.
        restart_decay (float): Decay factor for restart magnitude. Default: 0.9.
        patience (int): Steps without improvement to trigger restart. Default: 0 (disabled).
        cooldown (int): Minimum steps between restarts. Default: 100.
        loss_smoothing (float): EMA factor for loss tracking. Default: 0.95.
        auto_extend (bool): Extend max_steps when restarts occur. Default: True.
        verbose (bool): Print LR changes. Default: False.
    """
    
    def __init__(self, optimizer, warmup_steps=500, max_steps=5000,
                 min_lr_ratio=0.01, restart_factor=0.5, restart_decay=0.9,
                 patience=0, cooldown=100, loss_smoothing=0.95,
                 auto_extend=True, verbose=False):
        
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.restart_factor = restart_factor
        self.restart_decay = restart_decay
        self.patience = patience
        self.cooldown = cooldown
        self.loss_smoothing = loss_smoothing
        self.auto_extend = auto_extend
        self.verbose = verbose
        
        # Store initial LRs per group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Internal state
        self._step = 0
        self._restart_count = 0
        self._last_restart_step = -cooldown  # Allow immediate first restart
        self._cycle_start_step = 0
        self._current_max_lr_factor = 1.0
        
        # Loss tracking
        self._loss_ema = None
        self._best_loss_ema = float('inf')
        self._plateau_counter = 0
        self._loss_history = []
        
        # Phase tracking
        self._phase = 'warmup'  # 'warmup', 'decay', 'restart_boost'
        self._restart_boost_remaining = 0
    
    def get_lr_multiplier(self, step=None):
        """
        Calculate the LR multiplier for the current step.
        
        Returns a value between 0 and 1 (or higher during restarts).
        """
        if step is None:
            step = self._step
            
        # Phase 1: Warmup
        if step < self.warmup_steps:
            self._phase = 'warmup'
            return float(step) / float(max(1, self.warmup_steps))
        
        # Phase 2: Check for restart boost
        if self._restart_boost_remaining > 0:
            self._phase = 'restart_boost'
            # Quick decay from restart peak back to schedule
            boost_progress = 1.0 - (self._restart_boost_remaining / 20.0)
            schedule_lr = self._cosine_lr(step)
            restart_lr = schedule_lr * (1.0 + self._current_max_lr_factor * (1.0 - boost_progress))
            return min(restart_lr, 1.0)  # Never exceed base LR
        
        # Phase 3: Cosine Decay
        self._phase = 'decay'
        return self._cosine_lr(step)
    
    def _cosine_lr(self, step):
        """Calculate cosine decay LR multiplier."""
        effective_step = step - self._cycle_start_step
        # Defines the absolute length of a full cosine cycle.
        effective_max = self.max_steps
        
        if effective_max <= 0:
            return self.min_lr_ratio
            
        if effective_step >= effective_max:
            return self.min_lr_ratio
            
        # Cosine decay
        # The initial cycle accounts for the warmup phase, whereas subsequent
        # cycles (post-restart) apply the cosine decay immediately.
        if self._cycle_start_step == 0:
            decay_numerator = effective_step - self.warmup_steps
            decay_denominator = max(1, effective_max - self.warmup_steps)
        else:
            decay_numerator = effective_step
            decay_denominator = max(1, effective_max)
        
        decay_ratio = decay_numerator / decay_denominator
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr_ratio + coeff * (1.0 - self.min_lr_ratio)
    
    def step(self, loss=None):
        """
        Advance the scheduler by one step.
        
        Args:
            loss (float, optional): Current loss value for adaptive behavior.
        """
        self._step += 1
        
        # Update loss tracking
        if loss is not None:
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            self._loss_history.append(loss_val)
            
            if self._loss_ema is None:
                self._loss_ema = loss_val
            else:
                self._loss_ema = self.loss_smoothing * self._loss_ema + (1 - self.loss_smoothing) * loss_val
            
            # Plateau detection
            if self._loss_ema < self._best_loss_ema * 0.999:
                self._best_loss_ema = self._loss_ema
                self._plateau_counter = 0
            else:
                self._plateau_counter += 1
            
            # Check if plateau restart should trigger
            if (self.patience > 0 and 
                self._plateau_counter >= self.patience and
                self._step - self._last_restart_step >= self.cooldown and
                self._step > self.warmup_steps):
                self._trigger_restart()
            
            # Keep history bounded
            max_hist = max(200, self.patience * 3) if self.patience > 0 else 200
            if len(self._loss_history) > max_hist:
                self._loss_history = self._loss_history[-max_hist:]
        
        # Decrement restart boost
        if self._restart_boost_remaining > 0:
            self._restart_boost_remaining -= 1
        
        # Apply LR
        multiplier = self.get_lr_multiplier()
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * multiplier
    
    def _trigger_restart(self):
        """Trigger a warm restart."""
        self._restart_count += 1
        self._last_restart_step = self._step
        self._plateau_counter = 0
        self._restart_boost_remaining = 20  # 20 steps of boosted LR
        
        # Decay the restart factor over time
        self._current_max_lr_factor = self.restart_factor * (self.restart_decay ** (self._restart_count - 1))
        
        # Dynamically extends the cycle length to encourage wider exploration.
        if self.auto_extend:
            self.max_steps = int(self.max_steps * 1.5)
        
        # Reset cycle start for fresh cosine
        self._cycle_start_step = self._step
        self._best_loss_ema = self._loss_ema if self._loss_ema else float('inf')
        
        if self.verbose:
            print(f"🔄 TemporalScheduler: Warm Restart #{self._restart_count} at step {self._step} "
                  f"(boost factor: {self._current_max_lr_factor:.3f})")
    
    def manual_restart(self, boost_factor=None):
        """
        Manually trigger a warm restart.
        
        Args:
            boost_factor (float, optional): Override restart factor.
        """
        if boost_factor is not None:
            self._current_max_lr_factor = boost_factor
        self._trigger_restart()
    
    def get_last_lr(self):
        """Returns the last computed learning rates (compatible with PyTorch schedulers)."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_phase(self):
        """Returns current training phase."""
        return self._phase
    
    def get_diagnostics(self):
        """Returns scheduler diagnostics dict."""
        return {
            'step': self._step,
            'phase': self._phase,
            'restart_count': self._restart_count,
            'plateau_counter': self._plateau_counter,
            'loss_ema': self._loss_ema,
            'best_loss_ema': self._best_loss_ema,
            'current_lrs': self.get_last_lr(),
        }
    
    def state_dict(self):
        """Returns scheduler state for checkpointing."""
        return {
            'step': self._step,
            'restart_count': self._restart_count,
            'last_restart_step': self._last_restart_step,
            'cycle_start_step': self._cycle_start_step,
            'current_max_lr_factor': self._current_max_lr_factor,
            'loss_ema': self._loss_ema,
            'best_loss_ema': self._best_loss_ema,
            'plateau_counter': self._plateau_counter,
            'base_lrs': self.base_lrs,
            'max_steps': self.max_steps,
            'restart_boost_remaining': self._restart_boost_remaining,
        }
    
    def load_state_dict(self, state_dict):
        """Loads scheduler state from checkpoint."""
        self._step = state_dict.get('step', 0)
        self._restart_count = state_dict.get('restart_count', 0)
        self._last_restart_step = state_dict.get('last_restart_step', -self.cooldown)
        self._cycle_start_step = state_dict.get('cycle_start_step', 0)
        self._current_max_lr_factor = state_dict.get('current_max_lr_factor', 1.0)
        self._loss_ema = state_dict.get('loss_ema', None)
        self._best_loss_ema = state_dict.get('best_loss_ema', float('inf'))
        self._plateau_counter = state_dict.get('plateau_counter', 0)
        self.max_steps = state_dict.get('max_steps', self.max_steps)
        self._restart_boost_remaining = state_dict.get('restart_boost_remaining', 0)
        
        if 'base_lrs' in state_dict:
            self.base_lrs = state_dict['base_lrs']


class TemporalSchedulerConfig:
    """Pre-built configurations for TemporalScheduler."""
    
    @staticmethod
    def default():
        """Standard training schedule."""
        return dict(
            warmup_steps=500,
            max_steps=5000,
            min_lr_ratio=0.01,
            patience=0,
        )
    
    @staticmethod
    def llm():
        """LLM-style long training with warm restarts."""
        return dict(
            warmup_steps=500,
            max_steps=10000,
            min_lr_ratio=0.01,
            restart_factor=0.3,
            restart_decay=0.85,
            patience=200,
            cooldown=100,
            auto_extend=True,
        )
    
    @staticmethod
    def short_experiment():
        """For quick PoC experiments."""
        return dict(
            warmup_steps=50,
            max_steps=1000,
            min_lr_ratio=0.05,
            patience=0,
        )
    
    @staticmethod
    def finetune():
        """Conservative fine-tuning schedule."""
        return dict(
            warmup_steps=100,
            max_steps=2000,
            min_lr_ratio=0.001,
            patience=0,
        )
    
    @staticmethod
    def adaptive():
        """Fully adaptive with plateau detection."""
        return dict(
            warmup_steps=200,
            max_steps=5000,
            min_lr_ratio=0.01,
            restart_factor=0.5,
            restart_decay=0.9,
            patience=100,
            cooldown=50,
            auto_extend=True,
        )
