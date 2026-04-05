"""
ChaosGrad — Analytic Hypergradient Optimizer for OdyssNet

A zero-hyperparameter optimizer that autonomously adapts every learning
mechanism via Analytic Hypergradient Descent. Instead of developer-specified
constants (learning rates, betas, weight decay, plateau patience), all
meta-parameters are computed analytically from the intrinsic geometry of the
loss landscape at runtime.

Core principles:
- Cold-start calibration: per_lr_0 = min(1/g_rms, SIZE_CAP_SCALE·√numel), so the first
  effective step is normalized across all parameter tensors and bounded proportionally
  to tensor size. Large tensors are unaffected; small tensors receive a conservative ceiling.
- Confidence-weighted drive: all cosine signals are transformed as s = cos·|cos| before
  use. Strong signal (cos=0.9) acts at 81% of maximum; weak signal (cos=0.3) at 9%.
  The optimizer acts decisively on clear roads and conservatively in fog.
- Per-parameter LR    ← conf(∇_t,∇_{t-1}) + restore(→init_lr) + couple(LR×β step bound)
- Per-parameter beta  ← conf(∇_t,V_{t-1}) + restore(→0.9)  + couple(β×LR, symmetric)
- Per-parameter decay ← conf(∇_t,W_{t-1}) + restore(→seed/lr) [lr×decay product coupling]
- Centralization α    ← conf(noise signal) + restore(→0.5)   [independent]
- Frustration         ← loss stagnation → burst scaled to init_lr + meta-param reset toward init_lr

The only external scalar is genesis_lr: a mathematical starting point,
not a hyperparameter to tune. Set it near the rough scale of the problem
(default 1e-3) and let the system adapt from there.

State per parameter:
    step (int)                  — update count
    init_lr  (float)            — calibrated starting LR = 1/g_rms at T=0 (fixed, never updated)
    prev_grad (float32 tensor) — previous gradient
    momentum  (float32 tensor)  — exponential gradient accumulator
    per_param_lr    (float)     — autonomous LR multiplier (starts at init_lr; restore/couple
                                  reference this initial value, not 1.0)
    per_param_decay (float)     — autonomous weight decay rate (group-seeded, lr-coupled)
    per_param_beta  (float)     — autonomous momentum coefficient (init 0.9)
    per_param_alpha (float)     — autonomous centralization gate (init 0.5)
"""

import torch
import math


import torch.nn.functional as F

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Cosine similarity immune to the Curse of Dimensionality.
    Instead of flattening large tensors (which drives cosine to 0),
    it computes similarity locally along the last dimension (e.g., per-neuron)
    and averages the consensus.
    """
    if a.dim() == 0:
        return 0.0
        
    # dim=-1 ensures that for a 2D matrix (N, M), we get N similarities.
    # For a 1D vector (like bias), we get a single global similarity.
    # eps=1e-12 prevents division by zero for dead/silent neurons.
    sim = F.cosine_similarity(a, b, dim=-1, eps=1e-12)
    
    # Return the collective consensus of the similarities
    return sim.mean().item()


def _conf_signal(cos: float) -> float:
    """
    Confidence-weighted directional signal: s = cos · |cos| = sign(cos) · cos².

    Transforms raw cosine similarity so that the adaptation step scales with
    signal confidence. A driver on a clear road acts decisively; in fog, cautiously.

    cos=±1.0 → s=±1.0 (full action)   cos=±0.5 → s=±0.25 (quarter action)
    cos= 0.0 → s= 0.0 (no action)     cos=±0.3 → s=±0.09 (near-idle)
    """
    return cos * abs(cos)


class ChaosGrad(torch.optim.Optimizer):
    """
    Zero-hyperparameter OdyssNet optimizer.

    Every learning meta-parameter (lr multiplier, weight decay, momentum beta,
    gradient centralization gate) adapts autonomously via cosine-similarity-based
    hypergradient descent.

    Args:
        params: Iterable of parameters or classified parameter groups (from
                classify_params). Providing classified groups enables
                group-specific decay seeding and the Hebbian bypass rule.
        lr (float): Genesis learning rate — the single mathematical starting
                    point. Default: 1e-3. This is NOT a tunable hyperparameter;
                    it sets the initial update scale before autonomous
                    adaptation takes over.
    """

    # ------------------------------------------------------------------ #
    # Fixed meta-constants (not configurable by design)                   #
    # ------------------------------------------------------------------ #
    # ---- Driving forces (hypergradient step size per unit of cosine signal) ----
    _ETA_LR    = 0.05    # LR: log-scale step (proportional acceleration/deceleration)
    _ETA_MOM   = 0.02    # Momentum beta: additive step
    _ETA_DECAY = 0.002   # Weight decay: additive step
    _ETA_ALPHA = 0.02    # Centralization gate: additive step

    # ---- Restoring forces (spring constants toward equilibrium) ----
    _RESTORE_LR    = 0.02   # per_lr   → init_lr       (log-space spring)
    _RESTORE_BETA  = 0.10   # per_beta → 0.9          (linear spring)
    _RESTORE_ALPHA = 0.05   # per_alpha → 0.5         (linear spring)
    _RESTORE_DECAY = 0.10   # per_decay → seed/per_lr (product-coupled linear spring)

    # ---- Cross-couplings (prevent effective step amplitude from diverging) ----
    # Derived from restore constants — not independently tunable.
    # Effective step = per_lr / (1 - per_beta); neutral = init_lr / (10 * 0.1) = init_lr.
    _COUPLE_LR_BETA  = _RESTORE_LR   / 2     # = 0.01
    _COUPLE_BETA_LR  = _RESTORE_BETA / 10    # = 0.01

    # ---- Safety bounds (emergency rails, not primary constraints) ----
    _LR_MIN    = 0.01
    _LR_MAX    = 100.0
    _BETA_MIN  = 0.5
    _BETA_MAX  = 0.999
    _DECAY_MAX = 0.1

    # ---- Frustration Accumulator ----
    _FRUST_DECAY      = 0.995   # EMA decay rate
    _FRUST_THRESH     = 0.75    # Threshold → burst
    _FRUST_NOISE      = 0.01    # Momentum noise scale (relative to genesis_lr)
    _FRUST_META_RESET = 0.30    # Fraction to pull meta-params toward neutral on burst

    _GENESIS_SCALAR  = 1e-6  # Cold-start prev_grad seed scale
    _SIZE_CAP_SCALE  = 2.0   # init_lr ceiling = SIZE_CAP_SCALE × √numel; transparent for large
                              # tensors, conservative for small ones where every parameter matters

    # Initial per-group decay seeds (autonomous adaptation starts here)
    _INIT_DECAY = {
        'chaos_core':      0.01,
        'memory_feedback': 0.0,
        'projections':     0.01,
        'gates':           0.0,
        'hebbian':         0.0,
        'lightweight':     0.0,
    }

    def __init__(self, params, lr: float = 1e-3):
        if lr <= 0.0:
            raise ValueError(f"Genesis lr must be positive, got {lr}")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        self._global_step = 0
        self._frustration = 0.0    # Global Frustration Accumulator
        self._best_loss   = float('inf')
        self._force_plateau_escape = False

    # ------------------------------------------------------------------ #
    # Parameter classification                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_params(model):
        """
        Classifies OdyssNet parameters into semantically distinct groups.

        Groups determine the initial decay seed and enforce the Hebbian
        bypass rule (Hebbian logits must never receive autonomous weight
        decay because they govern unbounded temporal working-memory logits).

        Returns a list of param-group dicts suitable for ChaosGrad.__init__.
        """
        chaos_core      = []
        memory_feedback = []
        projections     = []
        gates           = []
        hebbian         = []
        lightweight     = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            leaf = name.split('.')[-1]

            if leaf == 'W':
                chaos_core.append(param)
            elif leaf == 'memory_feedback':
                memory_feedback.append(param)
            elif any(k in name for k in ('embed', 'proj', 'output_decoder')):
                projections.append(param)
            elif leaf in {'input_gate', 'output_gate', 'core_gate', 'memory_gate'}:
                gates.append(param)
            elif leaf in {'hebb_factor', 'hebb_decay'}:
                hebbian.append(param)
            else:
                lightweight.append(param)

        _flags = {
            '_is_chaos_core': False, '_is_memory_feedback': False,
            '_is_projection': False, '_is_gate': False,
            '_is_hebbian':    False, '_is_lightweight': False,
        }

        groups = []
        if chaos_core:
            groups.append({**_flags, 'params': chaos_core,
                           'group_name': 'chaos_core', '_is_chaos_core': True})
        if memory_feedback:
            groups.append({**_flags, 'params': memory_feedback,
                           'group_name': 'memory_feedback', '_is_memory_feedback': True})
        if projections:
            groups.append({**_flags, 'params': projections,
                           'group_name': 'projections', '_is_projection': True})
        if gates:
            groups.append({**_flags, 'params': gates,
                           'group_name': 'gates', '_is_gate': True})
        if hebbian:
            groups.append({**_flags, 'params': hebbian,
                           'group_name': 'hebbian', '_is_hebbian': True})
        if lightweight:
            groups.append({**_flags, 'params': lightweight,
                           'group_name': 'lightweight', '_is_lightweight': True})
        return groups

    # ------------------------------------------------------------------ #
    # Frustration Accumulator interface                                   #
    # ------------------------------------------------------------------ #

    def report_loss(self, loss_value):
        """
        Feed the current loss to the Frustration Accumulator.

        Frustration grows when loss fails to improve. When frustration
        exceeds the internal threshold, the next step injects a momentum
        noise burst across all parameters to escape local minima and
        shallow attractors.
        """
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()

        if loss_value < self._best_loss * 0.9999:
            self._best_loss = loss_value
            frustration_signal = 0.0
        else:
            # Loss stagnating or worsening relative to best — accumulate frustration.
            frustration_signal = min(1.0, loss_value / (self._best_loss + 1e-10))

        self._frustration = (
            self._frustration * self._FRUST_DECAY
            + frustration_signal * (1.0 - self._FRUST_DECAY)
        )

    def trigger_plateau_escape(self):
        """Manually force a frustration noise burst on the next step."""
        self._force_plateau_escape = True

    def get_diagnostics(self):
        """
        Returns optimizer health metrics.

        avg_effective_lr: mean of (per_param_lr / init_lr) — how much adaptation
            has moved each parameter's LR relative to its calibrated starting point.
            1.0 = no drift from cold start; >1 = training is going well; <1 = struggling.
        avg_init_lr: mean of init_lr across parameters — reflects the gradient scale
            of the network at cold start. Useful for detecting misconfigured genesis_lr.
        """
        total_drift = 0.0
        total_init  = 0.0
        count = 0
        for group in self.param_groups:
            genesis_lr = group.get('lr', self.defaults['lr'])
            for p in group['params']:
                s = self.state.get(p)
                if s:
                    init_lr = s.get('init_lr', s.get('per_param_lr', 1.0))
                    per_lr  = s.get('per_param_lr', init_lr)
                    total_drift += per_lr / max(init_lr, 1e-9)
                    total_init  += init_lr * genesis_lr
                    count += 1
        n = count if count > 0 else 1
        return {
            'global_step':       self._global_step,
            'frustration':       self._frustration,
            'best_loss':         self._best_loss,
            'avg_effective_lr':  total_drift / n,
            'avg_init_lr':       total_init  / n,
        }

    # ------------------------------------------------------------------ #
    # Optimization step                                                   #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single autonomous optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        # Determine whether to fire a frustration noise burst this step.
        do_burst = (
            self._frustration > self._FRUST_THRESH
            or self._force_plateau_escape
        )
        if do_burst:
            self._frustration *= 0.3   # Partial reset after burst
            self._force_plateau_escape = False

        for group in self.param_groups:
            is_core    = group.get('_is_chaos_core',      False)
            is_hebbian = group.get('_is_hebbian',         False)
            group_name = group.get('group_name',          'unknown')
            init_decay = self._INIT_DECAY.get(group_name, 0.0)

            # group['lr'] is the genesis_lr — ChaosGrad derives per-param rates from it.
            genesis_lr = group.get('lr', self.defaults['lr'])

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError('ChaosGrad does not support sparse gradients.')

                state = self.state[p]

                # -------------------------------------------------------- #
                # Cold start (T=0): calibrate from the first gradient.     #
                # -------------------------------------------------------- #
                if len(state) == 0:
                    # Calibrate per_lr so genesis_lr * per_lr * g_rms ≈ genesis_lr:
                    # the first effective step is normalized across all parameter tensors.
                    g_rms = (g.float() ** 2).mean().sqrt().item()
                    # Size cap: √numel grows with tensor area, so small tensors (W of a
                    # 3-neuron net has numel=9 → cap=6) receive a conservative ceiling while
                    # large tensors (numel=630k → cap≈1588 >> LR_MAX) are fully unaffected.
                    size_cap = self._SIZE_CAP_SCALE * math.sqrt(p.numel())
                    init_lr = max(self._LR_MIN, min(self._LR_MAX, size_cap,
                                  1.0 / max(g_rms, 1e-8)))
                    # Decay seeded so that the lr×decay product starts at the group seed.
                    init_decay_calibrated = init_decay / max(init_lr, 0.1)

                    state['step']           = 0
                    state['init_lr']        = init_lr   # fixed reference; never mutated
                    state['prev_grad']      = (g.detach() * self._GENESIS_SCALAR)
                    state['momentum']       = torch.zeros_like(g, dtype=torch.float32)
                    state['per_param_lr']   = init_lr
                    state['per_param_decay']= float(init_decay_calibrated)
                    state['per_param_beta'] = 0.9
                    state['per_param_alpha']= 0.5

                state['step'] += 1

                g_f       = g.float()
                prev_g    = state['prev_grad'].float()
                v         = state['momentum']
                init_lr   = state['init_lr']
                per_lr    = state['per_param_lr']
                per_decay = state['per_param_decay']
                per_beta  = state['per_param_beta']
                per_alpha = state['per_param_alpha']

                # Drift of per_lr relative to its calibrated starting point.
                # Zero at cold start; non-zero only when training signal has moved per_lr.
                # All restore and coupling terms reference this, so they are neutral
                # at T=0 and only activate as adaptation occurs.
                _log_drift = math.log(max(per_lr / init_lr, 1e-9))

                # -------------------------------------------------------- #
                # 1. Hypergradient signals                                  #
                #    Raw cosine similarities are transformed via            #
                #    s = cos·|cos| before driving any meta-parameter.      #
                #    This weights the adaptation step by signal confidence: #
                #    strong signal → act decisively; weak → act sparingly. #
                # -------------------------------------------------------- #
                sig_lr  = _conf_signal(_cosine_sim(g_f, prev_g))               # LR / alpha signal
                sig_wd  = _conf_signal(abs(_cosine_sim(g_f, p.data.float())))  # Decay signal
                sig_mom = _conf_signal(_cosine_sim(g_f, v.float()))            # Momentum signal

                # -------------------------------------------------------- #
                # 2. Autonomous hyperparameter update                       #
                #                                                           #
                # Each meta-parameter is governed by three forces:         #
                #   Drive:   cosine signal pushes toward the better state. #
                #   Restore: spring toward a principled equilibrium.       #
                #   Couple:  cross-terms prevent per_lr × per_beta from    #
                #            forming a runaway effective step amplitude.   #
                # -------------------------------------------------------- #

                # Per-param LR — multiplicative (log-space).
                # Restore pulls per_lr toward init_lr (calibrated start), not toward 1.0.
                # Coupling measures drift of the effective step amplitude from its calibrated
                # neutral: (per_lr/init_lr) / (10*(1-beta)) = 1.0 at cold start with beta=0.9.
                _drift_amp = (per_lr / init_lr) / (10.0 * max(1.0 - per_beta, 1e-4))
                per_lr = max(self._LR_MIN, min(self._LR_MAX,
                             per_lr * math.exp(
                                 self._ETA_LR          * sig_lr
                                 - self._RESTORE_LR    * _log_drift
                                 - self._COUPLE_LR_BETA * math.log(max(_drift_amp, 1e-9))
                             )))

                # Recompute log_drift with updated per_lr for use in beta coupling.
                _log_drift = math.log(max(per_lr / init_lr, 1e-9))

                # Per-param momentum beta — additive.
                # Coupling uses log_drift so that cold-start high per_lr (small gradients)
                # does not incorrectly suppress beta — only signal-driven per_lr changes do.
                per_beta = max(self._BETA_MIN, min(self._BETA_MAX,
                               per_beta
                               + self._ETA_MOM         * sig_mom
                               - self._RESTORE_BETA    * (per_beta - 0.9)
                               - self._COUPLE_BETA_LR  * _log_drift))

                # Centralization gate alpha — additive.
                # No cross-coupling: sig_lr already carries implicit LR effects
                # (high LR → oscillation → lower sig_lr → higher alpha).
                per_alpha = max(0.0, min(1.0,
                                per_alpha
                                - self._ETA_ALPHA    * sig_lr
                                - self._RESTORE_ALPHA * (per_alpha - 0.5)))

                # Per-param weight decay — additive; Hebbian bypass enforced.
                # Decay neutral = seed / per_lr (lr×decay product coupling).
                if not is_hebbian:
                    _decay_neutral = init_decay / max(per_lr, 0.1)
                    per_decay = max(0.0, min(self._DECAY_MAX,
                                   per_decay
                                   + self._ETA_DECAY    * sig_wd
                                   - self._RESTORE_DECAY * (per_decay - _decay_neutral)))

                # -------------------------------------------------------- #
                # 3. Gradient Centralization (continuous gate)              #
                # -------------------------------------------------------- #
                if g_f.dim() >= 2 and per_alpha > 1e-3:
                    g_proc = g_f - per_alpha * g_f.mean(
                        dim=tuple(range(1, g_f.dim())), keepdim=True
                    )
                else:
                    g_proc = g_f

                # Enforce zero diagonal on chaos core gradient.
                if is_core and g_proc.dim() == 2 and g_proc.shape[0] == g_proc.shape[1]:
                    g_proc = g_proc.clone()
                    g_proc.fill_diagonal_(0.0)

                # -------------------------------------------------------- #
                # 4. Momentum update                                        #
                # -------------------------------------------------------- #
                v.mul_(per_beta).add_(g_proc, alpha=1.0 - per_beta)

                # -------------------------------------------------------- #
                # 5. Frustration burst (escape local minima)                #
                # -------------------------------------------------------- #
                if do_burst:
                    # Noise scales with init_lr, not per_lr. When a small network is stuck,
                    # per_lr has already collapsed toward LR_MIN, making per_lr-scaled bursts
                    # too weak to escape. init_lr reflects the true gradient scale of the
                    # tensor and stays fixed, so bursts remain meaningful even under distress.
                    noise_scale = self._FRUST_NOISE * genesis_lr * init_lr
                    noise = torch.randn_like(v) * noise_scale
                    
                    # Centralized Noise (Gradient Centralization Protection)
                    # Shakes the system without destroying the zero-mean equilibrium it has built.
                    if g_f.dim() >= 2 and per_alpha > 1e-3:
                        noise = noise - per_alpha * noise.mean(
                            dim=tuple(range(1, noise.dim())), keepdim=True
                        )
                        
                    v.add_(noise)
                    
                    # Reboot per_lr toward init_lr (the calibrated starting point for this
                    # tensor), so the burst relaunches from the gradient-derived magnitude
                    # rather than from an arbitrary fixed point.
                    per_lr    = per_lr    + self._FRUST_META_RESET * (init_lr - per_lr)
                    per_beta  = per_beta  + self._FRUST_META_RESET * (0.9 - per_beta)
                    per_alpha = per_alpha + self._FRUST_META_RESET * (0.5 - per_alpha)

                # -------------------------------------------------------- #
                # 6. Autonomous weight decay                                 #
                # -------------------------------------------------------- #
                if per_decay > 0.0 and not is_hebbian:
                    p.data.mul_(1.0 - genesis_lr * per_decay)

                # -------------------------------------------------------- #
                # 7. Parameter update                                        #
                # -------------------------------------------------------- #
                p.data.add_(v, alpha=-(genesis_lr * per_lr))

                # -------------------------------------------------------- #
                # 8. Core matrix diagonal constraint                        #
                # -------------------------------------------------------- #
                if is_core and p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.data.fill_diagonal_(0.0)

                # -------------------------------------------------------- #
                # 9. Persist state                                           #
                # -------------------------------------------------------- #
                state['prev_grad']       = g.detach()
                state['per_param_lr']    = per_lr
                state['per_param_decay'] = per_decay
                state['per_param_beta']  = per_beta
                state['per_param_alpha'] = per_alpha

        return loss
