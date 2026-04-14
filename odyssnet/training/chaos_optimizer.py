"""
ChaosGrad v3 — Zero-hyperparameter optimizer for OdyssNet.

ChaosGrad is a fully optional, drop-in optimizer. The OdyssNetTrainer defaults
to Prodigy (lr=None) or AdamW (explicit lr); pass ChaosGrad as a custom
optimizer when you want autonomous per-parameter meta-adaptation:

    from odyssnet import OdyssNet, OdyssNetTrainer
    from odyssnet.training.chaos_optimizer import ChaosGrad

    model   = OdyssNet(num_neurons=32, input_ids=[0], output_ids=[31])
    opt     = ChaosGrad(ChaosGrad.classify_params(model), lr=1e-4)
    trainer = OdyssNetTrainer(model, optimizer=opt)

Algorithm (v3 improvements over the removed v2.2):
  - Second-moment adaptive normalization  → closes the AdamW performance gap
  - Bias-corrected momentum              → faster, cleaner early-training signal
  - Grad EMA for hypergradient signals   → stable reference in recurrent regimes
  - Group-aware frustration bursts       → Hebbian logits never disrupted
  - 9-group parameter classification     → bias / norm / scales are first-class
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


class ChaosGrad(torch.optim.Optimizer):
    """
    Zero-hyperparameter optimizer designed for OdyssNet.

    For each parameter the optimizer autonomously learns:
      - per_param_lr    : learning-rate multiplier
      - per_param_beta  : momentum coefficient
      - per_param_decay : weight decay rate
      - per_param_alpha : gradient-centralization gate

    The single user-facing parameter ``lr`` (genesis learning rate,
    default 1e-4) is a mathematical starting point, not a dial to tune.

    Args:
        params: Iterable of parameters **or** a list of classified param-group
                dicts returned by :meth:`classify_params`. Providing classified
                groups enables group-specific decay seeding, per-group beta
                equilibria, and the Hebbian bypass rule.
        lr (float): Genesis learning rate. Default: ``1e-4``.
    """

    # ------------------------------------------------------------------ #
    # Fixed meta-constants — not user-configurable                        #
    # ------------------------------------------------------------------ #

    # Driving forces
    _ETA_LR    = 0.05
    _ETA_MOM   = 0.02
    _ETA_DECAY = 0.002
    _ETA_ALPHA = 0.02

    # Restoring forces (spring constants toward equilibrium)
    _RESTORE_LR    = 0.02
    _RESTORE_BETA  = 0.10
    _RESTORE_ALPHA = 0.05
    _RESTORE_DECAY = 0.10

    # Cross-couplings
    _COUPLE_LR_BETA = 0.01   # _RESTORE_LR / 2
    _COUPLE_BETA_LR = 0.01   # _RESTORE_BETA / 10

    # Safety rails
    _LR_MIN    = 0.01
    _LR_MAX    = 100.0
    _BETA_MIN  = 0.5
    _BETA_MAX  = 0.999
    _DECAY_MAX = 0.1

    # Frustration accumulator
    _FRUST_DECAY      = 0.995
    _FRUST_THRESH     = 0.75
    _FRUST_NOISE      = 0.01
    _FRUST_META_RESET = 0.30

    # Cold-start helpers
    _GENESIS_SCALAR = 1e-6
    _SIZE_CAP_SCALE = 2.0

    # v3 additions
    _BETA2        = 0.999   # Second-moment EMA decay
    _EPS          = 1e-8    # Denominator safety floor
    _SIGNAL_ALPHA = 0.6     # Grad-EMA decay (signal computation reference)

    # Per-group initial decay seeds
    _INIT_DECAY: dict[str, float] = {
        'chaos_core':  0.01,
        'memory':      0.0,
        'projections': 0.01,
        'gates':       0.0,
        'hebbian':     0.0,
        'norm':        0.0,
        'bias':        0.0,
        'scales':      0.0,
        'lightweight': 0.0,
    }

    # Per-group momentum beta equilibrium
    _BETA_EQUIL: dict[str, float] = {
        'chaos_core':  0.95,   # Recurrent dynamics benefit from more smoothing
        'memory':      0.95,
        'projections': 0.90,
        'gates':       0.85,   # Gates need faster response, less inertia
        'hebbian':     0.90,
        'norm':        0.90,
        'bias':        0.90,
        'scales':      0.90,
        'lightweight': 0.90,
    }

    # Burst intensity per group: 'full', 'half', 'none'
    _BURST_TYPE: dict[str, str] = {
        'chaos_core':  'full',  # noise + meta-reset
        'memory':      'full',
        'projections': 'full',
        'gates':       'half',  # half-scale noise, no meta-reset
        'hebbian':     'none',  # Hebbian logits manage their own dynamics
        'norm':        'half',
        'bias':        'half',
        'scales':      'half',
        'lightweight': 'half',
    }

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(self, params, lr: float = 1e-4) -> None:
        if lr <= 0:
            raise ValueError(f"Genesis learning rate must be > 0, got {lr}")
        defaults = dict(
            lr=lr,
            group_name='lightweight',
            init_decay=0.0,
            beta_equil=0.90,
            burst_type='half',
            is_hebbian=False,
        )
        super().__init__(params, defaults)

        self._global_step: int   = 0
        self._frustration: float = 0.0
        self._best_loss:   float = float('inf')
        self._force_plateau_escape: bool = False

    # ------------------------------------------------------------------ #
    # state_dict / load_state_dict overrides                              #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        """
        Return the full optimizer state, including ChaosGrad global state.

        The standard PyTorch ``state_dict`` is extended with a
        ``'chaos_global'`` key so that frustration, best_loss, and global
        step survive :func:`~odyssnet.utils.odyssstore.save_checkpoint` /
        :func:`~odyssnet.utils.odyssstore.load_checkpoint` round-trips.
        """
        d = super().state_dict()
        d['chaos_global'] = {
            'global_step': self._global_step,
            'frustration':  self._frustration,
            'best_loss':    self._best_loss,
        }
        return d

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Restore optimizer state, including ChaosGrad global state.

        Accepts state dicts produced by both this class (which include
        ``'chaos_global'``) and by external tooling that omitted it (the
        global state is silently initialised to defaults in that case).
        """
        chaos_global = state_dict.get('chaos_global', {})
        super().load_state_dict(state_dict)
        self._global_step = int(chaos_global.get('global_step', 0))
        self._frustration = float(chaos_global.get('frustration', 0.0))
        self._best_loss   = float(chaos_global.get('best_loss', float('inf')))

    # ------------------------------------------------------------------ #
    # Parameter classification                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify_params(model) -> list[dict]:
        """
        Classify OdyssNet parameters into 9 semantic groups.

        Returns a list of param-group dicts suitable for passing directly to
        ``ChaosGrad.__init__``.  Groups without any parameters are omitted.

        Groups (in priority order):

        ============  =====================================  ===========
        Group         Detection                              Init Decay
        ============  =====================================  ===========
        chaos_core    leaf == 'W'                            0.01
        memory        leaf == 'memory_feedback'              0.0
        projections   'embed'/'proj'/'output_decoder'        0.01
        gates         leaf in {input_gate, …, memory_gate}  0.0
        hebbian       leaf in {hebb_factor, hebb_decay}      0.0  (bypass)
        norm          'norm.' in name                        0.0
        bias          leaf == 'B'                            0.0
        scales        leaf in {input_scale, output_scale}    0.0
        lightweight   everything else                        0.0
        ============  =====================================  ===========
        """
        groups: dict[str, dict] = {
            'chaos_core':  dict(params=[], group_name='chaos_core',  init_decay=0.01, beta_equil=0.95, burst_type='full', is_hebbian=False),
            'memory':      dict(params=[], group_name='memory',      init_decay=0.0,  beta_equil=0.95, burst_type='full', is_hebbian=False),
            'projections': dict(params=[], group_name='projections', init_decay=0.01, beta_equil=0.90, burst_type='full', is_hebbian=False),
            'gates':       dict(params=[], group_name='gates',       init_decay=0.0,  beta_equil=0.85, burst_type='half', is_hebbian=False),
            'hebbian':     dict(params=[], group_name='hebbian',     init_decay=0.0,  beta_equil=0.90, burst_type='none', is_hebbian=True),
            'norm':        dict(params=[], group_name='norm',        init_decay=0.0,  beta_equil=0.90, burst_type='half', is_hebbian=False),
            'bias':        dict(params=[], group_name='bias',        init_decay=0.0,  beta_equil=0.90, burst_type='half', is_hebbian=False),
            'scales':      dict(params=[], group_name='scales',      init_decay=0.0,  beta_equil=0.90, burst_type='half', is_hebbian=False),
            'lightweight': dict(params=[], group_name='lightweight', init_decay=0.0,  beta_equil=0.90, burst_type='half', is_hebbian=False),
        }

        _GATE_LEAVES  = {'input_gate', 'output_gate', 'core_gate', 'memory_gate'}
        _HEBB_LEAVES  = {'hebb_factor', 'hebb_decay'}
        _SCALE_LEAVES = {'input_scale', 'output_scale'}
        _PROJ_KEYS    = ('embed', 'proj', 'output_decoder')

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            leaf = name.split('.')[-1]

            if leaf == 'W':
                groups['chaos_core']['params'].append(param)
            elif leaf == 'memory_feedback':
                groups['memory']['params'].append(param)
            elif any(k in name for k in _PROJ_KEYS):
                groups['projections']['params'].append(param)
            elif leaf in _GATE_LEAVES:
                groups['gates']['params'].append(param)
            elif leaf in _HEBB_LEAVES:
                groups['hebbian']['params'].append(param)
            elif 'norm.' in name:
                groups['norm']['params'].append(param)
            elif leaf == 'B':
                groups['bias']['params'].append(param)
            elif leaf in _SCALE_LEAVES:
                groups['scales']['params'].append(param)
            else:
                groups['lightweight']['params'].append(param)

        return [g for g in groups.values() if g['params']]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        sim = F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-12)
        return sim.mean().item()

    @staticmethod
    def _conf_signal(cos: float) -> float:
        """s = cos × |cos|  —  confidence-weighted, range [-1, 1]."""
        return cos * abs(cos)

    def _init_param_state(
        self,
        p: torch.Tensor,
        g_f: torch.Tensor,
        group: dict,
    ) -> dict:
        """Cold-start state initialisation (T = 0)."""
        g_rms    = max(g_f.pow(2).mean().sqrt().item(), 1e-12)
        size_cap = self._SIZE_CAP_SCALE * math.sqrt(p.numel())
        init_lr  = max(self._LR_MIN, min(self._LR_MAX, size_cap, 1.0 / g_rms))

        is_hebbian = group.get('is_hebbian', False)
        init_decay = 0.0 if is_hebbian else group.get('init_decay', 0.0)
        beta_equil = group.get('beta_equil', 0.90)

        return {
            'step':            0,
            'init_lr':         init_lr,
            'grad_ema':        (g_f * self._GENESIS_SCALAR).detach().clone(),
            'momentum':        torch.zeros_like(g_f),
            'per_param_lr':    init_lr,
            'per_param_beta':  beta_equil,
            'per_param_decay': init_decay,
            'per_param_alpha': 0.5,
            'v2':              0.0,
        }

    def _compute_signals(
        self,
        g_f: torch.Tensor,
        state: dict,
        p: torch.Tensor,
    ) -> tuple[float, float, float]:
        """
        Three hypergradient signals.

        sig_lr  — gradient consistency (g_t vs slow grad EMA)
        sig_wd  — weight-gradient alignment
        sig_mom — momentum consistency
        """
        sig_lr  = self._conf_signal(self._cosine_sim(g_f, state['grad_ema']))
        sig_wd  = self._conf_signal(abs(self._cosine_sim(g_f, p.float())))
        sig_mom = self._conf_signal(self._cosine_sim(g_f, state['momentum']))
        return sig_lr, sig_wd, sig_mom

    def _update_meta_params(
        self,
        state: dict,
        sigs: tuple[float, float, float],
        group: dict,
    ) -> None:
        """Three-force (drive / restore / couple) meta-parameter update."""
        sig_lr, sig_wd, sig_mom = sigs

        per_lr    = state['per_param_lr']
        per_beta  = state['per_param_beta']
        per_decay = state['per_param_decay']
        per_alpha = state['per_param_alpha']
        init_lr   = state['init_lr']

        is_hebbian = group.get('is_hebbian', False)
        beta_equil = group.get('beta_equil', 0.90)
        init_decay = group.get('init_decay', 0.0)

        log_drift     = math.log(max(per_lr / init_lr, 1e-12))
        drift_amp     = (per_lr / init_lr) / max(10.0 * (1.0 - per_beta), 1e-12)
        log_drift_amp = math.log(max(drift_amp, 1e-12))

        # A. Learning rate (multiplicative, log-space)
        per_lr = per_lr * math.exp(
            self._ETA_LR        * sig_lr
            - self._RESTORE_LR  * log_drift
            - self._COUPLE_LR_BETA * log_drift_amp
        )
        per_lr = max(self._LR_MIN, min(self._LR_MAX, per_lr))

        # B. Momentum beta (additive, linear)
        per_beta = per_beta + (
              self._ETA_MOM        * sig_mom
            - self._RESTORE_BETA   * (per_beta - beta_equil)
            - self._COUPLE_BETA_LR * log_drift
        )
        per_beta = max(self._BETA_MIN, min(self._BETA_MAX, per_beta))

        # C. Weight decay (additive) — unconditional Hebbian bypass
        if not is_hebbian:
            decay_neutral = init_decay / max(per_lr, 1e-12)
            per_decay = per_decay + (
                  self._ETA_DECAY       * sig_wd
                - self._RESTORE_DECAY   * (per_decay - decay_neutral)
            )
            per_decay = max(0.0, min(self._DECAY_MAX, per_decay))
        else:
            per_decay = 0.0

        # D. Centralization gate alpha (additive, inverse drive)
        per_alpha = per_alpha + (
            - self._ETA_ALPHA     * sig_lr
            - self._RESTORE_ALPHA * (per_alpha - 0.5)
        )
        per_alpha = max(0.0, min(1.0, per_alpha))

        state['per_param_lr']    = per_lr
        state['per_param_beta']  = per_beta
        state['per_param_decay'] = per_decay
        state['per_param_alpha'] = per_alpha

    # ------------------------------------------------------------------ #
    # Optimizer step                                                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single ChaosGrad v3 optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        burst_now  = (self._frustration > self._FRUST_THRESH) or self._force_plateau_escape
        self._force_plateau_escape = False

        for group in self.param_groups:
            # Read genesis_lr from the group so that load_checkpoint lr-overrides
            # propagate correctly to ChaosGrad's scaling terms.
            genesis_lr = group.get('lr', self.defaults['lr'])
            group_name = group.get('group_name', 'lightweight')
            is_hebbian = group.get('is_hebbian', False)
            burst_type = group.get('burst_type', 'half')
            is_core    = (group_name == 'chaos_core')

            for p in group['params']:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError(
                        "ChaosGrad does not support sparse gradients. "
                        "Use a dense parameter or a different optimizer."
                    )

                g_f = p.grad.float().detach()

                # ---- Cold start ----
                if not self.state[p]:
                    self.state[p] = self._init_param_state(p, g_f, group)

                state = self.state[p]
                state['step'] += 1
                step = state['step']

                # ---- Hypergradient signals ----
                sigs = self._compute_signals(g_f, state, p)

                # ---- Meta-parameter update ----
                self._update_meta_params(state, sigs, group)

                per_lr    = state['per_param_lr']
                per_beta  = state['per_param_beta']
                per_alpha = state['per_param_alpha']
                per_decay = state['per_param_decay']
                init_lr   = state['init_lr']
                v         = state['momentum']

                # ---- Update grad EMA (for next step's signals) ----
                state['grad_ema'].mul_(self._SIGNAL_ALPHA).add_(
                    g_f, alpha=1.0 - self._SIGNAL_ALPHA
                )

                # ---- Gradient centralization ----
                g_proc = g_f
                if per_alpha > 1e-3 and g_f.dim() >= 2:
                    dims   = tuple(range(1, g_f.dim()))
                    g_proc = g_f - per_alpha * g_f.mean(dim=dims, keepdim=True)

                # ---- Zero diagonal on chaos core gradient ----
                if is_core and g_proc.dim() == 2 and g_proc.shape[0] == g_proc.shape[1]:
                    g_proc = g_proc.clone()
                    g_proc.fill_diagonal_(0.0)

                # ---- Momentum accumulation ----
                v.mul_(per_beta).add_(g_proc, alpha=1.0 - per_beta)

                # ---- Frustration burst ----
                if burst_now and burst_type != 'none':
                    noise_scale = self._FRUST_NOISE * genesis_lr * init_lr
                    if burst_type == 'half':
                        noise_scale *= 0.5
                    noise = torch.randn_like(v) * noise_scale
                    if per_alpha > 1e-3 and noise.dim() >= 2:
                        dims  = tuple(range(1, noise.dim()))
                        noise = noise - per_alpha * noise.mean(dim=dims, keepdim=True)
                    v.add_(noise)

                    if burst_type == 'full':
                        beta_equil = group.get('beta_equil', 0.90)
                        state['per_param_lr']    += self._FRUST_META_RESET * (init_lr    - per_lr)
                        state['per_param_beta']  += self._FRUST_META_RESET * (beta_equil - per_beta)
                        state['per_param_alpha'] += self._FRUST_META_RESET * (0.5        - per_alpha)

                # ---- Bias-corrected momentum ----
                bias_corr = max(1.0 - per_beta ** step, self._EPS)
                v_hat     = v / bias_corr

                # ---- Second-moment adaptive normalisation ----
                g_rms_sq = g_f.pow(2).mean().item()
                v2       = self._BETA2 * state['v2'] + (1.0 - self._BETA2) * g_rms_sq
                state['v2'] = v2
                v2_hat   = v2 / max(1.0 - self._BETA2 ** step, self._EPS)
                denom    = max(math.sqrt(v2_hat) * init_lr, self._EPS)

                # ---- Weight decay ----
                if per_decay > 0.0 and not is_hebbian:
                    decay_factor = max(self._EPS, 1.0 - genesis_lr * per_decay)
                    p.data.mul_(decay_factor)

                # ---- Parameter update ----
                p.data.add_(v_hat, alpha=-(genesis_lr * per_lr / denom))

                # ---- Enforce zero diagonal on chaos core ----
                if is_core and p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.data.fill_diagonal_(0.0)

        # Global frustration partial reset after burst
        if burst_now:
            self._frustration *= 0.3

        self._global_step += 1
        return loss

    # ------------------------------------------------------------------ #
    # Frustration API                                                      #
    # ------------------------------------------------------------------ #

    def report_loss(self, loss_value: float) -> None:
        """
        Report the current loss to the Frustration Accumulator.

        Call this once per optimizer step (the trainer does this automatically
        when ChaosGrad is detected as the active optimizer).
        """
        loss = float(loss_value)
        if loss < self._best_loss * 0.9999:
            self._best_loss      = loss
            frustration_signal   = 0.0
        else:
            frustration_signal = min(1.0, loss / max(self._best_loss, 1e-10))

        self._frustration = (
            self._frustration * self._FRUST_DECAY
            + frustration_signal * (1.0 - self._FRUST_DECAY)
        )

    def trigger_plateau_escape(self) -> None:
        """Manually trigger a frustration burst on the next :meth:`step` call."""
        self._force_plateau_escape = True

    def reset_param_state(self, param: torch.Tensor) -> None:
        """
        Clear the per-parameter optimizer state for ``param``.

        On the next :meth:`step` call the parameter undergoes cold-start
        recalibration — ``init_lr`` is recomputed from the current gradient
        RMS, momentum is zeroed, and meta-parameters are re-seeded from
        group defaults.

        Typical use case: after :meth:`~odyssnet.OdyssNetTrainer.regenerate_synapses`
        resets weak entries of ``W``, the stale momentum buffer would push the
        freshly re-initialised weights in the old direction.  Clearing the state
        lets ChaosGrad re-calibrate for the new gradient landscape.

        Args:
            param: The parameter tensor whose optimizer state should be cleared.
                   Must be a tensor currently tracked by this optimizer.
        """
        if param in self.state:
            del self.state[param]

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self, debug: bool = False) -> dict:
        """
        Return optimizer health metrics.

        Basic mode returns global scalars.  ``debug=True`` adds per-group
        breakdowns and per-parameter statistics.
        """
        all_states = [
            (self.state[p], group)
            for group in self.param_groups
            for p in group['params']
            if self.state.get(p)
        ]

        genesis_lr    = self.defaults['lr']
        effective_lrs = []
        init_lrs      = []
        for state, group in all_states:
            if 'per_param_lr' in state and 'init_lr' in state:
                group_genesis_lr = group.get('lr', self.defaults['lr'])
                effective_lrs.append(state['per_param_lr'] / state['init_lr'])
                init_lrs.append(state['init_lr'] * group_genesis_lr)

        diag: dict = {
            'global_step':      self._global_step,
            'frustration':      self._frustration,
            'best_loss':        self._best_loss,
            'avg_effective_lr': (sum(effective_lrs) / len(effective_lrs)) if effective_lrs else 0.0,
            'avg_init_lr':      (sum(init_lrs) / len(init_lrs)) if init_lrs else 0.0,
        }

        if not debug:
            return diag

        betas  = [s['per_param_beta']  for s, _ in all_states if 'per_param_beta'  in s]
        alphas = [s['per_param_alpha']  for s, _ in all_states if 'per_param_alpha' in s]
        decays = [s['per_param_decay']  for s, _ in all_states if 'per_param_decay' in s]
        steps  = [s['step']             for s, _ in all_states if 'step'            in s]

        diag['avg_beta']  = (sum(betas)  / len(betas))  if betas  else 0.0
        diag['avg_alpha'] = (sum(alphas) / len(alphas)) if alphas else 0.0
        diag['avg_decay'] = (sum(decays) / len(decays)) if decays else 0.0

        # Per-group breakdown
        group_stats = []
        for group in self.param_groups:
            gname    = group.get('group_name', 'unknown')
            g_states = [self.state[p] for p in group['params'] if self.state.get(p)]
            if not g_states:
                continue
            g_genesis_lr = group.get('lr', self.defaults['lr'])
            g_lrs   = [s['per_param_lr'] / s['init_lr'] for s in g_states if 'per_param_lr' in s]
            g_init_lrs = [s['init_lr'] * g_genesis_lr for s in g_states if 'init_lr' in s]
            g_betas  = [s['per_param_beta']  for s in g_states if 'per_param_beta'  in s]
            g_alphas = [s['per_param_alpha']  for s in g_states if 'per_param_alpha' in s]
            g_decays = [s['per_param_decay']  for s in g_states if 'per_param_decay' in s]
            group_stats.append({
                'group_name':       gname,
                'param_count':      len(g_states),
                'avg_effective_lr': (sum(g_lrs)    / len(g_lrs))    if g_lrs    else 0.0,
                'avg_init_lr':      (sum(g_init_lrs) / len(g_init_lrs)) if g_init_lrs else 0.0,
                'avg_beta':         (sum(g_betas)   / len(g_betas))  if g_betas  else 0.0,
                'avg_alpha':        (sum(g_alphas)  / len(g_alphas)) if g_alphas else 0.0,
                'avg_decay':        (sum(g_decays)  / len(g_decays)) if g_decays else 0.0,
            })
        diag['param_groups'] = group_stats

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {'min': 0.0, 'max': 0.0, 'std': 0.0}
            t = torch.tensor(vals, dtype=torch.float32)
            return {
                'min': t.min().item(),
                'max': t.max().item(),
                'std': t.std().item() if len(vals) > 1 else 0.0,
            }

        def _stats_mean(vals: list[float]) -> dict:
            d = _stats(vals)
            d['mean'] = (sum(vals) / len(vals)) if vals else 0.0
            return d

        diag['per_param_stats'] = {
            'effective_lr': _stats(effective_lrs),
            'beta':         _stats(betas),
            'alpha':        _stats(alphas),
            'decay':        _stats(decays),
            'steps':        _stats_mean(steps),
        }
        return diag
