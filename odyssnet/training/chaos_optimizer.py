"""
ChaosGrad — Zero-Config Optimizer for OdyssNet

The "learning teacher" rebuilt on provable ground. ChaosGrad combines:

1. **Per-synapse preconditioning** (Adam-style second moment): every synapse
   in the NxN chaos core gets its own effective step size, because temporal
   weight reuse across thinking steps makes gradient scales wildly
   heterogeneous across the matrix.

2. **Online distance adaptation** (D-adaptation / Prodigy-class estimator):
   the step scale is estimated at runtime from the observed distance
   traveled toward the solution. No learning rate is required — the
   optimizer discovers the natural step size of the loss landscape.

3. **Architecture-aware policy**: parameters are classified into families
   (chaos core, memory feedback, projections, plasticity logits,
   modulation). Weight decay is applied only to connective structure
   (core, projections), never to Hebbian plasticity logits, gates, biases,
   or norm weights — their zero is not a neutral point. The chaos core's
   zero-diagonal constraint (self-connections live in ``memory_feedback``,
   not ``W``) is enforced inside the step.

Zero mandatory hyperparameters: ``ChaosGrad.from_model(model)`` is all a
user needs. Every knob remains available for experts.

Design notes
------------
- Distance-adaptation accumulators are kept as on-device tensors and
  resolved with a single host sync per step.
- The estimator reference point ``p0`` and accumulator ``s`` are stored in
  the parameter's own shape (not flattened) so neurogenesis expansion can
  pad optimizer state with the same top-left-corner rule it applies to the
  parameters themselves.
- ``d_mode='global'`` (default) shares one step-scale estimate across all
  families — empirically the most stable on OdyssNet's coupled recurrent
  dynamics, because every family feeds the same echo chamber.
  ``d_mode='per_group'`` gives each family an independent estimate for
  research into decoupled family dynamics.
"""

import math

import torch


class ChaosGrad(torch.optim.Optimizer):
    """
    Zero-config OdyssNet optimizer: Adam-style per-synapse preconditioning
    with online distance adaptation (D-adaptation / Prodigy-class math).

    Args:
        params: Iterable of parameters or param-group dicts. Use
            :meth:`from_model` for architecture-aware grouping.
        lr (float, optional): If ``None`` (default), the step scale is
            estimated automatically. If a float is given, automatic
            estimation is disabled and the value is used as a fixed
            AdamW-style learning rate — an escape hatch for exact
            reproducibility studies.
        betas (Tuple[float, float]): Adam EMA coefficients. Default (0.9, 0.999).
        beta3 (float, optional): EMA coefficient for the distance
            estimator. Default ``sqrt(betas[1])``.
        eps (float): Numerical stability term. Default 1e-8.
        weight_decay (float): Decoupled weight decay. Default 0.0.
            :meth:`from_model` overrides this per family.
        d0 (float): Initial step-scale estimate. Default 1e-6.
        d_coef (float): Multiplier on the estimated scale. The preferred
            expert tuning knob (0.5 = cautious, 2.0 = bold). Default 1.0.
        growth_rate (float): Max multiplicative growth of the estimate per
            step. ``inf`` (default) lets the estimator settle immediately;
            finite values (e.g. 1.02) act as implicit warmup.
        d_mode (str): ``'global'`` (default) shares one step-scale estimate
            across all parameter groups; ``'per_group'`` estimates
            independently per group.
        trust_ratio (float, optional): Traction limit — the applied step
            scale is capped at ``trust_ratio × RMS(initial weights)``,
            where the RMS anchor is the smallest per-group initialization
            scale (groups initialized at/near zero or exactly one are
            ignored). On small chaotic networks the distance estimator can
            overshoot by orders of magnitude during the turbulent early
            phase; the anchored cap keeps every step proportional to the
            network's design scale. Set ``None`` to disable. Default 0.25.
        brake_factor (float, optional): Loss-spike brake. When
            :meth:`report_loss` observes a statistical spike in the loss
            stream, the distance estimate (and its ratchet ceiling) is
            multiplied by this factor, letting the estimator re-grow only
            if the landscape truly supports it. Counteracts the monotone
            step-scale growth that destabilizes sharpening temporal tasks.
            Set ``None`` to disable. Default 0.5.
        use_bias_correction (bool): Adam bias correction on the step
            magnitude. Default True.
        safeguard_warmup (bool): Use the warmup-robust variant of the
            estimator denominator. Default False.
    """

    def __init__(self, params, lr=None, betas=(0.9, 0.999), beta3=None,
                 eps=1e-8, weight_decay=0.0, d0=1e-6, d_coef=1.0,
                 growth_rate=float('inf'), d_mode='global', trust_ratio=0.25,
                 brake_factor=0.5, use_bias_correction=True,
                 safeguard_warmup=False):
        if lr is not None and lr <= 0.0:
            raise ValueError(f"lr must be positive or None, got {lr}")
        if not 0.0 < d0:
            raise ValueError(f"d0 must be positive, got {d0}")
        if not 0.0 < eps:
            raise ValueError(f"eps must be positive, got {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if beta3 is not None and not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3: {beta3}")
        if growth_rate <= 1.0:
            raise ValueError(f"growth_rate must be > 1.0, got {growth_rate}")
        if d_mode not in ('global', 'per_group'):
            raise ValueError(f"d_mode must be 'global' or 'per_group', got {d_mode!r}")
        if trust_ratio is not None and trust_ratio <= 0.0:
            raise ValueError(f"trust_ratio must be positive or None, got {trust_ratio}")
        if brake_factor is not None and not 0.0 < brake_factor < 1.0:
            raise ValueError(f"brake_factor must be in (0, 1) or None, got {brake_factor}")

        defaults = dict(
            lr=lr, betas=betas, beta3=beta3, eps=eps,
            weight_decay=weight_decay,
            d=d0, d0=d0, d_max=d0, d_numerator=0.0,
            d_coef=d_coef, growth_rate=growth_rate, d_mode=d_mode,
            trust_ratio=trust_ratio, brake_factor=brake_factor,
            use_bias_correction=use_bias_correction,
            safeguard_warmup=safeguard_warmup,
            zero_diag=False,
            group_name='default',
            k=0,
        )
        super().__init__(params, defaults)

        # Loss-stream statistics for the spike brake. Transient by design:
        # after a checkpoint reload the EWMA re-seeds from the first
        # reported loss, which is harmless (the braked d/d_max persist via
        # the param groups).
        self._loss_ema = None
        self._loss_var = 0.0

        # Anchor the traction limit to the *initial* weight scale. A cap
        # referencing live weights would be self-defeating: runaway steps
        # inflate the weights, which would inflate the cap in turn.
        if trust_ratio is not None:
            for group in self.param_groups:
                group['rms0'] = self._group_rms(group['params'])

    @staticmethod
    @torch.no_grad()
    def _group_rms(params):
        """RMS magnitude of a parameter collection (0.0 when empty)."""
        total_sq = 0.0
        total_n = 0
        for p in params:
            total_sq += p.detach().float().pow(2).sum().item()
            total_n += p.numel()
        return math.sqrt(total_sq / total_n) if total_n else 0.0

    # ------------------------------------------------------------------ #
    # Architecture-aware construction                                     #
    # ------------------------------------------------------------------ #

    #: Default decoupled weight decay per parameter family. Only connective
    #: structure decays; plasticity logits, gates, biases and norms must
    #: never be pulled toward zero (their zero is not a neutral point).
    FAMILY_WEIGHT_DECAY = {
        'chaos_core':      0.01,
        'projections':     0.01,
        'memory_feedback': 0.0,
        'plasticity':      0.0,
        'modulation':      0.0,
    }

    @classmethod
    def classify_params(cls, model):
        """
        Classifies OdyssNet parameters into semantically distinct families.

        Families:
            chaos_core      — the NxN recurrent matrix ``W`` (zero-diagonal).
            memory_feedback — per-neuron self-connections.
            projections     — embed / proj / output_decoder matrices.
            plasticity      — Hebbian factor/decay logits.
            modulation      — gates, scales, bias, norm weights (everything else).

        Returns a list of param-group dicts suitable for ``ChaosGrad(...)``.
        """
        families = {name: [] for name in cls.FAMILY_WEIGHT_DECAY}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            leaf = name.split('.')[-1]
            if leaf == 'W':
                families['chaos_core'].append(param)
            elif leaf == 'memory_feedback':
                families['memory_feedback'].append(param)
            elif any(k in name for k in ('embed', 'proj', 'output_decoder')):
                families['projections'].append(param)
            elif 'hebb' in leaf:
                families['plasticity'].append(param)
            else:
                families['modulation'].append(param)

        groups = []
        for family, params in families.items():
            if not params:
                continue
            groups.append({
                'params': params,
                'group_name': family,
                'weight_decay': cls.FAMILY_WEIGHT_DECAY[family],
                'zero_diag': family == 'chaos_core',
            })
        return groups

    @classmethod
    def from_model(cls, model, lr=None, **kwargs):
        """
        Builds a ChaosGrad instance with architecture-aware parameter
        grouping and per-family policy. This is the zero-config entry point:

            optimizer = ChaosGrad.from_model(model)

        Args:
            model (OdyssNet): The model to optimize.
            lr (float, optional): Fixed learning rate (disables automatic
                step-scale estimation). Default None (fully automatic).
            **kwargs: Forwarded to the constructor for expert overrides.
        """
        return cls(cls.classify_params(model), lr=lr, **kwargs)

    # ------------------------------------------------------------------ #
    # Step                                                                #
    # ------------------------------------------------------------------ #

    def _beta3(self, group):
        beta3 = group['beta3']
        return beta3 if beta3 is not None else math.sqrt(group['betas'][1])

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        adaptive_groups = []
        numerator_deltas = []
        denominators = []

        # ---- Pass 1: EMA updates + distance-estimator accumulation ----
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta3 = self._beta3(group)
            d = group['d']
            d0 = group['d0']
            k = group['k']
            fixed_lr = group['lr']
            adaptive = fixed_lr is None

            if group['use_bias_correction']:
                bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
            else:
                bias_correction = 1.0

            # Effective step magnitude for this group.
            dlr = (d if adaptive else fixed_lr) * bias_correction
            group['_dlr'] = dlr

            numerator_acc = None
            denominator_acc = None

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ChaosGrad does not support sparse gradients.")

                if group['zero_diag'] and grad.dim() == 2 and grad.shape[0] == grad.shape[1]:
                    # Defensive: the model registers a grad hook for this, but
                    # the optimizer must not rely on model-side wiring.
                    grad = grad.clone()
                    grad.fill_diagonal_(0.0)

                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if adaptive:
                        state['s'] = torch.zeros_like(p)
                        state['p0'] = p.detach().clone()

                state['step'] += 1
                state['_grad_proc'] = grad

                # Adam EMAs. Gradients are scaled by d (Prodigy convention) so
                # the ratio exp_avg / sqrt(exp_avg_sq) stays scale-consistent
                # while d evolves.
                d_scale = d if adaptive else 1.0
                state['exp_avg'].mul_(beta1).add_(grad, alpha=d_scale * (1 - beta1))
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=d_scale * d_scale * (1 - beta2))

                if adaptive:
                    s, p0 = state['s'], state['p0']
                    delta = (grad * (p0 - p)).sum() * ((d / d0) * dlr)
                    numerator_acc = delta if numerator_acc is None else numerator_acc + delta

                    s_alpha = (d / d0) * (d if group['safeguard_warmup'] else dlr)
                    s.mul_(beta3).add_(grad, alpha=s_alpha)
                    part = s.abs().sum()
                    denominator_acc = part if denominator_acc is None else denominator_acc + part

            if denominator_acc is not None:
                adaptive_groups.append(group)
                numerator_deltas.append(numerator_acc)
                denominators.append(denominator_acc)

        # ---- Single host sync: resolve step-scale estimates ----
        if adaptive_groups:
            n = len(adaptive_groups)
            stacked = torch.stack(
                [t.float() for t in numerator_deltas] + [t.float() for t in denominators]
            ).cpu().tolist()

            if adaptive_groups[0]['d_mode'] == 'global':
                self._update_d(adaptive_groups, sum(stacked[:n]), sum(stacked[n:]))
            else:
                for i, group in enumerate(adaptive_groups):
                    self._update_d([group], stacked[i], stacked[n + i])

        # ---- Pass 2: parameter updates ----
        for group in self.param_groups:
            decay = group['weight_decay']
            eps = group['eps']
            beta1 = group['betas'][0]
            adaptive = group['lr'] is None
            d = group['d']
            dlr = group.pop('_dlr')

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = state.pop('_grad_proc')

                d_eps = (d if adaptive else 1.0) * eps
                denom = state['exp_avg_sq'].sqrt().add_(d_eps)

                if decay != 0.0:
                    p.add_(p, alpha=-decay * dlr)

                if beta1 > 0:
                    p.addcdiv_(state['exp_avg'], denom, value=-dlr)
                else:
                    scale = d if adaptive else 1.0
                    p.addcdiv_(grad, denom, value=-dlr * scale)

                if group['zero_diag'] and p.dim() == 2 and p.shape[0] == p.shape[1]:
                    p.fill_diagonal_(0.0)

            group['k'] += 1

        return loss

    #: Groups whose initial RMS falls below this are treated as
    #: deliberately silent inits (zeros / micro-quiet) and do not anchor
    #: the traction limit; near-1.0 RMS groups are neutral multiplier
    #: inits (scales, norms) and are excluded for the same reason.
    _TRUST_RMS_FLOOR = 1e-3

    def _trust_cap(self, groups):
        """Traction limit for a set of groups sharing a d estimate."""
        trust_ratio = groups[0]['trust_ratio']
        if trust_ratio is None:
            return None
        anchors = [
            g['rms0'] for g in groups
            if g.get('rms0', 0.0) >= self._TRUST_RMS_FLOOR and not 0.9 <= g['rms0'] <= 1.1
        ]
        return trust_ratio * min(anchors) if anchors else None

    def _update_d(self, groups, numerator_delta, denominator):
        """
        Applies one D-adaptation update to a set of groups sharing an
        estimate. All groups in the set are kept in sync so the estimate
        survives ``state_dict`` round-trips regardless of group count.
        """
        lead = groups[0]
        beta3 = self._beta3(lead)
        numerator = lead['d_numerator'] * beta3 + numerator_delta

        d = lead['d']
        d_max = lead['d_max']
        if denominator > 0.0:
            d_hat = lead['d_coef'] * numerator / denominator
            if d == lead['d0']:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * lead['growth_rate'])

        # Traction limit: the applied step scale never exceeds a fixed
        # fraction of the network's initial weight scale. Shields tiny
        # chaotic networks from early estimator overshoot; transparent
        # whenever the estimate is already proportionate.
        cap = self._trust_cap(groups)
        if cap is not None:
            d = min(d, cap)

        for group in groups:
            group['d_numerator'] = numerator
            group['d'] = d
            group['d_max'] = d_max

    # ------------------------------------------------------------------ #
    # Loss-spike brake                                                    #
    # ------------------------------------------------------------------ #

    #: EWMA smoothing for the loss stream (≈ 20-step memory).
    _BRAKE_EMA_ALPHA = 0.05
    #: A spike is a loss exceeding EWMA by 3 sigma AND by 20 percent.
    _BRAKE_SIGMA = 3.0
    _BRAKE_RATIO = 1.2

    def report_loss(self, loss_value):
        """
        Feed the training loss to the spike brake. Called automatically by
        ``OdyssNetTrainer`` after every optimization step; call it manually
        in custom loops to enable the brake.

        A statistical spike (loss above the running EWMA by 3 sigma and by
        20 percent) signals that the current step scale has outgrown the
        sharpening landscape. The estimator's numerator and ratchet ceiling
        are scaled down by ``brake_factor``, and the estimate re-grows only
        if the landscape supports it.
        """
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        if not math.isfinite(loss_value):
            return

        if self._loss_ema is None:
            self._loss_ema = loss_value
            self._loss_var = 0.0
            return

        diff = loss_value - self._loss_ema
        std = math.sqrt(self._loss_var) + 1e-12
        is_spike = (
            diff > self._BRAKE_SIGMA * std
            and loss_value > self._BRAKE_RATIO * abs(self._loss_ema)
        )

        alpha = self._BRAKE_EMA_ALPHA
        self._loss_ema += alpha * diff
        self._loss_var = (1 - alpha) * (self._loss_var + alpha * diff * diff)

        if is_spike:
            self._apply_brake()
            # Re-seed the statistics at the spike level so a single plateau
            # shift does not trigger a brake cascade.
            self._loss_ema = loss_value
            self._loss_var = 0.0

    def _apply_brake(self):
        for group in self.param_groups:
            brake = group['brake_factor']
            if brake is None or group['lr'] is not None:
                continue
            d0 = group['d0']
            group['d_numerator'] *= brake
            group['d_max'] = max(d0, group['d_max'] * brake)
            group['d'] = max(d0, min(group['d'], group['d_max']))

    # ------------------------------------------------------------------ #
    # Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_diagnostics(self, debug=False):
        """
        Returns optimizer health metrics.

        Args:
            debug (bool): Include per-group detail. Default False.

        Returns:
            dict with:
                - global_step: steps taken
                - effective_lr: mean effective step scale across groups
                - groups (debug): per-family stats
        """
        d_values = []
        group_stats = []
        steps = 0
        for group in self.param_groups:
            adaptive = group['lr'] is None
            d_eff = group['d'] if adaptive else group['lr']
            d_values.append(d_eff)
            steps = max(steps, group['k'])
            if debug:
                group_stats.append({
                    'group_name': group.get('group_name', 'default'),
                    'param_count': sum(p.numel() for p in group['params']),
                    'adaptive': adaptive,
                    'effective_lr': d_eff,
                    'd_max': group['d_max'],
                    'weight_decay': group['weight_decay'],
                })

        diag = {
            'global_step': steps,
            'effective_lr': sum(d_values) / max(len(d_values), 1),
        }
        if debug:
            diag['groups'] = group_stats
        return diag
