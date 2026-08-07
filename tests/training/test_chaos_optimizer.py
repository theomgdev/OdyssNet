"""
Unit tests for odyssnet.training.chaos_optimizer.ChaosGrad.

Covers:
- Constructor validation
- Parameter classification (families, policy)
- from_model construction
- Adaptive stepping (d growth, convergence)
- Fixed-lr mode (AdamW-equivalent escape hatch)
- Zero-diagonal invariant on the chaos core
- d_mode global vs per_group
- state_dict round-trip
- Diagnostics
- Neurogenesis expansion migration
"""

import pytest
import torch

from odyssnet import ChaosGrad, OdyssNet, OdyssNetTrainer


def _model(**kwargs):
    defaults = dict(num_neurons=6, input_ids=[0, 1], output_ids=[4, 5], device="cpu")
    defaults.update(kwargs)
    return OdyssNet(**defaults)


def _train_steps(model, opt, steps=5, thinking=3):
    torch.manual_seed(0)
    x = torch.randn(4, model.num_neurons)
    losses = []
    for _ in range(steps):
        model.reset_state(4)
        out, _ = model(x, steps=thinking)
        loss = out[:, -1, model.output_ids].pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ===========================================================================
# Constructor validation
# ===========================================================================

class TestValidation:
    def test_negative_lr_rejected(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), lr=-1.0)

    def test_zero_lr_rejected(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), lr=0.0)

    def test_none_lr_accepted(self):
        m = _model()
        opt = ChaosGrad(m.parameters(), lr=None)
        assert opt.param_groups[0]['lr'] is None

    def test_invalid_d0_rejected(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), d0=0.0)

    def test_invalid_betas_rejected(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), betas=(1.0, 0.999))
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), betas=(0.9, 1.0))

    def test_invalid_growth_rate_rejected(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), growth_rate=1.0)

    def test_invalid_d_mode_rejected(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), d_mode='local')

    def test_invalid_trust_ratio_rejected(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), trust_ratio=0.0)

    def test_trust_ratio_none_accepted(self):
        m = _model()
        opt = ChaosGrad.from_model(m, trust_ratio=None)
        assert opt.param_groups[0]['trust_ratio'] is None


# ===========================================================================
# Parameter classification
# ===========================================================================

class TestClassification:
    def test_families_cover_all_params(self):
        m = _model(hebb_type='both', gate='sigmoid', vocab_size=7)
        groups = ChaosGrad.classify_params(m)
        grouped = sum(len(g['params']) for g in groups)
        total = sum(1 for p in m.parameters() if p.requires_grad)
        assert grouped == total

    def test_core_family_flags(self):
        m = _model()
        groups = ChaosGrad.classify_params(m)
        by_name = {g['group_name']: g for g in groups}
        core = by_name['chaos_core']
        assert core['zero_diag'] is True
        assert core['weight_decay'] == ChaosGrad.FAMILY_WEIGHT_DECAY['chaos_core']
        assert core['params'][0] is m.W

    def test_plasticity_family_has_no_decay(self):
        m = _model(hebb_type='both')
        by_name = {g['group_name']: g for g in ChaosGrad.classify_params(m)}
        assert by_name['plasticity']['weight_decay'] == 0.0
        leaves = {id(p) for p in by_name['plasticity']['params']}
        assert id(m.t_hebb_factor) in leaves
        assert id(m.s_hebb_decay) in leaves

    def test_projections_family(self):
        m = _model(vocab_size=7)
        by_name = {g['group_name']: g for g in ChaosGrad.classify_params(m)}
        proj_params = {id(p) for p in by_name['projections']['params']}
        assert id(m.embed.weight) in proj_params
        assert id(m.output_decoder.weight) in proj_params

    def test_from_model_builds_chaosgrad(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        assert isinstance(opt, ChaosGrad)
        assert all(g['lr'] is None for g in opt.param_groups)

    def test_from_model_with_fixed_lr(self):
        m = _model()
        opt = ChaosGrad.from_model(m, lr=1e-3)
        assert all(g['lr'] == 1e-3 for g in opt.param_groups)


# ===========================================================================
# Stepping
# ===========================================================================

class TestStepping:
    def test_adaptive_step_updates_params(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        before = m.W.detach().clone()
        _train_steps(m, opt, steps=3)
        assert not torch.equal(before, m.W.detach())

    def test_d_grows_from_d0(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=30)
        d = opt.param_groups[0]['d']
        assert d > opt.param_groups[0]['d0']

    def test_fixed_lr_mode_converges(self):
        torch.manual_seed(1)
        target = torch.randn(10)
        p = torch.nn.Parameter(torch.zeros(10))
        opt = ChaosGrad([p], lr=0.1)
        for _ in range(200):
            loss = (p - target).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert (p - target).pow(2).mean().item() < 1e-2

    def test_adaptive_mode_converges_on_quadratic(self):
        torch.manual_seed(1)
        target = torch.randn(10)
        p = torch.nn.Parameter(torch.zeros(10))
        opt = ChaosGrad([p])
        for _ in range(300):
            loss = (p - target).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert (p - target).pow(2).mean().item() < 1e-2

    def test_zero_diag_invariant(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=5)
        assert m.W.detach().diagonal().abs().sum().item() == 0.0

    def test_per_group_mode_runs(self):
        m = _model()
        opt = ChaosGrad.from_model(m, d_mode='per_group')
        losses = _train_steps(m, opt, steps=10)
        assert all(torch.isfinite(torch.tensor(losses)))

    def test_global_mode_shares_d(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=20)
        d_values = {g['d'] for g in opt.param_groups}
        assert len(d_values) == 1

    def test_sparse_grad_rejected(self):
        p = torch.nn.Parameter(torch.zeros(4, 4))
        opt = ChaosGrad([p])
        p.grad = torch.eye(4).to_sparse()
        with pytest.raises(RuntimeError):
            opt.step()

    def test_step_with_no_grads_is_noop(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        opt.step()  # no backward called; must not raise
        assert opt.param_groups[0]['d'] == opt.param_groups[0]['d0']

    def test_trust_cap_bounds_d(self):
        # Chaos core is initialized with std 0.02 ('quiet'), so the applied
        # step scale must never exceed trust_ratio * rms0 of the smallest
        # anchoring family, regardless of estimator overshoot.
        m = _model()
        opt = ChaosGrad.from_model(m)
        cap = opt._trust_cap([g for g in opt.param_groups if g['lr'] is None])
        assert cap is not None and 0.0 < cap < 0.05
        _train_steps(m, opt, steps=50)
        assert opt.param_groups[0]['d'] <= cap

    def test_trust_cap_anchors_recorded(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        for group in opt.param_groups:
            assert 'rms0' in group

    def test_anchor_weight_matches_old_hard_regions(self):
        # Deep inside what used to be the hard-included region, weight must
        # still be (near) 1; deep inside what used to be excluded, (near) 0.
        assert ChaosGrad._anchor_weight(0.02) == pytest.approx(1.0, abs=1e-6)
        assert ChaosGrad._anchor_weight(0.5) == pytest.approx(1.0, abs=1e-6)
        assert ChaosGrad._anchor_weight(1e-6) == pytest.approx(0.0, abs=1e-6)
        assert ChaosGrad._anchor_weight(1.0) == pytest.approx(0.0, abs=1e-6)
        assert ChaosGrad._anchor_weight(0.0) == 0.0

    @pytest.mark.parametrize("boundary", [1e-3, 1e-2, 0.9, 1.0, 1.1])
    def test_anchor_weight_continuous_across_old_hard_boundaries(self, boundary):
        # The old implementation had a hard jump (0 -> 1 or 1 -> 0) exactly at
        # each of these values. A tiny nudge across any of them must now
        # produce a tiny change in weight, not a jump.
        eps = 1e-6
        w_below = ChaosGrad._anchor_weight(boundary - eps)
        w_at = ChaosGrad._anchor_weight(boundary)
        w_above = ChaosGrad._anchor_weight(boundary + eps)
        assert abs(w_at - w_below) < 1e-3
        assert abs(w_above - w_at) < 1e-3

    def test_trust_cap_no_jump_sweeping_through_old_boundaries(self):
        # A fixed, clearly-anchoring group (rms0=0.02, deep inside the old
        # hard-included region) plus a probe group swept finely across every
        # old cutoff (0.001, 0.9, 1.0, 1.1). A hard cutoff concentrates its
        # whole effect into a single infinitesimal step; a smooth ramp spreads
        # it across the transition width instead -- so no single step should
        # account for more than a modest share of the total range the sweep
        # covers.
        fixed = 0.02
        trust_ratio = 0.25
        sweep = [x / 1000.0 for x in range(1, 1200)]  # 0.001 .. 1.199
        opt = ChaosGrad([torch.nn.Parameter(torch.zeros(1))], trust_ratio=trust_ratio)
        caps = []
        for probe in sweep:
            groups = [
                {'trust_ratio': trust_ratio, 'rms0': fixed},
                {'trust_ratio': trust_ratio, 'rms0': probe},
            ]
            caps.append(opt._trust_cap(groups))
        assert all(c is not None for c in caps)
        full_range = max(caps) - min(caps)
        assert full_range > 0  # sanity: the probe does move the cap
        max_step = max(abs(b - a) for a, b in zip(caps, caps[1:]))
        assert max_step < 0.5 * full_range

    def test_closure_is_called(self):
        p = torch.nn.Parameter(torch.ones(3))
        opt = ChaosGrad([p])
        called = []

        def closure():
            called.append(True)
            loss = p.sum()
            opt.zero_grad()
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert called and loss is not None


# ===========================================================================
# Loss-spike brake
# ===========================================================================

class TestBrake:
    def test_invalid_brake_factor_rejected(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), brake_factor=1.0)
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), brake_factor=0.0)

    def test_spike_shrinks_applied_step(self):
        # Intent changed, not just implementation: the brake used to mutate
        # `d`/`d_numerator` directly, which permanently scarred the D-adaptation
        # estimator on every trigger -- fatal on tasks where ordinary minibatch
        # noise crosses the spike threshold every few hundred steps (see
        # `_apply_brake`'s docstring). It now throttles a separate transient
        # `brake_ceiling` and leaves the estimator's own bookkeeping untouched,
        # so it keeps learning the true scale even while suppressed.
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=20)
        # Establish a calm loss stream, then spike it.
        for _ in range(30):
            opt.report_loss(1.0)
        d_before = opt.param_groups[0]['d']
        num_before = opt.param_groups[0]['d_numerator']
        ceiling_before = opt.param_groups[0]['brake_ceiling']
        opt.report_loss(100.0)
        assert opt.param_groups[0]['brake_ceiling'] < ceiling_before
        assert opt.param_groups[0]['d'] == d_before
        assert opt.param_groups[0]['d_numerator'] == num_before

    def test_ceiling_relaxes_after_isolated_spike(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=20)
        for _ in range(30):
            opt.report_loss(1.0)
        opt.report_loss(100.0)
        ceiling_after_spike = opt.param_groups[0]['brake_ceiling']
        assert ceiling_after_spike < 1.0
        # An isolated spike should heal: many calm calls relax the ceiling
        # back up rather than leaving it permanently depressed.
        for _ in range(300):
            opt.report_loss(1.0)
        assert opt.param_groups[0]['brake_ceiling'] > ceiling_after_spike

    def test_cascading_spikes_keep_compounding(self):
        # A fast cascade (genuine divergence, like the delayed-adder failure
        # this brake exists for) shouldn't get time to relax between hits.
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=20)
        for _ in range(30):
            opt.report_loss(1.0)
        opt.report_loss(100.0)
        ceiling_after_first = opt.param_groups[0]['brake_ceiling']
        opt.report_loss(1000.0)
        assert opt.param_groups[0]['brake_ceiling'] < ceiling_after_first

    def test_steady_loss_never_brakes(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=20)
        d_before = opt.param_groups[0]['d']
        for v in (1.0, 0.99, 1.01, 0.98, 1.0, 0.97):
            opt.report_loss(v)
        assert opt.param_groups[0]['d'] == d_before

    def test_nan_loss_ignored(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        opt.report_loss(float('nan'))
        opt.report_loss(1.0)  # must not raise

    def test_brake_disabled(self):
        m = _model()
        opt = ChaosGrad.from_model(m, brake_factor=None)
        _train_steps(m, opt, steps=10)
        d_before = opt.param_groups[0]['d']
        for _ in range(30):
            opt.report_loss(1.0)
        opt.report_loss(1000.0)
        assert opt.param_groups[0]['d'] == d_before

    def test_trainer_feeds_brake(self):
        m = _model()
        t = OdyssNetTrainer(m, device="cpu")
        x = torch.randn(2, 6)
        y = torch.randn(2, 2)
        t.train_batch(x, y, thinking_steps=2)
        assert t.optimizer._loss_ema is not None


# ===========================================================================
# Persistence
# ===========================================================================

class TestPersistence:
    def test_state_dict_round_trip(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=25)
        d_before = opt.param_groups[0]['d']

        sd = opt.state_dict()
        opt2 = ChaosGrad.from_model(m)
        opt2.load_state_dict(sd)

        assert opt2.param_groups[0]['d'] == d_before
        assert opt2.param_groups[0]['k'] == opt.param_groups[0]['k']
        # Per-parameter tensors restored
        p = opt2.param_groups[0]['params'][0]
        assert 'exp_avg' in opt2.state[p]
        assert 's' in opt2.state[p]

    def test_no_transient_keys_in_state_dict(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=2)
        sd = opt.state_dict()
        for group in sd['param_groups']:
            assert '_dlr' not in group
        for state in sd['state'].values():
            assert '_grad_proc' not in state


# ===========================================================================
# Diagnostics
# ===========================================================================

class TestDiagnostics:
    def test_basic_diagnostics(self):
        m = _model()
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=3)
        diag = opt.get_diagnostics()
        assert diag['global_step'] == 3
        assert diag['effective_lr'] > 0

    def test_debug_diagnostics_groups(self):
        m = _model(hebb_type='both')
        opt = ChaosGrad.from_model(m)
        _train_steps(m, opt, steps=2)
        diag = opt.get_diagnostics(debug=True)
        names = {g['group_name'] for g in diag['groups']}
        assert 'chaos_core' in names
        assert 'plasticity' in names

    def test_trainer_diag_includes_optimizer(self):
        m = _model()
        t = OdyssNetTrainer(m, device="cpu")
        x = torch.randn(2, 6)
        y = torch.randn(2, 2)
        t.train_batch(x, y, thinking_steps=2)
        diag = t.get_diagnostics()
        assert 'optimizer' in diag
        assert diag['current_lr'] == diag['optimizer']['effective_lr']


# ===========================================================================
# Neurogenesis migration
# ===========================================================================

class TestNeurogenesisMigration:
    def test_expand_preserves_chaosgrad(self):
        m = _model()
        t = OdyssNetTrainer(m, device="cpu")
        x = torch.randn(2, 6)
        y = torch.randn(2, 2)
        for _ in range(10):
            t.train_batch(x, y, thinking_steps=2)
        d_before = t.optimizer.param_groups[0]['d']

        t.expand(amount=2, verbose=False)

        assert isinstance(t.optimizer, ChaosGrad)
        assert m.num_neurons == 8
        assert t.optimizer.param_groups[0]['d'] == d_before
        # Training continues without error on the expanded model
        x2 = torch.randn(2, 8)
        loss = t.train_batch(x2, y, thinking_steps=2)
        assert torch.isfinite(torch.tensor(loss))

    def test_expand_preserves_custom_brake_config(self):
        # Neurogenesis rebuilds ChaosGrad via `from_model` on expansion; if it
        # doesn't forward the brake config a user constructed with, the
        # rebuilt optimizer silently reverts to the defaults post-expansion.
        m = _model()
        opt = ChaosGrad.from_model(
            m, brake_factor=0.3, brake_sigma=2.5, brake_ratio=1.5, brake_ema_alpha=0.1,
        )
        t = OdyssNetTrainer(m, optimizer=opt, device="cpu")
        x = torch.randn(2, 6)
        y = torch.randn(2, 2)
        t.train_batch(x, y, thinking_steps=2)

        t.expand(amount=2, verbose=False)

        assert all(g['brake_factor'] == 0.3 for g in t.optimizer.param_groups)
        assert t.optimizer._brake_sigma == 2.5
        assert t.optimizer._brake_ratio == 1.5
        assert t.optimizer._brake_ema_alpha == 0.1

    def test_expand_preserves_fixed_lr(self):
        m = _model()
        t = OdyssNetTrainer(m, device="cpu", lr=5e-4)
        x = torch.randn(2, 6)
        y = torch.randn(2, 2)
        t.train_batch(x, y, thinking_steps=2)
        t.expand(amount=1, verbose=False)
        assert isinstance(t.optimizer, ChaosGrad)
        assert all(g['lr'] == 5e-4 for g in t.optimizer.param_groups)
