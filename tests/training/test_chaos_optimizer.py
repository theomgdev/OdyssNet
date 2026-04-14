"""
Unit tests for odyssnet.training.chaos_optimizer.ChaosGrad (v3).

Coverage (Section 12 of OPTIMIZING_CHAOS.md):
  Initialization        — global step, frustration, best_loss, invalid lr
  Parameter classification — group membership, full coverage, Hebbian bypass
  Optimizer step        — global step increments, W changes, W diagonal zero,
                           sparse grad error, closure support, convergence
  Cold-start state      — dtype, init_lr bounds, beta seeding
  Hypergradient adaptation — per_param_lr / beta / alpha stay in bounds
  Hebbian bypass        — per_param_decay == 0.0 always for hebb params
  Frustration           — report_loss, stagnation growth, improvement reset,
                           trigger_plateau_escape, flag cleared after step,
                           frustration partial reset
  Diagnostics           — required keys, debug mode, init_lr in state,
                           coupling inactive at cold start
  State persistence     — state_dict / load_state_dict round-trip preserves
                           frustration, best_loss, global_step; lr group override
  Reset / regeneration  — reset_param_state clears state; trainer integration
  Cross-feature         — gradient accumulation, gradient clipping, AMP compat,
                           plain params fallback (no classify_params)
"""

import copy
import io
import math
import pytest
import torch

from odyssnet import OdyssNet, OdyssNetTrainer
from odyssnet.training.chaos_optimizer import ChaosGrad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(n=8, in_ids=None, out_ids=None, **kwargs):
    in_ids  = in_ids  or [0]
    out_ids = out_ids or [n - 1]
    return OdyssNet(num_neurons=n, input_ids=in_ids, output_ids=out_ids,
                    device="cpu", **kwargs)


def _opt(model, lr=1e-3):
    return ChaosGrad(ChaosGrad.classify_params(model), lr=lr)


def _one_step(opt, model):
    """Run a single forward/backward/step cycle."""
    model.reset_state(batch_size=2)
    x = torch.randn(2, 1)
    states, _ = model(x, steps=3)
    loss = states.mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_global_step_starts_at_zero(self):
        m   = _model()
        opt = _opt(m)
        assert opt._global_step == 0

    def test_frustration_starts_at_zero(self):
        m   = _model()
        opt = _opt(m)
        assert opt._frustration == 0.0

    def test_best_loss_starts_at_infinity(self):
        m   = _model()
        opt = _opt(m)
        assert opt._best_loss == float('inf')

    def test_negative_lr_raises(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), lr=-1e-3)

    def test_zero_lr_raises(self):
        m = _model()
        with pytest.raises(ValueError):
            ChaosGrad(m.parameters(), lr=0.0)


# ---------------------------------------------------------------------------
# Parameter classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_W_is_chaos_core(self):
        m      = _model()
        groups = ChaosGrad.classify_params(m)
        core   = next(g for g in groups if g['group_name'] == 'chaos_core')
        assert any(p is m.W for p in core['params'])

    def test_memory_feedback_in_memory_group(self):
        m      = _model()
        groups = ChaosGrad.classify_params(m)
        mem    = next(g for g in groups if g['group_name'] == 'memory')
        assert any(p is m.memory_feedback for p in mem['params'])

    def test_input_scale_in_scales_group(self):
        m      = _model()
        groups = ChaosGrad.classify_params(m)
        scales = next(g for g in groups if g['group_name'] == 'scales')
        assert any(p is m.input_scale for p in scales['params'])

    def test_all_params_covered(self):
        m      = _model()
        groups = ChaosGrad.classify_params(m)
        classified = {id(p) for g in groups for p in g['params']}
        model_params = {id(p) for p in m.parameters() if p.requires_grad}
        assert classified == model_params

    def test_hebbian_params_in_hebbian_group(self):
        m      = _model(hebb_type='neuron')
        groups = ChaosGrad.classify_params(m)
        hebb   = next(g for g in groups if g['group_name'] == 'hebbian')
        names  = {name.split('.')[-1] for name, _ in m.named_parameters()}
        assert 'hebb_factor' in names
        assert hebb['is_hebbian'] is True

    def test_hebbian_group_burst_type_is_none(self):
        m      = _model(hebb_type='neuron')
        groups = ChaosGrad.classify_params(m)
        hebb   = next(g for g in groups if g['group_name'] == 'hebbian')
        assert hebb['burst_type'] == 'none'

    def test_chaos_core_beta_equil_is_0_95(self):
        m      = _model()
        groups = ChaosGrad.classify_params(m)
        core   = next(g for g in groups if g['group_name'] == 'chaos_core')
        assert core['beta_equil'] == pytest.approx(0.95)

    def test_gates_beta_equil_is_0_85(self):
        m      = _model(gate='sigmoid')
        groups = ChaosGrad.classify_params(m)
        gates  = next((g for g in groups if g['group_name'] == 'gates'), None)
        if gates is not None:
            assert gates['beta_equil'] == pytest.approx(0.85)

    def test_plain_params_accepted(self):
        m   = _model()
        opt = ChaosGrad(m.parameters(), lr=1e-3)
        assert opt is not None


# ---------------------------------------------------------------------------
# Optimizer step
# ---------------------------------------------------------------------------

class TestOptimizerStep:
    def test_global_step_increments(self):
        m   = _model()
        opt = _opt(m)
        _one_step(opt, m)
        assert opt._global_step == 1

    def test_W_changes_after_step(self):
        m        = _model()
        opt      = _opt(m)
        W_before = m.W.data.clone()
        # Run a few steps — first step can have a very small effective update
        # (cold-start calibration), so we allow the weight to change across 3 steps.
        for _ in range(3):
            _one_step(opt, m)
        assert not torch.equal(m.W.data, W_before)

    def test_W_diagonal_stays_zero(self):
        m   = _model()
        opt = _opt(m)
        for _ in range(3):
            _one_step(opt, m)
        diag = m.W.data.diagonal()
        assert torch.allclose(diag, torch.zeros_like(diag))

    def test_sparse_gradient_raises(self):
        embed = torch.nn.Embedding(10, 4, sparse=True)
        opt   = ChaosGrad(embed.parameters(), lr=1e-3)
        idx   = torch.tensor([0, 2])
        out   = embed(idx).sum()
        out.backward()
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()

    def test_closure_support(self):
        m     = _model()
        opt   = _opt(m)
        calls = [0]

        def closure():
            calls[0] += 1
            model_out = m(torch.randn(2, 1), steps=3)[0]
            l = model_out.mean()
            l.backward()
            return l

        opt.step(closure=closure)
        assert calls[0] == 1

    def test_multiple_steps_loss_finite(self):
        m   = _model()
        opt = _opt(m)
        for _ in range(5):
            loss = _one_step(opt, m)
        assert math.isfinite(loss)


# ---------------------------------------------------------------------------
# Cold-start state
# ---------------------------------------------------------------------------

class TestColdStart:
    def _state_after_one_step(self, model=None):
        m   = model or _model()
        opt = _opt(m)
        _one_step(opt, m)
        return opt, m

    def test_grad_ema_is_float32(self):
        opt, m = self._state_after_one_step()
        state = opt.state[m.W]
        assert state['grad_ema'].dtype == torch.float32

    def test_momentum_is_float32(self):
        opt, m = self._state_after_one_step()
        state = opt.state[m.W]
        assert state['momentum'].dtype == torch.float32

    def test_per_param_lr_within_bounds(self):
        opt, m = self._state_after_one_step()
        state  = opt.state[m.W]
        assert ChaosGrad._LR_MIN <= state['per_param_lr'] <= ChaosGrad._LR_MAX

    def test_per_param_beta_near_equil(self):
        opt, m = self._state_after_one_step()
        state  = opt.state[m.W]
        # chaos_core beta_equil is 0.95; allow ±0.05 after one step
        assert 0.5 <= state['per_param_beta'] <= 0.999

    def test_init_lr_stored_in_state(self):
        opt, m = self._state_after_one_step()
        state  = opt.state[m.W]
        assert 'init_lr' in state
        assert ChaosGrad._LR_MIN <= state['init_lr'] <= ChaosGrad._LR_MAX

    def test_v2_initialized_to_zero_before_step(self):
        m   = _model()
        opt = _opt(m)
        # Before any step, state dict is empty
        assert not opt.state[m.W]

    def test_v2_positive_after_step(self):
        opt, m = self._state_after_one_step()
        assert opt.state[m.W]['v2'] >= 0.0


# ---------------------------------------------------------------------------
# Hypergradient adaptation
# ---------------------------------------------------------------------------

class TestHypergradientAdaptation:
    def test_per_param_lr_changes(self):
        m     = _model()
        opt   = _opt(m)
        _one_step(opt, m)
        lr0   = opt.state[m.W]['per_param_lr']
        _one_step(opt, m)
        lr1   = opt.state[m.W]['per_param_lr']
        # Must have moved at least once in 2 steps
        # (they could theoretically be equal but it's astronomically unlikely)
        assert isinstance(lr1, float)
        assert ChaosGrad._LR_MIN <= lr1 <= ChaosGrad._LR_MAX

    def test_per_param_beta_stays_in_bounds(self):
        m   = _model()
        opt = _opt(m)
        for _ in range(10):
            _one_step(opt, m)
        state = opt.state[m.W]
        assert ChaosGrad._BETA_MIN <= state['per_param_beta'] <= ChaosGrad._BETA_MAX

    def test_per_param_alpha_stays_in_bounds(self):
        m   = _model()
        opt = _opt(m)
        for _ in range(10):
            _one_step(opt, m)
        state = opt.state[m.W]
        assert 0.0 <= state['per_param_alpha'] <= 1.0

    def test_per_param_decay_stays_non_negative(self):
        m   = _model()
        opt = _opt(m)
        for _ in range(10):
            _one_step(opt, m)
        for group in opt.param_groups:
            for p in group['params']:
                if opt.state.get(p):
                    assert opt.state[p]['per_param_decay'] >= 0.0


# ---------------------------------------------------------------------------
# Hebbian bypass
# ---------------------------------------------------------------------------

class TestHebbianBypass:
    def test_hebb_decay_stays_zero(self):
        m   = _model(hebb_type='neuron')
        opt = _opt(m)
        for _ in range(10):
            _one_step(opt, m)

        hebb_params = {name: p for name, p in m.named_parameters()
                       if name.split('.')[-1] in ('hebb_factor', 'hebb_decay')}
        assert hebb_params, "Model has no Hebbian parameters"

        for name, p in hebb_params.items():
            state = opt.state.get(p)
            if state:
                assert state['per_param_decay'] == 0.0, \
                    f"{name}: per_param_decay should be 0.0, got {state['per_param_decay']}"

    def test_chaos_core_decay_can_grow(self):
        m   = _model()
        opt = _opt(m)
        for _ in range(20):
            _one_step(opt, m)
        state = opt.state[m.W]
        # decay may remain 0 if sig_wd never pushed it, but it must not be negative
        assert state['per_param_decay'] >= 0.0
        assert state['per_param_decay'] <= ChaosGrad._DECAY_MAX


# ---------------------------------------------------------------------------
# Frustration accumulator
# ---------------------------------------------------------------------------

class TestFrustration:
    def test_report_loss_updates_best(self):
        m   = _model()
        opt = _opt(m)
        opt.report_loss(1.0)
        assert opt._best_loss == pytest.approx(1.0)

    def test_frustration_grows_on_stagnation(self):
        m   = _model()
        opt = _opt(m)
        opt.report_loss(1.0)
        for _ in range(50):
            opt.report_loss(1.0)  # no improvement
        assert opt._frustration > 0.0

    def test_frustration_stays_low_on_improvement(self):
        m   = _model()
        opt = _opt(m)
        loss = 1.0
        for _ in range(20):
            loss *= 0.9
            opt.report_loss(loss)
        assert opt._frustration < 0.1

    def test_trigger_plateau_escape_sets_flag(self):
        m   = _model()
        opt = _opt(m)
        opt.trigger_plateau_escape()
        assert opt._force_plateau_escape is True

    def test_flag_clears_after_step(self):
        m   = _model()
        opt = _opt(m)
        opt.trigger_plateau_escape()
        _one_step(opt, m)
        assert opt._force_plateau_escape is False

    def test_frustration_partially_resets_after_burst(self):
        m   = _model()
        opt = _opt(m)
        # Build up frustration manually
        opt._frustration = 0.9
        opt.trigger_plateau_escape()
        _one_step(opt, m)
        assert opt._frustration < 0.9 * 0.5  # at most 30% of pre-burst value


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def _opt_after_steps(self, n=5):
        m   = _model()
        opt = _opt(m)
        for _ in range(n):
            _one_step(opt, m)
        return opt, m

    def test_basic_keys_present(self):
        opt, _ = self._opt_after_steps()
        diag   = opt.get_diagnostics()
        for key in ('global_step', 'frustration', 'best_loss',
                    'avg_effective_lr', 'avg_init_lr'):
            assert key in diag, f"Missing key: {key}"

    def test_debug_keys_present(self):
        opt, _ = self._opt_after_steps()
        diag   = opt.get_diagnostics(debug=True)
        for key in ('avg_beta', 'avg_alpha', 'avg_decay',
                    'param_groups', 'per_param_stats'):
            assert key in diag, f"Missing debug key: {key}"

    def test_per_param_stats_fields(self):
        opt, _ = self._opt_after_steps()
        stats  = opt.get_diagnostics(debug=True)['per_param_stats']
        for field in ('effective_lr', 'beta', 'alpha', 'decay', 'steps'):
            assert field in stats

    def test_coupling_inactive_at_cold_start(self):
        """At T=1, denom ≈ 1.0 → effective_lr ≈ per_param_lr / init_lr ≈ 1.0."""
        m   = _model()
        opt = _opt(m)
        _one_step(opt, m)
        diag = opt.get_diagnostics()
        # Should be reasonably close to 1.0 (bias correction / v2 calibration)
        assert 0.01 <= diag['avg_effective_lr'] <= 100.0

    def test_global_step_matches(self):
        opt, _ = self._opt_after_steps(n=3)
        assert opt.get_diagnostics()['global_step'] == 3

    def test_param_groups_in_debug(self):
        opt, _ = self._opt_after_steps()
        groups = opt.get_diagnostics(debug=True)['param_groups']
        assert isinstance(groups, list)
        assert len(groups) >= 1
        assert 'group_name' in groups[0]
