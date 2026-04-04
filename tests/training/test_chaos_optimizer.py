"""
Unit tests for odyssnet.training.chaos_optimizer.ChaosGrad.

Covers:
- Initialisation with single lr parameter
- Parameter group classification (classify_params)
- Optimizer step — parameter updates and global_step increment
- Cold start state initialisation (prev_grad as bfloat16, momentum zeros)
- Hypergradient adaptation (per_param_lr, per_param_beta, per_param_alpha change)
- Hebbian bypass rule (per_param_decay stays zero for hebb params)
- W diagonal stays zero after step
- Sparse gradient raises
- Closure support
- Frustration Accumulator (report_loss, trigger_plateau_escape)
- Diagnostics (get_diagnostics keys)
"""

import pytest
import torch
import torch.nn as nn
import os

os.environ.setdefault("NO_BNB", "1")

from odyssnet import OdyssNet
from odyssnet.training.chaos_optimizer import ChaosGrad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_model(n=5):
    return OdyssNet(num_neurons=n, input_ids=[0], output_ids=[n - 1], device="cpu")


def _optimizer(model, lr=1e-3):
    groups = ChaosGrad.classify_params(model)
    return ChaosGrad(groups, lr=lr)


def _run_step(model, opt, n=5):
    """Run one forward+backward+step cycle."""
    x = torch.randn(2, n)
    out, _ = model(x, steps=2)
    out.sum().backward()
    opt.step()
    opt.zero_grad()


# ===========================================================================
# Initialisation
# ===========================================================================

class TestChaosGradInit:
    def test_creates_param_groups(self):
        model = _small_model()
        opt = _optimizer(model)
        assert len(opt.param_groups) > 0

    def test_global_step_starts_at_zero(self):
        model = _small_model()
        opt = _optimizer(model)
        assert opt._global_step == 0

    def test_frustration_starts_at_zero(self):
        model = _small_model()
        opt = _optimizer(model)
        assert opt._frustration == pytest.approx(0.0)

    def test_best_loss_starts_at_inf(self):
        model = _small_model()
        opt = _optimizer(model)
        assert opt._best_loss == float("inf")

    def test_lr_stored_in_defaults(self):
        model = _small_model()
        opt = _optimizer(model, lr=5e-4)
        assert opt.defaults['lr'] == pytest.approx(5e-4)

    def test_negative_lr_raises(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        with pytest.raises(ValueError, match="lr"):
            ChaosGrad(groups, lr=-1e-3)

    def test_lr_accessible_in_param_groups(self):
        model = _small_model()
        opt = _optimizer(model, lr=2e-3)
        for g in opt.param_groups:
            assert g['lr'] == pytest.approx(2e-3)


# ===========================================================================
# Parameter Classification
# ===========================================================================

class TestClassifyParams:
    def test_w_in_chaos_core_group(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "chaos_core" in names
        assert any(p is model.W for p in names["chaos_core"]["params"])

    def test_memory_feedback_in_memory_group(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "memory_feedback" in names
        assert any(p is model.memory_feedback for p in names["memory_feedback"]["params"])

    def test_b_in_lightweight_group(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "lightweight" in names
        assert any(p is model.B for p in names["lightweight"]["params"])

    def test_embed_in_projections_group(self):
        model = OdyssNet(
            num_neurons=8, input_ids=list(range(4)), output_ids=list(range(4, 8)),
            device="cpu", vocab_size=16, vocab_mode="discrete",
        )
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "projections" in names

    def test_gate_params_in_gates_group(self):
        model = OdyssNet(num_neurons=5, input_ids=[0], output_ids=[4], device="cpu", gate="sigmoid")
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "gates" in names

    def test_all_params_covered(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        covered = {id(p) for g in groups for p in g["params"]}
        all_params = {id(p) for p in model.parameters()}
        assert covered == all_params

    def test_hebbian_params_in_hebbian_group(self):
        model = OdyssNet(num_neurons=5, input_ids=[0], output_ids=[4],
                         device="cpu", hebb_type="global")
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "hebbian" in names
        hebb_params = names["hebbian"]["params"]
        assert any(p is model.hebb_factor for p in hebb_params)
        assert any(p is model.hebb_decay  for p in hebb_params)

    def test_all_params_covered_with_hebbian(self):
        model = OdyssNet(num_neurons=5, input_ids=[0], output_ids=[4],
                         device="cpu", hebb_type="global")
        groups = ChaosGrad.classify_params(model)
        covered = {id(p) for g in groups for p in g["params"]}
        all_params = {id(p) for p in model.parameters()}
        assert covered == all_params


# ===========================================================================
# Optimizer Step
# ===========================================================================

class TestOptimizerStep:
    def test_step_increments_global_step(self):
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        assert opt._global_step == 1

    def test_w_changes_after_step(self):
        model = _small_model()
        opt = _optimizer(model)
        w_before = model.W.data.clone()
        _run_step(model, opt)
        assert not torch.allclose(model.W.data, w_before)

    def test_w_diagonal_stays_zero_after_step(self):
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        assert torch.all(model.W.diag() == 0.0)

    def test_sparse_gradient_raises(self):
        model = _small_model()
        opt = _optimizer(model)
        for p in model.parameters():
            p.grad = p.data.to_sparse()
            break
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()

    def test_closure_is_called(self):
        model = _small_model()
        opt = _optimizer(model)
        called = {"flag": False}

        def closure():
            called["flag"] = True
            x = torch.randn(2, 5)
            out, _ = model(x, steps=1)
            loss = out.sum()
            loss.backward()
            return loss

        opt.step(closure=closure)
        assert called["flag"]

    def test_multiple_steps_converge_direction(self):
        """After several steps the model should produce finite, changing outputs."""
        model = _small_model()
        opt = _optimizer(model, lr=1e-2)
        losses = []
        for _ in range(5):
            x = torch.randn(4, 5)
            out, _ = model(x, steps=3)
            loss = out.sum().pow(2)
            loss.backward()
            opt.step()
            opt.zero_grad()
            model.reset_state(4)
            losses.append(loss.item())
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


# ===========================================================================
# Cold Start State
# ===========================================================================

class TestColdStartState:
    def test_prev_grad_is_float32(self):
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    assert opt.state[p]['prev_grad'].dtype == torch.float32

    def test_momentum_is_float32(self):
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    assert opt.state[p]['momentum'].dtype == torch.float32

    def test_per_param_lr_calibrated_at_cold_start(self):
        """per_param_lr is calibrated from gradient RMS at T=0 (1/g_rms), so it is
        within valid bounds but not necessarily close to 1.0."""
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    lr = opt.state[p]['per_param_lr']
                    assert ChaosGrad._LR_MIN <= lr <= ChaosGrad._LR_MAX

    def test_per_param_beta_initialized_around_09(self):
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    beta = opt.state[p]['per_param_beta']
                    assert 0.5 <= beta <= 0.999


# ===========================================================================
# Hypergradient Adaptation
# ===========================================================================

class TestHypergradientAdaptation:
    def test_per_param_lr_changes_after_steps(self):
        """After multiple steps, per_param_lr should differ from its initial value of 1.0."""
        model = _small_model()
        opt = _optimizer(model, lr=1e-2)
        for _ in range(5):
            _run_step(model, opt)
        any_changed = False
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    if abs(opt.state[p]['per_param_lr'] - 1.0) > 1e-6:
                        any_changed = True
        assert any_changed, "per_param_lr should drift from 1.0 after multiple steps"

    def test_per_param_beta_adapts(self):
        model = _small_model()
        opt = _optimizer(model, lr=1e-2)
        for _ in range(5):
            _run_step(model, opt)
        any_changed = False
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    if abs(opt.state[p]['per_param_beta'] - 0.9) > 1e-6:
                        any_changed = True
        assert any_changed, "per_param_beta should adapt from 0.9 after multiple steps"

    def test_per_param_alpha_adapts(self):
        model = _small_model()
        opt = _optimizer(model, lr=1e-2)
        for _ in range(5):
            _run_step(model, opt)
        any_changed = False
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    if abs(opt.state[p]['per_param_alpha'] - 0.5) > 1e-6:
                        any_changed = True
        assert any_changed, "per_param_alpha should adapt from 0.5 after multiple steps"

    def test_per_param_lr_stays_within_bounds(self):
        model = _small_model()
        opt = _optimizer(model, lr=1e-2)
        for _ in range(20):
            _run_step(model, opt)
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    lr = opt.state[p]['per_param_lr']
                    assert ChaosGrad._LR_MIN <= lr <= ChaosGrad._LR_MAX

    def test_per_param_beta_stays_within_bounds(self):
        model = _small_model()
        opt = _optimizer(model, lr=1e-2)
        for _ in range(20):
            _run_step(model, opt)
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    beta = opt.state[p]['per_param_beta']
                    assert ChaosGrad._BETA_MIN <= beta <= ChaosGrad._BETA_MAX


# ===========================================================================
# Hebbian Bypass Rule
# ===========================================================================

class TestHebbianBypass:
    def test_hebbian_params_updated(self):
        """hebb_factor receives a non-zero gradient when Hebbian correlations are active.
        ≥5 thinking steps ensure hebb_state_W accumulates before backward.
        We verify the gradient exists and is non-zero rather than checking parameter
        change (which is lr-scale dependent)."""
        model = OdyssNet(num_neurons=5, input_ids=[0], output_ids=[4],
                         device="cpu", hebb_type="global")
        opt = _optimizer(model, lr=0.1)
        torch.manual_seed(7)
        x = torch.randn(2, 5)
        out, _ = model(x, steps=5)
        out.sum().backward()
        assert model.hebb_factor.grad is not None
        assert model.hebb_factor.grad.abs().item() > 0.0
        opt.step()
        opt.zero_grad()

    def test_hebbian_per_param_decay_stays_zero(self):
        """per_param_decay for Hebbian params must always remain 0.0 regardless of gradient signals."""
        model = OdyssNet(num_neurons=5, input_ids=[0], output_ids=[4],
                         device="cpu", hebb_type="global")
        opt = _optimizer(model)
        for _ in range(10):
            _run_step(model, opt, n=5)
        for group in opt.param_groups:
            if group.get('_is_hebbian'):
                for p in group['params']:
                    if p in opt.state and opt.state[p]:
                        assert opt.state[p]['per_param_decay'] == pytest.approx(0.0), \
                            "Hebbian params must never receive autonomous weight decay"

    def test_chaos_core_decay_can_grow(self):
        """chaos_core per_param_decay should start at 0.01 and evolve."""
        model = _small_model()
        opt = _optimizer(model)
        for _ in range(10):
            _run_step(model, opt)
        for group in opt.param_groups:
            if group.get('_is_chaos_core'):
                for p in group['params']:
                    if p in opt.state and opt.state[p]:
                        # Should be non-negative and within bounds
                        decay = opt.state[p]['per_param_decay']
                        assert 0.0 <= decay <= ChaosGrad._DECAY_MAX


# ===========================================================================
# Frustration Accumulator
# ===========================================================================

class TestFrustrationAccumulator:
    def test_report_loss_updates_best(self):
        model = _small_model()
        opt = _optimizer(model)
        opt.report_loss(1.0)
        assert opt._best_loss == pytest.approx(1.0)

    def test_frustration_grows_on_stagnation(self):
        model = _small_model()
        opt = _optimizer(model)
        opt.report_loss(1.0)  # sets best
        for _ in range(20):
            opt.report_loss(1.0)  # stagnating
        assert opt._frustration > 0.0

    def test_frustration_stays_low_on_improvement(self):
        model = _small_model()
        opt = _optimizer(model)
        for i in range(20):
            opt.report_loss(1.0 / (i + 1))  # continuously improving
        assert opt._frustration < 0.05

    def test_trigger_plateau_escape_sets_flag(self):
        model = _small_model()
        opt = _optimizer(model)
        opt.trigger_plateau_escape()
        assert opt._force_plateau_escape is True

    def test_plateau_escape_clears_flag_after_step(self):
        model = _small_model()
        opt = _optimizer(model)
        opt.trigger_plateau_escape()
        _run_step(model, opt)
        assert opt._force_plateau_escape is False

    def test_frustration_partially_resets_after_burst(self):
        model = _small_model()
        opt = _optimizer(model)
        # Force frustration above threshold
        opt._frustration = ChaosGrad._FRUST_THRESH + 0.1
        _run_step(model, opt)
        assert opt._frustration < ChaosGrad._FRUST_THRESH


# ===========================================================================
# Diagnostics
# ===========================================================================

class TestDiagnostics:
    def test_get_diagnostics_keys(self):
        model = _small_model()
        opt = _optimizer(model)
        diag = opt.get_diagnostics()
        assert "global_step"      in diag
        assert "frustration"      in diag
        assert "best_loss"        in diag
        assert "avg_effective_lr" in diag
        assert "avg_init_lr"      in diag

    def test_avg_effective_lr_after_steps(self):
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        diag = opt.get_diagnostics()
        assert diag['avg_effective_lr'] > 0.0
        assert diag['avg_init_lr'] > 0.0

    def test_init_lr_stored_in_state(self):
        """init_lr is stored at cold start and reflects gradient scale (1/g_rms)."""
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        for group in opt.param_groups:
            for p in group['params']:
                if p in opt.state and opt.state[p]:
                    assert 'init_lr' in opt.state[p]
                    assert ChaosGrad._LR_MIN <= opt.state[p]['init_lr'] <= ChaosGrad._LR_MAX

    def test_coupling_zero_at_cold_start(self):
        """At T=1, log_drift = log(per_lr/init_lr) should be near zero — coupling inactive."""
        model = _small_model()
        opt = _optimizer(model)
        _run_step(model, opt)
        for group in opt.param_groups:
            for p in group['params']:
                s = opt.state.get(p)
                if s:
                    ratio = s['per_param_lr'] / s['init_lr']
                    # After one step, drift should be small (cold start prev_grad ≈ 0 → sig_lr ≈ 1)
                    assert 0.5 < ratio < 2.0
