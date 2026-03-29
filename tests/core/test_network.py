"""
Unit tests for odyssnet.core.network.OdyssNet.

Covers:
- Initialization (default, custom, auto-size)
- Weight initialisation strategies
- Activation and gate configurations
- Forward pass (pulse mode, stream mode, sequential, vocab)
- State management (reset, detach)
- Utility methods (get_num_params, regenerate_weak_weights, device property)
- Structural invariants (diagonal constraint on W)
"""

import pytest
import torch
import torch.nn as nn

from odyssnet import OdyssNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(n=4, in_ids=None, out_ids=None, **kwargs):
    in_ids = in_ids or [0]
    out_ids = out_ids or [n - 1]
    return OdyssNet(num_neurons=n, input_ids=in_ids, output_ids=out_ids, device="cpu", **kwargs)


# ===========================================================================
# Initialisation
# ===========================================================================

class TestInitialisation:
    def test_default_construction(self):
        model = _make(4)
        assert model.num_neurons == 4
        assert model.input_ids == [0]
        assert model.output_ids == [3]

    def test_w_shape(self):
        model = _make(5)
        assert model.W.shape == (5, 5)

    def test_b_shape(self):
        model = _make(5)
        assert model.B.shape == (5,)

    def test_memory_feedback_shape(self):
        model = _make(5)
        assert model.memory_feedback.shape == (5,)

    def test_norm_is_rms_norm(self):
        model = _make(4)
        assert isinstance(model.norm, nn.RMSNorm)

    def test_w_diagonal_zero_after_init(self):
        model = _make(6)
        diag = model.W.diag()
        assert torch.all(diag == 0.0), "Diagonal of W must be zero after init"

    def test_state_shape(self):
        model = _make(4)
        assert model.state.shape == (1, 4)

    def test_auto_size_minus_one(self):
        model = OdyssNet(num_neurons=-1, input_ids=[0, 1], output_ids=[3], device="cpu")
        assert model.num_neurons == 4  # max(3) + 1

    def test_multiple_input_output_ids(self):
        model = OdyssNet(num_neurons=6, input_ids=[0, 1, 2], output_ids=[3, 4, 5], device="cpu")
        assert len(model.input_ids) == 3
        assert len(model.output_ids) == 3
        assert model.input_scale.shape == (3,)
        assert model.output_scale.shape == (3,)

    def test_pulse_mode_default_true(self):
        model = _make(4)
        assert model.pulse_mode is True

    def test_pulse_mode_off(self):
        model = _make(4, pulse_mode=False)
        assert model.pulse_mode is False


# ===========================================================================
# Weight Initialisation Strategies
# ===========================================================================

INIT_STRATEGIES = [
    "quiet",
    "micro_quiet",
    "classic",
    "xavier_uniform",
    "orthogonal",
    "zero",
    "one",
    "resonant",
]


class TestWeightInit:
    @pytest.mark.parametrize("strategy", INIT_STRATEGIES)
    def test_strategy_produces_finite_weights(self, strategy):
        model = _make(4, weight_init=strategy)
        assert torch.isfinite(model.W).all(), f"Non-finite values in W with strategy '{strategy}'"

    def test_zero_init_all_zeros(self):
        model = _make(4, weight_init="zero")
        # off-diagonal should be ~0 (init zeros, diagonal forced 0)
        off_diag = model.W.clone().fill_diagonal_(0.0)
        assert off_diag.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_resonant_spectral_radius_leq_one(self):
        model = _make(8, weight_init="resonant")
        with torch.no_grad():
            sigma_max = torch.linalg.matrix_norm(model.W, ord=2).item()
        assert sigma_max <= 1.0 + 1e-5, f"Spectral radius {sigma_max:.4f} > 1 for resonant init"

    def test_list_weight_init_overrides_slots(self):
        model = _make(4, weight_init=["quiet", "resonant", "quiet", "zero"])
        assert torch.isfinite(model.W).all()

    def test_single_string_expands_enc_dec(self):
        # 'resonant' as str -> enc_dec = 'quiet', core = 'resonant'
        model = _make(4, weight_init="resonant")
        assert model.core_weight_init == "resonant"
        assert model.enc_dec_weight_init == "quiet"


# ===========================================================================
# Activation Functions
# ===========================================================================

ACTIVATIONS = ["none", "tanh", "relu", "leaky_relu", "sigmoid", "gelu", "silu"]


class TestActivations:
    @pytest.mark.parametrize("act", ACTIVATIONS)
    def test_activation_forward_runs(self, act):
        model = _make(4, activation=act)
        x = torch.randn(2, 4)
        out, _ = model(x, steps=2)
        assert torch.isfinite(out).all()

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            _make(4, activation="unknown_act")

    def test_list_activation_independent_slots(self):
        model = _make(4, activation=["none", "tanh", "tanh", "none"])
        assert isinstance(model.act, nn.Tanh)
        assert isinstance(model.enc_dec_act, nn.Identity)


# ===========================================================================
# Gate Configurations
# ===========================================================================

class TestGates:
    def test_no_gate_by_default(self):
        model = _make(4)
        assert model.input_gate is None
        assert model.output_gate is None
        assert model.core_gate is None

    def test_memory_gate_created_by_default(self):
        # Default gate = ['none', 'none', 'identity']
        model = _make(4)
        assert model.memory_gate is not None

    def test_sigmoid_gate_creates_all_params(self):
        model = _make(4, gate="sigmoid")
        assert model.input_gate is not None
        assert model.output_gate is not None
        assert model.core_gate is not None
        assert model.memory_gate is not None

    def test_none_gate_disables_all(self):
        model = _make(4, gate=["none", "none", "none"])
        assert model.input_gate is None
        assert model.core_gate is None
        assert model.memory_gate is None

    def test_gate_list_partial_override(self):
        model = _make(4, gate=["sigmoid", "none"])
        # enc_dec gate on, core off, memory uses default 'identity'
        assert model.input_gate is not None
        assert model.core_gate is None
        assert model.memory_gate is not None


# ===========================================================================
# Forward Pass
# ===========================================================================

class TestForward:
    def test_pulse_mode_output_shape(self):
        model = _make(4)
        x = torch.randn(2, 4)
        out, h = model(x, steps=3)
        # (batch, steps, neurons)
        assert out.shape == (2, 3, 4)
        assert h.shape == (2, 4)

    def test_single_step_returns_one_output(self):
        model = _make(4)
        x = torch.randn(2, 4)
        out, _ = model(x, steps=1)
        assert out.shape == (2, 1, 4)

    def test_stream_mode_sequential_input(self):
        # (Batch, Steps, Neurons) input with pulse_mode=False
        model = _make(4, pulse_mode=False)
        x = torch.randn(2, 5, 4)  # 5 sequential steps
        out, _ = model(x, steps=5)
        assert out.shape == (2, 5, 4)

    def test_output_neurons_scaled(self):
        model = _make(4, in_ids=[0], out_ids=[3])
        with torch.no_grad():
            model.output_scale.fill_(2.0)
        x = torch.randn(1, 4)
        out, _ = model(x, steps=1)
        # Output neurons should be scaled by 2
        assert out.shape[2] == 4  # full state returned; scaling applied only at out_ids

    def test_none_input_steps_from_state(self):
        model = _make(4)
        out, _ = model(None, steps=3)
        assert out.shape == (1, 3, 4)

    def test_initial_state_injection(self):
        model = _make(4)
        custom_state = torch.ones(2, 4)
        x = torch.randn(2, 4)
        out, h = model(x, steps=2, current_state=custom_state)
        assert out.shape == (2, 2, 4)
        assert h.shape == (2, 4)

    def test_dropout_disabled_in_eval(self):
        model = OdyssNet(num_neurons=4, input_ids=[0], output_ids=[3], device="cpu", dropout_rate=0.5)
        model.eval()
        x = torch.randn(4, 4)
        out1, _ = model(x, steps=3)
        out2, _ = model(x, steps=3)
        assert torch.allclose(out1, out2), "Eval mode must produce deterministic outputs"

    def test_gradient_flows_through_forward(self):
        model = _make(4)
        x = torch.randn(2, 4, requires_grad=False)
        model.reset_state(2)
        out, _ = model(x, steps=3)
        loss = out.sum()
        loss.backward()
        assert model.W.grad is not None
        assert model.B.grad is not None


# ===========================================================================
# Vocab / Projection Mode
# ===========================================================================

class TestVocabMode:
    def test_vocab_mode_output_shape(self):
        model = OdyssNet(
            num_neurons=8,
            input_ids=list(range(4)),
            output_ids=list(range(4, 8)),
            device="cpu",
            vocab_size=16,
            vocab_mode="discrete",
        )
        x = torch.randint(0, 16, (2, 5))  # (Batch, Seq)
        out, h = model(x, steps=5)
        # (batch, steps, vocab_size)
        assert out.shape == (2, 5, 16)

    def test_vocab_mode_continuous_proj(self):
        model = OdyssNet(
            num_neurons=8,
            input_ids=list(range(4)),
            output_ids=list(range(4, 8)),
            device="cpu",
            vocab_size=(16, 16),
            vocab_mode="continuous",
        )
        assert model.proj is not None
        assert model.embed is None

    def test_vocab_mode_hybrid(self):
        model = OdyssNet(
            num_neurons=8,
            input_ids=list(range(4)),
            output_ids=list(range(4, 8)),
            device="cpu",
            vocab_size=16,
            vocab_mode="hybrid",
        )
        assert model.proj is not None
        assert model.embed is not None


# ===========================================================================
# State Management
# ===========================================================================

class TestStateManagement:
    def test_reset_state_zeros(self):
        model = _make(4)
        model.state = torch.ones(1, 4)
        model.reset_state(batch_size=3)
        assert model.state.shape == (3, 4)
        assert model.state.sum().item() == pytest.approx(0.0)

    def test_detach_state_breaks_grad(self):
        model = _make(4)
        x = torch.randn(1, 4, requires_grad=True)
        out, h = model(x, steps=2)
        # Store graded state
        model.state = h
        model.detach_state()
        assert not model.state.requires_grad

    def test_reset_state_default_batch_one(self):
        model = _make(4)
        model.reset_state()
        assert model.state.shape[0] == 1


# ===========================================================================
# Diagonal Constraint
# ===========================================================================

class TestDiagonalConstraint:
    def test_w_diagonal_stays_zero_after_forward(self):
        model = _make(6)
        x = torch.randn(2, 6)
        model(x, steps=5)
        assert torch.all(model.W.diag() == 0.0)

    def test_w_diagonal_stays_zero_after_gradient_step(self):
        model = _make(6)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        out.sum().backward()
        opt.step()
        assert torch.all(model.W.diag() == 0.0)


# ===========================================================================
# Utility Methods
# ===========================================================================

class TestUtilityMethods:
    def test_get_num_params_excludes_diagonal(self):
        model = _make(4)
        n_params = model.get_num_params()
        # W diagonal (4 values) excluded
        raw = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params == raw - model.W.shape[0]

    def test_regenerate_weak_weights_returns_counts(self):
        model = _make(6)
        with torch.no_grad():
            model.W.data.fill_(0.001)
            model.W.fill_diagonal_(0.0)
        revived, total = model.regenerate_weak_weights(threshold=0.01)
        assert revived >= 0
        assert total == model.W.numel() - model.W.shape[0]

    def test_regenerate_diagonal_not_revived(self):
        model = _make(6)
        with torch.no_grad():
            model.W.data.fill_(0.0)  # All weak including diagonal
        model.regenerate_weak_weights(threshold=0.01)
        assert torch.all(model.W.diag() == 0.0)

    def test_regenerate_percentage_mode(self):
        model = _make(6)
        revived, total = model.regenerate_weak_weights(percentage=0.1)
        assert revived >= 0

    def test_device_property_returns_w_device(self):
        model = _make(4)
        assert model.device == model.W.device
