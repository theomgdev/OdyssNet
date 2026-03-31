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
    "micro_quiet_8bit",   # used by mnist_record, mnist_reverse_record
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

    def test_three_element_activation_list(self):
        # Used by convergence_mnist_reverse_record.py
        model = _make(4, activation=["tanh", "tanh", "tanh"])
        assert isinstance(model.enc_dec_act, nn.Tanh)
        assert isinstance(model.act, nn.Tanh)
        assert isinstance(model.mem_act, nn.Tanh)
        x = torch.randn(2, 4)
        out, _ = model(x, steps=3)
        assert torch.isfinite(out).all()


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

    def test_non_pulse_2d_input_vco_pattern(self):
        # convergence_sine_wave.py: pulse_mode=False with a 2D (batch, neurons) input.
        # The input is held constant across all steps (continuous VCO mode).
        model = _make(4, pulse_mode=False)
        x = torch.randn(2, 4)
        out, h = model(x, steps=5)
        assert out.shape == (2, 5, 4)
        assert h.shape == (2, 4)

    def test_non_pulse_2d_all_steps_see_input(self):
        # In non-pulse 2D mode the cached input is injected every step,
        # so the output should differ from a zero-input run.
        model = _make(4, pulse_mode=False)
        x_nonzero = torch.ones(1, 4) * 5.0
        x_zero = torch.zeros(1, 4)
        model.reset_state(1)
        out_signal, _ = model(x_nonzero, steps=3)
        model.reset_state(1)
        out_zero, _ = model(x_zero, steps=3)
        assert not torch.allclose(out_signal, out_zero)

    def test_gradient_checkpointing_forward(self):
        # Used by experiment_llm.py; must produce finite outputs and allow backward.
        model = OdyssNet(
            num_neurons=4, input_ids=[0], output_ids=[3],
            device="cpu", gradient_checkpointing=True,
        )
        model.train()
        x = torch.randn(2, 4)
        out, _ = model(x, steps=3)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert model.W.grad is not None

    def test_return_sequence_false_shape(self):
        # return_sequence=False returns only the final step, saving VRAM.
        model = _make(4)
        x = torch.randn(2, 4)
        out_full, h_full = model(x, steps=5, return_sequence=True)
        model.reset_state(2)
        out_last, h_last = model(x, steps=5, return_sequence=False)
        # Full sequence: (B, T, N); last-only: (B, 1, N)
        assert out_full.shape == (2, 5, 4)
        assert out_last.shape == (2, 1, 4)
        # Both runs produce the same final hidden state
        assert torch.allclose(h_full, h_last, atol=1e-5)
        # The single output step matches the last step of the full run
        assert torch.allclose(out_last[:, 0, :], out_full[:, -1, :], atol=1e-5)

    def test_return_sequence_false_gradient_flows(self):
        # Gradients must flow through W when only the final step is collected.
        model = _make(4)
        model.train()
        x = torch.randn(2, 4)
        out, _ = model(x, steps=3, return_sequence=False)
        out.sum().backward()
        assert model.W.grad is not None


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

    def test_vocab_continuous_sequential_chunked_input(self):
        # convergence_mnist_record.py: continuous vocab with sequential 3D input
        # Input shape (batch, num_chunks, chunk_size) -> projected per chunk.
        model = OdyssNet(
            num_neurons=10,
            input_ids=list(range(3)),
            output_ids=list(range(3, 10)),
            device="cpu",
            vocab_size=[79, 10],
            vocab_mode="continuous",
        )
        x = torch.randn(4, 10, 79)   # (batch, 10 chunks, 79 pixels each)
        out, h = model(x, steps=10)
        assert out.shape == (4, 10, 10)

    def test_vocab_continuous_scalar_input_generation(self):
        # convergence_mnist_reverse_record.py: single scalar input, multi-step output
        model = OdyssNet(
            num_neurons=12,
            input_ids=[0, 1],
            output_ids=list(range(2, 8)),
            device="cpu",
            vocab_size=[1, 49],
            vocab_mode="continuous",
            activation=["tanh", "tanh", "tanh"],
            weight_init="micro_quiet_8bit",
        )
        x = torch.randn(4, 1, 1)   # (batch, 1 step, 1 feature)
        out, h = model(x, steps=21)
        assert out.shape == (4, 21, 49)

    def test_vocab_discrete_tie_embeddings(self):
        # experiment_llm.py: tie_embeddings=True shares embed <-> output_decoder weights.
        model = OdyssNet(
            num_neurons=8,
            input_ids=list(range(4)),
            output_ids=list(range(4, 8)),
            device="cpu",
            vocab_size=16,
            vocab_mode="discrete",
            tie_embeddings=True,
        )
        # Tied: embed.weight and output_decoder.weight are the same tensor
        assert model.embed.weight is model.output_decoder.weight

    def test_vocab_discrete_sequential_token_input(self):
        # experiment_llm.py: integer token indices as (batch, seq_len) input.
        model = OdyssNet(
            num_neurons=8,
            input_ids=list(range(4)),
            output_ids=list(range(4, 8)),
            device="cpu",
            vocab_size=32,
            vocab_mode="discrete",
        )
        x = torch.randint(0, 32, (2, 6))   # (batch, seq_len) token ids
        out, h = model(x, steps=6)
        assert out.shape == (2, 6, 32)


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


# ===========================================================================
# Hebbian Learning
# ===========================================================================

class TestHebbian:
    def test_disabled_by_default(self):
        model = _make(4)
        assert model.use_hebbian is False
        assert model.hebb_factor is None
        assert model.hebb_decay is None
        assert not hasattr(model, 'hebb_state_W')

    def test_parameters_created_when_enabled(self):
        model = _make(4, use_hebbian=True)
        assert model.use_hebbian is True
        assert isinstance(model.hebb_factor, nn.Parameter)
        assert isinstance(model.hebb_decay, nn.Parameter)

    def test_buffer_shapes(self):
        model = _make(6, use_hebbian=True)
        assert model.hebb_state_W.shape == (6, 6)
        assert model.hebb_state_mem.shape == (6,)

    def test_buffers_zero_at_init(self):
        model = _make(4, use_hebbian=True)
        assert model.hebb_state_W.abs().sum().item() == 0.0
        assert model.hebb_state_mem.abs().sum().item() == 0.0

    def test_initial_factor_bounds(self):
        # sigmoid(-3.0) ≈ 0.047, sigmoid(2.2) ≈ 0.90
        model = _make(4, use_hebbian=True)
        lr   = torch.sigmoid(model.hebb_factor).item()
        ret  = torch.sigmoid(model.hebb_decay).item()
        assert 0.0 < lr < 0.15
        assert 0.85 < ret < 1.0

    def test_forward_runs_with_hebbian(self):
        model = _make(4, use_hebbian=True)
        x = torch.randn(2, 4)
        out, h = model(x, steps=3)
        assert out.shape == (2, 3, 4)
        assert torch.isfinite(out).all()

    def test_state_updates_after_forward(self):
        model = _make(4, use_hebbian=True)
        x = torch.randn(2, 4)
        model(x, steps=3)
        # Hebbian state must be non-zero after a non-trivial forward pass.
        assert model.hebb_state_W.abs().sum().item() > 0.0
        assert model.hebb_state_mem.abs().sum().item() > 0.0

    def test_diagonal_zero_in_hebb_state_w(self):
        # hebb_state_W must mirror the W diagonal constraint.
        model = _make(6, use_hebbian=True)
        x = torch.randn(2, 6)
        model(x, steps=10)
        assert model.hebb_state_W.diagonal().abs().max().item() == 0.0

    def test_gradient_flows_to_hebb_factor(self):
        model = _make(4, use_hebbian=True)
        model.train()
        x = torch.randn(2, 4)
        out, _ = model(x, steps=3)
        out.sum().backward()
        assert model.hebb_factor.grad is not None
        assert torch.isfinite(model.hebb_factor.grad).all()

    def test_gradient_flows_to_hebb_decay(self):
        # hebb_decay only participates in the graph from step ≥ 1, so use steps ≥ 2.
        model = _make(4, use_hebbian=True)
        model.train()
        x = torch.randn(2, 4)
        out, _ = model(x, steps=3)
        out.sum().backward()
        assert model.hebb_decay.grad is not None
        assert torch.isfinite(model.hebb_decay.grad).all()

    def test_gradient_flows_through_w_as_well(self):
        model = _make(4, use_hebbian=True)
        model.train()
        x = torch.randn(2, 4)
        out, _ = model(x, steps=3)
        out.sum().backward()
        assert model.W.grad is not None

    def test_reset_clears_hebb_state(self):
        model = _make(4, use_hebbian=True)
        x = torch.randn(2, 4)
        model(x, steps=3)
        assert model.hebb_state_W.abs().sum().item() > 0.0
        model.reset_state()
        assert model.hebb_state_W.abs().sum().item() == 0.0
        assert model.hebb_state_mem.abs().sum().item() == 0.0

    def test_reset_also_resets_hidden_state(self):
        model = _make(4, use_hebbian=True)
        x = torch.randn(3, 4)
        model(x, steps=2)
        model.reset_state(batch_size=1)
        assert model.state.shape == (1, 4)
        assert model.state.abs().sum().item() == 0.0

    def test_gradient_checkpointing_compatible(self):
        model = OdyssNet(
            num_neurons=4, input_ids=[0], output_ids=[3],
            device="cpu", gradient_checkpointing=True, use_hebbian=True,
        )
        model.train()
        x = torch.randn(2, 4)
        out, _ = model(x, steps=3)
        assert torch.isfinite(out).all()
        out.sum().backward()
        assert model.hebb_factor.grad is not None

    def test_hebb_factor_in_named_parameters(self):
        model = _make(4, use_hebbian=True)
        param_names = {n for n, _ in model.named_parameters()}
        assert 'hebb_factor' in param_names
        assert 'hebb_decay' in param_names
