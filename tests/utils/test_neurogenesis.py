"""
Unit tests for odyssnet.utils.neurogenesis.Neurogenesis.

Covers:
- expand: neuron count increases correctly
- expand: existing weights preserved in the top-left submatrix
- expand: diagonal of W stays zero after expansion
- expand: bias and memory_feedback extended
- expand: network state tensor expanded
- expand: input/output IDs unchanged
- expand: a new valid optimizer is returned
- expand: optimizer state transfer (state transfer)
- expand: gate parameters expanded correctly
- expand: multiple sequential expansions
"""

import pytest
import torch
import os


from odyssnet import OdyssNet
from odyssnet.utils.neurogenesis import Neurogenesis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(n=5, in_ids=None, out_ids=None, **kwargs):
    in_ids = in_ids or [0]
    out_ids = out_ids or [n - 1]
    return OdyssNet(num_neurons=n, input_ids=in_ids, output_ids=out_ids, device="cpu", **kwargs)


def _adamw(model, lr=1e-3):
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _chaosgrad(model, lr=1e-3):
    return torch.optim.AdamW(model.parameters(), lr=lr)


# ===========================================================================
# Basic Expansion
# ===========================================================================

class TestExpandBasic:
    def test_num_neurons_increases(self):
        model = _model(n=4)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=3, verbose=False)
        assert model.num_neurons == 7

    def test_expand_by_one(self):
        model = _model(n=4)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=1, verbose=False)
        assert model.num_neurons == 5

    def test_w_shape_updated(self):
        model = _model(n=4)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.W.shape == (6, 6)

    def test_b_shape_updated(self):
        model = _model(n=4)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.B.shape == (6,)

    def test_memory_feedback_shape_updated(self):
        model = _model(n=4)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.memory_feedback.shape == (6,)

    def test_norm_weight_shape_updated(self):
        model = _model(n=4)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.norm.weight.shape == (6,)


# ===========================================================================
# Weight Preservation
# ===========================================================================

class TestWeightPreservation:
    def test_existing_w_block_preserved(self):
        model = _model(n=4)
        with torch.no_grad():
            model.W.data.fill_(0.5)
            model.W.fill_diagonal_(0.0)
        old_block = model.W.data.clone()
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        new_block = model.W.data[:4, :4]
        assert torch.allclose(new_block, old_block)

    def test_existing_b_preserved(self):
        model = _model(n=4)
        with torch.no_grad():
            model.B.data.fill_(1.23)
        old_b = model.B.data.clone()
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert torch.allclose(model.B.data[:4], old_b)

    def test_existing_memory_feedback_preserved(self):
        model = _model(n=4)
        with torch.no_grad():
            model.memory_feedback.data.fill_(0.99)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert torch.allclose(model.memory_feedback.data[:4],
                               torch.full((4,), 0.99))


# ===========================================================================
# Structural Invariants
# ===========================================================================

class TestStructuralInvariants:
    def test_w_diagonal_zero_after_expand(self):
        model = _model(n=4)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=3, verbose=False)
        assert torch.all(model.W.diag() == 0.0)

    def test_input_ids_unchanged(self):
        model = _model(n=5, in_ids=[0, 1])
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.input_ids == [0, 1]

    def test_output_ids_unchanged(self):
        model = _model(n=5, out_ids=[3, 4])
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.output_ids == [3, 4]

    def test_input_pos_buffer_correct(self):
        model = _model(n=5, in_ids=[0, 1])
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        expected = torch.tensor([0, 1], dtype=torch.long)
        assert torch.equal(model.input_pos, expected)

    def test_output_pos_buffer_correct(self):
        model = _model(n=5, out_ids=[3, 4])
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        expected = torch.tensor([3, 4], dtype=torch.long)
        assert torch.equal(model.output_pos, expected)


# ===========================================================================
# State Expansion
# ===========================================================================

class TestStateExpansion:
    def test_state_shape_updated(self):
        model = _model(n=4)
        model.reset_state(batch_size=2)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=3, verbose=False)
        assert model.state.shape == (2, 7)

    def test_existing_state_values_preserved(self):
        model = _model(n=4)
        model.reset_state(batch_size=1)
        with torch.no_grad():
            model.state[:, :4] = 0.42
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert torch.allclose(model.state[:, :4], torch.full((1, 4), 0.42))


# ===========================================================================
# Optimizer Return
# ===========================================================================

class TestOptimizerReturn:
    def test_returns_optimizer(self):
        model = _model(n=4)
        opt = _adamw(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert new_opt is not None

    def test_new_optimizer_can_step(self):
        model = _model(n=4)
        opt = _adamw(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=2)
        out.sum().backward()
        new_opt.step()
        new_opt.zero_grad()


# ===========================================================================
# Optimizer Expansion
# ===========================================================================

class TestOptimizerExpansion:
    def test_expand_with_chaosgrad_returns_chaosgrad(self):
        model = _model(n=4)
        opt = _chaosgrad(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert isinstance(new_opt, torch.optim.AdamW)

    def test_expand_with_chaosgrad_model_can_train(self):
        model = _model(n=4)
        opt = _chaosgrad(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=2)
        out.sum().backward()
        new_opt.step()
        new_opt.zero_grad()


# ===========================================================================
# Gate Parameters
# ===========================================================================

class TestGateExpansion:
    def test_core_gate_expanded(self):
        model = OdyssNet(
            num_neurons=4, input_ids=[0], output_ids=[3],
            device="cpu", gate=["none", "sigmoid", "none"],
        )
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.core_gate is not None
        assert model.core_gate.shape == (6,)

    def test_memory_gate_expanded(self):
        model = OdyssNet(
            num_neurons=4, input_ids=[0], output_ids=[3],
            device="cpu", gate=["none", "none", "sigmoid"],
        )
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.memory_gate is not None
        assert model.memory_gate.shape == (6,)

    def test_gate_old_values_preserved(self):
        model = OdyssNet(
            num_neurons=4, input_ids=[0], output_ids=[3],
            device="cpu", gate=["none", "sigmoid", "none"],
        )
        with torch.no_grad():
            model.core_gate.data.fill_(1.0)
        old_vals = model.core_gate.data.clone()
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert torch.allclose(model.core_gate.data[:4], old_vals)


# ===========================================================================
# Sequential Expansions
# ===========================================================================

class TestSequentialExpansions:
    def test_multiple_expansions_cumulative(self):
        model = _model(n=3)
        opt = _adamw(model)
        opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        opt = Neurogenesis.expand(model, opt, amount=3, verbose=False)
        assert model.num_neurons == 8

    def test_model_trains_after_two_expansions(self):
        model = _model(n=3)
        opt = _adamw(model)
        opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        opt = Neurogenesis.expand(model, opt, amount=1, verbose=False)
        x = torch.randn(2, model.num_neurons)
        out, _ = model(x, steps=3)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
        assert torch.isfinite(model.W.data).all()


# ===========================================================================
# Hebbian Plasticity + Neurogenesis (global / legacy)
# ===========================================================================

class TestHebbianExpansion:
    def test_hebb_buffers_resized(self):
        model = _model(n=4, hebb_type="global")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=3, verbose=False)
        assert model.hebb_state_W.shape == (7, 7)
        assert model.hebb_state_mem.shape == (7,)

    def test_hebb_buffer_old_region_preserved(self):
        model = _model(n=4, hebb_type="global")
        # Populate buffers with a known value before expansion.
        with torch.no_grad():
            model.hebb_state_W.fill_(0.5)
            model.hebb_state_W.fill_diagonal_(0.0)
            model.hebb_state_mem.fill_(0.3)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        # Old (4×4) off-diagonal subregion must equal 0.5 after expansion.
        block = model.hebb_state_W[:4, :4]
        off_diag_mask = ~torch.eye(4, dtype=torch.bool)
        assert torch.allclose(block[off_diag_mask], torch.full((12,), 0.5)), \
            "Old hebb_state_W off-diagonal region lost after expand"
        assert torch.allclose(model.hebb_state_mem[:4], torch.full((4,), 0.3))

    def test_hebb_buffer_new_region_zero(self):
        model = _model(n=4, hebb_type="global")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=3, verbose=False)
        # Newly added rows/cols must be zero.
        assert model.hebb_state_W[4:, :].abs().sum().item() == 0.0
        assert model.hebb_state_W[:, 4:].abs().sum().item() == 0.0
        assert model.hebb_state_mem[4:].abs().sum().item() == 0.0

    def test_forward_runs_after_expand(self):
        model = _model(n=4, hebb_type="global")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        assert out.shape == (2, 3, 6)
        assert torch.isfinite(out).all()

    def test_gradient_flows_after_expand(self):
        model = _model(n=4, hebb_type="global")
        model.train()
        opt = _adamw(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        out.sum().backward()
        assert model.hebb_factor.grad is not None
        assert torch.isfinite(model.hebb_factor.grad).all()
        new_opt.step()
        new_opt.zero_grad()

    def test_hebb_params_still_in_named_parameters(self):
        model = _model(n=4, hebb_type="global")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        names = {n for n, _ in model.named_parameters()}
        assert 'hebb_factor' in names
        assert 'hebb_decay' in names

    def test_chaosgrad_expand_with_hebbian(self):
        model = _model(n=4, hebb_type="global")
        opt = _chaosgrad(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert isinstance(new_opt, torch.optim.AdamW)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        out.sum().backward()
        new_opt.step()
        new_opt.zero_grad()

    def test_sequential_expand_with_hebbian(self):
        model = _model(n=3, hebb_type="global")
        opt = _adamw(model)
        opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.num_neurons == 7
        assert model.hebb_state_W.shape == (7, 7)
        assert model.hebb_state_mem.shape == (7,)

    def test_global_scalar_shape_unchanged(self):
        # "global" hebb_factor / hebb_decay are 0-dim scalars; expand must not resize them.
        model = _model(n=4, hebb_type="global")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=3, verbose=False)
        assert model.hebb_factor.shape == torch.Size([])
        assert model.hebb_decay.shape  == torch.Size([])


# ===========================================================================
# Neuron-Level Hebbian Expansion
# ===========================================================================

class TestNeuronHebbExpansion:
    def test_factor_shape_updated(self):
        model = _model(n=4, hebb_type="neuron")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=3, verbose=False)
        assert model.hebb_factor.shape == (7,)
        assert model.hebb_decay.shape  == (7,)

    def test_old_values_preserved(self):
        model = _model(n=4, hebb_type="neuron")
        with torch.no_grad():
            model.hebb_factor.data.fill_(1.5)
            model.hebb_decay.data.fill_(0.7)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert torch.allclose(model.hebb_factor.data[:4], torch.full((4,), 1.5))
        assert torch.allclose(model.hebb_decay.data[:4],  torch.full((4,), 0.7))

    def test_new_values_initialized(self):
        # New entries must be initialized to default logit values (-3.0 / 2.2).
        model = _model(n=4, hebb_type="neuron")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert torch.allclose(model.hebb_factor.data[4:], torch.full((2,), -3.0))
        assert torch.allclose(model.hebb_decay.data[4:],  torch.full((2,),  2.2))

    def test_forward_runs_after_expand(self):
        model = _model(n=4, hebb_type="neuron")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        assert out.shape == (2, 3, 6)
        assert torch.isfinite(out).all()

    def test_gradient_flows_after_expand(self):
        model = _model(n=4, hebb_type="neuron")
        model.train()
        opt = _adamw(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        out.sum().backward()
        assert model.hebb_factor.grad is not None
        assert model.hebb_factor.grad.shape == (6,)
        assert torch.isfinite(model.hebb_factor.grad).all()
        new_opt.step()
        new_opt.zero_grad()


# ===========================================================================
# Synapse-Level Hebbian Expansion
# ===========================================================================

class TestSynapseHebbExpansion:
    def test_factor_shape_updated(self):
        model = _model(n=4, hebb_type="synapse")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=3, verbose=False)
        assert model.hebb_factor.shape == (7, 7)
        assert model.hebb_decay.shape  == (7, 7)

    def test_old_quadrant_preserved_factor(self):
        # The original N×N submatrix must be intact after expansion to (N+K)×(N+K).
        n, k = 4, 3
        model = _model(n=n, hebb_type="synapse")
        with torch.no_grad():
            model.hebb_factor.data.fill_(0.8)
            model.hebb_decay.data.fill_(1.1)
        old_factor = model.hebb_factor.data.clone()
        old_decay  = model.hebb_decay.data.clone()
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=k, verbose=False)
        assert torch.allclose(model.hebb_factor.data[:n, :n], old_factor), \
            "Old N×N factor quadrant lost after synapse expand"
        assert torch.allclose(model.hebb_decay.data[:n, :n], old_decay), \
            "Old N×N decay quadrant lost after synapse expand"

    def test_new_region_initialized_factor(self):
        n, k = 4, 2
        model = _model(n=n, hebb_type="synapse")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=k, verbose=False)
        new_n = n + k
        # New rows (bottom) and new cols (right) must carry default logit values.
        assert torch.allclose(model.hebb_factor.data[n:, :], torch.full((k, new_n), -3.0))
        assert torch.allclose(model.hebb_factor.data[:, n:], torch.full((new_n, k), -3.0))
        assert torch.allclose(model.hebb_decay.data[n:, :],  torch.full((k, new_n),  2.2))
        assert torch.allclose(model.hebb_decay.data[:, n:],  torch.full((new_n, k),  2.2))

    def test_old_buffer_quadrant_preserved(self):
        n, k = 4, 2
        model = _model(n=n, hebb_type="synapse")
        with torch.no_grad():
            model.hebb_state_W.fill_(0.5)
            model.hebb_state_W.fill_diagonal_(0.0)
            model.hebb_state_mem.fill_(0.3)
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=k, verbose=False)
        block = model.hebb_state_W[:n, :n]
        off_diag_mask = ~torch.eye(n, dtype=torch.bool)
        assert torch.allclose(block[off_diag_mask], torch.full((n * n - n,), 0.5)), \
            "Old hebb_state_W off-diagonal quadrant lost after synapse expand"
        assert torch.allclose(model.hebb_state_mem[:n], torch.full((n,), 0.3))

    def test_new_buffer_region_zero(self):
        n, k = 4, 3
        model = _model(n=n, hebb_type="synapse")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=k, verbose=False)
        assert model.hebb_state_W[n:, :].abs().sum().item() == 0.0
        assert model.hebb_state_W[:, n:].abs().sum().item() == 0.0
        assert model.hebb_state_mem[n:].abs().sum().item() == 0.0

    def test_forward_runs_after_expand(self):
        model = _model(n=4, hebb_type="synapse")
        opt = _adamw(model)
        Neurogenesis.expand(model, opt, amount=2, verbose=False)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        assert out.shape == (2, 3, 6)
        assert torch.isfinite(out).all()

    def test_gradient_flows_after_expand(self):
        model = _model(n=4, hebb_type="synapse")
        model.train()
        opt = _adamw(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        out.sum().backward()
        assert model.hebb_factor.grad is not None
        assert model.hebb_factor.grad.shape == (6, 6)
        assert torch.isfinite(model.hebb_factor.grad).all()
        new_opt.step()
        new_opt.zero_grad()

    def test_chaosgrad_expand_with_synapse(self):
        model = _model(n=4, hebb_type="synapse")
        opt = _chaosgrad(model)
        new_opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert isinstance(new_opt, torch.optim.AdamW)
        x = torch.randn(2, 6)
        out, _ = model(x, steps=3)
        out.sum().backward()
        new_opt.step()
        new_opt.zero_grad()

    def test_sequential_expand_synapse(self):
        model = _model(n=3, hebb_type="synapse")
        opt = _adamw(model)
        opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        opt = Neurogenesis.expand(model, opt, amount=2, verbose=False)
        assert model.num_neurons == 7
        assert model.hebb_factor.shape == (7, 7)
        assert model.hebb_state_W.shape == (7, 7)
