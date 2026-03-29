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
- expand: with ChaosGrad optimizer (state transfer)
- expand: gate parameters expanded correctly
- expand: multiple sequential expansions
"""

import pytest
import torch
import os

os.environ.setdefault("NO_BNB", "1")

from odyssnet import OdyssNet
from odyssnet.utils.neurogenesis import Neurogenesis
from odyssnet.training.chaos_optimizer import ChaosGrad, ChaosGradConfig


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
    groups = ChaosGrad.classify_params(model)
    cfg = ChaosGradConfig.tiny_network(lr=lr)
    for g in groups:
        for k, v in cfg.items():
            if k not in g and k != "params":
                g[k] = v
    return ChaosGrad(groups, **cfg)


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
# ChaosGrad Optimizer Expansion
# ===========================================================================

class TestChaosGradExpansion:
    def test_expand_with_chaosgrad_returns_chaosgrad(self):
        model = _model(n=4)
        opt = _chaosgrad(model)
        new_opt = Neurogenesis.expand(
            model, opt, amount=2, verbose=False,
            chaos_config=ChaosGradConfig.tiny_network(lr=1e-3),
        )
        assert isinstance(new_opt, ChaosGrad)

    def test_expand_with_chaosgrad_model_can_train(self):
        model = _model(n=4)
        opt = _chaosgrad(model)
        new_opt = Neurogenesis.expand(
            model, opt, amount=2, verbose=False,
            chaos_config=ChaosGradConfig.tiny_network(lr=1e-3),
        )
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
