"""
Shared fixtures for the OdyssNet test suite.
"""

import pytest
import torch
import os


from odyssnet import OdyssNet, OdyssNetTrainer

# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    """Minimal 3-neuron model (1 input, 1 output, 1 hidden)."""
    return OdyssNet(
        num_neurons=3,
        input_ids=[0],
        output_ids=[2],
        device="cpu",
    )


@pytest.fixture
def small_model():
    """Small 5-neuron model (2 inputs, 2 outputs)."""
    return OdyssNet(
        num_neurons=5,
        input_ids=[0, 1],
        output_ids=[3, 4],
        device="cpu",
    )


@pytest.fixture
def xor_model():
    """3-neuron model suited for XOR-style tests."""
    return OdyssNet(
        num_neurons=3,
        input_ids=[0, 1],
        output_ids=[2],
        device="cpu",
        weight_init="resonant",
    )


# ---------------------------------------------------------------------------
# Trainer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_trainer(tiny_model):
    """Trainer with default AdamW."""
    return OdyssNetTrainer(tiny_model, device="cpu", lr=1e-4)


# ---------------------------------------------------------------------------
# Optimizer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_optimizer(tiny_model):
    """Plain AdamW for optimizer tests."""
    return torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_input():
    """Simple (4, 3) float tensor representing a batch."""
    torch.manual_seed(0)
    return torch.randn(4, 3)


@pytest.fixture
def batch_target():
    """Simple (4, 1) float tensor representing regression targets."""
    torch.manual_seed(1)
    return torch.randn(4, 1)
