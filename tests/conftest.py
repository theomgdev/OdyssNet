"""
Shared fixtures for the OdyssNet test suite.
"""

import pytest
import torch
import os

# Suppress bitsandbytes during tests
os.environ.setdefault("NO_BNB", "1")

from odyssnet import OdyssNet, OdyssNetTrainer
from odyssnet.training.chaos_optimizer import ChaosGrad, ChaosGradConfig
from odyssnet.training.chaos_scheduler import TemporalScheduler, TemporalSchedulerConfig


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
    """Trainer with standard AdamW (no bitsandbytes)."""
    return OdyssNetTrainer(tiny_model, device="cpu", lr=1e-3)


@pytest.fixture
def chaos_trainer(tiny_model):
    """Trainer configured with ChaosGrad optimizer."""
    return OdyssNetTrainer(
        tiny_model,
        device="cpu",
        lr=1e-3,
        chaos_config=ChaosGradConfig.tiny_network(lr=1e-3),
    )


# ---------------------------------------------------------------------------
# Optimizer / scheduler fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_optimizer(tiny_model):
    """Plain AdamW for use in optimizer/scheduler tests."""
    return torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)


@pytest.fixture
def chaos_optimizer(tiny_model):
    """ChaosGrad instance with default config."""
    groups = ChaosGrad.classify_params(tiny_model)
    cfg = ChaosGradConfig.default(lr=1e-3)
    for g in groups:
        for k, v in cfg.items():
            if k not in g and k != "params":
                g[k] = v
    return ChaosGrad(groups, **cfg)


@pytest.fixture
def temporal_scheduler(dummy_optimizer):
    """TemporalScheduler with short warmup for fast tests."""
    return TemporalScheduler(
        dummy_optimizer,
        warmup_steps=5,
        max_steps=50,
        min_lr_ratio=0.01,
        patience=0,
        verbose=False,
    )


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
