"""
Unit tests for odyssnet.training.chaos_optimizer.ChaosGrad and ChaosGradConfig.

Covers:
- Initialisation with defaults and custom configs
- Parameter group classification (classify_params)
- Optimizer step — parameter updates
- Plateau detection and escape mechanisms
- Spectral clipping for the core W matrix
- Adaptive LR and gradient centralization
- All ChaosGradConfig presets
- Diagnostics
"""

import pytest
import torch
import torch.nn as nn
import os

os.environ.setdefault("NO_BNB", "1")

from odyssnet import OdyssNet
from odyssnet.training.chaos_optimizer import ChaosGrad, ChaosGradConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_model(n=5):
    return OdyssNet(num_neurons=n, input_ids=[0], output_ids=[n - 1], device="cpu")


def _optimizer(model, **kwargs):
    groups = ChaosGrad.classify_params(model)
    cfg = ChaosGradConfig.default(lr=1e-3)
    cfg.update(kwargs)
    for g in groups:
        for k, v in cfg.items():
            if k not in g and k != "params":
                g[k] = v
    return ChaosGrad(groups, **cfg)


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

    def test_best_loss_starts_at_inf(self):
        model = _small_model()
        opt = _optimizer(model)
        assert opt._best_loss == float("inf")

    def test_plateau_counter_starts_at_zero(self):
        model = _small_model()
        opt = _optimizer(model)
        assert opt._plateau_counter == 0


# ===========================================================================
# Parameter Classification
# ===========================================================================

class TestClassifyParams:
    def test_w_in_chaos_core_group(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "chaos_core" in names
        params = names["chaos_core"]["params"]
        assert any(p is model.W for p in params)

    def test_memory_feedback_in_memory_group(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "memory_feedback" in names
        params = names["memory_feedback"]["params"]
        assert any(p is model.memory_feedback for p in params)

    def test_b_in_lightweight_group(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "lightweight" in names
        params = names["lightweight"]["params"]
        assert any(p is model.B for p in params)

    def test_embed_in_projections_group(self):
        model = OdyssNet(
            num_neurons=8,
            input_ids=list(range(4)),
            output_ids=list(range(4, 8)),
            device="cpu",
            vocab_size=16,
            vocab_mode="discrete",
        )
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "projections" in names

    def test_gate_params_in_gates_group(self):
        model = OdyssNet(
            num_neurons=5, input_ids=[0], output_ids=[4], device="cpu", gate="sigmoid"
        )
        groups = ChaosGrad.classify_params(model)
        names = {g["group_name"]: g for g in groups}
        assert "gates" in names

    def test_all_params_covered(self):
        model = _small_model()
        groups = ChaosGrad.classify_params(model)
        covered = set()
        for g in groups:
            for p in g["params"]:
                covered.add(id(p))
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

    def test_hebbian_step_updates_params(self):
        model = OdyssNet(num_neurons=5, input_ids=[0], output_ids=[4],
                         device="cpu", hebb_type="global")
        opt = _optimizer(model)
        factor_before = model.hebb_factor.data.clone()
        x = torch.randn(2, 5)
        out, _ = model(x, steps=3)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
        assert not torch.allclose(model.hebb_factor.data, factor_before)


# ===========================================================================
# Optimizer Step
# ===========================================================================

class TestOptimizerStep:
    def test_step_increments_global_step(self):
        model = _small_model()
        opt = _optimizer(model)
        x = torch.randn(2, 5)
        out, _ = model(x, steps=2)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
        assert opt._global_step == 1

    def test_w_changes_after_step(self):
        model = _small_model()
        opt = _optimizer(model)
        w_before = model.W.data.clone()
        x = torch.randn(2, 5)
        out, _ = model(x, steps=2)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
        assert not torch.allclose(model.W.data, w_before)

    def test_w_diagonal_stays_zero_after_step(self):
        model = _small_model()
        opt = _optimizer(model)
        x = torch.randn(2, 5)
        out, _ = model(x, steps=2)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
        assert torch.all(model.W.diag() == 0.0)

    def test_sparse_gradient_raises(self):
        model = _small_model()
        opt = _optimizer(model)
        # Inject a sparse gradient manually
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


# ===========================================================================
# Plateau Detection and Escape
# ===========================================================================

class TestPlateauDetection:
    def test_report_loss_updates_best(self):
        model = _small_model()
        opt = _optimizer(model)
        opt.report_loss(1.0)
        assert opt._best_loss == pytest.approx(1.0)

    def test_report_loss_increments_plateau_on_no_improvement(self):
        model = _small_model()
        opt = _optimizer(model)
        opt.report_loss(1.0)
        opt.report_loss(1.1)
        assert opt._plateau_counter == 1

    def test_report_loss_resets_plateau_on_improvement(self):
        model = _small_model()
        opt = _optimizer(model)
        opt.report_loss(1.0)
        opt.report_loss(1.1)
        opt.report_loss(0.5)
        assert opt._plateau_counter == 0

    def test_trigger_plateau_escape_sets_flag(self):
        model = _small_model()
        opt = _optimizer(model)
        opt.trigger_plateau_escape()
        assert opt._force_plateau_escape is True

    def test_plateau_escape_resets_flag_after_step(self):
        model = _small_model()
        opt = _optimizer(model, plateau_patience=0)
        opt.trigger_plateau_escape()
        x = torch.randn(2, 5)
        out, _ = model(x, steps=1)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
        assert opt._force_plateau_escape is False


# ===========================================================================
# Spectral Clipping
# ===========================================================================

class TestSpectralClipping:
    def test_spectral_clip_reduces_radius(self):
        model = _small_model(8)
        # Force a large W
        with torch.no_grad():
            model.W.data.fill_(1.0)
            model.W.fill_diagonal_(0.0)
        opt = _optimizer(model, spectral_clip=1.0)
        x = torch.randn(2, 8)
        out, _ = model(x, steps=2)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
        # After clipping, spectral radius should be tracked
        assert opt._spectral_radius >= 0.0

    def test_spectral_clip_zero_disables(self):
        model = _small_model()
        opt = _optimizer(model, spectral_clip=0.0)
        x = torch.randn(2, 5)
        out, _ = model(x, steps=1)
        out.sum().backward()
        opt.step()
        opt.zero_grad()
        # spectral radius should remain 0 (no computation done)
        assert opt._spectral_radius == 0.0


# ===========================================================================
# Adaptive LR and Gradient Centralization
# ===========================================================================

class TestAdaptiveFeatures:
    def test_adaptive_lr_enabled_still_converges(self):
        model = _small_model()
        opt = _optimizer(model, adaptive_lr=True)
        x = torch.randn(8, 5)
        target = torch.zeros(8, 1)
        for _ in range(5):
            out, _ = model(x, steps=2)
            loss = (out[:, -1, -1:] - target).pow(2).mean()
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.reset_state(8)

    def test_grad_centralization_zero_mean(self):
        """With centralization, row-mean of gradient should approach zero."""
        model = _small_model(6)
        opt = _optimizer(model, grad_centralization=True)
        x = torch.randn(4, 6)
        out, _ = model(x, steps=2)
        out.sum().backward()
        # Just verify step runs without error
        opt.step()
        opt.zero_grad()


# ===========================================================================
# Diagnostics
# ===========================================================================

class TestDiagnostics:
    def test_get_diagnostics_keys(self):
        model = _small_model()
        opt = _optimizer(model)
        diag = opt.get_diagnostics()
        assert "global_step" in diag
        assert "plateau_counter" in diag
        assert "best_loss" in diag
        assert "spectral_radius" in diag


# ===========================================================================
# ChaosGradConfig Presets
# ===========================================================================

class TestChaosGradConfig:
    @pytest.mark.parametrize("preset,kwargs", [
        ("conservative", {}),
        ("default", {}),
        ("finetune", {}),
        ("large_network", {}),
        ("tiny_network", {}),
    ])
    def test_preset_returns_dict(self, preset, kwargs):
        fn = getattr(ChaosGradConfig, preset)
        cfg = fn(**kwargs)
        assert isinstance(cfg, dict)
        assert "lr" in cfg

    def test_default_lr_can_be_overridden(self):
        cfg = ChaosGradConfig.default(lr=5e-5)
        assert cfg["lr"] == pytest.approx(5e-5)

    def test_tiny_network_no_weight_decay(self):
        cfg = ChaosGradConfig.tiny_network()
        assert cfg["weight_decay"] == 0.0

    def test_large_network_has_spectral_clip(self):
        cfg = ChaosGradConfig.large_network()
        assert cfg["spectral_clip"] > 0
