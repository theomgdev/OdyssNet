"""
Unit tests for odyssnet.training.trainer.OdyssNetTrainer.

Covers:
- Initialisation (optimizer auto-selection, chaos config)
- train_batch (basic step, loss returned, params updated)
- predict (shapes, eval mode)
- evaluate (returns scalar loss)
- fit (high-level training loop)
- Gradient accumulation
- Full sequence output mode
- Masked loss
- Synaptic noise
- regenerate_synapses
- expand (neurogenesis)
- Anomaly hook callbacks
- get_diagnostics / trigger_plateau_escape
- state_dict / load_state_dict round-trip
"""

import pytest
import torch
import os

os.environ.setdefault("NO_BNB", "1")

from odyssnet import OdyssNet, OdyssNetTrainer
from odyssnet.training.chaos_optimizer import ChaosGrad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(n=5, in_ids=None, out_ids=None):
    in_ids = in_ids or [0, 1]
    out_ids = out_ids or [3, 4]
    return OdyssNet(num_neurons=n, input_ids=in_ids, output_ids=out_ids, device="cpu")


def _trainer(model=None, **kwargs):
    if model is None:
        model = _model()
    defaults = dict(device="cpu", lr=1e-3)
    defaults.update(kwargs)
    return OdyssNetTrainer(model, **defaults)


def _batch(batch_size=4, num_neurons=5):
    torch.manual_seed(42)
    x = torch.randn(batch_size, num_neurons)
    return x


def _targets(batch_size=4, n_outputs=2):
    torch.manual_seed(99)
    return torch.randn(batch_size, n_outputs)


# ===========================================================================
# Initialisation
# ===========================================================================

class TestTrainerInit:
    def test_default_optimizer_is_chaos_grad(self):
        t = _trainer()
        assert isinstance(t.optimizer, ChaosGrad)
        assert t._using_chaos_grad is True

    def test_custom_optimizer_bypasses_chaos_grad(self):
        model = _model()
        custom_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        t = OdyssNetTrainer(model, optimizer=custom_opt, device="cpu")
        assert isinstance(t.optimizer, torch.optim.AdamW)
        assert t._using_chaos_grad is False

    def test_custom_optimizer_used_directly(self):
        model = _model()
        custom_opt = torch.optim.SGD(model.parameters(), lr=0.01)
        t = OdyssNetTrainer(model, optimizer=custom_opt, device="cpu")
        assert t.optimizer is custom_opt

    def test_custom_loss_fn_used(self):
        model = _model()
        fn = torch.nn.L1Loss()
        t = OdyssNetTrainer(model, loss_fn=fn, device="cpu")
        assert t.loss_fn is fn


# ===========================================================================
# train_batch
# ===========================================================================

class TestTrainBatch:
    def test_train_batch_returns_scalar_loss(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        loss = t.train_batch(x, y, thinking_steps=3)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_loss_is_finite(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        loss = t.train_batch(x, y, thinking_steps=3)
        assert torch.isfinite(torch.tensor(loss))

    def test_parameters_change_after_step(self):
        model = _model()
        t = _trainer(model)
        w_before = model.W.data.clone()
        x = _batch()
        y = _targets()
        t.train_batch(x, y, thinking_steps=3)
        assert not torch.allclose(model.W.data, w_before)

    def test_gradient_accumulation_delays_update(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        w_before = model.W.data.clone()
        # With accumulation_steps=2, first call should NOT update
        t.train_batch(x, y, thinking_steps=3, gradient_accumulation_steps=2)
        # Parameter should NOT change after first sub-step
        assert torch.allclose(model.W.data, w_before)
        # Second call triggers the actual update
        t.train_batch(x, y, thinking_steps=3, gradient_accumulation_steps=2)
        assert not torch.allclose(model.W.data, w_before)

    def test_full_sequence_mode(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        # For full_sequence, target shape is (batch, steps, n_outputs)
        y = torch.randn(4, 3, 2)
        loss = t.train_batch(x, y, thinking_steps=3, full_sequence=True)
        assert isinstance(loss, float)

    def test_masked_loss(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        mask = torch.ones(4, 2)
        mask[0] = 0  # ignore first sample
        loss = t.train_batch(x, y, thinking_steps=3, mask=mask)
        assert isinstance(loss, float)

    def test_synaptic_noise_applies_without_error(self):
        model = _model()
        t = OdyssNetTrainer(model, device="cpu", lr=1e-3, synaptic_noise=0.01)
        x = _batch()
        y = _targets()
        loss = t.train_batch(x, y, thinking_steps=2)
        assert isinstance(loss, float)

    def test_return_state_flag(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        result = t.train_batch(x, y, thinking_steps=2, return_state=True)
        assert isinstance(result, tuple)
        loss, state = result
        assert isinstance(loss, float)
        assert state.shape == (4, model.num_neurons)

    def test_tbptt_chained_initial_state(self):
        # experiment_llm.py: return_state=True feeds the final state back as
        # initial_state for the next chunk (Truncated BPTT).
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()

        loss1, state1 = t.train_batch(x, y, thinking_steps=2, return_state=True)
        state1 = state1.detach()

        # Second chunk starts from where the first chunk ended
        loss2, state2 = t.train_batch(
            x, y, thinking_steps=2, initial_state=state1, return_state=True
        )
        assert isinstance(loss2, float)
        assert state2.shape == (4, model.num_neurons)

    def test_output_transform_applied(self):
        # convergence_mnist_reverse_record.py uses output_transform to slice warmup
        # steps out of the full sequence before computing loss.
        model = _model()
        t = _trainer(model)
        x = _batch()
        # full_sequence target has 5 steps, but we only supervise the last 3
        y = torch.randn(4, 3, 2)

        transform_called = {"flag": False}

        def transform(pred):
            transform_called["flag"] = True
            # pred shape: (batch, steps=5, n_outputs=2) → slice last 3
            return pred[:, 2:, :]

        loss = t.train_batch(x, y, thinking_steps=5, full_sequence=True,
                             output_transform=transform)
        assert isinstance(loss, float)
        assert transform_called["flag"], "output_transform was not called"

    def test_sequential_3d_input_full_sequence(self):
        # convergence_adder / detective / latch / stopwatch pattern:
        # 3D (batch, seq_len, features) input with full_sequence=True.
        model = OdyssNet(num_neurons=8, input_ids=[0], output_ids=[7], device="cpu")
        t = OdyssNetTrainer(model, device="cpu", lr=1e-3)
        x = torch.randn(4, 6, 8)      # (batch, 6 steps, 8 neurons)
        y = torch.randn(4, 6, 1)      # (batch, 6 steps, 1 output)
        loss = t.train_batch(x, y, thinking_steps=6, full_sequence=True)
        assert isinstance(loss, float)

    def test_non_pulse_2d_input_full_sequence(self):
        # convergence_sine_wave.py: pulse_mode=False, 2D input, full_sequence=True.
        model = OdyssNet(
            num_neurons=8, input_ids=[0], output_ids=[7],
            device="cpu", pulse_mode=False,
        )
        t = OdyssNetTrainer(model, device="cpu", lr=1e-3)
        x = torch.randn(4, 8)          # (batch, neurons) — single scalar per sample
        y = torch.randn(4, 10, 1)      # (batch, steps, output)
        loss = t.train_batch(x, y, thinking_steps=10, full_sequence=True)
        assert isinstance(loss, float)


# ===========================================================================
# predict
# ===========================================================================

class TestPredict:
    def test_predict_returns_tensor(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        pred = t.predict(x, thinking_steps=3)
        assert isinstance(pred, torch.Tensor)

    def test_predict_output_shape(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        pred = t.predict(x, thinking_steps=3)
        # (batch, n_outputs) = (4, 2)
        assert pred.shape == (4, 2)

    def test_predict_full_sequence_shape(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        pred = t.predict(x, thinking_steps=5, full_sequence=True)
        assert pred.shape == (4, 5, 2)

    def test_predict_is_deterministic_in_eval(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        p1 = t.predict(x, thinking_steps=3)
        p2 = t.predict(x, thinking_steps=3)
        assert torch.allclose(p1, p2)

    def test_predict_does_not_affect_gradients(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        pred = t.predict(x, thinking_steps=2)
        assert not pred.requires_grad


# ===========================================================================
# evaluate
# ===========================================================================

class TestEvaluate:
    def test_evaluate_returns_float(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        result = t.evaluate(x, y, thinking_steps=3)
        assert isinstance(result, float)

    def test_evaluate_loss_positive(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        result = t.evaluate(x, y, thinking_steps=3)
        assert result >= 0.0


# ===========================================================================
# fit
# ===========================================================================

class TestFit:
    def test_fit_returns_history_list(self):
        model = _model()
        t = _trainer(model)
        x = _batch(batch_size=8)
        y = _targets(batch_size=8)
        history = t.fit(x, y, epochs=3, batch_size=4, thinking_steps=2, verbose=False)
        assert isinstance(history, list)
        assert len(history) == 3

    def test_fit_history_all_finite(self):
        model = _model()
        t = _trainer(model)
        x = _batch(batch_size=8)
        y = _targets(batch_size=8)
        history = t.fit(x, y, epochs=5, batch_size=4, thinking_steps=2, verbose=False)
        assert all(torch.isfinite(torch.tensor(v)) for v in history)

    def test_fit_loss_trend_downward_on_simple_data(self):
        """Loss should generally decrease when training on a simple constant target."""
        torch.manual_seed(0)
        model = _model()
        t = OdyssNetTrainer(model, device="cpu", lr=1e-2)
        n = 16
        x = torch.randn(n, 5)
        y = torch.zeros(n, 2)  # constant target
        history = t.fit(x, y, epochs=20, batch_size=n, thinking_steps=5, verbose=False)
        assert history[-1] < history[0], "Loss should decrease over training"


# ===========================================================================
# regenerate_synapses
# ===========================================================================

class TestRegenerate:
    def test_regenerate_returns_tuple(self):
        model = _model()
        t = _trainer(model)
        result = t.regenerate_synapses(threshold=0.01)
        assert isinstance(result, tuple)
        revived, total = result
        assert isinstance(revived, (int, float))
        assert total > 0


# ===========================================================================
# expand (neurogenesis)
# ===========================================================================

class TestExpand:
    def test_expand_increases_num_neurons(self):
        model = _model(n=5)
        t = _trainer(model)
        t.expand(amount=2, verbose=False)
        assert model.num_neurons == 7

    def test_expand_returns_new_optimizer(self):
        model = _model(n=5)
        t = _trainer(model)
        old_opt = t.optimizer
        t.expand(amount=1, verbose=False)
        # Optimizer is rebuilt; ids may differ
        assert t.optimizer is not None

    def test_expand_preserves_existing_weights(self):
        model = _model(n=4)
        t = _trainer(model)
        old_w = model.W.data[:4, :4].clone()
        t.expand(amount=2, verbose=False)
        new_w = model.W.data[:4, :4]
        assert torch.allclose(old_w, new_w)


# ===========================================================================
# Anomaly Hook
# ===========================================================================

class TestAnomalyHook:
    def test_hook_called_on_increase(self):
        model = _model()
        events = []

        def hook(event_type, loss_val):
            events.append(event_type)

        t = OdyssNetTrainer(model, device="cpu", lr=1e-3, anomaly_hook=hook)
        x = _batch()

        # Step 1: train toward zero → low loss, sets _prev_step_loss
        y_easy = torch.zeros(4, 2)
        t.train_batch(x, y_easy, thinking_steps=2)

        # Step 2: use an extreme target to force a much higher loss
        y_hard = torch.full((4, 2), 1e4)
        t.train_batch(x, y_hard, thinking_steps=2)

        assert "increase" in events


# ===========================================================================
# Diagnostics
# ===========================================================================

class TestDiagnostics:
    def test_get_diagnostics_keys(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        t.train_batch(x, y, thinking_steps=2)
        diag = t.get_diagnostics()
        assert "step_count" in diag
        assert "last_loss" in diag
        assert "using_chaos_grad" in diag
        assert "current_lr" in diag

    def test_trigger_plateau_escape_runs_without_error(self):
        model = _model()
        t = _trainer(model)
        t.trigger_plateau_escape()  # should not raise


# ===========================================================================
# State Dict Round-Trip
# ===========================================================================

class TestStateDictRoundTrip:
    def test_state_dict_contains_step_count(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        t.train_batch(x, y, thinking_steps=2)
        sd = t.state_dict()
        assert "step_count" in sd

    def test_load_state_dict_restores_step_count(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        for _ in range(3):
            t.train_batch(x, y, thinking_steps=2)
        sd = t.state_dict()

        # Build a fresh trainer and restore
        model2 = _model()
        t2 = _trainer(model2)
        t2.load_state_dict(sd)
        assert t2._step_count == t._step_count
