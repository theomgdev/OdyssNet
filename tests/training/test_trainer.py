"""
Unit tests for odyssnet.training.trainer.OdyssNetTrainer.

Covers:
- Initialisation (optimizer auto-selection, optimizer selection)
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
- get_diagnostics
- state_dict / load_state_dict round-trip
"""

import pytest
import torch
import os


from odyssnet import OdyssNet, OdyssNetTrainer


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
    defaults = dict(device="cpu", lr=1e-4)
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
    def test_default_optimizer_is_adamw(self):
        t = _trainer()
        assert isinstance(t.optimizer, torch.optim.AdamW)

    def test_custom_optimizer_overrides_default(self):
        model = _model()
        custom_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        t = OdyssNetTrainer(model, optimizer=custom_opt, device="cpu")
        assert isinstance(t.optimizer, torch.optim.AdamW)

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
        t = OdyssNetTrainer(model, device="cpu", lr=1e-4, synaptic_noise=0.01)
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
        t = OdyssNetTrainer(model, device="cpu", lr=1e-4)
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
        t = OdyssNetTrainer(model, device="cpu", lr=1e-4)
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

        t = OdyssNetTrainer(model, device="cpu", lr=1e-4, anomaly_hook=hook)
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
        assert "current_lr" in diag
        assert "gradient_persistence" in diag

    def test_get_diagnostics_debug_mode(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        t.train_batch(x, y, thinking_steps=2)
        diag = t.get_diagnostics(debug=True)

        # Check debug fields are present
        assert "persistent_grads_active" in diag
        assert "loss_tracking" in diag
        assert "scaler_state" in diag

    def test_get_diagnostics_default_no_debug_fields(self):
        model = _model()
        t = _trainer(model)
        x = _batch()
        y = _targets()
        t.train_batch(x, y, thinking_steps=2)
        diag = t.get_diagnostics(debug=False)

        # These should NOT be in default output
        assert "persistent_grads_active" not in diag
        assert "anomaly_tracking" not in diag
        assert "scaler_state" not in diag
        assert "gradient_stats" not in diag

    def test_get_diagnostics_with_gradient_persistence(self):
        model = _model()
        t = OdyssNetTrainer(model, device="cpu", lr=1e-4, gradient_persistence=0.5)
        x = _batch()
        y = _targets()
        t.train_batch(x, y, thinking_steps=2)
        diag = t.get_diagnostics(debug=True)

        assert diag['gradient_persistence'] == 0.5
        assert "persistent_grads_active" in diag

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


# ===========================================================================
# Private Helper Methods
# ===========================================================================

class TestTrainerHelpers:
    """Exercises the private helpers that centralise previously duplicated logic."""

    # --- _get_autocast_ctx ---------------------------------------------------

    def test_autocast_ctx_is_usable_context_manager(self):
        """_get_autocast_ctx must return a valid context manager on CPU."""
        t = _trainer()
        ctx = t._get_autocast_ctx()
        # Must support __enter__ / __exit__ without raising.
        with ctx:
            x = torch.randn(2, 5)
            _ = x @ x.T

    def test_autocast_ctx_is_consistent_on_repeated_calls(self):
        """Each call produces a fresh context manager of the same type."""
        t = _trainer()
        ctx1 = t._get_autocast_ctx()
        ctx2 = t._get_autocast_ctx()
        assert type(ctx1) is type(ctx2)

    # --- _extract_outputs: continuous mode -----------------------------------

    def test_extract_outputs_continuous_single_step(self):
        """Continuous mode, full_sequence=False: slice output_ids from final_state."""
        model = OdyssNet(num_neurons=5, input_ids=[0, 1], output_ids=[3, 4], device="cpu")
        t = _trainer(model)
        batch, n = 4, 5
        final_state = torch.randn(batch, n)
        all_states  = torch.randn(batch, 3, n)  # not used in this branch
        out = t._extract_outputs(all_states, final_state, full_sequence=False)
        assert out.shape == (batch, 2)
        assert torch.equal(out, final_state[:, [3, 4]])

    def test_extract_outputs_continuous_full_sequence(self):
        """Continuous mode, full_sequence=True: slice output_ids from all_states."""
        model = OdyssNet(num_neurons=5, input_ids=[0, 1], output_ids=[3, 4], device="cpu")
        t = _trainer(model)
        steps, batch, n = 3, 4, 5
        all_states  = torch.randn(batch, steps, n)
        final_state = torch.randn(batch, n)  # not used in this branch
        out = t._extract_outputs(all_states, final_state, full_sequence=True)
        assert out.shape == (batch, steps, 2)
        assert torch.equal(out, all_states[:, :, [3, 4]])

    # --- _extract_outputs: vocab mode ----------------------------------------

    def test_extract_outputs_vocab_single_step(self):
        """Vocab mode, full_sequence=False: last-timestep slice of all_states."""
        model = OdyssNet(
            num_neurons=4, input_ids=[0], output_ids=[3],
            device="cpu", vocab_size=8, vocab_mode="discrete",
        )
        t = _trainer(model)
        batch, steps, vocab = 3, 5, 8
        all_states = torch.randn(batch, steps, vocab)
        out = t._extract_outputs(all_states, None, full_sequence=False)
        assert out.shape == (batch, vocab)
        assert torch.equal(out, all_states[:, -1, :])

    def test_extract_outputs_vocab_full_sequence(self):
        """Vocab mode, full_sequence=True: return all_states as-is."""
        model = OdyssNet(
            num_neurons=4, input_ids=[0], output_ids=[3],
            device="cpu", vocab_size=8, vocab_mode="discrete",
        )
        t = _trainer(model)
        batch, steps, vocab = 2, 6, 8
        all_states = torch.randn(batch, steps, vocab)
        out = t._extract_outputs(all_states, None, full_sequence=True)
        assert out is all_states
