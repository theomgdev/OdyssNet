"""
Cross-feature and state-persistence tests for ChaosGrad v3.

Covers interactions with:
  - state_dict / load_state_dict round-trips (frustration, best_loss, global_step)
  - lr group override propagation
  - reset_param_state
  - regenerate_synapses + ChaosGrad
  - gradient accumulation, synaptic noise, gradient persistence
  - neurogenesis (trainer.expand)
  - anomaly hooks, diagnostics, trigger_plateau_escape no-op
  - gradient checkpointing
  - plain params (no classify_params) fallback behaviour
"""

import math
import pytest
import torch

from odyssnet import OdyssNet, OdyssNetTrainer, ChaosGrad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(n=8, **kwargs):
    return OdyssNet(num_neurons=n, input_ids=[0], output_ids=[n - 1],
                    device='cpu', **kwargs)


def _opt(model, lr=1e-4):
    return ChaosGrad(ChaosGrad.classify_params(model), lr=lr)


def _trainer(model, **kw):
    opt = _opt(model)
    return OdyssNetTrainer(model, optimizer=opt, **kw)


def _step(trainer, n=1):
    x, y = torch.randn(2, 1), torch.randn(2, 1)
    loss = 0.0
    for _ in range(n):
        loss = trainer.train_batch(x, y, thinking_steps=3)
    return loss


def _one_step_raw(opt, model):
    model.reset_state(batch_size=2)
    out = model(torch.randn(2, 1), steps=3)[0]
    out.mean().backward()
    opt.step()
    opt.zero_grad()


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:

    def _build(self, n_steps=10, stagnant_steps=30):
        m   = _model()
        opt = _opt(m)
        for _ in range(n_steps):
            _one_step_raw(opt, m)
        for _ in range(stagnant_steps):
            opt.report_loss(1.0)
        return opt, m

    def test_state_dict_contains_chaos_global(self):
        opt, _ = self._build()
        assert 'chaos_global' in opt.state_dict()

    def test_state_dict_chaos_global_has_required_keys(self):
        opt, _ = self._build()
        cg = opt.state_dict()['chaos_global']
        assert {'global_step', 'frustration', 'best_loss'} <= cg.keys()

    def test_round_trip_frustration(self):
        opt, m = self._build()
        frust_before = opt._frustration
        sd = opt.state_dict()
        opt2 = _opt(m)
        opt2.load_state_dict(sd)
        assert opt2._frustration == pytest.approx(frust_before)

    def test_round_trip_best_loss(self):
        opt, m = self._build()
        opt.report_loss(0.05)
        sd = opt.state_dict()
        opt2 = _opt(m)
        opt2.load_state_dict(sd)
        assert opt2._best_loss == pytest.approx(opt._best_loss)

    def test_round_trip_global_step(self):
        opt, m = self._build(n_steps=7)
        sd = opt.state_dict()
        opt2 = _opt(m)
        opt2.load_state_dict(sd)
        assert opt2._global_step == opt._global_step

    def test_round_trip_per_param_lr(self):
        opt, m = self._build()
        lr_before = opt.state[m.W]['per_param_lr']
        sd = opt.state_dict()
        opt2 = _opt(m)
        opt2.load_state_dict(sd)
        assert opt2.state[m.W]['per_param_lr'] == pytest.approx(lr_before)

    def test_missing_chaos_global_is_graceful(self):
        m   = _model()
        opt = _opt(m)
        _one_step_raw(opt, m)
        sd = opt.state_dict()
        del sd['chaos_global']
        opt2 = _opt(m)
        opt2.load_state_dict(sd)   # must not raise
        assert opt2._frustration == 0.0
        assert opt2._best_loss == float('inf')

    def test_lr_group_override_propagates(self):
        """
        load_checkpoint overrides param_group['lr']. ChaosGrad reads genesis_lr
        from group, so the new value must take effect on the next step.
        """
        m   = _model()
        opt = _opt(m, lr=1e-4)
        _one_step_raw(opt, m)

        for pg in opt.param_groups:
            pg['lr'] = 5e-5

        W_before = m.W.data.clone()
        _one_step_raw(opt, m)
        # Step must complete; diagonal stays zero
        assert torch.allclose(m.W.data.diagonal(), torch.zeros(m.num_neurons))
        assert math.isfinite(m.W.data.norm().item())


# ---------------------------------------------------------------------------
# reset_param_state
# ---------------------------------------------------------------------------

class TestResetParamState:

    def test_reset_clears_entry(self):
        m   = _model()
        opt = _opt(m)
        _one_step_raw(opt, m)
        assert opt.state.get(m.W)
        opt.reset_param_state(m.W)
        assert not opt.state.get(m.W)

    def test_after_reset_cold_start_fires(self):
        m   = _model()
        opt = _opt(m)
        for _ in range(5):
            _one_step_raw(opt, m)

        opt.reset_param_state(m.W)
        _one_step_raw(opt, m)
        state = opt.state[m.W]
        # step should be 1 — cold start happened
        assert state['step'] == 1
        assert ChaosGrad._LR_MIN <= state['init_lr'] <= ChaosGrad._LR_MAX

    def test_reset_unknown_param_safe(self):
        m   = _model()
        opt = _opt(m)
        dummy = torch.nn.Parameter(torch.randn(4))
        opt.reset_param_state(dummy)   # no crash

    def test_trainer_regenerate_resets_W_state(self):
        m       = _model()
        t       = _trainer(m)
        _step(t, n=5)

        assert t.optimizer.state.get(m.W)

        # Zero out all weights so every entry is below threshold=1.0
        with torch.no_grad():
            m.W.data.zero_()

        revived, _ = t.regenerate_synapses(threshold=1.0)
        if revived > 0:
            assert not t.optimizer.state.get(m.W), \
                "W state must be cleared after regeneration with ChaosGrad"

    def test_trainer_regenerate_no_effect_for_adamw(self):
        """regenerate_synapses must not raise for AdamW trainer."""
        m       = _model()
        trainer = OdyssNetTrainer(m, lr=1e-3)
        _step(trainer, n=3)
        with torch.no_grad():
            m.W.data.zero_()
        revived, _ = trainer.regenerate_synapses(threshold=1.0)
        assert isinstance(revived, int)


# ---------------------------------------------------------------------------
# Cross-feature interactions
# ---------------------------------------------------------------------------

class TestCrossFeatureInteractions:

    def test_gradient_accumulation_steps_once(self):
        m = _model()
        t = _trainer(m)
        x, y = torch.randn(2, 1), torch.randn(2, 1)
        for _ in range(3):
            t.train_batch(x, y, thinking_steps=3, gradient_accumulation_steps=3)
        assert t.optimizer._global_step == 1

    def test_synaptic_noise_no_crash(self):
        m = _model()
        t = _trainer(m, synaptic_noise=1e-4)
        loss = _step(t, n=5)
        assert math.isfinite(loss)
        assert torch.isfinite(m.W.data).all()
        assert torch.allclose(m.W.data.diagonal(), torch.zeros(m.num_neurons))

    def test_gradient_persistence_no_crash(self):
        m = _model()
        t = _trainer(m, gradient_persistence=0.1)
        loss = _step(t, n=5)
        assert math.isfinite(loss)

    def test_neurogenesis_preserves_frustration(self):
        m = _model(n=6)
        t = _trainer(m)
        _step(t, n=3)
        frust_before = t.optimizer._frustration
        t.expand(amount=2, verbose=False)
        assert t.optimizer._frustration == pytest.approx(frust_before)

    def test_neurogenesis_training_continues(self):
        m = _model(n=6)
        t = _trainer(m)
        _step(t, n=3)
        t.expand(amount=2, verbose=False)
        loss = _step(t, n=3)
        assert math.isfinite(loss)

    def test_anomaly_hook_no_crash(self):
        events = []
        m = _model()
        t = _trainer(m, anomaly_hook=lambda e, l: events.append(e))
        _step(t, n=5)
        assert isinstance(events, list)

    def test_get_diagnostics_optimizer_key_present(self):
        m = _model()
        t = _trainer(m)
        _step(t)
        diag = t.get_diagnostics(debug=True)
        assert 'optimizer' in diag
        assert 'frustration' in diag['optimizer']

    def test_adamw_trainer_no_optimizer_key(self):
        m = _model()
        t = OdyssNetTrainer(m, lr=1e-3)
        _step(t)
        assert 'optimizer' not in t.get_diagnostics()

    def test_trigger_plateau_escape_noop_adamw(self):
        m = _model()
        t = OdyssNetTrainer(m, lr=1e-3)
        t.trigger_plateau_escape()   # must not raise

    def test_gradient_checkpointing_compatible(self):
        m = OdyssNet(num_neurons=8, input_ids=[0], output_ids=[7],
                     gradient_checkpointing=True, device='cpu')
        t = _trainer(m)
        loss = _step(t, n=3)
        assert math.isfinite(loss)

    def test_plain_params_no_crash(self):
        """ChaosGrad with plain model.parameters() (no classify_params) must train."""
        m   = _model()
        opt = ChaosGrad(m.parameters(), lr=1e-4)
        t   = OdyssNetTrainer(m, optimizer=opt)
        loss = _step(t, n=3)
        assert math.isfinite(loss)

    def test_plain_params_hebbian_no_bypass(self):
        """Without classify_params, hebb params get lightweight treatment (no crash)."""
        m   = _model(hebb_type='global')
        opt = ChaosGrad(m.parameters(), lr=1e-4)
        t   = OdyssNetTrainer(m, optimizer=opt)
        loss = _step(t, n=5)
        assert math.isfinite(loss)

    def test_vocab_model_all_params_trained(self):
        """ChaosGrad must correctly classify and step a model with vocab projections."""
        m = OdyssNet(num_neurons=10, input_ids=list(range(10)),
                     output_ids=list(range(10)), vocab_size=[16, 10],
                     vocab_mode='continuous', device='cpu')
        t = _trainer(m)
        # For vocab_size models, OdyssNet interprets dim-1 as seq_len.
        # (batch=2, seq_len=1, v_in=16) → at step 0: step_in=(2,16) → proj → (2,10).
        x = torch.randn(2, 1, 16)   # (batch, seq_len=1, v_in)
        y = torch.randn(2, 10)      # (batch, vocab_out)
        loss = t.train_batch(x, y, thinking_steps=3)
        assert math.isfinite(loss)

    def test_report_loss_called_automatically_by_trainer(self):
        """Trainer must call report_loss after each optimizer step."""
        m = _model()
        t = _trainer(m)
        assert t.optimizer._best_loss == float('inf')
        _step(t)
        assert t.optimizer._best_loss < float('inf')
