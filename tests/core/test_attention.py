"""
Unit tests for odyssnet.core.attention.TemporalAttention and its wiring into
OdyssNet.

Covers:
- Construction, validation and auto-sizing of the head geometry
- Zero-output initialization: attention on == attention off at step zero
- Exactness of the log-sum-exp segment merge against a single softmax
- KV cache: incremental decode matches a one-shot pass, carry across calls,
  window eviction, ring/segmented parity, reset semantics
- Grouped-query / multi-query attention
- Gradient flow into every attention parameter, and gradient-checkpointing
  equivalence
- Row-wise state and cache resets (staggered cold starts)
"""

import pytest
import torch

from odyssnet import OdyssNet
from odyssnet.core.attention import TemporalAttention
from odyssnet.utils.data import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model(seed=7, loud=True, **kwargs):
    """A small vocab-mode model with attention enabled.

    `loud=True` breaks the zero initialization of `o_proj` so the attention
    branch actually contributes — most tests here are meaningless while it is
    an exact no-op.
    """
    set_seed(seed)
    kwargs.setdefault("attn_heads", 4)
    model = OdyssNet(
        num_neurons=32,
        input_ids=list(range(8)),
        output_ids=list(range(8, 16)),
        device="cpu",
        vocab_size=50,
        vocab_mode="discrete",
        **kwargs,
    )
    if loud and model.attn is not None:
        torch.nn.init.normal_(model.attn.o_proj.weight, std=0.05)
    return model


def _tokens(batch=3, length=6, seed=0):
    torch.manual_seed(seed)
    return torch.randint(0, 50, (batch, length))


# ===========================================================================
# Construction
# ===========================================================================

class TestConstruction:
    def test_disabled_by_default(self):
        model = _model(loud=False, attn_heads=None)
        assert model.attn is None
        assert not any("attn" in name for name in model.state_dict())

    def test_enabled_creates_module(self):
        model = _model()
        assert isinstance(model.attn, TemporalAttention)
        assert model.attn.heads == 4

    def test_head_dim_autosizes_even(self):
        attn = TemporalAttention(100, heads=3)
        assert attn.head_dim % 2 == 0
        assert attn.head_dim == 32          # min(64, 100 // 3) rounded even

    def test_head_dim_capped_at_64(self):
        assert TemporalAttention(4096, heads=4).head_dim == 64

    def test_kv_heads_must_divide_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            TemporalAttention(32, heads=4, kv_heads=3)

    def test_odd_head_dim_rejected_with_rope(self):
        with pytest.raises(ValueError, match="even"):
            TemporalAttention(32, heads=2, head_dim=7, rope=True)

    def test_odd_head_dim_allowed_without_rope(self):
        assert TemporalAttention(32, heads=2, head_dim=7, rope=False).head_dim == 7

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="attn_window"):
            TemporalAttention(32, heads=2, window=0)

    def test_invalid_write_mode(self):
        with pytest.raises(ValueError, match="attn_write"):
            _model(attn_write="every-other-tuesday")

    def test_invalid_read_mode(self):
        with pytest.raises(ValueError, match="attn_read"):
            _model(attn_read="continuously")

    def test_rope_table_is_not_persistent(self):
        model = _model()
        assert "attn.inv_freq" not in model.state_dict()

    def test_weight_init_accepts_fifth_entry(self):
        model = _model(weight_init=["quiet", "resonant", "quiet", "zero", "zero"])
        assert torch.all(model.attn.q_proj.weight == 0.0)

    def test_four_entry_weight_init_still_works(self):
        model = _model(weight_init=["quiet", "resonant", "quiet", "zero"])
        assert model.attn.q_proj.weight.abs().sum() > 0


# ===========================================================================
# Zero-output initialization
# ===========================================================================

class TestZeroInit:
    def test_o_proj_starts_at_zero(self):
        model = _model(loud=False)
        assert torch.all(model.attn.o_proj.weight == 0.0)

    def test_attention_is_a_no_op_at_init(self):
        """An attention model must start life numerically identical to the
        same model without attention — including its core weights, which is
        why the module is built after the core is initialized."""
        x = _tokens()
        plain = _model(loud=False, attn_heads=None)
        attentive = _model(loud=False)

        plain.reset_state(3)
        expected, _ = plain(x, steps=10)
        attentive.reset_state(3)
        actual, _ = attentive(x, steps=10)

        assert torch.allclose(expected, actual, atol=1e-7)

    def test_core_weights_identical_across_the_switch(self):
        assert torch.equal(_model(loud=False, attn_heads=None).W,
                           _model(loud=False).W)


class TestBranchScale:
    """`o_proj` starts at zero and Adam-family updates move it by ~lr per step
    whatever its width, so without the 1/sqrt(heads*head_dim) factor a wide
    branch reaches a destructive contribution in the same number of steps a
    narrow one takes to reach a useful one. Measured on associative recall:
    unscaled, 4x64 collapsed to chance while 1x16 matched the baseline."""

    def _contribution(self, heads, head_dim, seed=3):
        set_seed(seed)
        attn = TemporalAttention(128, heads=heads, head_dim=head_dim, kv_heads=1)
        torch.nn.init.normal_(attn.o_proj.weight, std=0.05)
        torch.manual_seed(11)
        h = torch.randn(8, 128)
        keys = torch.randn(8, 1, 12, head_dim)
        values = torch.randn(8, 1, 12, head_dim)
        return attn.attend(h, ((keys, values),), pos=12).std().item()

    def test_contribution_is_width_independent(self):
        scales = [self._contribution(h, d)
                  for h, d in ((1, 16), (2, 32), (4, 64), (8, 64))]
        assert max(scales) / min(scales) < 1.5, scales

    def test_out_scale_matches_branch_width(self):
        attn = TemporalAttention(128, heads=4, head_dim=64)
        assert attn.out_scale == pytest.approx(1.0 / (4 * 64) ** 0.5)


# ===========================================================================
# Attention mathematics
# ===========================================================================

class TestSegmentMerge:
    def test_split_segments_match_one_softmax(self):
        """The cache is attended as segments and joined at the scores. That
        has to be exactly what one softmax over the concatenation would
        give — otherwise the training and inference paths, which segment
        differently, would disagree."""
        set_seed(1)
        attn = TemporalAttention(16, heads=2, kv_heads=1, head_dim=8, window=64)
        torch.nn.init.normal_(attn.o_proj.weight, std=0.1)
        h = torch.randn(2, 16)
        keys, values = torch.randn(2, 1, 9, 8), torch.randn(2, 1, 9, 8)

        one = attn.attend(h, ((keys, values),), pos=9)
        split = attn.attend(h, ((keys[:, :, :4], values[:, :, :4]),
                                (keys[:, :, 4:], values[:, :, 4:])), pos=9)

        assert torch.allclose(one, split, atol=1e-6)

    def test_split_is_order_faithful(self):
        """Segments carry their own values: probabilities from one block must
        never be paired with another block's values."""
        set_seed(4)
        attn = TemporalAttention(16, heads=2, kv_heads=1, head_dim=8, window=64)
        torch.nn.init.normal_(attn.o_proj.weight, std=0.1)
        h = torch.randn(2, 16)
        keys, values = torch.randn(2, 1, 6, 8), torch.randn(2, 1, 6, 8)

        reference = attn.attend(h, ((keys, values),), pos=6)
        swapped = attn.attend(h, ((keys[:, :, :3], values[:, :, 3:]),
                                  (keys[:, :, 3:], values[:, :, :3])), pos=6)

        assert not torch.allclose(reference, swapped, atol=1e-4)

    def test_empty_cache_returns_none(self):
        attn = TemporalAttention(16, heads=2, head_dim=8)
        assert attn.attend(torch.randn(2, 16), (), pos=0) is None

    def test_permutation_invariance(self):
        """Order in the cache carries no meaning — RoPE is already inside each
        key — which is what lets the inference cache be a ring."""
        set_seed(2)
        attn = TemporalAttention(16, heads=2, kv_heads=1, head_dim=8)
        torch.nn.init.normal_(attn.o_proj.weight, std=0.1)
        h = torch.randn(2, 16)
        keys, values = torch.randn(2, 1, 5, 8), torch.randn(2, 1, 5, 8)
        order = torch.tensor([3, 0, 4, 1, 2])

        straight = attn.attend(h, ((keys, values),), pos=5)
        shuffled = attn.attend(h, ((keys[:, :, order], values[:, :, order]),), pos=5)

        assert torch.allclose(straight, shuffled, atol=1e-6)


# ===========================================================================
# KV cache
# ===========================================================================

class TestKVCache:
    def test_one_entry_per_token_by_default(self):
        model = _model()
        model.reset_state(3)
        model(_tokens(length=6), steps=12)      # think_gap 1 -> 2 steps/token
        assert model.attn.cache_len == 6

    def test_write_step_records_every_step(self):
        model = _model(attn_write="step")
        model.reset_state(3)
        model(_tokens(length=6), steps=12)
        assert model.attn.cache_len == 12

    def test_read_token_queries_once_per_token(self):
        """`attn_read` changes how often the cache is queried, not what is in
        it: the entries are the same, the outputs are not."""
        x = _tokens(length=6)
        every_step = _model(attn_read="step")
        per_token = _model(attn_read="token")
        per_token.load_state_dict(every_step.state_dict())

        every_step.reset_state(3)
        dense, _ = every_step(x, steps=12)
        per_token.reset_state(3)
        sparse, _ = per_token(x, steps=12)

        assert per_token.attn.cache_len == every_step.attn.cache_len == 6
        assert not torch.allclose(dense, sparse, atol=1e-5)

    def test_read_modes_agree_at_one_step_per_token(self):
        x = _tokens(length=6)
        every_step = _model(attn_read="step")
        per_token = _model(attn_read="token")
        per_token.load_state_dict(every_step.state_dict())

        every_step.reset_state(3)
        dense, _ = every_step(x, steps=6)
        per_token.reset_state(3)
        sparse, _ = per_token(x, steps=6)

        assert torch.allclose(dense, sparse, atol=1e-7)

    def test_incremental_decode_matches_one_shot(self):
        """The point of a KV cache: feeding tokens one at a time must produce
        exactly what feeding them together does."""
        model = _model()
        x = _tokens(batch=1, length=6)

        model.reset_state(1)
        with torch.no_grad():
            full, _ = model(x, steps=12)

        model.reset_state(1)
        with torch.no_grad():
            pieces = [model(x[:, i:i + 1], steps=2)[0] for i in range(6)]
        assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-5)

    def test_history_carries_across_calls(self):
        model = _model()
        x = _tokens(batch=2, length=6)

        model.reset_state(2)
        one_call, _ = model(x, steps=12)
        model.reset_state(2)
        first, _ = model(x[:, :3], steps=6)
        second, _ = model(x[:, 3:], steps=6)

        assert torch.allclose(one_call, torch.cat([first, second], dim=1), atol=1e-5)

    def test_reset_state_clears_the_cache(self):
        model = _model()
        model.reset_state(2)
        model(_tokens(batch=2, length=3), steps=6)
        assert model.attn.cache_len == 3
        model.reset_state(2)
        assert model.attn.cache_len == 0
        assert model.attn.position == 0

    def test_window_evicts_oldest(self):
        model = _model(attn_window=4)
        model.reset_state(3)
        model(_tokens(length=9), steps=18)
        assert model.attn.cache_len == 4
        assert model.attn.position == 9

    @pytest.mark.parametrize("window", [2, 5, 64])
    def test_ring_and_segmented_paths_agree(self, window):
        """Inference uses a preallocated ring, training a segmented graph
        cache. They are two representations of one cache and must score the
        same, eviction included."""
        model = _model(attn_window=window)
        x = _tokens(length=9)

        model.reset_state(3)
        trained = model(x, steps=18)[0]
        model.reset_state(3)
        with torch.no_grad():
            inferred = model(x, steps=18)[0]

        assert torch.allclose(trained, inferred, atol=1e-5)

    def test_ring_is_preallocated_once(self):
        model = _model(attn_window=8)
        model.reset_state(1)
        with torch.no_grad():
            model(_tokens(batch=1, length=4), steps=8)
            storage = model.attn._ring_k.data_ptr()
            model(_tokens(batch=1, length=4, seed=1), steps=8)
        assert model.attn._ring_k.data_ptr() == storage

    def test_snapshot_restore(self):
        model = _model()
        model.reset_state(2)
        with torch.no_grad():
            model(_tokens(batch=2, length=4), steps=8)
        snap = model.attn.snapshot()
        with torch.no_grad():
            model(_tokens(batch=2, length=4, seed=9), steps=8)
        model.attn.restore(snap)
        assert model.attn.cache_len == 4
        assert model.attn.position == 4

    def test_cache_is_float32_under_autocast(self):
        """Attention runs with autocast off, so the cache has one dtype
        whatever the surrounding precision is — a cache written in fp16 during
        training and read in fp32 during evaluation would otherwise have to be
        converted at exactly the wrong moment. It is not free accuracy either:
        half-precision attention trailed this path by a stable ~0.017 of loss
        on embedded MNIST with four heads."""
        model = _model()
        model.reset_state(2)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            model(_tokens(batch=2, length=4), steps=8)
        assert model.attn._pend_k or model.attn._mem_k is not None
        cached = model.attn._mem_k if model.attn._mem_k is not None else model.attn._pend_k[0]
        assert cached.dtype == torch.float32

    def test_read_and_write_stay_out_of_the_compiled_graph(self):
        """The fp32 region above is what `torch.compile` miscompiles — Inductor
        reports float for a buffer it emits as half and `o_proj` dies on the
        mismatch — so both methods are hidden from Dynamo. Removing the
        decorator silently breaks every compiled run with attention on, and
        costs nothing to keep: the rest of the step still fuses around the
        break (17.4 ms/batch against 17.6 fully traced in half, 60.4 eager)."""
        assert getattr(TemporalAttention.attend, "_torchdynamo_disable", False)
        assert getattr(TemporalAttention.write, "_torchdynamo_disable", False)

    def test_cost_model_reports_bytes(self):
        attn = TemporalAttention(64, heads=4, kv_heads=1, head_dim=16, window=10)
        assert attn.cache_bytes(batch=2) == 2 * 2 * 1 * 10 * 16 * 4
        assert attn.training_cache_bytes(batch=1, writes=0) == attn.cache_bytes(1)


# ===========================================================================
# Grouped-query attention
# ===========================================================================

class TestGroupedQuery:
    @pytest.mark.parametrize("kv_heads", [1, 2, 4])
    def test_forward_shapes(self, kv_heads):
        model = _model(attn_kv_heads=kv_heads)
        model.reset_state(3)
        out, _ = model(_tokens(), steps=12)
        assert out.shape == (3, 6, 50)

    def test_kv_projection_width_follows_kv_heads(self):
        model = _model(attn_kv_heads=2)
        assert model.attn.k_proj.weight.shape[0] == 2 * model.attn.head_dim
        assert model.attn.q_proj.weight.shape[0] == 4 * model.attn.head_dim

    def test_multi_query_shrinks_the_cache(self):
        mqa = _model(attn_kv_heads=1).attn.cache_bytes(batch=8)
        mha = _model(attn_kv_heads=4).attn.cache_bytes(batch=8)
        assert mha == 4 * mqa


# ===========================================================================
# Gradients
# ===========================================================================

class TestGradients:
    def test_every_attention_parameter_receives_gradient(self):
        model = _model()
        model.reset_state(3)
        out, _ = model(_tokens(), steps=12)
        out.sum().backward()
        for name, param in model.named_parameters():
            if "attn" in name:
                assert param.grad is not None, name
                assert param.grad.abs().sum() > 0, name

    def test_gradient_checkpointing_matches(self):
        plain = _model()
        checkpointed = _model(gradient_checkpointing=True)
        checkpointed.load_state_dict(plain.state_dict())
        plain.train()
        checkpointed.train()
        x = _tokens()

        plain.reset_state(3)
        plain(x, steps=12)[0].sum().backward()
        checkpointed.reset_state(3)
        checkpointed(x, steps=12)[0].sum().backward()

        assert torch.allclose(plain.attn.q_proj.weight.grad,
                              checkpointed.attn.q_proj.weight.grad, atol=1e-5)
        assert torch.allclose(plain.W.grad, checkpointed.W.grad, atol=1e-5)

    def test_carried_cache_does_not_leak_graph(self):
        """Truncated BPTT: what a previous call wrote is a constant, so a
        second backward must not walk into the first call's graph."""
        model = _model()
        model.reset_state(2)
        model(_tokens(batch=2, length=3), steps=6)[0].sum().backward()
        out, _ = model(_tokens(batch=2, length=3, seed=1), steps=6)
        out.sum().backward()          # would raise if the graph were retained


# ===========================================================================
# Row-wise reset
# ===========================================================================

class TestResetRows:
    def test_zeroes_selected_state_rows_only(self):
        model = _model()
        model.reset_state(3)
        with torch.no_grad():
            model(_tokens(), steps=12)
        model.reset_rows(torch.tensor([True, False, False]))
        assert torch.all(model.state[0] == 0)
        assert model.state[1].abs().sum() > 0

    def test_zeroes_selected_cache_rows_only(self):
        model = _model()
        model.reset_state(3)
        with torch.no_grad():
            model(_tokens(), steps=12)
        model.reset_rows(torch.tensor([True, False, False]))
        assert torch.all(model.attn._ring_k[0] == 0)
        assert model.attn._ring_k[1].abs().sum() > 0

    def test_reset_row_attends_to_nothing(self):
        """A zeroed row spreads its softmax over all-zero values, so its
        attention output is exactly zero — the same thing an empty cache
        gives it."""
        model = _model()
        model.reset_state(2)
        with torch.no_grad():
            model(_tokens(batch=2, length=4), steps=8)
        model.reset_rows(torch.tensor([True, False]))
        segments, pos = model.attn.cache_view()
        out = model.attn.attend(torch.randn(2, 32), segments, pos)
        assert torch.allclose(out[0], torch.zeros_like(out[0]), atol=1e-7)

    def test_rejects_wrong_mask_length(self):
        model = _model()
        model.reset_state(3)
        with pytest.raises(ValueError, match="mask"):
            model.reset_rows(torch.tensor([True, False]))

    def test_works_without_attention(self):
        model = _model(loud=False, attn_heads=None)
        model.reset_state(3)
        model.reset_rows(torch.tensor([True, False, True]))
        assert torch.all(model.state[0] == 0)


# ===========================================================================
# Training integration
# ===========================================================================

class TestTrainingIntegration:
    def test_trainer_step_runs_and_moves_the_loss(self):
        from odyssnet import OdyssNetTrainer

        model = _model()
        trainer = OdyssNetTrainer(model, device="cpu", lr=1e-3)
        trainer.loss_fn = torch.nn.CrossEntropyLoss()
        x = _tokens(batch=4, length=4)
        y = x.reshape(-1)
        flatten = lambda o: o.reshape(-1, o.shape[-1])

        first = trainer.train_batch(x, y, thinking_steps=8, full_sequence=True,
                                    output_transform=flatten)
        for _ in range(20):
            last = trainer.train_batch(x, y, thinking_steps=8, full_sequence=True,
                                       output_transform=flatten)
        assert last < first

    def test_optimizer_puts_projections_in_the_attention_family(self):
        from odyssnet import ChaosGrad

        model = _model()
        names = {g["group_name"]: g for g in ChaosGrad.classify_params(model)}
        assert "attention" in names
        assert names["attention"]["weight_decay"] > 0
        attention_params = {id(p) for p in names["attention"]["params"]}
        assert id(model.attn.q_proj.weight) in attention_params
        # QK-norm gains are not connective structure and must not decay.
        assert id(model.attn.q_norm.weight) in {id(p) for p in names["modulation"]["params"]}
