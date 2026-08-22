"""
Temporal attention for the chaos core (OdyssNet 3.0).

OdyssNet has no layers to stack attention *between*. Depth here is time: one
NxN matrix `W` that the state echoes through, one step at a time. So attention
is attached along the only axis the architecture has — the state's own past.

At every step the current state queries a cache of earlier states:

    q_t = h_t W_q                      one query, always
    k_i = h_i W_k , v_i = h_i W_v      one entry per *written* step
    a_t = softmax(q_t K^T / sqrt(d)) V
    h_t += a_t W_o                      (added to the pre-activation signal)

Two consequences shape every design decision in this file.

**The query length is always 1.** The core cannot be unrolled in parallel — step
t's input is step t-1's output — so a forward pass is a sequence of single-query
attentions, exactly like a transformer's decode phase and never like its prefill.
The score matrix is therefore `(B, H, 1, L)`: one row per key, small enough that
there is nothing for a fused flash kernel to tile away
(`F.scaled_dot_product_attention` solves a problem this shape does not have).
What *is* expensive is re-materializing K and V, so the cache is kept in
segments that are never concatenated — only their score blocks are, which is
the cheap end of the same join and keeps the result a single exact softmax.

**Order does not matter, so the cache can be a ring.** Softmax over keys is
permutation-invariant, and every entry in the cache is strictly in the past
(the write for step t happens after the read for step t), so no causal mask is
ever needed and the buffer never has to be reordered. Position information is
carried by RoPE, applied to each key *at write time* against its absolute
position, so a rotated key stays correct wherever it later sits in the buffer.

Cache representations
---------------------
The cache exists in two forms and switches between them automatically:

* **Ring** (grad disabled — inference, validation, generation): a preallocated
  `(B, H_kv, window, D)` buffer written in place through a narrowed view. Zero
  allocation per token, which is the whole point of a KV cache.
* **Segmented** (grad enabled — training): in-place writes are illegal under
  autograd, so the carry from previous calls is held as one frozen tensor
  (saved once for backward, however many steps read it) and this call's writes
  accumulate in a list. Reads join the two at the scores.

The switch is one copy, at the first write after the mode changes.

Memory
------
Training retains, per read, the keys and values it attended. The frozen segment
is a single shared tensor, so it costs `2·B·H_kv·window·D` floats *once*; the
growing segment costs `2·B·H_kv·D·(1+2+...+n)` for n writes in the call. That
quadratic is in writes-per-forward (tokens per truncated-BPTT window), not in
the run length, and it is why `kv_heads` defaults to 1 (multi-query): the KV
cache, not the projections, is the term that decides whether a batch fits.

Speed
-----
Every one of these operations is tiny and there are many of them per step, so
this path is bound by kernel launches rather than by arithmetic — measured on
an RTX 3060 Ti at 1024 neurons, attention runs 3.3x slower per token at batch
128 and 2.8x at batch 512, because the overhead is per *step* and a larger
batch amortizes it. Head count barely registers (4 heads cost the same as 1),
which is the signature of that regime. Two consequences are baked in here:
attention runs with autocast disabled (a query of length 1 has nothing to gain
from fp16, and the implicit casts around it cost more than the math), and the
segments are joined once at the scores rather than merged afterwards.
"""

import math

import torch
import torch.nn as nn


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class TemporalAttention(nn.Module):
    """
    Multi-head attention over the chaos core's own state history.

    Args:
        num_neurons (int): Width of the state — attention reads and writes the
            full `(B, N)` vector, since OdyssNet has no other representation.
        heads (int): Query heads. Default 4.
        head_dim (int, optional): Per-head width. Defaults to
            `min(64, num_neurons // heads)`, rounded down to an even number
            (RoPE rotates coordinate pairs).
        kv_heads (int, optional): Key/value heads, shared across query heads
            (grouped-query attention). Must divide `heads`. Default 1
            (multi-query): the cache is the dominant memory term in a
            step-sequential model, and this divides it by `heads`.
        window (int): How many entries the cache holds. Attention never sees
            more than this many steps back; the oldest are evicted first, in
            both cache representations. Default 256.
        rope (bool): Rotary position embedding, applied at write time against
            the entry's absolute write index. Default True.
        rope_theta (float): RoPE base frequency. Default 10000.0.
        qk_norm (bool): RMSNorm on queries and keys before the dot product.
            Default True — the core is chaotic and its state norm is only
            controlled at the end of a step, so unnormalized logits here are
            the first thing to blow up.
        dropout (float): Dropout on the attention weights during training.
            Applied to the joined score block. Default 0.0.
        device: Torch device for the projections.

    The output projection is zero-initialized and the branch is divided by
    `sqrt(heads * head_dim)`, so switching attention on — or widening it —
    does not change the scale of what reaches the recurrence. See `out_scale`
    below; that factor is the difference between attention helping and
    attention drowning the core.
    """

    def __init__(self, num_neurons, heads=4, head_dim=None, kv_heads=1,
                 window=256, rope=True, rope_theta=10000.0, qk_norm=True,
                 dropout=0.0, device=None):
        super().__init__()

        if num_neurons <= 0:
            raise ValueError(f"attention needs a positive num_neurons, got {num_neurons}")
        if heads < 1:
            raise ValueError(f"attn_heads must be >= 1, got {heads}")
        if kv_heads < 1:
            raise ValueError(f"attn_kv_heads must be >= 1, got {kv_heads}")
        if heads % kv_heads:
            raise ValueError(
                f"attn_heads ({heads}) must be divisible by attn_kv_heads ({kv_heads})"
            )
        if window < 1:
            raise ValueError(f"attn_window must be >= 1, got {window}")

        if head_dim is None:
            # Rounded down to even so the default is always RoPE-compatible;
            # an explicit odd value is the user's call and only rejected when
            # RoPE is actually on.
            head_dim = min(64, max(2, num_neurons // heads))
            head_dim -= head_dim % 2
        head_dim = int(head_dim)
        if head_dim < 2:
            raise ValueError(f"attn_head_dim must be >= 2, got {head_dim}")
        if head_dim % 2:
            # RoPE rotates (x_i, x_{i+D/2}) pairs; an odd width has no pairing.
            if rope:
                raise ValueError(
                    f"attn_head_dim must be even when RoPE is on, got {head_dim}"
                )

        self.num_neurons = num_neurons
        self.heads = heads
        self.kv_heads = kv_heads
        self.group = heads // kv_heads
        self.head_dim = head_dim
        self.window = int(window)
        self.rope = bool(rope)
        self.rope_theta = float(rope_theta)
        self.scale = 1.0 / math.sqrt(head_dim)

        # The branch's contribution is divided by the square root of its own
        # width, and this is load-bearing rather than cosmetic. `o_proj` starts
        # at zero and every optimizer in this library is Adam-family, so its
        # magnitude after k steps is set by the step size, not by the gradient:
        # |o_proj| ~ k*lr whatever the width. The contribution it produces
        # would then grow as sqrt(heads*head_dim) — a wide branch reaches a
        # destructive scale in the same number of steps a narrow one takes to
        # reach a useful one. Measured on associative recall at 128 neurons:
        # without this, 1 head x 16 matched the no-attention baseline (54%)
        # while 4 heads x 64 collapsed to chance (6%), having drowned the
        # recurrence it was supposed to assist. Dividing here makes the
        # reachable contribution width-independent, which is the same reasoning
        # behind GPT-2's 1/sqrt(2*n_layers) residual-branch init.
        self.out_scale = 1.0 / math.sqrt(heads * head_dim)

        self.q_proj = nn.Linear(num_neurons, heads * head_dim, bias=False, device=device)
        self.k_proj = nn.Linear(num_neurons, kv_heads * head_dim, bias=False, device=device)
        self.v_proj = nn.Linear(num_neurons, kv_heads * head_dim, bias=False, device=device)
        # Zero-initialized: at step zero the attention branch contributes
        # exactly nothing, so an attention-enabled model *is* the 2.x model
        # until training decides otherwise. Every known-good initialization
        # story (resonant core, edge of chaos) survives switching this on.
        self.o_proj = nn.Linear(heads * head_dim, num_neurons, bias=False, device=device)
        nn.init.zeros_(self.o_proj.weight)

        self.q_norm = nn.RMSNorm(head_dim, device=device) if qk_norm else None
        self.k_norm = nn.RMSNorm(head_dim, device=device) if qk_norm else None
        self.attn_drop = nn.Dropout(p=dropout)

        if self.rope:
            inv = 1.0 / (self.rope_theta ** (
                torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim
            ))
            # Not persistent: it is a constant derived from the config, and
            # keeping it out of the state dict keeps checkpoints comparable
            # with models that only differ in `rope_theta`.
            self.register_buffer('inv_freq', inv, persistent=False)
        else:
            self.inv_freq = None

        self._rope_memo = None
        self._reset_cache_storage()

    # ------------------------------------------------------------------ #
    # Cache storage                                                       #
    # ------------------------------------------------------------------ #

    def _reset_cache_storage(self):
        # Segmented (grad-enabled) representation.
        self._mem_k = None            # frozen carry, (B, H_kv, L, D)
        self._mem_v = None
        self._pend_k = []             # this call's writes, differentiable
        self._pend_v = []
        self._pend_cat = None         # memoized concat of the pending list
        # Ring (grad-disabled) representation.
        self._ring_k = None           # (B, H_kv, window, D)
        self._ring_v = None
        self._ring_fill = 0
        self._ring_cursor = 0
        # Absolute write index, shared by both representations. Keys are
        # rotated against it; the query uses the count itself, so a query is
        # always one position ahead of the newest key.
        self._writes = 0

    def reset(self):
        """Forget everything. Called by `OdyssNet.reset_state()`."""
        self._reset_cache_storage()

    @property
    def cache_len(self):
        """Entries currently attendable."""
        if self._ring_k is not None:
            return self._ring_fill
        n = 0 if self._mem_k is None else self._mem_k.shape[2]
        return min(n + len(self._pend_k), self.window)

    @property
    def position(self):
        """Absolute index the next write will take (also the query position)."""
        return self._writes

    # ------------------------------------------------------------------ #
    # Rotary embedding                                                    #
    # ------------------------------------------------------------------ #

    def _rope_pair(self, pos, dtype, device):
        """
        cos/sin for a single absolute position.

        Computed on demand rather than from a table: positions grow with the
        run (a stateful stream can reach millions of tokens) and a table would
        have to grow with them. The angle is formed in float64 before the
        trigonometry — at a position of 1e6 the fastest-rotating dimension is
        already past float32's resolution, and a silently wrong phase is worse
        than the microsecond this costs.
        """
        # One-entry memo. Within a token group the query repeats its position
        # for every echo step, and the write that follows takes the same index,
        # so the hit rate is high enough to be worth the four lines.
        cached = self._rope_memo
        if cached is not None and cached[0] == pos and cached[1] == dtype and cached[2] == device:
            return cached[3], cached[4]
        angle = self.inv_freq.to(torch.float64) * float(pos)
        cos = torch.cos(angle).to(dtype).repeat(2)
        sin = torch.sin(angle).to(dtype).repeat(2)
        self._rope_memo = (pos, dtype, device, cos, sin)
        return cos, sin

    def _apply_rope(self, x, pos):
        cos, sin = self._rope_pair(pos, x.dtype, x.device)
        return x * cos + _rotate_half(x) * sin

    # ------------------------------------------------------------------ #
    # Read                                                                #
    # ------------------------------------------------------------------ #

    def cache_view(self):
        """
        The attendable cache as `(segments, position)`.

        `segments` is a tuple of `(K, V)` pairs — zero, one or two of them —
        handed to `attend()` as explicit arguments rather than read from `self`
        so that gradient checkpointing recomputes a step against the same cache
        it originally saw.
        """
        if self._ring_k is not None:
            if self._ring_fill == 0:
                return (), self._writes
            if self._ring_fill < self.window:
                seg = (self._ring_k[:, :, :self._ring_fill],
                       self._ring_v[:, :, :self._ring_fill])
            else:
                seg = (self._ring_k, self._ring_v)
            return (seg,), self._writes

        pending = None
        if self._pend_k:
            if self._pend_cat is None:
                self._pend_cat = (torch.cat(self._pend_k, dim=2),
                                  torch.cat(self._pend_v, dim=2))
            pending = self._pend_cat

        # Eviction is by slicing, never by copying: a narrowed tensor shares
        # storage with the one it came from, so holding the window to size
        # costs nothing even though every step asks for it again. That is what
        # lets this path evict as strictly as the ring does, instead of the
        # window meaning "carried between calls" here and "total" there.
        segments = []
        room = self.window
        if pending is not None:
            n = pending[0].shape[2]
            if n > room:
                pending = (pending[0][:, :, -room:], pending[1][:, :, -room:])
                n = room
            room -= n
        if self._mem_k is not None and room > 0:
            if self._mem_k.shape[2] > room:
                segments.append((self._mem_k[:, :, -room:], self._mem_v[:, :, -room:]))
            else:
                segments.append((self._mem_k, self._mem_v))
        if pending is not None:
            segments.append(pending)
        return tuple(segments), self._writes

    def attend(self, h, segments, pos):
        """
        Attention contribution for state `h`, shaped `(B, N)`.

        Returns None when the cache is empty — the first step of a cold state
        has nothing to look back at, and a softmax over zero keys is NaN, not
        zero.

        Segments are joined at the *scores*, not at the keys. One softmax runs
        over the concatenation of `(B, H, g, L_i)` score blocks — which are
        one row per key and therefore cheap to concatenate — and each block's
        probabilities are matched back against their own values. The keys and
        values themselves are never copied, which is the whole reason the
        cache is kept in segments; scores are simply the smaller thing to
        join, and joining them keeps the result a single exact softmax with no
        rescaling step.
        """
        if not segments:
            return None
        with torch.amp.autocast(device_type=h.device.type, enabled=False):
            return self._attend(h.float(), segments, pos)

    def _attend(self, h, segments, pos):
        b = h.shape[0]
        q = self.q_proj(h).view(b, self.kv_heads, self.group, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.rope:
            q = self._apply_rope(q, pos)
        q = q * self.scale

        # Grouped-query attention needs no key repetition: the `group` axis of
        # the query plays the role a query-length axis would, and broadcasting
        # over it is what sharing a KV head means.
        blocks = [torch.matmul(q, k.transpose(-1, -2)) for k, _ in segments]
        scores = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
        weights = self.attn_drop(torch.softmax(scores, dim=-1))

        out = None
        start = 0
        for block, (_, v) in zip(blocks, segments):
            width = block.shape[-1]
            part = torch.matmul(weights.narrow(-1, start, width), v)
            out = part if out is None else out + part
            start += width

        out = self.o_proj(out.reshape(b, self.heads * self.head_dim))
        return out * self.out_scale

    # ------------------------------------------------------------------ #
    # Write                                                               #
    # ------------------------------------------------------------------ #

    def write(self, h):
        """Append one cache entry built from state `h`, shaped `(B, N)`."""
        with torch.amp.autocast(device_type=h.device.type, enabled=False):
            self._write(h.float())
        self._writes += 1

    def _write(self, h):
        b = h.shape[0]
        k = self.k_proj(h).view(b, self.kv_heads, 1, self.head_dim)
        v = self.v_proj(h).view(b, self.kv_heads, 1, self.head_dim)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.rope:
            k = self._apply_rope(k, self._writes)
        k, v = k.contiguous(), v.contiguous()

        if torch.is_grad_enabled():
            self._write_segmented(k, v)
        else:
            self._write_ring(k, v)

    def _write_segmented(self, k, v):
        if self._ring_k is not None:
            # Leaving inference: adopt the ring's contents as the frozen carry.
            # Its order is rotated, which costs nothing — softmax over keys is
            # permutation-invariant and RoPE already sits inside each key.
            if self._ring_fill:
                fill = min(self._ring_fill, self.window)
                self._mem_k = self._ring_k[:, :, :fill].clone()
                self._mem_v = self._ring_v[:, :, :fill].clone()
            self._ring_k = self._ring_v = None
            self._ring_fill = self._ring_cursor = 0
        if self._mem_k is not None and self._mem_k.shape[0] != k.shape[0]:
            self._drop_all()
        self._pend_k.append(k)
        self._pend_v.append(v)
        self._pend_cat = None

    def _write_ring(self, k, v):
        if self._mem_k is not None or self._pend_k:
            # Leaving training: flatten the segmented cache into a ring so the
            # generation loop that follows is allocation-free.
            segments, _ = self.cache_view()
            keys = torch.cat([s[0] for s in segments], dim=2).detach()
            values = torch.cat([s[1] for s in segments], dim=2).detach()
            self._mem_k = self._mem_v = None
            self._pend_k, self._pend_v, self._pend_cat = [], [], None
            self._allocate_ring(keys.shape[0], keys.dtype, keys.device)
            self._seed_ring(keys, values)

        if (self._ring_k is None
                or self._ring_k.shape[0] != k.shape[0]
                or self._ring_k.dtype != k.dtype
                or self._ring_k.device != k.device):
            self._allocate_ring(k.shape[0], k.dtype, k.device)

        # A narrowed view, not index_copy_: an index tensor would mean a
        # host-to-device copy on every single decoded token, which is exactly
        # the per-token allocation a KV cache exists to avoid.
        at = self._ring_cursor
        self._ring_k.narrow(2, at, 1).copy_(k)
        self._ring_v.narrow(2, at, 1).copy_(v)
        self._ring_cursor = (at + 1) % self.window
        self._ring_fill = min(self._ring_fill + 1, self.window)

    def _allocate_ring(self, batch, dtype, device):
        shape = (batch, self.kv_heads, self.window, self.head_dim)
        self._ring_k = torch.zeros(shape, dtype=dtype, device=device)
        self._ring_v = torch.zeros(shape, dtype=dtype, device=device)
        self._ring_fill = 0
        self._ring_cursor = 0

    def _seed_ring(self, keys, values):
        take = min(keys.shape[2], self.window)
        self._ring_k[:, :, :take] = keys[:, :, -take:]
        self._ring_v[:, :, :take] = values[:, :, -take:]
        self._ring_fill = take
        self._ring_cursor = take % self.window

    def _drop_all(self):
        self._mem_k = self._mem_v = None
        self._pend_k, self._pend_v, self._pend_cat = [], [], None
        self._ring_k = self._ring_v = None
        self._ring_fill = self._ring_cursor = 0

    # ------------------------------------------------------------------ #
    # Between forward passes                                              #
    # ------------------------------------------------------------------ #

    def detach_cache(self):
        """
        Fold this call's writes into the frozen carry and cut the graph.

        The mirror of `OdyssNet.detach_state()`: attention over the current
        call is differentiable, everything carried in from earlier calls is a
        constant. Truncation happens here too — `window` bounds what survives
        into the next call.
        """
        if self._ring_k is not None:
            return                    # inference path is already graph-free
        if not self._pend_k and self._mem_k is None:
            return
        segments, _ = self.cache_view()
        if not segments:
            return
        keys = torch.cat([s[0] for s in segments], dim=2).detach()
        values = torch.cat([s[1] for s in segments], dim=2).detach()
        if keys.shape[2] > self.window:
            keys = keys[:, :, -self.window:]
            values = values[:, :, -self.window:]
        self._mem_k = keys.contiguous()
        self._mem_v = values.contiguous()
        self._pend_k, self._pend_v, self._pend_cat = [], [], None

    def drop_rows(self, mask):
        """
        Forget the history of the batch rows selected by `mask` (a bool `(B,)`).

        Zeroing is enough and is why the ring needs no per-row bookkeeping: a
        row of all-zero keys spreads its softmax uniformly over all-zero
        values, so the attention output for that row is exactly zero — the same
        thing an empty cache gives it.
        """
        if not mask.any():
            return
        self.detach_cache()
        sel = mask.view(-1, 1, 1, 1)
        for name in ('_mem_k', '_mem_v', '_ring_k', '_ring_v'):
            buf = getattr(self, name)
            if buf is not None:
                setattr(self, name, buf.masked_fill(sel.to(buf.device), 0.0))

    def snapshot(self):
        """Capture the cache so an evaluation pass cannot disturb training."""
        self.detach_cache()
        return {
            'mem_k': None if self._mem_k is None else self._mem_k.clone(),
            'mem_v': None if self._mem_v is None else self._mem_v.clone(),
            'ring_k': None if self._ring_k is None else self._ring_k.clone(),
            'ring_v': None if self._ring_v is None else self._ring_v.clone(),
            'ring_fill': self._ring_fill,
            'ring_cursor': self._ring_cursor,
            'writes': self._writes,
        }

    def restore(self, snap):
        self._reset_cache_storage()
        self._mem_k = snap['mem_k']
        self._mem_v = snap['mem_v']
        self._ring_k = snap['ring_k']
        self._ring_v = snap['ring_v']
        self._ring_fill = snap['ring_fill']
        self._ring_cursor = snap['ring_cursor']
        self._writes = snap['writes']

    # ------------------------------------------------------------------ #
    # Cost model                                                          #
    # ------------------------------------------------------------------ #

    def cache_bytes(self, batch, element_size=4):
        """Bytes held by a full KV cache at this batch size (inference)."""
        return 2 * batch * self.kv_heads * self.window * self.head_dim * element_size

    def training_cache_bytes(self, batch, writes, element_size=4):
        """
        Bytes of keys and values retained for the backward pass of one call
        that performs `writes` cache writes.

        The frozen carry is a single tensor however many steps read it; the
        growing segment is re-concatenated on every write and each version is
        held until backward, which is the `writes²/2` term. This is the number
        that decides whether a batch fits, so the LLM harness prints it rather
        than making users rediscover it as an OOM.
        """
        per_entry = 2 * batch * self.kv_heads * self.head_dim * element_size
        frozen = self.window * per_entry
        growing = per_entry * writes * (writes + 1) // 2
        return frozen + growing

    def extra_repr(self):
        return (f"neurons={self.num_neurons}, heads={self.heads}, "
                f"kv_heads={self.kv_heads}, head_dim={self.head_dim}, "
                f"window={self.window}, rope={self.rope}, "
                f"qk_norm={self.q_norm is not None}")


