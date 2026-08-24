"""The plastic trace as the writes it is made of, rather than as a matrix.

Held as a matrix the trace is `(B, N, N)` and the backward pass keeps one per
step, which costs `steps x B x N^2`. But every Hebbian write is an outer
product and the recurrence only ever asks the trace for `h @ L`, so it can be
kept as the sum it is,

    L_t[j,i] = r_j^t C[j,i] + sum_{s<t} r_j^(t-1-s) c_s[j] h_s[i]

with the diagonal of that sum removed. Both `h @ L` and the row norms its
RMSNorm needs come out of contractions over the stored `c_s` and `h_s` --
`(B, N)` vectors, never matrices -- and `C`, the persistent buffer, is one
matrix for the whole call rather than one per step.

Three properties make it affordable rather than merely smaller:

* The history lives in a `(steps, B, N)` buffer written once per step, and
  every contraction runs over all of it behind a mask. Every step therefore has
  the same shapes, which is what `torch.compile` needs, and the buffer is the
  whole memory cost: `steps x B x N`.
* The read is recomputed in the backward pass rather than kept, so none of it
  survives the step. What it recomputes from is immutable once written, so the
  recompute sees exactly what the forward pass saw.
* The novelty gate damps a write by the RMS of the row of `W + cur` it lands
  on, and is detached, so it is carried as running sums updated once per write
  rather than read back out of the trace: `O(B * N)` and no graph.

Paths share the history -- `h_s` is the same vector for temporal and spatial
and only the coefficient `c_s` differs -- so summing the per-path coefficients
before the outer product covers `hebb_type='both'`, cross terms included.

Only the newest write carries gradient: the trace's own history is truncated,
which is what makes the correlation affordable to differentiate. The decay
logit sees one live factor for the same reason.
"""

import torch
import torch.utils.checkpoint as checkpoint


class PlasticTrace:
    """One call's worth of plastic writes, kept as vectors."""

    def __init__(self, lr, ret, carry, gain, core, steps, batch, dtype):
        # lr/ret: (P, N) row factors per path. carry: (P, N, N), the buffer.
        # gain: the RMSNorm weight. core: W, read by the novelty gate.
        self.paths, self.n = lr.shape
        self.lr, self.ret, self.carry, self.gain = lr, ret, carry, gain
        self.eps = torch.finfo(dtype).eps
        self.retired = 0
        self.live_c = self.live_h = self.live_gate = None

        opts = dict(device=lr.device, dtype=dtype)
        self.h_hist = torch.zeros(steps, batch, self.n, **opts)
        self.c_hist = torch.zeros(self.paths, steps, batch, self.n, **opts)
        # C @ h_s, cached when a write retires: the carry reaches the row norms
        # through it, and it is the same vector at every later step.
        self.ch_hist = torch.zeros(self.paths, steps, batch, self.n, **opts)
        self.index = torch.arange(steps, **opts)

        # Constants of the call: the carry's row norms, per path pair so that
        # 'both' keeps its cross term, and its diagonal.
        self.carry_gram = torch.einsum('pji,qji->pqj', carry, carry)
        self.carry_diag = carry.diagonal(dim1=-2, dim2=-1)
        self._gate_setup(core, batch)

    # -- the novelty gate ---------------------------------------------------- #
    #
    #     N * rms_j^2 = ||W[j,:]||^2 + 2 <W[j,:], cur[j,:]> + ||cur[j,:]||^2
    #
    # and with `cur = A * gain / rms(A)` the last two are a gain-weighted inner
    # product against W and a gain-weighted row square of the trace. Neither
    # needs the matrix, and both survive as running sums.

    def _gate_setup(self, core, batch):
        with torch.no_grad():
            g = self.gain.detach()
            self.wg = core * g                          # W scaled by the gain
            self.g2 = g * g
            self.w_sq = (core * core).sum(-1)           # ||W[j,:]||^2, (N,)
            wide = lambda t: t.unsqueeze(-2).expand(*t.shape[:-1], batch,
                                                    self.n).clone()
            # The trace starts at the carry, and so do the running sums.
            self.x_sum = wide(torch.einsum('ji,pji->pj', self.wg, self.carry))
            self.g_sum = wide(torch.einsum('pji,qji,i->pqj',
                                           self.carry, self.carry, self.g2))
            self.diag = wide(self.carry_diag)           # L_p[j,j]

    def _gate(self, rms):
        """`1 / (1 + rms_i(W[j,:] + cur[j,:]))`, detached, `(B, N)`."""
        with torch.no_grad():
            lr = self.lr.detach().unsqueeze(1)                       # (P, 1, N)
            cross = (lr * self.x_sum).sum(0)                         # <W, A>
            square = (lr.unsqueeze(0) * lr.unsqueeze(1) * self.g_sum).sum((0, 1))
            row = (self.w_sq + 2.0 * cross / rms + square / (rms * rms)) / self.n
            return 1.0 / (1.0 + row.clamp(min=0).sqrt())

    def _gate_update(self, written, c_t, pos):
        """Carry the running sums across one write of `c_t (x) written`."""
        with torch.no_grad():
            ret = self.ret.detach().unsqueeze(1)                     # (P, 1, N)
            h = written.detach()
            hg = h * self.g2
            decay = self._weights(self.ret.detach(), pos, pos, live=False)

            # <L_p[j,:] * g^2, h>: the one term here that reads the history.
            reach = torch.einsum('pji,bi->pbj', self.carry, hg) * ret ** pos
            dots = torch.einsum('sbi,bi->bs', self.h_hist, hg)
            reach = reach + torch.einsum('psbj,psj,bs->pbj',
                                         self.c_hist, decay, dots)
            diag = self.diag + torch.einsum('psbj,psj,sbj->pbj',
                                            self.c_hist, decay, self.h_hist)
            # A write never lands on the diagonal, so what would have crossed
            # it comes back out -- of the reach, and of the write's own square.
            reach = reach - diag * self.g2 * h
            square = (h * hg).sum(-1, keepdim=True) - self.g2 * h * h

            rp, rq = ret.unsqueeze(1), ret.unsqueeze(0)
            cp, cq = c_t.unsqueeze(1), c_t.unsqueeze(0)
            self.g_sum = (rp * rq * self.g_sum
                          + rp * cq * reach.unsqueeze(1)
                          + rq * cp * reach.unsqueeze(0)
                          + cp * cq * square)
            self.x_sum = ret * self.x_sum + c_t * torch.einsum('ji,bi->bj',
                                                               self.wg, h)
            self.diag = ret * self.diag

    # -- geometry ------------------------------------------------------------ #

    def _weights(self, ret, pos, kept, live=True):
        """`r_j^(pos-1-s)` for the first `kept` slots, zero after; `(P, steps, N)`.

        `ret` is passed rather than read off the trace: the read is recomputed
        under a checkpoint, and a factor reached through the closure would not
        carry a gradient back out of it.

        `kept` is the number of *retired* writes, which is not `pos` while a
        write is still live: the live one is passed separately, and counting it
        here as well would add it twice on replay, once the slot it later took
        has been filled.

        Only the outermost power is live: the trace decays a detached history
        once per step, so a gradient reaching the decay logit sees one factor
        and the rest are constants. Slots past `kept` are masked rather than
        trimmed, which keeps every step the same shape.
        """
        age = (pos - 1 - self.index).clamp(min=0).view(1, -1, 1)
        mask = (self.index < kept).to(ret.dtype).view(1, -1, 1)
        ret = ret.unsqueeze(1)
        if not live:
            return ret.detach() ** age * mask
        frozen = ret.detach() ** (age - 1).clamp(min=0)
        return torch.where(age > 0, ret * frozen, torch.ones_like(frozen)) * mask

    def _carry_weight(self, ret, pos):
        """`r^pos` on the carry, one live factor for the same reason."""
        if pos == 0:
            return torch.ones_like(ret)
        return ret * ret.detach() ** (pos - 1)

    # -- reading ------------------------------------------------------------- #

    def read(self, h_t):
        """`h_t @ RMSNorm_rows(sum_p lr_p L_p) * gain`, and the gate for the
        write that follows -- both from the same `cur` the step will see."""
        pos = self.retired + (self.live_c is not None)
        live_c = (self.live_c if self.live_c is not None
                  else h_t.new_zeros(self.paths, *h_t.shape))
        live_h = self.live_h if self.live_h is not None else torch.zeros_like(h_t)
        args = (h_t, live_c, live_h, self.lr, self.ret, self.gain, pos,
                self.retired)
        if torch.is_grad_enabled() and any(
                t.requires_grad for t in args[:6] if torch.is_tensor(t)):
            out, rms = checkpoint.checkpoint(self._read, *args, use_reentrant=False)
        else:
            out, rms = self._read(*args)
        self.live_gate = self._gate(rms.detach())
        return out

    def _read(self, h_t, live_c, live_h, lr, ret, gain, pos, kept):
        weights = self._weights(ret, pos, kept)                      # (P, S, N)
        coef = ((lr.unsqueeze(1) * weights).unsqueeze(2) * self.c_hist).sum(0)
        carry_w = lr * self._carry_weight(ret, pos)                  # (P, N)
        live_w = (lr.unsqueeze(1) * live_c).sum(0)                   # (B, N)

        # --- rows of A = carry + writes, before the diagonal is taken out ---- #
        gram = torch.einsum('sbi,ubi->bsu', self.h_hist, self.h_hist)
        row_sq = torch.einsum('sbj,ubj,bsu->bj', coef, coef, gram)
        row_diag = torch.einsum('sbj,sbj->bj', coef, self.h_hist)
        reach = (torch.einsum('sbj,bs->bj', coef,
                              torch.einsum('sbi,bi->bs', self.h_hist, live_h))
                 + torch.einsum('pj,pji,bi->bj', carry_w, self.carry, live_h))
        row_sq = (row_sq
                  + torch.einsum('pj,qj,pqj->j', carry_w, carry_w, self.carry_gram)
                  + 2.0 * torch.einsum('sbj,psbj,pj->bj', coef, self.ch_hist,
                                       carry_w))
        carry_diag = (carry_w * self.carry_diag).sum(0).expand_as(row_diag)
        row_diag = row_diag + carry_diag
        row_sq = (row_sq + 2.0 * live_w * reach
                  + live_w * live_w * (live_h * live_h).sum(-1, keepdim=True))
        row_diag = row_diag + live_w * live_h

        # Writes never land on the diagonal; the carry keeps whatever it holds.
        written_diag = row_diag - carry_diag
        row_sq = row_sq - row_diag ** 2 + carry_diag ** 2
        rms = torch.sqrt(row_sq / self.n + self.eps)

        # --- z @ A ----------------------------------------------------------- #
        z = h_t / rms
        out = (torch.einsum('bs,sbi->bi',
                            torch.einsum('bj,sbj->bs', z, coef), self.h_hist)
               + torch.einsum('bj,pj,pji->bi', z, carry_w, self.carry)
               + (z * live_w).sum(-1, keepdim=True) * live_h)
        return (out - z * written_diag) * gain, rms

    # -- writing ------------------------------------------------------------- #

    def write(self, src, h_out):
        """Append one write: `lr * gate * src / N` paired with `h_out`.

        `src` is `(P, B, N)` -- temporal pairs the previous state with the
        current one and spatial the current state with itself, so the two paths
        differ here and nowhere else.
        """
        if self.live_c is not None:
            self._retire()
        c_t = self.lr.unsqueeze(1) * src * self.live_gate / self.n
        self._gate_update(h_out, c_t.detach(), self.retired)
        self.live_c, self.live_h = c_t, h_out

    def _retire(self):
        # Written functionally rather than in place: the read is recomputed
        # under a checkpoint, and a buffer mutated after the graph read it is
        # something autograd refuses and Inductor refuses harder.
        with torch.no_grad():
            h = self.live_h.detach()
            slot = (self.index == self.retired).view(-1, 1, 1)
            self.h_hist = torch.where(slot, h.unsqueeze(0), self.h_hist)
            self.c_hist = torch.where(slot, self.live_c.detach().unsqueeze(1),
                                      self.c_hist)
            self.ch_hist = torch.where(
                slot, torch.einsum('pji,bi->pbj', self.carry, h).unsqueeze(1),
                self.ch_hist)
        self.retired += 1
        self.live_c = self.live_h = None

    def finish(self):
        """The per-path trace `(P, N, N)`, averaged over the batch: what the
        persistent buffer keeps."""
        if self.live_c is not None:
            self._retire()
        with torch.no_grad():
            coef = self._weights(self.ret, self.retired, self.retired,
                                 live=False).unsqueeze(2) * self.c_hist
            written = torch.einsum('psbj,sbi->pji', coef,
                                   self.h_hist) / self.h_hist.shape[1]
            out = (self.carry * (self.ret ** self.retired).unsqueeze(-1)
                   + written
                   - torch.diag_embed(written.diagonal(dim1=-2, dim2=-1)))
        return out
