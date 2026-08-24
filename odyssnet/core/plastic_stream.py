"""The plastic trace as a history of writes rather than as a matrix.

The dense trace is `(B, N, N)` and the backward pass holds one per step, so
plasticity costs `steps x B x N^2`. But every write is an outer product, and
what the recurrence asks of the trace is a single vector-matrix product. Write
the trace as the sum it is,

    L_t[j,i] = r_j^t C[j,i] + sum_{s<t} r_j^(t-1-s) c_s[j] h_s[i]

with the diagonal of that sum removed, and both `h_t @ L_t` and the row norms
its RMSNorm needs come out of contractions over the stored `c_s` and `h_s` --
`(B, N)` vectors, never matrices. `C` is the persistent buffer: one matrix for
the whole call rather than one per step.

Exact rather than approximate, under two restrictions. `hebb_gate='none'`,
because the elementwise novelty gate is the one term that does not factor
through an outer product; and `hebb_res` 'global' or 'neuron', because
'synapse' gives every element its own decay and `(h * r^t) @ C` stops being a
matrix product.

`'none'` is not the gate to want: over six seeds on a trained-write recall
probe it collapses to the no-plasticity control on two of them, where `'row'`
-- one damping per presynaptic row, which keeps the write an outer product --
is the steadiest of the three. Carrying `'row'` here needs the same
contractions twice more, a gain-weighted row square and `(W * gain) @ h_s`
cached per write. Until then this path demonstrates the form rather than
replacing the dense one.

Paths share the history: `h_s` is the same vector for temporal and spatial and
only the coefficient `c_s` differs, so summing the per-path coefficients before
the outer product covers `hebb_type='both'`, cross terms included.

Only the newest write carries gradient, exactly as the dense trace detaches its
own history every step; everything older is a constant here too.
"""

import torch


class StreamingTrace:
    """One call's worth of plastic writes, kept as vectors."""

    def __init__(self, lr, ret, carry, dtype):
        # lr/ret: (P, N) row factors per path. carry: (P, N, N), the buffer.
        self.paths, self.n = lr.shape
        self.lr, self.ret, self.carry = lr, ret, carry
        self.eps = torch.finfo(dtype).eps
        self.retired = 0
        self.live_c = self.live_h = None

        # History as a list of writes, each tensor written once and never
        # touched again: a preallocated buffer would be mutated under the
        # graph that read it, which autograd refuses. Stacked once per step so
        # the contractions stay batched.
        self.h_list, self.c_list, self.ch_list = [], [], []

        # A cold buffer skips every carry term. The test costs a device sync
        # and makes the step data-dependent, which `torch.compile` will not
        # like; it stays until the streaming form is measured under compile.
        self.cold = bool(carry.abs().max() == 0)
        # Constants of the call: the carry's row norms (per path pair, so that
        # 'both' keeps its cross term) and its diagonal.
        self.carry_gram = torch.einsum('pji,qji->pqj', carry, carry)
        self.carry_diag = carry.diagonal(dim1=-2, dim2=-1)

    # -- geometry ----------------------------------------------------------- #

    def _pos(self):
        """Steps the trace has decayed through: retired writes plus the live one."""
        return self.retired + (self.live_c is not None)

    def _decay(self, pos):
        """r_j^(pos-1-s) for each retired write s, shape (P, retired, N).

        Only the outermost power is live. The dense trace decays a *detached*
        history once per step, so a gradient reaching the decay logit sees one
        factor and treats the rest as constants; differentiating the whole
        chain here would be a different architecture, not a cheaper one.
        """
        age = torch.arange(pos - 1, pos - 1 - self.retired, -1,
                           device=self.ret.device, dtype=self.ret.dtype)
        return self._power(age.view(1, -1, 1))

    def _power(self, age):
        """r^age with one live factor: r * detach(r)^(age-1), and 1 at age 0."""
        frozen = self.ret.detach().unsqueeze(1) ** (age - 1).clamp(min=0)
        return torch.where(age > 0, self.ret.unsqueeze(1) * frozen,
                           torch.ones_like(frozen))

    def _history(self):
        """The retired writes, stacked: `(S, B, N)` states and `(P, S, B, N)`
        coefficients."""
        return (torch.stack(self.h_list),
                torch.stack(self.c_list, dim=1),
                torch.stack(self.ch_list, dim=1) if self.ch_list else None)

    def _coefficients(self, pos, c_hist):
        """Row weight of each retired write in `sum_p lr_p L_p`, `(S, B, N)`.

        The application-side factor is live and the accumulation-side factor
        was detached with the write, which is where the dense path's
        `local.detach()` lands.
        """
        weight = (self.lr.unsqueeze(1) * self._decay(pos)).unsqueeze(2)
        return (weight * c_hist).sum(0)

    # -- reading ------------------------------------------------------------ #

    def read(self, h_t, gain):
        """`h_t @ RMSNorm_rows(sum_p lr_p L_p) * gain`.

        The same vector the dense path gets out of `bmm(h_t, cur_hebb_W)`.
        """
        pos = self._pos()
        if self.retired:
            hist, c_hist, ch_hist = self._history()
            coef = self._coefficients(pos, c_hist)          # (S, B, N)
        else:
            hist = ch_hist = coef = None
        weight = self.lr * self._power(torch.tensor(
            float(pos), device=self.ret.device)).squeeze(1)   # (P, N), the carry
        zero = torch.zeros(h_t.shape[0], self.n, device=h_t.device, dtype=h_t.dtype)

        # --- rows of the unmasked A = carry + writes ------------------------ #
        if self.cold:
            row_sq, row_diag = zero, zero
            carry_diag = zero
        else:
            row_sq = torch.einsum('pj,qj,pqj->j', weight, weight,
                                  self.carry_gram).expand_as(zero)
            carry_diag = (weight * self.carry_diag).sum(0).expand_as(zero)
            row_diag = carry_diag

        if coef is not None:
            gram = torch.einsum('sbi,ubi->bsu', hist, hist)
            row_sq = row_sq + torch.einsum('sbj,ubj,bsu->bj', coef, coef, gram)
            if not self.cold:
                row_sq = row_sq + 2.0 * torch.einsum(
                    'sbj,psbj,pj->bj', coef, ch_hist, weight)
            row_diag = row_diag + torch.einsum('sbj,sbj->bj', coef, hist)

        if self.live_c is not None:
            live_w = (self.lr.unsqueeze(1) * self.live_c).sum(0)          # (B, N)
            reach = self._reach(coef, hist, weight, self.live_h)          # A_old @ h
            row_sq = (row_sq + 2.0 * live_w * reach
                      + live_w * live_w * (self.live_h ** 2).sum(-1, keepdim=True))
            row_diag = row_diag + live_w * self.live_h

        # Writes never touch the diagonal; the carry keeps whatever it holds.
        written_diag = row_diag - carry_diag
        row_sq = row_sq - row_diag ** 2 + carry_diag ** 2
        z = h_t / torch.sqrt(row_sq / self.n + self.eps)

        # --- z @ A --------------------------------------------------------- #
        out = zero
        if not self.cold:
            out = torch.einsum('bj,pj,pji->bi', z, weight, self.carry)
        if coef is not None:
            out = out + torch.einsum('bs,sbi->bi',
                                     torch.einsum('bj,sbj->bs', z, coef), hist)
        if self.live_c is not None:
            live_w = (self.lr.unsqueeze(1) * self.live_c).sum(0)
            out = out + (z * live_w).sum(-1, keepdim=True) * self.live_h
        return (out - z * written_diag) * gain

    def _reach(self, coef, hist, weight, vec):
        """`A_old[j,:] @ vec` on the unmasked trace, for the row-norm cross term."""
        total = torch.zeros_like(vec)
        if not self.cold:
            total = torch.einsum('pj,pji,bi->bj', weight, self.carry, vec)
        if coef is not None:
            total = total + torch.einsum('sbj,bs->bj', coef,
                                         torch.einsum('sbi,bi->bs', hist, vec))
        return total

    # -- writing ------------------------------------------------------------ #

    def write(self, src, h_out):
        """Append one write: `lr * src / N` paired with `h_out`.

        `src` is `(P, B, N)` -- temporal pairs the previous state with the
        current one and spatial the current state with itself, so the two paths
        differ here and nowhere else.
        """
        if self.live_c is not None:
            self._retire()
        self.live_c = self.lr.unsqueeze(1) * src / self.n
        self.live_h = h_out

    def _retire(self):
        h = self.live_h.detach()
        self.h_list.append(h)
        self.c_list.append(self.live_c.detach())
        if not self.cold:
            self.ch_list.append(torch.einsum('pji,bi->pbj', self.carry, h))
        self.retired += 1
        self.live_c = self.live_h = None

    def finish(self):
        """The per-path trace `(P, N, N)`, averaged over the batch: what the
        persistent buffer keeps."""
        if self.live_c is not None:
            self._retire()
        out = self.carry * (self.ret ** self.retired).unsqueeze(-1)   # buffer: no graph
        if self.retired:
            hist, c_hist, _ = self._history()
            coef = self._decay(self.retired).unsqueeze(2) * c_hist
            written = torch.einsum('psbj,sbi->pji', coef, hist) / hist.shape[1]
            out = out + written - torch.diag_embed(written.diagonal(dim1=-2, dim2=-1))
        return out
