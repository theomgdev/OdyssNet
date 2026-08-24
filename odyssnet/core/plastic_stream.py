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

    def __init__(self, lr, ret, carry, gain, core, gate, batch, dtype):
        # lr/ret: (P, N) row factors per path. carry: (P, N, N), the buffer.
        # gain: the RMSNorm weight. core: W, read only by the novelty gate.
        self.paths, self.n = lr.shape
        self.lr, self.ret, self.carry = lr, ret, carry
        self.gain, self.gate_form = gain, gate
        self.eps = torch.finfo(dtype).eps
        self.retired = 0
        self.live_c = self.live_h = self.live_gate = None

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

        if gate == "row":
            self._gate_setup(core, batch)

    # -- the novelty gate ---------------------------------------------------- #
    #
    # 'row' damps a write by the RMS of the row of `W + cur` it lands on, and
    # that needs three numbers per row, none of which needs the matrix:
    #
    #     N * rms_j^2 = ||W[j,:]||^2 + 2 <W[j,:], cur[j,:]> + ||cur[j,:]||^2
    #
    # With `cur = A * gain / rms(A)` the last two are a gain-weighted inner
    # product against W and a gain-weighted row square of the trace. The gate
    # is detached, so both are carried as running sums updated once per write:
    # reading them costs O(B*N) and the matrix is never assembled.

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
        if self.gate_form != "row":
            return None
        with torch.no_grad():
            lr = self.lr.detach().unsqueeze(1)                       # (P, 1, N)
            cross = (lr * self.x_sum).sum(0)                         # <W, A>
            square = (lr.unsqueeze(0) * lr.unsqueeze(1) * self.g_sum).sum((0, 1))
            row = (self.w_sq + 2.0 * cross / rms + square / (rms * rms)) / self.n
            return 1.0 / (1.0 + row.clamp(min=0).sqrt())

    def _gate_update(self, per_path, hist, written, c_t, pos):
        """Carry the running sums across one write of `c_t (x) written`."""
        if self.gate_form != "row":
            return
        with torch.no_grad():
            ret = self.ret.detach().unsqueeze(1)                     # (P, 1, N)
            h = written.detach()
            hg = h * self.g2

            # <L_p[j,:] * g^2, h>: the one term that reads the history. The
            # carry has decayed `pos` steps by now, the same as everything the
            # running sums hold.
            reach = (torch.einsum('pji,bi->pbj', self.carry, hg) * ret ** pos
                     if not self.cold else torch.zeros_like(c_t))
            diag = self.diag
            if per_path is not None:
                dots = torch.einsum('sbi,bi->bs', hist, hg)
                reach = reach + torch.einsum('psbj,bs->pbj', per_path, dots)
                diag = diag + torch.einsum('psbj,sbj->pbj', per_path, hist)
            # A write never lands on the diagonal, so what would have crossed
            # it comes back out -- of the reach and of the write's own square.
            reach = reach - diag * self.g2 * h
            square = (h * hg).sum(-1, keepdim=True) - self.g2 * h * h

            rp, rq = ret.unsqueeze(1), ret.unsqueeze(0)
            cp, cq = c_t.unsqueeze(1), c_t.unsqueeze(0)
            self.g_sum = (rp * rq * self.g_sum
                          + rp * cq * reach.unsqueeze(1)
                          + rq * cp * reach.unsqueeze(0)
                          + cp * cq * square)
            self.x_sum = ret * self.x_sum + c_t * torch.einsum('ji,bi->bj', self.wg, h)
            self.diag = ret * self.diag

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

    def read(self, h_t):
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
        rms = torch.sqrt(row_sq / self.n + self.eps)
        # The gate reads the same `cur` this step is about to be given, which
        # is why it is taken here and spent by the write that follows.
        self.live_gate = self._gate(rms)
        z = h_t / rms

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
        return (out - z * written_diag) * self.gain

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
        c_t = self.lr.unsqueeze(1) * src / self.n
        if self.live_gate is not None:
            c_t = c_t * self.live_gate
        if self.gate_form == "row":
            if self.retired:
                hist, c_hist, _ = self._history()
                per_path = self._decay(self.retired).unsqueeze(2) * c_hist
            else:
                hist = per_path = None
            self._gate_update(per_path, hist, h_out, c_t.detach(), self.retired)
        self.live_c = c_t
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
