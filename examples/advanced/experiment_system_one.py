"""
OdyssNet-SystemOne — typed decisions whose parameters do not know how many
questions or options are coming.

Every classifier reserves an output column per class. That is the ceiling: an
option the model never saw in training is not *hard* to answer, it is
unaddressable, and widening the schema means retraining. A System One layer is
supposed to take a state, take whatever typed questions a caller has, and
return calibrated answers — so the schema cannot live in the weights.

The mapping is native, not bolted on. Context, question and option are all
bytes entering the same N x N core behind one-byte segment markers, and every
answer is read as ONE scalar from a single-row decoder:

    [S] context          -> h_ctx     stored, read once per example
    [Q] question         -> h_qry     from h_ctx, stored, read once
    [O] option 1         -> scalar    from h_qry
    [O] option 2         -> scalar    from h_qry, rewound
    ...
    [Q] next question    -> h_qry'    from h_ctx, rewound

Two levels of stored state, two levels of rewind. `forward` returns `h_t`
inside the graph while only `self.state` is detached, so a stored state handed
back as `current_state` still carries gradient — the caching is a training
mechanism and not an inference trick. `TemporalAttention.mark()/rewind()` does
the same for the KV cache, without which option k+1 attends to what option k
wrote and a score depends on the order it was asked in.

What follows from the shape, and what `--mode smoke` checks:

* K is runtime data. A 2-option question and a 150-option question run through
  the same weights, and an option never seen in training is scored by reading
  its text.
* A question costs a segment, not a retrain. There is no schema to widen.
* The stored state is `neurons` floats whatever the context's length; a KV
  cache grows with it.

Out-of-scope and confidence are not heads here. Per-option BCE makes the scalar
absolute rather than a rank, so a question whose answer is none of the options
reads as a flat low distribution; confidence is that distribution's own max and
margin. A rejection option is just another option in the data.

Usage
-----
    python -u experiment_system_one.py --mode smoke
    python -u experiment_system_one.py --mode train --tag base --minutes 20
    python -u experiment_system_one.py --mode train --tag base --resume
    python -u experiment_system_one.py --mode sweep --sweep think --minutes 4
    python -u experiment_system_one.py --mode eval --tag base
    python -u experiment_system_one.py --mode ask --tag base \
        --context "cancel my flight to denver" \
        --question "what does the user want to do?" \
        --option "cancel a reservation" --option "book a flight"

Data
----
JSONL, one object per context, in the vocabulary of the task rather than of any
one dataset:

    {"context": "cancel my flight to denver",
     "questions": [
       {"q": "what does the user want to do?",
        "options": ["cancel a reservation", "book a flight", "none of these"],
        "correct": 0},
       {"q": "can this assistant handle that?",
        "options": ["yes", "no"], "correct": 0}]}

`correct: -1` means no option is right: the absolute signal with no ranking
term. Options are text, so shape them however the dataset needs — nothing in
this file knows what a dataset looks like, and a converter belongs next to the
data rather than in the training path.

`docs/LIBRARY.md` carries what the sweeps measured.
"""

import sys

# Keep emoji-rich console output from crashing legacy Windows code pages.
# line_buffering=True is not optional: reconfigure() rebuilds the TextIOWrapper
# and would otherwise discard `python -u`'s write-through, leaving a long run's
# progress invisible until the process exits.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace",
                            line_buffering=True)

import argparse
import glob
import json
import os
import random
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from odyssnet import (OdyssNet, OdyssNetTrainer, load_checkpoint,
                      save_checkpoint, set_seed)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "ckpt")
DATA_DIR = os.path.join(HERE, "..", "..", "data", "decisions")

# Segment markers. Ids 0-3 are reserved so a marker is never ambiguous with a
# text byte; text is shifted up by OFFSET.
#: -1 is the core's "inject nothing" sentinel, so a padded row reads exactly
#: as the short row it was padded from and a score never depends on what else
#: shares its batch.
PAD_ID = -1
SEG_CONTEXT, SEG_QUESTION, SEG_OPTION = 0, 1, 2
OFFSET = 3
VOCAB = 256 + OFFSET


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Cfg:
    # --- data ---
    data: str = os.path.join(DATA_DIR, "train.jsonl")
    val: str = os.path.join(DATA_DIR, "val.jsonl")
    chunk: int = 16                 # bytes per forward; state carries
                                    # across pieces, so this bounds the
                                    # tensor, never the context length

    # --- architecture ---
    neurons: int = 192
    n_in: int = 96                  # neurons the byte embedding reaches
    n_out: int = 96                 # neurons the scalar decoder reads
    think: int = 8                  # echo steps before each readout
    activation: tuple = ("none", "gelu_tanh", "tanh")
    weight_init: tuple = ("quiet", "resonant", "quiet", "zero")
    gates: tuple = ("none", "none", "identity")
    hebb_type: str = ""             # "" = off
    hebb_res: str = "neuron"
    dropout: float = 0.0

    # --- attention ---
    attn_heads: int = 0             # 0 = no attention at all
    attn_kv_heads: int = 1
    attn_window: int = 256
    attn_rope: bool = True
    attn_qk_norm: bool = True

    # --- optimization ---
    batch: int = 24                 # contexts per step
    # None = ChaosGrad's online estimate (default, zero-config); float = fixed-rate mode.
    lr: float | None = None
    grad_ckpt: bool = False

    # --- run control ---
    minutes: float = 0.0            # 0 = until Ctrl-C
    max_steps: int = 0              # 0 = unlimited
    eval_every: int = 500
    eval_contexts: int = 300        # 0 = the whole validation file
    log_every: int = 100
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tag: str = "base"

    def io_ids(self):
        needed = self.n_in + self.n_out
        if needed > self.neurons:
            raise ValueError(
                f"n_in + n_out = {needed} exceeds neurons = {self.neurons}")
        return (list(range(self.n_in)),
                list(range(self.n_in, self.n_in + self.n_out)))


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Read:
    """One (context, question, options, correct) tuple."""
    __slots__ = ("context", "question", "options", "correct")
    context: str
    question: str
    options: list
    correct: int                    # index into options, or -1 for "none"


def load_reads(path, default_leaf="train.jsonl"):
    """
    Parse the decision format, refusing anything ambiguous.

    Errors name the file and line: a malformed corpus should cost a message,
    not a training run that silently learns from three usable rows.

    Strings are interned as they are read, because a corpus is mostly repeated
    text — one context is carried by every question under it, and an option
    set is drawn from a pool far smaller than the file. Interning plus slots
    is what decides whether a multi-million-question corpus fits in memory.
    """
    if os.path.isdir(path):
        candidates = [default_leaf, "train.jsonl", "val.jsonl"] if default_leaf else ["train.jsonl", "val.jsonl"]
        for leaf in candidates:
            if leaf and os.path.exists(os.path.join(path, leaf)):
                path = os.path.join(path, leaf)
                break

    if not os.path.exists(path):
        raise SystemExit(
            f"\n✋ No data at {path}.\n"
            f"   --data / --val take JSONL (or a directory containing train.jsonl / val.jsonl):\n"
            f'     {{"context": "...", "questions": [\n'
            f'        {{"q": "...", "options": ["a", "b"], "correct": 0}}]}}\n'
            f"   correct: -1 means no option is right.\n")

    reads = []
    pool = {}
    def keep(text):
        return pool.setdefault(text, text)

    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"\n✋ {path}:{n}: {e}\n") from e
            ctx, questions = row.get("context"), row.get("questions")
            if not isinstance(ctx, str) or not isinstance(questions, list):
                raise SystemExit(
                    f"\n✋ {path}:{n}: needs 'context' (string) and "
                    f"'questions' (list).\n")
            ctx = keep(ctx)
            for q in questions:
                options = q.get("options")
                if not isinstance(options, list) or len(options) < 2:
                    raise SystemExit(
                        f"\n✋ {path}:{n}: a question needs at least two "
                        f"options.\n")
                correct = int(q.get("correct", -1))
                if correct >= len(options):
                    raise SystemExit(
                        f"\n✋ {path}:{n}: correct={correct} is out of range "
                        f"for {len(options)} options.\n")
                reads.append(Read(ctx, keep(str(q.get("q", ""))),
                                  [keep(str(o)) for o in options], correct))
    if not reads:
        raise SystemExit(f"\n✋ {path}: no usable rows.\n")
    return reads


def group_by_context(reads):
    """
    [(context, [Read, ...]), ...] — the unit the context cache amortises.

    Grouping is what makes the stored state worth anything: every question
    under one context shares a single context read.
    """
    out, index = [], {}
    for r in reads:
        if r.context not in index:
            index[r.context] = len(out)
            out.append((r.context, []))
        out[index[r.context]][1].append(r)
    return out


def load_corpus(cfg, verbose=True):
    """
    (train_groups, val_groups), reported as the shape they really are.

    `--data` and `--val` take several comma-separated paths, so a final corpus
    is assembled at the command line rather than by concatenating files on
    disk: the parts stay separately regenerable, and a mix can be changed
    without rewriting gigabytes.
    """
    def resolve(spec, leaf):
        """Comma-separated paths, with a directory standing for its `leaf`."""
        out = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            candidate = os.path.join(part, leaf)
            out.append(candidate if os.path.isdir(part) else part)
        return out

    train_paths = resolve(cfg.data, "train.jsonl")
    # An unset --val follows --data: pointing --data at data/wordnet should
    # validate on data/wordnet/val.jsonl, not on the default corpus.
    val_paths = (resolve(cfg.data, "val.jsonl") if cfg.val == Cfg.val
                 and os.path.isdir(cfg.data.split(",")[0].strip())
                 else resolve(cfg.val, "val.jsonl"))

    def read_all(paths, leaf):
        reads = []
        for path in paths:
            reads += load_reads(path, leaf)
        return group_by_context(reads)

    train = read_all(train_paths, "train.jsonl")
    val = read_all(val_paths, "val.jsonl")
    if verbose:
        for label, paths, groups in (("train", train_paths, train),
                                     ("val", val_paths, val)):
            reads = [r for _, rs in groups for r in rs]
            k = [len(r.options) for r in reads]
            none = sum(1 for r in reads if r.correct < 0)
            print(f"📚 {label}: {len(groups):,} contexts | {len(reads):,} "
                  f"questions | K {min(k)}-{max(k)} (mean {sum(k)/len(k):.1f}) "
                  f"| {none:,} answered by none of the options"
                  + (f" | {len(paths)} files" if len(paths) > 1 else ""))
    return train, val


def encode(text, marker):
    return [marker] + [b + OFFSET for b in text.encode("utf-8")]


def pad_to(rows, width):
    out = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def pack(groups, cfg):
    """
    (context, questions, options, mask, correct) as device tensors.

    `questions` is (B, Q, L_q) and `options` is (B, Q, K, L_o), padded to the
    batch maximum with `mask` carrying the real counts — so a batch mixing a
    2-option and a 150-option question is one tensor and nothing in the model
    or the loss ever learns a fixed geometry.

    Each of the three axes is padded to the longest entry *in this batch*, not
    to a configured width: segments are chained through `current_state`, so a
    three-byte option has no reason to run a long segment's worth of steps.
    """
    b = len(groups)
    q_max = max(len(rs) for _, rs in groups)
    k_max = max(len(r.options) for _, rs in groups for r in rs)

    ctx_rows = [encode(c, SEG_CONTEXT) for c, _ in groups]
    q_rows = {}
    o_rows = {}
    correct = np.full((b, q_max), -1, dtype=np.int64)
    mask = np.zeros((b, q_max, k_max), dtype=bool)

    for i, (_, rs) in enumerate(groups):
        for j, r in enumerate(rs):
            q_rows[(i, j)] = encode(r.question, SEG_QUESTION)
            for k, opt in enumerate(r.options):
                o_rows[(i, j, k)] = encode(opt, SEG_OPTION)
            mask[i, j, : len(r.options)] = True
            correct[i, j] = r.correct

    up = lambda n: max(cfg.chunk, -(-n // cfg.chunk) * cfg.chunk)  # noqa: E731
    l_q = up(max(len(v) for v in q_rows.values()))
    l_o = up(max(len(v) for v in o_rows.values()))
    l_c = up(max(len(r) for r in ctx_rows))

    ctx = pad_to(ctx_rows, l_c)
    questions = np.full((b, q_max, l_q), PAD_ID, dtype=np.int64)
    options = np.full((b, q_max, k_max, l_o), PAD_ID, dtype=np.int64)
    for (i, j), row in q_rows.items():
        questions[i, j, : len(row)] = row
    for (i, j, k), row in o_rows.items():
        options[i, j, k, : len(row)] = row

    to = lambda a: torch.from_numpy(a).to(cfg.device)  # noqa: E731
    return (to(ctx), to(questions), to(options), to(mask), to(correct))


def widths(group, chunk):
    """
    The step geometry a group would be read at, as a bucket key.

    Lengths enter as a count of chunks rather than as bytes: two contexts of
    91 and 96 bytes both run six pieces of 16, so they belong in one batch and
    a key on exact length would have split them into two.

    Question and option *counts* are deliberately not part of the key. They
    pad along their own axes and the step count does not depend on them, so
    keying on them only narrows the buckets without making a row's scores any
    more independent of what shares its batch.
    """
    ctx, reads = group
    up = lambda n: -(-(n + 1) // chunk)  # noqa: E731  (+1 for the marker)
    return (up(len(ctx.encode("utf-8"))),
            max(up(len(r.question.encode("utf-8"))) for r in reads),
            max(up(len(o.encode("utf-8"))) for r in reads for o in r.options))


def bucket_by_width(groups, size, chunk):
    """
    Groups split into batches whose rows share a step geometry.

    Rows of one batch run the same number of steps, so a batch mixing
    geometries would pay its longest row's steps for every row and its scores
    would stop being reproducible across `--batch`.
    """
    buckets = {}
    for g in groups:
        buckets.setdefault(widths(g, chunk), []).append(g)
    return cut(buckets, size)


def cut(buckets, size):
    """{key: [group, ...]} flattened into batches of at most `size`."""
    out = []
    for rows in buckets.values():
        out += [rows[s: s + size] for s in range(0, len(rows), size)]
    return out


class Batches:
    """
    Shuffled epochs over the context groups, grouped by exact segment width.

    A batch is read at one width because the core spends a step per input
    column, and the attention cache is indexed by batch row — so splitting a
    read by row length is not available and padding a short row would make its
    score depend on the longest row beside it. Grouping by the widths the rows
    actually have removes the pad instead of hiding it.

    Each epoch re-cuts the batches rather than reordering a fixed set, so a
    long run sees new combinations of rows instead of the same few hundred
    batches in a new order — on a large corpus the fixed cut is most of the
    stochasticity a step would otherwise have.
    """

    def __init__(self, groups, cfg):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.pools = {}
        for g in groups:
            self.pools.setdefault(widths(g, cfg.chunk), []).append(g)
        self.epochs = 0
        self.queue = []
        self._refill()

    def _refill(self):
        for rows in self.pools.values():
            self.rng.shuffle(rows)
        self.queue = cut(self.pools, self.cfg.batch)
        self.rng.shuffle(self.queue)

    def next(self):
        if not self.queue:
            self.epochs += 1
            self._refill()
        return pack(self.queue.pop(), self.cfg)


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

def build(cfg):
    input_ids, output_ids = cfg.io_ids()
    model = OdyssNet(
        num_neurons=cfg.neurons,
        input_ids=input_ids,
        output_ids=output_ids,
        device=cfg.device,
        # Bytes arrive one per step and the answer is read after the last one,
        # so pulse injection is the only sane choice: holding a byte across its
        # echo steps would drown the state it is supposed to perturb.
        pulse_mode=True,
        # One output column. The decoder cannot encode "which class", only
        # "how well does what I just read fit what I was asked" — which is the
        # whole reason the option count never reaches the parameters.
        vocab_size=(VOCAB, 1),
        vocab_mode="hybrid",
        # Identity on the encoder/decoder keeps that scalar an unbounded logit.
        activation=list(cfg.activation),
        weight_init=list(cfg.weight_init),
        gate=list(cfg.gates),
        hebb_type=cfg.hebb_type or None,
        hebb_res=cfg.hebb_res,
        attn_heads=cfg.attn_heads or None,
        attn_kv_heads=cfg.attn_kv_heads,
        attn_window=cfg.attn_window,
        attn_rope=cfg.attn_rope,
        attn_qk_norm=cfg.attn_qk_norm,
        dropout_rate=cfg.dropout,
        gradient_checkpointing=cfg.grad_ckpt,
    )
    trainer = OdyssNetTrainer(model, lr=cfg.lr, device=cfg.device)
    # The loss is assembled here, over a variable number of options, so the
    # trainer's criterion has to pass the scalar through rather than square it
    # against a zero target.
    trainer.loss_fn = lambda pred, _target: pred.mean()
    return model, trainer


def describe(cfg, model):
    """Every setting that decides what a run means, before it starts."""
    print(f"\n🧠 {model.get_num_params():,} trainable params | {cfg.neurons} "
          f"neurons | in {cfg.n_in} / out {cfg.n_out} | think {cfg.think} | "
          f"batch {cfg.batch} | lr {'auto' if cfg.lr is None else cfg.lr}")
    print(f"   readout 1 scalar | gates {','.join(cfg.gates)}"
          + (f" | hebb {cfg.hebb_type}/{cfg.hebb_res}" if cfg.hebb_type else "")
          + (f" | attn {cfg.attn_heads}h/{cfg.attn_kv_heads}kv "
             f"w{cfg.attn_window}" if cfg.attn_heads else ""))


# --------------------------------------------------------------------------- #
# Reads: two levels of stored state, two levels of rewind                     #
# --------------------------------------------------------------------------- #

def _mark(model):
    return None if model.attn is None else model.attn.mark()


def _rewind(model, mark):
    if model.attn is not None and mark is not None:
        model.attn.rewind(mark)


def read_segment(model, ids, cfg, state=None, return_sequence=False):
    """
    One segment plus its echo steps; returns (scalar_or_seq, end state).

    The segment is read in `cfg.chunk`-wide pieces with the state carried
    between them, which is what makes a context of any length cost steps in
    proportion to its bytes and nothing else. Every piece is exactly one chunk
    wide: `pack` pads to a chunk multiple and `bucket_by_width` puts rows of
    equal chunk count in one batch, so the step count a row runs depends on
    nothing but its own length — pad bytes inject nothing, and a score cannot
    move with what else shares the batch.
    """
    total = ids.shape[1]
    h = state
    seq_outs = [] if return_sequence else None
    for start in range(0, total, cfg.chunk):
        piece = ids[:, start: start + cfg.chunk]
        last = start + cfg.chunk >= total
        out, h = model(piece,
                       steps=piece.shape[1] + (cfg.think if last else 0),
                       current_state=h, return_sequence=return_sequence)
        if return_sequence:
            seq_outs.append(out)
    if return_sequence:
        return torch.cat(seq_outs, dim=1).squeeze(-1), h
    return out[:, -1, 0], h


def smooth_trajectory_score(step_scores, active_lengths, power=2.0):
    """
    Smoothly integrates option trajectory step scores into a final decision scalar.

    Weights scale smoothly as tau^power where tau = (t+1)/T, eliminating early
    prefix ambiguity without sharp boundary discontinuities.
    """
    b, q_n, k_n, t_max = step_scores.shape
    device = step_scores.device
    t_idx = torch.arange(1, t_max + 1, device=device, dtype=step_scores.dtype).view(1, 1, 1, t_max)
    lens = active_lengths.unsqueeze(-1).clamp(min=1).to(step_scores.dtype)

    tau = (t_idx / lens).clamp(max=1.0)
    weights = tau.pow(power)
    mask = t_idx <= lens
    weights = weights * mask

    weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (step_scores * (weights / weight_sum)).sum(dim=-1)


def _branch(model, h, times):
    """
    Replicate a stored state into `times` continuations, interleaved.

    Branching is a batch operation here: every option reads the same question
    state, so they go down the batch axis in one forward instead of a Python
    loop of K forwards. The attention cache is indexed by batch row and has to
    be widened the same way. Hebbian state is `(N, N)` and shared, so it needs
    nothing.
    """
    if model.attn is not None:
        model.attn.repeat_rows(times)
    return h.repeat_interleave(times, dim=0)


def score_all(model, cfg, ctx_ids, q_ids, opt_ids, mask, return_trajectories=False):
    """
    Every option of every question, from a single context read.

    (B, L), (B, Q, L), (B, Q, K, L) -> (B, Q, K). Masked options come back at
    -1e4, so a softmax over the last axis ignores them.

    The cost is 3 segment reads whatever Q and K are: the context once, all Q
    questions together as one `(B*Q, N)` branch of the context state, and all
    Q*K options together as one `(B*Q*K, N)` branch of the question states.
    Options and questions still cannot see each other — each branch starts
    from the same stored state and writes into its own row — but the GPU gets
    three wide matmuls instead of 1 + Q + Q*K narrow ones.
    """
    b, q_n, k_n, _ = opt_ids.shape
    model.reset_state(batch_size=b)
    if model.attn is not None:
        model.attn.reset()

    _, h_ctx = read_segment(model, ctx_ids, cfg)
    ctx_mark = _mark(model)

    # Every question branches off the context state at once.
    h_wide = _branch(model, h_ctx, q_n)
    _, h_qry = read_segment(model, q_ids.reshape(b * q_n, -1), cfg,
                            state=h_wide)

    # And every option branches off its own question's state at once. The
    # (B, Q, K) layout survives because both branches are interleaved: row
    # (i*Q + j) is example i's question j, and (i*Q + j)*K + k is its option k.
    o_wide = _branch(model, h_qry, k_n)
    opt_flat = opt_ids.reshape(b * q_n * k_n, -1)
    step_scores, _ = read_segment(model, opt_flat, cfg,
                                  state=o_wide, return_sequence=True)

    _rewind(model, ctx_mark)
    if model.attn is not None:
        model.attn.select_first_of(q_n * k_n)

    # Trajectory shape: (B, Q, K, TotalSteps)
    total_steps = step_scores.shape[1]
    step_scores = step_scores.view(b, q_n, k_n, total_steps)
    active_lens = (opt_ids != PAD_ID).sum(dim=-1)

    s = smooth_trajectory_score(step_scores, active_lens, power=2.0)
    scores = s.masked_fill(~mask, -1e4)
    if return_trajectories:
        return scores, step_scores, active_lens
    return scores


def decision_loss(scores, correct, mask):
    """
    Two terms on the same scalars.

    absolute — per option, "is this the right one". This is what makes the
               scalar mean something on its own, and therefore what lets an
               all-low distribution read as "none of these" without a
               dedicated head to drown in the class imbalance one would face.
    rank     — softmax across a question's options, "which is best".

    A question with `correct < 0` contributes only the absolute term, which is
    the only true thing to say about it.

    The absolute term weights its positive class by negatives/positives: with
    K options only one is right, and a corpus where some questions are
    answered by none of them pushes that rate lower still, so the unweighted
    term is minimised by driving every score down together. The counts are
    offset by one so a batch whose every question is a rejection — no positive
    to balance — stays finite instead of needing a floor.
    """
    target = torch.zeros_like(scores)
    has = correct >= 0
    if has.any():
        i, j = has.nonzero(as_tuple=True)
        target[i, j, correct[has]] = 1.0

    pos = target[mask].sum()
    neg = mask.sum() - pos
    absolute = F.binary_cross_entropy_with_logits(
        scores[mask], target[mask], pos_weight=(neg + 1) / (pos + 1))
    rank = (F.cross_entropy(scores[has], correct[has]) if has.any()
            else scores.new_zeros(()))
    return absolute + rank, {"abs": float(absolute), "rank": float(rank)}


def train_step(trainer, model, cfg, batch):
    """One optimizer step over a batch of contexts."""
    ctx, questions, options, mask, correct = batch
    terms = {}

    def transform(_out):
        # The trainer owns the step for its optimizer bookkeeping, but this
        # protocol needs many forwards per step, so the real work happens here
        # and the trainer's criterion passes the scalar through.
        total, parts = decision_loss(
            score_all(model, cfg, ctx, questions, options, mask),
            correct, mask)
        terms.update(parts)
        return total.reshape(1, 1, 1)

    trainer.train_batch(ctx, torch.zeros(1, device=cfg.device),
                        thinking_steps=ctx.shape[1] + cfg.think,
                        output_transform=transform)
    return terms


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

def _ece(prob, hit, bins=15):
    """Expected calibration error: does a stated probability hold up."""
    if not len(prob):
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    return float(sum(
        (idx == b).mean() * abs(prob[idx == b].mean() - hit[idx == b].mean())
        for b in range(bins) if (idx == b).any()))


class Validator:
    """
    Fixed held-out evaluation over the same contexts for every arm, so numbers
    from different configurations are directly comparable.
    """

    def __init__(self, groups, cfg):
        self.cfg = cfg
        if cfg.eval_contexts and cfg.eval_contexts < len(groups):
            # A deterministic sample, not the first N: a corpus is written
            # grouped by whatever produced it, so the head of the file is one
            # question type and a prefix would score that type rather than the
            # validation set. The seed is fixed so every arm scores the same
            # contexts.
            groups = random.Random(0).sample(groups, cfg.eval_contexts)
        # Batches of one width, as in training: a mixed-width batch pays its
        # longest row's steps for every row, and that is also what would make
        # the score move with --batch.
        self.batches = bucket_by_width(groups, cfg.batch, cfg.chunk)
        reads = [r for b in self.batches for _, rs in b for r in rs]
        self.contexts = len(groups)
        self.answerable = sum(1 for r in reads if r.correct >= 0)
        self.rejectable = len(reads) - self.answerable

    @torch.no_grad()
    def run(self, model):
        was_training = model.training
        model.eval()
        # TF32 off for the same reason AMP would be: the score is what arms are
        # ranked by, so it must not move with kernel choice.
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return self._score(model)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            if was_training:
                model.train()

    def _score(self, model):
        cfg = self.cfg
        hit = n = rej_hit = rej_n = 0
        conf, ok = [], []
        abs_sum = rank_sum = loss_n = 0

        for chunk in self.batches:
            ctx, questions, options, mask, correct = pack(chunk, cfg)
            scores = score_all(model, cfg, ctx, questions, options, mask)

            # Val loss: same criterion as training, no gradient.
            _, parts = decision_loss(scores, correct, mask)
            abs_sum += parts["abs"]
            rank_sum += parts["rank"]
            loss_n += 1

            prob = torch.softmax(scores, dim=2)
            pred = prob.argmax(dim=2)
            top = prob.max(dim=2).values

            for i, (_, rs) in enumerate(chunk):
                for j in range(len(rs)):
                    c = int(correct[i, j])
                    if c >= 0:
                        n += 1
                        good = int(pred[i, j]) == c
                        hit += int(good)
                        conf.append(float(top[i, j]))
                        ok.append(float(good))
                    else:
                        # Nothing is right, so a low top score is the answer.
                        rej_n += 1
                        rej_hit += int(float(top[i, j]) < 0.5)

        m = {"acc": hit / n if n else float("nan"),
             "abs": abs_sum / max(loss_n, 1),
             "rank": rank_sum / max(loss_n, 1),
             "questions": n + rej_n}
        if rej_n:
            m["reject"] = rej_hit / rej_n
        if conf:
            c, o = np.asarray(conf), np.asarray(ok)
            m["ece"] = _ece(c, o)
            m["brier"] = float(np.mean((c - o) ** 2))
            m["conf"] = float(c.mean())
        return m


def fmt(m):
    bits = [f"acc {m['acc']*100:6.2f}%"]
    if "abs" in m:
        bits.append(f"abs {m['abs']:.4f}")
    if "rank" in m:
        bits.append(f"rank {m['rank']:.4f}")
    if "reject" in m:
        bits.append(f"reject {m['reject']*100:5.1f}%")
    if "ece" in m:
        bits.append(f"ece {m['ece']:.3f}")
    return " | ".join(bits)


# --------------------------------------------------------------------------- #
# ask                                                                         #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ask(model, cfg, context, questions, return_trajectories=False):
    """
    `questions` is [(text, [option, ...]), ...]; the context is read once.

    This is the caller's view of the protocol, and the reason the context cache
    exists: a second question costs its own segment, not another reading of the
    state it is asked against.
    """
    was_training = model.training
    model.eval()
    groups = [(context, [Read(context, q, list(o), -1) for q, o in questions])]
    ctx, q_ids, opt_ids, mask, _ = pack(groups, cfg)
    with torch.no_grad():
        if return_trajectories:
            scores_batch, step_scores, active_lens = score_all(
                model, cfg, ctx, q_ids, opt_ids, mask, return_trajectories=True)
            scores = scores_batch[0]
        else:
            scores = score_all(model, cfg, ctx, q_ids, opt_ids, mask)[0]
    if was_training:
        model.train()

    answers = []
    for j, (q, options) in enumerate(questions):
        s = scores[j, : len(options)]
        p = torch.softmax(s, dim=0)
        order = torch.argsort(p, descending=True)
        ans = {
            "question": q,
            "ranked": [(options[int(i)], float(p[int(i)])) for i in order],
            "absolute": {options[int(i)]: float(torch.sigmoid(s[int(i)]))
                         for i in order},
            "confidence": float(p.max()),
            "margin": (float(p[order[0]] - p[order[1]])
                       if len(options) > 1 else 1.0),
        }
        if return_trajectories:
            ans["trajectories"] = {
                options[int(i)]: step_scores[0, j, int(i), :int(active_lens[0, j, int(i)])].cpu().tolist()
                for i in range(len(options))
            }
        answers.append(ans)
    return answers


def print_answers(answers):
    for a in answers:
        print(f"\n  Q: {a['question']}")
        for name, p in a["ranked"][:8]:
            print(f"     {p*100:6.2f}%  (abs {a['absolute'][name]:.2f})  "
                  f"{name}")
        print(f"     ── confidence {a['confidence']:.2f} | "
              f"margin {a['margin']:.2f}")
        if max(a["absolute"].values()) < 0.3:
            print("     ── reads as none of these")


def plot_trajectories(context, answers):
    """
    Renders character-by-character score trajectories for each option.
    Honors ODYSSNET_DISABLE_PLOT=1 to stay non-blocking in automated test runs.
    """
    if os.environ.get("ODYSSNET_DISABLE_PLOT") == "1":
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not installed; skipping trajectory plot)")
        return

    n_q = len(answers)
    fig, axes = plt.subplots(n_q, 1, figsize=(10, max(4 * n_q, 4)), squeeze=False)

    for ax, a in zip(axes[:, 0], answers):
        q = a["question"]
        ranked = a["ranked"]
        top_opt = ranked[0][0]
        trajs = a.get("trajectories", {})

        for name, prob in ranked:
            traj = trajs.get(name, [])
            if not traj:
                continue
            is_top = (name == top_opt)
            lw = 2.5 if is_top else 1.2
            alpha = 1.0 if is_top else 0.6
            label = f"{prob*100:5.1f}%  {name}"
            marker = "o" if len(traj) <= 32 else None
            color = ax._get_lines.get_next_color()
            ax.plot(range(len(traj)), traj, label=label, linewidth=lw,
                    alpha=alpha, marker=marker, markersize=3, color=color)
            # Dashed horizontal mean line for each option — shows where the
            # trajectory settled on average, making it easy to spot the
            # "eureka" inflection points against the baseline.
            mean_s = sum(traj) / len(traj)
            ax.axhline(mean_s, linestyle="--", linewidth=0.9,
                       alpha=0.5, color=color)

        ax.set_title(f"Q: {q}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Option Token / Character Step (t)", fontsize=9)
        ax.set_ylabel("Instantaneous Score s(t)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8)

    ctx_preview = context if len(context) <= 70 else context[:67] + "..."
    fig.suptitle(f"OdyssNet-SystemOne — Decision Trajectory\nContext: \"{ctx_preview}\"",
                 fontsize=12)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Checkpoints                                                                 #
# --------------------------------------------------------------------------- #

#: Everything that defines how a checkpoint was built and scored; adopted by
#: eval/ask, whose job is to reproduce it.
ARCH_FIELDS = ("neurons", "n_in", "n_out", "think", "chunk", "activation",
               "weight_init", "gates", "hebb_type", "hebb_res", "attn_heads",
               "attn_kv_heads", "attn_window", "attn_rope", "attn_qk_norm")

#: The subset `--resume` adopts: what `build()` allocates. `think` and
#: `think` is a forward-pass argument and every accepted activation is
#: parameter-free, so pinning those would only take away a knob — but `gates`
#: adds or removes four gate parameters and the attention geometry sizes its
#: projections, so both are adopted.
RESUME_FIELDS = ("neurons", "n_in", "n_out", "gates", "hebb_type", "hebb_res",
                 "attn_heads", "attn_kv_heads", "attn_qk_norm")


def ckpt_paths(cfg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    stem = os.path.join(CKPT_DIR, f"s1_odyss_{cfg.tag}")
    return stem + "_latest.pth", stem + "_best.pth"


def guard_overwrite(cfg, overwrite=False):
    """
    Refuse to start a fresh run on top of an existing tag's checkpoints.

    Without this, `--mode train --tag base` with no `--resume` rebuilds from
    scratch with best = -inf, so the first validation counts as a record and
    overwrites both files — a long run destroyed seconds in, with nothing in
    the log to say so. A hard exit rather than a prompt, so the script stays
    usable from another script.
    """
    latest, best = ckpt_paths(cfg)
    existing = [p for p in (latest, best) if os.path.exists(p)]
    if not existing or overwrite:
        return
    raise SystemExit(
        f"\n✋ Refusing to overwrite existing checkpoints for tag "
        f"'{cfg.tag}':\n"
        + "".join(f"     {p}\n" for p in existing)
        + f"\n   Continue that run:   --mode train --tag {cfg.tag} --resume\n"
        f"   Start somewhere new: --mode train --tag {cfg.tag}_b\n"
        f"   Overwrite anyway:    --mode train --tag {cfg.tag} --overwrite\n")


def adopt_saved_arch(cfg, path, fields):
    """
    Override architecture fields with the ones a checkpoint was built from.

    `fields` is required rather than defaulted: resume and eval want different
    sets, and a third caller inheriting the broader one would over-adopt —
    failing in the direction that quietly removes a knob.
    """
    if not os.path.exists(path):
        return cfg
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # save_checkpoint merges extra_data into the top level rather than nesting.
    saved = payload.get("cfg") or {}
    changed = {}
    for f in fields:
        if f not in saved:
            continue
        want = tuple(saved[f]) if isinstance(getattr(cfg, f), tuple) \
            else saved[f]
        if want != getattr(cfg, f):
            changed[f] = want
    if changed:
        print(f"🔧 adopting checkpoint architecture: "
              f"{', '.join(f'{k}={v}' for k, v in changed.items())}")
    return replace(cfg, **changed)


def _save(path, model, trainer, cfg, step, metrics, best):
    payload = asdict(cfg)
    for f in ("activation", "weight_init", "gates"):
        payload[f] = list(payload[f])
    save_checkpoint(model, trainer.optimizer, step,
                    metrics.get("acc", float("nan")), path,
                    extra_data={"cfg": payload, "step": step,
                                "best_acc": best, "metrics": metrics},
                    trainer_state=trainer.state_dict())


# --------------------------------------------------------------------------- #
# Training session                                                            #
# --------------------------------------------------------------------------- #

def run_session(cfg, corpus, budget_sec=0.0, resume=False, resume_best=False,
                quiet=False, save=True):
    """
    Train under a wall-clock and/or step budget. Returns the final metrics.

    Used by `--mode train` (open-ended) and by every sweep arm (budgeted), so
    both paths exercise exactly the same code.
    """
    train_groups, val_groups = corpus
    set_seed(cfg.seed)

    model, trainer = build(cfg)
    validator = Validator(val_groups, cfg)
    batches = Batches(train_groups, cfg)
    latest_path, best_path = ckpt_paths(cfg)

    step, best = 0, -float("inf")
    # `_latest` is rewritten at every evaluation and `_best` only on an
    # improvement, so the two diverge exactly when it matters: a collapse
    # overwrites `_latest` within one eval interval, while `_best` structurally
    # cannot hold a collapsed model.
    resume_path = best_path if resume_best else latest_path
    flag = "--resume-best" if resume_best else "--resume"
    if resume and not os.path.exists(resume_path):
        if resume_best and os.path.exists(latest_path):
            raise SystemExit(
                f"\n✋ {flag}: no best checkpoint at {resume_path}, but tag "
                f"'{cfg.tag}' has a latest one.\n"
                f"   Starting fresh would overwrite it at the first "
                f"evaluation, so this run stops instead.\n"
                f"   Continue from latest:  --mode train --tag {cfg.tag} "
                f"--resume\n")
        print(f"ℹ️  {flag} given but no checkpoint at {resume_path}; "
              f"starting fresh.")
    if resume and os.path.exists(resume_path):
        # Loading must not fail softly: --resume bypasses guard_overwrite, so a
        # run that continues past a failed load trains a random model and saves
        # it over the checkpoint it could not read.
        try:
            info = load_checkpoint(model, trainer.optimizer, resume_path,
                                   device=cfg.device, strict=True,
                                   lr=cfg.lr, trainer=trainer)
        except Exception as e:
            raise SystemExit(
                f"\n✋ Could not load the checkpoint for tag '{cfg.tag}':\n"
                f"     {e}\n\n"
                f"   The file is intact; this run refuses to overwrite it "
                f"with a fresh model.\n"
                f"   Train elsewhere:   --mode train --tag {cfg.tag}_b\n"
                f"   Start over anyway: --mode train --tag {cfg.tag} "
                f"--overwrite (destroys it)\n") from e
        step = int(info.get("step", 0))
        best = float(info.get("best_acc", -float("inf")))
        print(f"📂 resumed {os.path.basename(resume_path)} at step {step:,} "
              f"(best acc {best:.2%})")

    if not quiet:
        describe(cfg, model)
        print(f"   scoring {validator.contexts:,} contexts | "
              f"{validator.answerable:,} answerable + "
              f"{validator.rejectable:,} rejectable questions\n")

    metrics = {"acc": float("nan")}
    window = {"abs": 0.0, "rank": 0.0, "n": 0}
    started = time.time()
    interrupted = False
    start_step = step

    try:
        while True:
            if budget_sec and time.time() - started >= budget_sec:
                break
            if cfg.max_steps and step - start_step >= cfg.max_steps:
                break

            terms = train_step(trainer, model, cfg, batches.next())
            step += 1
            window["abs"] += terms["abs"]
            window["rank"] += terms["rank"]
            window["n"] += 1

            if not quiet and cfg.log_every and step % cfg.log_every == 0:
                k = max(window["n"], 1)
                elapsed = time.time() - started
                print(f"step {step:>7,} | abs {window['abs']/k:6.4f} | "
                      f"rank {window['rank']/k:6.4f} | "
                      f"lr {trainer._current_lr():.2e} | "
                      f"epoch {batches.epochs} | {elapsed/60:5.1f}m")
                window = {"abs": 0.0, "rank": 0.0, "n": 0}

            if cfg.eval_every and step % cfg.eval_every == 0:
                metrics = validator.run(model)
                mark = ""
                if metrics["acc"] > best:
                    best, mark = metrics["acc"], "  🏆"
                    if save:
                        _save(best_path, model, trainer, cfg, step, metrics,
                              best)
                if not quiet:
                    print(f"   ↳ VAL  {fmt(metrics)}{mark}")
                if save:
                    _save(latest_path, model, trainer, cfg, step, metrics,
                          best)
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏹️  Interrupted — finishing cleanly.")

    elapsed = max(time.time() - started, 1e-6)
    if step > start_step:
        metrics = validator.run(model)
        if save:
            if metrics["acc"] > best:
                best = metrics["acc"]
                _save(best_path, model, trainer, cfg, step, metrics, best)
            _save(latest_path, model, trainer, cfg, step, metrics, best)

    metrics.update({
        "steps": step - start_step,
        "minutes": elapsed / 60.0,
        "steps_s": (step - start_step) / elapsed,
        "params": model.get_num_params(),
        "epochs": batches.epochs,
        "interrupted": interrupted,
    })
    return metrics, model, trainer


# --------------------------------------------------------------------------- #
# Sweeps                                                                      #
# --------------------------------------------------------------------------- #

SWEEPS = {
    # The thesis test: echo steps are depth bought with no new parameters, and
    # the readout happens after them. Every arm gets the same wall clock, so
    # more thinking means fewer optimizer steps.
    "think": [
        ("think0", dict(think=0)),
        ("think4", dict(think=4)),
        ("think8", dict(think=8)),
        ("think16", dict(think=16)),
    ],
    # Does the parameter count buy accuracy, and where does it stop?
    "size": [
        ("n96", dict(neurons=96, n_in=48, n_out=48)),
        ("n192", dict(neurons=192, n_in=96, n_out=96)),
        ("n384", dict(neurons=384, n_in=192, n_out=192)),
    ],
    # Scoring an option against a context is a comparison, and both of these
    # are comparison mechanisms — attention over the states the context wrote,
    # or a plastic trace the option queries.
    "mech": [
        ("plain", dict()),
        ("attn4", dict(attn_heads=4)),
        ("hebb", dict(hebb_type="temporal")),
        ("gated", dict(gates=("none", "sigmoid", "identity"))),
    ],
    # ChaosGrad's zero-config estimate against pinned rates. Supervision here
    # is one signal per option rather than one per token, which is not the
    # regime the estimator was tuned in.
    "lr": [
        ("auto", dict(lr=None)),
        ("3e-3", dict(lr=3e-3)),
        ("1e-3", dict(lr=1e-3)),
        ("3e-4", dict(lr=3e-4)),
    ],
}


def run_sweep(cfg, corpus, name, minutes, arms=None):
    plan = SWEEPS[name]
    if arms:
        wanted = {a.strip() for a in arms.split(",")}
        plan = [p for p in plan if p[0] in wanted]
        if not plan:
            raise SystemExit(f"No arms named {sorted(wanted)} in '{name}'")

    budget = minutes * 60.0
    print(f"\n{'='*78}")
    print(f"🔬 SWEEP '{name}' — {len(plan)} arms x {minutes:.1f} min "
          f"(compute-matched, seed {cfg.seed})")
    print(f"{'='*78}")

    results = []
    for i, (arm, over) in enumerate(plan, 1):
        # eval_every=0: mid-run validation would spend budget, and each arm a
        # different amount of it. Every arm is scored once, after the clock.
        arm_cfg = replace(cfg, tag=f"sweep_{name}_{arm}", eval_every=0, **over)
        print(f"\n[{i}/{len(plan)}] ▶ {arm}: {over or 'baseline'}")
        try:
            m, model, _ = run_session(arm_cfg, corpus, budget_sec=budget,
                                      quiet=True, save=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print("   ✗ OOM — arm skipped")
            _empty_cache(cfg.device)
            continue
        m["arm"] = arm
        results.append(m)
        print(f"   {fmt(m)} | {m['steps']:,} steps | {m['epochs']} epochs | "
              f"{m['params']:,} params")
        del model
        _empty_cache(cfg.device)

    if not results:
        return results

    results.sort(key=lambda r: -r["acc"])
    print(f"\n{'='*78}")
    print(f"🏁 SWEEP '{name}' RESULTS — ranked by accuracy")
    print(f"{'='*78}")
    print(f"{'arm':<12} {'acc':>8} {'reject':>8} {'ece':>7} {'steps':>8} "
          f"{'params':>10}")
    print("-" * 78)
    for r in results:
        print(f"{r['arm']:<12} {r['acc']*100:>7.2f}% "
              f"{r.get('reject', float('nan'))*100:>7.1f}% "
              f"{r.get('ece', float('nan')):>7.3f} {r['steps']:>8,} "
              f"{r['params']:>10,}")
    print("-" * 78)
    print(f"🥇 {results[0]['arm']}")

    out = os.path.join(CKPT_DIR, f"sweep_{name}_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📄 {out}")
    return results


def _empty_cache(device):
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def mean_loss(model, cfg, groups):
    """Loss over fixed batches with no training — the learning-check baseline."""
    was_training = model.training
    model.eval()
    total = n = 0
    for start in range(0, len(groups), cfg.batch):
        ctx, q, o, mask, correct = pack(groups[start:start + cfg.batch], cfg)
        t, _ = decision_loss(score_all(model, cfg, ctx, q, o, mask), correct,
                             mask)
        total += float(t)
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


def run_smoke(cfg, corpus):
    """
    Fast end-to-end check of every path the heavy run depends on, plus the
    three properties that make this a decision layer rather than a classifier.
    Seconds, not hours — this is what makes the expensive script safe to edit.

    Runs against the real corpus, cut to a handful of contexts: a built-in
    sample would be a dataset living in the training path, and it would stop
    exercising the loader that every real run depends on.
    """
    print(f"\n{'='*78}")
    print("🔥 SMOKE TEST")
    print(f"{'='*78}")

    # Clear our own litter first: the round-trip below trains with save=True
    # and no --resume, so leftovers would trip guard_overwrite and fail the
    # test on its second invocation. Running smoke twice has to be a no-op.
    for stale in glob.glob(os.path.join(CKPT_DIR, "s1_odyss_smoke*.pth")):
        os.remove(stale)
    os.makedirs(CKPT_DIR, exist_ok=True)

    base = replace(cfg, neurons=96, n_in=48, n_out=48, think=2, batch=3,
                   max_steps=6, minutes=0.0, eval_every=0, log_every=0,
                   eval_contexts=0, tag="smoke")
    # Six contexts is enough to exercise every path and keeps the whole test
    # in seconds.
    groups = corpus[0][:6]
    corpus = (groups, groups)
    failures = []

    def check(name, ok, detail=""):
        print(f"{'✅' if ok else '❌'} {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    variants = [
        ("plain", dict()),
        ("attn", dict(attn_heads=4)),
        ("hebbian", dict(hebb_type="both")),
        ("gated", dict(gates=("none", "sigmoid", "sigmoid"))),
        ("grad_ckpt", dict(grad_ckpt=True)),
        ("think0", dict(think=0)),
    ]
    print()
    for name, over in variants:
        c = replace(base, **over)
        t0 = time.time()
        try:
            m, model, trainer = run_session(c, corpus, quiet=True, save=False)
            assert m["steps"] == c.max_steps, \
                f"ran {m['steps']} of {c.max_steps} steps"
            assert np.isfinite(m["acc"]), "non-finite accuracy"
            diag = trainer.get_diagnostics()
            check(f"variant {name:<10}", True,
                  f"{m['params']:>7,} params | {fmt(m)} | "
                  f"lr {diag['current_lr']:.1e} | {time.time()-t0:4.1f}s")
            del model
        except Exception as e:                       # noqa: BLE001
            check(f"variant {name:<10}", False, f"{type(e).__name__}: {e}")

    # 1. The property the design exists for: the option count never reaches
    #    the parameters.
    print()
    counts, shapes = {}, {}
    try:
        for k in (2, 25, 150):
            options = [f"option number {i}" for i in range(k)]
            g = [("some context",
                  [Read("some context", "which one", options, 0)])]
            model, _ = build(base)
            counts[k] = model.get_num_params()
            model.eval()
            with torch.no_grad():
                shapes[k] = tuple(
                    score_all(model, base, *pack(g, base)[:4]).shape)
            del model
        check("option count is not a parameter",
              len(set(counts.values())) == 1,
              f"K=2/25/150 all {counts[2]:,} params, scores "
              f"{shapes[2]}/{shapes[25]}/{shapes[150]}")
    except Exception as e:                           # noqa: BLE001
        check("option count is not a parameter", False,
              f"{type(e).__name__}: {e}")

    # 2. Branches must be blind to each other, or a score depends on the order
    #    it was asked in. This is what mark()/rewind() buys, and the bug it
    #    prevents is silent.
    for attn in (0, 4):
        label = "plain" if not attn else "attn"
        try:
            c = replace(base, attn_heads=attn)
            model, _ = build(c)
            model.eval()
            options = ["cancel a reservation", "book a flight", "weather"]
            fwd = [("cancel my flight",
                    [Read("cancel my flight", "what now", options, 0)])]
            rev = [("cancel my flight",
                    [Read("cancel my flight", "what now", options[::-1], 2)])]
            with torch.no_grad():
                a = score_all(model, c, *pack(fwd, c)[:4])[0, 0]
                b = score_all(model, c, *pack(rev, c)[:4])[0, 0]
            d = (a - b.flip(0)).abs().max().item()
            check(f"options blind to each other ({label})", d < 1e-4,
                  f"max Δ {d:.2e} between forward and reversed")
            del model
        except Exception as e:                       # noqa: BLE001
            check(f"options blind to each other ({label})", False,
                  f"{type(e).__name__}: {e}")

    # 3. A score must not move with what else shares the batch, or every
    #    number a run reports depends on --batch. `bucket_by_width` is what
    #    buys this, so the check goes through it rather than around it.
    try:
        c = replace(base, attn_heads=0)
        model, _ = build(c)
        model.eval()
        scores = {}
        for b in (1, 2, len(groups)):
            got = []
            for rows in bucket_by_width(groups, b, c.chunk):
                ctx, q, o, mk, _ = pack(rows, c)
                with torch.no_grad():
                    s = score_all(model, c, ctx, q, o, mk)
                # Masked slots come back at -1e4 and their count depends on
                # the batch, so only the real options are comparable.
                got += sorted(float(x) for x in s[mk])
            scores[b] = sorted(got)
        # Tolerance rather than equality: a batched matmul reorders its
        # reductions, so identical inputs land a few ULPs apart. What must not
        # happen is a row running a different number of steps, which moves the
        # score by 1e-2 or more.
        worst = max(abs(a - b) for ref in (scores[1],)
                    for other in (scores[2], scores[len(groups)])
                    for a, b in zip(ref, other))
        check("a score is blind to its batch", worst < 1e-4,
              f"max Δ {worst:.1e} over batch 1/2/{len(groups)}")
        del model
    except Exception as e:                           # noqa: BLE001
        check("a score is blind to its batch", False,
              f"{type(e).__name__}: {e}")

    # 5. Branching down the batch axis has to give what the sequential loop
    #    gave, under every state-carrying mechanism — attention's cache is
    #    indexed by batch row and Hebbian state is shared, so this is where a
    #    wrong replication would show up. The tolerance is float32 reality:
    #    BLAS picks a different reduction order per batch width, and in
    #    float64 the same comparison closes to 6e-14, so anything at 1e-4 or
    #    above is a real divergence rather than arithmetic.
    def _sequential(model, cfg, ctx, q, o, mk):
        bb, qq, kk, _ = o.shape
        model.reset_state(batch_size=bb)
        if model.attn is not None:
            model.attn.reset()
        _, h_c = read_segment(model, ctx, cfg)
        cm = _mark(model)
        out = []
        for j in range(qq):
            _rewind(model, cm)
            _, h_q = read_segment(model, q[:, j], cfg, state=h_c)
            qm = _mark(model)
            cols = []
            for k in range(kk):
                _rewind(model, qm)
                step_s, _ = read_segment(model, o[:, j, k], cfg, state=h_q, return_sequence=True)
                act_lens = (o[:, j, k] != PAD_ID).sum(dim=-1).view(bb, 1, 1)
                s = smooth_trajectory_score(step_s.view(bb, 1, 1, -1), act_lens, power=2.0).squeeze(1).squeeze(1)
                cols.append(s)
            out.append(torch.stack(cols, dim=1))
        _rewind(model, cm)
        return torch.stack(out, dim=1).masked_fill(~mk, -1e4)

    for attn, hebb in ((0, None), (4, None), (0, "temporal"), (4, "temporal")):
        label = f"attn={attn or 0} hebb={hebb or 'none'}"
        try:
            c = replace(base, attn_heads=attn, hebb_type=hebb)
            model, _ = build(c)
            model.eval()
            batch = pack(bucket_by_width(groups, c.batch, c.chunk)[0], c)
            with torch.no_grad():
                want = _sequential(model, c, *batch[:4])
                got = score_all(model, c, *batch[:4])
            d = (want[batch[3]] - got[batch[3]]).abs().max().item()
            check(f"batched options match the loop ({label})", d < 1e-4,
                  f"max Δ {d:.1e} (float32 reduction order; 6e-14 in float64)")
            del model
        except Exception as e:                       # noqa: BLE001
            check(f"batched options match the loop ({label})", False,
                  f"{type(e).__name__}: {e}")

    # 6. The caching is real: 1 context + Q questions + Q option reads, not
    #    Q*K*3. Counting forward calls is the only way to assert it.
    try:
        c = replace(base)
        model, _ = build(c)
        model.eval()
        calls = {"n": 0}
        real = model.forward

        def counted(*a, **kw):
            calls["n"] += 1
            return real(*a, **kw)

        model.forward = counted
        reads = [Read("ctx", "q1", ["a", "b"], 0),
                 Read("ctx", "q2", ["a", "b"], 1),
                 Read("ctx", "q3", ["a", "b"], 0)]
        with torch.no_grad():
            score_all(model, c, *pack([("ctx", reads)], c)[:4])
        # One context read, one batched read for every question, and one
        # batched read for every option of every question — three, whatever
        # Q and K are.
        expect = 3
        naive = sum(len(r.options) for r in reads) * 3
        check("context and question read once", calls["n"] == expect,
              f"{calls['n']} segment reads (cached {expect}, uncached would "
              f"be {naive})")
        model.forward = real
        del model
    except Exception as e:                           # noqa: BLE001
        check("context and question read once", False,
              f"{type(e).__name__}: {e}")

    # Gradient has to survive both rewinds, or the cache is an inference trick.
    try:
        c = replace(base)
        model, trainer = build(c)
        ctx, q, o, mask, correct = pack(groups[:3], c)
        total, _ = decision_loss(score_all(model, c, ctx, q, o, mask),
                                 correct, mask)
        trainer.optimizer.zero_grad(set_to_none=True)
        total.backward()
        named = {"W": model.W, "embed": model.embed.weight,
                 "decoder": model.output_decoder.weight}
        dead = [k for k, p in named.items()
                if p.grad is None or p.grad.abs().max() == 0]
        check("gradient reaches the core through both rewinds", not dead,
              ", ".join(f"{k} {p.grad.abs().max():.1e}"
                        for k, p in named.items()) if not dead
              else f"no gradient into {', '.join(dead)}")
        del model
    except Exception as e:                           # noqa: BLE001
        check("gradient reaches the core through both rewinds", False,
              f"{type(e).__name__}: {e}")

    # An option set the model never trained on must still be scoreable — the
    # thing a reserved-column classifier cannot do at all.
    try:
        c = replace(base)
        _, model, _ = run_session(c, corpus, quiet=True, save=False)
        novel = [("pay my electricity bill",
                  [Read("pay my electricity bill", "what does the user want?",
                        ["pay a bill", "rent a car", "none of these"], 0)])]
        r = Validator(novel, replace(c, eval_contexts=0)).run(model)
        check("unseen option set scores", np.isfinite(r["acc"]),
              f"{fmt(r)} on options absent from training")
        del model
    except Exception as e:                           # noqa: BLE001
        check("unseen option set scores", False, f"{type(e).__name__}: {e}")

    # Checkpoint round-trip, once per architecture that changes the state dict.
    print()
    for label, over in (("plain", dict()), ("attn", dict(attn_heads=4))):
        try:
            c = replace(base, tag=f"smoke_ckpt_{label}", eval_every=0, **over)
            _, model1, _ = run_session(c, corpus, quiet=True, save=True)
            validator = Validator(groups, replace(c, eval_contexts=0))
            a1 = validator.run(model1)["acc"]
            del model1

            # The adoption path is what makes a bare `--tag ... --resume`
            # safe: the CLI defaults are full size, the checkpoint is not.
            c2 = adopt_saved_arch(replace(cfg, tag=c.tag), ckpt_paths(c)[0],
                                  fields=ARCH_FIELDS)
            model2, trainer2 = build(c2)
            load_checkpoint(model2, trainer2.optimizer, ckpt_paths(c)[0],
                            device=c2.device, strict=True)
            a2 = validator.run(model2)["acc"]
            check(f"round-trip {label:<5}", abs(a1 - a2) < 1e-9,
                  f"{a1:.6f} == {a2:.6f} after rebuild from the checkpoint")
            del model2
        except Exception as e:                       # noqa: BLE001
            check(f"round-trip {label:<5}", False, f"{type(e).__name__}: {e}")

    # Resume has to continue, not restart: the step counter carries.
    try:
        c = replace(base, tag="smoke_resume", max_steps=4, eval_every=2)
        run_session(c, corpus, quiet=True, save=True)
        m2, _, _ = run_session(replace(c, max_steps=3), corpus, resume=True,
                               quiet=True, save=False)
        check("resume continues the step counter", m2["steps"] == 3,
              "4 steps saved, 3 more ran")
    except Exception as e:                           # noqa: BLE001
        check("resume continues the step counter", False,
              f"{type(e).__name__}: {e}")

    # Wiring check: the loss has to reach every family that is supposed to
    # learn, and a step has to move them. Asserting that rather than a fit,
    # because on this task neither the loss nor the accuracy moves measurably
    # inside a smoke-sized budget — a gate on either reads the seed, not the
    # pipeline, and the budget it would need is a training run.
    print()
    try:
        c = replace(base, max_steps=12, eval_every=0, log_every=0, batch=12)
        fit = groups
        set_seed(c.seed)
        model, trainer = build(c)
        batches = Batches(fit, c)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}

        b = pack(bucket_by_width(fit, c.batch, c.chunk)[0], c)
        loss, _ = decision_loss(score_all(model, c, *b[:4]), b[4], b[3])
        reached = {n for n, g in zip(
            (n for n, _ in model.named_parameters()),
            torch.autograd.grad(loss, [p for _, p in model.named_parameters()],
                                allow_unused=True))
            if g is not None and float(g.norm()) > 0.0}
        # The memory gate opens on its own schedule and carries the latch
        # behind it, so neither is expected to have gradient from step one.
        want = {"W", "B", "embed.weight", "output_decoder.weight", "norm.weight"}
        check("the loss reaches the core", want <= reached,
              f"gradient in {len(reached)} families, "
              f"missing {sorted(want - reached) or 'none'}")

        for _ in range(c.max_steps):
            train_step(trainer, model, c, batches.next())
        moved = [n for n, p in model.named_parameters()
                 if n in want and not torch.equal(p.detach(), before[n])]
        check("a step moves the core", sorted(moved) == sorted(want),
              f"{len(moved)}/{len(want)} families changed over "
              f"{c.max_steps} steps")

        # The absolute term alone is minimised by pushing every option down
        # together, which reads as progress while the scores stop telling the
        # options apart. The class weight is what prevents it.
        with torch.no_grad():
            s = score_all(model, c, *b[:4])[b[3]]
        spread = float(s.std())
        check("scores stay apart", spread > 1e-4,
              f"option score std {spread:.1e}")
        del model
    except Exception as e:                           # noqa: BLE001
        check("the loss reaches the core", False,
              f"{type(e).__name__}: {e}")

    for stale in glob.glob(os.path.join(CKPT_DIR, "s1_odyss_smoke*.pth")):
        os.remove(stale)

    print(f"\n{'🎉 SMOKE PASSED' if not failures else '💥 SMOKE FAILED: ' + ', '.join(failures)}")
    return not failures


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

EPILOG = """\
data format (JSONL, one object per context):
  {"context": "...", "questions": [
      {"q": "...", "options": ["...", "..."], "correct": 0}]}
  correct: -1 means no option is right.

examples:
  # end-to-end self-test on a built-in corpus; safe to run twice
  %(prog)s --mode smoke
  %(prog)s --mode smoke --device cpu

  # a fresh run (refuses to clobber an existing tag)
  %(prog)s --mode train --tag base --minutes 20

  # continue it, or rewind to the best checkpoint after a collapse
  %(prog)s --mode train --tag base --resume
  %(prog)s --mode train --tag base --resume-best --lr 3e-4

  # your own data
  %(prog)s --mode train --data mine/train.jsonl --val mine/val.jsonl

  # several corpora as one, assembled at the command line
  %(prog)s --mode train --data ../../data/decisions,../../data/basics,../../data/wordnet

  # compute-matched ablations; every arm gets the same wall clock
  %(prog)s --mode sweep --sweep think --minutes 4
  %(prog)s --mode sweep --sweep mech --arms plain,attn4 --minutes 6

  # score or query a saved checkpoint (architecture read from the file)
  %(prog)s --mode eval --tag base --val mine/test.jsonl
  %(prog)s --mode ask --tag base --context "cancel my flight to denver" \\
      --question "what does the user want to do?" \\
      --option "cancel a reservation" --option "book a flight"
"""

#: Appended to every "unknown name" rejection. These lists are transcribed
#: from OdyssNet's `_build_activation` and `_apply_init`, which are if/elif
#: chains over string literals and cannot be introspected — so a strategy
#: added to the library is rejected here until someone updates the copy.
_STALE = (". If the library gained this name recently, the accepted list in "
          "parse_args() needs updating.")

ACTIVATIONS = ("none", "identity", "tanh", "relu", "leaky_relu", "sigmoid",
               "gelu", "gelu_tanh", "silu")
INITS = ("quiet", "micro_quiet", "micro_quiet_warm", "classic",
         "xavier_uniform", "xavier_normal", "kaiming_uniform",
         "kaiming_normal", "orthogonal", "sparse", "zero", "one", "resonant")


def parse_args():
    d = Cfg()
    p = argparse.ArgumentParser(
        prog="experiment_system_one.py",
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("mode")
    g.add_argument("--mode", default="train",
                   choices=["train", "sweep", "smoke", "eval", "ask"],
                   help="train: (resumable) training loop. sweep: "
                        "compute-matched ablation. smoke: fast self-test. "
                        "eval/ask: score or query a saved checkpoint. "
                        "(default: %(default)s)")
    g.add_argument("--sweep", default="think", choices=sorted(SWEEPS),
                   help="which ablation preset to run (default: %(default)s)")
    g.add_argument("--arms", default=None, metavar="A,B",
                   help="comma-separated subset of the preset's arms")

    g = p.add_argument_group("data")
    g.add_argument("--data", default=d.data, metavar="PATH[,PATH]",
                   help="training JSONL, or several comma-separated; a "
                        "directory stands for its train.jsonl "
                        "(default: data/decisions/train.jsonl)")
    g.add_argument("--val", default=d.val, metavar="PATH[,PATH]",
                   help="validation JSONL; follows --data when that names a "
                        "directory (default: data/decisions/val.jsonl)")

    g = p.add_argument_group("architecture")
    g.add_argument("--neurons", type=int, default=d.neurons, metavar="N",
                   help="size of the NxN chaos core (default: %(default)s)")
    g.add_argument("--n-in", type=int, default=d.n_in, metavar="N",
                   help="neurons receiving the byte embedding "
                        "(default: %(default)s)")
    g.add_argument("--n-out", type=int, default=d.n_out, metavar="N",
                   help="neurons the scalar decoder reads; n_in + n_out must "
                        "be <= --neurons (default: %(default)s)")
    g.add_argument("--chunk", type=int, default=d.chunk, metavar="N",
                   help="bytes read per forward; the state carries across "
                        "pieces, so a longer context costs more steps, never "
                        "a bigger tensor (default: %(default)s)")
    g.add_argument("--think", type=int, default=d.think, metavar="N",
                   help="echo steps after each segment's bytes and before its "
                        "readout. Depth bought with no new parameters, and "
                        "the knob `--sweep think` measures "
                        "(default: %(default)s)")
    g.add_argument("--activation", default=",".join(d.activation),
                   metavar="ENC,CORE,MEM",
                   help="three activations: encoder/decoder, the core step, "
                        "memory feedback. 'none' is identity here; keep the "
                        "encoder at identity to leave the scalar an unbounded "
                        "logit (default: %(default)s)")
    g.add_argument("--weight-init", default=",".join(d.weight_init),
                   metavar="ENC,CORE,MEM,GATE",
                   help="four initialization strategies "
                        "(default: %(default)s)")
    g.add_argument("--gates", default=",".join(d.gates), metavar="IN,CORE,MEM",
                   help="three gate activations; 'none' creates no parameter "
                        "for that gate (default: %(default)s)")
    g.add_argument("--hebb", default="none",
                   choices=["none", "temporal", "spatial", "both"],
                   help="Hebbian plasticity: a trace the option segment can "
                        "query against what the context wrote "
                        "(default: %(default)s)")
    g.add_argument("--hebb-res", default=d.hebb_res,
                   choices=["global", "neuron"],
                   help="plasticity resolution (default: %(default)s)")
    g.add_argument("--dropout", type=float, default=d.dropout, metavar="P",
                   help="dropout applied every step (default: %(default)s)")

    g = p.add_argument_group(
        "attention",
        "At every step the state queries a cache of the states before it. Off "
        "unless --attn-heads is given, and its output projection starts at "
        "zero, so switching it on changes nothing until training decides "
        "otherwise. Branch rewinds truncate the cache, so an option never "
        "attends to another option's writes.")
    g.add_argument("--attn-heads", type=int, default=d.attn_heads, metavar="N",
                   help="query heads; 0 disables attention entirely "
                        "(default: %(default)s)")
    g.add_argument("--attn-kv-heads", type=int, default=d.attn_kv_heads,
                   metavar="N",
                   help="key/value heads, shared across query heads (must "
                        "divide --attn-heads) (default: %(default)s)")
    g.add_argument("--attn-window", type=int, default=d.attn_window,
                   metavar="N",
                   help="cache entries a query can see (default: %(default)s)")
    g.add_argument("--attn-rope", action=argparse.BooleanOptionalAction,
                   default=d.attn_rope,
                   help="rotary positions on each key (default: %(default)s)")
    g.add_argument("--attn-qk-norm", action=argparse.BooleanOptionalAction,
                   default=d.attn_qk_norm,
                   help="RMSNorm on queries and keys before the dot product "
                        "(default: %(default)s)")

    g = p.add_argument_group("optimization")
    g.add_argument("--batch", type=int, default=d.batch, metavar="N",
                   help="contexts per step; every question under a context "
                        "shares its single context read "
                        "(default: %(default)s)")
    g.add_argument("--lr", default="keep", metavar="RATE",
                   help="'auto' for ChaosGrad's online estimate, a float for "
                        "fixed-rate mode. Default 'keep': a fresh run uses "
                        "auto (zero-config); --resume keeps the checkpoint's mode")
    g.add_argument("--grad-ckpt", action="store_true",
                   help="gradient checkpointing: less memory, one extra "
                        "sequential forward per step")

    g = p.add_argument_group("run control")
    g.add_argument("--minutes", type=float, default=d.minutes, metavar="M",
                   help="wall-clock budget; 0 runs until Ctrl-C, which "
                        "checkpoints cleanly on the way out "
                        "(default: %(default)s)")
    g.add_argument("--max-steps", type=int, default=d.max_steps, metavar="N",
                   help="optimizer-step budget; 0 is unlimited "
                        "(default: %(default)s)")
    g.add_argument("--tag", default=d.tag, metavar="NAME",
                   help="names the checkpoint pair "
                        "s1_odyss_<NAME>_{latest,best}.pth "
                        "(default: %(default)s)")
    g.add_argument("--resume", action="store_true",
                   help="continue --tag's latest checkpoint, restoring "
                        "weights, optimizer and step counter")
    g.add_argument("--resume-best", action="store_true",
                   help="resume from --tag's *best* checkpoint instead; "
                        "implies --resume. The recovery path after a "
                        "collapse, since `_latest` is rewritten every "
                        "evaluation while `_best` only ever improves")
    g.add_argument("--overwrite", action="store_true",
                   help="allow a fresh run to destroy an existing tag's "
                        "checkpoints")
    g.add_argument("--seed", type=int, default=d.seed, metavar="N",
                   help="seeds init and shuffling (default: %(default)s)")
    g.add_argument("--device", default=d.device, metavar="DEV",
                   help="cuda or cpu (default: %(default)s)")

    g = p.add_argument_group("evaluation and logging")
    g.add_argument("--eval-every", type=int, default=d.eval_every, metavar="N",
                   help="steps between held-out validations; 0 disables "
                        "mid-run scoring (default: %(default)s)")
    g.add_argument("--eval-contexts", type=int, default=d.eval_contexts,
                   metavar="N",
                   help="contexts scored per validation; 0 scores the whole "
                        "validation file (default: %(default)s)")
    g.add_argument("--log-every", type=int, default=d.log_every, metavar="N",
                   help="steps between progress lines; 0 is silent "
                        "(default: %(default)s)")

    g = p.add_argument_group("ask")
    g.add_argument("--context", default="", metavar="TEXT",
                   help="the state every --question is asked against")
    g.add_argument("--question", action="append", default=[], metavar="TEXT",
                   help="repeatable; all of them share one context read")
    g.add_argument("--option", action="append", default=[], metavar="TEXT",
                   help="repeatable; the option set every question is scored "
                        "over")

    a = p.parse_args()

    # --resume-best is a variant of --resume, not an alternative: folding it in
    # here keeps every downstream check from having to test both.
    if a.resume_best:
        a.resume = True

    # Validate here rather than at construction time, so a typo costs a second
    # and a message instead of a corpus load and a CUDA init.
    for name in ("neurons", "n_in", "n_out", "batch", "chunk"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive, "
                    f"got {getattr(a, name)}")
    if a.n_in + a.n_out > a.neurons:
        p.error(f"--n-in ({a.n_in}) + --n-out ({a.n_out}) = "
                f"{a.n_in + a.n_out} exceeds --neurons ({a.neurons})")
    if a.think < 0:
        p.error(f"--think must be >= 0, got {a.think}")
    if str(a.lr) not in ("auto", "keep"):
        try:
            if float(a.lr) <= 0:
                raise ValueError
        except ValueError:
            p.error(f"--lr must be 'auto', 'keep' or a positive float, "
                    f"got {a.lr!r}")

    a.gates = tuple(s.strip() for s in a.gates.split(","))
    if len(a.gates) != 3:
        p.error(f"--gates needs exactly three comma-separated entries "
                f"(input,core,memory), got {len(a.gates)}")
    unknown = [x for x in a.gates if x.lower() not in ACTIVATIONS]
    if unknown:
        p.error(f"--gates: unknown activation(s) {unknown}; "
                f"choose from {list(ACTIVATIONS)}{_STALE}")

    a.activation = tuple(s.strip() for s in a.activation.split(","))
    if len(a.activation) != 3:
        p.error(f"--activation needs exactly three comma-separated entries "
                f"(encoder,core,memory), got {len(a.activation)}")
    unknown = [x for x in a.activation if x.lower() not in ACTIVATIONS]
    if unknown:
        p.error(f"--activation: unknown activation(s) {unknown}; "
                f"choose from {list(ACTIVATIONS)}{_STALE}")

    a.weight_init = tuple(s.strip() for s in a.weight_init.split(","))
    if len(a.weight_init) != 4:
        p.error(f"--weight-init needs exactly four comma-separated entries "
                f"(encoder,core,memory,gate), got {len(a.weight_init)}")
    unknown = [x for x in a.weight_init if x.lower() not in INITS]
    if unknown:
        p.error(f"--weight-init: unknown strategy/strategies {unknown}; "
                f"choose from {list(INITS)}{_STALE}")

    if a.attn_heads < 0:
        p.error(f"--attn-heads must be >= 0 (0 disables), got {a.attn_heads}")
    if a.attn_heads:
        if a.attn_kv_heads < 1:
            p.error(f"--attn-kv-heads must be >= 1, got {a.attn_kv_heads}")
        if a.attn_heads % a.attn_kv_heads:
            p.error(f"--attn-heads ({a.attn_heads}) must be divisible by "
                    f"--attn-kv-heads ({a.attn_kv_heads})")
        if a.attn_window < 1:
            p.error(f"--attn-window must be >= 1, got {a.attn_window}")
    elif (a.attn_kv_heads != d.attn_kv_heads
          or a.attn_window != d.attn_window
          or a.attn_rope != d.attn_rope
          or a.attn_qk_norm != d.attn_qk_norm):
        print("ℹ️  attention flags ignored: --attn-heads is 0, so no "
              "attention is built. Pass --attn-heads 4 to switch it on.")

    if a.mode == "ask":
        if not a.context:
            p.error("--mode ask needs --context")
        if not a.question:
            p.error("--mode ask needs at least one --question")
        if len(a.option) < 2:
            p.error("--mode ask needs at least two --option values")

    return a


def cfg_from_args(a):
    d = Cfg()
    return Cfg(
        data=a.data, val=a.val, chunk=a.chunk,
        neurons=a.neurons, n_in=a.n_in, n_out=a.n_out, think=a.think,
        activation=a.activation, weight_init=a.weight_init, gates=a.gates,
        hebb_type="" if a.hebb == "none" else a.hebb, hebb_res=a.hebb_res,
        dropout=a.dropout,
        attn_heads=a.attn_heads, attn_kv_heads=a.attn_kv_heads,
        attn_window=a.attn_window, attn_rope=a.attn_rope,
        attn_qk_norm=a.attn_qk_norm,
        batch=a.batch,
        lr=d.lr if str(a.lr) == "keep" else (
            None if str(a.lr) == "auto" else float(a.lr)),
        grad_ckpt=a.grad_ckpt,
        minutes=a.minutes, max_steps=a.max_steps,
        eval_every=a.eval_every, eval_contexts=a.eval_contexts,
        log_every=a.log_every,
        seed=a.seed, device=a.device, tag=a.tag,
    )


def main():
    a = parse_args()
    cfg = cfg_from_args(a)
    set_seed(cfg.seed)

    if cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("🚀 OdyssNet-SystemOne — typed decisions, options supplied at "
          "runtime")
    print(f"   mode {a.mode} | device {cfg.device} | seed {cfg.seed}")

    # Rebuild from the architecture a checkpoint was trained with rather than
    # from whatever flags are on this command line. An architecture flag
    # omitted on --resume would otherwise revert to a CLI default, fail the
    # strict load, and leave a run that --resume already excused from
    # guard_overwrite free to overwrite the checkpoint.
    path = None
    if a.mode in ("eval", "ask"):
        latest, best = ckpt_paths(cfg)
        path = best if os.path.exists(best) else latest
        if not os.path.exists(path):
            raise SystemExit(
                f"\n✋ No checkpoint for tag '{cfg.tag}' (looked for "
                f"{path}).\n"
                f"   Train one first: --mode train --tag {cfg.tag}\n")
        cfg = adopt_saved_arch(cfg, path, fields=ARCH_FIELDS)
    elif a.mode == "train" and a.resume:
        latest, best = ckpt_paths(cfg)
        # Read the architecture from the file the run will actually load, or
        # --resume-best rebuilds from the wrong one and fails the strict load
        # it is trying to rescue.
        src = best if a.resume_best else latest
        if os.path.exists(src):
            cfg = adopt_saved_arch(cfg, src, fields=RESUME_FIELDS)

    if a.mode == "ask":
        model, trainer = build(cfg)
        load_checkpoint(model, trainer.optimizer, path, device=cfg.device,
                        strict=True)
        print(f"📂 {path}")
        print(f"\n  context: {a.context!r}")
        print(f"  (read once; {len(a.question)} question(s) over "
              f"{len(a.option)} options)")
        answers = ask(model, cfg, a.context,
                      [(q, list(a.option)) for q in a.question],
                      return_trajectories=True)
        print_answers(answers)
        plot_trajectories(a.context, answers)
        return

    if a.mode == "eval":
        val_groups = group_by_context(load_reads(cfg.val))
        model, trainer = build(cfg)
        load_checkpoint(model, trainer.optimizer, path, device=cfg.device,
                        strict=True)
        print(f"📂 {path}")
        validator = Validator(val_groups, cfg)
        print(f"\n📚 scoring {len(validator.groups):,} contexts | "
              f"{validator.answerable:,} answerable + "
              f"{validator.rejectable:,} rejectable questions")
        m = validator.run(model)
        print(f"\n📊 {fmt(m)}")
        if "brier" in m:
            print(f"   brier {m['brier']:.4f} | mean confidence "
                  f"{m['conf']:.3f} | {model.get_num_params():,} params")
        return

    corpus = load_corpus(cfg)

    if a.mode == "smoke":
        sys.exit(0 if run_smoke(cfg, corpus) else 1)

    if a.mode == "sweep":
        run_sweep(cfg, corpus, a.sweep, a.minutes or 3.0, arms=a.arms)
        return

    # --- train ---
    if not a.resume:
        guard_overwrite(cfg, overwrite=a.overwrite)

    metrics, _, _ = run_session(
        cfg, corpus, budget_sec=cfg.minutes * 60.0, resume=a.resume,
        resume_best=a.resume_best)

    print(f"\n{'='*78}")
    print(f"📊 FINAL  {fmt(metrics)}")
    if "brier" in metrics:
        print(f"   brier {metrics['brier']:.4f} | mean confidence "
              f"{metrics['conf']:.3f}")
    print(f"   {metrics['steps']:,} steps over {metrics['epochs']} epochs in "
          f"{metrics['minutes']:.1f} min ({metrics['steps_s']:.1f} steps/s), "
          f"{metrics['params']:,} params")
    print(f"{'='*78}")
    print("   checkpoints: "
          + ", ".join(os.path.basename(p) for p in ckpt_paths(cfg)))


if __name__ == "__main__":
    main()
