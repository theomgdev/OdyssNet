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
import hashlib
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
from odyssnet.training.chaos_optimizer import ChaosGrad

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "ckpt")
DATA_DIR = os.path.join(HERE, "..", "..", "data", "decisions")
CACHE_DIR = os.path.join(DATA_DIR, "read_cache")

# Segment markers. Ids 0-3 are reserved so a marker is never ambiguous with a
# text byte; text is shifted up by OFFSET.
#: -1 is the core's "inject nothing" sentinel, so a padded row reads exactly
#: as the short row it was padded from and a score never depends on what else
#: shares its batch.
PAD_ID = -1
SEG_CONTEXT, SEG_QUESTION, SEG_OPTION = 0, 1, 2
OFFSET = 3
VOCAB = 256 + OFFSET

#: A question is read as declined when its best option's probability falls
#: below this. Note what a flat distribution does: over K>=3 options the top
#: probability is 1/K < 0.5 already, so the rate a model has to beat is not
#: zero and `Validator` reports both.
REJECT_AT = 0.5


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
    batch: int = -1                 # -1 = empirical autotune; contexts/step
    # None = ChaosGrad's online estimate (default, zero-config); float = fixed-rate mode.
    lr: float | None = None
    grad_ckpt: bool = False
    # The decision loss is one scalar per question accumulated over three
    # recurrent segments, so its natural gradient norm sits near 80 — two
    # orders above a per-token loss, and clipping at 1.0 throws that away.
    clip: float = 80.0              # gradient clipping threshold
    # ChaosGrad's estimate reads `(grad * (p0 - p)).sum()` — how well the
    # gradient aligns with where the weights have actually travelled. Opposing
    # gradients across steps keep that sum near zero or negative, so the
    # estimate never ratchets and the run sits at chance however long it goes.
    d0: float = 1e-6                # initial step-scale estimate
    compile: bool = False           # torch.compile the forward. The step is
                                    # launch-bound at every size measured, so
                                    # fusing it is the largest speedup on
                                    # offer; warmup costs a minute or two.

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
    __slots__ = ("context", "question", "options", "correct", "band")
    context: str
    question: str
    options: list
    correct: int                    # index into options, or -1 for "none"
    band: int                       # the corpus's own difficulty label, or 0


class Corpus:
    """
    A parsed corpus whose text lives on disk, not in the heap.

    JSONL is parsed once into four files:
      <stamp>.blob   every distinct string, concatenated UTF-8
      <stamp>.str    (n_strings + 1,) int64 offsets into the blob
      <stamp>.opt    (total_options,) int64 string ids of options (flat, 0 pad)
      <stamp>.read   (n_reads, 6) int64: ctx, qry, correct, band, opt_lo, opt_hi

    The table is what a run walks; the blob and option table are memory-mapped,
    so the text is paged in by the OS on demand and a corpus larger than RAM
    costs nothing extra.

    Strings are deduplicated as they are written, because a decision corpus is
    mostly repeated text — one context is carried by every question under it,
    and option sets are drawn from a small pool. That dedup is also why a
    string id is worth storing: the pool fits in the table even when the text
    does not fit in memory.

    There is no ceiling on option count K: options form a contiguous slice
    in `.opt`, so K=2 and K=150 use exactly their actual length at zero pad.
    """

    WIDTH = 6                       # ctx, qry, correct, band, opt_lo, opt_hi

    def __init__(self, stamp):
        self.stamp = stamp
        base = os.path.join(CACHE_DIR, stamp)
        self._str = np.memmap(base + ".str", dtype=np.int64, mode="r")
        self._blob = np.memmap(base + ".blob", dtype=np.uint8, mode="r")
        self._opt = np.memmap(base + ".opt", dtype=np.int64, mode="r")
        reads = np.memmap(base + ".read", dtype=np.int64, mode="r")
        self.reads = reads.reshape(-1, self.WIDTH)

    def text(self, sid):
        """The string with this id.

        Deliberately uncached: a cache keyed by string id grows to the whole
        corpus over a long run, which is the heap cost this class exists to
        avoid. Only the batch being packed is ever decoded, and a short slice
        of a mapped array is cheap.
        """
        lo, hi = int(self._str[sid]), int(self._str[sid + 1])
        return bytes(self._blob[lo:hi]).decode("utf-8")

    def byte_len(self, sid):
        """UTF-8 length without decoding — `widths` needs only this."""
        return int(self._str[sid + 1]) - int(self._str[sid])

    def read(self, i):
        """Row `i` as a `Read`, with its text resolved."""
        row = self.reads[i]
        opts = [self.text(int(o)) for o in self._opt[row[4]:row[5]]]
        return Read(self.text(int(row[0])), self.text(int(row[1])),
                    opts, int(row[2]), int(row[3]))

    def widths(self, group, chunk):
        """
        The step geometry of a group, from the offset table alone.

        This runs over every group on every epoch, so it must not decode
        anything: lengths are offset differences, and the bucket key is a
        count of chunks rather than of bytes.
        """
        ctx_id, idx = group
        up = lambda n: -(-(n + 1) // chunk)  # noqa: E731  (+1 for the marker)
        q_max = o_max = 0
        for i in idx:
            row = self.reads[i]
            q_max = max(q_max, up(self.byte_len(int(row[1]))))
            for o in self._opt[row[4]:row[5]]:
                o_max = max(o_max, up(self.byte_len(int(o))))
        return (up(self.byte_len(ctx_id)), q_max, o_max)

    def group_by_context(self):
        """
        [(ctx_id, [read_idx, ...]), ...] — the unit the context cache amortises.

        Grouping is what makes the stored state worth anything: every question
        under one context shares a single context read. Contexts are already
        adjacent in the table because the cache writes a JSONL line's
        questions together, so this is one pass with no index dict.
        """
        out = []
        ctx_col = np.asarray(self.reads[:, 0])
        if not ctx_col.size:
            return out
        cuts = np.flatnonzero(np.diff(ctx_col)) + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, ctx_col.size]):
            out.append((int(ctx_col[lo]), list(range(int(lo), int(hi)))))
        return out

    def __len__(self):
        return self.reads.shape[0]


def _cache_stamp(paths):
    """A name derived from what the cache was built from.

    Path, size and mtime of every input: regenerating a corpus changes the
    mtime, so a stale cache is never read as current. The alternative —
    hashing the contents — would read gigabytes to decide whether to read
    gigabytes.
    """
    parts = []
    for p in paths:
        st = os.stat(p)
        parts.append(f"{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}")
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]
    return digest


def build_cache(paths, stamp, verbose=False):
    """
    Parse JSONL into four cache files. Nothing accumulates in RAM.

    The string pool is a dict of text -> id held while writing, which is the
    one unavoidable cost: dedup needs to remember what it has seen. It holds
    distinct strings only, so it is bounded by the corpus's vocabulary of
    contexts and options rather than by its question count.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.join(CACHE_DIR, stamp)

    pool, offsets, blob_at = {}, [0], 0
    n_reads = 0
    opt_at = 0
    t0 = time.time()

    with open(base + ".blob.part", "wb") as blob, \
         open(base + ".read.part", "wb") as table, \
         open(base + ".opt.part", "wb") as opt_file:

        def intern(text):
            nonlocal blob_at
            got = pool.get(text)
            if got is not None:
                return got
            raw = text.encode("utf-8")
            blob.write(raw)
            blob_at += len(raw)
            offsets.append(blob_at)
            got = len(pool)
            pool[text] = got
            return got

        for path in paths:
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
                    ctx_id = intern(ctx)
                    for q in questions:
                        options = q.get("options")
                        if not isinstance(options, list) or len(options) < 2:
                            raise SystemExit(
                                f"\n✋ {path}:{n}: a question needs at least "
                                f"two options.\n")
                        correct = int(q.get("correct", -1))
                        if correct >= len(options):
                            raise SystemExit(
                                f"\n✋ {path}:{n}: correct={correct} is out of "
                                f"range for {len(options)} options.\n")
                        opt_ids = np.array([intern(str(o)) for o in options], dtype=np.int64)
                        opt_ids.tofile(opt_file)
                        opt_start = opt_at
                        opt_at += len(options)
                        opt_end = opt_at
                        rec = np.array([ctx_id, intern(str(q.get("q", ""))),
                                        correct, int(q.get("band", 0)),
                                        opt_start, opt_end], dtype=np.int64)
                        rec.tofile(table)
                        n_reads += 1
                    if verbose and n % 50000 == 0:
                        print(f"   {os.path.basename(path)}: {n:,} lines, "
                              f"{n_reads:,} questions  {time.time()-t0:4.0f}s")

    if not n_reads:
        raise SystemExit(f"\n✋ {', '.join(paths)}: no usable rows.\n")

    np.asarray(offsets, dtype=np.int64).tofile(base + ".str.part")
    for ext in (".blob", ".read", ".str", ".opt"):
        os.replace(base + ext + ".part", base + ext)
    if verbose:
        print(f"   cached {n_reads:,} questions | {opt_at:,} options | "
              f"{len(pool):,} distinct strings | {blob_at/1e6:.1f} MB blob")
    return n_reads


def open_corpus(paths, verbose=False):
    """The cache for these paths, built if it is missing or stale."""
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(
                f"\n✋ No data at {p}.\n"
                f"   --data / --val take JSONL (or a directory containing "
                f"train.jsonl / val.jsonl):\n"
                f'     {{"context": "...", "questions": [\n'
                f'        {{"q": "...", "options": ["a", "b"], "correct": 0}}]}}\n'
                f"   correct: -1 means no option is right.\n")
    stamp = _cache_stamp(paths)
    base = os.path.join(CACHE_DIR, stamp)
    if not all(os.path.exists(base + e) for e in (".blob", ".str", ".read", ".opt")):
        build_cache(paths, stamp, verbose=verbose)
    return Corpus(stamp)


def load_corpus(cfg, verbose=True):
    """
    (train, val) as `Corpus` objects with their context groups.

    Returns `((train_corpus, train_groups), (val_corpus, val_groups))`. Each
    corpus is parsed once into a memmap cache keyed by its inputs, so a
    re-run of the same data costs a stat call rather than a full parse, and
    the text never enters the heap.

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

    out = []
    for label, paths in (("train", train_paths), ("val", val_paths)):
        corpus = open_corpus(paths, verbose=verbose)
        groups = corpus.group_by_context()
        out.append((corpus, groups))
        if verbose:
            k = corpus.reads[:, 5] - corpus.reads[:, 4]
            none = int((corpus.reads[:, 2] < 0).sum())
            print(f"📚 {label}: {len(groups):,} contexts | {len(corpus):,} "
                  f"questions | K {int(k.min())}-{int(k.max())} "
                  f"(mean {float(k.mean()):.1f}) "
                  f"| {none:,} answered by none of the options"
                  + (f" | {len(paths)} files" if len(paths) > 1 else ""))
    return out[0], out[1]


def encode(text, marker):
    return [marker] + [b + OFFSET for b in text.encode("utf-8")]


def pad_to(rows, width):
    out = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


@dataclass
class Batch:
    """
    One step's work as flat rows, the way the card wants it.

    A decision batch is ragged in two directions at once — contexts carry
    different question counts and questions carry different option counts —
    and a `(B, Q, K, L)` tensor pays the batch maximum on both. The rows are
    therefore flat and the structure lives in index tensors:

        ctx        (B, Lc)                 one row per context
        qry        (Nq, Lq)                every question, flattened
        opt        (No, Lo)                every option, flattened
        q_owner    (Nq,)   -> context row
        o_owner    (No,)   -> question row
        correct    (Nq,)   index into that question's options, or -1
        q_span     (Nq+1,) option slice per question, for the ranking term

    `score_all` branches by gathering on `q_owner` / `o_owner`, so no padding
    row is ever forwarded and the step count still depends only on segment
    width.
    """
    ctx: torch.Tensor
    qry: torch.Tensor
    opt: torch.Tensor
    q_owner: torch.Tensor
    o_owner: torch.Tensor
    correct: torch.Tensor
    q_span: torch.Tensor

    @property
    def n_questions(self):
        return self.qry.shape[0]

    @property
    def n_options(self):
        return self.opt.shape[0]


def pack(groups, cfg, corpus=None):
    """
    A `Batch` of flat rows on the device.

    Each of the three segment kinds is padded to the longest entry *of its own
    kind in this batch*, rounded up to a chunk: segments are chained through
    `current_state`, so a three-byte option has no reason to run a long one's
    steps. Nothing is padded along the question or option axis, because there
    is no such axis.

    With a `corpus`, `groups` are `(ctx_id, [read_index, ...])` and the text is
    decoded here — one batch's worth — and never held between steps. Without
    one, they are `(context, [Read, ...])`, which is what a caller that built
    its reads in memory has: `--mode ask` and the smoke checks.
    """
    if corpus is None:
        rows = [(ctx, list(reads)) for ctx, reads in groups]
    else:
        rows = [(corpus.text(c), [corpus.read(j) for j in idx])
                for c, idx in groups]

    ctx_rows = [encode(c, SEG_CONTEXT) for c, _ in rows]
    qry_rows, opt_rows = [], []
    q_owner, o_owner, correct, q_span = [], [], [], [0]

    for i, (_, reads) in enumerate(rows):
        for r in reads:
            qry_rows.append(encode(r.question, SEG_QUESTION))
            q_owner.append(i)
            q_index = len(qry_rows) - 1
            for opt in r.options:
                opt_rows.append(encode(opt, SEG_OPTION))
                o_owner.append(q_index)
            correct.append(r.correct)
            q_span.append(len(opt_rows))

    up = lambda n: max(cfg.chunk, -(-n // cfg.chunk) * cfg.chunk)  # noqa: E731
    to = lambda a, d=np.int64: torch.from_numpy(  # noqa: E731
        np.asarray(a, dtype=d)).to(cfg.device, non_blocking=True)

    return Batch(
        ctx=to(pad_to(ctx_rows, up(max(len(r) for r in ctx_rows)))),
        qry=to(pad_to(qry_rows, up(max(len(r) for r in qry_rows)))),
        opt=to(pad_to(opt_rows, up(max(len(r) for r in opt_rows)))),
        q_owner=to(q_owner), o_owner=to(o_owner),
        correct=to(correct), q_span=to(q_span))


def widths(group, chunk, corpus=None):
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
    if corpus is not None:
        return corpus.widths(group, chunk)
    ctx, reads = group
    up = lambda n: -(-(n + 1) // chunk)  # noqa: E731  (+1 for the marker)
    return (up(len(ctx.encode("utf-8"))),
            max(up(len(r.question.encode("utf-8"))) for r in reads),
            max(up(len(o.encode("utf-8"))) for r in reads for o in r.options))


def bucket_by_width(groups, size, chunk, corpus=None):
    """
    Groups split into batches whose rows share a step geometry.

    Rows of one batch run the same number of steps, so a batch mixing
    geometries would pay its longest row's steps for every row and its scores
    would stop being reproducible across `--batch`.
    """
    buckets = {}
    for g in groups:
        buckets.setdefault(widths(g, chunk, corpus), []).append(g)
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

    def __init__(self, groups, cfg, corpus=None):
        self.cfg = cfg
        self.corpus = corpus
        self.rng = random.Random(cfg.seed)
        self.pools = {}
        for g in groups:
            self.pools.setdefault(widths(g, cfg.chunk, corpus), []).append(g)
        self.epochs = 0
        self.queue = []
        self._refill()

    def _refill(self):
        for rows in self.pools.values():
            self.rng.shuffle(rows)
        self.queue = cut(self.pools, self.cfg.batch)
        self.rng.shuffle(self.queue)

    def state(self):
        """
        Where the corpus walk is, for the checkpoint.

        The queue is regenerated from `rng` plus `epochs`, so replaying the
        RNG to the same draw count restores the same remaining batches. That
        is cheaper and smaller than storing the queue, and it stays correct
        across a corpus that has not changed.
        """
        return {"epochs": self.epochs, "left": len(self.queue),
                "batch": self.cfg.batch}

    def restore(self, snap):
        """Replay the walk to where the checkpoint left it."""
        if not snap:
            return
        saved_batch = snap.get("batch")
        if saved_batch is not None and saved_batch != self.cfg.batch:
            # Silently ignoring this would resume the *weights* while quietly
            # rewinding the *data* to the first context — the model would
            # re-read what it has already seen and nothing would say so.
            print(f"⚠️  Corpus position not restored: checkpoint ran "
                  f"batch={saved_batch}, this run has {self.cfg.batch}. "
                  f"Re-run with the checkpoint's batch size to continue where "
                  f"it left off; training will otherwise restart the corpus "
                  f"from the beginning.")
            return
        target_epochs = int(snap.get("epochs", 0))
        left = int(snap.get("left", 0))
        while self.epochs < target_epochs:
            self.epochs += 1
            self._refill()
        if 0 <= left <= len(self.queue):
            del self.queue[left:]

    def next(self):
        if not self.queue:
            self.epochs += 1
            self._refill()
        return pack(self.queue.pop(), self.cfg, self.corpus)

    def peek(self):
        """The next batch's shape without consuming it, for the cost report."""
        return pack(self.queue[-1], self.cfg, self.corpus)


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
    if cfg.compile:
        # Compile the bound method rather than the module: everything else in
        # this script reaches through `model` for state, caches and
        # checkpoints, and an OptimizedModule wrapper would sit between them
        # and the real object.
        model.forward = torch.compile(model.forward)
    trainer = OdyssNetTrainer(
        model, device=cfg.device, max_grad_norm=cfg.clip,
        optimizer=ChaosGrad.from_model(model, lr=cfg.lr, d0=cfg.d0))
    # The loss is assembled here, over a variable number of options, so the
    # trainer's criterion has to pass the scalar through rather than square it
    # against a zero target.
    trainer.loss_fn = lambda pred, _target: pred.mean()
    return model, trainer


def cost_advisory(cfg, model, batch):
    """
    What one step holds, before it holds it.

    87k parameters are a third of a megabyte; what fills a card here is the
    option read. `score_all` gathers one row per option and keeps every
    recurrent step for backward, so the tensor that decides whether a run fits
    is `options x neurons x steps` — and `--batch` counts contexts, which is
    between one and two orders of magnitude below it.
    """
    params = sum(p.numel() for p in model.parameters())
    rows = batch.n_options
    steps = batch.opt.shape[1] + cfg.think
    state_gb = rows * cfg.neurons * steps * 4 / 1e9
    print(f"📐 {params*4/1e6:.2f} MB of weights | one step reads "
          f"{batch.ctx.shape[0]} contexts -> {batch.n_questions} questions -> "
          f"{rows} options")
    print(f"   option read is {rows} rows x {steps} steps x {cfg.neurons} "
          f"neurons = {state_gb:.2f} GB of state kept for backward")
    if state_gb > 1.5:
        suggest = max(1, int(cfg.batch * 1.5 / state_gb))
        print(f"⚠️  that term alone is {state_gb:.1f} GB and it is linear in "
              f"the batch — try --batch {suggest}, a shorter --chunk, or "
              f"--grad-ckpt.")
    if model.attn is not None:
        writes = steps
        kv_gb = model.attn.training_cache_bytes(rows, writes) / 1e9
        print(f"👁️  attention {model.attn.heads}x{model.attn.head_dim} "
              f"({model.attn.kv_heads} kv) | KV kept for backward "
              f"{kv_gb:.2f} GB ({writes} writes at {rows} rows)")
        if kv_gb > 1.5:
            print("⚠️  the KV term grows with the square of the writes per "
                  "step — shorten --chunk before blaming the core.")
    if model.hebb_type is not None:
        paths = 2 if model.hebb_type == "both" else 1
        trace_gb = paths * steps * rows * cfg.neurons ** 2 * 4 / 1e9
        print(f"🧬 plasticity {model.hebb_type}/{model.hebb_res} | trace kept "
              f"for backward ~{trace_gb:.1f} GB ({paths} path(s) x {steps} "
              f"steps x {rows} rows x {cfg.neurons}²)")
        if trace_gb > 1.5:
            print("⚠️  the plastic trace is the largest activation in this "
                  "model by far — --grad-ckpt exists for exactly it.")


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
    Integrate each option's per-step scores into one decision scalar.

    `(rows, T)` and `(rows,)` -> `(rows,)`. Weights rise as tau^power over
    normalized time tau = (t+1)/length, which suppresses the prefix a short
    option shares with its competitors without a hard cutoff, and steps past a
    row's own length weigh nothing.
    """
    t_max = step_scores.shape[-1]
    t_idx = torch.arange(1, t_max + 1, device=step_scores.device,
                         dtype=step_scores.dtype)
    lens = active_lengths.unsqueeze(-1).clamp(min=1).to(step_scores.dtype)

    weights = (t_idx / lens).clamp(max=1.0).pow(power) * (t_idx <= lens)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (step_scores * weights).sum(dim=-1)


def _branch(model, h, index):
    """
    Gather a stored state into one row per continuation.

    Branching is a batch operation here: every option reads its own question's
    state, so they go down the batch axis in one forward instead of a Python
    loop. `index` may repeat a row any number of times and the count need not
    be the same for every row — which is what lets a 2-option question share a
    batch with a 12-option one at no padding cost. The attention cache is
    indexed by batch row and is gathered the same way; Hebbian state is
    `(N, N)` and shared, so it needs nothing.
    """
    if model.attn is not None:
        model.attn.gather_rows(index)
    return h.index_select(0, index)


def score_all(model, cfg, batch, return_trajectories=False):
    """
    One scalar per option, from a single context read.

    Returns `(No,)` — one score per option row, in the batch's own flat order.
    The cost is three segment reads whatever the question and option counts
    are: the context once, every question as one branch of the context states,
    every option as one branch of the question states. Options and questions
    still cannot see each other, because each branch starts from a stored
    state and writes into its own row.
    """
    model.reset_state(batch_size=batch.ctx.shape[0])
    if model.attn is not None:
        model.attn.reset()

    _, h_ctx = read_segment(model, batch.ctx, cfg)
    ctx_mark = _mark(model)

    h_wide = _branch(model, h_ctx, batch.q_owner)
    _, h_qry = read_segment(model, batch.qry, cfg, state=h_wide)

    o_wide = _branch(model, h_qry, batch.o_owner)
    step_scores, _ = read_segment(model, batch.opt, cfg, state=o_wide,
                                  return_sequence=True)

    _rewind(model, ctx_mark)
    if model.attn is not None:
        # Every branch is gone; one row per context is what the next step
        # expects, and the rewind above already dropped what the branches wrote.
        model.attn.gather_rows(
            torch.arange(batch.ctx.shape[0], device=batch.ctx.device))

    lens = (batch.opt != PAD_ID).sum(dim=-1)
    scores = smooth_trajectory_score(step_scores, lens, power=2.0)
    if return_trajectories:
        return scores, step_scores, lens
    return scores


def decision_loss(scores, batch):
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

    The ranking term runs over flat rows: options of one question are a
    contiguous slice, so a segment-wise log-sum-exp is the cross-entropy
    without ever materialising a `(Nq, K_max)` matrix.
    """
    starts, ends = batch.q_span[:-1], batch.q_span[1:]
    has = batch.correct >= 0

    target = torch.zeros_like(scores)
    if has.any():
        target[starts[has] + batch.correct[has]] = 1.0
    pos = target.sum()
    neg = target.numel() - pos
    absolute = F.binary_cross_entropy_with_logits(
        scores, target, pos_weight=(neg + 1) / (pos + 1))

    if has.any():
        # log-sum-exp per question, max-shifted, over a ragged partition.
        q_of = torch.repeat_interleave(
            torch.arange(len(starts), device=scores.device), ends - starts)
        peak = torch.full((len(starts),), -float("inf"), device=scores.device)
        peak = peak.scatter_reduce(0, q_of, scores, reduce="amax")
        total = torch.zeros_like(peak).index_add_(
            0, q_of, (scores - peak[q_of]).exp())
        chosen = scores[starts[has] + batch.correct[has]]
        rank = (peak[has] + total[has].log() - chosen).mean()
    else:
        rank = scores.new_zeros(())
    return absolute + rank, {"abs": float(absolute), "rank": float(rank)}


def train_step(trainer, model, cfg, batch):
    """One optimizer step over a batch of contexts."""
    terms = {}

    def transform(_out):
        # The trainer owns the step for its optimizer bookkeeping, but this
        # protocol needs many forwards per step, so the real work happens here
        # and the trainer's criterion passes the scalar through.
        total, parts = decision_loss(score_all(model, cfg, batch), batch)
        terms.update(parts)
        return total.reshape(1, 1, 1)

    trainer.train_batch(batch.ctx, torch.zeros(1, device=cfg.device),
                        thinking_steps=batch.ctx.shape[1] + cfg.think,
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

    Every metric is reported against what it would be without a model. A
    decision corpus mixes 2-option and 12-option questions, so chance is not
    1/K for any single K — it is the mean of 1/K over the questions actually
    scored, and an accuracy a point above it is noise rather than learning.
    Rejection is worse: a flat distribution over three or more options already
    has `max p < 0.5`, so the reject rate scores free and only its gap over
    chance means anything.
    """

    def __init__(self, groups, cfg, corpus=None):
        self.cfg = cfg
        self.corpus = corpus
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
        self.batches = bucket_by_width(groups, cfg.batch, cfg.chunk, corpus)
        if corpus is None:
            reads = [r for b in self.batches for _, rs in b for r in rs]
        else:
            reads = [corpus.read(j) for b in self.batches
                     for _, idx in b for j in idx]
        self.contexts = len(groups)
        answerable = [r for r in reads if r.correct >= 0]
        rejectable = [r for r in reads if r.correct < 0]
        self.answerable = len(answerable)
        self.rejectable = len(rejectable)
        # What a model that cannot read at all scores on exactly these
        # questions: the baseline every number below is quoted against.
        self.chance = (sum(1 / len(r.options) for r in answerable)
                       / max(len(answerable), 1))
        self.chance_reject = (sum(1.0 for r in rejectable
                                  if 1 / len(r.options) < REJECT_AT)
                              / max(len(rejectable), 1))

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
        spread = []
        bands = {}

        for rows in self.batches:
            batch = pack(rows, cfg, self.corpus)
            scores = score_all(model, cfg, batch)

            # Val loss: same criterion as training, no gradient.
            _, parts = decision_loss(scores, batch)
            abs_sum += parts["abs"]
            rank_sum += parts["rank"]
            loss_n += 1
            spread.append(float(scores.std()))

            span = batch.q_span.tolist()
            correct = batch.correct.tolist()
            if self.corpus is None:
                reads = [r for _, rs in rows for r in rs]
            else:
                reads = [self.corpus.read(j) for _, idx in rows for j in idx]
            for j, r in enumerate(reads):
                opt = scores[span[j]: span[j + 1]]
                prob = torch.softmax(opt, dim=0)
                top = float(prob.max())
                c = correct[j]
                if c >= 0:
                    n += 1
                    good = int(prob.argmax()) == c
                    hit += int(good)
                    conf.append(top)
                    ok.append(float(good))
                    tally = bands.setdefault(r.band, [0, 0])
                    tally[0] += int(good)
                    tally[1] += 1
                else:
                    # Nothing is right, so a low top score is the answer.
                    rej_n += 1
                    rej_hit += int(top < REJECT_AT)

        m = {"acc": hit / n if n else float("nan"),
             "chance": self.chance,
             "abs": abs_sum / max(loss_n, 1),
             "rank": rank_sum / max(loss_n, 1),
             # A near-zero spread is score collapse, which accuracy cannot
             # see: argmax resolves ties at the last representable digit and
             # produces a rising curve out of rounding residue.
             "spread": sum(spread) / max(len(spread), 1),
             "questions": n + rej_n}
        if rej_n:
            m["reject"] = rej_hit / rej_n
            m["chance_reject"] = self.chance_reject
        if conf:
            c, o = np.asarray(conf), np.asarray(ok)
            m["ece"] = _ece(c, o)
            m["brier"] = float(np.mean((c - o) ** 2))
            m["conf"] = float(c.mean())
        if len(bands) > 1:
            m["bands"] = {b: (h / t, t) for b, (h, t) in sorted(bands.items())}
        return m


def fmt(m):
    """
    One line, every rate next to what it would be without a model.

    `acc 24.83% (chance 23.39%)` is the whole difference between a result and
    a number that looks like one.
    """
    bits = [f"acc {m['acc']*100:6.2f}%"]
    if "chance" in m:
        bits[0] += f" (chance {m['chance']*100:5.2f}%)"
    for key, label in (("abs", "abs"), ("rank", "rank")):
        if key in m:
            bits.append(f"{label} {m[key]:.4f}")
    if "reject" in m:
        bits.append(f"reject {m['reject']*100:5.1f}%"
                    + (f" (chance {m['chance_reject']*100:.0f}%)"
                       if "chance_reject" in m else ""))
    if "spread" in m:
        bits.append(f"spread {m['spread']:.1e}")
    if "ece" in m:
        bits.append(f"ece {m['ece']:.3f}")
    return " | ".join(bits)


def fmt_bands(m):
    """Per-band accuracy, which is what a mixed corpus hides in one average."""
    if "bands" not in m:
        return ""
    return "   bands " + "  ".join(
        f"{b}:{acc*100:.1f}%/{t}" for b, (acc, t) in m["bands"].items())


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
    groups = [(context, [Read(context, q, list(o), -1, 0) for q, o in questions])]
    batch = pack(groups, cfg)
    if return_trajectories:
        scores, step_scores, lens = score_all(model, cfg, batch,
                                              return_trajectories=True)
    else:
        scores = score_all(model, cfg, batch)
        step_scores = lens = None
    if was_training:
        model.train()

    span = batch.q_span.tolist()
    answers = []
    for j, (q, options) in enumerate(questions):
        s = scores[span[j]: span[j + 1]]
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
                options[k]: step_scores[span[j] + k,
                                        : int(lens[span[j] + k])].cpu().tolist()
                for k in range(len(options))
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


def _save(path, model, trainer, cfg, step, metrics, best, stream=None):
    payload = asdict(cfg)
    for f in ("activation", "weight_init", "gates"):
        payload[f] = list(payload[f])
    extra = {"cfg": payload, "step": step,
             "best_acc": best, "metrics": metrics}
    if stream is not None:
        extra["stream"] = stream
    save_checkpoint(model, trainer.optimizer, step,
                    metrics.get("acc", float("nan")), path,
                    extra_data=extra,
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
    if isinstance(corpus[0], tuple) and len(corpus[0]) == 2 and hasattr(corpus[0][0], "reads"):
        (train_corpus, train_groups), (val_corpus, val_groups) = corpus
    else:
        train_corpus, val_corpus = None, None
        train_groups, val_groups = corpus

    set_seed(cfg.seed)

    if cfg.batch <= 0:
        cfg = replace(cfg, batch=autotune_batch(cfg, train_groups, train_corpus))
        set_seed(cfg.seed)

    model, trainer = build(cfg)
    validator = Validator(val_groups, cfg, val_corpus)
    batches = Batches(train_groups, cfg, train_corpus)
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
        if "stream" in info:
            batches.restore(info["stream"])
        print(f"📂 resumed {os.path.basename(resume_path)} at step {step:,} "
              f"(best acc {best:.2%})")

    if not quiet:
        describe(cfg, model)
        cost_advisory(cfg, model, batches.peek())
        print(f"   scoring {validator.contexts:,} contexts | "
              f"{validator.answerable:,} answerable + "
              f"{validator.rejectable:,} rejectable questions | "
              f"chance {validator.chance:.2%}\n")

    metrics = {"acc": float("nan")}
    window = {"abs": 0.0, "rank": 0.0, "n": 0}
    started = time.time()
    interrupted = False
    # Whether the step-scale estimate has left `--d0`; see the note below.
    # True on a resumed run whose estimate already moved, so the notice fires
    # once in a run's life rather than after every restart.
    warmed = (cfg.lr is not None
              or trainer.optimizer.param_groups[0]["d"] > 20 * cfg.d0)
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

            # Say when the online step-scale estimate has found its scale.
            # With `--lr` unset, ChaosGrad reads `grad · (p0 - p)` and needs a
            # few hundred steps before `d` leaves `--d0`; until it does, the
            # loss is genuinely flat and the run looks broken when it is only
            # warming up. Measured on the bundled corpus, the step where `d`
            # first exceeds 20x `d0` is 105 at batch 128 and 146 at batch 6,
            # and 407 at another seed — it arrives in jumps rather than on a
            # ramp, so the quiet stretch is expected rather than a symptom.
            # Announced once, when it ends.
            if not quiet and cfg.lr is None and not warmed:
                d_now = trainer.optimizer.param_groups[0]["d"]
                if d_now > 20 * cfg.d0:
                    warmed = True
                    print(f"   ↳ step scale settled at {d_now:.2e} after "
                          f"{step:,} steps (was --d0 {cfg.d0:.0e}); "
                          f"the loss moves from here.")

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
                              best, stream=batches.state())
                if not quiet:
                    print(f"   ↳ VAL  {fmt(metrics)}{mark}")
                    if "bands" in metrics:
                        print(fmt_bands(metrics))
                if save:
                    _save(latest_path, model, trainer, cfg, step, metrics,
                          best, stream=batches.state())
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏹️  Interrupted — finishing cleanly.")

    elapsed = max(time.time() - started, 1e-6)
    if step > start_step:
        metrics = validator.run(model)
        if save:
            if metrics["acc"] > best:
                best = metrics["acc"]
                _save(best_path, model, trainer, cfg, step, metrics, best,
                      stream=batches.state())
            _save(latest_path, model, trainer, cfg, step, metrics, best,
                  stream=batches.state())

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


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def autotune_batch(cfg, groups, corpus=None, ladder=(8, 16, 32, 64, 128, 256, 512)):
    """
    Measure, don't model. `--batch` counts contexts, but a step's real width is
    the option count it expands to — which varies per batch and cannot be
    predicted from the flag. The step is also launch-bound, so extra rows ride
    along nearly free until they suddenly do not.

    Throughput here is questions per second rather than contexts, because a
    context carrying six questions is six decisions' worth of work and
    comparing contexts would rank a corpus's shape instead of the batch size.
    """
    print("\n⚖️  Autotuning batch size (measured questions/s, not estimated)")
    best, best_rate = ladder[0], 0.0
    for size in ladder:
        c = replace(cfg, batch=size)
        batches = Batches(groups, c, corpus)
        set_seed(cfg.seed)
        model, trainer = build(c)
        try:
            for _ in range(2):                                  # warmup
                train_step(trainer, model, c, batches.next())
            _sync(cfg.device)
            t0 = time.time()
            seen = 0
            for _ in range(3):
                batch = batches.next()
                train_step(trainer, model, c, batch)
                seen += batch.n_questions
            _sync(cfg.device)
            rate = seen / (time.time() - t0)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"   batch {size:>4}: OOM — stopping ladder")
            del model, trainer
            _empty_cache(cfg.device)
            break
        del model, trainer
        _empty_cache(cfg.device)

        mark = ""
        if rate > best_rate * 1.03:
            best, best_rate, mark = size, rate, "  <-- best"
        print(f"   batch {size:>4}: {rate:9,.0f} q/s{mark}")
        if rate < best_rate * 0.85:          # past the knee, stop burning time
            break

    print(f"   selected batch = {best} ({best_rate:,.0f} q/s)")
    return best


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def mean_loss(model, cfg, groups, corpus=None):
    """Loss over fixed batches with no training — the learning-check baseline.

    Carries `corpus` for the same reason `pack` does: with one, a group is
    `(ctx_id, [read_index, ...])` and the text has to be resolved through the
    memmap; without one it is already `(context, [Read, ...])`.
    """
    was_training = model.training
    model.eval()
    total = n = 0
    for rows in bucket_by_width(groups, cfg.batch, cfg.chunk, corpus):
        batch = pack(rows, cfg, corpus)
        t, _ = decision_loss(score_all(model, cfg, batch), batch)
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
    if isinstance(corpus[0], tuple) and len(corpus[0]) == 2 and hasattr(corpus[0][0], "reads"):
        train_corp, train_grps = corpus[0]
        groups = [(train_corp.text(c), [train_corp.read(j) for j in idx])
                  for c, idx in train_grps[:6]]
    else:
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
                  [Read("some context", "which one", options, 0, 0)])]
            model, _ = build(base)
            counts[k] = model.get_num_params()
            model.eval()
            with torch.no_grad():
                shapes[k] = tuple(score_all(model, base, pack(g, base)).shape)
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
                    [Read("cancel my flight", "what now", options, 0, 0)])]
            rev = [("cancel my flight",
                    [Read("cancel my flight", "what now", options[::-1], 2, 0)])]
            with torch.no_grad():
                a = score_all(model, c, pack(fwd, c))
                b = score_all(model, c, pack(rev, c))
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
                with torch.no_grad():
                    s = score_all(model, c, pack(rows, c))
                got += [float(x) for x in s]
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
    def _sequential(model, cfg, batch):
        """One option at a time, the loop the gather replaced."""
        b = batch.ctx.shape[0]
        q_owner = batch.q_owner.tolist()
        o_owner = batch.o_owner.tolist()
        got = torch.zeros(batch.n_options, device=batch.ctx.device)
        for j, owner in enumerate(q_owner):
            for k, q_of in enumerate(o_owner):
                if q_of != j:
                    continue
                # A fresh read per option, so the comparison does not depend
                # on the cache bookkeeping it is meant to check.
                model.reset_state(batch_size=b)
                if model.attn is not None:
                    model.attn.reset()
                _, h_c = read_segment(model, batch.ctx, cfg)
                _, h_q = read_segment(
                    model, batch.qry[j: j + 1], cfg,
                    state=_branch(model, h_c,
                                  batch.q_owner.new_tensor([owner])))
                step_s, _ = read_segment(model, batch.opt[k: k + 1], cfg,
                                         state=h_q, return_sequence=True)
                lens = (batch.opt[k: k + 1] != PAD_ID).sum(dim=-1)
                got[k] = smooth_trajectory_score(step_s, lens, power=2.0)
        return got

    for attn, hebb in ((0, None), (4, None), (0, "temporal"), (4, "temporal")):
        label = f"attn={attn or 0} hebb={hebb or 'none'}"
        try:
            c = replace(base, attn_heads=attn, hebb_type=hebb)
            model, _ = build(c)
            model.eval()
            batch = pack(bucket_by_width(groups, c.batch, c.chunk)[0], c)
            with torch.no_grad():
                want = _sequential(model, c, batch)
                got = score_all(model, c, batch)
            d = (want - got).abs().max().item()
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
        reads = [Read("ctx", "q1", ["a", "b"], 0, 0),
                 Read("ctx", "q2", ["a", "b"], 1, 0),
                 Read("ctx", "q3", ["a", "b"], 0, 0)]
        with torch.no_grad():
            score_all(model, c, pack([("ctx", reads)], c))
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
        batch = pack(groups[:3], c)
        total, _ = decision_loss(score_all(model, c, batch), batch)
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
                        ["pay a bill", "rent a car", "none of these"], 0, 0)])]
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
        loss, _ = decision_loss(score_all(model, c, b), b)
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
            s = score_all(model, c, b)
        spread = float(s.std())
        check("scores stay apart", spread > 1e-4,
              f"option score std {spread:.1e}")
        del model
    except Exception as e:                           # noqa: BLE001
        check("the loss reaches the core", False,
              f"{type(e).__name__}: {e}")

    # Learning check: a handful of contexts is memorizable, so the loss has to
    # fall below where it started. This is the one gate that would catch a
    # pipeline that is wired correctly and still cannot learn — the failure the
    # wiring checks above are blind to by construction.
    #
    # Deliberately at a fixed lr. ChaosGrad's online estimate reads
    # `grad · (p0 - p)`, the alignment between the gradient and the distance
    # already travelled, and that sum needs a few hundred steps to accumulate
    # before `d` leaves `--d0`. How many is not a property of this slice's
    # size: measured as the first step where `d` exceeds 20x `d0`, over a
    # 900-step budget, it is step 105 at batch 128 and step 146 at batch 6 on
    # the full corpus, step 237 on these six contexts, step 407 on the full
    # corpus at another seed — and on six contexts at seed 7 it never happens
    # at all. Every run that does ratchet lands on the same 1.2-1.9e-3, so the
    # estimate is right once it arrives; only its arrival is seed business.
    # A 300-step budget therefore straddles the warm-up, and gating the
    # script's wiring on it would gate on the seed. The warm-up is removed
    # from this test and left to the library's own suite.
    print()
    try:
        c = replace(base, max_steps=300, eval_every=0, log_every=0, batch=6,
                    lr=1e-2)
        set_seed(c.seed)
        model, trainer = build(c)
        batches = Batches(groups, c)
        start = mean_loss(model, c, groups)
        for _ in range(c.max_steps):
            train_step(trainer, model, c, batches.next())
        end = mean_loss(model, c, groups)
        check("loss falls on a memorizable slice", end < start * 0.98,
              f"{start:.4f} -> {end:.4f} over {c.max_steps} steps on "
              f"{len(groups)} contexts at lr={c.lr:g}")
        del model
    except Exception as e:                           # noqa: BLE001
        check("loss falls on a memorizable slice", False,
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
                        "shares its single context read. -1 measures "
                        "throughput and picks the fastest "
                        "(default: %(default)s)")
    g.add_argument("--lr", default="keep", metavar="RATE",
                   help="'auto' for ChaosGrad's online estimate, a float for "
                        "fixed-rate mode. Default 'keep': a fresh run uses "
                        "auto (zero-config); --resume keeps the checkpoint's mode")
    g.add_argument("--clip", type=float, default=d.clip, metavar="NORM",
                   help="gradient clipping threshold. The default is sized to "
                        "this loss: clipping far below the natural gradient "
                        "norm starves the automatic step-scale estimate "
                        "(default: %(default)s)")
    g.add_argument("--d0", type=float, default=d.d0, metavar="SCALE",
                   help="initial step-scale estimate for --lr auto. The "
                        "estimate only ratchets up, so too low a start never "
                        "catches up on a sparse loss (default: %(default)s)")
    g.add_argument("--grad-ckpt", action="store_true",
                   help="gradient checkpointing: less memory, one extra "
                        "sequential forward per step")
    g.add_argument("--compile", action="store_true",
                   help="torch.compile the forward pass. Three segment reads "
                        "per step, each a loop of small kernels, make this "
                        "launch-bound — so fusing it is the largest speedup "
                        "on offer. Costs a warmup of a minute or two, and "
                        "does not combine with --grad-ckpt")

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
    for name in ("neurons", "n_in", "n_out", "chunk"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive, "
                    f"got {getattr(a, name)}")
    if a.batch == 0:
        p.error("--batch must be positive, or -1 to autotune")
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
        grad_ckpt=a.grad_ckpt, compile=a.compile, clip=a.clip,
        d0=a.d0,
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
        if cfg.batch <= 0:
            cfg = replace(cfg, batch=1)
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
        # Autotuning measures training throughput, which eval does not do.
        if cfg.batch <= 0:
            cfg = replace(cfg, batch=32)
        val_paths = [p.strip() for p in cfg.val.split(",") if p.strip()]
        val_paths = [os.path.join(p, "val.jsonl") if os.path.isdir(p) else p for p in val_paths]
        val_corpus = open_corpus(val_paths, verbose=False)
        val_groups = val_corpus.group_by_context()
        model, trainer = build(cfg)
        load_checkpoint(model, trainer.optimizer, path, device=cfg.device,
                        strict=True)
        print(f"📂 {path}")
        validator = Validator(val_groups, cfg, val_corpus)
        print(f"\n📚 scoring {validator.contexts:,} contexts | "
              f"{validator.answerable:,} answerable + "
              f"{validator.rejectable:,} rejectable questions | "
              f"chance {validator.chance:.2%}")
        m = validator.run(model)
        print(f"\n📊 {fmt(m)}")
        if "bands" in m:
            print(fmt_bands(m))
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
    if "bands" in metrics:
        print(fmt_bands(metrics))
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
