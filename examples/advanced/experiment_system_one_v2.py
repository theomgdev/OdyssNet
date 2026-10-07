"""
System One on OdyssNet: a decision layer whose parameter count does not know
how many questions you will ask, or how many options each one has.

THE SHAPE OF THE THING
----------------------
A classifier reserves a column per class. Ask it about an option it never saw
in training and it has nowhere to put the answer — the question is not hard,
it is unaddressable. That ceiling is in the parameter matrix, not in the data,
and it is the reason a 150-way intent head is a classifier rather than a
decision layer.

This script removes the column. The context, the question and each option are
all just bytes entering the same recurrent core, and every answer is read as
ONE scalar from a single-row decoder:

    [S] "cancel my flight to denver"        -> h_s        (context, read once)
     |
     +--[Q] "what does the user want?"      -> h_sq       (context + question)
     |   |
     |   +--[O] "cancel a reservation"      -> scalar     rewind to h_sq
     |   +--[O] "book a flight"             -> scalar     rewind to h_sq
     |   +--[O] "check the weather"         -> scalar     rewind to h_sq
     |        (K options, K scalars, softmax across them -> `choice`)
     |
     +--[Q] "can this assistant do that?"   -> h_sq2      rewind to h_s
         +--[O] "yes"                       -> scalar     (sigmoid -> `noul`)

Three consequences, and they are the whole point:

  * The option set is *runtime data*. A 150-option question and a 4-option
    question run through the same weights, and an option the model has never
    seen is scored by reading its text. Zero-shot over options is not a trick
    here, it is the default.
  * h_s is a fixed-size vector — `neurons` floats, independent of how long the
    context was. A transformer's KV cache grows with the context; this does
    not.
  * Adding a question costs a question segment, not a retrain. There is no
    schema to widen.

WHERE THE READOUT SITS
----------------------
The sketch above reads the option last, which is the arrangement that pays
the amortisation in full: the context is in the shared prefix, so it is read
once per context no matter how many questions and options follow.

Measured, that arrangement is also the weakest at discriminating. On the
intent question at 900 steps and a matched seed:

    context last   [Q] -> [O] -> [S]    27.50%
    question last  [S] -> [O] -> [Q]    22.08%
    option last    [S] -> [Q] -> [O]    20.00%     (chance ~18.2%)

The reason is temporal, not representational: the scalar is read at the end
of the final segment, and whatever must be *compared against* has to still be
present in the state at that moment. Reading the option last leaves the
context ~50 steps in the past, faded through a chaotic core.

So `--order` is a real trade, not a tuning knob:

    --order ctx_last   best discrimination; the context is re-read per option
    --order opt_last   context read once, state reusable across questions
    --order opt_last --echo-bytes N   replay N context bytes after the option,
                                      buying freshness back cheaply

`ask_stored` — a second question against a stored state — therefore requires
`opt_last` by construction, and `--mode ask` falls back to the full read for
the other orders. `--sweep order` is the arm.

OOS AND CONFIDENCE ARE NOT HEADS HERE
-------------------------------------
Both fall out of the same scalar, which is why they get *better* rather than
being dropped:

  * Out-of-scope is "every option scored low". Each option is trained with a
    per-option BCE against its own correctness, so the scalar is absolute and
    not just a rank. A query with no good option therefore reads as a flat low
    distribution — no dedicated head to drown in the 150:1 class imbalance
    that a separate OOS output has to fight. CLINC150's own out-of-scope split
    becomes ordinary training data: target zero on every option. An explicit
    `none of these` option is also available (`--none-option`), which turns
    abstention into something selectable rather than a threshold.
  * Confidence comes from the distribution the model already produced:
    `max(p)` (calibrated by the same BCE), the top-two margin, and the
    entropy. A measurement on the previous iteration of this experiment is why
    there is no learned confidence head: a 5-level head reached ECE 0.101
    while plain max-softmax on the same run reached 0.033. The head was
    spending parameters to be worse. A *learned* confidence signal is still
    available, but as a question — `is the evidence enough?` — asked against
    the stored h_s for the price of one segment.

WHAT WOULD FALSIFY THE DESIGN
-----------------------------
If the model ignores the question segment it collapses back into a classifier
that happens to be fed option text, and every claim above is void. Two things
guard that, both reported by `--mode eval`:

  * Every question type is trained from several paraphrases and evaluated on
    held-out ones (`--paraphrase-holdout`), so a model that pattern-matches a
    fixed wording scores at chance there.
  * `--holdout-intents N` removes N intents from training entirely. Their
    options appear for the first time at test, which is the zero-shot number.

Usage
-----
    python experiment_system_one_v2.py --mode smoke
    python experiment_system_one_v2.py --mode train --minutes 20
    python experiment_system_one_v2.py --mode eval  --tag base
    python experiment_system_one_v2.py --mode ask   --tag base \
        --context "cancel my flight to denver" \
        --question "what does the user want to do?" \
        --option "cancel a reservation" --option "book a flight"

Data: CLINC150 (Larson et al., EMNLP 2019), cached under data/clinc150/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from odyssnet import OdyssNet, ChaosGrad  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "clinc150"
CKPT = HERE / "ckpt"

DATA_URL = ("https://raw.githubusercontent.com/clinc/oos-eval/master/"
            "data/data_full.json")
DOMAIN_URL = ("https://raw.githubusercontent.com/clinc/oos-eval/master/"
              "data/domains.json")

# ---------------------------------------------------------------------------
# Byte protocol
#
# Bytes 0-3 are reserved as segment markers, so a byte value of 1 in the
# stream is never ambiguous with a marker: text is shifted up by OFFSET. The
# marker is what tells the core which *kind* of information is arriving, and it
# is the only structure the protocol imposes.
# ---------------------------------------------------------------------------
PAD_ID = 0
SEG_STATE = 1      # [S] context follows
SEG_QUESTION = 2   # [Q] question follows
SEG_OPTION = 3     # [O] option follows
OFFSET = 4         # text bytes land at 4..259
VOCAB = 256 + OFFSET


def encode(text, max_bytes, marker):
    """`marker` + utf-8 bytes of `text`, right-padded, as int64 ids."""
    raw = text.encode("utf-8")[: max_bytes - 1]
    ids = [marker] + [b + OFFSET for b in raw]
    ids += [PAD_ID] * (max_bytes - len(ids))
    return ids


def encode_batch(texts, max_bytes, marker):
    return np.asarray([encode(t, max_bytes, marker) for t in texts],
                      dtype=np.int64)


# ---------------------------------------------------------------------------
# Question bank
#
# Every question type carries several paraphrases. Training samples from the
# first `1 - holdout` of them and evaluation can use only the rest, which is
# what makes "does it read the question at all" a measurable thing rather than
# an assumption. Options are *text*, so a question type is a way of generating
# (question, options, correct) triples — not a tensor shape.
# ---------------------------------------------------------------------------
QUESTIONS = {
    "intent": [
        "what does the user want to do?",
        "which action is the user asking for?",
        "identify the user's intent",
        "what is the request about?",
        "which task should be run for this message?",
        "pick the intent of this query",
    ],
    "domain": [
        "which domain does this belong to?",
        "what area is this request in?",
        "classify the topic of this message",
        "which category covers this query?",
        "what kind of service is being asked for?",
    ],
    "can_handle": [
        "can this assistant handle that request?",
        "is this query something the system supports?",
        "is this request in scope?",
        "does the assistant have a skill for this?",
    ],
    "is_about": [
        "is the user asking about {}?",
        "does this query concern {}?",
        "is this request related to {}?",
        "would {} answer this message?",
    ],
    "enough": [
        "is the evidence enough to answer confidently?",
        "is this message clear enough to act on?",
        "can you answer this without guessing?",
    ],
}

YES_NO = ["yes", "no"]
NONE_OPTION = "none of these"


def pick_phrasing(kind, rng, holdout, held_out_side=False):
    """
    A paraphrase for `kind`. `holdout` is the fraction reserved for evaluation;
    `held_out_side` selects which pool to draw from, so the same function
    serves the trainer and the held-out probe.
    """
    pool = QUESTIONS[kind]
    cut = max(1, int(round(len(pool) * (1.0 - holdout))))
    pool = pool[cut:] if held_out_side else pool[:cut]
    if not pool:                      # holdout too large for a short bank
        pool = QUESTIONS[kind]
    return pool[rng.randrange(len(pool))]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Cfg:
    # --- core ---
    neurons: int = 192
    n_in: int = 96
    n_out: int = 96
    # `enc_dec_act` stays identity so the single-row readout is unbounded: the
    # scalar is a logit, and squashing it before the sigmoid would cost range.
    activation: tuple = ("none", "gelu_tanh", "tanh")
    gates: tuple = ("none", "none", "identity")
    hebb_type: str | None = None
    attn_heads: int | None = None
    attn_window: int = 256
    grad_ckpt: bool = False

    # --- protocol ---
    ctx_bytes: int = 48       # context segment length
    q_bytes: int = 40         # question segment length
    opt_bytes: int = 28       # option segment length
    deliberate: int = 2       # free steps after each segment (no input)
    # Which segment arrives last, i.e. what is freshest in the state when the
    # scalar is read. Measured on the intent question, 900 steps, seed 42:
    # `ctx_last` 27.50%, `q_last` 22.08%, `opt_last` 20.00% (chance 18.2%) —
    # the thing an option must be *compared against* wants to be the recent
    # signal, not the one that entered 50 steps earlier.
    #
    # `ctx_last` wins on accuracy but re-reads the context per option, which
    # gives up the amortisation the design exists for. `opt_last` keeps the
    # context in the shared prefix, and replaying a short echo of it after
    # each option buys most of the accuracy back: echo8 reached 25.42% at the
    # same budget. That combination is the default — reusable stored state,
    # near-`ctx_last` discrimination. `--sweep order` is the arm.
    order: str = "opt_last"   # ctx_last | q_last | opt_last
    echo_bytes: int = 8       # with opt_last: replay this many context bytes

    # --- questions ---
    # Weights over question kinds; a kind at 0 never appears. `is_about` and
    # `enough` are the noul-shaped ones, and `enough` is the learned
    # confidence signal that replaces the old head.
    mix: tuple = (("intent", 4.0), ("domain", 2.0), ("can_handle", 2.0),
                  ("is_about", 1.5), ("enough", 1.0))
    k_min: int = 3            # options per choice question, sampled per example
    k_max: int = 12
    none_option: bool = True  # include an explicit abstention option
    paraphrase_holdout: float = 0.34
    holdout_intents: int = 20  # intents never trained on -> zero-shot set

    # --- optimization ---
    batch: int = 24           # contexts per step (each spawns q_per_ctx reads)
    q_per_ctx: int = 2        # questions asked per stored context
    # A pinned rate: measured on the previous iteration, ChaosGrad's online
    # estimate overshoots when supervision is one signal per example rather
    # than one per token. `--lr auto` restores the estimator.
    lr: float | None = 1e-3
    grad_persistence: float = 0.0
    oos_frac: float = 0.25    # share of contexts drawn from the OOS split
    w_bce: float = 1.0        # absolute per-option correctness
    w_list: float = 1.0       # relative ranking across the option set

    # --- run ---
    max_steps: int = 4000
    minutes: float = 0.0
    eval_every: int = 500
    eval_contexts: int = 300
    log_every: int = 100
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tag: str = "v2"

    @property
    def io_ids(self):
        """Input and output neuron id ranges, taken off opposite ends."""
        return list(range(self.n_in)), \
            list(range(self.neurons - self.n_out, self.neurons))

    def steps_for(self, n_bytes):
        return n_bytes + self.deliberate


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def _fetch(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {path.name} ...", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r:
        path.write_bytes(r.read())


def humanise(intent):
    """`cancel_reservation` -> `cancel reservation`; the option text itself."""
    return intent.replace("_", " ")


@dataclass
class Corpus:
    train: list          # (text, intent, domain)
    val: list
    test: list
    oos_train: list      # (text, None, None)
    oos_val: list
    oos_test: list
    intents: list        # trained intents
    zero_shot: list      # intents held out of training entirely
    domains: list
    intent_domain: dict

    def options_for(self, kind, include_zero_shot=False):
        if kind == "domain":
            return self.domains
        pool = list(self.intents)
        if include_zero_shot:
            pool += self.zero_shot
        return pool


def load_corpus(cfg, verbose=True):
    data_p, dom_p = DATA / "data_full.json", DATA / "domains.json"
    _fetch(DATA_URL, data_p)
    _fetch(DOMAIN_URL, dom_p)

    raw = json.loads(data_p.read_text(encoding="utf-8"))
    dom_raw = json.loads(dom_p.read_text(encoding="utf-8"))

    intent_domain = {}
    for dom, members in dom_raw.items():
        for it in members:
            intent_domain[it] = dom
    domains = sorted(dom_raw.keys())
    all_intents = sorted(intent_domain.keys())

    # Hold intents out *by name*, so their option text is unseen at test.
    rng = random.Random(cfg.seed)
    shuffled = all_intents[:]
    rng.shuffle(shuffled)
    zero_shot = sorted(shuffled[: cfg.holdout_intents])
    trained = sorted(shuffled[cfg.holdout_intents:])
    zs_set = set(zero_shot)

    def rows(key, keep_zero_shot):
        out = []
        for text, intent in raw[key]:
            if (intent in zs_set) != keep_zero_shot:
                continue
            out.append((text, intent, intent_domain[intent]))
        return out

    def oos_rows(key):
        return [(text, None, None) for text, _ in raw.get(key, [])]

    corpus = Corpus(
        train=rows("train", False),
        val=rows("val", False),
        test=rows("test", False),
        oos_train=oos_rows("oos_train"),
        oos_val=oos_rows("oos_val"),
        oos_test=oos_rows("oos_test"),
        intents=trained,
        zero_shot=zero_shot,
        domains=domains,
        intent_domain=intent_domain,
    )
    # Test rows whose intent was never trained: the zero-shot probe.
    corpus.zs_test = rows("test", True)

    if verbose:
        print(f"  CLINC150: {len(corpus.train)} train / {len(corpus.val)} val "
              f"/ {len(corpus.test)} test")
        print(f"  intents: {len(trained)} trained, {len(zero_shot)} held out "
              f"(zero-shot test rows: {len(corpus.zs_test)})")
        print(f"  oos: {len(corpus.oos_train)} train / {len(corpus.oos_val)} "
              f"val / {len(corpus.oos_test)} test")
    return corpus


# ---------------------------------------------------------------------------
# Reads
#
# A Read is one (context, question, options, correct) tuple. Building them is
# pure data work — no tensor shape depends on the option count, which is the
# property the whole design rests on.
# ---------------------------------------------------------------------------
@dataclass
class Read:
    kind: str
    question: str
    options: list        # list[str]
    correct: int         # index into options, or -1 when none is correct


def _sample_k(cfg, rng):
    return rng.randint(cfg.k_min, cfg.k_max)


def build_read(cfg, corpus, row, kind, rng, held_out_side=False,
               option_pool=None):
    """One Read for `row` = (text, intent, domain); intent None means OOS."""
    _, intent, domain = row
    in_scope = intent is not None

    if kind == "intent":
        pool = option_pool or corpus.intents
        k = _sample_k(cfg, rng)
        if in_scope and intent in pool:
            wrong = [o for o in pool if o != intent]
            opts = rng.sample(wrong, min(k - 1, len(wrong))) + [intent]
        else:
            opts = rng.sample(pool, min(k, len(pool)))
        rng.shuffle(opts)
        texts = [humanise(o) for o in opts]
        if cfg.none_option:
            texts.append(NONE_OPTION)
        correct = (texts.index(humanise(intent))
                   if in_scope and intent in opts
                   else (len(texts) - 1 if cfg.none_option else -1))
        return Read(kind, pick_phrasing("intent", rng, cfg.paraphrase_holdout,
                                        held_out_side), texts, correct)

    if kind == "domain":
        k = _sample_k(cfg, rng)
        pool = corpus.domains
        if in_scope:
            wrong = [d for d in pool if d != domain]
            opts = rng.sample(wrong, min(k - 1, len(wrong))) + [domain]
        else:
            opts = rng.sample(pool, min(k, len(pool)))
        rng.shuffle(opts)
        texts = [humanise(o) for o in opts]
        if cfg.none_option:
            texts.append(NONE_OPTION)
        correct = (texts.index(humanise(domain)) if in_scope
                   else (len(texts) - 1 if cfg.none_option else -1))
        return Read(kind, pick_phrasing("domain", rng, cfg.paraphrase_holdout,
                                        held_out_side), texts, correct)

    if kind == "can_handle":
        # noul shaped as a two-option choice, so one code path serves both.
        return Read(kind, pick_phrasing("can_handle", rng,
                                        cfg.paraphrase_holdout, held_out_side),
                    list(YES_NO), 0 if in_scope else 1)

    if kind == "is_about":
        pool = option_pool or corpus.intents
        probe = intent if (in_scope and rng.random() < 0.5) else \
            rng.choice([o for o in pool if o != intent] or pool)
        tmpl = pick_phrasing("is_about", rng, cfg.paraphrase_holdout,
                             held_out_side)
        return Read(kind, tmpl.format(humanise(probe)), list(YES_NO),
                    0 if (in_scope and probe == intent) else 1)

    if kind == "enough":
        # The learned confidence signal: in-scope queries are answerable,
        # out-of-scope ones are not. This is the replacement for the head that
        # measured worse than max-softmax.
        return Read(kind, pick_phrasing("enough", rng, cfg.paraphrase_holdout,
                                        held_out_side), list(YES_NO),
                    0 if in_scope else 1)

    raise ValueError(f"unknown question kind {kind!r}")


class ReadSampler:
    """Draws (row, [Read, ...]) pairs — a context plus the questions on it."""

    def __init__(self, cfg, corpus, split, oos_split, seed=0):
        self.cfg, self.corpus = cfg, corpus
        self.split, self.oos = split, oos_split
        self.rng = random.Random(seed)
        self.kinds = [k for k, w in cfg.mix if w > 0]
        self.weights = [w for _, w in cfg.mix if w > 0]

    def row(self):
        if self.oos and self.rng.random() < self.cfg.oos_frac:
            return self.oos[self.rng.randrange(len(self.oos))]
        return self.split[self.rng.randrange(len(self.split))]

    def batch(self, n, q_per_ctx=None):
        q_per_ctx = q_per_ctx or self.cfg.q_per_ctx
        out = []
        for _ in range(n):
            row = self.row()
            kinds = self.rng.choices(self.kinds, self.weights, k=q_per_ctx)
            out.append((row, [build_read(self.cfg, self.corpus, row, k,
                                         self.rng) for k in kinds]))
        return out


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build(cfg):
    in_ids, out_ids = cfg.io_ids
    model = OdyssNet(
        num_neurons=cfg.neurons,
        input_ids=in_ids,
        output_ids=out_ids,
        # One output column. The decoder cannot encode "which class" — only
        # "how well does what I just read fit what I was asked".
        vocab_size=(VOCAB, 1),
        vocab_mode="hybrid",
        pulse_mode=True,
        activation=list(cfg.activation),
        gate=list(cfg.gates),
        hebb_type=cfg.hebb_type,
        attn_heads=cfg.attn_heads,
        attn_window=cfg.attn_window,
        gradient_checkpointing=cfg.grad_ckpt,
        device=cfg.device,
    )
    trainer_lr = cfg.lr
    opt = (ChaosGrad.from_model(model, lr=trainer_lr)
           if trainer_lr is not None else ChaosGrad.from_model(model))
    return model, opt


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# The rewind protocol
#
# `forward` returns `h_t` inside the graph (only `self.state` is detached), so
# a stored state can be handed back in as `current_state` and still carry
# gradient. That is what makes "read the context once, branch per option" a
# training-time mechanism and not just an inference trick.
# ---------------------------------------------------------------------------
def _attn_mark(model):
    return None if model.attn is None else model.attn.mark()


def _attn_rewind(model, mark):
    """
    Drop cache entries written after `mark`, so option i+1 cannot attend to
    what option i wrote. Without this, scores depend on option order — the
    smoke test checks exactly that.
    """
    if model.attn is None or mark is None:
        return
    model.attn.rewind(mark)


def read_segment(model, ids, cfg, state=None):
    """Run one segment and return (readout_logits, final_state)."""
    steps = cfg.steps_for(ids.shape[1])
    out, h = model(ids, steps=steps, current_state=state,
                   return_sequence=False)
    # (B, 1, 1) -> (B,): the single decoder column at the final step.
    return out[:, -1, 0], h


def score_options(model, cfg, h_prefix, option_ids, tail=()):
    """
    Score every option against the same prefix state.

    `option_ids` is (B, K, L); each k runs from a rewind of `h_prefix`, so the
    options are mutually invisible and K is free to change between calls.
    `tail` is the segments replayed after each option — which is how the
    thing being compared against ends up fresh at the readout.
    """
    k = option_ids.shape[1]
    mark = _attn_mark(model)
    scores = []
    for i in range(k):
        _attn_rewind(model, mark)
        s, h = read_segment(model, option_ids[:, i], cfg, state=h_prefix)
        for seg in tail:
            s, h = read_segment(model, seg, cfg, state=h)
        scores.append(s)
    _attn_rewind(model, mark)
    return torch.stack(scores, dim=1)          # (B, K)


def run_reads(model, cfg, ctx_ids, q_ids, opt_ids):
    """
    The full protocol for one batch of (context, question, options).

    The segment order is `cfg.order`, and what it decides is which signal is
    freshest when the scalar is read:

      ctx_last  [Q]question -> [O]option -> [S]context   (measured best)
      q_last    [S]context  -> [O]option -> [Q]question
      opt_last  [S]context  -> [Q]question -> [O]option  (+ optional echo)

    Everything before the option is the shared prefix, run once; every option
    branches from a rewind of it, so options stay mutually invisible and K is
    free to change between calls.

    Returns (scores, prefix_state). The prefix state is reusable only in
    `opt_last`, where the context is part of the prefix — which is the
    trade-off the order knob exposes: freshness costs the re-read.
    """
    model.reset_state(batch_size=ctx_ids.shape[0])
    if model.attn is not None:
        model.attn.reset()

    if cfg.order == "ctx_last":
        _, h = read_segment(model, q_ids, cfg)
        tail = [ctx_ids]
    elif cfg.order == "q_last":
        _, h = read_segment(model, ctx_ids, cfg)
        tail = [q_ids]
    elif cfg.order == "opt_last":
        _, h = read_segment(model, ctx_ids, cfg)
        _, h = read_segment(model, q_ids, cfg, state=h)
        tail = ([ctx_ids[:, : cfg.echo_bytes]] if cfg.echo_bytes else [])
    else:
        raise ValueError(f"unknown order {cfg.order!r}")

    return score_options(model, cfg, h, opt_ids, tail), h


def ask_stored(model, cfg, h_s, q_ids, opt_ids, ctx_ids=None):
    """
    Another question against an already-stored context state.

    Only meaningful when the context is part of the shared prefix, i.e.
    `order='opt_last'`; the other orders read the context per option by
    construction and `run_reads` is the entry point for them.

    `ctx_ids` is needed when `echo_bytes` is set, because the echo replays a
    slice of the context after each option — the stored state alone cannot
    reproduce it, and skipping it would make this path behave differently
    from the one the model trained under.
    """
    if cfg.order != "opt_last":
        raise ValueError(
            f"ask_stored needs order='opt_last' (context in the prefix), "
            f"got {cfg.order!r}")
    tail = []
    if cfg.echo_bytes:
        if ctx_ids is None:
            raise ValueError(
                f"echo_bytes={cfg.echo_bytes} needs ctx_ids to replay")
        tail = [ctx_ids[:, : cfg.echo_bytes]]
    _, h_sq = read_segment(model, q_ids, cfg, state=h_s)
    return score_options(model, cfg, h_sq, opt_ids, tail)


# ---------------------------------------------------------------------------
# Loss
#
# Two terms on the same scalars:
#   bce  — per option, "is this one correct": makes the scalar absolute, which
#          is what lets a flat low distribution mean out-of-scope.
#   list — softmax across the option set, "which is best": the ranking signal.
# A row with no correct option (OOS without `none of these`) contributes only
# the bce term, which is the correct thing to say about it.
# ---------------------------------------------------------------------------
def read_loss(scores, correct, mask, cfg):
    b, k = scores.shape
    tgt = torch.zeros_like(scores)
    has = correct >= 0
    if has.any():
        tgt[has, correct[has]] = 1.0

    valid = mask.float()
    bce = F.binary_cross_entropy_with_logits(scores, tgt, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)

    lst = scores.new_zeros(())
    if has.any():
        s = scores[has].masked_fill(~mask[has], -1e4)
        lst = F.cross_entropy(s, correct[has])

    return cfg.w_bce * bce + cfg.w_list * lst, {"bce": bce.item(),
                                                "list": float(lst)}


# ---------------------------------------------------------------------------
# Batch assembly
#
# Reads inside one step are padded to a common K so they ride one tensor; the
# mask carries the real option counts. K varies between steps by design, so
# the model never sees a fixed option-count geometry.
# ---------------------------------------------------------------------------
def pack(batch, cfg, device):
    rows, reads = [], []
    for row, rds in batch:
        for r in rds:
            rows.append(row)
            reads.append(r)

    k_max = max(len(r.options) for r in reads)
    ctx = encode_batch([r[0] for r in rows], cfg.ctx_bytes, SEG_STATE)
    qs = encode_batch([r.question for r in reads], cfg.q_bytes, SEG_QUESTION)

    opts = np.full((len(reads), k_max, cfg.opt_bytes), PAD_ID, dtype=np.int64)
    mask = np.zeros((len(reads), k_max), dtype=bool)
    for i, r in enumerate(reads):
        enc = encode_batch(r.options, cfg.opt_bytes, SEG_OPTION)
        opts[i, : len(r.options)] = enc
        mask[i, : len(r.options)] = True

    t = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    return (t(ctx), t(qs), t(opts), t(mask),
            torch.tensor([r.correct for r in reads], device=device),
            [r.kind for r in reads])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ece(prob, correct, bins=15):
    if len(prob) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        tot += sel.mean() * abs(prob[sel].mean() - correct[sel].mean())
    return float(tot)


@torch.no_grad()
def evaluate(model, cfg, corpus, split, oos_split, n_contexts,
             held_out_phrasing=False, option_pool=None, seed=1234):
    """
    Accuracy per question kind, plus calibration of the distribution the model
    produced. `held_out_phrasing` asks with paraphrases never trained on, and
    `option_pool` can include the zero-shot intents.
    """
    model.eval()
    rng = random.Random(seed)
    kinds = [k for k, w in cfg.mix if w > 0]

    hit = {k: [0, 0] for k in kinds}
    conf, ok = [], []
    oos_scores, oos_flag = [], []

    rows = []
    n_oos = int(round(n_contexts * cfg.oos_frac)) if oos_split else 0
    for _ in range(n_contexts - n_oos):
        rows.append(split[rng.randrange(len(split))])
    for _ in range(n_oos):
        rows.append(oos_split[rng.randrange(len(oos_split))])

    bs = max(1, cfg.batch)
    for start in range(0, len(rows), bs):
        chunk = rows[start: start + bs]
        batch = []
        for row in chunk:
            rds = [build_read(cfg, corpus, row, k, rng,
                              held_out_side=held_out_phrasing,
                              option_pool=option_pool) for k in kinds]
            batch.append((row, rds))
        ctx, qs, opts, mask, correct, kind_list = pack(batch, cfg, cfg.device)

        scores, _ = run_reads(model, cfg, ctx, qs, opts)
        scores = scores.masked_fill(~mask, -1e4)
        prob = torch.softmax(scores, dim=1)
        pred = prob.argmax(dim=1)

        for i, kind in enumerate(kind_list):
            c = int(correct[i])
            if c < 0:
                continue
            hit[kind][1] += 1
            good = int(pred[i]) == c
            hit[kind][0] += int(good)
            if kind in ("intent", "domain"):
                conf.append(float(prob[i].max()))
                ok.append(float(good))
            if kind == "can_handle":
                # p(yes) against whether the row really is in scope.
                oos_scores.append(float(prob[i, 0]))
                oos_flag.append(1.0 if c == 0 else 0.0)

    model.train()
    out = {}
    for k, (h, n) in hit.items():
        out[f"acc_{k}"] = h / n if n else float("nan")
    if conf:
        c = np.asarray(conf)
        o = np.asarray(ok)
        out["ece"] = _ece(c, o)
        out["brier"] = float(np.mean((c - o) ** 2))
        out["conf_mean"] = float(c.mean())
    if oos_flag:
        s = np.asarray(oos_scores)
        f = np.asarray(oos_flag)
        pred_in = s >= 0.5
        out["oos_recall"] = float((~pred_in[f == 0]).mean()) if (f == 0).any() \
            else float("nan")
        out["in_recall"] = float(pred_in[f == 1].mean()) if (f == 1).any() \
            else float("nan")
    return out


def fmt(m):
    bits = []
    for k in ("intent", "domain", "can_handle", "is_about", "enough"):
        key = f"acc_{k}"
        if key in m and not math.isnan(m[key]):
            bits.append(f"{k} {m[key] * 100:.1f}%")
    if "oos_recall" in m and not math.isnan(m["oos_recall"]):
        bits.append(f"oos-rec {m['oos_recall'] * 100:.1f}%")
    if "ece" in m:
        bits.append(f"ece {m['ece']:.3f}")
    return " | ".join(bits)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
ARCH = ("neurons", "n_in", "n_out", "activation", "gates", "hebb_type",
        "attn_heads", "attn_window", "ctx_bytes", "q_bytes", "opt_bytes",
        "deliberate", "order", "echo_bytes", "holdout_intents",
        "none_option")


def ckpt_path(cfg):
    CKPT.mkdir(parents=True, exist_ok=True)
    return CKPT / f"s1v2_{cfg.tag}.pth"


def save(cfg, model, step, metrics, path):
    torch.save({"model": model.state_dict(), "step": step,
                "metrics": metrics,
                "arch": {k: getattr(cfg, k) for k in ARCH},
                "params": n_params(model)}, path)


def load_for_eval(cfg):
    path = ckpt_path(cfg)
    if not path.exists():
        print(f"no checkpoint at {path}", file=sys.stderr)
        return None, None
    blob = torch.load(path, map_location=cfg.device, weights_only=False)
    cfg = replace(cfg, **{k: v for k, v in blob["arch"].items()})
    model, _ = build(cfg)
    model.load_state_dict(blob["model"])
    return cfg, model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg, corpus, quiet=False, save_ckpt=True, budget_sec=0.0):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model, opt = build(cfg)
    sampler = ReadSampler(cfg, corpus, corpus.train, corpus.oos_train,
                          seed=cfg.seed)

    if not quiet:
        print(f"\n  params {n_params(model):,} | neurons {cfg.neurons} | "
              f"device {cfg.device}")
        print(f"  protocol: [S]{cfg.ctx_bytes} [Q]{cfg.q_bytes} "
              f"[O]{cfg.opt_bytes} +{cfg.deliberate} free steps")
        print(f"  options per question: {cfg.k_min}-{cfg.k_max}"
              f"{' + none' if cfg.none_option else ''}")

    t0 = time.time()
    best, best_metrics = -1.0, {}
    run = {"bce": 0.0, "list": 0.0, "n": 0}

    for step in range(1, cfg.max_steps + 1):
        batch = sampler.batch(cfg.batch)
        ctx, qs, opts, mask, correct, _ = pack(batch, cfg, cfg.device)

        scores, _ = run_reads(model, cfg, ctx, qs, opts)
        loss, terms = read_loss(scores, correct, mask, cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if hasattr(opt, "report_loss"):
            opt.report_loss(float(loss))
        opt.step()
        model.detach_state()

        run["bce"] += terms["bce"]
        run["list"] += terms["list"]
        run["n"] += 1

        if not quiet and cfg.log_every and step % cfg.log_every == 0:
            n = max(1, run["n"])
            print(f"  step {step:>5} | bce {run['bce'] / n:.4f} "
                  f"| list {run['list'] / n:.4f} "
                  f"| {time.time() - t0:.0f}s", flush=True)
            run = {"bce": 0.0, "list": 0.0, "n": 0}

        due_eval = cfg.eval_every and step % cfg.eval_every == 0
        out_of_time = budget_sec and (time.time() - t0) > budget_sec
        if due_eval or out_of_time or step == cfg.max_steps:
            m = evaluate(model, cfg, corpus, corpus.val, corpus.oos_val,
                         cfg.eval_contexts)
            if not quiet:
                print(f"    eval @ {step}: {fmt(m)}", flush=True)
            key = m.get("acc_intent", 0.0)
            if not math.isnan(key) and key > best:
                best, best_metrics = key, m
                if save_ckpt:
                    save(cfg, model, step, m, ckpt_path(cfg))
            if out_of_time:
                break

    return model, best_metrics, time.time() - t0


# ---------------------------------------------------------------------------
# ask: the caller's view
# ---------------------------------------------------------------------------
@torch.no_grad()
def ask(model, cfg, context, questions):
    """
    `questions` is a list of (question_text, [option, ...]).

    With `order='opt_last'` the context is read once and every question runs
    off the stored state — the amortisation claim in executable form. The
    other orders put the context after the option (measured to score better),
    so there the context is re-read per option and this walks `run_reads`
    instead. Both paths answer any number of questions with the same weights;
    only the cost differs.
    """
    model.eval()
    ctx = torch.from_numpy(
        encode_batch([context], cfg.ctx_bytes, SEG_STATE)).to(cfg.device)

    stored = cfg.order == "opt_last"
    if stored:
        model.reset_state(batch_size=1)
        if model.attn is not None:
            model.attn.reset()
        _, h_s = read_segment(model, ctx, cfg)

    answers = []
    for q, opts in questions:
        q_ids = torch.from_numpy(
            encode_batch([q], cfg.q_bytes, SEG_QUESTION)).to(cfg.device)
        o_ids = torch.from_numpy(
            encode_batch(opts, cfg.opt_bytes, SEG_OPTION)
        ).to(cfg.device).unsqueeze(0)
        if stored:
            s = ask_stored(model, cfg, h_s, q_ids, o_ids, ctx_ids=ctx)[0]
        else:
            s = run_reads(model, cfg, ctx, q_ids, o_ids)[0][0]
        p = torch.softmax(s, dim=0)
        order = torch.argsort(p, descending=True)
        answers.append({
            "question": q,
            "ranked": [(opts[int(i)], float(p[int(i)])) for i in order],
            "abs": {opts[int(i)]: float(torch.sigmoid(s[int(i)]))
                    for i in order},
            "confidence": float(p.max()),
            "margin": float(p[order[0]] - p[order[1]]) if len(opts) > 1 else 1.0,
            "entropy": float(-(p * (p + 1e-9).log()).sum()),
        })
    model.train()
    return answers


def print_answers(answers):
    for a in answers:
        print(f"\n  Q: {a['question']}")
        for name, p in a["ranked"][:6]:
            abs_p = a["abs"][name]
            print(f"     {p * 100:6.2f}%  (abs {abs_p:.2f})  {name}")
        print(f"     ── confidence {a['confidence']:.2f} "
              f"| margin {a['margin']:.2f} | entropy {a['entropy']:.2f}")
        if a["confidence"] < 0.35 or max(a["abs"].values()) < 0.3:
            print("     ── reads as out of scope / not answerable")


def _mean_loss(model, cfg, corpus, n_batches=8, seed=999):
    """
    Mean loss over a few fixed batches, with no training.

    Used by the smoke test's learning check: a pipeline that is wired wrong
    cannot move this number, which is a sharper thing to assert than any
    accuracy threshold on a task whose ranking term is still near chance.
    """
    sampler = ReadSampler(cfg, corpus, corpus.train, corpus.oos_train,
                          seed=seed)
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = sampler.batch(cfg.batch)
            ctx, qs, opts, mask, correct, _ = pack(batch, cfg, cfg.device)
            scores, _ = run_reads(model, cfg, ctx, qs, opts)
            loss, _ = read_loss(scores, correct, mask, cfg)
            total += float(loss)
    if was_training:
        model.train()
    return total / n_batches


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------
def smoke(cfg, corpus):
    print("\n" + "=" * 68)
    print("  SMOKE")
    print("=" * 68)
    ok = True
    base = replace(cfg, neurons=96, n_in=48, n_out=48, ctx_bytes=24,
                   q_bytes=20, opt_bytes=16, batch=6, q_per_ctx=2,
                   k_min=2, k_max=4, max_steps=6, eval_every=0, log_every=0,
                   eval_contexts=24)

    print("\n  → variants build, train and score")
    variants = {
        "plain": {},
        "attn": {"attn_heads": 2},
        "hebb": {"hebb_type": "temporal"},
        "no_none": {"none_option": False},
        "grad_ckpt": {"grad_ckpt": True},
        "wide_k": {"k_min": 6, "k_max": 10},
    }
    for name, over in variants.items():
        try:
            c = replace(base, **over)
            m, _, _ = train(c, corpus, quiet=True, save_ckpt=False)
            r = evaluate(m, c, corpus, corpus.val, corpus.oos_val, 24)
            print(f"  ✓ {name:<10} params {n_params(m):>7,} | {fmt(r)}")
            del m
        except Exception as e:                      # noqa: BLE001
            ok = False
            print(f"  ❌ {name:<10} {type(e).__name__}: {e}")

    # The readout must not know the option count.
    print("\n  → parameter count is independent of K")
    try:
        counts = {}
        for k in (2, 25, 150):
            c = replace(base, k_min=k, k_max=k)
            m, _ = build(c)
            counts[k] = n_params(m)
            del m
        uniq = set(counts.values())
        if len(uniq) == 1:
            print(f"  ✓ K=2/25/150 all {counts[2]:,} params")
        else:
            ok = False
            print(f"  ❌ params vary with K: {counts}")
    except Exception as e:                          # noqa: BLE001
        ok = False
        print(f"  ❌ param independence {type(e).__name__}: {e}")

    # Rewind must make options mutually invisible.
    print("\n  → options do not contaminate each other")
    try:
        for attn in (None, 2):
            c = replace(base, attn_heads=attn)
            m, _ = build(c)
            m.eval()
            with torch.no_grad():
                ctx = torch.from_numpy(encode_batch(
                    ["cancel my flight"], c.ctx_bytes, SEG_STATE)).to(c.device)
                q = torch.from_numpy(encode_batch(
                    ["what is this"], c.q_bytes, SEG_QUESTION)).to(c.device)
                opts = ["cancel reservation", "book flight", "weather"]
                fwd = encode_batch(opts, c.opt_bytes, SEG_OPTION)
                rev = encode_batch(opts[::-1], c.opt_bytes, SEG_OPTION)
                s1, _ = run_reads(m, c, ctx, q,
                                  torch.from_numpy(fwd).to(c.device).unsqueeze(0))
                s2, _ = run_reads(m, c, ctx, q,
                                  torch.from_numpy(rev).to(c.device).unsqueeze(0))
                d = (s1[0] - s2[0].flip(0)).abs().max().item()
            label = "plain" if attn is None else "attn"
            if d < 1e-4:
                print(f"  ✓ {label}: order-independent (max Δ {d:.2e})")
            else:
                ok = False
                print(f"  ❌ {label}: option order changes scores (Δ {d:.2e})")
            del m
    except Exception as e:                          # noqa: BLE001
        ok = False
        print(f"  ❌ contamination check {type(e).__name__}: {e}")

    # A stored state must answer a second question without re-reading.
    # This is the `opt_last` property specifically: the other orders put the
    # context after the option, so there is no context-inclusive prefix to
    # store. Asserting it under the default order would be asserting the
    # wrong thing.
    print("\n  → stored state answers more questions (opt_last)")
    try:
        c = replace(base, order="opt_last")
        m, _ = build(c)
        m.eval()
        with torch.no_grad():
            ctx = torch.from_numpy(encode_batch(
                ["cancel my flight"], c.ctx_bytes, SEG_STATE)).to(c.device)
            m.reset_state(batch_size=1)
            if m.attn is not None:
                m.attn.reset()
            _, h_s = read_segment(m, ctx, c)
            outs = []
            for q, opts in (("what is this", ["cancel", "book"]),
                            ("which domain", ["travel", "banking"]),
                            ("is it clear", ["yes", "no"])):
                qi = torch.from_numpy(encode_batch(
                    [q], c.q_bytes, SEG_QUESTION)).to(c.device)
                oi = torch.from_numpy(encode_batch(
                    opts, c.opt_bytes, SEG_OPTION)).to(c.device).unsqueeze(0)
                outs.append(ask_stored(m, c, h_s, qi, oi, ctx_ids=ctx))
        shapes = [tuple(o.shape) for o in outs]
        print(f"  ✓ 3 questions off one stored context: {shapes}")
        del m
    except Exception as e:                          # noqa: BLE001
        ok = False
        print(f"  ❌ stored-state reuse {type(e).__name__}: {e}")

    # Gradient must flow back through the rewind into the context read.
    print("\n  → gradient reaches the core through a rewind")
    try:
        c = replace(base)
        m, opt = build(c)
        batch = ReadSampler(c, corpus, corpus.train, corpus.oos_train,
                            seed=0).batch(4)
        ctx, qs, opts_t, mask, correct, _ = pack(batch, c, c.device)
        scores, _ = run_reads(m, c, ctx, qs, opts_t)
        loss, _ = read_loss(scores, correct, mask, c)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        named = {"W": m.W, "embed": m.embed.weight,
                 "decoder": m.output_decoder.weight}
        dead = [n for n, p in named.items()
                if p.grad is None or p.grad.abs().max().item() == 0.0]
        if dead:
            ok = False
            print(f"  ❌ no gradient into: {', '.join(dead)}")
        else:
            for n, p in named.items():
                print(f"  ✓ {n:<8} |grad| max {p.grad.abs().max().item():.3e}")
        del m
    except Exception as e:                          # noqa: BLE001
        ok = False
        print(f"  ❌ gradient check {type(e).__name__}: {e}")

    # Zero-shot options must at least run.
    print("\n  → unseen options are scoreable")
    try:
        c = replace(base)
        m, _, _ = train(c, corpus, quiet=True, save_ckpt=False)
        pool = corpus.intents + corpus.zero_shot
        r = evaluate(m, c, corpus, corpus.zs_test, None, 24,
                     option_pool=pool)
        print(f"  ✓ zero-shot rows scored | {fmt(r)}")
        del m
    except Exception as e:                          # noqa: BLE001
        ok = False
        print(f"  ❌ zero-shot {type(e).__name__}: {e}")

    # Checkpoint round-trip.
    print("\n  → checkpoint round-trip")
    try:
        c = replace(base, tag="_smoke")
        m, _, _ = train(c, corpus, quiet=True, save_ckpt=False)
        p = CKPT / "s1v2__smoke.pth"
        save(c, m, 1, {}, p)
        c2, m2 = load_for_eval(replace(cfg, tag="_smoke"))
        m.eval()
        m2.eval()
        with torch.no_grad():
            ctx = torch.from_numpy(encode_batch(
                ["test query"], c.ctx_bytes, SEG_STATE)).to(c.device)
            q = torch.from_numpy(encode_batch(
                ["what is this"], c.q_bytes, SEG_QUESTION)).to(c.device)
            o = torch.from_numpy(encode_batch(
                ["a", "b"], c.opt_bytes, SEG_OPTION)
            ).to(c.device).unsqueeze(0)
            a, _ = run_reads(m, c, ctx, q, o)
            b, _ = run_reads(m2, c2, ctx, q, o)
        d = (a - b).abs().max().item()
        if d == 0.0:
            print("  ✓ bit-exact after reload")
        else:
            ok = False
            print(f"  ❌ reload differs by {d:.3e}")
        p.unlink(missing_ok=True)
        del m, m2
    except Exception as e:                          # noqa: BLE001
        ok = False
        print(f"  ❌ round-trip {type(e).__name__}: {e}")

    # Learning check.
    #
    # This gates the plumbing, not the accuracy. Run twice, the binary
    # questions came out at 52.0% and 49.2% — straddling any threshold put
    # there, so a threshold would be testing the seed. What a broken pipeline
    # cannot do is move the loss at all, so that is what is asserted; the
    # accuracies are printed for the reader.
    #
    # Honest status: K-way discrimination sits just off chance (19.5-19.6%
    # against 18.18% at 2000 steps) while the binary questions reach the
    # high 40s-50s from a 25% start. The option-order and echo findings came
    # out of chasing exactly that gap and did not close it — the ranking term
    # is the open problem, not a failing test.
    print("\n  → learning check")
    try:
        c = replace(cfg, neurons=128, n_in=64, n_out=64, ctx_bytes=32,
                    q_bytes=28, opt_bytes=20, batch=32, q_per_ctx=2,
                    k_min=3, k_max=6, max_steps=2000, eval_every=0,
                    log_every=0, eval_contexts=240)
        torch.manual_seed(c.seed)
        m0, _ = build(c)
        start = _mean_loss(m0, c, corpus, n_batches=8)
        del m0
        m, _, dt = train(c, corpus, quiet=True, save_ckpt=False)
        end = _mean_loss(m, c, corpus, n_batches=8)
        r = evaluate(m, c, corpus, corpus.val, corpus.oos_val, 240)
        moved = end < start * 0.97
        chance = 1.0 / ((c.k_min + c.k_max) / 2 + (1 if c.none_option else 0))
        print(f"  {'✓' if moved else '❌'} loss {start:.4f} → {end:.4f} "
              f"({dt:.0f}s)")
        print(f"     {fmt(r)}")
        print(f"     k-way {r.get('acc_intent', 0.0) * 100:.2f}% vs chance "
              f"{chance * 100:.2f}% — the open problem")
        ok = ok and moved
        del m
    except Exception as e:                          # noqa: BLE001
        ok = False
        print(f"  ❌ learning {type(e).__name__}: {e}")

    print("\n" + "=" * 68)
    print("  🎉 SMOKE PASSED" if ok else "  ⚠️  SMOKE FAILED")
    print("=" * 68)
    return ok


# ---------------------------------------------------------------------------
# Sweeps: one knob at a time, same seed, same budget
# ---------------------------------------------------------------------------
SWEEPS = {
    # Where the readout sits relative to what it must compare against. This
    # is the knob that mattered most: `opt_last` scores the option while the
    # context is ~50 steps back, and a chaotic core has faded it.
    "order": [
        ("ctx_last", {"order": "ctx_last"}),
        ("q_last", {"order": "q_last"}),
        ("opt_last", {"order": "opt_last"}),
        ("opt_last+echo8", {"order": "opt_last", "echo_bytes": 8}),
        ("opt_last+echo16", {"order": "opt_last", "echo_bytes": 16}),
    ],
    # Does the option set's size change how well it learns to discriminate?
    "k": [
        ("k2-4", {"k_min": 2, "k_max": 4}),
        ("k3-8", {"k_min": 3, "k_max": 8}),
        ("k6-16", {"k_min": 6, "k_max": 16}),
    ],
    # Free steps after each segment: thinking time with no new parameters.
    "deliberate": [
        ("0", {"deliberate": 0}),
        ("2", {"deliberate": 2}),
        ("6", {"deliberate": 6}),
    ],
    # The two loss terms. `bce` makes the scalar absolute (what lets a flat
    # low distribution mean out-of-scope); `list` is the ranking signal.
    "loss": [
        ("both", {"w_bce": 1.0, "w_list": 1.0}),
        ("list only", {"w_bce": 0.0, "w_list": 1.0}),
        ("bce only", {"w_bce": 1.0, "w_list": 0.0}),
        ("bce heavy", {"w_bce": 3.0, "w_list": 1.0}),
    ],
    # Recurrence extras, off by default.
    "mech": [
        ("plain", {}),
        ("attn2", {"attn_heads": 2}),
        ("hebb", {"hebb_type": "temporal"}),
    ],
    "lr": [
        ("auto", {"lr": None}),
        ("3e-3", {"lr": 3e-3}),
        ("1e-3", {"lr": 1e-3}),
        ("3e-4", {"lr": 3e-4}),
    ],
}


def sweep(cfg, corpus, name, minutes, steps=None):
    arms = SWEEPS.get(name)
    if arms is None:
        print(f"unknown sweep {name!r}; have: {', '.join(SWEEPS)}",
              file=sys.stderr)
        sys.exit(2)

    per = (minutes * 60.0 / len(arms)) if minutes else 0.0
    print("\n" + "=" * 68)
    print(f"  SWEEP {name} — {len(arms)} arms"
          f"{f', {per:.0f}s each' if per else ''}, seed {cfg.seed}")
    print("=" * 68)

    rows = []
    for label, over in arms:
        c = replace(cfg, tag=f"{cfg.tag}_{name}", **over)
        if steps:
            c = replace(c, max_steps=steps)
        m, met, dt = train(c, corpus, quiet=True, save_ckpt=False,
                           budget_sec=per)
        rows.append((label, met, dt))
        print(f"  {label:<18} {fmt(met)} | {dt:.0f}s", flush=True)
        del m
        _empty_cache(cfg.device)

    print("\n  " + "-" * 60)
    best = max(rows, key=lambda r: r[1].get("acc_intent", 0.0))
    print(f"  best on intent: {best[0]} "
          f"({best[1].get('acc_intent', 0.0) * 100:.2f}%)")
    return rows


def _empty_cache(device):
    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Frontier: what does parameter count buy
# ---------------------------------------------------------------------------
def frontier(cfg, corpus, minutes, sizes=None):
    sizes = sizes or [64, 96, 128, 192, 256]
    per = (minutes * 60.0 / len(sizes)) if minutes else 0.0
    print("\n" + "=" * 68)
    print(f"  FRONTIER — {len(sizes)} sizes"
          f"{f', {per:.0f}s each' if per else ''}")
    print("=" * 68)

    rows = []
    for n in sizes:
        c = replace(cfg, neurons=n, n_in=max(32, n // 2),
                    n_out=max(32, n // 2), tag=f"{cfg.tag}_n{n}")
        m, met, dt = train(c, corpus, quiet=True, save_ckpt=False,
                           budget_sec=per)
        p = n_params(m)
        rows.append((n, p, met, dt))
        print(f"  n={n:<4} params {p:>8,} | {fmt(met)} | {dt:.0f}s",
              flush=True)
        del m
        if cfg.device == "cuda":
            torch.cuda.empty_cache()

    print("\n  " + "-" * 64)
    print(f"  {'neurons':>8} {'params':>10} {'intent':>8} {'oos-rec':>8} "
          f"{'acc/kparam':>11}")
    for n, p, met, _ in rows:
        acc = met.get("acc_intent", float("nan")) * 100
        oos = met.get("oos_recall", float("nan")) * 100
        print(f"  {n:>8} {p:>10,} {acc:>7.1f}% {oos:>7.1f}% "
              f"{acc / (p / 1000):>11.2f}")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    d = Cfg()
    p = argparse.ArgumentParser(
        prog="experiment_system_one_v2.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="System One decisions on OdyssNet: context stored once, "
                    "questions and options as runtime text.",
        epilog="""examples:
  # everything works end to end
  %(prog)s --mode smoke

  # a real run
  %(prog)s --mode train --tag base --neurons 192 --minutes 20

  # held-out paraphrases and unseen options
  %(prog)s --mode eval --tag base

  # ask it something
  %(prog)s --mode ask --tag base \\
      --context "cancel my flight to denver" \\
      --question "what does the user want to do?" \\
      --option "cancel a reservation" --option "book a flight"
""")
    p.add_argument("--mode", default="train",
                   choices=["train", "eval", "ask", "smoke", "frontier",
                            "sweep"])
    p.add_argument("--tag", default=d.tag)
    p.add_argument("--device", default=d.device, choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=d.seed)

    g = p.add_argument_group("core")
    g.add_argument("--neurons", type=int, default=d.neurons)
    g.add_argument("--n-in", type=int, default=d.n_in)
    g.add_argument("--n-out", type=int, default=d.n_out)
    g.add_argument("--activation", default=",".join(d.activation),
                   help="encoder/decoder, core step, memory")
    g.add_argument("--gates", default=",".join(d.gates))
    g.add_argument("--hebb", default=d.hebb_type,
                   choices=[None, "temporal", "spatial", "both"])
    g.add_argument("--attn-heads", type=int, default=d.attn_heads)
    g.add_argument("--attn-window", type=int, default=d.attn_window)
    g.add_argument("--grad-ckpt", action="store_true", default=d.grad_ckpt)

    g = p.add_argument_group("protocol")
    g.add_argument("--ctx-bytes", type=int, default=d.ctx_bytes)
    g.add_argument("--q-bytes", type=int, default=d.q_bytes)
    g.add_argument("--opt-bytes", type=int, default=d.opt_bytes)
    g.add_argument("--deliberate", type=int, default=d.deliberate,
                   help="free steps after each segment, no input injected")
    g.add_argument("--order", default=d.order,
                   choices=["ctx_last", "q_last", "opt_last"],
                   help="which segment arrives last, i.e. what is freshest "
                        "when the scalar is read; measured best is ctx_last, "
                        "while opt_last is the one that reads the context "
                        "once and reuses the stored state")
    g.add_argument("--echo-bytes", type=int, default=d.echo_bytes,
                   help="with --order opt_last, replay this many context "
                        "bytes after each option to buy freshness back")

    g = p.add_argument_group("questions")
    g.add_argument("--k-min", type=int, default=d.k_min)
    g.add_argument("--k-max", type=int, default=d.k_max)
    g.add_argument("--no-none-option", dest="none_option",
                   action="store_false", default=d.none_option,
                   help="drop the explicit 'none of these' option")
    g.add_argument("--q-per-ctx", type=int, default=d.q_per_ctx,
                   help="questions asked per stored context")
    g.add_argument("--paraphrase-holdout", type=float,
                   default=d.paraphrase_holdout)
    g.add_argument("--holdout-intents", type=int, default=d.holdout_intents,
                   help="intents never trained on; their options are the "
                        "zero-shot set")
    g.add_argument("--oos-frac", type=float, default=d.oos_frac)

    g = p.add_argument_group("optimization")
    g.add_argument("--batch", type=int, default=d.batch)
    g.add_argument("--lr", default="keep",
                   help="'keep' uses the pinned default, 'auto' hands the "
                        "rate to ChaosGrad, or give a float")
    g.add_argument("--w-bce", type=float, default=d.w_bce)
    g.add_argument("--w-list", type=float, default=d.w_list)
    g.add_argument("--max-steps", type=int, default=d.max_steps)
    g.add_argument("--minutes", type=float, default=d.minutes)
    g.add_argument("--eval-every", type=int, default=d.eval_every)
    g.add_argument("--eval-contexts", type=int, default=d.eval_contexts)
    g.add_argument("--log-every", type=int, default=d.log_every)

    g = p.add_argument_group("ask")
    g.add_argument("--context", default="")
    g.add_argument("--question", action="append", default=[])
    g.add_argument("--option", action="append", default=[])

    g = p.add_argument_group("frontier / sweep")
    g.add_argument("--sizes", default="",
                   help="comma-separated neuron counts")
    g.add_argument("--sweep", default="order",
                   help=f"which knob: {', '.join(SWEEPS)}")

    a = p.parse_args()
    a.activation = tuple(s.strip() for s in a.activation.split(","))
    a.gates = tuple(s.strip() for s in a.gates.split(","))
    return a


def cfg_from_args(a):
    d = Cfg()
    return Cfg(
        neurons=a.neurons, n_in=a.n_in, n_out=a.n_out,
        activation=a.activation, gates=a.gates, hebb_type=a.hebb,
        attn_heads=a.attn_heads, attn_window=a.attn_window,
        grad_ckpt=a.grad_ckpt,
        ctx_bytes=a.ctx_bytes, q_bytes=a.q_bytes, opt_bytes=a.opt_bytes,
        deliberate=a.deliberate, order=a.order, echo_bytes=a.echo_bytes,
        k_min=a.k_min, k_max=a.k_max, none_option=a.none_option,
        q_per_ctx=a.q_per_ctx, paraphrase_holdout=a.paraphrase_holdout,
        holdout_intents=a.holdout_intents, oos_frac=a.oos_frac,
        batch=a.batch,
        lr=(d.lr if str(a.lr) == "keep"
            else None if str(a.lr) == "auto" else float(a.lr)),
        w_bce=a.w_bce, w_list=a.w_list,
        max_steps=a.max_steps, minutes=a.minutes,
        eval_every=a.eval_every, eval_contexts=a.eval_contexts,
        log_every=a.log_every,
        seed=a.seed, device=a.device, tag=a.tag,
    )


def main():
    a = parse_args()
    cfg = cfg_from_args(a)

    print("=" * 68)
    print("  SYSTEM ONE on OdyssNet — stored context, runtime options")
    print("=" * 68)
    corpus = load_corpus(cfg)

    if a.mode == "smoke":
        sys.exit(0 if smoke(cfg, corpus) else 1)

    if a.mode == "sweep":
        sweep(cfg, corpus, a.sweep, a.minutes)
        return

    if a.mode == "frontier":
        sizes = [int(s) for s in a.sizes.split(",")] if a.sizes else None
        frontier(cfg, corpus, a.minutes, sizes)
        return

    if a.mode == "train":
        budget = a.minutes * 60.0
        model, met, dt = train(cfg, corpus, budget_sec=budget)
        print(f"\n  done in {dt:.0f}s | best val: {fmt(met)}")
        print(f"  checkpoint: {ckpt_path(cfg)}")

        print("\n  → test, trained options")
        print("    " + fmt(evaluate(model, cfg, corpus, corpus.test,
                                    corpus.oos_test, cfg.eval_contexts)))
        print("  → test, paraphrases held out of training")
        print("    " + fmt(evaluate(model, cfg, corpus, corpus.test,
                                    corpus.oos_test, cfg.eval_contexts,
                                    held_out_phrasing=True)))
        if corpus.zero_shot:
            print(f"  → zero-shot: {len(corpus.zero_shot)} intents never "
                  f"trained on")
            print("    " + fmt(evaluate(
                model, cfg, corpus, corpus.zs_test, None, cfg.eval_contexts,
                option_pool=corpus.intents + corpus.zero_shot)))
        return

    cfg2, model = load_for_eval(cfg)
    if model is None:
        sys.exit(1)

    if a.mode == "eval":
        print(f"\n  params {n_params(model):,}")
        for label, kw in (
            ("test, trained options", {}),
            ("test, held-out paraphrases", {"held_out_phrasing": True}),
        ):
            m = evaluate(model, cfg2, corpus, corpus.test, corpus.oos_test,
                         cfg.eval_contexts, **kw)
            print(f"  {label:<30} {fmt(m)}")
        if corpus.zero_shot:
            m = evaluate(model, cfg2, corpus, corpus.zs_test, None,
                         cfg.eval_contexts,
                         option_pool=corpus.intents + corpus.zero_shot)
            print(f"  {'zero-shot options':<30} {fmt(m)}")
        return

    if a.mode == "ask":
        if not a.context:
            print("--mode ask needs --context", file=sys.stderr)
            sys.exit(2)
        qs = []
        if a.question and a.option:
            for q in a.question:
                qs.append((q, list(a.option)))
        else:
            # A default battery, so `ask` is useful with just --context.
            top = [humanise(i) for i in corpus.intents[:8]]
            qs = [(QUESTIONS["intent"][0], top + [NONE_OPTION]),
                  (QUESTIONS["domain"][0],
                   [humanise(d) for d in corpus.domains]),
                  (QUESTIONS["can_handle"][0], list(YES_NO)),
                  (QUESTIONS["enough"][0], list(YES_NO))]
        print(f"\n  context: {a.context!r}")
        print(f"  (read once, {len(qs)} questions off the stored state)")
        print_answers(ask(model, cfg2, a.context, qs))
        return


if __name__ == "__main__":
    main()
