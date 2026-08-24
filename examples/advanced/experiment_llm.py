"""
OdyssNet-LLM — temporal-depth language modeling on TinyStories.

A single NxN chaos core is used as a language model: tokens are injected one
at a time, the signal echoes for ``think_gap`` extra steps, and the readout
neurons are decoded to vocabulary logits. Depth is temporal — the same W is
reused every step — so the interesting question is not "how many layers" but
"how many steps per token, and is that worth the wall-clock".

This script is built to answer that question rather than to assert it.

    python -u experiment_llm.py --mode smoke                    # ~60s self-test
    python -u experiment_llm.py --mode train --tag base --batch 256 --minutes 25
    python -u experiment_llm.py --mode train --tag base --resume --batch 256
    python -u experiment_llm.py --mode sweep --sweep gap --minutes 3 --batch 128
    python -u experiment_llm.py --mode sweep --sweep attn --minutes 3 --batch 128
    python -u experiment_llm.py --mode eval  --tag base
    python -u experiment_llm.py --mode gen   --tag base --prompt "Once upon a time"

`--help` lists every flag with its default and a fuller set of examples. Use
`python -u` when piping: this script line-buffers stdout, but the interpreter
flag is what keeps a long run's progress visible through a pipe.

Design notes
------------
* **Two tokenizers, one flag.** ``--tokenizer bpe`` (the default) trains a
  byte-level BPE on the corpus itself, once, and pins it. ``--tokenizer
  cl100k_base`` (or any other tiktoken encoding) skips that step entirely —
  nothing to train, nothing to cache, no ``tokenizers`` dependency. See
  "Choosing a tokenizer" below for why the default is the small trained one.
* **Offline + instant restart.** The corpus is tokenized once, streaming, into
  a flat integer memmap next to ``data/`` (element width chosen from the
  vocabulary size); later runs mmap it in milliseconds. Peak memory during
  tokenization is one encode batch, so corpus size is bounded by disk rather
  than by RAM. No network, no HuggingFace streaming, no per-run tokenizer cost.
* **Stateful contiguous streams.** The batch is B parallel readers, each
  walking its own contiguous shard of the corpus. Hidden state carries across
  chunks (``keep_state=True``) and the model detaches it internally, which is
  textbook truncated BPTT — the model effectively sees unbounded context
  instead of restarting cold every 512 tokens.
* **Honest metrics.** Validation is a fixed held-out tail of the corpus, and
  perplexity is always computed from *unsmoothed* cross-entropy. Bits-per-byte
  is reported alongside, because per-token perplexity is not comparable across
  tokenizers and bits-per-byte is.
* **Budget-matched comparison.** Sweep arms are given equal wall-clock, and
  both val PPL and tokens-consumed are reported, so "learns more per token"
  can be told apart from "simply runs faster".
* **Attention is a flag, not a fork.** ``--attn-heads 4`` gives every thinking
  step a query over the states before it; at zero (the default) nothing is
  built. Its output projection starts at zero and the module is constructed
  after the core, so an attention run and a non-attention run at the same seed
  begin from the same W — the comparison has one variable in it. See
  "Attention" below.

Choosing a tokenizer
--------------------
The industrial answer to "which vocabulary" is a large one: Llama 3 uses a
128k tiktoken-style BPE, Qwen ~151k, Gemma a 256k SentencePiece, and even the
deliberately small models (Mistral, Phi-3, SmolLM) sit at 32k-49k. That answer
does not transfer here, and the reason is arithmetic rather than taste. Those
models carry a hidden width of 2048 or more, so an embedding table amortizes
against a large body of compute; this one is a 1024-neuron core, and the table
would *become* the model.

Measured on a held-out 906 KB slice of ``tinystories.txt`` — bytes per token,
higher is better compression:

    tokenizer                vocabulary   bytes/token   embed+decoder
    -----------------------------------------------------------------
    byte                            256         1.000          0.20M
    trained BPE                   1,024         3.079          0.79M
    trained BPE (the default)     2,048         3.601          1.57M
    trained BPE                   4,096         3.908          3.15M
    tiktoken gpt2                50,257         3.914         38.6M
    tiktoken cl100k_base        100,277         4.146         77.1M
    tiktoken o200k_base         200,019         4.190        153.8M

A 4,096-token BPE trained on this corpus matches GPT-2's 50k vocabulary to
within 0.2% (3.908 vs 3.914 bytes/token) at one twelfth the ids, because an
in-domain vocabulary spends every merge on text the model will actually see.
``cl100k_base`` buys 6% more compression for 24x the embedding parameters: at
the defaults that is 2.6M parameters (1.05M of them the chaos core) becoming
78M, 98% of it lookup table — and unrunnable at the documented batch, since
the logits tensor alone is ``batch x chunk x vocab``, or 4.9 GB in fp32 at
256 x 48 x 100,277.

So the default stays ``--tokenizer bpe --vocab-size 2048``, and the reference
run below is defined against it. The tiktoken path is for the cases where the
trade goes the other way: skipping the training step, comparing against a
standard vocabulary, or scaling the core up to where a large table pays for
itself. Bits-per-byte is reported so those runs stay comparable — perplexity
is not, across vocabularies.

``--tokenizer byte`` sits at the opposite end and is chosen for a different
reason than cost. It is the only option under which a character is reliably
its own token, which is what character-level work needs: a subword merge hides
the units the model has to manipulate, so a model that has only ever seen
``100`` as one id has learned nothing about the three digits in it. The
vocabulary table nearly vanishes (0.20M against a 1.05M core), leaving the
chaos core as most of what is being measured. The price is sequence length —
one byte per token against the BPE's 3.6, so the same text is ~3.6x more
timesteps, and this architecture is latency-bound on sequential steps. Reach
for it when the token boundary matters, not as a general default.

Attention
---------
A transformer stacks attention between layers. This model has none, so
attention goes along the axis it does have: at every thinking step the state
queries a cache of the states that came before it, and what comes back is
added to the same pre-activation signal the recurrence and the token embedding
feed. What the core does by mixing everything through one matrix, attention
does by naming what it wants.

The flags that matter, in the order they usually matter:

    --attn-heads 4          switch it on (0, the default, builds nothing)
    --attn-kv-heads 1       multi-query; the cache is what OOMs, not the weights
    --attn-window 256       how far back a query can see
    --attn-write token      one entry per token, not per thinking step

The cost is not the parameters. A query is issued every step, so `--think-gap`
multiplies the *reads*; `--attn-write token` keeps the cache measured in tokens
so it does not also multiply the *entries*. And the keys and values a backward
pass has to keep are quadratic in the writes inside one truncated-BPTT window
— chunk², not run length — which is the number this script prints before
training starts rather than after the OOM.

One more thing the branch does not cost you: its output projection starts at
zero and is divided by the square root of its own width, so switching attention
on — or widening it — changes nothing until training decides otherwise. Without
that division a wide branch drowns the recurrence (`docs/LIBRARY.md` carries
the measurement).

`--sweep attn` is the measurement: seven arms, equal wall-clock, `off` being
the 2.x architecture exactly. Add `--max-steps N` to compare at equal *tokens*
instead — the two questions have different answers.

Speed
-----
The echo loop issues thousands of small kernels per step and is bound by
launching them, not by the arithmetic inside any of them: profiled on a
10-neuron core at batch 32 over 16 steps, ~5,800 launches for about 20 ms of
GPU work. So `--compile` is the largest speedup on offer, by a wide margin:

    --compile               2.2x on a bare core
                            4.0x with --hebb both
                            3.6x with --hebb both and attention

It is also what makes plasticity affordable — eager it adds 132% to the step,
compiled 26%. The price is a warmup of a minute or two before the first step,
longer with plasticity and attention on; after that the graph is stable across
the train/eval alternation.

One trap worth knowing if you compile your own loop rather than using this one:
give the model fixed shapes. A ragged final batch is a new shape, and enough
recompiles exhaust Dynamo's budget and silently drop the process back to eager
for good — the only symptom is a step time that never improved. This script
feeds fixed-size batches from the token cache and does not have the problem.

What the sweeps measured (RTX 3060 Ti, 1024 neurons, 2.6M params, vocab 2048)
----------------------------------------------------------------------------
* ``--sweep gap``, 1.3 min/arm, batch 128 — temporal depth pays, but only the
  first step of it. ``gap=1`` reached val ppl 16.85 on 5.68M tokens while
  ``gap=0`` reached only 48.04 on 11.85M: one extra echo step per token beat
  twice the data. ``gap=3``/``gap=5`` cost 4-6x per token and, at equal
  wall-clock, never got far enough to compete (ppl ~410-415 at 1.5-2.3M
  tokens) — a verdict about wall-clock, not about depth per token.
* ``--sweep coldstart``, 1500 steps, batch 128 — the default ``gap=1``
  configuration develops a degenerate absorbing state that only cold starts
  reveal. With state carried indefinitely (``cold_start_every=0``): 15.6% of
  held-out shards collapsed to ~240 nats/token and pooled val loss read 39.55,
  while *training* loss looked healthy at 2.689. With staggered per-row resets
  (``cold_start_every=32``): 0% collapsed, pooled val loss 2.68 (ppl 14.60),
  training loss 2.706. Same seed, same steps, same tokens — the failure is
  removed at no measurable cost in fit quality.
* Batch is nearly free: throughput scaled ~linearly 6k -> 274k tok/s from
  batch 8 to 512, because the model is latency-bound on sequential steps.
* ``--sweep attn``, 2 min/arm, batch 128 — **at equal wall-clock the 2.x
  architecture wins**, by median cold-start shard ppl:

      arm          p50 ppl   bits/byte   Mtokens    tok/s
      ---------------------------------------------------
      off            14.55      1.0765     10.91   90,853
      read_token     17.75      1.1736      4.19   34,901
      mqa4           20.11      1.2302      3.16   26,280
      no_rope        21.27      1.2469      3.58   29,847
      window32       23.47      1.2911      2.94   24,481
      mha4           27.38      1.3553      3.06   25,436
      write_step     32.16      1.4171      2.30   19,125

  Every attention arm ran 2.6-4.7x slower per token, so every one of them
  spent the budget on a third of the data — the ranking among them is mostly
  a ranking of how cheap they are. ``--attn-read token`` is the clearest
  example: identical attention, queried once per token instead of once per
  step, 1.33x the throughput and the best attention result here. 0% collapsed
  cold starts everywhere, attention on or off.
  Read that as a verdict about wall-clock on a 2.6M-parameter core, not about
  attention. The cost is kernel-launch overhead per *step*, which shrinks with
  batch (3.45x at batch 64, 2.79x at 512) and would shrink further on a core
  large enough for the launches to disappear behind real arithmetic. Compare
  at equal *tokens* with ``--max-steps`` for the other half of the picture,
  and note that the tok/s column inside a long sweep drifts downward as the
  GPU heats; the controlled throughput numbers are in ``docs/LIBRARY.md``.
* ``--sweep attn --max-steps 600``, batch 128 — **at equal tokens the verdict
  reverses, and every attention arm wins.** Same seed, same 3.69M tokens:

      arm          val ppl   bits/byte    tok/s
      -----------------------------------------
      mqa4           18.57      1.1778   26,302
      read_token     19.75      1.2027   34,811
      mha4           21.40      1.2351   25,169
      off            22.19      1.2497   90,007

  So attention learns more per token (16% better perplexity at the default
  four multi-query heads) and costs more per second, and which of those two
  facts decides your run depends on whether you are token-limited or
  time-limited. On a corpus this size, on this GPU, at this core width, the
  clock is the binding constraint and ``off`` still wins the wall-clock sweep
  above — but the learning-per-token claim is the one that scales with
  hardware, and it points the other way.

Reference run — ``--mode train --minutes 25 --batch 256`` at the defaults
above, 2,625,280 parameters: held-out **loss 2.2009, ppl 9.03,
bits/byte 0.8764**, median cold start 2.3154, **0.0% collapsed cold starts**,
226.15M tokens at 150,761 tok/s. Reproduce with that command; `--mode eval
--tag base` re-scores the saved checkpoint. Those figures are defined against
the exact token cache that run was trained on: a rebuilt cache moves the
held-out split, and the score with it (see CHANGELOG 2.6.4).
"""

import sys

# Keep emoji-rich console output from crashing legacy Windows code pages.
# line_buffering=True is not optional: reconfigure() rebuilds the TextIOWrapper
# and would otherwise discard `python -u`'s write-through, leaving a long
# training run's progress invisible until the process exits.
# stderr too: argparse errors and SystemExit messages go there, and on a
# legacy code page they came out backslash-escaped ("✋ Refusing to...").
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import argparse
import glob
import json
import math
import os
import time
from dataclasses import dataclass, asdict, replace

import numpy as np
import torch
import torch.nn.functional as F

from odyssnet import (
    OdyssNet,
    OdyssNetTrainer,
    save_checkpoint,
    load_checkpoint,
    set_seed,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "ckpt")
DATA_DIR = os.path.join(HERE, "..", "..", "data")
CACHE_DIR = os.path.join(DATA_DIR, "token_cache")

# A shard scoring worse than this has not "learned poorly" — it has fallen
# into the degenerate absorbing state (observed at ~240 nats/token, against
# ~7.6 for uniform guessing over a 2048 vocab). Anything past uniform by this
# margin is a collapse, not a bad fit.
FAIL_NATS = 15.0


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Cfg:
    # --- corpus ---
    text_path: str = os.path.join(DATA_DIR, "tinystories.txt")
    tokenizer: str = "bpe"           # "bpe" = train on the corpus; otherwise
                                     # the name of a tiktoken encoding
    vocab_size: int = 2048           # only a knob for "bpe"; a tiktoken
                                     # encoding brings its own and overwrites it
    val_chars: int = 1_500_000       # held-out tail of the raw text
    train_chars: int = -1            # -1 = all remaining text

    # --- architecture ---
    neurons: int = 1024
    n_in: int = 256                  # neurons that receive the token embedding
    n_out: int = 512                 # neurons the vocab decoder reads
    activation: tuple = ("none", "gelu_tanh", "tanh")
    weight_init: tuple = ("quiet", "resonant", "quiet", "zero")
    gates: tuple = ("none", "none", "identity")
    hebb_type: str | None = None
    hebb_res: str = "neuron"
    dropout: float = 0.0
    tie_embeddings: bool = False

    # --- temporal attention (OdyssNet 3.0) ---
    attn_heads: int = 0              # 0 = no attention at all
    attn_kv_heads: int = 1           # 1 = multi-query; == heads is classic MHA
    attn_head_dim: int = 0           # 0 = derive from neurons / heads
    attn_window: int = 256           # cache entries visible to a query
    attn_write: str = "token"        # "token" or "step"
    attn_read: str = "step"          # "token" or "step"
    attn_rope: bool = True
    attn_qk_norm: bool = True
    attn_dropout: float = 0.0

    # --- temporal depth ---
    pulse_mode: bool = True          # True injects a token's embedding only on
                                     # the step it arrives, leaving the echo
                                     # steps to run on state alone; False holds
                                     # it across all of them. Identical at
                                     # think_gap 0, divergent above it.
    think_gap: int = 1               # extra echo steps per token
    chunk: int = 48                  # tokens per optimizer step (TBPTT window)
    cold_start_every: int = 32       # mean chunks before a row's state is
                                     # zeroed again (0 = never reset)

    # --- optimization ---
    batch: int = -1                  # -1 = empirical autotune
    lr: float | None = None          # None = zero-config ChaosGrad
    lr_force_auto: bool = False      # explicit `--lr auto`: on resume, move a
                                     # fixed-rate checkpoint back to the
                                     # estimator instead of keeping its mode
    label_smoothing: float = 0.0
    grad_persistence: float = 0.0    # "ghost gradients": a fraction of the
                                     # previous step's gradient is added to the
                                     # next one, on top of ChaosGrad's own
                                     # momentum
    grad_ckpt: bool = False
    compile: bool = False            # torch.compile the forward. The step is
                                     # launch-bound at every size measured, so
                                     # fusing it is the single largest speedup
                                     # available — and the one that makes
                                     # plasticity affordable (measured on a
                                     # 10-neuron core: +132% eager, +26%
                                     # compiled). Warmup costs a minute or two.

    # --- runtime ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    minutes: float = 0.0             # 0 = run until interrupted
    max_steps: int = 0               # 0 = unlimited
    eval_every: int = 400            # optimizer steps between validations
    val_tokens: int = 200_000
    val_batch: int = 64              # = 64 independent cold starts per score
    val_warmup_chunks: int = 2       # discarded so a cold state isn't scored
    val_amp: bool = False            # eval in fp32: the metric must be exact
    log_every: int = 50
    tag: str = "base"

    def io_ids(self):
        input_ids = list(range(self.n_in))
        output_ids = list(range(self.n_in, self.n_in + self.n_out))
        needed = self.n_in + self.n_out
        if needed > self.neurons:
            raise ValueError(
                f"n_in + n_out = {needed} exceeds neurons = {self.neurons}"
            )
        return input_ids, output_ids


# --------------------------------------------------------------------------- #
# Tokenizers                                                                  #
# --------------------------------------------------------------------------- #

#: Ready-made tiktoken encodings offered by `--tokenizer`: no training pass,
#: and a vocabulary fixed by the encoding rather than chosen. See "Choosing a
#: tokenizer" in the module docstring for why the default is a small in-domain
#: BPE instead.
TIKTOKEN_ENCODINGS = ("gpt2", "r50k_base", "p50k_base", "cl100k_base", "o200k_base")


class TokenizerBase:
    """
    The tokenizer interface this script uses. The cache builder, validator and
    generator talk to this and never to `tokenizers` or `tiktoken` directly,
    so a third backend is a class rather than a grep.

    Subclasses provide `encode(text) -> [id]`, `encode_batch([text]) -> [[id]]`
    (plain lists — the `tokenizers` `Encoding` wrapper is unwrapped here),
    `decode([id]) -> text`, `get_vocab_size()`, and a `bos_id` to fall back on
    when a prompt encodes to nothing.

    `get_vocab_size` is the id space the model must cover, which is not always
    the size requested: special tokens can sit above the nominal vocabulary.
    `bos_id` has no default on purpose — id 0 is `<s>` in one backend and an
    ordinary byte in the other, so omitting it should raise, not guess.

    Named `TokenizerBase` because `tokenizers` exports a `Tokenizer` this file
    imports.
    """

    def encode(self, text):
        raise NotImplementedError

    def encode_batch(self, texts):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError

    def get_vocab_size(self):
        raise NotImplementedError


class TrainedBPE(TokenizerBase):
    """Byte-level BPE trained on the corpus itself and pinned to a file."""

    def __init__(self, tok):
        self._tok = tok
        tid = tok.token_to_id("<s>")
        self.bos_id = 0 if tid is None else tid

    def encode(self, text):
        return self._tok.encode(text).ids

    def encode_batch(self, texts):
        return [e.ids for e in self._tok.encode_batch(texts)]

    def decode(self, ids):
        return self._tok.decode(ids)

    def get_vocab_size(self):
        return self._tok.get_vocab_size()


class TikToken(TokenizerBase):
    """
    A pretrained tiktoken encoding — no training pass, no tokenizer file.

    Encoding goes through `encode_ordinary`, not `encode`: the latter raises
    on a special-token string such as `<|endoftext|>`, and a corpus is text
    rather than a prompt template.
    """

    def __init__(self, name):
        self._enc = load_tiktoken(name)
        # These encodings have no <s>, and their reserved special ids are not
        # safe to feed a model that has never seen them. A newline is a real
        # token and a plausible document start.
        self.bos_id = self._enc.encode_ordinary("\n")[0]

    def encode(self, text):
        return self._enc.encode_ordinary(text)

    def encode_batch(self, texts):
        return self._enc.encode_ordinary_batch(texts)

    def decode(self, ids):
        return self._enc.decode(list(ids))

    def get_vocab_size(self):
        # n_vocab spans the special tokens, so the decoder covers every id the
        # encoder can emit. Slightly wasteful, never out of range.
        return self._enc.n_vocab


class ByteTokenizer(TokenizerBase):
    """
    One id per byte: 256 ids, nothing to train, nothing to pin.

    Worth choosing for two reasons rather than one. It is the only vocabulary
    under which a character is reliably its own token, which matters whenever
    the task is character-level — arithmetic, spelling, transliteration,
    anything where a subword merge would hide the units the model has to
    manipulate. And it makes the vocabulary table negligible: 0.20M parameters
    against a 1.05M core at 1024 neurons, where the 2048-token BPE costs 1.57M
    and is the larger half of the model.

    What it costs is sequence length. Bytes compress far worse than a trained
    BPE — on TinyStories, 1.0 byte/token against 3.6 — so the same text is
    ~3.6x more timesteps, and this architecture is latency-bound on sequential
    steps. Use it when the token boundary matters or when the core is small
    enough that a large table would dominate; not as a general default.
    """

    def __init__(self):
        # No <s> exists here. A newline is a real, decodable byte and a
        # plausible document start.
        self.bos_id = ord("\n")

    def encode(self, text):
        return list(text.encode("utf-8"))

    def encode_batch(self, texts):
        return [list(t.encode("utf-8")) for t in texts]

    def decode(self, ids):
        # A scored window can begin partway through a multi-byte character, so
        # an invalid UTF-8 prefix is normal here and must not raise.
        return bytes(bytearray(ids)).decode("utf-8", errors="replace")

    def get_vocab_size(self):
        return 256


def load_tiktoken(name):
    """Fetch a tiktoken encoding, with an install hint instead of a traceback."""
    try:
        import tiktoken
    except ImportError as e:
        raise SystemExit(
            f"\n✋ --tokenizer {name} needs tiktoken, which is not installed.\n"
            f"     pip install tiktoken        (or: pip install -e \".[llm]\")\n"
            f"   Or use the zero-dependency default: --tokenizer bpe\n"
        ) from e
    # tiktoken caches encodings internally, so repeated calls are free.
    return tiktoken.get_encoding(name)


def resolve_vocab(cfg):
    """
    Pin `cfg.vocab_size` to the tokenizer's actual id space.

    Only `bpe` has a vocabulary to choose; the others carry their own, and it
    must be known before the corpus is tokenized because the cache's element
    width derives from it — a 100k vocabulary written as uint16 is silently
    wrong rather than an error.
    """
    if cfg.tokenizer == "bpe":
        return cfg
    if cfg.tokenizer == "byte":
        return replace(cfg, vocab_size=ByteTokenizer().get_vocab_size())
    return replace(cfg, vocab_size=load_tiktoken(cfg.tokenizer).n_vocab)


def bpe_path(cfg):
    """Where this vocab size's trained tokenizer is pinned."""
    if cfg.vocab_size % 1000:
        return os.path.join(CKPT_DIR, f"poc_tokenizer_{cfg.vocab_size}.json")
    return os.path.join(CKPT_DIR, f"poc_tokenizer_{cfg.vocab_size // 1000}k.json")


def get_tokenizer(cfg, blocks=None):
    """
    The tokenizer for this config: pretrained, or trained once and pinned.

    `blocks` is text to train a BPE vocabulary from, used only when this config
    asks for one and no pinned file exists yet. The tiktoken path never
    consumes it.
    """
    if cfg.tokenizer == "byte":
        return ByteTokenizer()
    if cfg.tokenizer != "bpe":
        return TikToken(cfg.tokenizer)

    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.implementations import ByteLevelBPETokenizer

    os.makedirs(CKPT_DIR, exist_ok=True)
    path = bpe_path(cfg)

    if os.path.exists(path):
        return TrainedBPE(HFTokenizer.from_file(path))

    if blocks is None:
        raise FileNotFoundError(f"Tokenizer {path} missing and no text to train it on.")

    print(f"📚 Training {cfg.vocab_size}-token BPE tokenizer...")
    tok = ByteLevelBPETokenizer()
    # train_from_iterator consumes the block stream lazily — the sample the
    # vocabulary is learned from is never materialized as one string.
    tok.train_from_iterator(blocks, vocab_size=cfg.vocab_size, min_frequency=2,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tok.save(path)
    print(f"✅ Tokenizer saved: {path}")
    return TrainedBPE(HFTokenizer.from_file(path))


def vocab_advisory(cfg, vocab):
    """
    Where the parameters went, plus a warning when the vocabulary is about to
    dominate the run. Advisory, not a limit: a large table is the right call on
    a large core, which is what the tiktoken path exists for.
    """
    table = vocab * cfg.n_in + vocab * cfg.n_out + vocab      # embed + decoder
    # W only; B and memory_feedback are O(N) beside it.
    core = cfg.neurons * cfg.neurons
    share = table / max(table + core, 1)
    print(f"🔤 tokenizer {cfg.tokenizer} | vocab {vocab:,} | "
          f"embed+decoder {table/1e6:.2f}M vs core {core/1e6:.2f}M "
          f"({share:.0%} of the model is lookup table)")

    # batch x chunk x vocab floats, kept by autograd for the backward pass. On
    # a large vocabulary this tensor, not the weights, decides whether the run
    # fits.
    logits_gb = cfg.batch * cfg.chunk * vocab * 4 / 1e9
    if logits_gb > 0.5:
        suggest = max(1, int(0.5e9 / (cfg.chunk * vocab * 4)))
        print(f"⚠️  logits tensor is {logits_gb:.1f} GB per step "
              f"({cfg.batch} x {cfg.chunk} x {vocab:,} fp32). If this OOMs, "
              f"try --batch {suggest} or a shorter --chunk.")
    # Gated on the vocabulary being the large half of the imbalance, not on the
    # ratio alone: a deliberately tiny core puts any vocabulary above 90% and
    # needs no advice — least of all to switch to the tokenizer already in use.
    if share > 0.9 and vocab > 4096:
        print(f"⚠️  {share:.0%} of the parameters are the vocabulary table. "
              f"This measures the tokenizer more than the chaos core — either "
              f"grow --neurons to match, or use the default "
              f"--tokenizer bpe --vocab-size 2048 that the reference numbers "
              f"are defined against.")


def plasticity_advisory(cfg, model):
    """
    What the plastic trace costs, before it costs it.

    The trace is per example, so what the backward pass holds grows with the
    batch as well as with the step count. It is the one term in this model that
    does, and raising the batch for throughput is the natural move that makes it
    unaffordable.
    """
    if model.hebb_type is None:
        return
    paths = 2 if model.hebb_type == "both" else 1
    steps = cfg.chunk * (cfg.think_gap + 1)
    budget_gb = 2.0

    # The trace is kept as the writes it is made of, so it grows with the step
    # count rather than with the neuron count squared. Constants measured at
    # 1024 neurons over 96 steps with a populated buffer, which is what a
    # training run holds; smaller runs sit under them.
    row_gb = (26 if paths == 1 else 35) * steps * cfg.neurons * 4 / 1e9
    trace_gb = row_gb * cfg.batch
    print(f"🧬 plasticity {model.hebb_type}/{model.hebb_res} | "
          f"{steps} steps x batch {cfg.batch} x {cfg.neurons} neurons | "
          f"trace kept for backward {trace_gb:.1f} GB")
    if trace_gb > budget_gb:
        print(f"⚠️  that trace alone is {trace_gb:.1f} GB. It is linear in the "
              f"batch and in the step count — try --batch "
              f"{max(1, int(budget_gb / row_gb))} or a shorter --chunk.")


def attention_advisory(cfg, model):
    """
    What the attention branch costs, before it costs it.

    Two numbers matter and neither is the parameter count. The KV cache the
    backward pass has to keep is quadratic in *writes per optimizer step* —
    that is the tokens in a truncated-BPTT window, not the run length — and
    the query is issued once per thinking step, so `--think-gap` buys reads
    rather than entries.
    """
    attn = getattr(model, "attn", None)
    if attn is None:
        return
    params = sum(p.numel() for p in attn.parameters())
    writes = cfg.chunk if cfg.attn_write == "token" else cfg.chunk * (cfg.think_gap + 1)
    train_gb = attn.training_cache_bytes(cfg.batch, writes) / 1e9
    print(f"👁️  attention {attn.heads}x{attn.head_dim} "
          f"({attn.kv_heads} kv head{'s' if attn.kv_heads > 1 else ''}) | "
          f"window {attn.window} | write per {cfg.attn_write} | "
          f"{params/1e6:.2f}M params | "
          f"KV kept for backward {train_gb:.2f} GB "
          f"({writes} writes/step at batch {cfg.batch})")
    if train_gb > 2.0:
        smaller = max(1, int(cfg.batch * 2.0 / max(train_gb, 1e-9)))
        print(f"⚠️  that KV term alone is {train_gb:.1f} GB. It grows with "
              f"chunk² — try --batch {smaller}, a shorter --chunk, or "
              f"--attn-kv-heads 1 (already the default) before blaming the "
              f"core.")


# --------------------------------------------------------------------------- #
# Corpus: uint16/uint32 memmap cache                                          #
# --------------------------------------------------------------------------- #


def _iter_blocks(path, start=0, end=None, target=200_000):
    """
    Yield ~target-sized text blocks from a byte range of a file.

    Reads line by line in binary and decodes per line: a corpus is never held
    in memory, and line boundaries are safe UTF-8 split points (``\\n`` cannot
    occur inside a multi-byte sequence). Peak cost is one block.
    """
    end = os.path.getsize(path) if end is None else end
    buf, size, pos = [], 0, start
    with open(path, "rb") as f:
        f.seek(start)
        while pos < end:
            line = f.readline()
            if not line:
                break
            pos += len(line)
            # Normalize newlines exactly as text-mode reads would. Reading in
            # binary is what keeps memory flat, but it also preserves the CR of
            # a CRLF file — and a byte-level BPE turns every one of those into
            # its own token. On a CRLF corpus that is one wasted token per
            # line: ~5% of the whole stream, and it silently shifts every
            # metric relative to a text-mode tokenization of the same file.
            s = line.decode("utf-8", errors="replace").replace("\r\n", "\n")
            buf.append(s)
            size += len(s)
            if size >= target:
                yield "".join(buf)
                buf, size = [], 0
    if buf:
        yield "".join(buf)


def _tail_offset(path, val_chars):
    """
    Byte offset where the held-out tail starts, aligned to a line boundary.

    `val_chars` is a character budget; for ASCII-heavy text one byte is one
    character, so the byte offset approximates it and the alignment only ever
    trims. Non-Latin corpora get a slightly smaller held-out set than asked
    for, never a larger one.

    Aligning on a bare b"\\n" is deliberate and covers both line-ending
    conventions: in CRLF the b"\\n" is the second byte, so the offset after it
    is still the start of a line. Searching for a paragraph break instead
    looks tidier and is a trap — b"\\n\\n" never matches CRLF bytes (they read
    b"\\r\\n\\r\\n"), and three of the four corpora in data/ have no blank
    lines at all, which left the split mid-line and potentially mid-UTF-8.
    """
    size = os.path.getsize(path)
    # Never hand back a corpus with no training half: a small file plus the
    # default val budget would otherwise make the whole thing validation.
    budget = min(val_chars, size // 5)
    start = max(0, size - budget)
    if start <= 0:
        return 0
    with open(path, "rb") as f:
        f.seek(start)
        window = f.read(min(4_000_000, size - start))
    idx = window.find(b"\n")
    if idx >= 0:
        return start + idx + 1
    # No line break at all in 4 MB. Vanishingly unlikely, but at least refuse
    # to cut a multi-byte character in half.
    off = 0
    while off < len(window) and (window[off] & 0xC0) == 0x80:
        off += 1
    return start + off


def _token_dtype(vocab_size):
    """
    Narrowest integer type that can hold every id of this vocabulary.

    uint16 covers ids up to 65535, which is every BPE vocabulary this example
    ships with — but a tiktoken-scale vocabulary (100k+) would wrap around
    silently, corrupting the corpus with no error anywhere. numpy does not
    warn on out-of-range casts, so the guard has to be the type choice itself.
    """
    return np.uint16 if vocab_size <= np.iinfo(np.uint16).max + 1 else np.uint32


def _encode_to_file(tok, blocks, out_path, dtype=np.uint16, label="", verbose=False):
    """
    Tokenize an iterable of text blocks straight to a flat uint16 file.

    Nothing accumulates in RAM: each batch of blocks is encoded and written
    through. The previous version built a Python list of token ids, which at
    full-corpus scale (~490M tokens) is 13+ GB of boxed ints before the array
    conversion even starts. Peak here is one encode batch, a few MB, so the
    corpus size a user can train on stops being bounded by their RAM.
    """
    total, batch = 0, []
    t0 = time.time()
    tmp = out_path + ".part"
    with open(tmp, "wb") as out:
        limit = np.iinfo(dtype).max

        def flush():
            nonlocal total
            if not batch:
                return
            for enc in tok.encode_batch(batch):
                ids = np.asarray(enc, dtype=np.int64)
                # Special tokens are not always inside the nominal vocab size,
                # so check the ids actually produced rather than trusting the
                # declared vocabulary. One vectorized max per block.
                if ids.size and ids.max() > limit:
                    raise ValueError(
                        f"token id {int(ids.max())} exceeds {np.dtype(dtype).name} "
                        f"(max {limit}). The corpus would be silently corrupted. "
                        f"The cache's element width is derived from the "
                        f"tokenizer's reported vocabulary size, so this means "
                        f"the tokenizer emits ids outside the range it declares."
                    )
                ids.astype(dtype, copy=False).tofile(out)
                total += ids.size
            batch.clear()

        for i, block in enumerate(blocks):
            batch.append(block)
            if len(batch) >= 64:
                flush()
                if verbose and i % 2560 == 0 and i:
                    print(f"   {label}: {total/1e6:8.1f}M tokens  {time.time()-t0:5.0f}s")
        flush()
    os.replace(tmp, out_path)
    return total


def _open_tokens(path, dtype=np.uint16):
    """Memory-map a flat token file."""
    return np.memmap(path, dtype=dtype, mode="r")


def load_corpus(cfg, verbose=True):
    """
    Returns (train_tokens, val_tokens, tokenizer).

    The raw text is split by *character* offset first — the tail becomes
    validation — and each half is tokenized independently, so no validation
    token can leak into training via a shared BPE merge boundary.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    # The tokenizer is part of the cache's identity, but the trained-BPE stamp
    # keeps its historical `.v<size>` form byte-for-byte so existing caches —
    # and the numbers measured against them — stay valid. Do not unify these
    # two branches; no encoding name can collide with `v<digits>`.
    base = os.path.basename(cfg.text_path)
    if cfg.tokenizer == "bpe":
        stamp = f"{base}.v{cfg.vocab_size}.val{cfg.val_chars}"
    else:
        stamp = f"{base}.{cfg.tokenizer}.val{cfg.val_chars}"
    # The element width is part of the cache's identity: a flat token file
    # carries no header, so reading uint32 bytes as uint16 yields plausible
    # garbage rather than an error.
    dtype = _token_dtype(cfg.vocab_size)
    width = np.dtype(dtype).name
    train_bin = os.path.join(CACHE_DIR, f"{stamp}.{width}.train.bin")
    val_bin = os.path.join(CACHE_DIR, f"{stamp}.{width}.val.bin")

    # .npy is the pre-2.6.3 cache format. Still read if present, so an existing
    # cache — and every score already measured against it — stays valid.
    train_npy = os.path.join(CACHE_DIR, f"{stamp}.train.npy")
    val_npy = os.path.join(CACHE_DIR, f"{stamp}.val.npy")
    # Token ids are only meaningful next to the tokenizer that produced them,
    # and a trained BPE lives in a file that can go missing (cleaning ckpt/ is
    # enough). Retraining it is not a repair: nothing records what an existing
    # cache's tokenizer was trained on, and a retrained one measurably differs
    # (CHANGELOG 2.6.4). A missing tokenizer therefore invalidates the cache,
    # and the two are rebuilt together.
    usable = cfg.tokenizer != "bpe" or os.path.exists(bpe_path(cfg))
    if not usable and any(os.path.exists(p) for p in (train_bin, train_npy)):
        print(f"⚠️  Token cache found, but the tokenizer that built it "
              f"({os.path.basename(bpe_path(cfg))}) is gone. Its ids cannot be "
              f"decoded, so the cache is being rebuilt from the corpus.")
        if os.path.exists(train_npy):
            print(f"   Also delete the legacy cache {train_npy} (and its .val "
                  f"pair): .npy takes priority on the next run and a rebuilt "
                  f"tokenizer will not match it.")

    # .npy first: if a user already has one, every number they have measured
    # was measured against it, and a format upgrade must not move their metric.
    # Delete the .npy pair to re-tokenize under the streaming scheme (which
    # normalizes CRLF, so a fresh cache differs from a pre-2.6.3 one by ~0.3%).
    for tp, vp, opener in ((train_npy, val_npy,
                            lambda p: np.load(p, mmap_mode="r")),
                           (train_bin, val_bin,
                            lambda p: _open_tokens(p, dtype))):
        if usable and os.path.exists(tp) and os.path.exists(vp):
            tok = get_tokenizer(cfg)
            train, val = opener(tp), opener(vp)
            if verbose:
                print(f"💾 Token cache: {len(train):,} train / {len(val):,} val "
                      f"tokens (vocab {cfg.vocab_size})")
            return train, val, tok

    if not os.path.exists(cfg.text_path):
        raise FileNotFoundError(
            f"Corpus not found: {cfg.text_path}\n"
            f"   Point --text at a UTF-8 text file."
        )

    # A crashed or interrupted tokenization leaves a .part behind; it is never
    # read, so clear it rather than accumulating dead files in the cache dir.
    for stale in glob.glob(os.path.join(CACHE_DIR, "*.part")):
        os.remove(stale)

    # Split at a line boundary and tokenize each side independently, so no
    # validation token reaches training through a shared BPE merge. Neither
    # side is ever read into memory whole. `train_chars` is compared against a
    # byte offset here, so on a multi-byte corpus it trims slightly more text
    # than its name suggests.
    size = os.path.getsize(cfg.text_path)
    split = _tail_offset(cfg.text_path, cfg.val_chars)
    train_end = min(split, cfg.train_chars) if cfg.train_chars > 0 else split

    tok = get_tokenizer(cfg, blocks=_iter_blocks(cfg.text_path, 0,
                                                 min(train_end, 20_000_000)))

    print(f"🔤 Tokenizing {cfg.text_path} "
          f"({train_end/1e6:.1f}MB train + {(size-split)/1e6:.1f}MB val)...")
    t0 = time.time()
    n_train = _encode_to_file(tok, _iter_blocks(cfg.text_path, 0, train_end),
                              train_bin, dtype, "train", verbose=True)
    n_val = _encode_to_file(tok, _iter_blocks(cfg.text_path, split, size),
                            val_bin, dtype, "val", verbose=True)
    print(f"✅ Cached in {time.time()-t0:.0f}s: {n_train:,} train / {n_val:,} val "
          f"tokens ({width})")

    return _open_tokens(train_bin, dtype), _open_tokens(val_bin, dtype), tok


class StreamShards:
    """
    B parallel readers over contiguous shards of a token array.

    Each row of the batch walks its own region of the corpus in order, so
    hidden state carried across chunks is always state about *that row's*
    ongoing text. This is what makes truncated BPTT meaningful here.
    """

    def __init__(self, tokens, batch, chunk, device):
        self.tokens = tokens
        self.batch = batch
        self.chunk = chunk
        self.device = device
        shard = len(tokens) // batch
        if shard < chunk + 1:
            raise ValueError(
                f"{len(tokens):,} tokens cannot feed batch={batch} at chunk={chunk}"
            )
        self.starts = np.array([i * shard for i in range(batch)], dtype=np.int64)
        self.limits = self.starts + shard
        self.pos = self.starts.copy()
        self.wraps = 0

    def state(self):
        return self.pos.tolist(), self.wraps

    def restore(self, pos, wraps=0):
        """Restore per-row cursors. Only valid if the batch size still matches."""
        pos = np.asarray(pos, dtype=np.int64)
        if pos.shape != self.pos.shape:
            # Silently ignoring this would resume the *weights* while quietly
            # rewinding the *data* to token 0 — the model would re-read text it
            # has already seen and nothing in the logs would say so.
            print(f"⚠️  Stream position not restored: checkpoint has "
                  f"{pos.shape[0] if pos.ndim else 0} rows, this run has "
                  f"{self.batch}. Re-run with the checkpoint's batch size to "
                  f"continue where it left off; training will otherwise "
                  f"restart the corpus from the beginning.")
            return
        self.pos = np.clip(pos, self.starts, self.limits - self.chunk - 1)
        self.wraps = wraps

    def next(self):
        """Returns (x, y, wrapped) — wrapped rows need their state reset."""
        wrapped = False
        rows_x = np.empty((self.batch, self.chunk), dtype=np.int64)
        rows_y = np.empty((self.batch, self.chunk), dtype=np.int64)
        for b in range(self.batch):
            if self.pos[b] + self.chunk + 1 > self.limits[b]:
                self.pos[b] = self.starts[b]
                wrapped = True
            p = self.pos[b]
            window = np.asarray(self.tokens[p:p + self.chunk + 1], dtype=np.int64)
            rows_x[b] = window[:-1]
            rows_y[b] = window[1:]
            self.pos[b] = p + self.chunk
        if wrapped:
            self.wraps += 1
        x = torch.from_numpy(rows_x).to(self.device, non_blocking=True)
        y = torch.from_numpy(rows_y).to(self.device, non_blocking=True)
        return x, y, wrapped

    def tokens_per_step(self):
        return self.batch * self.chunk


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

def build(cfg, vocab_size):
    input_ids, output_ids = cfg.io_ids()
    model = OdyssNet(
        num_neurons=cfg.neurons,
        input_ids=input_ids,
        output_ids=output_ids,
        device=cfg.device,
        pulse_mode=cfg.pulse_mode,
        activation=list(cfg.activation),
        weight_init=list(cfg.weight_init),
        gate=list(cfg.gates),
        hebb_type=cfg.hebb_type,
        hebb_res=cfg.hebb_res,
        attn_heads=cfg.attn_heads or None,
        attn_kv_heads=cfg.attn_kv_heads,
        attn_head_dim=cfg.attn_head_dim or None,
        attn_window=cfg.attn_window,
        attn_write=cfg.attn_write,
        attn_read=cfg.attn_read,
        attn_rope=cfg.attn_rope,
        attn_qk_norm=cfg.attn_qk_norm,
        attn_dropout=cfg.attn_dropout,
        dropout_rate=cfg.dropout,
        gradient_checkpointing=cfg.grad_ckpt,
        vocab_size=vocab_size,
        vocab_mode="discrete",
        tie_embeddings=cfg.tie_embeddings,
    )
    if cfg.compile:
        # Compile the bound method rather than the module: everything else in
        # this script reaches through `model` for state, caches, plastic
        # buffers and checkpoints, and an OptimizedModule wrapper would sit
        # between them and the real object.
        model.forward = torch.compile(model.forward)

    trainer = OdyssNetTrainer(model, lr=cfg.lr, device=cfg.device,
                              gradient_persistence=cfg.grad_persistence)
    trainer.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    return model, trainer


def snapshot_state(model):
    """Capture the recurrent (and plastic) state so eval can't disturb training."""
    snap = {"state": model.state.clone()}
    if model.hebb_type is not None:
        for name in ("t_hebb_state_W", "t_hebb_state_mem",
                     "s_hebb_state_W", "s_hebb_state_mem"):
            buf = getattr(model, name, None)
            if buf is not None:
                snap[name] = buf.clone()
    # The KV cache is recurrent state too, and evaluation runs at a different
    # batch size — without this, scoring a checkpoint would leave the training
    # stream reading somebody else's history.
    if getattr(model, "attn", None) is not None:
        snap["attn"] = model.attn.snapshot()
    return snap


def restore_state(model, snap):
    model.state = snap["state"]
    for name, buf in snap.items():
        if name == "state":
            continue
        if name == "attn":
            model.attn.restore(buf)
            continue
        getattr(model, name).copy_(buf)


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

class Validator:
    """
    Fixed held-out evaluation. Deterministic across runs and configurations,
    so numbers from different arms are directly comparable.
    """

    def __init__(self, cfg, val_tokens, tokenizer):
        self.cfg = cfg
        self.chunk = cfg.chunk
        self.batch = min(cfg.val_batch, max(1, len(val_tokens) // (cfg.chunk * 4)))
        usable = min(len(val_tokens), cfg.val_tokens)
        self.tokens = np.asarray(val_tokens[:usable], dtype=np.int64)

        shard = len(self.tokens) // self.batch
        self.n_chunks = (shard - 1) // self.chunk
        if self.n_chunks <= cfg.val_warmup_chunks:
            raise ValueError("Validation split too small for this chunk/batch size.")
        self.starts = [b * shard for b in range(self.batch)]
        self.scored_chunks = self.n_chunks - cfg.val_warmup_chunks

        # Bytes covered by the *scored* token windows only — makes
        # bits-per-byte exact rather than approximately right.
        skip = cfg.val_warmup_chunks * self.chunk
        span = self.scored_chunks * self.chunk
        nbytes = 0
        for s in self.starts:
            ids = self.tokens[s + skip: s + skip + span].tolist()
            nbytes += len(tokenizer.decode(ids).encode("utf-8"))
        self.scored_bytes = max(1, nbytes)
        self.scored_tokens = self.batch * span

    @torch.no_grad()
    def run(self, model):
        cfg = self.cfg
        was_training = model.training
        snap = snapshot_state(model)
        model.eval()
        # TF32 off for the same reason AMP is off: the score is the thing runs
        # are ranked by, so it should not move with kernel choice. Restored on
        # the way out so training keeps its speed.
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return self._score(model)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            restore_state(model, snap)
            if was_training:
                model.train()

    def _score(self, model):
        cfg = self.cfg

        vocab = model.output_decoder.out_features
        steps = self.chunk * (cfg.think_gap + 1)
        model.reset_state(batch_size=self.batch)

        total_nll = 0.0
        shard_nll = torch.zeros(self.batch, device=cfg.device)
        # Validation runs in full fp32 with TF32 off (see run()). Not because
        # precision was ever the bug — measured, AMP on/off and TF32 on/off
        # agree to four decimals — but because eval throughput is irrelevant
        # here and determinism is worth more than the speed.
        #
        # (The batch-size dependence that led here was *not* numerics: eval
        # batch size sets how many shards the val split is cut into and
        # therefore which corpus positions each shard starts at, and only some
        # starting positions fall into the absorbing state. See val_coldfail.)
        amp = torch.amp.autocast(
            device_type="cuda" if cfg.device.startswith("cuda") else "cpu",
            enabled=cfg.val_amp and cfg.device.startswith("cuda"))
        for c in range(self.n_chunks):
            off = c * self.chunk
            xs = np.stack([self.tokens[s + off: s + off + self.chunk] for s in self.starts])
            ys = np.stack([self.tokens[s + off + 1: s + off + self.chunk + 1] for s in self.starts])
            x = torch.from_numpy(xs).to(cfg.device)
            y = torch.from_numpy(ys).to(cfg.device).reshape(-1)

            with amp:
                logits, _ = model(x, steps=steps, return_sequence=True)
            if c < cfg.val_warmup_chunks:
                continue
            # Per-shard, not just pooled: every shard is an independent cold
            # start, so the spread across shards *is* the cold-start
            # robustness measurement. A pooled mean hides the failure mode —
            # a handful of shards stuck at ~240 nats swamp the average and
            # look like "the model diverged" when most shards are healthy.
            per_row = F.cross_entropy(
                logits.reshape(-1, vocab).float(), y,
                reduction="none").view(self.batch, -1).sum(dim=1)
            shard_nll += per_row
            total_nll += float(per_row.sum())

        span = self.scored_chunks * self.chunk
        shard_loss = (shard_nll / span).tolist()
        nll_per_token = total_nll / self.scored_tokens
        median = sorted(shard_loss)[len(shard_loss) // 2]
        failed = sum(1 for s in shard_loss if s > FAIL_NATS)
        return {
            "val_loss": nll_per_token,
            "val_ppl": math.exp(min(nll_per_token, 30.0)),
            "val_bpb": total_nll / math.log(2.0) / self.scored_bytes,
            # Median shard: what the model does on a *typical* cold start.
            "val_p50": median,
            # Fraction of cold starts that collapsed entirely.
            "val_coldfail": failed / len(shard_loss),
        }


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate(model, tokenizer, cfg, prompt="Once upon a time", length=160,
             temperature=0.8, top_k=40, top_p=0.9):
    was_training = model.training
    snap = snapshot_state(model)
    model.eval()

    # A prompt that encodes to nothing still needs a starting token. Ask the
    # tokenizer: id 0 is "<s>" in the trained BPE and an ordinary byte under
    # tiktoken.
    ids = tokenizer.encode(prompt) or [tokenizer.bos_id]
    gap = cfg.think_gap + 1
    model.reset_state(batch_size=1)

    x = torch.tensor([ids], dtype=torch.long, device=cfg.device)
    model(x, steps=len(ids) * gap, return_sequence=False)

    last = ids[-1]
    for _ in range(length):
        step_in = torch.tensor([[last]], dtype=torch.long, device=cfg.device)
        logits, _ = model(step_in, steps=gap, return_sequence=True)
        logits = logits[0, -1, :].float()

        if temperature > 0:
            logits = logits / temperature
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if 0 < top_p < 1.0:
            order = torch.argsort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(logits[order], dim=-1), dim=-1)
            drop = cum > top_p
            drop[1:] = drop[:-1].clone()
            drop[0] = False
            logits = logits.masked_fill(
                torch.zeros_like(drop).scatter(0, order, drop), float("-inf"))

        probs = torch.softmax(logits, dim=0)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:
            probs = torch.ones_like(probs) / probs.numel()
        last = int(torch.multinomial(probs, 1).item())
        ids.append(last)

    restore_state(model, snap)
    if was_training:
        model.train()
    return tokenizer.decode(ids)


# --------------------------------------------------------------------------- #
# Empirical batch autotune                                                    #
# --------------------------------------------------------------------------- #

def autotune_batch(cfg, vocab_size, train_tokens,
                   ladder=(8, 32, 128, 256, 512, 1024, 2048, 4096)):
    """
    Measure, don't model. This network is latency-bound — a chunk is
    ``chunk x (think_gap+1)`` *sequential* small matmuls, so the GPU is idle
    between kernel launches and extra batch rows ride along nearly free.
    Measured on a 3060 Ti, throughput scales ~linearly from batch 8 to 512
    (6k -> 274k tok/s); no analytic VRAM formula predicts that, which is why
    this probes instead of estimating.

    Note this maximizes *throughput*, not learning. Bigger batches buy tokens
    but spend fewer optimizer steps on them — run ``--mode sweep --sweep batch``
    to see where that trade actually turns against you.
    """
    print("\n⚖️  Autotuning batch size (measured tokens/s, not estimated)")
    set_seed(cfg.seed)
    model, trainer = build(cfg, vocab_size)
    steps = cfg.chunk * (cfg.think_gap + 1)
    flatten = lambda o: o.reshape(-1, o.shape[-1])

    best, best_rate = ladder[0], 0.0
    for b in ladder:
        try:
            shards = StreamShards(train_tokens, b, cfg.chunk, cfg.device)
        except ValueError:
            break
        try:
            for i in range(3):                                  # warmup
                x, y, _ = shards.next()
                trainer.train_batch(x, y.reshape(-1), thinking_steps=steps,
                                    full_sequence=True, output_transform=flatten,
                                    keep_state=(i > 0))
            _sync(cfg.device)
            t0 = time.time()
            for _ in range(4):
                x, y, _ = shards.next()
                trainer.train_batch(x, y.reshape(-1), thinking_steps=steps,
                                    full_sequence=True, output_transform=flatten,
                                    keep_state=True)
            _sync(cfg.device)
            rate = 4 * b * cfg.chunk / (time.time() - t0)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"   batch {b:>4}: OOM — stopping ladder")
            trainer.optimizer.zero_grad(set_to_none=True)
            _empty_cache(cfg.device)
            break

        mark = ""
        if rate > best_rate * 1.03:
            best, best_rate, mark = b, rate, "  <-- best"
        print(f"   batch {b:>4}: {rate:9,.0f} tok/s{mark}")
        if rate < best_rate * 0.85:            # past the knee, stop burning time
            break

    del model, trainer
    _empty_cache(cfg.device)
    print(f"   selected batch = {best} ({best_rate:,.0f} tok/s)")
    return best


def _sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _empty_cache(device):
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Training session                                                            #
# --------------------------------------------------------------------------- #

#: Everything that defines how a checkpoint was built and scored; adopted by
#: `eval`/`gen`, whose job is to reproduce that. `tokenizer` is here for a
#: different reason than the rest: it changes no tensor shape, but decoding
#: with the wrong one yields fluent-looking garbage and no error.
ARCH_FIELDS = ("neurons", "n_in", "n_out", "tokenizer", "vocab_size",
               "activation", "weight_init", "gates", "hebb_type", "hebb_res",
               "tie_embeddings", "think_gap", "pulse_mode",
               "attn_heads", "attn_kv_heads", "attn_head_dim", "attn_window",
               "attn_write", "attn_read", "attn_rope", "attn_qk_norm")

#: The subset `--resume` adopts: what `build()` allocates, plus which corpus is
#: read. The rest cannot change the state dict, so pinning them would only take
#: away a knob — `weight_init` is dead once weights load, `think_gap` and
#: `pulse_mode` are forward-pass arguments, and every accepted activation is
#: parameter-free.
#: `gates` is not in that group and is adopted: it adds or removes `core_gate`,
#: `input_gate`, `output_gate` and `memory_gate`.
#: The attention fields split the same way: heads / kv_heads / head_dim size
#: the projections and qk_norm adds two RMSNorm gains, so all four are adopted;
#: window, write mode and RoPE change no tensor and stay yours to re-choose on
#: a resume.
RESUME_FIELDS = ("neurons", "n_in", "n_out", "tokenizer", "vocab_size",
                 "gates", "hebb_type", "hebb_res", "tie_embeddings",
                 "attn_heads", "attn_kv_heads", "attn_head_dim", "attn_qk_norm")


def adopt_saved_arch(cfg, path, fields):
    """
    Override architecture fields with the ones stored in a checkpoint.

    `fields` is required, not defaulted: the two callers want different sets,
    and a third silently inheriting the broader one would over-adopt — failing
    in the direction that removes a knob quietly rather than the one that
    raises.
    """
    saved = torch.load(path, map_location="cpu").get("cfg")
    if not saved:
        return cfg
    changes = {}
    for k in fields:
        if k not in saved:
            continue
        want = tuple(saved[k]) if isinstance(getattr(cfg, k), tuple) else saved[k]
        if want != getattr(cfg, k):
            changes[k] = want
    if changes:
        print(f"🔧 Adopting checkpoint architecture: {changes}")
    return replace(cfg, **changes)


def ckpt_paths(cfg):
    os.makedirs(CKPT_DIR, exist_ok=True)
    base = os.path.join(CKPT_DIR, f"llm_odyss_{cfg.tag}")
    return base + "_latest.pth", base + "_best.pth"


def guard_overwrite(cfg, overwrite=False):
    """
    Refuse to start a fresh run on top of an existing tag's checkpoints.

    Without this, `--mode train --tag base` (no --resume) rebuilds the model
    from scratch with best_val = inf, so the very first validation counts as a
    new record and overwrites both <tag>_best.pth and <tag>_latest.pth — a
    25-minute run destroyed ~35 seconds in, with nothing in the logs to warn
    you. Checked before the batch autotune so it fails in a second rather than
    after a two-minute probe.

    Deliberately a hard exit and not an interactive prompt: the blocking
    input() this rewrite removed is exactly what made the old script
    unusable from a script or a CI job.
    """
    latest, best = ckpt_paths(cfg)
    existing = [p for p in (latest, best) if os.path.exists(p)]
    if not existing or overwrite:
        return
    raise SystemExit(
        f"\n✋ Refusing to overwrite existing checkpoints for tag '{cfg.tag}':\n"
        + "".join(f"     {p}\n" for p in existing)
        + f"\n   Continue that run:   --mode train --tag {cfg.tag} --resume "
          f"--batch <same as checkpoint>\n"
        f"   Start somewhere new: --mode train --tag {cfg.tag}_v2\n"
        f"   Overwrite anyway:    --mode train --tag {cfg.tag} --overwrite\n"
    )


def run_session(cfg, corpus, budget_sec=0.0, resume=False, resume_best=False,
                quiet=False, sample_every=0, save=True):
    """
    Train under a wall-clock and/or step budget. Returns the final metrics.

    Used by --mode train (open-ended) and by every sweep arm (budgeted), so
    both paths exercise exactly the same code.
    """
    train_tokens, val_tokens, tokenizer = corpus
    vocab_size = tokenizer.get_vocab_size()

    set_seed(cfg.seed)
    if cfg.batch <= 0:
        cfg = replace(cfg, batch=autotune_batch(cfg, vocab_size, train_tokens))
        set_seed(cfg.seed)

    model, trainer = build(cfg, vocab_size)
    validator = Validator(cfg, val_tokens, tokenizer)
    shards = StreamShards(train_tokens, cfg.batch, cfg.chunk, cfg.device)
    latest_path, best_path = ckpt_paths(cfg)

    best_val = float("inf")
    start_step = 0
    # `_latest` is rewritten at every evaluation and `_best` only on an
    # improvement, so the two diverge exactly when it matters: a loss
    # explosion overwrites `_latest` within one eval interval, while `_best`
    # structurally cannot hold a diverged model. --resume-best is the recovery
    # path that difference makes possible.
    resume_path = best_path if resume_best else latest_path
    flag = "--resume-best" if resume_best else "--resume"
    if resume and not os.path.exists(resume_path):
        # Starting fresh is harmless when the tag has nothing at all, but with
        # a latest present it would train a random model and overwrite that
        # file at the first evaluation — the destruction the flag exists to
        # avoid, arrived at by asking to avoid it.
        if resume_best and os.path.exists(latest_path):
            raise SystemExit(
                f"\n✋ --resume-best: no best checkpoint at {resume_path}, but "
                f"tag '{cfg.tag}' has a latest one.\n"
                f"   Starting fresh would overwrite it at the first "
                f"evaluation, so this run stops instead.\n"
                f"   Continue from latest:  --mode train --tag {cfg.tag} "
                f"--resume\n"
            )
        print(f"ℹ️  {flag} given but no checkpoint at {resume_path}; "
              f"starting fresh.")
    if resume and os.path.exists(resume_path):
        # Loading the weights must not fail softly. --resume bypasses
        # guard_overwrite, so a run that continues past a failed load trains a
        # randomly initialized model and, at the first validation, saves it
        # over the checkpoint it could not read.
        try:
            data = load_checkpoint(model, trainer.optimizer, resume_path,
                                   device=cfg.device, strict=True, lr=cfg.lr)
        except Exception as e:
            raise SystemExit(
                f"\n✋ Could not load the checkpoint for tag '{cfg.tag}':\n"
                f"     {e}\n\n"
                f"   The file is intact; this run refuses to overwrite it with "
                f"a fresh model.\n"
                f"   Architecture is read from the checkpoint, so a shape "
                f"mismatch here usually means the\n"
                f"   file predates that record or was written by a different "
                f"version.\n"
                f"   Train elsewhere:   --mode train --tag {cfg.tag}_v2\n"
                f"   Start over anyway: --mode train --tag {cfg.tag} --overwrite "
                f"(destroys it)\n"
            ) from e

        if cfg.lr is None and cfg.lr_force_auto:
            # Explicit `--lr auto`: return a fixed-rate checkpoint to the
            # estimator. load_checkpoint(lr=None) means "don't touch", so
            # this is the only path back to auto; ChaosGrad lazily builds
            # the estimator state it never created under fixed rate.
            switched = any(g.get('lr') is not None
                           for g in trainer.optimizer.param_groups)
            for g in trainer.optimizer.param_groups:
                g['lr'] = None
            if switched:
                print("🔁 --lr auto: checkpoint was fixed-rate; optimizer "
                      "returned to ChaosGrad's online estimate.")
        extra = data.get("stream", {})
        shards.restore(extra.get("pos", []), extra.get("wraps", 0))
        best_val = data.get("best_val", float("inf"))
        start_step = int(data.get("step", 0))

        # Deliberately outside the fatal guard: this payload can gain or lose
        # keys between versions, and a failure here costs counters rather than
        # weights, with nothing trained yet.
        ts = data.get("trainer_state_dict")
        if ts:
            try:
                trainer.load_state_dict(ts)
            except Exception as e:
                print(f"⚠️  Weights and optimizer resumed, but the trainer "
                      f"state did not ({e}). Continuing with the loaded model; "
                      f"AMP scaler and anomaly-detector history restart.")
        print(f"📂 Resumed from {os.path.basename(resume_path)} at step "
              f"{start_step:,} (best val PPL {math.exp(min(best_val,30)):.2f})")
        if resume_best:
            print(f"   ↩️  Rewound to the best checkpoint: step counter, "
                  f"stream positions and optimizer state all come from it. "
                  f"The next evaluation rewrites {os.path.basename(latest_path)}.")

    # Report the mode the optimizer is ACTUALLY in, not what the CLI asked
    # for. Under the default `--lr keep`, a resumed checkpoint keeps its own
    # mode (`load_checkpoint(lr=None)` means "don't touch"), so printing
    # cfg.lr here once claimed "lr auto" over a run that was really
    # fixed-rate. Explicit `--lr auto` / `--lr <float>` do override, and the
    # resume path above already announced any switch.
    live_lr = trainer.optimizer.param_groups[0].get('lr') \
        if trainer.optimizer.param_groups else cfg.lr
    if not quiet:
        vocab_advisory(cfg, vocab_size)
        attention_advisory(cfg, model)
        plasticity_advisory(cfg, model)
        print(f"\n🧠 {model.get_num_params():,} trainable params | "
              f"{cfg.neurons} neurons | in {cfg.n_in} / out {cfg.n_out} | "
              f"gap {cfg.think_gap} | chunk {cfg.chunk} | batch {cfg.batch} | "
              f"lr {'auto' if live_lr is None else live_lr}"
              + (f" | hebb {cfg.hebb_type}/{cfg.hebb_res}" if cfg.hebb_type else ""))
        if (live_lr is None) != (cfg.lr is None):
            print(f"ℹ️  Resumed optimizer keeps the checkpoint's "
                  f"{'auto' if live_lr is None else f'fixed-rate ({live_lr})'} "
                  f"mode; pass an explicit --lr to change it.")

    steps_per_chunk = cfg.chunk * (cfg.think_gap + 1)
    flatten = lambda o: o.reshape(-1, o.shape[-1])

    metrics = {k: float("nan") for k in
               ("val_loss", "val_ppl", "val_bpb", "val_p50", "val_coldfail")}
    tokens_seen = 0
    step = start_step
    running = 0.0
    running_n = 0
    keep = False
    t_start = time.time()
    interrupted = False

    try:
        while True:
            if budget_sec and time.time() - t_start >= budget_sec:
                break
            if cfg.max_steps and step - start_step >= cfg.max_steps:
                break

            x, y, wrapped = shards.next()
            if wrapped:
                keep = False           # a shard restarted; its state is stale
            elif keep and cfg.cold_start_every:
                # Staggered cold starts. Without them the training state is
                # carried forever and the model never sees a zero state, so
                # gradient descent puts no pressure on cold-start behaviour —
                # measured consequence: some starting contexts fall into a
                # degenerate absorbing state (~240 nats) that training loss
                # cannot see, because training never re-enters it. Resetting
                # rows independently keeps a mix of cold and warm contexts in
                # every batch and gives context length an exponential spread.
                hit = torch.rand(model.state.shape[0], device=model.state.device) \
                    < (1.0 / cfg.cold_start_every)
                # reset_rows, not a masked_fill on the state: with attention on,
                # a row whose state is zeroed but whose KV history is intact is
                # neither cold nor warm, and the cold start being simulated
                # would not be one.
                model.reset_rows(hit)

            loss = trainer.train_batch(
                x, y.reshape(-1),
                thinking_steps=steps_per_chunk,
                full_sequence=True,
                output_transform=flatten,
                keep_state=keep,
            )
            keep = True
            step += 1
            tokens_seen += shards.tokens_per_step()
            running += loss
            running_n += 1

            if not quiet and cfg.log_every and step % cfg.log_every == 0:
                el = time.time() - t_start
                avg = running / max(running_n, 1)
                # The effective LR is ChaosGrad's live estimate under lr=None.
                # Watching it is the cheapest divergence tripwire there is:
                # a runaway estimate shows up here before the loss does.
                print(f"step {step:>7,} | loss {avg:6.4f} | ppl {math.exp(min(avg,30)):8.2f} | "
                      f"lr {trainer._current_lr():.2e} | "
                      f"{tokens_seen/max(el,1e-6):7,.0f} tok/s | {tokens_seen/1e6:6.2f}M tok | "
                      f"{el/60:5.1f}m")
                running, running_n = 0.0, 0

            if cfg.eval_every and step % cfg.eval_every == 0:
                metrics = validator.run(model)
                mark = ""
                if metrics["val_loss"] < best_val:
                    best_val = metrics["val_loss"]
                    mark = "  🏆"
                    if save:
                        _save(cfg, model, trainer, shards, step, best_val, metrics, best_path)
                if not quiet:
                    print(f"   ↳ VAL  loss {metrics['val_loss']:.4f} | "
                          f"ppl {metrics['val_ppl']:.2f} | bpb {metrics['val_bpb']:.4f} | "
                          f"p50 {metrics['val_p50']:.3f} | "
                          f"coldfail {metrics['val_coldfail']:.1%}{mark}")
                if save:
                    _save(cfg, model, trainer, shards, step, best_val, metrics, latest_path)
                if sample_every and (step // cfg.eval_every) % sample_every == 0:
                    print("   ── sample ──")
                    print("   " + generate(model, tokenizer, cfg, length=90)
                          .replace("\n", " ")[:400])
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏹️  Interrupted — finishing cleanly.")

    elapsed = max(time.time() - t_start, 1e-6)
    if step > start_step:
        metrics = validator.run(model)
        if save:
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                _save(cfg, model, trainer, shards, step, best_val, metrics, best_path)
            _save(cfg, model, trainer, shards, step, best_val, metrics, latest_path)

    metrics.update({
        "tokens": tokens_seen,
        "steps": step - start_step,
        "tok_s": tokens_seen / elapsed,
        "minutes": elapsed / 60.0,
        "params": model.get_num_params(),
        "batch": cfg.batch,
        "interrupted": interrupted,
    })
    return metrics, model, trainer, tokenizer


def _save(cfg, model, trainer, shards, step, best_val, metrics, path):
    pos, wraps = shards.state()
    save_checkpoint(
        model, trainer.optimizer, step, metrics.get("val_loss", float("nan")), path,
        extra_data={
            "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(cfg).items()},
            "stream": {"pos": pos, "wraps": wraps},
            "best_val": best_val,
            "step": step,
            "metrics": metrics,
            "trainer_state_dict": trainer.state_dict(),
        },
    )


# --------------------------------------------------------------------------- #
# Sweeps                                                                      #
# --------------------------------------------------------------------------- #

SWEEPS = {
    # The thesis test: is temporal depth worth its wall-clock? Every arm gets
    # the same number of seconds, so more thinking steps means fewer tokens.
    "gap": [
        ("gap0", dict(think_gap=0)),
        ("gap1", dict(think_gap=1)),
        ("gap3", dict(think_gap=3)),
        ("gap5", dict(think_gap=5)),
    ],
    # Architecture / recipe choices, all against the same baseline.
    "arch": [
        ("baseline",   dict()),
        ("readout128", dict(n_out=128)),
        ("core_gate",  dict(gates=("none", "sigmoid", "identity"))),
        ("mem_off",    dict(gates=("none", "none", "none"))),
        ("hebb_temp",  dict(hebb_type="temporal", hebb_res="neuron")),
        ("hebb_both",  dict(hebb_type="both", hebb_res="neuron")),
        ("lr_fixed",   dict(lr=1e-4)),
        ("tied_embed", dict(tie_embeddings=True, n_in=256, n_out=256)),
    ],
    # Does training on cold starts remove the degenerate absorbing state?
    # `never` carries the state for the entire run.
    "coldstart": [
        ("never", dict(cold_start_every=0)),
        ("c128",  dict(cold_start_every=128)),
        ("c32",   dict(cold_start_every=32)),
        ("c8",    dict(cold_start_every=8)),
    ],
    # ChaosGrad's zero-config estimator vs a pinned rate. The auto estimate
    # climbs to its traction cap (~3.9e-3 here) within ~300 steps and pins
    # there; that is the regime the absorbing state develops in, so it is
    # worth knowing what a lower pinned rate costs in exchange.
    "optim": [
        ("auto_b128",   dict(batch=128)),
        ("auto_b1024",  dict(batch=1024)),
        ("fix1e4_b128", dict(batch=128, lr=1e-4)),
        ("fix1e4_b1k",  dict(batch=1024, lr=1e-4)),
        ("fix3e4_b1k",  dict(batch=1024, lr=3e-4)),
        ("fix1e3_b1k",  dict(batch=1024, lr=1e-3)),
    ],
    # Throughput is not learning. Big batches buy tokens but spend fewer
    # optimizer steps on them; this finds where that trade turns.
    "batch": [
        ("b64",   dict(batch=64)),
        ("b128",  dict(batch=128)),
        ("b256",  dict(batch=256)),
        ("b512",  dict(batch=512)),
        ("b1024", dict(batch=1024)),
    ],
    # Does letting the state query its own past pay for the step it costs?
    # `off` is the 2.x architecture and the only arm without a KV cache; the
    # rest hold everything else fixed, so a difference here is attention's.
    "attn": [
        ("off",        dict(attn_heads=0)),
        ("mqa4",       dict(attn_heads=4)),
        ("mha4",       dict(attn_heads=4, attn_kv_heads=4)),
        ("window32",   dict(attn_heads=4, attn_window=32)),
        ("no_rope",    dict(attn_heads=4, attn_rope=False)),
        ("read_token",  dict(attn_heads=4, attn_read="token")),
        ("write_step", dict(attn_heads=4, attn_write="step")),
    ],
    # Does the truncated-BPTT window length matter more than depth?
    "window": [
        ("chunk16",  dict(chunk=16)),
        ("chunk48",  dict(chunk=48)),
        ("chunk128", dict(chunk=128)),
        ("ckpt_on",  dict(chunk=128, grad_ckpt=True)),
    ],
}


def run_sweep(cfg, corpus, name, minutes, arms=None):
    plan = SWEEPS[name]
    if arms:
        wanted = {a.strip() for a in arms.split(",")}
        plan = [p for p in plan if p[0] in wanted]
        if not plan:
            raise SystemExit(f"No arms named {sorted(wanted)} in sweep '{name}'")

    budget = minutes * 60.0
    print(f"\n{'='*78}")
    print(f"🔬 SWEEP '{name}' — {len(plan)} arms x {minutes:.1f} min "
          f"(compute-matched, seed {cfg.seed})")
    print(f"{'='*78}")

    results = []
    for i, (arm, over) in enumerate(plan, 1):
        # eval_every=0: mid-run validation would spend budget, and each arm
        # would spend a *different* amount of it. Every arm is scored once,
        # after the clock stops.
        arm_cfg = replace(cfg, tag=f"sweep_{name}_{arm}", eval_every=0, **over)
        print(f"\n[{i}/{len(plan)}] ▶ {arm}: {over or 'baseline'}")
        try:
            m, model, _, tokenizer = run_session(
                arm_cfg, corpus, budget_sec=budget, quiet=True, save=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"   ✗ OOM — arm skipped")
            _empty_cache(cfg.device)
            continue
        m["arm"] = arm
        results.append(m)
        print(f"   loss {m['val_loss']:.4f} | ppl {m['val_ppl']:8.2f} | "
              f"bpb {m['val_bpb']:.4f} | p50 {m['val_p50']:.3f} | "
              f"coldfail {m['val_coldfail']:.1%} | {m['tokens']/1e6:.2f}M tok | "
              f"{m['tok_s']:,.0f} tok/s | {m['params']:,} params")
        print("   " + generate(model, tokenizer, arm_cfg, length=60).replace("\n", " ")[:220])
        del model
        _empty_cache(cfg.device)

    if not results:
        return results

    # Ranked by median shard, not the pooled mean: the mean is dominated by
    # however many cold starts collapsed, which `coldfail` reports separately
    # and far more legibly. Read the two columns together — a low p50 with a
    # high coldfail is a model that is good when it works and unreliable.
    results.sort(key=lambda r: (r["val_p50"], r["val_coldfail"]))
    print(f"\n{'='*92}")
    print(f"🏁 SWEEP '{name}' RESULTS — ranked by median cold-start shard loss")
    print(f"{'='*92}")
    print(f"{'arm':<12} {'p50 loss':>9} {'p50 ppl':>9} {'coldfail':>9} "
          f"{'pooled':>9} {'bits/byte':>10} {'Mtokens':>9} {'tok/s':>9}")
    print("-" * 92)
    for r in results:
        print(f"{r['arm']:<12} {r['val_p50']:>9.4f} "
              f"{math.exp(min(r['val_p50'], 30)):>9.2f} {r['val_coldfail']:>8.1%} "
              f"{r['val_loss']:>9.3f} {r['val_bpb']:>10.4f} "
              f"{r['tokens']/1e6:>9.2f} {r['tok_s']:>9,.0f}")
    print("-" * 92)
    print(f"🥇 {results[0]['arm']}")

    out = os.path.join(CKPT_DIR, f"sweep_{name}_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📄 {out}")
    return results


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

def run_smoke(cfg, corpus):
    """
    Fast end-to-end check of every path the heavy run depends on: streaming,
    training, stateful TBPTT, validation, generation, checkpoint round-trip,
    Hebbian plasticity and gates. Seconds, not hours — this is what makes the
    expensive script safe to modify.
    """
    print(f"\n{'='*78}")
    print("🔥 SMOKE TEST")
    print(f"{'='*78}")

    # Clear our own litter first. The round-trip below trains with save=True
    # and no --resume, so leftovers from a previous smoke run would trip
    # guard_overwrite and fail the test on the second invocation. Running
    # smoke twice in a row has to be a no-op, or it isn't a smoke test.
    for stale in glob.glob(os.path.join(CKPT_DIR, "llm_odyss_smoke*.pth")):
        os.remove(stale)

    base = replace(
        cfg, neurons=96, n_in=32, n_out=48, chunk=8, batch=4, think_gap=1,
        max_steps=12, eval_every=6, log_every=6, val_tokens=20000, val_batch=8,
        val_warmup_chunks=1, tag="smoke",
    )

    variants = [
        ("plain",     dict()),
        ("hebbian",   dict(hebb_type="both", hebb_res="neuron")),
        ("gated",     dict(gates=("none", "sigmoid", "sigmoid"))),
        ("grad_ckpt", dict(grad_ckpt=True, lr=1e-4)),
        ("gap0",      dict(think_gap=0)),
        ("attn",      dict(attn_heads=4)),
        # Everything the attention path can be configured into that changes
        # its shapes or its bookkeeping: multi-head instead of multi-query, a
        # window narrower than the chunk (so eviction actually happens mid-run),
        # an entry per thinking step, and no rotary positions.
        ("attn_mha",  dict(attn_heads=4, attn_kv_heads=4, attn_window=4,
                           attn_write="step", attn_rope=False)),
        ("attn_ckpt", dict(attn_heads=2, grad_ckpt=True, lr=1e-4)),
    ]

    ok = True
    for name, over in variants:
        c = replace(base, **over)
        t0 = time.time()
        try:
            m, model, trainer, tok = run_session(c, corpus, quiet=True, save=False)
            txt = generate(model, tok, c, prompt="Once", length=12)
            assert math.isfinite(m["val_loss"]), "non-finite validation loss"
            assert m["steps"] == c.max_steps, f"ran {m['steps']} of {c.max_steps} steps"
            diag = trainer.get_diagnostics()
            print(f"  ✅ {name:<10} loss {m['val_loss']:6.3f} | p50 {m['val_p50']:6.3f} | "
                  f"bpb {m['val_bpb']:5.3f} | lr {diag['current_lr']:.2e} | "
                  f"{time.time()-t0:4.1f}s | {txt[:40]!r}")
        except Exception as e:
            ok = False
            print(f"  ❌ {name:<10} {type(e).__name__}: {e}")

    # Checkpoint round-trip: save, rebuild, reload, and require identical loss.
    # Run once per architecture that changes the state dict — attention adds
    # six tensors and a runtime cache, and a checkpoint that silently drops
    # either is a corrupted resume rather than an error.
    print("\n  → checkpoint round-trip")
    for label, over in (("plain", dict()), ("attn", dict(attn_heads=4))):
        try:
            c = replace(base, tag=f"smoke_ckpt_{label}", eval_every=0, **over)
            _, model1, _, _ = run_session(c, corpus, quiet=True, save=True)
            validator = Validator(c, corpus[1], corpus[2])
            v1 = validator.run(model1)["val_loss"]
            del model1

            model2, trainer2 = build(c, corpus[2].get_vocab_size())
            load_checkpoint(model2, trainer2.optimizer, ckpt_paths(c)[0],
                            device=c.device, strict=True)
            v2 = validator.run(model2)["val_loss"]
            if abs(v1 - v2) < 1e-4:
                print(f"  ✅ round-trip {label:<6} {v1:.6f} == {v2:.6f}")
            else:
                ok = False
                print(f"  ❌ round-trip {label:<6} {v1:.6f} != {v2:.6f}")
        except Exception as e:
            ok = False
            print(f"  ❌ round-trip {label:<6} {type(e).__name__}: {e}")

    # Learning check: the loss must actually move on a memorizable slice.
    print("\n  → learning check (200 steps on a tiny slice)")
    c = replace(base, max_steps=200, eval_every=0, log_every=0, tag="smoke_learn")
    m, model, trainer, tok = run_session(c, corpus, quiet=True, save=False)
    v = Validator(c, corpus[1], corpus[2]).run(model)
    uniform = math.log(corpus[2].get_vocab_size())
    print(f"  {'✅' if v['val_loss'] < uniform else '❌'} val loss "
          f"{v['val_loss']:.3f} vs uniform-guess {uniform:.3f}")
    ok = ok and v["val_loss"] < uniform

    print(f"\n{'🎉 SMOKE PASSED' if ok else '💥 SMOKE FAILED'}")
    return ok


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

EPILOG = """\
examples:
  # end-to-end self-test; safe to run twice, touches no real checkpoint
  %(prog)s --mode smoke
  %(prog)s --mode smoke --device cpu

  # start a fresh run under a new tag (refuses to clobber an existing one)
  %(prog)s --mode train --tag base --batch 256 --minutes 25

  # continue it -- --batch must match the checkpoint or the corpus rewinds
  %(prog)s --mode train --tag base --resume --batch 256 --minutes 25

  # continue with a pinned rate instead of ChaosGrad's estimate
  %(prog)s --mode train --tag base --resume --batch 256 --lr 1.5e-3

  # continue with the state never reset (longer context, riskier dynamics --
  # watch the val_coldfail column)
  %(prog)s --mode train --tag base --resume --batch 256 --cold-start-every 0

  # recover from a loss explosion: rewind to the best checkpoint, pin the rate
  %(prog)s --mode train --tag base --resume-best --batch 256 --lr 1e-4

  # compute-matched ablations; every arm gets the same wall-clock
  %(prog)s --mode sweep --sweep gap  --minutes 3 --batch 128
  %(prog)s --mode sweep --sweep coldstart --arms never,c32 --minutes 2

  # temporal attention: every step queries the states before it
  %(prog)s --mode train --tag attn --attn-heads 4 --batch 256 --minutes 25

  # is it worth what it costs? 'off' is the 2.x architecture, same seed, same W
  %(prog)s --mode sweep --sweep attn --minutes 3 --batch 128
  %(prog)s --mode sweep --sweep attn --arms off,mqa4 --minutes 5

  # a longer reach back, classic multi-head, an entry per thinking step
  %(prog)s --mode train --tag attn_wide --attn-heads 8 --attn-kv-heads 8 \\
      --attn-window 1024 --attn-write step --batch 64

  # score or sample a saved checkpoint (architecture is read from the file)
  %(prog)s --mode eval --tag base
  %(prog)s --mode gen  --tag base --prompt "Once upon a time" --length 200

  # another corpus entirely
  %(prog)s --mode train --tag shake --text ../../data/shakespeare.txt

  # skip the BPE training pass and use a ready-made tiktoken vocabulary --
  # note the vocabulary table then dwarfs the core, so shrink the batch
  %(prog)s --mode smoke --tokenizer cl100k_base
  %(prog)s --mode train --tag tk --tokenizer cl100k_base --batch 16 --chunk 32

  # a bigger in-domain vocabulary instead (still cheaper than any tiktoken one)
  %(prog)s --mode train --tag v4k --vocab-size 4096

  # one id per byte: every character its own token, negligible embedding table,
  # ~3.6x more timesteps for the same text
  %(prog)s --mode train --tag bytes --tokenizer byte
"""


#: Appended to every "unknown name" rejection. These lists are transcribed
#: from OdyssNet's `_build_activation` and `_apply_init`, which are if/elif
#: chains over string literals and cannot be introspected — so a strategy added
#: to the library is rejected here until someone updates the copy. Validating
#: locally is still the right trade (the model's own error would arrive only
#: after the corpus is tokenized and CUDA is initialized), but the failure
#: should point at its cause rather than look like the name is invalid.
_STALE = (". If the library gained this name recently, the accepted list in "
          "parse_args() needs updating.")


def parse_args():
    d = Cfg()
    p = argparse.ArgumentParser(
        prog="experiment_llm.py",
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("mode")
    g.add_argument("--mode", default="train",
                   choices=["train", "sweep", "smoke", "eval", "gen"],
                   help="train: (resumable) training loop. sweep: compute-matched "
                        "ablation. smoke: fast self-test. eval/gen: score or sample "
                        "a saved checkpoint. (default: %(default)s)")
    g.add_argument("--sweep", default="gap", choices=sorted(SWEEPS),
                   help="which ablation preset to run in --mode sweep "
                        "(default: %(default)s)")
    g.add_argument("--arms", default=None, metavar="A,B",
                   help="comma-separated subset of the preset's arms, e.g. never,c32")

    g = p.add_argument_group("corpus")
    g.add_argument("--text", default=d.text_path, metavar="PATH",
                   help="UTF-8 corpus; tokenized once into a memmap cache "
                        "(default: data/tinystories.txt)")
    g.add_argument("--tokenizer", default=d.tokenizer,
                   choices=("bpe", "byte") + TIKTOKEN_ENCODINGS,
                   help="'bpe' trains a byte-level BPE on the corpus itself "
                        "(once, then pinned) at --vocab-size; 'byte' is one id "
                        "per byte, which keeps every character its own token "
                        "and costs ~3.6x more timesteps for the same text; the "
                        "rest are ready-made tiktoken encodings. The default is "
                        "small and in-domain on purpose -- a 4096-token corpus "
                        "BPE matches gpt2's 50k vocabulary on TinyStories at "
                        "1/12 the embedding cost; see 'Choosing a tokenizer' in "
                        "--help's description. (default: %(default)s)")
    g.add_argument("--vocab-size", type=int, default=None, metavar="N",
                   help="BPE vocabulary size; trained once and pinned per size. "
                        "Ignored for tiktoken tokenizers, which bring their own. "
                        "Perplexity is only comparable within one vocab -- use "
                        f"bits/byte across vocabs. (default: {d.vocab_size})")

    g = p.add_argument_group("architecture")
    g.add_argument("--neurons", type=int, default=d.neurons, metavar="N",
                   help="size of the NxN chaos core (default: %(default)s)")
    g.add_argument("--n-in", type=int, default=d.n_in, metavar="N",
                   help="neurons receiving the token embedding (default: %(default)s)")
    g.add_argument("--n-out", type=int, default=d.n_out, metavar="N",
                   help="neurons the vocab decoder reads; n_in + n_out must be "
                        "<= --neurons (default: %(default)s)")
    g.add_argument("--think-gap", type=int, default=d.think_gap, metavar="N",
                   help="extra echo steps per token; total steps per token is N+1. "
                        "Measured best at 1 for equal wall-clock. (default: %(default)s)")
    g.add_argument("--injection", default="pulse" if d.pulse_mode else "continuous",
                   choices=("pulse", "continuous"),
                   help="sets the model's pulse_mode. 'pulse' injects a token's "
                        "embedding only on the step it arrives, leaving the "
                        "--think-gap echo steps to run on recurrent state "
                        "alone; 'continuous' holds it across every step. "
                        "Identical at --think-gap 0 and increasingly different "
                        "above it, so it is a knob about what the thinking "
                        "steps actually think on. (default: %(default)s)")
    g.add_argument("--activation", default=",".join(d.activation),
                   metavar="ENC,CORE,MEM",
                   help="three activations: encoder/decoder, the core step, and "
                        "memory feedback. 'none' means identity here (unlike "
                        "--gates, where it removes the gate). "
                        "(default: %(default)s)")
    g.add_argument("--weight-init", default=",".join(d.weight_init),
                   metavar="ENC,CORE,MEM,GATE",
                   help="four initialization strategies: encoder/decoder, the "
                        "chaos core, memory feedback, and gates. The core's "
                        "choice sets the initial spectral behaviour the whole "
                        "architecture is built around. (default: %(default)s)")
    g.add_argument("--gates", default=",".join(d.gates), metavar="IN,CORE,MEM",
                   help="three gate activations, applied to input/output "
                        "scaling, the core signal, and memory feedback. 'none' "
                        "disables a gate outright and creates no parameter for "
                        "it; any other value is an activation name, in practice "
                        "'sigmoid' or 'identity'. Measured best on TinyStories: "
                        "none,sigmoid,identity, which beat the default on both "
                        "loss and tokens consumed (--sweep arch). "
                        "(default: %(default)s)")
    g.add_argument("--hebb", default="none",
                   choices=["none", "temporal", "spatial", "both"],
                   help="Hebbian plasticity mechanism. Note this is the one thing "
                        "that couples batch rows. (default: %(default)s)")
    g.add_argument("--hebb-res", default=d.hebb_res,
                   choices=["global", "neuron"],
                   help="plasticity resolution: one factor for the whole core, "
                        "or one per neuron (default: %(default)s)")
    g.add_argument("--tie-embeddings", action="store_true",
                   help="share embed/decoder weights; requires n_in == n_out")
    g.add_argument("--dropout", type=float, default=d.dropout, metavar="P",
                   help="dropout applied every thinking step (default: %(default)s)")

    g = p.add_argument_group(
        "attention",
        "Temporal attention: at every thinking step the state queries a cache "
        "of the states before it. Off unless --attn-heads is given, and its "
        "output projection starts at zero, so switching it on changes nothing "
        "until training decides otherwise.")
    g.add_argument("--attn-heads", type=int, default=d.attn_heads, metavar="N",
                   help="query heads; 0 disables attention entirely and costs "
                        "nothing (default: %(default)s)")
    g.add_argument("--attn-kv-heads", type=int, default=d.attn_kv_heads,
                   metavar="N",
                   help="key/value heads, shared across query heads (must "
                        "divide --attn-heads). 1 is multi-query and is the "
                        "default because the KV cache, not the projections, is "
                        "what decides whether a batch fits; pass the same "
                        "value as --attn-heads for classic multi-head. "
                        "(default: %(default)s)")
    g.add_argument("--attn-head-dim", type=int, default=d.attn_head_dim,
                   metavar="N",
                   help="width per head; 0 derives it from --neurons and "
                        "--attn-heads, capped at 64 and rounded even. Pin it "
                        "explicitly if you intend to transplant between cores "
                        "of different sizes. (default: %(default)s)")
    g.add_argument("--attn-window", type=int, default=d.attn_window, metavar="N",
                   help="cache entries a query can see; the oldest are evicted "
                        "first. Memory is linear in this and the batch. "
                        "(default: %(default)s)")
    g.add_argument("--attn-write", default=d.attn_write, choices=("token", "step"),
                   help="'token' writes one cache entry per input token, so "
                        "--think-gap buys extra reads rather than a longer "
                        "cache; 'step' writes every thinking step, letting the "
                        "model attend to its own intermediate steps at "
                        "(gap+1)x the cache. Identical at --think-gap 0. "
                        "(default: %(default)s)")
    g.add_argument("--attn-read", default=d.attn_read, choices=("token", "step"),
                   help="'step' queries the cache on every thinking step; "
                        "'token' only on the step a token arrives, leaving the "
                        "echo steps to run on state alone the way pulse "
                        "injection does. The query, not the cache entry, is "
                        "what --think-gap multiplies, so this is the cheaper "
                        "of the two whenever the gap is above 0 — measured "
                        "1.3x the throughput at --think-gap 1. Identical at "
                        "--think-gap 0. (default: %(default)s)")
    g.add_argument("--attn-rope", action=argparse.BooleanOptionalAction,
                   default=d.attn_rope,
                   help="rotary position embedding, applied to each key when "
                        "it is written. Without it attention is a pure "
                        "content lookup with no sense of when. "
                        "(default: %(default)s)")
    g.add_argument("--attn-qk-norm", action=argparse.BooleanOptionalAction,
                   default=d.attn_qk_norm,
                   help="RMSNorm on queries and keys before the dot product. "
                        "The core is chaotic and its state norm is only "
                        "bounded at the end of a step, so this is the first "
                        "thing to keep when logits misbehave. "
                        "(default: %(default)s)")
    g.add_argument("--attn-dropout", type=float, default=d.attn_dropout,
                   metavar="P",
                   help="dropout on the attention weights during training "
                        "(default: %(default)s)")

    g = p.add_argument_group("optimization")
    g.add_argument("--batch", type=int, default=d.batch, metavar="N",
                   help="parallel corpus streams; -1 measures throughput and picks "
                        "the fastest. Must match the checkpoint when resuming. "
                        "(default: %(default)s)")
    g.add_argument("--chunk", type=int, default=d.chunk, metavar="N",
                   help="tokens per optimizer step; this is the truncated-BPTT "
                        "window the gradient sees (default: %(default)s)")
    g.add_argument("--cold-start-every", type=int, default=d.cold_start_every,
                   metavar="N",
                   help="each row's state is zeroed with probability 1/N per step, "
                        "so mean context is N*chunk tokens; 0 never resets, which "
                        "measurably invites cold-start collapse (default: %(default)s)")
    g.add_argument("--lr", default="keep", metavar="RATE",
                   help="'auto' for ChaosGrad's online estimate, a float for "
                        "fixed-rate mode. Default 'keep': a fresh run uses auto; "
                        "--resume keeps the mode stored in the checkpoint. Pass "
                        "'auto' explicitly to move a fixed-rate checkpoint back "
                        "to the estimator on resume.")
    g.add_argument("--label-smoothing", type=float, default=d.label_smoothing,
                   metavar="P",
                   help="applied to the training loss only; reported perplexity is "
                        "always unsmoothed (default: %(default)s)")
    g.add_argument("--grad-persistence", type=float, default=d.grad_persistence,
                   metavar="P",
                   help="'ghost gradients': keep this fraction of the previous "
                        "step's gradient and add it to the next, injected after "
                        "AMP unscale and before clipping, then re-captured from "
                        "the clipped result -- so the carry is a bounded "
                        "geometric series rather than a runaway. This is "
                        "momentum on top of ChaosGrad's own, so treat it as a "
                        "second-order knob; 0 disables it. Note --resume does "
                        "NOT let you switch it off: while this is left at 0 the "
                        "trainer restores the checkpoint's stored fraction AND "
                        "its accumulated ghost buffer, so a run you believe has "
                        "ghosting disabled resumes with both the setting and a "
                        "warm carry. Pass a positive value to override it, or "
                        "start a new --tag to be rid of it. "
                        "(default: %(default)s)")
    g.add_argument("--grad-ckpt", action="store_true",
                   help="gradient checkpointing: less memory, one extra sequential "
                        "forward per step. Dynamo does not trace a checkpointed "
                        "region, so it does not combine with --compile")
    g.add_argument("--compile", action="store_true",
                   help="torch.compile the forward pass. The echo loop issues "
                        "thousands of small kernels per step and is bound by "
                        "launching them, so fusing it is the biggest speedup on "
                        "offer: 2.2x on a bare core, 4.0x with plasticity, 3.6x "
                        "with plasticity and attention (10 neurons, batch 32, 16 "
                        "steps). The price is a warmup of a minute or two before "
                        "the first step, longer with --hebb and attention on")

    g = p.add_argument_group("run control")
    g.add_argument("--minutes", type=float, default=d.minutes, metavar="M",
                   help="wall-clock budget; 0 runs until Ctrl-C, which checkpoints "
                        "cleanly on the way out (default: %(default)s)")
    g.add_argument("--max-steps", type=int, default=d.max_steps, metavar="N",
                   help="optimizer-step budget; 0 is unlimited (default: %(default)s)")
    g.add_argument("--tag", default=d.tag, metavar="NAME",
                   help="names the checkpoint pair llm_odyss_<NAME>_{latest,best}.pth "
                        "(default: %(default)s)")
    g.add_argument("--resume", action="store_true",
                   help="continue --tag's latest checkpoint, restoring weights, "
                        "optimizer and per-row stream positions")
    g.add_argument("--resume-best", action="store_true",
                   help="resume from --tag's *best* checkpoint instead of its "
                        "latest; implies --resume. This is the recovery path "
                        "after a loss explosion: `_latest` is rewritten at "
                        "every evaluation, so a diverged run overwrites it "
                        "within one eval interval, while `_best` is only ever "
                        "written on an improvement and therefore cannot hold a "
                        "diverged model. It rewinds the step counter, stream "
                        "positions and optimizer state to that checkpoint, and "
                        "the next evaluation repairs `_latest`. Refuses to run "
                        "if the tag has a latest but no best, rather than "
                        "starting fresh over it")
    g.add_argument("--overwrite", action="store_true",
                   help="allow a fresh run to destroy an existing tag's checkpoints")
    g.add_argument("--seed", type=int, default=d.seed, metavar="N",
                   help="seeds init, shuffling and cold-start draws (default: %(default)s)")
    g.add_argument("--device", default=d.device, metavar="DEV",
                   help="cuda or cpu (default: %(default)s)")

    g = p.add_argument_group("evaluation and logging")
    g.add_argument("--eval-every", type=int, default=d.eval_every, metavar="N",
                   help="steps between held-out validations; 0 disables mid-run "
                        "scoring (default: %(default)s)")
    g.add_argument("--log-every", type=int, default=d.log_every, metavar="N",
                   help="steps between progress lines; 0 is silent (default: %(default)s)")
    g.add_argument("--sample-every", type=int, default=2, metavar="N",
                   help="print a generated sample every N validations; 0 never "
                        "(default: %(default)s)")
    g.add_argument("--val-tokens", type=int, default=d.val_tokens, metavar="N",
                   help="held-out tokens scored per validation (default: %(default)s)")
    g.add_argument("--val-batch", type=int, default=d.val_batch, metavar="N",
                   help="shards the validation split is cut into, which is also "
                        "the number of independent cold starts each score "
                        "averages. It changes which corpus positions are scored, "
                        "and only some positions fall into the absorbing state, "
                        "so val_coldfail moves with it -- raise it to probe "
                        "cold-start robustness, keep it fixed to compare runs. "
                        "(default: %(default)s)")
    g.add_argument("--val-amp", action="store_true",
                   help="evaluate under AMP; faster, and measured to agree with the "
                        "fp32 default to ~4 decimals")

    g = p.add_argument_group("generation")
    g.add_argument("--prompt", default="Once upon a time", metavar="TEXT",
                   help="seed text for --mode gen (default: %(default)s)")
    g.add_argument("--length", type=int, default=200, metavar="N",
                   help="tokens to generate in --mode gen (default: %(default)s)")

    a = p.parse_args()

    # --resume-best is a variant of --resume, not an alternative to it: every
    # downstream check asks `a.resume`, so folding it in here keeps the two
    # from having to be tested separately at each site.
    if a.resume_best:
        a.resume = True

    # `--vocab-size` defaults to None so an explicit request is distinguishable
    # from silence, which decides whether a tiktoken run should say the flag
    # does nothing.
    if a.vocab_size is None:
        a.vocab_size = d.vocab_size
    elif a.tokenizer != "bpe":
        print(f"ℹ️  --vocab-size {a.vocab_size} ignored: {a.tokenizer} is a "
              f"pretrained encoding and carries its own vocabulary.")

    # Validate here rather than at model-construction time so a typo costs a
    # second and an error message, not a corpus tokenization and a CUDA init.
    for name in ("neurons", "n_in", "n_out", "chunk", "vocab_size",
                 "val_tokens", "val_batch"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive, "
                    f"got {getattr(a, name)}")
    if a.n_in + a.n_out > a.neurons:
        p.error(f"--n-in ({a.n_in}) + --n-out ({a.n_out}) = {a.n_in + a.n_out} "
                f"exceeds --neurons ({a.neurons})")
    if str(a.lr) not in ("auto", "keep"):
        try:
            if float(a.lr) <= 0:
                raise ValueError
        except ValueError:
            p.error(f"--lr must be 'auto', 'keep' or a positive float, got {a.lr!r}")
    # Validated here, with the other flags, rather than at construction: an
    # unknown activation would otherwise surface only after the corpus is
    # tokenized and CUDA is initialized. The accepted set is OdyssNet's
    # `_build_activation`, plus 'none', which means "no gate at all".
    a.gates = tuple(s.strip() for s in a.gates.split(","))
    if len(a.gates) != 3:
        p.error(f"--gates needs exactly three comma-separated entries "
                f"(input,core,memory), got {len(a.gates)}: {','.join(a.gates)}")
    known = ("none", "identity", "tanh", "relu", "leaky_relu", "sigmoid",
             "gelu", "gelu_tanh", "silu")
    unknown = [g for g in a.gates if g.lower() not in known]
    if unknown:
        p.error(f"--gates: unknown activation(s) {unknown}; "
                f"choose from {list(known)}{_STALE}")

    a.activation = tuple(s.strip() for s in a.activation.split(","))
    if len(a.activation) != 3:
        p.error(f"--activation needs exactly three comma-separated entries "
                f"(encoder,core,memory), got {len(a.activation)}: "
                f"{','.join(a.activation)}")
    unknown = [s for s in a.activation if s.lower() not in known]
    if unknown:
        p.error(f"--activation: unknown activation(s) {unknown}; "
                f"choose from {list(known)}{_STALE}")

    a.weight_init = tuple(s.strip() for s in a.weight_init.split(","))
    if len(a.weight_init) != 4:
        p.error(f"--weight-init needs exactly four comma-separated entries "
                f"(encoder,core,memory,gate), got {len(a.weight_init)}: "
                f"{','.join(a.weight_init)}")
    inits = ("quiet", "micro_quiet", "micro_quiet_warm", "classic",
             "xavier_uniform", "xavier_normal", "kaiming_uniform",
             "kaiming_normal", "orthogonal", "sparse", "zero", "one",
             "resonant")
    unknown = [s for s in a.weight_init if s.lower() not in inits]
    if unknown:
        p.error(f"--weight-init: unknown strategy/strategies {unknown}; "
                f"choose from {list(inits)}{_STALE}")
    if a.think_gap < 0:
        p.error(f"--think-gap must be >= 0, got {a.think_gap}")
    # Upper bound is the trainer's documented range. Values at or above 1.0
    # make the carried gradient non-decaying, which is a divergence rather
    # than a setting.
    if not 0.0 <= a.grad_persistence <= 0.9:
        p.error(f"--grad-persistence must be between 0.0 and 0.9, "
                f"got {a.grad_persistence}")
    if a.cold_start_every < 0:
        p.error(f"--cold-start-every must be >= 0, got {a.cold_start_every}")
    if a.tie_embeddings and a.n_in != a.n_out:
        p.error(f"--tie-embeddings requires --n-in == --n-out, "
                f"got {a.n_in} and {a.n_out}")

    # Attention geometry, checked here for the same reason as everything else
    # in this function: the library raises the same errors, but only after the
    # corpus is tokenized and CUDA is up.
    if a.attn_heads < 0:
        p.error(f"--attn-heads must be >= 0 (0 disables), got {a.attn_heads}")
    if a.attn_heads:
        if a.attn_kv_heads < 1:
            p.error(f"--attn-kv-heads must be >= 1, got {a.attn_kv_heads}")
        if a.attn_heads % a.attn_kv_heads:
            p.error(f"--attn-heads ({a.attn_heads}) must be divisible by "
                    f"--attn-kv-heads ({a.attn_kv_heads})")
        if a.attn_head_dim < 0:
            p.error(f"--attn-head-dim must be >= 0 (0 = automatic), "
                    f"got {a.attn_head_dim}")
        if a.attn_head_dim and a.attn_rope and a.attn_head_dim % 2:
            p.error(f"--attn-head-dim must be even when RoPE is on "
                    f"(it rotates coordinate pairs), got {a.attn_head_dim}. "
                    f"Use --no-attn-rope or an even width.")
        if a.attn_window < 1:
            p.error(f"--attn-window must be >= 1, got {a.attn_window}")
        if not 0.0 <= a.attn_dropout < 1.0:
            p.error(f"--attn-dropout must be in [0.0, 1.0), got {a.attn_dropout}")
    elif any((a.attn_kv_heads != d.attn_kv_heads, a.attn_head_dim != d.attn_head_dim,
              a.attn_window != d.attn_window, a.attn_write != d.attn_write,
              a.attn_read != d.attn_read, a.attn_rope != d.attn_rope,
              a.attn_qk_norm != d.attn_qk_norm,
              a.attn_dropout != d.attn_dropout)):
        print("ℹ️  attention flags ignored: --attn-heads is 0, so no attention "
              "is built. Pass --attn-heads 4 to switch it on.")
    return a


def cfg_from_args(a):
    return Cfg(
        text_path=a.text, tokenizer=a.tokenizer,
        vocab_size=a.vocab_size, neurons=a.neurons,
        n_in=a.n_in, n_out=a.n_out, think_gap=a.think_gap, chunk=a.chunk,
        cold_start_every=a.cold_start_every, val_amp=a.val_amp, gates=a.gates,
        activation=a.activation, weight_init=a.weight_init,
        pulse_mode=(a.injection == "pulse"), val_batch=a.val_batch,
        batch=a.batch,
        lr=None if str(a.lr) in ("auto", "keep") else float(a.lr),
        lr_force_auto=(str(a.lr) == "auto"),
        dropout=a.dropout, label_smoothing=a.label_smoothing,
        grad_persistence=a.grad_persistence,
        hebb_type=None if a.hebb == "none" else a.hebb,
        hebb_res=a.hebb_res, grad_ckpt=a.grad_ckpt, compile=a.compile,
        attn_heads=a.attn_heads, attn_kv_heads=a.attn_kv_heads,
        attn_head_dim=a.attn_head_dim, attn_window=a.attn_window,
        attn_write=a.attn_write, attn_read=a.attn_read,
        attn_rope=a.attn_rope,
        attn_qk_norm=a.attn_qk_norm, attn_dropout=a.attn_dropout,
        tie_embeddings=a.tie_embeddings, minutes=a.minutes, max_steps=a.max_steps,
        eval_every=a.eval_every, log_every=a.log_every,
        val_tokens=a.val_tokens, seed=a.seed, device=a.device, tag=a.tag,
    )


def main():
    a = parse_args()
    cfg = cfg_from_args(a)
    set_seed(cfg.seed)

    if cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("🚀 OdyssNet-LLM — temporal depth on TinyStories")
    print(f"   mode {a.mode} | device {cfg.device} | seed {cfg.seed}")

    # A smoke test wants the cheapest vocabulary that still exercises the
    # pipeline, but only the trained BPE has one to choose. A tiktoken smoke
    # run is testing that path specifically and keeps its encoding's real size;
    # the smoke config's small n_in/n_out is what keeps the table affordable.
    if a.mode == "smoke" and cfg.tokenizer == "bpe":
        cfg = replace(cfg, vocab_size=1024)

    # Rebuild from the architecture a checkpoint was trained with, not from
    # whatever flags are on this command line, and do it before the corpus is
    # loaded: `tokenizer` and `vocab_size` decide which token cache is read and
    # which tokenizer decodes it. An architecture flag omitted on --resume
    # otherwise reverts to a CLI default, fails the strict load, and leaves a
    # run that --resume has already excused from guard_overwrite free to
    # overwrite the checkpoint.
    #
    # `path` is what eval/gen load below; the resume branch only adopts a
    # config, since run_session opens the checkpoint itself.
    path = None
    if a.mode in ("eval", "gen"):
        latest, best = ckpt_paths(cfg)
        path = best if os.path.exists(best) else latest
        if not os.path.exists(path):
            raise SystemExit(f"No checkpoint for tag '{cfg.tag}' (looked for {path})")
        cfg = adopt_saved_arch(cfg, path, fields=ARCH_FIELDS)
    elif a.mode == "train" and a.resume:
        latest, best = ckpt_paths(cfg)
        # Read the architecture from the file the run will actually load, or
        # --resume-best would rebuild from the wrong one and fail the strict
        # load it is trying to rescue.
        src = best if a.resume_best else latest
        if os.path.exists(src):
            cfg = adopt_saved_arch(cfg, src, fields=RESUME_FIELDS)

    # A tiktoken vocabulary belongs to its encoding; pin it before the cache
    # path and element width are derived from it.
    cfg = resolve_vocab(cfg)

    corpus = load_corpus(cfg)
    vocab = corpus[2].get_vocab_size()

    if a.mode == "smoke":
        sys.exit(0 if run_smoke(cfg, corpus) else 1)

    if a.mode == "sweep":
        run_sweep(cfg, corpus, a.sweep, a.minutes or 3.0, arms=a.arms)
        return

    if a.mode in ("eval", "gen"):
        set_seed(cfg.seed)
        model, trainer = build(cfg, vocab)
        load_checkpoint(model, trainer.optimizer, path, device=cfg.device, strict=True)
        print(f"📂 {path}")
        if a.mode == "eval":
            m = Validator(cfg, corpus[1], corpus[2]).run(model)
            print(f"\n📊 val loss {m['val_loss']:.4f} | ppl {m['val_ppl']:.2f} | "
                  f"bits/byte {m['val_bpb']:.4f}")
            print(f"   median cold start {m['val_p50']:.4f} "
                  f"(ppl {math.exp(min(m['val_p50'],30)):.2f}) | "
                  f"collapsed cold starts {m['val_coldfail']:.1%}")
        else:
            print("\n" + generate(model, corpus[2], cfg,
                                  prompt=a.prompt, length=a.length))
        return

    # --- train ---
    if not a.resume:
        guard_overwrite(cfg, overwrite=a.overwrite)

    metrics, model, trainer, tok = run_session(
        cfg, corpus, budget_sec=cfg.minutes * 60.0, resume=a.resume,
        resume_best=a.resume_best, sample_every=a.sample_every)

    print(f"\n{'='*78}")
    print(f"📊 FINAL  val loss {metrics['val_loss']:.4f} | ppl {metrics['val_ppl']:.2f} | "
          f"bits/byte {metrics['val_bpb']:.4f}")
    print(f"   median cold start {metrics['val_p50']:.4f} "
          f"(ppl {math.exp(min(metrics['val_p50'],30)):.2f}) | "
          f"collapsed cold starts {metrics['val_coldfail']:.1%}")
    print(f"   {metrics['tokens']/1e6:.2f}M tokens in {metrics['minutes']:.1f} min "
          f"({metrics['tok_s']:,.0f} tok/s), {metrics['params']:,} params")
    print(f"{'='*78}")
    print("\n" + generate(model, tok, cfg, prompt=a.prompt, length=a.length))


if __name__ == "__main__":
    main()
