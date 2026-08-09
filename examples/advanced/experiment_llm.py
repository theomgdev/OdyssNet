"""
OdyssNet-LLM — temporal-depth language modeling on TinyStories.

A single NxN chaos core is used as a language model: tokens are injected one
at a time, the signal echoes for ``think_gap`` extra steps, and the readout
neurons are decoded to vocabulary logits. Depth is temporal — the same W is
reused every step — so the interesting question is not "how many layers" but
"how many steps per token, and is that worth the wall-clock".

This script is built to answer that question rather than to assert it.

    # end-to-end self-test, exercises every code path (~40s CUDA, ~15s CPU)
    python -u experiment_llm.py --mode smoke
    python -u experiment_llm.py --mode smoke --device cpu

    # compute-matched ablation: every arm gets the same wall-clock budget
    python -u experiment_llm.py --mode sweep --sweep gap  --minutes 3
    python -u experiment_llm.py --mode sweep --sweep arch --minutes 3

    # the long run (resumable; Ctrl-C is safe, it checkpoints on exit)
    python -u experiment_llm.py --mode train
    python -u experiment_llm.py --mode train --minutes 60

    # inspect a trained checkpoint
    python -u experiment_llm.py --mode eval --resume
    python -u experiment_llm.py --mode gen  --resume --prompt "Once upon a time"

Design notes
------------
* **Offline + instant restart.** The corpus is tokenized once into a uint16
  memmap next to ``data/``; later runs mmap it in milliseconds. No network,
  no HuggingFace streaming, no per-run tokenizer cost.
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

Reference run — ``--mode train --minutes 25 --batch 256`` at the defaults
above, 2,625,280 parameters: held-out **loss 2.2009, ppl 9.03,
bits/byte 0.8764**, median cold start 2.3154, **0.0% collapsed cold starts**,
226.15M tokens at 150,761 tok/s. Reproduce with that command; `--mode eval
--tag base` re-scores the saved checkpoint.
"""

import sys

# Keep emoji-rich console output from crashing legacy Windows code pages.
# line_buffering=True is not optional: reconfigure() rebuilds the TextIOWrapper
# and would otherwise discard `python -u`'s write-through, leaving a long
# training run's progress invisible until the process exits.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict, replace

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
    vocab_size: int = 2048
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

    # --- temporal depth ---
    think_gap: int = 1               # extra echo steps per token
    chunk: int = 48                  # tokens per optimizer step (TBPTT window)
    cold_start_every: int = 32       # mean chunks before a row's state is
                                     # zeroed again (0 = never reset)

    # --- optimization ---
    batch: int = -1                  # -1 = empirical autotune
    lr: float | None = None          # None = zero-config ChaosGrad
    label_smoothing: float = 0.0
    grad_ckpt: bool = False

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
# Corpus: tokenizer + uint16 memmap cache                                     #
# --------------------------------------------------------------------------- #

def get_tokenizer(cfg, text_for_training=None):
    """Load the pinned BPE tokenizer for this vocab size, training it once."""
    from tokenizers import Tokenizer
    from tokenizers.implementations import ByteLevelBPETokenizer

    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"poc_tokenizer_{cfg.vocab_size // 1000}k.json")
    if cfg.vocab_size % 1000:
        path = os.path.join(CKPT_DIR, f"poc_tokenizer_{cfg.vocab_size}.json")

    if os.path.exists(path):
        return Tokenizer.from_file(path)

    if text_for_training is None:
        raise FileNotFoundError(f"Tokenizer {path} missing and no text to train it on.")

    print(f"📚 Training {cfg.vocab_size}-token BPE tokenizer...")
    tok = ByteLevelBPETokenizer()
    blocks = _blocks(text_for_training[:20_000_000])
    tok.train_from_iterator(iter(blocks), vocab_size=cfg.vocab_size, min_frequency=2,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tok.save(path)
    print(f"✅ Tokenizer saved: {path}")
    return Tokenizer.from_file(path)


def _blocks(text, target=200_000):
    """Split text into ~target-sized blocks on paragraph boundaries."""
    out, buf, size = [], [], 0
    for para in text.split("\n\n"):
        buf.append(para)
        size += len(para) + 2
        if size >= target:
            out.append("\n\n".join(buf))
            buf, size = [], 0
    if buf:
        out.append("\n\n".join(buf))
    return out


def _encode_corpus(tok, text):
    """Tokenize a large string block-wise into a flat uint16 array."""
    blocks = _blocks(text)
    ids = []
    for i in range(0, len(blocks), 64):
        for enc in tok.encode_batch(blocks[i:i + 64]):
            ids.extend(enc.ids)
    return np.asarray(ids, dtype=np.uint16)


def load_corpus(cfg, verbose=True):
    """
    Returns (train_tokens, val_tokens, tokenizer).

    The raw text is split by *character* offset first — the tail becomes
    validation — and each half is tokenized independently, so no validation
    token can leak into training via a shared BPE merge boundary.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    stamp = f"{os.path.basename(cfg.text_path)}.v{cfg.vocab_size}.val{cfg.val_chars}"
    train_npy = os.path.join(CACHE_DIR, f"{stamp}.train.npy")
    val_npy = os.path.join(CACHE_DIR, f"{stamp}.val.npy")

    if os.path.exists(train_npy) and os.path.exists(val_npy):
        tok = get_tokenizer(cfg)
        train = np.load(train_npy, mmap_mode="r")
        val = np.load(val_npy, mmap_mode="r")
        if verbose:
            print(f"💾 Token cache: {len(train):,} train / {len(val):,} val tokens "
                  f"(vocab {cfg.vocab_size})")
        return train, val, tok

    if not os.path.exists(cfg.text_path):
        raise FileNotFoundError(
            f"Corpus not found: {cfg.text_path}\n"
            f"   Point --text at a UTF-8 text file, or fetch TinyStories with:\n"
            f"   python -c \"from datasets import load_dataset as L; "
            f"open(r'{cfg.text_path}','w',encoding='utf-8').write("
            f"'\\n\\n'.join(r['text'] for r in L('roneneldan/TinyStories',split='train[:400000]')))\""
        )

    print(f"📖 Reading corpus: {cfg.text_path}")
    with open(cfg.text_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    split = max(0, len(text) - cfg.val_chars)
    train_text, val_text = text[:split], text[split:]
    if cfg.train_chars > 0:
        train_text = train_text[:cfg.train_chars]

    tok = get_tokenizer(cfg, text_for_training=train_text)

    print(f"🔤 Tokenizing {len(train_text)/1e6:.1f}M chars (train) "
          f"+ {len(val_text)/1e6:.1f}M chars (val)...")
    t0 = time.time()
    train = _encode_corpus(tok, train_text)
    val = _encode_corpus(tok, val_text)
    np.save(train_npy, train)
    np.save(val_npy, val)
    print(f"✅ Cached in {time.time()-t0:.1f}s: {len(train):,} train / {len(val):,} val tokens")

    return np.load(train_npy, mmap_mode="r"), np.load(val_npy, mmap_mode="r"), tok


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
        pos = np.asarray(pos, dtype=np.int64)
        if pos.shape == self.pos.shape:
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
        activation=list(cfg.activation),
        weight_init=list(cfg.weight_init),
        gate=list(cfg.gates),
        hebb_type=cfg.hebb_type,
        hebb_res=cfg.hebb_res,
        dropout_rate=cfg.dropout,
        gradient_checkpointing=cfg.grad_ckpt,
        vocab_size=vocab_size,
        vocab_mode="discrete",
        tie_embeddings=cfg.tie_embeddings,
    )
    trainer = OdyssNetTrainer(model, lr=cfg.lr, device=cfg.device)
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
    return snap


def restore_state(model, snap):
    model.state = snap["state"]
    for name, buf in snap.items():
        if name != "state":
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

    ids = tokenizer.encode(prompt).ids or [0]
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

#: Fields that must match the saved weights bit-for-bit to load them at all.
ARCH_FIELDS = ("neurons", "n_in", "n_out", "vocab_size", "activation",
               "weight_init", "gates", "hebb_type", "hebb_res",
               "tie_embeddings", "think_gap")


def adopt_saved_arch(cfg, path):
    """Override architecture fields with the ones stored in a checkpoint."""
    saved = torch.load(path, map_location="cpu").get("cfg")
    if not saved:
        return cfg
    changes = {}
    for k in ARCH_FIELDS:
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


def run_session(cfg, corpus, budget_sec=0.0, resume=False, quiet=False,
                sample_every=0, save=True):
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
    if resume and os.path.exists(latest_path):
        try:
            data = load_checkpoint(model, trainer.optimizer, latest_path,
                                   device=cfg.device, strict=True, lr=cfg.lr)
            extra = data.get("stream", {})
            shards.restore(extra.get("pos", []), extra.get("wraps", 0))
            best_val = data.get("best_val", float("inf"))
            start_step = int(data.get("step", 0))
            ts = data.get("trainer_state_dict")
            if ts:
                trainer.load_state_dict(ts)
            print(f"📂 Resumed at step {start_step:,} "
                  f"(best val PPL {math.exp(min(best_val,30)):.2f})")
        except Exception as e:
            print(f"⚠️  Resume failed ({e}); starting fresh.")

    if not quiet:
        print(f"\n🧠 {model.get_num_params():,} trainable params | "
              f"{cfg.neurons} neurons | in {cfg.n_in} / out {cfg.n_out} | "
              f"gap {cfg.think_gap} | chunk {cfg.chunk} | batch {cfg.batch} | "
              f"lr {'auto' if cfg.lr is None else cfg.lr}"
              + (f" | hebb {cfg.hebb_type}/{cfg.hebb_res}" if cfg.hebb_type else ""))

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
                if bool(hit.any()):
                    model.state = model.state.masked_fill(hit.unsqueeze(1), 0.0)

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
        ("lr_fixed",   dict(lr=1e-4)),
        ("tied_embed", dict(tie_embeddings=True, n_in=256, n_out=256)),
    ],
    # Does training on cold starts remove the degenerate absorbing state?
    # `never` is the old behaviour: state carried for the entire run.
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
    print("\n  → checkpoint round-trip")
    try:
        c = replace(base, tag="smoke_ckpt", eval_every=0)
        _, model1, _, _ = run_session(c, corpus, quiet=True, save=True)
        validator = Validator(c, corpus[1], corpus[2])
        v1 = validator.run(model1)["val_loss"]
        del model1

        model2, trainer2 = build(c, corpus[2].get_vocab_size())
        load_checkpoint(model2, trainer2.optimizer, ckpt_paths(c)[0],
                        device=c.device, strict=True)
        v2 = validator.run(model2)["val_loss"]
        if abs(v1 - v2) < 1e-4:
            print(f"  ✅ round-trip  {v1:.6f} == {v2:.6f}")
        else:
            ok = False
            print(f"  ❌ round-trip  {v1:.6f} != {v2:.6f}")
    except Exception as e:
        ok = False
        print(f"  ❌ round-trip  {type(e).__name__}: {e}")

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

def parse_args():
    d = Cfg()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="train",
                   choices=["train", "sweep", "smoke", "eval", "gen"])
    p.add_argument("--sweep", default="gap", choices=sorted(SWEEPS))
    p.add_argument("--arms", default=None, help="comma-separated arm subset")

    p.add_argument("--text", default=d.text_path)
    p.add_argument("--vocab-size", type=int, default=d.vocab_size)
    p.add_argument("--neurons", type=int, default=d.neurons)
    p.add_argument("--n-in", type=int, default=d.n_in)
    p.add_argument("--n-out", type=int, default=d.n_out)
    p.add_argument("--think-gap", type=int, default=d.think_gap)
    p.add_argument("--chunk", type=int, default=d.chunk)
    p.add_argument("--cold-start-every", type=int, default=d.cold_start_every,
                   help="mean chunks before a row's state resets (0 = never)")
    p.add_argument("--val-amp", action="store_true",
                   help="evaluate under AMP (faster; measured to agree with fp32 to ~4 decimals)")
    p.add_argument("--batch", type=int, default=d.batch, help="-1 autotunes")
    p.add_argument("--lr", default="auto", help="'auto' (ChaosGrad) or a float")
    p.add_argument("--dropout", type=float, default=d.dropout)
    p.add_argument("--label-smoothing", type=float, default=d.label_smoothing)
    p.add_argument("--hebb", default=None, choices=[None, "temporal", "spatial", "both"])
    p.add_argument("--hebb-res", default=d.hebb_res, choices=["global", "neuron", "synapse"])
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--tie-embeddings", action="store_true")

    p.add_argument("--minutes", type=float, default=d.minutes, help="0 = unlimited")
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--log-every", type=int, default=d.log_every)
    p.add_argument("--val-tokens", type=int, default=d.val_tokens)
    p.add_argument("--sample-every", type=int, default=2,
                   help="generate a sample every N validations (0 = never)")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--tag", default=d.tag)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--length", type=int, default=200)
    return p.parse_args()


def cfg_from_args(a):
    return Cfg(
        text_path=a.text, vocab_size=a.vocab_size, neurons=a.neurons,
        n_in=a.n_in, n_out=a.n_out, think_gap=a.think_gap, chunk=a.chunk,
        cold_start_every=a.cold_start_every, val_amp=a.val_amp,
        batch=a.batch, lr=None if str(a.lr) == "auto" else float(a.lr),
        dropout=a.dropout, label_smoothing=a.label_smoothing,
        hebb_type=a.hebb, hebb_res=a.hebb_res, grad_ckpt=a.grad_ckpt,
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

    if a.mode == "smoke":
        cfg = replace(cfg, vocab_size=1024)

    corpus = load_corpus(cfg)
    vocab = corpus[2].get_vocab_size()

    if a.mode == "smoke":
        sys.exit(0 if run_smoke(cfg, corpus) else 1)

    if a.mode == "sweep":
        run_sweep(cfg, corpus, a.sweep, a.minutes or 3.0, arms=a.arms)
        return

    if a.mode in ("eval", "gen"):
        latest, best = ckpt_paths(cfg)
        path = best if os.path.exists(best) else latest
        if not os.path.exists(path):
            raise SystemExit(f"No checkpoint for tag '{cfg.tag}' (looked for {path})")
        # Rebuild from the architecture the checkpoint was *trained* with, not
        # from whatever flags happen to be on this command line — otherwise a
        # forgotten --neurons turns into an opaque state_dict shape error.
        cfg = adopt_saved_arch(cfg, path)
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
    metrics, model, trainer, tok = run_session(
        cfg, corpus, budget_sec=cfg.minutes * 60.0, resume=a.resume,
        sample_every=a.sample_every)

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
