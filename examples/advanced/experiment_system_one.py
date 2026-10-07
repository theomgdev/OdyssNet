"""
OdyssNet-SystemOne — three typed decisions from one pass over the query.

A System One model does not write prose. It reads a state and returns typed,
calibrated decisions: a `choice` over a label set, a `score` on an ordered
scale, a `noul` that is the probability a proposition holds. Those are
if-statements with probabilities attached, and what makes them expensive today
is that the usual way to get one is to run a transformer that was built to
generate text and then throw the text away.

OdyssNet does not have to work that way. The query is injected byte by byte,
the signal echoes through one NxN core, and the readout neurons are decoded
*once* into every decision head at the same time — a 150-way intent choice, a
10-way domain choice, an in-scope noul and a confidence score, all off the
same final state. There is no per-decision pass, because there is no per-
decision network: the heads are slices of one decoder matrix, so a second
question costs its own slice and nothing else.

Deliberation is where the parameters go instead. `--think-gap` keeps the core
running after the last byte has arrived, on state alone, and those steps reuse
the same W — so depth is bought in time rather than in weights.

    python -u experiment_system_one.py --mode smoke                  # ~60s self-test
    python -u experiment_system_one.py --mode train --tag base --minutes 20
    python -u experiment_system_one.py --mode eval  --tag base
    python -u experiment_system_one.py --mode ask   --tag base --query "cancel my flight"
    python -u experiment_system_one.py --mode sweep --sweep scale --minutes 3
    python -u experiment_system_one.py --mode frontier --tag base

`--help` lists every flag with its default. Use `python -u` when piping.

The task
--------
CLINC150: 150 intents over 10 domains, plus out-of-scope queries written to
look exactly like the in-scope ones. It is the right benchmark for this
because it refuses to reward a model that is merely accurate — the dataset
exists to show that classifiers which score 96% in-scope collapse on deciding
whether a query belongs at all, which is precisely the decision a routing
layer is deployed to make. Both halves are scored here, and so is calibration,
because a decision whose probability cannot be trusted is not a decision.

Design notes
------------
* **Bytes, not a tokenizer.** 256 ids, nothing to train, nothing to pin, and
  every character is its own timestep. A subword vocabulary would make the
  embedding table the model at this scale; at `--neurons 128` the whole lookup
  cost is 256x64 in and a decoder whose width is the number of decisions.
* **One readout, N heads.** `vocab_size=[256, DECISION_WIDTH]` makes OdyssNet's
  own `embed` and `output_decoder` the encoder and the decision layer. The
  heads are contiguous slices of that one matrix, so they share the core's
  final state and are read in a single projection — the architectural form of
  "parallel sampling" rather than a loop over questions.
* **The OOS head is trained on OOS data, and scored on held-out OOS.** 100
  training out-of-scope queries against 15,000 in-scope ones is the real
  imbalance, so the noul head carries a positive weight rather than a
  rebalanced corpus; `--oos-weight` is the knob and `--sweep oos` measures it.
* **Calibration is a metric, not a hope.** Expected calibration error and the
  Brier score are reported next to accuracy, because a System One layer is
  consumed by `if p > threshold` and a model that is 90% accurate while
  claiming 99% is worse than useless to that caller.
* **Budget-matched sweeps.** Every arm gets the same wall-clock and is scored
  once after the clock stops, so "learns more per example" is distinguishable
  from "runs faster". `--mode frontier` instead holds the recipe fixed and
  walks the core size, which is the parameter-efficiency curve this file
  exists to produce.
"""

import sys

# Keep emoji-rich console output from crashing legacy Windows code pages.
# line_buffering=True is not optional: reconfigure() rebuilds the TextIOWrapper
# and would otherwise discard `python -u`'s write-through, leaving a long
# training run's progress invisible until the process exits.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import argparse
import glob
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

from odyssnet import (
    OdyssNet,
    OdyssNetTrainer,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "ckpt")
DATA_DIR = os.path.join(HERE, "..", "..", "data")
CLINC_DIR = os.path.join(DATA_DIR, "clinc150")

CLINC_URL = ("https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
             "data_full.json")
DOMAINS_URL = ("https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
               "domains.json")

# Byte id reserved for padding. The corpus is ASCII-range text and byte 0 never
# occurs in it, so a pad is distinguishable from content without widening the
# vocabulary. It matters that this is a real id: OdyssNet's vocab path has no
# -1 "inject nothing" sentinel (that exists only in direct-neuron mode), so a
# negative pad would index the embedding out of range.
PAD_ID = 0


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Cfg:
    # --- corpus ---
    max_bytes: int = 64              # query bytes kept; 93% of CLINC150 fits
                                     # whole and the rest is truncated, not
                                     # dropped
    oos_train: bool = True           # include the 100 out-of-scope training
                                     # queries

    # --- architecture ---
    neurons: int = 128
    n_in: int = 64                   # neurons receiving the byte embedding
    n_out: int = 64                  # neurons the decision decoder reads
    activation: tuple = ("none", "gelu_tanh", "tanh")
    weight_init: tuple = ("quiet", "resonant", "quiet", "zero")
    gates: tuple = ("none", "none", "identity")
    hebb_type: str | None = None
    hebb_res: str = "neuron"
    dropout: float = 0.0

    # --- temporal attention ---
    attn_heads: int = 0              # 0 = no attention at all
    attn_kv_heads: int = 1
    attn_head_dim: int = 0           # 0 = derive from neurons / heads
    attn_window: int = 128
    attn_write: str = "token"
    attn_read: str = "step"
    attn_rope: bool = True
    attn_qk_norm: bool = True
    attn_dropout: float = 0.0

    # --- temporal depth ---
    # Bytes arrive one per step and the answer is read after the last one, so
    # pulse injection is the only sane choice here: holding a byte across its
    # echo steps would smear the character the core is supposed to have moved
    # past.
    think_gap: int = 0               # echo steps per byte
    deliberate: int = 8              # steps after the last byte, on state
                                     # alone — the decision is read from the
                                     # final one

    # --- optimization ---
    batch: int = 128
    # A single supervision signal arrives after every byte has been consumed,
    # which is a longer credit path than the LLM path's signal-per-token. The
    # online estimate climbs to its traction cap on that gradient and the cap
    # is too large here — measured on the val split at 1200 steps, the
    # estimator reached 1.90% intent accuracy where a pinned 1e-3 reached
    # 6.27%. Pass `--lr auto` to put the estimator back.
    lr: float | None = 1e-3
    lr_force_auto: bool = False
    label_smoothing: float = 0.0
    oos_weight: float = 8.0          # positive-class weight on the noul head;
                                     # 150:1 in the raw data, so the loss needs
                                     # to care about the minority decision
    w_intent: float = 1.0            # loss weights per head
    w_domain: float = 0.3
    w_oos: float = 1.0
    w_conf: float = 0.3
    grad_persistence: float = 0.0
    grad_ckpt: bool = False
    compile: bool = False

    # --- runtime ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    minutes: float = 0.0             # 0 = run until interrupted
    max_steps: int = 0               # 0 = unlimited
    epochs: float = 0.0              # 0 = unlimited
    eval_every: int = 200            # optimizer steps between validations
    eval_batch: int = 256
    eval_amp: bool = False           # score in fp32: the metric must be exact
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
# The decision schema                                                         #
# --------------------------------------------------------------------------- #

class Schema:
    """
    The typed questions this model answers, and where each one lives in the
    single decoder output.

    A System One request is a state plus a list of questions, answered in one
    pass. Here the state is the query's bytes and the questions are fixed at
    construction, so the "schema" is a layout over one output vector: each head
    owns a contiguous slice, and `width` is what the decoder projects to. A
    fourth question would be a fourth slice and a wider decoder — not a second
    forward pass, which is the whole point.

    Heads:
        intent  choice over `n_intent` labels
        domain  choice over `n_domain` labels
        oos     noul — p(the query is out of scope)
        conf    score over `CONF_LEVELS` ordered levels, read as a
                probability-weighted mean in [0, 1]

    `conf` is a self-report and is trained against whether the intent head
    actually got that example right, which makes it a prediction about the
    model's own correctness rather than a restatement of its logits. The
    alternative — reading max softmax — is the method CLINC150's own paper
    measured and found to be the weakest OOS signal available, so it is worth
    one extra slice to ask the question directly.
    """

    CONF_LEVELS = 5

    def __init__(self, intents, domains, intent_to_domain):
        self.intents = list(intents)
        self.domains = list(domains)
        self.intent_to_domain = np.asarray(intent_to_domain, dtype=np.int64)
        n_i, n_d = len(self.intents), len(self.domains)
        self.n_intent, self.n_domain = n_i, n_d
        self.sl_intent = slice(0, n_i)
        self.sl_domain = slice(n_i, n_i + n_d)
        self.i_oos = n_i + n_d
        self.sl_conf = slice(n_i + n_d + 1, n_i + n_d + 1 + self.CONF_LEVELS)
        self.width = n_i + n_d + 1 + self.CONF_LEVELS

    def split(self, out):
        """
        (B, width) -> the four raw head outputs.

        A trailing step axis is accepted and the last step taken, so the same
        layout reads both a raw forward output and the trainer's, which slices
        the final step before handing it to `output_transform`.
        """
        if out.dim() == 3:
            out = out[:, -1, :]
        return (out[:, self.sl_intent], out[:, self.sl_domain],
                out[:, self.i_oos], out[:, self.sl_conf])

    def describe(self):
        return (f"{self.n_intent}-way intent choice + {self.n_domain}-way "
                f"domain choice + in-scope noul + {self.CONF_LEVELS}-level "
                f"confidence score = {self.width} outputs, one pass")


def conf_targets(correct, levels=Schema.CONF_LEVELS):
    """
    A right answer asks for the top confidence level, a wrong one for the
    bottom. The levels between exist so the head can express the middle rather
    than being pushed to the extremes; a soft target would train a specific
    distribution shape that nothing here has a reason to prefer.
    """
    return torch.where(correct, torch.full_like(correct, levels - 1,
                                                dtype=torch.long),
                       torch.zeros_like(correct, dtype=torch.long))


def score_from_logits(logits, levels=Schema.CONF_LEVELS):
    """Probability-weighted mean of the ordered levels, mapped to [0, 1]."""
    p = torch.softmax(logits.float(), dim=-1)
    rung = torch.arange(levels, device=p.device, dtype=p.dtype)
    return (p * rung).sum(dim=-1) / (levels - 1)


# --------------------------------------------------------------------------- #
# Corpus                                                                      #
# --------------------------------------------------------------------------- #

def _fetch(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print(f"⬇️  {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
    except (urllib.error.URLError, TimeoutError) as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise SystemExit(
            f"\n✋ Could not download {os.path.basename(path)}: {e}\n"
            f"   The dataset is 2.5 MB and this is the only network access "
            f"this script needs.\n"
            f"   Fetch it by hand and drop it at:\n     {path}\n"
            f"   Source: {url}\n"
        ) from e
    os.replace(tmp, path)
    return path


def encode_queries(texts, max_bytes):
    """
    Queries as right-padded byte ids, plus each one's true length.

    Padding goes on the right and the decision is read after the whole window
    has been consumed, so a short query spends its remaining steps on pad
    bytes. That is deliberate: pad is a real embedding the core learns to
    ignore, and it keeps the batch rectangular without a mask the vocab path
    does not have.
    """
    out = np.full((len(texts), max_bytes), PAD_ID, dtype=np.int64)
    lens = np.empty(len(texts), dtype=np.int64)
    for i, t in enumerate(texts):
        b = t.encode("utf-8")[:max_bytes]
        out[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
        lens[i] = len(b)
    return out, lens


class Split:
    """One evaluation or training split, already encoded."""

    def __init__(self, x, intent, domain, oos):
        self.x = x
        self.intent = intent
        self.domain = domain
        self.oos = oos

    def __len__(self):
        return len(self.x)

    def to(self, device):
        # Idempotent: one corpus is shared by every arm of a sweep or a smoke
        # run, so this is called once per arm on the same object.
        as_t = lambda v: (v if isinstance(v, torch.Tensor)
                          else torch.from_numpy(v)).to(device)
        self.x = as_t(self.x)
        self.intent = as_t(self.intent)
        self.domain = as_t(self.domain)
        self.oos = as_t(self.oos)
        return self


def load_corpus(cfg, verbose=True):
    """
    Returns (train, val, test, schema).

    Out-of-scope rows carry intent and domain label -1 and are excluded from
    those two losses by mask; only the noul head sees them as a positive.
    """
    data_path = _fetch(CLINC_URL, os.path.join(CLINC_DIR, "data_full.json"))
    dom_path = _fetch(DOMAINS_URL, os.path.join(CLINC_DIR, "domains.json"))

    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)
    with open(dom_path, encoding="utf-8") as f:
        domains_map = json.load(f)

    domains = sorted(domains_map)
    intents = sorted({i for v in domains_map.values() for i in v})
    of_intent = {name: k for k, name in enumerate(intents)}
    of_domain = {name: k for k, name in enumerate(domains)}
    intent_to_domain = np.empty(len(intents), dtype=np.int64)
    for dom, members in domains_map.items():
        for name in members:
            intent_to_domain[of_intent[name]] = of_domain[dom]
    schema = Schema(intents, domains, intent_to_domain)

    def build(in_key, oos_key, use_oos=True):
        rows = list(raw[in_key]) + (list(raw[oos_key]) if use_oos else [])
        texts = [t for t, _ in rows]
        x, _ = encode_queries(texts, cfg.max_bytes)
        n = len(rows)
        intent = np.full(n, -1, dtype=np.int64)
        domain = np.full(n, -1, dtype=np.int64)
        oos = np.zeros(n, dtype=np.float32)
        for i, (_, label) in enumerate(rows):
            if label == "oos":
                oos[i] = 1.0
            else:
                intent[i] = of_intent[label]
                domain[i] = intent_to_domain[intent[i]]
        return Split(x, intent, domain, oos)

    train = build("train", "oos_train", use_oos=cfg.oos_train)
    val = build("val", "oos_val")
    test = build("test", "oos_test")

    if verbose:
        for name, s in (("train", train), ("val", val), ("test", test)):
            n_oos = int(s.oos.sum())
            print(f"📚 {name:<5} {len(s):>6,} queries "
                  f"({len(s) - n_oos:,} in-scope / {n_oos:,} out-of-scope)")
        print(f"🗂️  {schema.describe()}")
    return train, val, test, schema


class Batcher:
    """Shuffled epochs over a split, on device."""

    def __init__(self, split, batch, device, seed=0):
        self.split = split.to(device)
        self.batch = min(batch, len(split))
        self.device = device
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.order = self._shuffle()
        self.cursor = 0
        self.epochs = 0.0

    def _shuffle(self):
        return torch.randperm(len(self.split), generator=self.g).to(self.device)

    def next(self):
        if self.cursor + self.batch > len(self.order):
            self.order = self._shuffle()
            self.cursor = 0
        idx = self.order[self.cursor:self.cursor + self.batch]
        self.cursor += self.batch
        self.epochs += self.batch / len(self.split)
        s = self.split
        return s.x[idx], s.intent[idx], s.domain[idx], s.oos[idx]


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #

def build(cfg, schema):
    input_ids, output_ids = cfg.io_ids()
    model = OdyssNet(
        num_neurons=cfg.neurons,
        input_ids=input_ids,
        output_ids=output_ids,
        device=cfg.device,
        pulse_mode=True,
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
        vocab_size=[256, schema.width],
        vocab_mode="discrete",
    )
    if cfg.compile:
        # Compile the bound method rather than the module: everything else here
        # reaches through `model` for state and checkpoints, and an
        # OptimizedModule wrapper would sit between them and the real object.
        model.forward = torch.compile(model.forward)

    trainer = OdyssNetTrainer(model, lr=cfg.lr, device=cfg.device,
                              gradient_persistence=cfg.grad_persistence)
    # The four heads are combined by `head_losses`/`weighted` inside the
    # output transform, which is the only place that sees all of them, so what
    # reaches `loss_fn` is already the scalar to descend. The trainer's default
    # MSELoss would square it — monotone, but it rescales every gradient by 2L
    # and feeds a squared stream to ChaosGrad's spike brake.
    trainer.loss_fn = lambda predicted, _target: predicted.mean()
    return model, trainer


def total_steps(cfg):
    """
    Steps the core runs for one query: one per byte, times the echo factor,
    plus the deliberation tail.

    The byte count has to be a whole multiple of the echo factor, because
    OdyssNet derives its injection `ratio` as `steps // seq_len` — a tail that
    is not a multiple would shift which step each byte lands on.
    """
    gap = cfg.think_gap + 1
    return cfg.max_bytes * gap + cfg.deliberate * gap


def decide(model, x, cfg, schema):
    """
    One pass, four decisions.

    `return_sequence=False` keeps the (B, T, N) activity tensor from being
    allocated: the answer is only ever read off the final step, which is the
    one that has seen every byte plus the deliberation tail.
    """
    out, _ = model(x, steps=total_steps(cfg), return_sequence=False)
    return schema.split(out)


def head_losses(raw, target, cfg, schema):
    """
    The per-head loss terms, and the intent head's hit mask.

    Out-of-scope rows have no intent and no domain, so those two terms are
    masked rather than given a 151st class. That is a deliberate split of the
    two questions CLINC150's paper tangles together: "which intent" and
    "is this any intent at all" are different decisions here, and keeping them
    separate is what lets the noul head be weighted without distorting the
    intent distribution.
    """
    intent_logits, domain_logits, oos_logit, conf_logits = raw
    y_intent, y_domain, y_oos = target
    in_scope = y_oos < 0.5

    terms = {}
    if bool(in_scope.any()):
        terms["intent"] = F.cross_entropy(
            intent_logits[in_scope].float(), y_intent[in_scope],
            label_smoothing=cfg.label_smoothing)
        terms["domain"] = F.cross_entropy(
            domain_logits[in_scope].float(), y_domain[in_scope],
            label_smoothing=cfg.label_smoothing)
    else:
        zero = oos_logit.sum() * 0.0
        terms["intent"] = zero
        terms["domain"] = zero

    terms["oos"] = F.binary_cross_entropy_with_logits(
        oos_logit.float(), y_oos.float(),
        pos_weight=torch.as_tensor(cfg.oos_weight, device=oos_logit.device,
                                   dtype=torch.float32))

    # The confidence head is trained against the intent head's live hit/miss,
    # detached: it predicts correctness, and letting its gradient reach the
    # intent logits would let the model lower its own accuracy to make the
    # self-report easier.
    hit = torch.zeros_like(y_oos, dtype=torch.bool)
    if bool(in_scope.any()):
        pred = intent_logits.detach().argmax(dim=-1)
        hit = in_scope & (pred == y_intent)
    terms["conf"] = F.cross_entropy(conf_logits.float(), conf_targets(hit))
    return terms, hit


def weighted(terms, cfg):
    return (cfg.w_intent * terms["intent"] + cfg.w_domain * terms["domain"]
            + cfg.w_oos * terms["oos"] + cfg.w_conf * terms["conf"])


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #

def _ece(prob, correct, bins=15):
    """
    Expected calibration error: the gap between claimed and actual, averaged
    over equal-width probability bins and weighted by occupancy.

    This is the number that decides whether `if p > 0.8` means anything, which
    is the only way a typed decision is ever consumed.
    """
    prob = prob.detach().float().clamp(0.0, 1.0)
    correct = correct.detach().float()
    edges = torch.linspace(0.0, 1.0, bins + 1, device=prob.device)
    total = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (prob > lo) & (prob <= hi) if b else (prob >= lo) & (prob <= hi)
        n = int(sel.sum())
        if not n:
            continue
        total += (n / len(prob)) * abs(float(prob[sel].mean())
                                      - float(correct[sel].mean()))
    return total


class Evaluator:
    """
    Fixed held-out scoring. Deterministic across runs and configurations, so
    numbers from different arms are directly comparable.
    """

    def __init__(self, cfg, split, schema):
        self.cfg = cfg
        self.schema = schema
        # `Split.to` is idempotent, so a split shared with the training batcher
        # is moved once and this is a no-op on it.
        self.split = split.to(cfg.device)

    @torch.no_grad()
    def run(self, model):
        cfg, schema = self.cfg, self.schema
        was_training = model.training
        model.eval()
        # TF32 off for the same reason AMP is: these numbers rank the arms, so
        # they must not move with kernel choice. Restored on the way out.
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return self._score(model)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            if was_training:
                model.train()

    def _score(self, model):
        cfg, schema = self.cfg, self.schema
        s = self.split
        amp = torch.amp.autocast(
            device_type="cuda" if cfg.device.startswith("cuda") else "cpu",
            enabled=cfg.eval_amp and cfg.device.startswith("cuda"))

        i_pred, d_pred, oos_p, conf, msp = [], [], [], [], []
        for lo in range(0, len(s), cfg.eval_batch):
            x = s.x[lo:lo + cfg.eval_batch]
            model.reset_state(batch_size=x.shape[0])
            with amp:
                intent_logits, domain_logits, oos_logit, conf_logits = decide(
                    model, x, cfg, schema)
            p_intent = torch.softmax(intent_logits.float(), dim=-1)
            i_pred.append(p_intent.argmax(dim=-1))
            msp.append(p_intent.max(dim=-1).values)
            d_pred.append(domain_logits.float().argmax(dim=-1))
            oos_p.append(torch.sigmoid(oos_logit.float()))
            conf.append(score_from_logits(conf_logits))

        i_pred = torch.cat(i_pred)
        d_pred = torch.cat(d_pred)
        oos_p = torch.cat(oos_p)
        conf = torch.cat(conf)
        msp = torch.cat(msp)

        in_scope = s.oos < 0.5
        n_in = int(in_scope.sum())
        i_hit = (i_pred == s.intent) & in_scope
        d_hit = (d_pred == s.domain) & in_scope

        # Out-of-scope recall at the natural 0.5 threshold, and in-scope
        # accuracy over the queries that belong. CLINC150's paper reports both
        # because every method it tested did well on one and badly on the
        # other; a single blended number would hide exactly that.
        oos_true = ~in_scope
        n_oos = int(oos_true.sum())
        flagged = oos_p > 0.5
        oos_recall = float((flagged & oos_true).sum()) / max(n_oos, 1)
        false_flag = float((flagged & in_scope).sum()) / max(n_in, 1)

        # Routing accuracy: the decision a deployed layer actually makes —
        # reject if flagged, otherwise take the intent. One number that both
        # heads can lose.
        routed = torch.where(flagged, torch.full_like(i_pred, -1), i_pred)
        route_hit = routed == torch.where(oos_true,
                                          torch.full_like(s.intent, -1),
                                          s.intent)

        return {
            "intent_acc": float(i_hit.sum()) / max(n_in, 1),
            "domain_acc": float(d_hit.sum()) / max(n_in, 1),
            "oos_recall": oos_recall,
            "oos_false_flag": false_flag,
            "route_acc": float(route_hit.float().mean()),
            # Calibration of the self-reported score against whether the
            # intent head was right, on in-scope queries only — the population
            # the question is about.
            "conf_ece": _ece(conf[in_scope], i_hit[in_scope]),
            "conf_brier": float(((conf[in_scope]
                                  - i_hit[in_scope].float()) ** 2).mean()),
            # Max-softmax calibration, for contrast: it is the signal the
            # dataset's own baselines used, and the comparison is the argument
            # for having asked the question directly.
            "msp_ece": _ece(msp[in_scope], i_hit[in_scope]),
            "oos_ece": _ece(oos_p, oos_true),
            "oos_brier": float(((oos_p - oos_true.float()) ** 2).mean()),
        }


def fmt_metrics(m):
    return (f"intent {m['intent_acc']:.2%} | domain {m['domain_acc']:.2%} | "
            f"oos-recall {m['oos_recall']:.2%} | false-flag "
            f"{m['oos_false_flag']:.2%} | route {m['route_acc']:.2%} | "
            f"ece {m['conf_ece']:.3f}")


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def ask(model, cfg, schema, queries):
    """Answer one or more queries the way a caller would."""
    was_training = model.training
    model.eval()
    x, _ = encode_queries(list(queries), cfg.max_bytes)
    x = torch.from_numpy(x).to(cfg.device)
    model.reset_state(batch_size=x.shape[0])
    intent_logits, domain_logits, oos_logit, conf_logits = decide(
        model, x, cfg, schema)
    p_intent = torch.softmax(intent_logits.float(), dim=-1)
    top = p_intent.topk(min(3, schema.n_intent), dim=-1)
    p_domain = torch.softmax(domain_logits.float(), dim=-1)
    out = []
    for b, q in enumerate(queries):
        out.append({
            "query": q,
            "intent": [(schema.intents[int(i)], float(p))
                       for p, i in zip(top.values[b], top.indices[b])],
            "domain": (schema.domains[int(p_domain[b].argmax())],
                       float(p_domain[b].max())),
            "oos": float(torch.sigmoid(oos_logit[b].float())),
            "confidence": float(score_from_logits(conf_logits[b:b + 1])[0]),
        })
    if was_training:
        model.train()
    return out


def print_answers(answers):
    for a in answers:
        head = a["intent"][0]
        verdict = ("OUT OF SCOPE" if a["oos"] > 0.5
                   else f"{head[0]} ({head[1]:.1%})")
        print(f"\n  ❯ {a['query']!r}")
        print(f"    → {verdict}")
        print(f"      domain {a['domain'][0]} ({a['domain'][1]:.1%}) | "
              f"p(oos) {a['oos']:.3f} | self-confidence "
              f"{a['confidence']:.2f}")
        alts = ", ".join(f"{n} {p:.1%}" for n, p in a["intent"][1:])
        if alts:
            print(f"      runners-up: {alts}")


# --------------------------------------------------------------------------- #
# Training session                                                            #
# --------------------------------------------------------------------------- #

#: Everything that defines how a checkpoint was built and scored; adopted by
#: `eval`/`ask`, whose job is to reproduce that.
ARCH_FIELDS = ("neurons", "n_in", "n_out", "max_bytes", "activation",
               "weight_init", "gates", "hebb_type", "hebb_res", "think_gap",
               "deliberate", "attn_heads", "attn_kv_heads", "attn_head_dim",
               "attn_window", "attn_write", "attn_read", "attn_rope",
               "attn_qk_norm")

#: The subset `--resume` adopts: what `build()` allocates. The rest cannot
#: change the state dict, so pinning them would only take away a knob —
#: `weight_init` is dead once weights load, and `think_gap` / `deliberate` are
#: forward-pass arguments.
RESUME_FIELDS = ("neurons", "n_in", "n_out", "max_bytes", "gates",
                 "hebb_type", "hebb_res", "attn_heads", "attn_kv_heads",
                 "attn_head_dim", "attn_qk_norm")


def adopt_saved_arch(cfg, path, fields):
    """Override architecture fields with the ones stored in a checkpoint."""
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
    base = os.path.join(CKPT_DIR, f"s1_odyss_{cfg.tag}")
    return base + "_latest.pth", base + "_best.pth"


def guard_overwrite(cfg, overwrite=False):
    """
    Refuse to start a fresh run on top of an existing tag's checkpoints.

    Without this, a fresh `--tag base` rebuilds from scratch with best = -inf,
    so the first validation counts as a record and overwrites both files —
    a long run destroyed seconds in, with nothing in the logs to warn you.
    A hard exit rather than a prompt, so the script stays usable from CI.
    """
    latest, best = ckpt_paths(cfg)
    existing = [p for p in (latest, best) if os.path.exists(p)]
    if not existing or overwrite:
        return
    raise SystemExit(
        f"\n✋ Refusing to overwrite existing checkpoints for tag '{cfg.tag}':\n"
        + "".join(f"     {p}\n" for p in existing)
        + f"\n   Continue that run:   --mode train --tag {cfg.tag} --resume\n"
        f"   Start somewhere new: --mode train --tag {cfg.tag}_v2\n"
        f"   Overwrite anyway:    --mode train --tag {cfg.tag} --overwrite\n"
    )


def _save(cfg, model, trainer, step, best, metrics, path):
    save_checkpoint(
        model, trainer.optimizer, step, metrics.get("route_acc", float("nan")),
        path,
        extra_data={
            "cfg": {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(cfg).items()},
            "best": best,
            "step": step,
            "metrics": metrics,
            "trainer_state_dict": trainer.state_dict(),
        },
    )


def run_session(cfg, corpus, budget_sec=0.0, resume=False, resume_best=False,
                quiet=False, save=True):
    """
    Train under a wall-clock, epoch and/or step budget. Returns final metrics.

    Used by --mode train (open-ended) and by every sweep arm (budgeted), so
    both paths exercise exactly the same code.
    """
    train_split, val_split, _, schema = corpus

    set_seed(cfg.seed)
    model, trainer = build(cfg, schema)
    batcher = Batcher(train_split, cfg.batch, cfg.device, seed=cfg.seed)
    evaluator = Evaluator(cfg, val_split, schema)
    latest_path, best_path = ckpt_paths(cfg)

    best = -float("inf")
    start_step = 0
    # `_latest` is rewritten at every evaluation and `_best` only on an
    # improvement, so the two diverge exactly when it matters: a collapse
    # overwrites `_latest` within one eval interval, while `_best`
    # structurally cannot hold a collapsed model.
    resume_path = best_path if resume_best else latest_path
    flag = "--resume-best" if resume_best else "--resume"
    if resume and not os.path.exists(resume_path):
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
        # Loading must not fail softly: --resume bypasses guard_overwrite, so a
        # run that continued past a failed load would train a random model and
        # save it over the checkpoint it could not read.
        try:
            data = load_checkpoint(model, trainer.optimizer, resume_path,
                                   device=cfg.device, strict=True, lr=cfg.lr)
        except Exception as e:
            raise SystemExit(
                f"\n✋ Could not load the checkpoint for tag '{cfg.tag}':\n"
                f"     {e}\n\n"
                f"   The file is intact; this run refuses to overwrite it "
                f"with a fresh model.\n"
                f"   Train elsewhere:   --mode train --tag {cfg.tag}_v2\n"
                f"   Start over anyway: --mode train --tag {cfg.tag} "
                f"--overwrite (destroys it)\n"
            ) from e

        if cfg.lr is None and cfg.lr_force_auto:
            # Explicit `--lr auto`: return a fixed-rate checkpoint to the
            # estimator. load_checkpoint(lr=None) means "don't touch", so this
            # is the only path back to auto.
            switched = any(g.get("lr") is not None
                           for g in trainer.optimizer.param_groups)
            for g in trainer.optimizer.param_groups:
                g["lr"] = None
            if switched:
                print("🔁 --lr auto: checkpoint was fixed-rate; optimizer "
                      "returned to ChaosGrad's online estimate.")

        best = data.get("best", -float("inf"))
        start_step = int(data.get("step", 0))
        ts = data.get("trainer_state_dict")
        if ts:
            # Outside the fatal guard: this payload can gain or lose keys
            # between versions, and a failure costs counters rather than
            # weights, with nothing trained yet.
            try:
                trainer.load_state_dict(ts)
            except Exception as e:
                print(f"⚠️  Weights and optimizer resumed, but the trainer "
                      f"state did not ({e}). Continuing with the loaded "
                      f"model; AMP scaler and anomaly history restart.")
        print(f"📂 Resumed from {os.path.basename(resume_path)} at step "
              f"{start_step:,} (best route {best:.2%})")

    # Report the mode the optimizer is ACTUALLY in, not what the CLI asked for:
    # under the default `--lr keep` a resumed checkpoint keeps its own mode.
    live_lr = (trainer.optimizer.param_groups[0].get("lr")
               if trainer.optimizer.param_groups else cfg.lr)
    steps = total_steps(cfg)
    if not quiet:
        params = model.get_num_params()
        table = 256 * cfg.n_in + schema.width * cfg.n_out
        print(f"\n🧠 {params:,} trainable params | {cfg.neurons} neurons | "
              f"core {cfg.neurons * (cfg.neurons - 1):,} + lookup "
              f"{table:,} ({table / params:.0%}) | "
              f"lr {'auto' if live_lr is None else live_lr}")
        print(f"⏱️  {steps} steps per query "
              f"({cfg.max_bytes} bytes x {cfg.think_gap + 1} + "
              f"{cfg.deliberate} deliberation) | batch {cfg.batch}")
        if model.attn is not None:
            print(f"👁️  attention {model.attn.heads}x{model.attn.head_dim} "
                  f"({model.attn.kv_heads} kv) | window {model.attn.window} | "
                  f"{sum(p.numel() for p in model.attn.parameters()):,} params")
        if cfg.hebb_type:
            print(f"🧬 plasticity {cfg.hebb_type}/{cfg.hebb_res}")

    metrics = {k: float("nan") for k in
               ("intent_acc", "domain_acc", "oos_recall", "oos_false_flag",
                "route_acc", "conf_ece", "conf_brier", "msp_ece", "oos_ece",
                "oos_brier")}
    step = start_step
    seen = 0
    run = {k: 0.0 for k in ("total", "intent", "domain", "oos", "conf")}
    run_n = 0
    t_start = time.time()
    interrupted = False

    try:
        while True:
            if budget_sec and time.time() - t_start >= budget_sec:
                break
            if cfg.max_steps and step - start_step >= cfg.max_steps:
                break
            if cfg.epochs and batcher.epochs >= cfg.epochs:
                break

            x, y_i, y_d, y_o = batcher.next()
            terms_box = {}

            def transform(out, _box=terms_box, _t=(y_i, y_d, y_o)):
                # train_batch owns the forward pass and the AMP context, so the
                # heads are split here rather than in a second call: the
                # transform sees the decoder output and returns the scalar the
                # loss closes over.
                _box["terms"], _ = head_losses(schema.split(out), _t, cfg,
                                               schema)
                return weighted(_box["terms"], cfg).unsqueeze(0)

            loss = trainer.train_batch(
                x, torch.zeros(1, device=cfg.device),
                thinking_steps=steps,
                full_sequence=False,
                output_transform=transform,
            )
            step += 1
            seen += x.shape[0]
            run["total"] += loss
            for k, v in terms_box.get("terms", {}).items():
                run[k] += float(v)
            run_n += 1

            if not quiet and cfg.log_every and step % cfg.log_every == 0:
                el = time.time() - t_start
                n = max(run_n, 1)
                print(f"step {step:>7,} | loss {run['total'] / n:6.4f} "
                      f"(intent {run['intent'] / n:5.3f} dom "
                      f"{run['domain'] / n:5.3f} oos {run['oos'] / n:5.3f} "
                      f"conf {run['conf'] / n:5.3f}) | "
                      f"lr {trainer._current_lr():.2e} | "
                      f"{seen / max(el, 1e-6):6,.0f} q/s | "
                      f"{batcher.epochs:5.2f} ep | {el / 60:5.1f}m")
                run = {k: 0.0 for k in run}
                run_n = 0

            if cfg.eval_every and step % cfg.eval_every == 0:
                metrics = evaluator.run(model)
                mark = ""
                if metrics["route_acc"] > best:
                    best = metrics["route_acc"]
                    mark = "  🏆"
                    if save:
                        _save(cfg, model, trainer, step, best, metrics,
                              best_path)
                if not quiet:
                    print(f"   ↳ VAL  {fmt_metrics(metrics)}{mark}")
                if save:
                    _save(cfg, model, trainer, step, best, metrics, latest_path)
    except KeyboardInterrupt:
        interrupted = True
        print("\n⏹️  Interrupted — finishing cleanly.")

    elapsed = max(time.time() - t_start, 1e-6)
    if step > start_step:
        metrics = evaluator.run(model)
        if save:
            if metrics["route_acc"] > best:
                best = metrics["route_acc"]
                _save(cfg, model, trainer, step, best, metrics, best_path)
            _save(cfg, model, trainer, step, best, metrics, latest_path)

    metrics.update({
        "steps": step - start_step,
        "queries": seen,
        "epochs": batcher.epochs,
        "q_s": seen / elapsed,
        "minutes": elapsed / 60.0,
        "params": model.get_num_params(),
        "neurons": cfg.neurons,
        "interrupted": interrupted,
    })
    return metrics, model, trainer


# --------------------------------------------------------------------------- #
# Sweeps                                                                      #
# --------------------------------------------------------------------------- #

SWEEPS = {
    # The thesis test: does deliberation after the last byte pay for the steps
    # it costs? Every arm gets the same wall-clock, so more thinking means
    # fewer queries.
    "think": [
        ("delib0", dict(deliberate=0)),
        ("delib4", dict(deliberate=4)),
        ("delib8", dict(deliberate=8)),
        ("delib16", dict(deliberate=16)),
    ],
    # Depth per byte instead of depth after the query.
    "gap": [
        ("gap0", dict(think_gap=0)),
        ("gap1", dict(think_gap=1)),
        ("gap2", dict(think_gap=2)),
    ],
    # Parameter efficiency at equal wall-clock. `frontier` mode walks the same
    # axis to convergence; this is the cheap version.
    "scale": [
        ("n96", dict(neurons=96, n_in=48, n_out=48)),
        ("n128", dict(neurons=128, n_in=64, n_out=64)),
        ("n192", dict(neurons=192, n_in=96, n_out=96)),
        ("n256", dict(neurons=256, n_in=128, n_out=128)),
    ],
    # Does letting each step query the states before it pay for itself? `off`
    # builds no attention and is the only arm without a KV cache; its output
    # projection starts at zero and the module is built after the core, so
    # every arm begins from the same W.
    "attn": [
        ("off", dict(attn_heads=0)),
        ("mqa2", dict(attn_heads=2)),
        ("mqa4", dict(attn_heads=4)),
        ("mha4", dict(attn_heads=4, attn_kv_heads=4)),
    ],
    # Plasticity against the same seed's plain core. The gain is
    # zero-initialized and construction draws no RNG, so this is a one-variable
    # ablation.
    "hebb": [
        ("off", dict(hebb_type=None)),
        ("temporal", dict(hebb_type="temporal")),
        ("both", dict(hebb_type="both")),
        ("global", dict(hebb_type="temporal", hebb_res="global")),
    ],
    # How much the minority decision should be worth. CLINC150 is 150:1
    # in-scope, and the two failure directions trade against each other:
    # read oos_recall and oos_false_flag together, not separately.
    "oos": [
        ("w1", dict(oos_weight=1.0)),
        ("w4", dict(oos_weight=4.0)),
        ("w8", dict(oos_weight=8.0)),
        ("w16", dict(oos_weight=16.0)),
    ],
    # Where the step scale should sit. One supervision signal per query is a
    # longer credit path than the LLM path's signal-per-token, and the online
    # estimator climbs to a traction cap that is too large for it here.
    "lr": [
        ("auto", dict(lr=None)),
        ("1e-3", dict(lr=1e-3)),
        ("3e-4", dict(lr=3e-4)),
        ("1e-4", dict(lr=1e-4)),
    ],
    # Is the deliberation tail better spent on a wider query window?
    "bytes": [
        ("b48", dict(max_bytes=48)),
        ("b64", dict(max_bytes=64)),
        ("b96", dict(max_bytes=96)),
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
    print(f"\n{'=' * 78}")
    print(f"🔬 SWEEP '{name}' — {len(plan)} arms x {minutes:.1f} min "
          f"(compute-matched, seed {cfg.seed})")
    print(f"{'=' * 78}")

    results = []
    for i, (arm, over) in enumerate(plan, 1):
        # eval_every=0: mid-run validation would spend budget, and each arm
        # would spend a different amount of it. Every arm is scored once,
        # after the clock stops.
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
        print(f"   {fmt_metrics(m)}")
        print(f"   {m['params']:,} params | {m['epochs']:.1f} epochs | "
              f"{m['q_s']:,.0f} q/s")
        del model
        _empty_cache(cfg.device)

    if not results:
        return results

    results.sort(key=lambda r: -r["route_acc"])
    print(f"\n{'=' * 100}")
    print(f"🏁 SWEEP '{name}' RESULTS — ranked by routing accuracy")
    print(f"{'=' * 100}")
    print(f"{'arm':<10} {'route':>8} {'intent':>8} {'domain':>8} "
          f"{'oos-rec':>8} {'false':>7} {'ece':>7} {'params':>10} {'epochs':>7}")
    print("-" * 100)
    for r in results:
        print(f"{r['arm']:<10} {r['route_acc']:>7.2%} {r['intent_acc']:>7.2%} "
              f"{r['domain_acc']:>7.2%} {r['oos_recall']:>7.2%} "
              f"{r['oos_false_flag']:>6.2%} {r['conf_ece']:>7.3f} "
              f"{r['params']:>10,} {r['epochs']:>7.1f}")
    print("-" * 100)
    print(f"🥇 {results[0]['arm']}")

    out = os.path.join(CKPT_DIR, f"s1_sweep_{name}_results.json")
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"📄 {out}")
    return results


# --------------------------------------------------------------------------- #
# Parameter-efficiency frontier                                               #
# --------------------------------------------------------------------------- #

FRONTIER = ((48, 24, 24), (64, 32, 32), (96, 48, 48), (128, 64, 64),
            (192, 96, 96), (256, 128, 128))


def run_frontier(cfg, corpus, minutes, sizes=None):
    """
    The same recipe at every core size, each arm given the same wall-clock.

    This is the curve the file exists to produce: decisions-per-parameter
    rather than a single headline number. Printed with the lookup table split
    out, because at these sizes the embedding is a real fraction of the model
    and a reader comparing against a transformer needs to know which half is
    which.
    """
    grid = sizes or FRONTIER
    budget = minutes * 60.0
    print(f"\n{'=' * 78}")
    print(f"📈 FRONTIER — {len(grid)} core sizes x {minutes:.1f} min "
          f"(seed {cfg.seed})")
    print(f"{'=' * 78}")

    rows = []
    for i, (n, n_in, n_out) in enumerate(grid, 1):
        arm_cfg = replace(cfg, neurons=n, n_in=n_in, n_out=n_out,
                          tag=f"frontier_n{n}", eval_every=0)
        print(f"\n[{i}/{len(grid)}] ▶ {n} neurons ({n_in} in / {n_out} out)")
        try:
            m, model, _ = run_session(arm_cfg, corpus, budget_sec=budget,
                                      quiet=True, save=False)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            print("   ✗ OOM — size skipped")
            _empty_cache(cfg.device)
            continue
        schema = corpus[3]
        m["lookup"] = 256 * n_in + schema.width * n_out
        rows.append(m)
        print(f"   {fmt_metrics(m)}")
        print(f"   {m['params']:,} params | {m['epochs']:.1f} epochs")
        del model
        _empty_cache(cfg.device)

    if not rows:
        return rows

    print(f"\n{'=' * 96}")
    print("📈 PARAMETER-EFFICIENCY FRONTIER")
    print(f"{'=' * 96}")
    print(f"{'neurons':>8} {'params':>10} {'lookup':>9} {'route':>8} "
          f"{'intent':>8} {'oos-rec':>8} {'ece':>7} {'route/kparam':>13}")
    print("-" * 96)
    for r in rows:
        print(f"{r['neurons']:>8} {r['params']:>10,} "
              f"{r['lookup'] / r['params']:>8.0%} {r['route_acc']:>7.2%} "
              f"{r['intent_acc']:>7.2%} {r['oos_recall']:>7.2%} "
              f"{r['conf_ece']:>7.3f} "
              f"{100 * r['route_acc'] / (r['params'] / 1000):>13.3f}")
    print("-" * 96)

    out = os.path.join(CKPT_DIR, "s1_frontier_results.json")
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"📄 {out}")
    return rows


def _empty_cache(device):
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Smoke test                                                                  #
# --------------------------------------------------------------------------- #

def run_smoke(cfg, corpus):
    """
    Fast end-to-end check of every path the heavy run depends on: encoding,
    all four heads, training, evaluation, inference, checkpoint round-trip,
    attention and plasticity. Seconds, not hours — this is what makes the
    expensive script safe to modify.
    """
    print(f"\n{'=' * 78}")
    print("🔥 SMOKE TEST")
    print(f"{'=' * 78}")

    # Clear our own litter first: the round-trip below trains with save=True
    # and no --resume, so leftovers would trip guard_overwrite and fail the
    # test on a second invocation. Running smoke twice has to be a no-op.
    for stale in glob.glob(os.path.join(CKPT_DIR, "s1_odyss_smoke*.pth")):
        os.remove(stale)

    schema = corpus[3]
    base = replace(cfg, neurons=64, n_in=32, n_out=32, max_bytes=16,
                   deliberate=2, batch=16, max_steps=12, eval_every=6,
                   log_every=6, eval_batch=64, tag="smoke")

    variants = [
        ("plain", dict()),
        ("gap1", dict(think_gap=1)),
        ("hebbian", dict(hebb_type="both", hebb_res="neuron")),
        ("hebb_global", dict(hebb_type="temporal", hebb_res="global")),
        ("gated", dict(gates=("none", "sigmoid", "sigmoid"))),
        ("attn", dict(attn_heads=4)),
        # Everything the attention path can be configured into that changes its
        # shapes or bookkeeping: multi-head instead of multi-query, a window
        # narrower than the sequence so eviction happens mid-run, an entry per
        # step, and no rotary positions.
        ("attn_mha", dict(attn_heads=4, attn_kv_heads=4, attn_window=4,
                          attn_write="step", attn_rope=False)),
        ("grad_ckpt", dict(grad_ckpt=True, lr=1e-4)),
        ("no_oos_train", dict(oos_train=False)),
    ]

    ok = True
    for name, over in variants:
        c = replace(base, **over)
        t0 = time.time()
        try:
            # oos_train changes the corpus, not the model, so that arm needs
            # its own encode rather than the shared one.
            arm_corpus = load_corpus(c, verbose=False) if "oos_train" in over \
                else corpus
            m, model, trainer = run_session(c, arm_corpus, quiet=True,
                                            save=False)
            answers = ask(model, c, schema, ["cancel my flight", "hi there"])
            assert m["steps"] == c.max_steps, \
                f"ran {m['steps']} of {c.max_steps} steps"
            for k in ("intent_acc", "oos_recall", "route_acc", "conf_ece"):
                assert math.isfinite(m[k]), f"non-finite {k}"
            assert 0.0 <= answers[0]["oos"] <= 1.0, "p(oos) out of range"
            assert 0.0 <= answers[0]["confidence"] <= 1.0, "score out of range"
            diag = trainer.get_diagnostics()
            print(f"  ✅ {name:<13} route {m['route_acc']:6.2%} | "
                  f"intent {m['intent_acc']:6.2%} | ece {m['conf_ece']:5.3f} | "
                  f"lr {diag['current_lr']:.2e} | {time.time() - t0:4.1f}s")
        except Exception as e:
            ok = False
            print(f"  ❌ {name:<13} {type(e).__name__}: {e}")

    # Every head must actually receive gradient. A head whose slice never moves
    # is the quiet failure mode here: the run trains, the loss falls, and one
    # of the four decisions is a constant nobody checked.
    print("\n  → every head trains")
    try:
        c = replace(base, max_steps=0, eval_every=0, tag="smoke_heads")
        set_seed(c.seed)
        model, trainer = build(c, schema)
        before = model.output_decoder.weight.detach().clone()
        batcher = Batcher(corpus[0], c.batch, c.device, seed=c.seed)
        x, y_i, y_d, y_o = batcher.next()
        steps = total_steps(c)

        def transform(out, _t=(y_i, y_d, y_o)):
            terms, _ = head_losses(schema.split(out), _t, c, schema)
            return weighted(terms, c).unsqueeze(0)


        for _ in range(3):
            trainer.train_batch(x, torch.zeros(1, device=c.device),
                                thinking_steps=steps, full_sequence=False,
                                output_transform=transform)
        moved = (model.output_decoder.weight.detach() - before).abs()
        for label, sl in (("intent", schema.sl_intent),
                          ("domain", schema.sl_domain),
                          ("oos", slice(schema.i_oos, schema.i_oos + 1)),
                          ("conf", schema.sl_conf)):
            delta = float(moved[sl].max())
            good = delta > 0.0
            ok = ok and good
            print(f"  {'✅' if good else '❌'} {label:<6} head moved "
                  f"{delta:.2e}")
    except Exception as e:
        ok = False
        print(f"  ❌ head gradients {type(e).__name__}: {e}")

    # Checkpoint round-trip: save, rebuild, reload, require identical metrics.
    # Run once per architecture that changes the state dict — attention adds
    # six tensors and a runtime cache, and a checkpoint that silently drops
    # either is a corrupted resume rather than an error.
    print("\n  → checkpoint round-trip")
    for label, over in (("plain", dict()), ("attn", dict(attn_heads=4))):
        try:
            c = replace(base, tag=f"smoke_ckpt_{label}", eval_every=0, **over)
            _, model1, _ = run_session(c, corpus, quiet=True, save=True)
            evaluator = Evaluator(c, corpus[1], schema)
            v1 = evaluator.run(model1)["route_acc"]
            del model1

            model2, trainer2 = build(c, schema)
            load_checkpoint(model2, trainer2.optimizer, ckpt_paths(c)[0],
                            device=c.device, strict=True)
            v2 = evaluator.run(model2)["route_acc"]
            if abs(v1 - v2) < 1e-9:
                print(f"  ✅ round-trip {label:<6} {v1:.6f} == {v2:.6f}")
            else:
                ok = False
                print(f"  ❌ round-trip {label:<6} {v1:.6f} != {v2:.6f}")
        except Exception as e:
            ok = False
            print(f"  ❌ round-trip {label:<6} {type(e).__name__}: {e}")

    # Learning check: intent accuracy must clear chance on a real run, which
    # catches a pipeline that trains without learning — label misalignment, a
    # readout fed the wrong step, a dead decoder. The bar is deliberately
    # modest: one supervision signal arrives per query, so this task needs
    # tens of epochs before its accuracy is interesting and a smoke test gets
    # eight. What is being tested is that the signal is connected.
    print("\n  → learning check (1000 steps)")
    c = replace(base, neurons=96, n_in=48, n_out=48, max_bytes=32,
                deliberate=4, batch=128, max_steps=1000, eval_every=0,
                log_every=0, eval_batch=512, tag="smoke_learn")
    m, model, _ = run_session(c, corpus, quiet=True, save=False)
    chance = 1.0 / schema.n_intent
    good = m["intent_acc"] > 4 * chance
    ok = ok and good
    print(f"  {'✅' if good else '❌'} intent {m['intent_acc']:.2%} vs chance "
          f"{chance:.2%} ({m['epochs']:.1f} epochs) | domain "
          f"{m['domain_acc']:.2%} | route {m['route_acc']:.2%}")
    print_answers(ask(model, c, schema,
                      ["what's my checking account balance",
                       "how many kilometers to the moon"]))

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
  %(prog)s --mode train --tag base --minutes 20

  # continue it
  %(prog)s --mode train --tag base --resume --minutes 20

  # a fixed number of passes over the corpus instead of a clock
  %(prog)s --mode train --tag base --epochs 40

  # compute-matched ablations; every arm gets the same wall-clock
  %(prog)s --mode sweep --sweep think --minutes 3
  %(prog)s --mode sweep --sweep attn  --arms off,mqa4 --minutes 5
  %(prog)s --mode sweep --sweep oos   --minutes 3

  # the parameter-efficiency curve: same recipe, every core size
  %(prog)s --mode frontier --minutes 4

  # temporal attention on, deliberation lengthened
  %(prog)s --mode train --tag attn --attn-heads 4 --deliberate 16

  # score or query a saved checkpoint (architecture is read from the file)
  %(prog)s --mode eval --tag base
  %(prog)s --mode eval --tag base --split val
  %(prog)s --mode ask  --tag base --query "cancel my flight to denver"
  %(prog)s --mode ask  --tag base --query "what is the capital of peru"
"""

#: Appended to every "unknown name" rejection. These lists are transcribed from
#: OdyssNet's `_build_activation` and `_apply_init`, which are if/elif chains
#: over string literals and cannot be introspected — so a strategy added to the
#: library is rejected here until someone updates the copy.
_STALE = (". If the library gained this name recently, the accepted list in "
          "parse_args() needs updating.")

_ACTS = ("none", "identity", "tanh", "relu", "leaky_relu", "sigmoid", "gelu",
         "gelu_tanh", "silu")
_INITS = ("quiet", "micro_quiet", "micro_quiet_warm", "classic",
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
                   choices=["train", "sweep", "frontier", "smoke", "eval",
                            "ask"],
                   help="train: (resumable) training loop. sweep: "
                        "compute-matched ablation. frontier: the same recipe "
                        "at every core size. smoke: fast self-test. "
                        "eval/ask: score or query a saved checkpoint. "
                        "(default: %(default)s)")
    g.add_argument("--sweep", default="think", choices=sorted(SWEEPS),
                   help="which ablation preset to run in --mode sweep "
                        "(default: %(default)s)")
    g.add_argument("--arms", default=None, metavar="A,B",
                   help="comma-separated subset of the preset's arms")
    g.add_argument("--split", default="test", choices=("val", "test"),
                   help="which split --mode eval scores. 'val' is what "
                        "training selected the checkpoint on, so 'test' is "
                        "the honest number and the default. "
                        "(default: %(default)s)")

    g = p.add_argument_group("corpus")
    g.add_argument("--max-bytes", type=int, default=d.max_bytes, metavar="N",
                   help="query bytes kept; longer queries are truncated, not "
                        "dropped. 64 holds 93%% of CLINC150 whole, and every "
                        "byte is a timestep, so this is also the sequence "
                        "length (default: %(default)s)")
    g.add_argument("--no-oos-train", action="store_true",
                   help="exclude the 100 out-of-scope training queries, "
                        "making out-of-scope detection a pure zero-shot "
                        "question about the in-scope decision boundary")

    g = p.add_argument_group("architecture")
    g.add_argument("--neurons", type=int, default=d.neurons, metavar="N",
                   help="size of the NxN chaos core (default: %(default)s)")
    g.add_argument("--n-in", type=int, default=d.n_in, metavar="N",
                   help="neurons receiving the byte embedding "
                        "(default: %(default)s)")
    g.add_argument("--n-out", type=int, default=d.n_out, metavar="N",
                   help="neurons the decision decoder reads; n_in + n_out "
                        "must be <= --neurons (default: %(default)s)")
    g.add_argument("--think-gap", type=int, default=d.think_gap, metavar="N",
                   help="extra echo steps per byte; total steps per byte is "
                        "N+1 (default: %(default)s)")
    g.add_argument("--deliberate", type=int, default=d.deliberate, metavar="N",
                   help="steps run after the last byte, on recurrent state "
                        "alone, before the decision is read. This is the "
                        "knob that buys depth without buying parameters "
                        "(default: %(default)s)")
    g.add_argument("--activation", default=",".join(d.activation),
                   metavar="ENC,CORE,MEM",
                   help="three activations: encoder/decoder, the core step, "
                        "and memory feedback. 'none' means identity here "
                        "(unlike --gates, where it removes the gate). "
                        "(default: %(default)s)")
    g.add_argument("--weight-init", default=",".join(d.weight_init),
                   metavar="ENC,CORE,MEM,GATE",
                   help="four initialization strategies: encoder/decoder, the "
                        "chaos core, memory feedback, and gates "
                        "(default: %(default)s)")
    g.add_argument("--gates", default=",".join(d.gates), metavar="IN,CORE,MEM",
                   help="three gate activations, applied to input/output "
                        "scaling, the core signal, and memory feedback. "
                        "'none' disables a gate outright and creates no "
                        "parameter for it (default: %(default)s)")
    g.add_argument("--hebb", default="none",
                   choices=["none", "temporal", "spatial", "both"],
                   help="Hebbian plasticity mechanism. Note this is the one "
                        "thing that couples batch rows (default: %(default)s)")
    g.add_argument("--hebb-res", default=d.hebb_res,
                   choices=["global", "neuron"],
                   help="plasticity resolution: one factor for the whole "
                        "core, or one per neuron (default: %(default)s)")
    g.add_argument("--dropout", type=float, default=d.dropout, metavar="P",
                   help="dropout applied every thinking step "
                        "(default: %(default)s)")

    g = p.add_argument_group(
        "attention",
        "Temporal attention: at every step the state queries a cache of the "
        "states before it. Off unless --attn-heads is given, and its output "
        "projection starts at zero, so switching it on changes nothing until "
        "training decides otherwise.")
    g.add_argument("--attn-heads", type=int, default=d.attn_heads, metavar="N",
                   help="query heads; 0 disables attention entirely and costs "
                        "nothing (default: %(default)s)")
    g.add_argument("--attn-kv-heads", type=int, default=d.attn_kv_heads,
                   metavar="N",
                   help="key/value heads, shared across query heads (must "
                        "divide --attn-heads). 1 is multi-query "
                        "(default: %(default)s)")
    g.add_argument("--attn-head-dim", type=int, default=d.attn_head_dim,
                   metavar="N",
                   help="width per head; 0 derives it from --neurons and "
                        "--attn-heads (default: %(default)s)")
    g.add_argument("--attn-window", type=int, default=d.attn_window,
                   metavar="N",
                   help="cache entries a query can see; the oldest are "
                        "evicted first (default: %(default)s)")
    g.add_argument("--attn-write", default=d.attn_write,
                   choices=("token", "step"),
                   help="'token' writes one cache entry per byte; 'step' "
                        "writes every thinking step. Identical at "
                        "--think-gap 0 (default: %(default)s)")
    g.add_argument("--attn-read", default=d.attn_read,
                   choices=("token", "step"),
                   help="'step' queries the cache on every step; 'token' only "
                        "on the step a byte arrives (default: %(default)s)")
    g.add_argument("--attn-rope", action=argparse.BooleanOptionalAction,
                   default=d.attn_rope,
                   help="rotary position embedding, applied to each key when "
                        "it is written (default: %(default)s)")
    g.add_argument("--attn-qk-norm", action=argparse.BooleanOptionalAction,
                   default=d.attn_qk_norm,
                   help="RMSNorm on queries and keys before the dot product "
                        "(default: %(default)s)")
    g.add_argument("--attn-dropout", type=float, default=d.attn_dropout,
                   metavar="P",
                   help="dropout on the attention weights during training "
                        "(default: %(default)s)")

    g = p.add_argument_group("decisions")
    g.add_argument("--oos-weight", type=float, default=d.oos_weight,
                   metavar="W",
                   help="positive-class weight on the out-of-scope noul head. "
                        "The corpus is ~150:1 in-scope, so at 1.0 the loss "
                        "barely notices the minority decision. Raising it "
                        "trades false flags for recall — read both columns "
                        "(default: %(default)s)")
    g.add_argument("--w-intent", type=float, default=d.w_intent, metavar="W",
                   help="loss weight, 150-way intent choice "
                        "(default: %(default)s)")
    g.add_argument("--w-domain", type=float, default=d.w_domain, metavar="W",
                   help="loss weight, 10-way domain choice. Low by default "
                        "because it is implied by the intent and is here as a "
                        "coarse auxiliary signal (default: %(default)s)")
    g.add_argument("--w-oos", type=float, default=d.w_oos, metavar="W",
                   help="loss weight, in-scope noul (default: %(default)s)")
    g.add_argument("--w-conf", type=float, default=d.w_conf, metavar="W",
                   help="loss weight, confidence score. This head predicts "
                        "the intent head's own correctness "
                        "(default: %(default)s)")

    g = p.add_argument_group("optimization")
    g.add_argument("--batch", type=int, default=d.batch, metavar="N",
                   help="queries per optimizer step (default: %(default)s)")
    g.add_argument("--lr", default="keep", metavar="RATE",
                   help="'auto' for ChaosGrad's online estimate, a float for "
                        "fixed-rate mode. Default 'keep': a fresh run uses "
                        "auto; --resume keeps the mode stored in the "
                        "checkpoint")
    g.add_argument("--label-smoothing", type=float, default=d.label_smoothing,
                   metavar="P",
                   help="applied to the two choice heads' training loss only "
                        "(default: %(default)s)")
    g.add_argument("--grad-persistence", type=float, default=d.grad_persistence,
                   metavar="P",
                   help="'ghost gradients': keep this fraction of the "
                        "previous step's gradient and add it to the next. "
                        "Momentum on top of ChaosGrad's own, so a "
                        "second-order knob (default: %(default)s)")
    g.add_argument("--grad-ckpt", action="store_true",
                   help="gradient checkpointing: less memory, one extra "
                        "sequential forward per step. Dynamo does not trace a "
                        "checkpointed region, so it does not combine with "
                        "--compile")
    g.add_argument("--compile", action="store_true",
                   help="torch.compile the forward pass. The echo loop issues "
                        "many small kernels per step and is bound by "
                        "launching them, so fusing it is the biggest speedup "
                        "on offer. The price is a warmup of a minute or two")

    g = p.add_argument_group("run control")
    g.add_argument("--minutes", type=float, default=d.minutes, metavar="M",
                   help="wall-clock budget; 0 runs until Ctrl-C, which "
                        "checkpoints cleanly on the way out "
                        "(default: %(default)s)")
    g.add_argument("--max-steps", type=int, default=d.max_steps, metavar="N",
                   help="optimizer-step budget; 0 is unlimited "
                        "(default: %(default)s)")
    g.add_argument("--epochs", type=float, default=d.epochs, metavar="E",
                   help="passes over the training split; 0 is unlimited. The "
                        "corpus is 15k queries, so this is the budget that "
                        "makes two runs comparable at equal data rather than "
                        "equal time (default: %(default)s)")
    g.add_argument("--tag", default=d.tag, metavar="NAME",
                   help="names the checkpoint pair "
                        "s1_odyss_<NAME>_{latest,best}.pth "
                        "(default: %(default)s)")
    g.add_argument("--resume", action="store_true",
                   help="continue --tag's latest checkpoint, restoring "
                        "weights and optimizer state")
    g.add_argument("--resume-best", action="store_true",
                   help="resume from --tag's *best* checkpoint instead of its "
                        "latest; implies --resume. This is the recovery path "
                        "after a collapse: `_latest` is rewritten at every "
                        "evaluation while `_best` is only ever written on an "
                        "improvement. Refuses to run if the tag has a latest "
                        "but no best, rather than starting fresh over it")
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
    g.add_argument("--eval-batch", type=int, default=d.eval_batch, metavar="N",
                   help="queries per evaluation forward pass "
                        "(default: %(default)s)")
    g.add_argument("--eval-amp", action="store_true",
                   help="evaluate under AMP; faster, and the metric is no "
                        "longer bit-stable across kernel choices")
    g.add_argument("--log-every", type=int, default=d.log_every, metavar="N",
                   help="steps between progress lines; 0 is silent "
                        "(default: %(default)s)")

    g = p.add_argument_group("inference")
    g.add_argument("--query", action="append", default=None, metavar="TEXT",
                   help="query to answer in --mode ask; repeat for several")

    a = p.parse_args()

    # --resume-best is a variant of --resume, not an alternative: every
    # downstream check asks `a.resume`, so folding it in here keeps the two
    # from having to be tested separately at each site.
    if a.resume_best:
        a.resume = True

    # Validate here rather than at model-construction time so a typo costs a
    # second and an error message, not a corpus download and a CUDA init.
    for name in ("neurons", "n_in", "n_out", "max_bytes", "batch",
                 "eval_batch"):
        if getattr(a, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive, "
                    f"got {getattr(a, name)}")
    if a.n_in + a.n_out > a.neurons:
        p.error(f"--n-in ({a.n_in}) + --n-out ({a.n_out}) = "
                f"{a.n_in + a.n_out} exceeds --neurons ({a.neurons})")
    if a.think_gap < 0:
        p.error(f"--think-gap must be >= 0, got {a.think_gap}")
    if a.deliberate < 0:
        p.error(f"--deliberate must be >= 0, got {a.deliberate}")
    if str(a.lr) not in ("auto", "keep"):
        try:
            if float(a.lr) <= 0:
                raise ValueError
        except ValueError:
            p.error(f"--lr must be 'auto', 'keep' or a positive float, "
                    f"got {a.lr!r}")
    if a.oos_weight <= 0:
        p.error(f"--oos-weight must be positive, got {a.oos_weight}")
    for name in ("w_intent", "w_domain", "w_oos", "w_conf"):
        if getattr(a, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be >= 0, "
                    f"got {getattr(a, name)}")
    if max(a.w_intent, a.w_domain, a.w_oos, a.w_conf) <= 0:
        p.error("at least one head weight must be positive, or there is "
                "nothing to train")
    # Upper bound is the trainer's documented range. Values at or above 1.0
    # make the carried gradient non-decaying, which is a divergence rather
    # than a setting.
    if not 0.0 <= a.grad_persistence <= 0.9:
        p.error(f"--grad-persistence must be between 0.0 and 0.9, "
                f"got {a.grad_persistence}")

    a.gates = tuple(s.strip() for s in a.gates.split(","))
    if len(a.gates) != 3:
        p.error(f"--gates needs exactly three comma-separated entries "
                f"(input,core,memory), got {len(a.gates)}: {','.join(a.gates)}")
    unknown = [s for s in a.gates if s.lower() not in _ACTS]
    if unknown:
        p.error(f"--gates: unknown activation(s) {unknown}; "
                f"choose from {list(_ACTS)}{_STALE}")

    a.activation = tuple(s.strip() for s in a.activation.split(","))
    if len(a.activation) != 3:
        p.error(f"--activation needs exactly three comma-separated entries "
                f"(encoder,core,memory), got {len(a.activation)}: "
                f"{','.join(a.activation)}")
    unknown = [s for s in a.activation if s.lower() not in _ACTS]
    if unknown:
        p.error(f"--activation: unknown activation(s) {unknown}; "
                f"choose from {list(_ACTS)}{_STALE}")

    a.weight_init = tuple(s.strip() for s in a.weight_init.split(","))
    if len(a.weight_init) != 4:
        p.error(f"--weight-init needs exactly four comma-separated entries "
                f"(encoder,core,memory,gate), got {len(a.weight_init)}: "
                f"{','.join(a.weight_init)}")
    unknown = [s for s in a.weight_init if s.lower() not in _INITS]
    if unknown:
        p.error(f"--weight-init: unknown strategy/strategies {unknown}; "
                f"choose from {list(_INITS)}{_STALE}")

    # Attention geometry, checked here for the same reason as everything else
    # in this function: the library raises the same errors, but only after the
    # corpus is loaded and CUDA is up.
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
            p.error(f"--attn-dropout must be in [0.0, 1.0), "
                    f"got {a.attn_dropout}")
    elif any((a.attn_kv_heads != d.attn_kv_heads,
              a.attn_head_dim != d.attn_head_dim,
              a.attn_window != d.attn_window, a.attn_write != d.attn_write,
              a.attn_read != d.attn_read, a.attn_rope != d.attn_rope,
              a.attn_qk_norm != d.attn_qk_norm,
              a.attn_dropout != d.attn_dropout)):
        print("ℹ️  attention flags ignored: --attn-heads is 0, so no attention "
              "is built. Pass --attn-heads 4 to switch it on.")
    return a


def cfg_from_args(a):
    d = Cfg()
    return Cfg(
        max_bytes=a.max_bytes, oos_train=not a.no_oos_train,
        neurons=a.neurons, n_in=a.n_in, n_out=a.n_out,
        activation=a.activation, weight_init=a.weight_init, gates=a.gates,
        hebb_type=None if a.hebb == "none" else a.hebb, hebb_res=a.hebb_res,
        dropout=a.dropout,
        attn_heads=a.attn_heads, attn_kv_heads=a.attn_kv_heads,
        attn_head_dim=a.attn_head_dim, attn_window=a.attn_window,
        attn_write=a.attn_write, attn_read=a.attn_read, attn_rope=a.attn_rope,
        attn_qk_norm=a.attn_qk_norm, attn_dropout=a.attn_dropout,
        think_gap=a.think_gap, deliberate=a.deliberate,
        batch=a.batch,
        lr=(d.lr if str(a.lr) == "keep"
            else None if str(a.lr) == "auto" else float(a.lr)),
        lr_force_auto=(str(a.lr) == "auto"),
        label_smoothing=a.label_smoothing, oos_weight=a.oos_weight,
        w_intent=a.w_intent, w_domain=a.w_domain, w_oos=a.w_oos,
        w_conf=a.w_conf,
        grad_persistence=a.grad_persistence, grad_ckpt=a.grad_ckpt,
        compile=a.compile,
        seed=a.seed, device=a.device, minutes=a.minutes,
        max_steps=a.max_steps, epochs=a.epochs, eval_every=a.eval_every,
        eval_batch=a.eval_batch, eval_amp=a.eval_amp, log_every=a.log_every,
        tag=a.tag,
    )


def main():
    a = parse_args()
    cfg = cfg_from_args(a)
    set_seed(cfg.seed)

    if cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print("🚀 OdyssNet-SystemOne — typed decisions from one pass")
    print(f"   mode {a.mode} | device {cfg.device} | seed {cfg.seed}")

    # Rebuild from the architecture a checkpoint was trained with, not from
    # whatever flags are on this command line, and do it before the corpus is
    # encoded: `max_bytes` decides the sequence length. An architecture flag
    # omitted on --resume otherwise reverts to a CLI default, fails the strict
    # load, and leaves a run that --resume has already excused from
    # guard_overwrite free to overwrite the checkpoint.
    path = None
    if a.mode in ("eval", "ask"):
        latest, best = ckpt_paths(cfg)
        path = best if os.path.exists(best) else latest
        if not os.path.exists(path):
            raise SystemExit(
                f"No checkpoint for tag '{cfg.tag}' (looked for {path})")
        cfg = adopt_saved_arch(cfg, path, fields=ARCH_FIELDS)
    elif a.mode == "train" and a.resume:
        latest, best = ckpt_paths(cfg)
        # Read the architecture from the file the run will actually load, or
        # --resume-best would rebuild from the wrong one and fail the strict
        # load it is trying to rescue.
        src = best if a.resume_best else latest
        if os.path.exists(src):
            cfg = adopt_saved_arch(cfg, src, fields=RESUME_FIELDS)

    corpus = load_corpus(cfg)
    schema = corpus[3]

    if a.mode == "smoke":
        sys.exit(0 if run_smoke(cfg, corpus) else 1)

    if a.mode == "sweep":
        run_sweep(cfg, corpus, a.sweep, a.minutes or 3.0, arms=a.arms)
        return

    if a.mode == "frontier":
        run_frontier(cfg, corpus, a.minutes or 4.0)
        return

    if a.mode in ("eval", "ask"):
        set_seed(cfg.seed)
        model, trainer = build(cfg, schema)
        load_checkpoint(model, trainer.optimizer, path, device=cfg.device,
                        strict=True)
        print(f"📂 {path}")
        if a.mode == "eval":
            split = corpus[1] if a.split == "val" else corpus[2]
            m = Evaluator(cfg, split, schema).run(model)
            print(f"\n📊 {a.split} split, {len(split):,} queries, "
                  f"{model.get_num_params():,} params")
            print(f"   intent      {m['intent_acc']:.2%}   (150-way choice)")
            print(f"   domain      {m['domain_acc']:.2%}   (10-way choice)")
            print(f"   oos recall  {m['oos_recall']:.2%}   "
                  f"false flags {m['oos_false_flag']:.2%}")
            print(f"   routing     {m['route_acc']:.2%}   "
                  f"(reject-or-route, the decision a caller consumes)")
            print(f"   calibration ece {m['conf_ece']:.4f} | brier "
                  f"{m['conf_brier']:.4f}   "
                  f"(max-softmax ece {m['msp_ece']:.4f})")
            print(f"   oos calib   ece {m['oos_ece']:.4f} | brier "
                  f"{m['oos_brier']:.4f}")
        else:
            queries = a.query or ["cancel my flight to denver",
                                  "what is the capital of peru"]
            print_answers(ask(model, cfg, schema, queries))
        return

    # --- train ---
    if not a.resume:
        guard_overwrite(cfg, overwrite=a.overwrite)

    metrics, model, _ = run_session(
        cfg, corpus, budget_sec=cfg.minutes * 60.0, resume=a.resume,
        resume_best=a.resume_best)

    print(f"\n{'=' * 78}")
    print(f"📊 FINAL (val)  {fmt_metrics(metrics)}")
    print(f"   {metrics['params']:,} params | {metrics['queries']:,} queries "
          f"in {metrics['minutes']:.1f} min ({metrics['epochs']:.1f} epochs, "
          f"{metrics['q_s']:,.0f} q/s)")
    print(f"{'=' * 78}")

    test = Evaluator(cfg, corpus[2], schema).run(model)
    print(f"📊 TEST         {fmt_metrics(test)}")
    print_answers(ask(model, cfg, schema,
                      ["cancel my flight to denver",
                       "what is the capital of peru"]))


if __name__ == "__main__":
    main()
