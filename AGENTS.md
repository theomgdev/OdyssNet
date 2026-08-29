# AGENTS.md

Instructions for any assistant working in this repository, human or otherwise.
Read it before you touch anything. The general half comes from
[Keel](https://github.com/theomgdev/keel); fix a general rule there first and
bring it back, so the two do not drift. Everything under Project specifics is
OdyssNet's own.

## Rule one: maximise the value to garbage ratio

Everything else here follows from this. Every change carries some value and some
garbage, and the job is to push that ratio up — not to produce more, faster.

The measurable form: add up everything you write for a change — commit message,
pull request body, comments left in the source, any markdown you touch — and it
has to come out shorter than the code that change contains. Two lines of code do
not get twenty lines of explanation. When the writing is longer than the thing
it describes, cut the writing, not the code.

Read the same ratio along the time axis and it becomes value over time. Garbage
is not only what lands in the diff — it is also the hour spent re-deriving a
figure already measured, or the third pass over a paragraph nobody will read.
Time spent is denominator too, so a change that arrives clean and a day late has
still lost.

## What garbage means

Three properties describe it better than any list of symptoms. Slop has
*superficial competence*: consistent naming, tests that exist, documentation
that is present, a clean diff, and it is still wrong or pointless underneath. It
has *asymmetry effort*: it takes vastly less effort to generate than it would
have without AI, while the effort to review it has not moved, so the cost lands
on whoever reads it. And it is *mass producible*, which is why
maintainers drown rather than merely disagree — an agent can open six pull
requests in a day and nobody can review six.

Those abstractions have concrete forms, and they are what get work rejected.

In code, the dangerous case is not the hallucinated API call or the out-of-scope
variable, because CI catches those cheaply. It is code that compiles, passes
every test, and is quietly wrong. Next to it sit the abstraction layer built for
a problem that needed ten lines, the duplicated block that should have called
something that already exists, the unused helper left behind, and the invented
naming convention. Measurement backs this up: across hundreds of millions of
lines, duplicated blocks have grown several times over, refactoring has
collapsed, and AI-heavy code churns far more than the code around it.

In tests, the failure is quieter and it is the one that stings a year later.
Generated tests love to pass. They assert too little, they mock away the very
logic they were meant to exercise, and they check that the implementation is
shaped the way it currently happens to be shaped rather than that the behaviour
is right. A test that only notices when you delete the code is not a test; it is
a tripwire around today's implementation, and it will block the refactor you
needed while catching none of the bugs you had.

History belongs in the commit message and the changelog, nowhere else. Code,
README, contributing guides and every other document describe the project as it
is now — not what used to be there, not what you measured on the way, not which
alternative you tried and rejected. The words that mean you are about to leak
your working process into the artifact are "used to", "previously", "it turned
out", "measured on", and "no longer". Design rationale for a constant is
legitimate and welcome; state the reason, not the story that produced it. Say
that a hard edge would make the cap discontinuous, not that both boundaries were
hard until you found otherwise.

In comments, the loudest complaint is that there are simply too many, and that
they answer "what" when the line below already says what. A comment restating
its own line is garbage by definition: it costs a reader time and returns
nothing. Comment the why, the constraint, the thing that would surprise
someone. Heavy commenting usually signals that the writer did not understand the
code, which is exactly the impression you do not want to leave.

In commit messages, the two failures are the bloated register — "In this commit,
improvements were made to the authentication module" — and restating the diff
instead of explaining why it exists. The reader can see what changed. They
cannot see what was wrong.

In pull requests, the complaint maintainers repeat most is that the person who
opened it cannot explain it when asked. Behind that come descriptions padded
with verbosity that says nothing, invented details stated as fact, and claims of
testing that never happened.

## What follows

Keep the change surgical. Every changed line should trace back to what was
asked. Do not improve adjacent code, do not reformat what you scrolled past, do
not refactor what is not broken, and match the surrounding style even where you
would have done it differently. Clean up the imports and helpers your own change
orphaned, and leave pre-existing dead code alone — mention it instead. A diff
full of unrelated tidying is the fastest way to make a reviewer stop reading.

Write the least code that solves the problem. No speculative abstraction, no
error handling for situations that cannot occur. If two hundred lines could have been fifty, it should have been fifty.

Research switches settle a question on a branch and then come out. Adding a flag
to A/B two implementations is good practice while it is measuring; the moment it
has answered, delete the losing path and make the winner the only one. What is
left otherwise is API surface, documentation, tests, and a way to be configured
wrong. A parameter earns its place when an expert would genuinely flip it in
production, not because two implementations happen to exist. Efficiency is never
one of those: everyone wants it, so it is not a preference to expose.

Say what you assumed, and stop when you do not know. If a request has two
readings, name both instead of silently picking one. If something is genuinely
unclear, say what is unclear rather than guessing well. Guessing quietly is how
a change ends up looking finished and being wrong.

Do not claim a result you did not observe. If you did not run the test, do not
say it passed. If you ran it, say how many assertions passed rather than that
"all tests pass". Numbers beat adjectives, and a file and line number beats a
description of where something lives.

Passing tests are not proof of correctness. They prove nothing you already
thought of is broken. Read the change again for the case nobody wrote a test
for.

Assume the happy path is right and go looking at the edges, because that is
where generated code fails. Worth checking specifically: a brand new dependency
pulled in for something trivial, a heavyweight library added to get one
function, errors caught generically or swallowed in silence, names like `data`
and `result` that say nothing, values hardcoded where configuration was meant,
and user input interpolated straight into a query string. None of those break
the build.

Do not over-verify either. The bar is whether a check would change what the
documentation says or what the reader does next. Publishing a predicted number
as though it were measured clears that bar and must be fixed; re-deriving a
figure already measured, or restating a caveat three ways, does not. Paranoia
reads as paranoia, not rigour, and it buries the findings that matter under
qualifications.

Be able to defend it without the model. If you could not answer a reviewer's
question about why a line is there, it is not ready, whoever typed it. This is
the single rule that separates useful assistance from slop.

Say that you had help, and own the result anyway. Most projects that allow
assisted contributions ask for both: around half require the assistance to be
disclosed, and about three quarters require a human in the loop, which is the
same demand from the other side. A line in the pull request, or an `Assisted-by:`
trailer on the commit, costs nothing and tells a reviewer where to look harder.
The trailer is disclosure, not credit: it records that a model helped, while the
author and the accountable party stay human. It does not transfer responsibility
either — the output is yours the moment you open the pull request. Where a
project uses the Developer Certificate of Origin the line is harder still: an
agent must never add a `Signed-off-by`, because only a person can certify it.

A model may help you review, but it cannot be the thing that approves a change;
an automated review comment is not a second pair of eyes, it is the same pair.
If a human cannot read every line, use more than one model rather than none —
one looking for mistakes, one watching this ratio, and something in an advisory
seat by default. Treat what they return as a source to filter, not a checklist
to execute: keep the points that survive contact with the evidence, say plainly
which ones you dropped and why. And keep the limit in view. Independent evidence
is what turns a guess into a finding; a second model agreeing with the first is
still a guess, just a more confident one.

Finish one thing before starting the next. Volume is why this became a crisis:
projects have closed bug bounties, gone zero-tolerance, started auto-closing
outside pull requests, and in at least one case shut down entirely, all over
being flooded. Do not flood anyone.

Write like a person, not like a report. Open with a short plain sentence saying
what the change is and why, then keep going in ordinary sentences: what was
wrong, what you did, and anything a reviewer would trip over. No blog structure,
no bulleted lists where a paragraph works, no headline subheadings inside a
commit message, no tables for three items. If a reader has to skim past
formatting to reach the content, the formatting lost.

Do not scatter per-tool instruction files across the repository. This file is
the contract and every assistant reads it. Anything specific to your own tool —
`CLAUDE.md`, `CURSOR.md`, `GEMINI.md`, `.cursor/`, `.claude/` and the rest —
stays local and gitignored. Those files are noise to everyone not using that
tool, they drift out of sync with each other, and a repository collecting one
per vendor has already lost the thing they were meant to protect.

## Project specifics

### Commands

```bash
pip install -e ".[dev]"                 # install (torch>=2.4, Python 3.10+)
python -m pytest tests/ -q              # full test suite (~2s, CPU-only)
python -m pytest tests/training/test_chaos_optimizer.py -q          # one file
python -m pytest tests/training/test_trainer.py::TestTrainerInit -q # one class

# Examples (GPU-friendly but run on CPU too). Disable the blocking
# matplotlib window in automated runs:
ODYSSNET_DISABLE_PLOT=1 python -u examples/convergence_gates.py
python examples/test_all.py             # runs every example as a subprocess (slow)
```

Some example scripts reconfigure stdout to UTF-8 (emoji output on Windows code pages); use `python -u` when piping their output, otherwise it stays buffered until exit.

### Architecture

OdyssNet is a research library for a non-layered recurrent architecture: a single fully connected NxN weight matrix `W` (the "chaos core") through which signals echo for `thinking_steps` timesteps. Depth is temporal, not spatial — a 794-neuron zero-hidden network solves MNIST because the same parameters are reused every step.

**`odyssnet/core/network.py` — `OdyssNet`** (single class, one big `forward`):
- Per step: `h @ W + B` → optional gates → per-neuron `memory_feedback` (self-connections live here, NOT in W) → input injection → activation → dropout → RMSNorm. RMSNorm every step is what makes the chaotic routing trainable.
- `pulse_mode=True` injects input only at t=0; `False` re-injects continuously.
- `vocab_size` enables projection mode (`embed`/`proj` in, `output_decoder` out); otherwise inputs/outputs map directly onto neuron indices `input_ids`/`output_ids`.
- Hebbian plasticity (`hebb_type`: temporal/spatial/both; `hebb_res`: global/neuron/synapse): a per-example `(B, N, N)` trace accumulates during forward, is RMS-normalized through `hebb_norm` and added to `W`/`memory_feedback`; the persistent buffer is the `(N, N)` batch mean, which is what checkpoints, neurogenesis and transplants see — and what every row reads back at the start of the next call, so a batch run as a colony has one shared memory (`convergence_hive_mind`). Four properties of the 3.1.0 repair, all of them load-bearing — do not undo one while refactoring:
  - The **gain** (`hebb_norm.weight`) is zero-initialized and construction draws no RNG, so `hebb_type` is an ablation with one variable: a plastic model and a plain one at the same seed share a core. It lands in ChaosGrad's `modulation` family by falling through the classifier; weight decay there would pull the gain back to zero and switch plasticity off silently.
  - Factor/decay parameters are raw logits mapped through sigmoid — their zero is not neutral, so they must never receive weight decay. The factor decides *which* synapses are plastic; the gain decides *how much*. Applying the factor at both ends (accumulation and application) is the old `sigmoid(logit)**2` bug.
  - The correlation is **not** detached from the state; only the trace's own history is (`local_*.detach()`). One stop-gradient costs memory, the other costs capability.
  - Temporal and spatial run stacked on a leading path axis so `'both'` is one set of kernels. `_offdiag` carries the 1/N scale and the diagonal mask together.
  - The trace is an activation: `steps x B x N²` per path, and nothing else in the graph is close. Reading it, taking the step and writing it back are **one checkpointed region** (`_step_and_learn`), so `gradient_checkpointing=True` keeps only the trace that crosses the step boundary (6.3 units of `B x N²` per step down to 1.2). Split that region and the flag stops reaching the term it exists for.
  - The repair made plasticity work; it did not make it a win. Controlled A/Bs on the record task: ahead with attention off (1 seed), *behind* with 4 attention heads held fixed (2 seeds, +0.02 loss and -0.5pp for +50 params), slightly behind on single-injection classification. Attention seems to fill the same role better where both apply. Ablate `hebb_type` before claiming anything — that is what the zero-init gain is for.
- The step is **kernel-launch-bound at every size measured**, so `torch.compile` is the largest speedup available (plasticity: +132% eager, +26% compiled) and fewer-kernels beats better-arithmetic. `TemporalAttention.attend`/`write` carry `@torch._dynamo.disable`: Inductor miscompiles their `autocast(enabled=False)` region (reports float for a buffer it emits as half; `o_proj` dies on `Half != float`, and a source-level `.float()` is folded away). Removing the decorator breaks every compiled run with attention on. Letting attention run in ambient precision instead was measured and rejected — no faster, and ~0.017 of loss worse.
- **Invariant: `W.diagonal() == 0`**, enforced three ways (grad hook on the model, inside `ChaosGrad.step`, and defensively after the step in the trainer). Preserve all three when refactoring.

**`odyssnet/core/attention.py` — `TemporalAttention`** (optional, `attn_heads=None` builds nothing):
- No layers to stack attention between, so it attends along *time*: each step issues one query over a cache of earlier states, added to the same pre-activation signal the recurrence feeds. Query length is always 1 — a decode phase, never a prefill — so `F.scaled_dot_product_attention` solves a problem this shape does not have, and segments are joined at the *scores* under one softmax instead of copying K/V.
- Two invariants: `o_proj` is zero-initialized **and** the module is built after `_init_weights()`, so an attention model and a plain one at the same seed share `W`; and the branch is divided by `sqrt(heads*head_dim)` (`out_scale`) — without it a 4x64 branch collapsed associative recall to chance while 1x16 matched baseline.
- Everything cached is strictly past and softmax over keys is permutation-invariant, so no causal mask is ever needed and a wrapped ring buffer never has to be reordered. RoPE goes in at *write* time, computed in float64 on demand.
- Two cache representations, tested to agree: a preallocated ring under `no_grad`, a frozen carry plus a differentiable list of this call's writes under autograd.

**`odyssnet/training/chaos_optimizer.py` — `ChaosGrad`** (the default optimizer, zero-config):
- Adam per-element preconditioning + Prodigy-class D-adaptation: `lr=None` estimates the step scale online; `lr=float` is fixed-rate mode, byte-for-byte AdamW-equivalent (used where reproducibility matters).
- `ChaosGrad.from_model(model)` classifies parameters into families (chaos_core / memory_feedback / projections / attention / plasticity / modulation) carrying policy: weight decay only on chaos_core, projections and attention, `zero_diag` on the core. `attn` is checked before `projections` (q/k/v/o_proj would match the substring); norm gains inside attention and `hebb_norm` go to `modulation`, where nothing decays them.
- Two stability systems, both validated against real failure modes: the **traction limit** (`trust_ratio`, cap anchored to *initial* weight RMS — a live-RMS cap is self-defeating because runaway steps inflate their own cap) and the **loss-spike brake** (`report_loss`, fed automatically by the trainer; shrinks the estimate on 3σ/+20% spikes so monotone d growth cannot diverge sharpening temporal tasks).
- Estimator state `s`/`p0` is deliberately stored in the parameter's own shape (not flattened) so neurogenesis can pad it with the same top-left-corner rule as the parameters. Keep it that way.
- When changing optimizer behavior, re-run the quick probes: XOR seeds 42/123 must solve zero-config (`examples/convergence_gates.py` pattern, 3 neurons), the delayed adder must not diverge by epoch ~400, and a 3k-subset/6-epoch MNIST run should stay ≥ ~90.5%.

**`odyssnet/training/trainer.py` — `OdyssNetTrainer`**: AMP autocast + GradScaler, grad clipping at 1.0, gradient accumulation, optional gradient persistence ("ghost gradients") and synaptic noise, anomaly hooks (spike/plateau/increase), and the `report_loss` wiring into ChaosGrad. `train_batch` is the core primitive; `fit` is the convenience loop.

**`odyssnet/utils/neurogenesis.py`**: grows the network in place (pads W/B/buffers/Hebbian state top-left) and migrates the optimizer — ChaosGrad gets rebuilt via `from_model` with its per-family `d` estimates carried over; per-param state tensors are padded by `transfer_state` (which distinguishes param-shaped tensors from 0-dim metadata — a shape-comparison bug here once silently corrupted post-expansion training).

**`odyssnet/utils/odyssstore.py`**: checkpointing plus `transplant_weights` (loads a smaller trained model into the top-left corner of a bigger one — the skill-transfer experiments depend on this).

### Conventions (from CONTRIBUTING.md)

- Examples use `OdyssNetTrainer`, never hand-written `loss.backward()/optimizer.step()` loops, and must call `set_seed(...)` at the start of `main()` — 42 by default, a different seed only when a measurement chose it and a comment says so (the three MNIST record/tiny examples use 123).
- `convergence_mnist_record`, `convergence_mnist_tiny` and `convergence_mnist_reverse_record` run **attention, not plasticity**, on seed 123 — a controlled A/B put attention ahead on the record task. Head geometry is sized to the core (`attn_heads=4` at 10-12 neurons, `attn_heads=1, attn_head_dim=4` at 59), because attention's projections scale with the neuron count. `convergence_skill_transfer` runs both.
- `convergence_hive_mind` reads the Hebbian buffer as a memory shared between batch rows — the batch is a colony, the buffer is what the bodies share. Four things there are load-bearing: `hebb_type='temporal'` (an edge is directed and so is `h_prev ⊗ h_t`; `'spatial'` alone stays at chance and `'both'` makes the deepest hop a coin flip between seeds); the study pass runs under `torch.no_grad()`, so only the reading is trained — say so if that ever changes; it is **two steps** (a cold start makes the first state a near-one-hot, so the correlation written next step is row-addressed — a six-step write keeps single-edge recall and loses the composition); and every eval call passes `current_state` explicitly, because a batch-size change sends `forward` through `reset_state`, which zeroes the memory.
- Examples are zero-config (`OdyssNetTrainer(model, device=...)`) unless a documented precision record depends on a tuned rate (record/reverse-record keep explicit `lr` via fixed-rate mode — say why in a comment). `experiment_llm.py` defaults to zero-config as of 2.6.2; its reference run was measured under `lr=None`, and `--lr <float>` selects fixed-rate mode when you want it.
- `experiment_llm.py`'s tokenizer default (`--tokenizer bpe --vocab-size 2048`) is the configuration its reference numbers are defined against — don't "modernize" it to a tiktoken vocabulary. Measured on TinyStories, a 4096-token in-domain BPE matches gpt2's 50k vocab (3.908 vs 3.914 bytes/token) at 1/12 the embedding cost, and `cl100k_base` would make 98% of the model a lookup table. `--tokenizer <tiktoken encoding>` is available for larger cores and for skipping the training pass.
- Version lives in four places that must stay in sync: `odyssnet/__init__.py`, `pyproject.toml`, `CITATION.cff`, and the `CHANGELOG.md` entry. Docs (`docs/LIBRARY.md`, `README.md`, `README_TR.md`, `CONTRIBUTING.md`) are kept aligned with code in the same change.
- README performance tables (98.92% MNIST etc.) are advertised metrics tied to specific example configs — don't change those configs without flagging that the numbers need re-validation.
- **Documentation discipline.** `CHANGELOG.md` is the only file that carries history. Code comments, docstrings, `README*.md`, `LIBRARY.md` and `CONTRIBUTING.md` describe the project as it *is* — what the code does and which invariants must not be broken, in as few words as that takes. What used to be wrong, what was measured on the way, which alternative was tried and rejected: CHANGELOG and the commit message, nowhere else. Design rationale for a constant is legitimate; state the reason, not the story that produced it. No comment ever sits between a call's arguments. Benchmark tables live in `docs/LIBRARY.md`, not in docstrings.
- `tmp/` is scratch space, not part of the library.

### CI

GitHub Actions runs `python -m pytest tests/ -v` on Python 3.10–3.12 (CPU). Tests must stay CPU-safe and fast; anything CUDA-dependent needs a skip guard.

## Is this working?

You will know it is when diffs contain only what was asked for, when reviews
stop turning into rewrites, and when questions arrive before the work rather
than after the mistake.
