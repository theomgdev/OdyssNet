# OdyssNet Library Documentation

OdyssNet is a PyTorch-based library that implements **Zero-Hidden Layer** neural networks using **Temporal Depth**. By treating the neural network as a dynamic system that evolves over time, OdyssNet achieves deep learning capabilities without stacking spatial layers.

## Core Modules

The library is organized into three primary modules:
1.  **`odyssnet.core.network`**: The recurrent core architecture and update dynamics.
2.  **`odyssnet.training.trainer`**: Optimization engine built around **ChaosGrad** with bio-inspired regularization.
3.  **`odyssnet.utils`**: Data utilities, model persistence (`odyssstore`), and dynamic expansion (`neurogenesis`).

---

## OdyssNet Model (`odyssnet.core.network`)

The `OdyssNet` class defines the structure and dynamics of the network. It is a single layer where every neuron is connected to every other neuron (including itself).

### Initialization

```python
from odyssnet import OdyssNet

model = OdyssNet(
    num_neurons=10, 
    input_ids=[0, 1], 
    output_ids=[9], 
    pulse_mode=True, 
    dropout_rate=0.0, 
    device='cuda',
    weight_init=['quiet', 'resonant', 'quiet', 'zero', 'quiet'],
    activation=['none', 'tanh', 'tanh', 'none'],
    gate=None,           # Default resolves to ['none', 'none', 'identity']
    vocab_size=None,     # Optional: Decouples input/output size from neurons
    vocab_mode='hybrid', # 'hybrid', 'discrete', or 'continuous'
    hebb_type=None,      # Toggle: None, 'temporal', 'spatial', or 'both'
    hebb_res='neuron',   # Plasticity resolution: 'global' or 'neuron'
    attn_heads=None,     # Temporal attention: None/0 builds nothing at all
    attn_kv_heads=1,     # Shared KV heads (1 = multi-query)
    attn_head_dim=None,  # None derives it from num_neurons / attn_heads
    attn_window=256,     # Cache entries a query can see
    attn_write='token',  # 'token' (one entry per input token) or 'step'
    attn_read='step',    # query every step, or only when a token arrives
    attn_rope=True,      # Rotary positions, applied when an entry is written
    attn_qk_norm=True,   # RMSNorm on queries and keys
    attn_dropout=0.0,    # Dropout on the attention weights
    debug=False,         # NaN/Inf diagnosis — raises RuntimeError at the first offending operation
)
```

**Parameters:**
*   `num_neurons` (int): Total number of neurons in the single layer (No hidden layers).
*   `input_ids` (list[int]): Indices of neurons that receive external input.
*   `output_ids` (list[int]): Indices of neurons whose state is read as output.
*   `pulse_mode` (bool): 
    *   `True`: Input is applied only at $t=0$ (Impulse).
    *   `False`: Input is applied continuously at every step (Stream).
*   `dropout_rate` (float): Probability of synaptic failure during training (Biological simulation).
*   `device` (str): 'cpu' or 'cuda'.
*   `weight_init` (str or list[str]): Weight initialization strategy. Default is `['quiet', 'resonant', 'quiet', 'zero', 'quiet']` for [Encoder/Decoder, Core, Memory, Gates, Attention]. Shorter lists are filled from the defaults, so a four-entry list written before 3.0 still means what it meant. Single string values are expanded intelligently. The attention entry covers the query/key/value projections only — the output projection is always zero (see **Temporal Attention**).
    *   `'resonant'` **(Default for Core)**: Edge-of-Chaos initialization with spectral radius ρ(W) = 1.0. Uses bipolar Rademacher (±1) skeleton + small Gaussian noise (std=0.02) + spectral normalization. Ensures signals neither explode nor vanish while maintaining excitatory/inhibitory balance.
    *   `'orthogonal'`: Orthogonal matrix initialization. Excellent stability for large networks.
    *   `'xavier_uniform'` / `'xavier_normal'`: Xavier-scaled initialization. Good for small logic networks.
    *   `'kaiming_uniform'` / `'kaiming_normal'`: Kaiming-scaled initialization. ReLU-oriented.
    *   `'quiet'`: Normal(0, 0.02). Small random initialization.
    *   `'micro_quiet'`: Normal(0, 1e-6). Near-zero initialization.
    *   `'sparse'`: 90% sparse with std=0.02.
    *   `'zero'`, `'one'`, `'classic'`: Special initialization cases.
*   `activation` (str or list[str]): Activation function. Default is `['none', 'tanh', 'tanh', 'none']` for [encoder_decoder, core, memory, gate_hint]. The 4th entry is reserved for config symmetry and doesn't affect gate behavior. Supported activations: `'tanh'`, `'relu'`, `'leaky_relu'`, `'sigmoid'`, `'gelu'`, `'gelu_tanh'`, `'silu'`, `'none'`, `'identity'`. Single string applies to core path; list format allows per-component control with 1-4 entries (missing entries filled from defaults).
*   `vocab_size` (int or list/tuple, optional): Size of the input/output vocabulary. 
    *   **Symmetric**: `vocab_size=50257` (GPT-2 style).
    *   **Asymmetric**: `vocab_size=[v_in, v_out]` (e.g., `[784, 10]` for MNIST to map 784 pixels to 10 classes).
    *   **Disable**: Use `-1` to disable one side (e.g., `[-1, 1000]` for direct neuron input but decoded output).
*   `vocab_mode` (str): Controls which input encoding layers are initialized (default: `'hybrid'`).
    *   `'hybrid'`: Initializes both Embedding (for integer/token inputs) and Linear Projection (for float inputs). Use when input type varies.
    *   `'discrete'`: Initializes only Embedding layer. Use for token-only inputs (e.g., NLP tasks). Saves VRAM.
    *   `'continuous'`: Initializes only Linear Projection. Use for float-only inputs (e.g., vision, audio). Saves VRAM.
*   `tie_embeddings` (bool): 
    *   If `True`, ties the input embedding weights to the output decoder weights, saving significant VRAM and parameter count (Symmetric `vocab_size` only). Default is `False`.
*   `hebb_type` (str or None): Controls the active mechanism for **Heterogeneous Synaptic Plasticity**. Default is `None` (plasticity disabled).
    *   `'temporal'`: STDP-style learning; correlates current state $h_t$ with previous state $h_{t-1}$.
    *   `'spatial'`: Co-activation learning (classic Hebbian); correlates current state $h_t$ with itself $h_t$ (neurons firing simultaneously).
    *   `'both'`: Combines both temporal and spatial mechanisms.
*   `hebb_res` (str): Controls the structural resolution of plasticity. Default is `'neuron'`.

    | `hebb_res` | Parameter Shape | Extra Params per Path | Mechanics |
    |---|---|---|---|
    | `"global"` | scalar `()` | +2 | Uniform plasticity — the whole network is equally plastic. |
    | `"neuron"` | vector `(N,)` | +2N | Per-neuron plasticity — each neuron learns its own adaptation rate. |

    *   For each active path (`t_` for temporal, `s_` for spatial), two learnable logit parameters are created according to the resolution:
        *   `t_hebb_factor` / `s_hebb_factor` (raw logit → `sigmoid` → learning rate ≈ 0.047 initially)
        *   `t_hebb_decay` / `s_hebb_decay` (raw logit → `sigmoid` → retention ≈ 0.90 initially)
    *   During each forward pass the model accumulates correlations (temporal $h_t \otimes h_{t-1}$ and/or spatial $h_t \otimes h_t$) and applies them to the effective weights.
    *   The Hebbian states are persisted across forward calls via registered buffers (`t_hebb_state_W`, `s_hebb_state_W`, etc.) and are cleared by `reset_state()`.
    *   Both factors and decays are fully differentiable — gradients flow into them via the recurrent computation so the network **learns how to learn** online.

#### How the plastic contribution reaches the recurrence

*   **Normalized, zero-initialized gain (`hebb_norm`, +N parameters).** The trace is RMS-normalized and scaled by a learnable gain before it is added to `W`. The factor decides *which* synapses are plastic, the gain decides *how much*. The gain starts at zero, so `hebb_type` is a one-variable switch: a plastic model and a plain one at the same seed share a core and agree exactly until training moves the gain.
*   **Differentiable correlation.** The correlation carries gradient; only the trace's own history is truncated (`local.detach()` in the update).
*   **Novelty gate.** A write is damped by `1 / (1 + rms(W_eff[j,:]))`, the presynaptic row it lands on — co-activation into an already-strong row says the weights fired the neuron, not that the pattern is new. Parameter-free and detached, so it opens no second-order path through `W`.
*   **Per-example trace.** The live trace is `(B, N, N)`; every example writes its own associations, so a batched forward computes the same function as the same examples run singly. The persistent buffer stays `(N, N)` and holds the batch mean, leaving checkpoints, neurogenesis padding and weight transplants untouched.
*   **Pooled across the batch, between calls.** The buffer is written at the end of a forward pass and read back at the start of the next one, expanded to every row. Within a call each example reads only what it wrote; across a call boundary all of them read the mean of what they all wrote.

**The trace is never assembled.** Every write is an outer product and the recurrence only ever asks the trace for `h @ L`, so the trace is kept as the `(B, N)` writes it is made of and contracted on demand — both the product and the row norms its RMSNorm needs. The persistent buffer is materialized once per call, already averaged over the batch. Memory is therefore `steps x B x N`, not `steps x B x N²`: at 1024 neurons, batch 8 and 96 echo steps the whole plastic path holds 109 MB.

Three properties keep it affordable. The history lives in a preallocated buffer and every contraction runs over all of it behind a mask, so every step has the same shapes and `torch.compile` can trace them. The read is recomputed in the backward pass rather than kept, which is sound because the history is immutable once written. And the novelty gate, being detached, is carried as running sums rather than read back out of the trace.

**Cost.** The step is bound by kernel launches rather than arithmetic, and the trace issues more of them than a plain step does. `torch.compile` closes most of that — 7.5x on the plastic path at 64 neurons over 16 steps, 4.4x at 512 over 96 — but the loop unrolls, so warmup grows with the step count and is minutes rather than seconds at 96. Plasticity remains the one term that scales with the batch.

**When to use it.** Plasticity earns its place where step *T* extends what step *T-1* built, and acts as overfit noise where each step handles an independent chunk. Temporal attention fills the same role on sequential classification and measured better there, so the two are alternatives more than complements. Ablate rather than assume — that is what the zero-initialized gain is for.

#### One memory across many bodies

Because the buffer pools the batch and is broadcast back, a batch can be read as a *colony*: independent bodies with private inputs, private outputs, their own hidden state and their own attention cache, sharing one core and one memory on it. Nothing else crosses between rows — the recurrence, the norm and the attention cache are per-row — so anything a body knows that its own inputs never contained came through the trace.

`examples/advanced/convergence_hive_mind.py` measures that. Eight bodies are each shown one edge of an 8-symbol ring, drawn fresh every episode; every hidden state and attention cache is then wiped, and each body is asked to walk from a query symbol, one edge per echo step. Only the reading is trained — the study pass runs under `torch.no_grad()`, so the write is the plasticity rule itself.

| 320 queries per column | hop 0 | hop 1 | hop 2 | hop 3 |
|---|---|---|---|---|
| together, one memory | 1.000 | **1.000** | **1.000** | **1.000** |
| apart, one bee alone | 1.000 | 0.297 | 0.134 | 0.094 |
| together, memory blank | 1.000 | 0.147 | 0.141 | 0.112 |

Chance is 0.125. Hop 1 is an edge another body observed; hops 2 and 3 compose edges held by different bodies. Running the colony as one batch and running each body separately then averaging the memories afterwards agree to 3.0e-08 on a memory of scale 0.317, so the batch axis is only the vectorized form of pooling that could equally be done across machines.

**Which mechanism carries it**, same protocol, 1,000 optimizer steps on CPU:

| | hop 1 | hop 2 | hop 3 |
|---|---|---|---|
| `hebb_type='temporal'` (shipped), seeds 42, 7 and 123 | 1.000 | 1.000 | 1.000 |
| `hebb_type='temporal'`, attention off | 1.000 | 1.000 | 1.000 |
| `hebb_type='both'`, seed 42 / seed 7 | 1.000 | 1.000 | 1.000 / 0.200 |
| `hebb_type='spatial'` | 0.169 | 0.134 | 0.122 |
| `hebb_type=None` | 0.125 | — | — |

An edge is directed and so is `h_prev ⊗ h_t`, which is why the temporal path carries the colony; `h_t ⊗ h_t` binds a state to itself, stores no direction, and alone never leaves chance. Carried alongside the temporal path it does not add to it — with `'both'` the deepest hop is a coin flip between runs (1.000 and 0.200 at two seeds) where temporal alone repeats 1.000 at three. Attention is not a channel on this task at all: the cache is per body and is wiped with the state before the query, and the colony measures the same with it off. `hebb_type=None` is the leak test — with no plastic trace there is no channel, and nothing above chance may appear.

**How deep it goes.** Trained to seven hops instead of three, the same colony holds 1.000 through hop 4, then 0.884, 0.775 and 0.228 — one echo step per edge, so this is the depth of composition the shared memory supports, not a property of the ring.

Two protocol properties transfer to any use of the buffer this way:

*   **Keep the write short.** A cold start makes the first state a near-one-hot of the injected pattern, so the temporal correlation written on the next step lands on that pattern's row. Longer study passes add correlations between dense states, which are common to every body and bury the addressed row. Two and three steps both hold on the ring task; a six-step write leaves single-edge recall standing (0.994) and takes the composition with it (hop 2: 0.503, hop 3: 0.194).
*   **Pass `current_state` explicitly when reading.** `forward` calls `reset_state` when the batch size differs from the stored state, and `reset_state` zeroes the Hebbian buffers. An evaluation pass at a different batch size — one body instead of eight — silently wipes the memory it was about to read unless the state is handed in.

*   `gate` (None, str, or list[str]): Optional parametric gating mechanism. Default is `None`, which resolves to `['none', 'none', 'identity']`.
    *   `None`: Default configuration with memory identity gate enabled, others disabled.
    *   `str` (e.g., `'sigmoid'`): Applies the same gate activation to all three branches `[encoder_decoder, core, memory]`.
    *   `list[str]`: Specify individual gate activations for up to 3 branches in `[encoder_decoder, core, memory]` order. Missing entries use defaults.
    *   `'none'`: Completely disables the gate branch (no learnable parameters).
    *   `'identity'`: Enables identity gating with learnable parameters (starts at identity function but can adapt).
    *   Gate parameters are initialized using the 4th entry in `weight_init` (default: `'zero'`).
*   `attn_heads` (int or None): Number of query heads for **Temporal Attention**. Default `None` — no module, no parameters, no per-step cost. Any positive value switches attention on; see the **Temporal Attention** section for what it does and what it costs.
*   `attn_kv_heads` (int): Key/value heads, shared across query heads (grouped-query attention). Must divide `attn_heads`. Default `1` (multi-query), because the KV cache — not the projections — is what decides whether a batch fits. Set it equal to `attn_heads` for classic multi-head.
*   `attn_head_dim` (int or None): Width per head. Default `None` derives `min(64, num_neurons // attn_heads)`, rounded down to even. Pin it explicitly if you plan to `transplant_weights` between cores of different sizes, or the attention geometry moves with the neuron count.
*   `attn_window` (int): How many cache entries a query can attend. Oldest evicted first. Default `256`.
*   `attn_write` (str): `'token'` (default) writes one cache entry per input token — the state at the end of that token's thinking steps. `'step'` writes every thinking step, letting the model attend to its own intermediate steps at `(think_gap+1)x` the cache. Identical when there is one step per token.
*   `attn_read` (str): `'step'` (default) issues a query on every thinking step. `'token'` queries only on the step a token arrives, leaving the echo steps to run on state alone exactly as `pulse_mode` does with the input itself. The query — not the cache entry — is what extra thinking steps multiply, so this is the cheaper of the two whenever there are several steps per token (~1.3x throughput at `think_gap=1`), and the two are identical at one step per token.
*   `attn_rope` (bool): Rotary position embedding, applied to each key at write time against its absolute position. Default `True`. With it off, attention is a pure content lookup with no sense of *when*.
*   `attn_qk_norm` (bool): RMSNorm on queries and keys before the dot product. Default `True` — the core is chaotic and its state norm is only bounded at the end of a step.
*   `attn_dropout` (float): Dropout on the attention weights during training. Default `0.0`.
*   `debug` (bool): Enables NaN/Inf diagnosis mode. Default is `False`. When `True`, every critical operation in the forward pass (linear recurrence, memory feedback, activation, StepNorm, Hebbian correlation and accumulation) is checked after execution; the first non-finite value raises `RuntimeError` with the operation name and step index. Also automatically calls `torch.autograd.set_detect_anomaly(True)` so backward-pass NaN is caught with a full stack trace. Disable after the root cause is found — overhead is zero when `False`.

### Vocabulary Decoupling

When `vocab_size` is typically much larger than `num_neurons` (e.g., 50k vocab vs 1024 neurons), OdyssNet uses decoupled layers. This can be configured as symmetric (same size for in/out) or asymmetric.

1.  **Encoder (Input)**: Maps `v_in` -> `len(input_ids)` (Neurons).
    *   Integers (Tokens) use `nn.Embedding`.
    *   Floats (Vectors) use `nn.Linear` (Projection).
    *   *Disabled if `v_in == -1`.*
2.  **Decoder (Output)**: Maps `len(output_ids)` (Neurons) -> `v_out`.
    *   Uses `nn.Linear` (Decoding).
    *   *Disabled if `v_out == -1`.*

**Benefit:** This allows the "Thinking Core" (Neurons) to remain small and efficient while handling complex input formats or large output spaces without manual slicing.

```python
# Asymmetric Example: MNIST (784 pixels -> 10 classes)
model = OdyssNet(
    num_neurons=10,
    input_ids=range(10),
    output_ids=range(10),
    vocab_size=[784, 10], # Input 784, Output 10
    vocab_mode='continuous'
)
# No need for slice_output: model(x) returns (Batch, Steps, 10)
```

---

## Input Modalities and Data Handling

OdyssNet processes data through three distinct modalities. Choosing the right one is critical for performance and VRAM efficiency.

### 1. Pulse Mode (Impulse Computing)
**Use case**: Static data like images (MNIST) or single-shot logic (XOR).
*   **Behavior**: Set `pulse_mode=True`. Input is injected at $t=0$ only.
*   **Thinking**: The model continues computation for the specified number of `steps` without further input.
*   **VRAM Efficiency**: Optimal. Only (Batch, Neurons) is stored.

```python
# Image Classification (784 pixels -> 100 steps thinking)
model = OdyssNet(..., pulse_mode=True)
output = model(image_tensor, steps=100)
```

### 2. Continuous Mode (Static Control)
**Use case**: Control systems, VCO (Sine Wave), or real-time sensor monitoring.
*   **Behavior**: Set `pulse_mode=False`. The same input is injected at every time step $t$.
*   **Thinking**: The model state is constantly influenced by the static input.
*   **VRAM Efficiency**: High. Only (Batch, Neurons) is stored.

```python
# Frequency Control for Oscillator
model = OdyssNet(..., pulse_mode=False)
output = model(freq_input, steps=30)
```

### 3. Sequential Mode (Temporal Stretching)
**Use case**: Large Language Models (LLM), Time-Series, and reasoning agents.
*   **Behavior**: Provide a sequence `(Batch, Tokens)`. If `steps` > `tokens`, OdyssNet automatically scales the temporal resolution.
*   **Mechanism**: If 100 tokens are provided with 500 `steps`, the model intersperses 4 "silent" thinking steps between each token.
*   **VRAM Efficiency**: High. Eliminates the need for manually dilated/padded input tensors.

```python
# LLM: 128 tokens with 5 thinking steps per token (Total 640 steps)
tokens = torch.randint(0, 50257, (batch, 128))
output = model(tokens, steps=640)
```

#### Comparison of Sequential Input Formats
| Input Type | Format | Modality | Recommended Use Case |
| :--- | :--- | :--- | :--- |
| **Index (ID)** | `(Batch, Steps)` (Long) | Sequential | LLMs, Tokenized text. |
| **Dense** | `(Batch, Steps, Dim)` (Float) | Sequential | Audio, Video, Vector Streams. |
| **Pulse** | `(Batch, Dim)` (Float) | Instant | Static Images, Logic Gates. |
| **Continuous**| `(Batch, Dim)` (Float) | Periodic | Oscillators, Constant Signals. |

---

### Key Methods

#### `model.get_num_params()`
Returns the **effective** parameter count of the network. It accounts for the `memory_feedback` separation by properly discounting the inactive diagonal of the `W` matrix to give you a true representation of learning capacity.

#### `model.compile()`
Optimizes the model using `torch.compile` (PyTorch 2.0+) for faster execution. Returns the compiled model.

#### `model.forward(x_input, steps=1, current_state=None, return_sequence=True)`
Runs the dynamic system.
*   `x_input`: Input tensor. Can be a single pulse or a sequence (index-based or dense).
*   `steps`: **Thinking Time**. How many times the signal reverberates in the echo chamber.
*   `current_state`: Optional. Pass a previous state to continue from.
*   `return_sequence` (bool, default `True`): Controls output allocation strategy.
    *   `True`: Collects the full output sequence and returns `all_states` of shape `(Batch, Steps, Neurons)`. Required when loss is computed over all time steps (`full_sequence=True` in the trainer).
    *   `False`: Skips building the `(Batch, Steps, Neurons)` tensor entirely and returns only the final step as `(Batch, 1, Neurons)`. Saves VRAM proportional to `thinking_steps` — use this whenever only the last output matters (e.g., classification, pulse-mode inference).
*   **Returns**: `(all_states, final_state)`
    *   `all_states`: Shape `(Batch, Steps, Neurons)` when `return_sequence=True`; shape `(Batch, 1, Neurons)` when `False`.
    *   `final_state`: Tensor of shape `(Batch, Neurons)` — the last hidden state, regardless of `return_sequence`.

> **Trainer transparency**: `OdyssNetTrainer` automatically passes `return_sequence=full_sequence` in `train_batch()` and `predict()`. You only need to set this manually when calling `model.forward()` directly.

---

## OdyssNet Trainer (`odyssnet.training.trainer`)

The `OdyssNetTrainer` handles the training loop, gradient accumulation, mixed precision (AMP), and experimental features like Ghost Gradients. **ChaosGrad** is the default optimizer — fully zero-config: it estimates the step scale online, so no learning rate is required.

### Initialization

```python
from odyssnet import OdyssNetTrainer

# Zero-config (recommended): ChaosGrad estimates the step scale online
trainer = OdyssNetTrainer(model, device='cuda')

# Fixed learning rate: ChaosGrad in fixed-rate mode (AdamW-equivalent,
# byte-for-byte reproducible loss curves)
trainer = OdyssNetTrainer(model, lr=1e-4, device='cuda')

# With optional features
trainer = OdyssNetTrainer(
    model,
    device='cuda',
    gradient_persistence=0.0,
    synaptic_noise=0.0,
    anomaly_hook=my_hook
)

# Expert overrides on the default optimizer
from odyssnet import ChaosGrad
trainer = OdyssNetTrainer(model, optimizer=ChaosGrad.from_model(model, d_coef=0.5))

# Any torch optimizer still works
import torch
trainer = OdyssNetTrainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4))
```

**Parameters:**
*   `lr` (float or None): Learning rate. Default: `None`.
    *   `None`: **ChaosGrad** estimates the step scale online — no manual tuning required. Recommended default. Loss curves vary slightly across runs because the estimate adapts to the observed landscape.
    *   float (e.g. `1e-4`): ChaosGrad runs in **fixed-rate mode** (automatic estimation disabled; AdamW-equivalent updates under the same family policy). Use for byte-for-byte reproducibility studies and benchmarking against fixed baselines.
*   `gradient_persistence` (float): **Ghost Gradients / Persistence**.
    *   `0.0`: Standard behavior (`zero_grad()` after every step).
    *   `> 0.0` (e.g., `0.1`): Keeps a percentage of the previous step's gradient. This creates a "momentum" over time, effectively simulating a larger batch size or longer temporal context. Useful for difficult convergence landscapes.
*   `synaptic_noise` (float): **Thermal Noise**.
    *   Adds Gaussian noise (std dev = `synaptic_noise`) to all weights *before* every training step.
    *   Simulates biological thermal noise and prevents overfitting (Stochastic Resonance).
    *   **Default:** `0.0` (Enable for regularization, e.g. `1e-6`, on large or overfitting-prone networks).
*   `anomaly_hook` (Callable, optional): A user-defined function `hook(anomaly_type, loss_val)` triggered automatically when training encounters anomalies. Supported `anomaly_type` values:
    *   `"spike"`: A sudden, violent surge in loss (e.g., exploded gradient).
    *   `"increase"`: Triggered *every single time* the current step's loss is strictly greater than the previous step's loss (even by 0.0001). Perfect for custom patience counters or algorithmic early stopping.
    *   `"plateau"`: The loss has stagnated and is barely moving over a window.
    *   **Usage**: Allows for smart interventions (like custom logging or early stopping when stuck).

### Key Methods

#### `trainer.fit(...)`
Runs a full training loop.

```python
history = trainer.fit(
    input_features=X, 
    target_values=Y, 
    epochs=100, 
    batch_size=32, 
    thinking_steps=10       # Temporal Depth
)
```

#### `trainer.train_batch(...)`
Runs a single custom training step. Useful for custom loops (RL, Generative, etc.).
*   `thinking_steps`: How long the model "thinks" before loss is calculated.
*   `gradient_accumulation_steps`: Simulates larger batch sizes.
*   `full_sequence` (bool): If `True`, calculates loss on the entire sequence output `(Batch, Steps, Out)` instead of just the last step. Essential for Seq2Seq tasks.
*   `mask` (Tensor, optional): A binary or weighted mask `(Batch, Steps, Out)` to ignore specific steps or outputs during loss calculation. Useful for tasks with "thinking delays" or variable-length sequences.
*   `output_transform` (Callable, optional): A function to transform the predicted outputs before loss calculation. Useful for reshaping logits (e.g., flatten for CrossEntropy) or applying custom activations.

#### `trainer.predict(input_features, thinking_steps, full_sequence=False)`
Runs inference in evaluation mode.
*   `full_sequence` (bool): If `True`, returns outputs for all time steps `(Batch, Steps, Out)`.

#### `trainer.regenerate_synapses(threshold=0.01)`
Triggers **Darwinian Regeneration**. Instead of pruning weak weights, this method **re-initializes** them.
*   **Logic**: If $|W| < threshold$, the synapse is considered "dead/useless". It is wiped and assigned a new random value using the model's original initialization strategy (e.g., Xavier/Orthogonal).
*   **Purpose**: Allows the network to escape local minima and constantly explore new pathways. Transforms "dead" capacity into "fresh" capacity.
*   **Returns**: `(revived_count, total_synapses)`

#### `trainer.get_diagnostics(debug=False)`
Returns comprehensive training diagnostics.

**Parameters:**
*   `debug` (bool): If `True`, includes computationally intensive diagnostics such as gradient statistics, persistent gradient info, and detailed optimizer metrics. Default: `False`.

**Returns:**
A dictionary containing:
*   `step_count`: Number of optimization steps taken
*   `last_loss`: Most recent loss value
*   `current_lr`: Effective learning rate (ChaosGrad's live estimate when automatic)
*   `gradient_persistence`: Gradient persistence coefficient
*   `optimizer`: ChaosGrad health metrics (present when ChaosGrad is active; per-family detail in debug mode)
*   `persistent_grads_active`: Number of active persistent gradients (debug mode only)
*   `anomaly_tracking`: Anomaly detection state (debug mode only)
*   `loss_tracking`: Loss buffer statistics (debug mode only)
*   `scaler_state`: AMP scaler information (debug mode only)
*   `gradient_stats`: Gradient norms and means across parameters (debug mode only)

---

## ChaosGrad Optimizer (`odyssnet.training.chaos_optimizer`)

ChaosGrad is OdyssNet's bespoke zero-config optimizer — "the learning teacher." It combines three mechanisms tuned to the chaos core's dynamics:

1.  **Per-synapse preconditioning** (Adam-style second moment): temporal weight reuse across thinking steps makes gradient scales wildly heterogeneous across the NxN matrix; every synapse gets its own effective step size.
2.  **Online distance adaptation** (D-adaptation class estimator): the global step scale is *estimated* from the observed distance traveled toward the solution. No learning rate is ever required.
3.  **Architecture-aware policy**: parameters are auto-classified into families — `chaos_core` (W), `memory_feedback`, `projections` (embed/proj/decoder), `plasticity` (Hebbian logits), and `modulation` (gates, scales, bias, norms). Weight decay applies only to connective structure; the chaos core's zero-diagonal constraint is enforced inside the step.

Two safety systems keep the estimator honest on chaotic landscapes:

*   **Traction limit** (`trust_ratio`): the applied step scale never exceeds `trust_ratio × RMS(initial weights)`, anchored at construction. Shields tiny networks (e.g. the 9-parameter XOR core) from early estimator overshoot.
*   **Loss-spike brake** (`brake_factor`): on a statistical loss spike (3σ and +20% over the running EWMA), a transient ceiling on the *applied* step is cut — the estimator's own bookkeeping is never touched, so it keeps learning the true scale while suppressed. The ceiling releases back toward 1.0 only once the loss has recovered to its pre-spike level; while the loss stays elevated it barely moves, so a slow divergence can't win its step size back just by avoiding further spikes. Counteracts the monotone step-scale growth that destabilizes sharpening temporal tasks. The trainer feeds the loss stream automatically; custom loops call `optimizer.report_loss(loss)`.

### Usage

```python
from odyssnet import ChaosGrad

# Zero-config (what the trainer does internally)
optimizer = ChaosGrad.from_model(model)

# Fixed-rate mode: AdamW-equivalent, reproducible
optimizer = ChaosGrad.from_model(model, lr=1e-4)

# Expert knobs (all optional)
optimizer = ChaosGrad.from_model(
    model,
    d_coef=1.0,            # step-scale multiplier: 0.5 cautious, 2.0 bold
    d0=1e-6,               # initial step-scale estimate
    growth_rate=float('inf'),  # finite values (e.g. 1.02) act as warmup
    d_mode='global',       # 'per_group' = independent estimate per family
    trust_ratio=0.25,      # traction limit (None disables)
    brake_factor=0.5,      # loss-spike brake (None disables)
    betas=(0.9, 0.999),
    use_bias_correction=True,
)

# Plain parameter iterables also work (single group, global estimation,
# but no family policy — prefer from_model for OdyssNet models)
optimizer = ChaosGrad(model.parameters())
```

### Key Methods

*   `ChaosGrad.from_model(model, lr=None, **kwargs)`: zero-config entry point with architecture-aware family grouping.
*   `ChaosGrad.classify_params(model)`: returns the family param-group dicts (useful for custom policies).
*   `optimizer.report_loss(loss)`: feeds the loss-spike brake. Automatic under `OdyssNetTrainer`.
*   `optimizer.get_diagnostics(debug=False)`: `global_step`, `effective_lr`, and per-family stats in debug mode.

---

## Temporal Attention (`odyssnet.core.attention`)

A transformer stacks attention *between layers*. OdyssNet has no layers, so attention goes along the axis it does have: **at every thinking step, the state queries a cache of the states that came before it.**

```python
model = OdyssNet(
    num_neurons=1024,
    input_ids=range(256), output_ids=range(256, 768),
    vocab_size=2048, vocab_mode='discrete',
    attn_heads=4,          # this is the whole switch
)
```

Per step: `q_t = h_t W_q` queries the cache, and the result is added to the same pre-activation signal the recurrence, the memory feedback and the token embedding feed — then the step's activation and RMSNorm bound it like everything else. What the chaos core does by mixing everything through one matrix, attention does by naming what it wants.

### Switching it on changes nothing

`o_proj` is zero-initialized, and the module is constructed *after* the core is initialized so it draws no RNG the core would otherwise have used. Two models built with the same seed, one with `attn_heads=4` and one without, have **the same `W` and produce the same output** until training moves the attention weights. An ablation of attention is therefore a one-variable comparison, and no known-good initialization story (`resonant`, edge of chaos) has to be re-validated to try it.

### …and widening it does not change the scale either

The branch's output is divided by `sqrt(attn_heads · attn_head_dim)`, which is load-bearing rather than cosmetic. `o_proj` starts at zero and every optimizer here is Adam-family, so its magnitude after *k* steps is set by the step size rather than by the gradient — `|o_proj| ~ k·lr` whatever the width — and the contribution it produces would otherwise grow as `sqrt(heads · head_dim)`. A wide branch would then reach a destructive scale in the same number of steps a narrow one takes to reach a useful one.

This was measured, not assumed. On multi-query associative recall at 128 neurons (2 key/value pairs, 400 steps, chance 6.2%), **before** the division:

| | accuracy |
|---|---|
| no attention | 54.6% |
| attention, `o_proj` frozen at zero | 54.1% |
| **1 head × 16** | **54.6%** |
| **4 heads × 64** | **5.9%** — collapsed to chance |

The wide branch had drowned the recurrence it was meant to assist; the narrow one, reaching the same `|o_proj|` but a 4× smaller contribution, was harmless. After the division every width lands on the baseline (54.1–56.5%, including 8 heads × 64). It is the same reasoning behind GPT-2's `1/sqrt(2·n_layers)` residual-branch initialization, with steps in place of layers.

### The KV cache

The query length is always 1 — the core cannot be unrolled in parallel, since step *t*'s input is step *t-1*'s output — so a forward pass is a sequence of single-query attentions, exactly like a transformer's decode phase. Two consequences shape the implementation:

*   **Nothing needs masking or reordering.** Every cached entry is strictly in the past, and softmax over keys is permutation-invariant. Position is carried by RoPE applied to each key *at write time*, so a rotated key stays correct wherever it later sits in the buffer.
*   **The cache is never re-materialized.** The carried history and the current call's writes are attended as separate segments and joined at the *scores* — one row per key, the cheap end of the join — under a single softmax. The result is exactly what one softmax over the concatenated keys gives, while the keys themselves are copied nowhere, so the carry costs one saved tensor no matter how many steps read it.

There are two representations, switched automatically:

| | grad enabled (training) | grad disabled (eval, generation) |
|---|---|---|
| storage | frozen carry + differentiable pending list | preallocated ring, written in place |
| per-step allocation | one concat of the pending segment | none |
| why | in-place writes are illegal under autograd | a KV cache exists to not allocate |

Both evict identically, and the test suite pins them to the same numbers (`tests/core/test_attention.py::TestKVCache::test_ring_and_segmented_paths_agree`). Incremental decoding matches a one-shot pass to float tolerance, so generation and scoring are the same computation.

### What it costs

| term | scaling | notes |
|---|---|---|
| parameters | `2·N·(H·D)` + `2·N·(H_kv·D)` | q/o are full-width, k/v shrink with `attn_kv_heads` |
| cache (inference) | `2·B·H_kv·window·D` | `model.attn.cache_bytes(batch)` |
| keys/values kept for backward | `2·B·H_kv·D·(window + n²/2)` | `n` = writes per forward call; `model.attn.training_cache_bytes(batch, n)` |

That `n²` is in **writes per truncated-BPTT window**, not run length — 48 tokens per optimizer step, not the 226M tokens of a run. It is also why `attn_kv_heads` defaults to 1 and why `attn_write='token'` is the default: `think_gap` then buys extra *reads* of the cache rather than extra *entries* in it.

**Time is the larger cost, and it is per step rather than per token.** Every operation on this path is tiny and there are many of them, so it is bound by kernel launches, not arithmetic — a larger batch rides along nearly free. Measured on an RTX 3060 Ti at 1024 neurons, `think_gap=1`, `chunk=48`, four heads, multi-query:

| batch | attention off | on | on, `attn_read='token'` | cost |
|---|---|---|---|---|
| 64 | 46,160 tok/s | 13,265 | 17,408 | 3.48x |
| 128 | 91,803 | 26,530 | 34,700 | 3.46x |
| 256 | 162,014 | 52,838 | 68,687 | 3.07x |
| 512 | 280,870 | 98,236 | 125,388 | 2.86x |

One head costs the same as four (26,286 vs 27,175 tok/s at batch 128), which is the signature of that regime rather than a rounding error. Take the absolute numbers as this GPU's, not as constants: a consumer card measured immediately after 40 minutes of continuous load reported 55-166k tok/s for the same rows, so let it settle before comparing runs — the ratios held (2.9-4.5x) but the throughput did not. Two design decisions follow from it and are already applied: attention runs with **autocast disabled** (a query of length 1 gains nothing from fp16, while the implicit casts around it cost more than the math — worth 21% on its own), and segments are joined once at the scores instead of being merged afterwards. Together those took batch-128 attention from 18,376 to 27,175 tok/s.

**`torch.compile` is the largest speedup available**, and how much of it survives depends on the configuration. The echo loop issues thousands of small kernels per step and is bound by launching them: fusing it is worth 2.2x on a bare core and 4.0x with plasticity. `attend` and `write` carry `@torch._dynamo.disable` — Inductor miscompiles their autocast-disabled region and `o_proj` dies on `Half != float` — so the graph breaks twice per step around them. With plasticity on there is enough left to fuse that the break is free (17.4 ms/batch against 17.6 fully traced in half); with attention as the only addition to a bare core there is not, and compiling buys close to nothing (48.5 against 45.4 eager). Measure rather than assume on an attention-only run.

Attention itself stays in float32 whatever the surrounding precision, which keeps the cache single-dtype and the softmax exact. Half-precision attention was measured at ~0.017 of loss behind on embedded MNIST with four heads, so the decorator is the cheaper side of that trade in both directions.

The knob that remains yours is `attn_read`: at `think_gap=1` querying once per token instead of once per step is worth ~1.3x, and at `think_gap=0` the two are the same thing.

### Interaction with the rest of the library

*   **Truncated BPTT**: what a call writes is differentiable inside it; what carries into the next call is a constant. `detach_state()` and the end of every `forward()` handle this, mirroring the hidden state exactly.
*   **`model.reset_rows(mask)`**: zeroes the state *and* the attention history of selected batch rows. Staggered cold starts need both — a row whose state is zeroed but whose cache is intact is neither cold nor warm.
*   **ChaosGrad**: the projections land in an `attention` family with weight decay; the QK-norm gains do not (they are not connective structure).
*   **Neurogenesis**: q/k/v grow new input columns as small noise and `o_proj` grows new output rows as zeros — the same asymmetry the core uses — and the cache is dropped, since every entry in it was written by a projection of a different shape.
*   **Checkpoints**: six tensors (`attn.{q,k,v,o}_proj.weight`, `attn.{q,k}_norm.weight`). The cache itself is runtime state and is never serialized, exactly like `model.state`.

### Does it help? Measured, on TinyStories

`examples/advanced/experiment_llm.py` exposes every knob on the command line and ships an ablation preset whose `off` arm is the 2.x architecture exactly — same seed, same `W`:

```bash
python -u experiment_llm.py --mode sweep --sweep attn --minutes 3 --batch 128     # equal wall-clock
python -u experiment_llm.py --mode sweep --sweep attn --max-steps 600 --batch 128 # equal tokens
python -u experiment_llm.py --mode train --tag attn --attn-heads 4 --batch 256 --minutes 25
```

The two questions have opposite answers, and both are worth knowing. 1024 neurons, 2.6M parameters, vocab 2048, `think_gap=1`, batch 128, RTX 3060 Ti:

| | equal tokens (600 steps, 3.69M tok) | equal wall-clock (2 min) |
|---|---|---|
| no attention | 22.19 ppl | **14.55 ppl** on 10.91M tokens |
| 4 heads, multi-query | **18.57 ppl** | 20.11 ppl on 3.16M tokens |
| 4 heads, `attn_read='token'` | 19.75 ppl | 17.75 ppl on 4.19M tokens |
| 4 heads, multi-head | 21.40 ppl | 27.38 ppl on 3.06M tokens |

**Attention learns more per token — 16% better perplexity at the default heads — and costs more per second.** Which fact decides a run depends on whether it is token-limited or time-limited. On a 2.6M-parameter core on a consumer GPU, the clock binds and the 2.x architecture still wins the wall-clock comparison; the per-token advantage is the half that scales with hardware and with core width, since the throughput cost is launch overhead rather than arithmetic. `0.0%` collapsed cold starts in every arm, attention on or off.

---

## Image Diffusion (`examples/advanced/experiment_diffusion.py`)

Diffusion is a loop over time, and OdyssNet is a network whose depth *is* time, so the denoising trajectory and the thinking trajectory can be the same object. With `pulse_mode=False` and a `(B, K, F)` input run for `K*E` steps, `forward` resolves `ratio = E` on its own:

| | |
|---|---|
| frame *k* injected at step `k*E` | one denoising timestep |
| `E` echo steps through the `N x N` core | temporal depth in place of UNet layers |
| output collected at step `(k+1)*E - 1` | the prediction for that timestep |
| `h_t` crosses every frame boundary | the denoiser remembers its own trajectory |
| `attn_write='token'` | one cache entry per denoising step |

The whole reverse trajectory is a single differentiable forward pass, so `train_batch(..., full_sequence=True)` against the trainer's default `MSELoss` is the entire training call. `vocab_size=[F_in, P]` with `vocab_mode='continuous'` makes the model's own `proj` and `output_decoder` the encoder and decoder — there is no VAE, and every learned parameter is inside OdyssNet.

### Why `--predict x0` is the default

The answer is read off `n_out` neurons, so whatever the network emits is a **rank-`n_out` view** of a `P`-dimensional image, and the parameterisation decides whether that rank is enough.

Epsilon is white noise: isotropic, full rank, incompressible. A rank-192 view of it keeps `192/784` of the variance, which pins the achievable MSE at **0.755** however long training runs — and a 573k-parameter run measured **0.767**, saturated rather than undertrained. Natural images are low rank: the same 192 directions carry all but **3.4%** of MNIST's variance.

Measured at 700 steps, each target against its own do-nothing predictor:

| target | val MSE / trivial | at pure noise | at nearly clean |
|---|---|---|---|
| **x_0** | **13.3%** | 0.237 | **0.096** |
| v | 59.7% | 0.236 | 0.998 |
| eps | 80.5% | 0.787 | 0.896 |

`v` is x_0-like at high `t` and epsilon-like at low `t`, so it inherits the rank problem over half the range. `--sweep size` carries epsilon arms at four widths, and they behave as the rank argument requires: always above the bound, monotone in `n_out`, closing on it with training, and sampling at chance throughout.

| `n_out` | bound `1 - n_out/P` | measured, 1200 steps |
|---|---|---|
| 96 | 0.878 | 0.900 |
| 144 | 0.816 | 0.852 |
| 192 | 0.755 | 0.806 (0.767 by 8.5k steps) |
| 288 | 0.633 | 0.710 |

### Does the trajectory memory help? Measured, on MNIST

```bash
python -u experiment_diffusion.py --mode sweep --sweep memory --minutes 4                     # equal wall-clock
python -u experiment_diffusion.py --mode sweep --sweep memory --max-steps 600 --minutes 25    # equal gradients
python -u experiment_diffusion.py --mode train --tag attn --attn-heads 4 --minutes 20
```

`independent` is the control: the same frames, the same targets and the same gradient budget, issued as K separate calls so the denoiser starts each frame with nothing — which is what a UNet sampler does. 512 neurons, 573k parameters, 16 frames x 4 echo, guidance 2.0, seed 42, 600 steps per arm, RTX 3060 Ti:

| arm | val MSE | conditioning fidelity | Frechet | params |
|---|---|---|---|---|
| 4 attention heads | 0.1575 | 71.8% | **38.97** | 901,184 |
| **trajectory (default)** | 0.1550 | **78.4%** | 40.82 | **573,376** |
| attention + plasticity | 0.1518 | 73.2% | 47.69 | 902,720 |
| plasticity, temporal | 0.1564 | 70.4% | 65.79 | 574,912 |
| shared-epsilon trajectory | **0.1351** | 54.2% | 76.85 | 573,376 |
| `independent` (memoryless) | 0.2010 | 39.4% | 83.10 | 573,376 |

**Carrying the trajectory doubles conditioning fidelity and halves the Frechet distance against a memoryless denoiser at an identical parameter count.** That is the claim this example exists to test, and it survived its control.

The same claim, without a second training run: `--mode eval` samples one checkpoint twice, carried and wiped between denoising steps — identical weights, identical guidance, one line of difference at inference time.

| same checkpoint, sampling only | conditioning fidelity | Frechet |
|---|---|---|
| trajectory carried | **85.6%** | **11.0** |
| wiped each step | 58.6% | 22.7 |

Two of the other rows are cautionary. The **shared-epsilon** arm holds the best held-out loss at every budget measured: one epsilon per trajectory leaves two frames enough to recover `x_0` by linear algebra, so the model can learn an inversion rather than a denoiser, and an inversion cannot follow it into sampling. Whether that costs it samples did not replicate — 54.2% conditioning fidelity against `trajectory`'s 78.4% at seed 42, but 83.4% against 83.6% at seed 54321, both at equal wall clock. `--traj-noise iid` is the default because it has never been worse, not because the gap is settled, and both columns are reported so an arm that trades one for the other stays visible.

**Attention** leads on Frechet by 4.5% while behind on conditioning fidelity and held-out loss for 57% more parameters — on one seed at 500 samples that is not a separation, and per parameter it is a loss. It is available (`--attn-heads 4`) and is not the default. **Plasticity** is behind on every column, and it is slow: at equal wall clock the plastic arms reach roughly 6% of the plain arm's step count and attention roughly 28%, because the retained trace grows with the step count, the batch and the neuron count together.

Frechet distances here are taken in a small fixed classifier's feature space, not InceptionV3's, so they are **not FID** and are comparable only between arms of the same sweep.

### Is temporal depth worth more than denoising steps?

`K` frames and `E` echo steps between them multiply into the same compute, so `--sweep depth` holds `K*E = 64` fixed and varies the split. On a UNet this question does not exist — the number of denoising steps and the depth spent inside one are different resources. MNIST, seed 54321, x_0, guidance 2.0, 3 minutes per arm, 573,376 parameters throughout:

| arm | frames | echo | val MSE | conditioning fidelity | Frechet | steps |
|---|---|---|---|---|---|---|
| k32_e2 | 32 | 2 | 0.1104 | 73.0% | **13.06** | 3,593 |
| k16_e4 | 16 | 4 | 0.1058 | 83.6% | 15.97 | 3,800 |
| k8_e8 | 8 | 8 | 0.0986 | 88.0% | 13.96 | 3,932 |
| **k4_e16** | **4** | **16** | 0.1001 | **95.0%** | 15.66 | 4,004 |

**Fidelity climbs monotonically with echo depth at identical compute**, and the deepest arm samples in four denoising steps rather than thirty-two — the cheapest inference in the table by a factor of eight. Frechet stays flat across all four in no order, which is what rules out the climb being the sample distribution narrowing onto a few modes.

The default stays `16 x 4`. This is one seed at equal wall clock, the step counts spread 11% in the deepest arm's favour, and the ranked table crowns `k32_e2` because `RANK_KEY` is Frechet and Frechet is the one column that does not separate here — read the fidelity column for this sweep.

### The width curve

`--sweep size`, same budget and seed, the x_0 arms:

| arm | neurons | `n_out` | params | val MSE | conditioning fidelity | Frechet |
|---|---|---|---|---|---|---|
| n256 | 256 | 96 | 221,152 | 0.1234 | 75.6% | 27.63 |
| n384 | 384 | 144 | 380,880 | 0.1098 | 83.0% | 21.17 |
| n512 | 512 | 192 | 573,376 | 0.1067 | 85.8% | 14.73 |
| n768 | 768 | 288 | 1,056,672 | 0.1007 | **87.6%** | **11.52** |

Returns are still positive at a million parameters and already shallow: 4.8x the parameters of `n256` buys twelve points of fidelity. `n_out` scales with the width in this grid, so the curve mixes capacity with output rank — and the epsilon arms carried alongside it move rank alone, staying at chance fidelity whatever the width.

---

## Advanced Capabilities

### 1. Temporal Depth (Space-Time Tradeoff)
OdyssNet replaces spatial layers with temporal steps. 
*   **Vertical vs Horizontal**: A standard 10-layer network has fixed depth. OdyssNet can be run for 10 or 100 steps on-the-fly.
*   **Dynamic Complexity**: Higher `steps` allow the network more time to reverberate signals through its recurrent core, enabling deeper reasoning without increasing parameter count.

### 2. Gradient Accumulation (Virtual Batch Size)
OdyssNet allows you to simulate massive batch sizes on limited hardware (e.g., consumer GPUs).
*   **How it works:** Instead of updating weights after every batch, it accumulates gradients for `N` steps and then performs a single update.
*   **Usage:**
    ```python
    # Simulates a batch size of 32 * 4 = 128
    trainer.train_batch(x, y, thinking_steps=10, gradient_accumulation_steps=4)
    ```
*   **Benefit:** Allows training large models or using large batch stability without running out of VRAM.

### 3. Gradient Persistence (Ghost Gradients)
By setting `gradient_persistence > 0`, the network retains a fraction of the previous batch's gradient. 
*   **Mechanism**: Uses a decaying echo (linear scaling) of previous gradients.
*   **Use Case**: Smoothing optimization in non-convex landscapes or simulated long-context training.

### 4. Synaptic Regeneration (Darwinian Revive)
OdyssNet can re-initialize synapses that are no longer contributing to the loss signal (stagnant weights).
*   **Concept**: Instead of pruning, near-zero weights are re-initialized using the original weight strategy.
*   **Benefit**: Maximizes network plasticity and parameter efficiency by converting dead capacity into fresh exploration.
*   **Usage**: 
    *   **Threshold Mode**: `trainer.regenerate_synapses(threshold=0.01)`
    *   **Percent Mode**: `trainer.regenerate_synapses(percentage=0.05)`

---

## Model Persistence (`odyssnet.utils.odyssstore`)

The `odyssstore` module provides checkpoint management utilities, including a unique **Weight Transplantation** feature for transferring learned knowledge between models of different sizes.

### Functions

#### `save_checkpoint(model, optimizer, epoch, loss, path, extra_data=None, trainer_state=None)`
Saves a training checkpoint to disk. Pass `trainer_state=trainer.state_dict()` to also persist the trainer's runtime state (step counter, scaler, persistent gradients).

#### `load_checkpoint(model, optimizer, path, device='cpu', strict=True, lr=None, trainer=None)`
Loads a checkpoint. Set `strict=False` to ignore size mismatches (will partially load what fits). Pass `lr` to overwrite the saved learning rate after loading. Pass `trainer` (an `OdyssNetTrainer` instance) to restore runtime trainer state (step counter, scaler, persistent gradients).

#### `transplant_weights(model, checkpoint_path, device='cpu', verbose=True)`
🧬 **Weight Transplantation**: Transfers learned weights from a checkpoint to a model, **even if the number of neurons is different**.

*   **Scaling Up**: Start a 512-neuron model with knowledge from a 256-neuron model. The overlapping 256×256 region is copied, the rest stays initialized.
*   **Scaling Down**: Compress a 1024-neuron model into a 256-neuron model. The most "central" 256×256 weights are preserved.
*   **Warm Starts**: Any learned weights are better than random. Gradients will find their way faster.

```python
from odyssnet import OdyssNet, transplant_weights

# Create a NEW, larger model
big_model = OdyssNet(num_neurons=512, ...)

# Transplant weights from a smaller, trained checkpoint
transplant_weights(big_model, 'small_model_checkpoint.pth')

# big_model now has a "warm start" - training will converge faster!
```

#### `get_checkpoint_info(path, device='cpu')`
Reads checkpoint metadata (epoch, loss, num_neurons) without loading into a model.

---

## Neurogenesis (Network Expansion)

OdyssNet supports dynamic growth, allowing you to add neurons to a live network during training. This mimics biological neurogenesis.

### `trainer.expand(amount=1, verbose=True)`
Dynamically adds `amount` empty neurons to the model.
*   **Continuity**: Optimizers are migrated, so momentum and history are preserved.
*   **State**: The training state is preserved.
*   **Initialization**: 
    *   **Incoming Weights**: 0 (Maintains forward pass stability, new neuron starts inactive).
    *   **Outgoing Weights**: Small random noise (Enables backpropagation / gradient flow).

```python
# Add 1 neuron if loss stagnates
if loss > prev_loss:
    trainer.expand(amount=1)
```

> **Initialization:** New connections are initialized with `micro_quiet_warm` (Normal(0, 1e-3)) noise so they remain dormant relative to trained weights and do not destabilize the existing dynamics. Optimizer momentum is migrated from the old parameters to the expanded ones.

---

## Utilities (`odyssnet.utils`)

### 1. Data Utilities (`odyssnet.utils.data`)

#### `prepare_input(input_features, model_input_ids, num_neurons, device)`
Maps raw input features (numpy or tensor) to the full network state tensor.
*   **Pulse Mode:** Plugs data into `t=0`, leaves rest as 0.
*   **Stream Mode:** Maps sequence data `(Batch, Steps, Features)` to correct neurons.
*   **Auto-Device:** Automatically moves data to the model's device.

```python
from odyssnet.utils.data import prepare_input

x_in, batch_size = prepare_input(X_train, model.input_ids, model.num_neurons, 'cuda')
```

#### `to_tensor(data, device)`
Safely converts any list/array/int/float into a PyTorch tensor on the target device.

```python
from odyssnet.utils.data import to_tensor

data_tensor = to_tensor(data, 'cuda')
```

#### `set_seed(seed=42)`
Sets a fixed seed for **reproducible results** across all random sources (Python, NumPy, PyTorch, CUDA).

*   **Purpose**: Ensures consistent behavior across runs for reliable experimentation and debugging.
*   **Seed Value**: The provided seed is applied to all randomization sources simultaneously.
*   **CUDA Support**: Automatically configures CUDA random state if GPU is available.

```python
from odyssnet import set_seed

# At the start of your script for full reproducibility
set_seed(42)

# Train or run experiments - results will be identical across runs
model = OdyssNet(...)
trainer = OdyssNetTrainer(model)
trainer.fit(x, y, epochs=100)
```

**Best Practice:**
*   Call `set_seed()` **at the start of your script**, before any random operations.
*   Use consistent seed values (e.g., 42) for reproducible example and experiment validation.
*   Different seeds can be used for ensemble training or robustness testing.

### 2. Neurogenesis (`odyssnet.utils.neurogenesis`)
See **Neurogenesis** section above.

### 3. OdyssStore (`odyssnet.utils.odyssstore`)
This module manages model serialization and the transdimensional weight transplantation feature described in the **Advanced Capabilities** section.

### 4. TrainingHistory (`odyssnet.utils.history`)

Lightweight metric accumulator with built-in multi-panel plotting. All example scripts use this to visualize training dynamics.

```python
from odyssnet import TrainingHistory

history = TrainingHistory()

for epoch in range(epochs):
    loss = trainer.train_batch(x, y, thinking_steps=10)
    history.record(loss=loss, lr=current_lr, accuracy=acc)

# Interactive display
history.plot(title="My Experiment")

# Save to file
history.plot(save_path="results/training.png", title="My Experiment")
```

**Methods:**
*   `record(**kwargs)`: Record one or more named metrics for the current step. Values are converted to float.
*   `get(key)`: Return the list of recorded values for a metric name.
*   `metrics`: Property returning names of all recorded metrics.
*   `plot(save_path=None, title="Training History")`: Generate a multi-subplot figure with one panel per metric. If `save_path` is given, saves to disk; otherwise shows interactively. If the environment variable `ODYSSNET_DISABLE_PLOT=1` is set, plotting is skipped entirely (useful for automated testing).

---

## Usage Examples

### Example 1: XOR Logic
```python
# 2 Inputs, 1 Output. 0 Hidden Layers.
model = OdyssNet(num_neurons=3, input_ids=[0, 1], output_ids=[2], device='cuda')
trainer = OdyssNetTrainer(model, lr=5e-3, gradient_persistence=0.1)

# Training logic...
trainer.fit(X, Y, epochs=100, thinking_steps=5)
```

### Example 2: MNIST Asymmetric Vocab
```python
# 784 pixels -> 10 neurons -> 10 logits
model = OdyssNet(num_neurons=10, input_ids=range(10), output_ids=range(10), vocab_size=[784, 10])
# Model handles projection and decoding automatically.
```
