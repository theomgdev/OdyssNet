# OdyssNet 2.2 Library Documentation

OdyssNet is a PyTorch-based library that implements **Zero-Hidden Layer** neural networks using **Temporal Depth**. By treating the neural network as a dynamic system that evolves over time, OdyssNet achieves deep learning capabilities without stacking spatial layers.

## Core Modules

The library is organized into three primary modules:
1.  **`odyssnet.core.network`**: The recurrent core architecture and update dynamics.
2.  **`odyssnet.training.trainer`**: Optimization engine with 8-bit support and bio-inspired regularization.
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
    weight_init=['quiet', 'resonant', 'quiet', 'zero'],
    activation=['none', 'tanh', 'tanh', 'none'],
    gate=None,           # Default resolves to ['none', 'none', 'identity']
    vocab_size=None,     # Optional: Decouples input/output size from neurons
    vocab_mode='hybrid', # 'hybrid', 'discrete', or 'continuous'
    hebb_type=None,      # Optional: Plasticity resolution — None, 'global', 'neuron', or 'synapse'
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
*   `weight_init` (str or list[str]): Weight initialization strategy. Default is `['quiet', 'resonant', 'quiet', 'zero']` for [Encoder/Decoder, Core, Memory, Gates]. Single string values are expanded intelligently.
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
*   `hebb_type` (str or None): Controls the structural resolution of **Heterogeneous Synaptic Plasticity**. Default is `None` (plasticity disabled).

    | `hebb_type` | Parameter Shape | Extra Params | Mechanics |
    |---|---|---|---|
    | `None` | — | 0 | Disabled. |
    | `"global"` | scalar `()` | +2 | Uniform plasticity — the whole network is equally plastic. |
    | `"neuron"` | vector `(N,)` | +2N | Per-neuron plasticity — each neuron learns its own adaptation rate. |
    | `"synapse"` | matrix `(N, N)` | +2N² | Per-synapse plasticity — each connection has its own factor and decay. |

    *   Two learnable logit parameters are created according to the resolution:
        *   `hebb_factor` (raw logit → `sigmoid` → learning rate ≈ 0.047 initially)
        *   `hebb_decay` (raw logit → `sigmoid` → retention ≈ 0.90 initially)
    *   During each forward pass the model accumulates temporal cross-neuron correlations $C_t = h_t \otimes h_{t-1}$ and applies them as $W_\text{eff} = W + (f_h \odot C_t)$ (where $f_h$ is `hebb_factor`), with $\odot$ element-wise multiplication broadcast to the chosen resolution.
    *   The Hebbian state is persisted across forward calls via registered buffers (`hebb_state_W`, `hebb_state_mem`) and is cleared by `reset_state()`.
    *   Both `hebb_factor` and `hebb_decay` are fully differentiable — gradients flow into them via the recurrent computation so the network **learns how to learn** online.
    *   **Memory cost**: `"global"` adds negligible overhead; `"neuron"` adds $O(N)$; `"synapse"` triples total parameter count to $3N^2$.
    *   See [ChaosGrad — Parameter Groups](#parameter-groups) for how the optimizer treats these parameters.
*   `gate` (None, str, or list[str]): Optional parametric gating mechanism. Default is `None`, which resolves to `['none', 'none', 'identity']`.
    *   `None`: Default configuration with memory identity gate enabled, others disabled.
    *   `str` (e.g., `'sigmoid'`): Applies the same gate activation to all three branches `[encoder_decoder, core, memory]`.
    *   `list[str]`: Specify individual gate activations for up to 3 branches in `[encoder_decoder, core, memory]` order. Missing entries use defaults.
    *   `'none'`: Completely disables the gate branch (no learnable parameters).
    *   `'identity'`: Enables identity gating with learnable parameters (starts at identity function but can adapt).
    *   Gate parameters are initialized using the 4th entry in `weight_init` (default: `'zero'`).
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

The `OdyssNetTrainer` handles the training loop, gradient accumulation, mixed precision (AMP), and experimental features like Ghost Gradients. **ChaosGrad** is the default and only built-in optimizer — no configuration dictionary is required.

### Initialization

```python
from odyssnet import OdyssNetTrainer, TemporalSchedulerConfig

# ChaosGrad is the default — just pass a genesis lr
trainer = OdyssNetTrainer(model, lr=1e-3, device='cuda')

# With TemporalScheduler
trainer = OdyssNetTrainer(
    model,
    lr=1e-4,
    device='cuda',
    scheduler_config=TemporalSchedulerConfig.adaptive(),
    gradient_persistence=0.0,
    synaptic_noise=0.0,
    anomaly_hook=my_hook
)

# Custom optimizer (bypasses ChaosGrad)
import torch
trainer = OdyssNetTrainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3))
```

**Parameters:**
*   `lr` (float): Genesis learning rate for ChaosGrad. This is the single mathematical starting point from which all per-parameter learning rates autonomously adapt. Default: `1e-3`.
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
    *   **Usage**: Allows for smart interventions (like calling `trigger_plateau_escape()` when stuck).

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

#### `trainer.trigger_plateau_escape()`
Manually triggers the plateau escape algorithms (noise injection & warm restarts) in both the optimizer and scheduler. Can be tied with the `anomaly_hook`.

#### `trainer.get_diagnostics()`
Returns training diagnostics including optimizer and scheduler state.

---

## ChaosGrad Optimizer (`odyssnet.training.chaos_optimizer`)

A zero-hyperparameter optimizer that autonomously adapts every learning mechanism via
Analytic Hypergradient Descent. All meta-parameters — learning rate multiplier, momentum
beta, weight decay, and gradient centralization — emerge from the intrinsic geometry of the
loss landscape at each step. No configuration dictionary required.

### Autonomous Mechanisms

| Mechanism | Formula | What it replaces |
| :--- | :--- | :--- |
| **Cold-start LR** | `per_lr₀ = min(1/g_rms, 2√numel)` — normalized first step; size cap protects small tensors | manual per-group lr tuning |
| **Per-param LR** | conf-weighted drive + restore(→`init_lr`) + derived coupling(LR×β step bound) | `lr_mult`, `adaptive_lr_clip` |
| **Per-param Momentum** | conf-weighted drive + restore(→0.9) + derived coupling(β×LR) | static `betas` |
| **Per-param Decay** | conf-weighted drive + restore(→`seed/per_lr`); lr×decay product preserved | global `weight_decay` |
| **Centralization Gate** | conf-weighted drive + restore(→0.5) | boolean `grad_centralization` |
| **Frustration Accumulator** | loss stagnation → burst scaled to `genesis_lr × init_lr`; meta-param reset toward `init_lr` | integer `plateau_patience` |

### Hebbian Bypass Rule
Hebbian logits (`hebb_factor`, `hebb_decay`) are raw unbounded scalars governing temporal
working memory. The autonomous decay mechanism is **unconditionally bypassed** for these
parameters — their `per_param_decay` is permanently fixed at `0.0`.

### Parameter Groups
ChaosGrad still classifies parameters to seed initial decay values correctly:

| Group | Initial Decay Seed |
| :--- | :--- |
| **chaos_core** (W matrix) | 0.01 |
| **memory_feedback** | 0.0 |
| **projections** | 0.01 |
| **gates** | 0.0 |
| **hebbian** | 0.0 (bypass enforced) |
| **lightweight** | 0.0 |

### Optimizer State per Parameter
```
step            (int)        — update count
prev_grad       (bfloat16)   — previous gradient (VRAM-compressed)
momentum        (float32)    — gradient accumulator
init_lr         (float)      — calibrated starting LR = 1/g_rms at T=0, capped by 2√numel (fixed)
per_param_lr    (float)      — autonomous LR multiplier, init = init_lr
per_param_decay (float)      — autonomous weight decay rate
per_param_beta  (float)      — autonomous momentum coefficient, init 0.9
per_param_alpha (float)      — autonomous centralization gate, init 0.5
```

### Usage

```python
from odyssnet import ChaosGrad

# Single parameter — the genesis lr
param_groups = ChaosGrad.classify_params(model)
optimizer = ChaosGrad(param_groups, lr=1e-3)

# Feed loss to Frustration Accumulator (optional but recommended)
optimizer.report_loss(loss_value)

# Manual escape from local minima
optimizer.trigger_plateau_escape()

# Diagnostics
diag = optimizer.get_diagnostics()
# Returns: global_step, frustration, best_loss, avg_effective_lr, avg_init_lr
```

---

## TemporalScheduler (`odyssnet.training.chaos_scheduler`)

An **adaptive LR scheduler** that monitors the training process and adjusts in real-time.

### Training Phases
1.  **Warmup**: Linear ramp from 0 to `max_lr` (prevents chaos explosion at start).
2.  **Cosine Decay**: Smooth decay to `min_lr_ratio × max_lr`.
3.  **Warm Restart**: When a plateau is detected, temporarily boosts LR above the current base learning rate with a decaying factor, then begins a new cosine cycle.

### Pre-built Configurations

```python
from odyssnet import TemporalSchedulerConfig

TemporalSchedulerConfig.default()          # Standard cosine decay
TemporalSchedulerConfig.llm()              # LLM-style long training
TemporalSchedulerConfig.short_experiment() # Quick PoC runs
TemporalSchedulerConfig.finetune()         # Conservative schedule
TemporalSchedulerConfig.adaptive()         # Full auto-restart mode
```

### Features
*   **Loss-Trend Awareness**: Adapts decay speed based on convergence rate.
*   **Plateau Detection**: Auto-triggers warm restarts when training stalls.
*   **Convergence Rate Tracking**: `scheduler.get_convergence_rate()` returns positive (bad) or negative (good).
*   **Checkpoint Support**: Full `state_dict()` / `load_state_dict()` for seamless resume.

```python
# Direct usage (standalone)
from odyssnet import TemporalScheduler

scheduler = TemporalScheduler(
    optimizer,
    warmup_steps=500,
    max_steps=5000,
    patience=100  # Auto-restart on plateau
)

# In training loop:
scheduler.step(loss=current_loss)  # Pass loss for adaptive behavior

# Or integrated via Trainer:
trainer = OdyssNetTrainer(model, scheduler_config=TemporalSchedulerConfig.adaptive())
# Scheduler steps automatically inside train_batch()
```

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

#### `save_checkpoint(model, optimizer, epoch, loss, path, extra_data=None)`
Saves a training checkpoint to disk.

#### `load_checkpoint(model, optimizer, path, device='cpu', strict=True)`
Loads a checkpoint. Set `strict=False` to ignore size mismatches (will partially load what fits).

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
*   Use consistent seed values (e.g., 42) for reproducible PoC and experiment validation.
*   Different seeds can be used for ensemble training or robustness testing.

### 2. Neurogenesis (`odyssnet.utils.neurogenesis`)
See **Neurogenesis** section above.

### 3. OdyssStore (`odyssnet.utils.odyssstore`)
This module manages model serialization and the transdimensional weight transplantation feature described in the **Advanced Capabilities** section.

---

## Usage Examples

### Example 1: XOR Logic
```python
# 2 Inputs, 1 Output. 0 Hidden Layers.
model = OdyssNet(num_neurons=3, input_ids=[0, 1], output_ids=[2], device='cuda')
trainer = OdyssNetTrainer(model, gradient_persistence=0.1)

# Training logic...
trainer.fit(X, Y, epochs=100, thinking_steps=5)
```

### Example 2: MNIST Asymmetric Vocab
```python
# 784 pixels -> 10 neurons -> 10 logits
model = OdyssNet(num_neurons=10, input_ids=range(10), output_ids=range(10), vocab_size=[784, 10])
# Model handles projection and decoding automatically.
```