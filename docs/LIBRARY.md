# OdyssNet Library Documentation

OdyssNet is a PyTorch-based library that implements **Zero-Hidden Layer** neural networks using **Temporal Depth**. By treating the neural network as a dynamic system that evolves over time, OdyssNet achieves deep learning capabilities without stacking spatial layers.

## Core Modules

The library is organized into three primary modules:
1.  **`odyssnet.core.network`**: The recurrent core architecture and update dynamics.
2.  **`odyssnet.training.trainer`**: Optimization engine with AdamW and bio-inspired regularization.
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

The `OdyssNetTrainer` handles the training loop, gradient accumulation, mixed precision (AMP), and experimental features like Ghost Gradients. **Prodigy** is the default optimizer (auto-calibrating, no LR tuning required). Pass an explicit `lr` to use AdamW instead, or supply any custom optimizer — including **ChaosGrad**.

### Initialization

```python
from odyssnet import OdyssNetTrainer

# Quick prototyping: Prodigy — auto-calibrates LR, no tuning needed
trainer = OdyssNetTrainer(model, device='cuda')

# Reproducible experiments and production: pin an explicit lr to use AdamW
trainer = OdyssNetTrainer(model, lr=1e-4, device='cuda')

# With optional features
trainer = OdyssNetTrainer(
    model,
    device='cuda',
    gradient_persistence=0.0,
    synaptic_noise=0.0,
    anomaly_hook=my_hook
)

# Custom optimizer (bypasses both Prodigy and AdamW)
import torch
trainer = OdyssNetTrainer(model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4))

# ChaosGrad — zero-hyperparameter optimizer (optional, see ChaosGrad section below)
from odyssnet import ChaosGrad
opt     = ChaosGrad(ChaosGrad.classify_params(model), lr=1e-3)
trainer = OdyssNetTrainer(model, optimizer=opt)
```

**Parameters:**
*   `lr` (float or None): Learning rate. Default: `None`.
    *   `None`: **Prodigy** optimizer is used. Auto-calibrates the learning rate continuously — no manual tuning required. Requires `pip install prodigyopt`. Best for quick prototyping; produces non-deterministic loss curves across runs even with a fixed seed.
    *   float (e.g. `1e-4`): **AdamW** optimizer is used with `weight_decay=0.01`. Recommended for reproducible experiments, benchmarking, and production runs.
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
*   `current_lr`: Current learning rate
*   `gradient_persistence`: Gradient persistence coefficient
*   `persistent_grads_active`: Number of active persistent gradients (debug mode only)
*   `anomaly_tracking`: Anomaly detection state (debug mode only)
*   `loss_tracking`: Loss buffer statistics (debug mode only)
*   `scaler_state`: AMP scaler information (debug mode only)
*   `gradient_stats`: Gradient norms and means across parameters (debug mode only)

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

---

## ChaosGrad Optimizer (`odyssnet.training.chaos_optimizer`)

ChaosGrad is a **fully optional**, zero-hyperparameter optimizer designed specifically for OdyssNet. Pass it as a custom optimizer to bypass the default Prodigy / AdamW selection.

The trainer's default behavior is **unchanged** — Prodigy when `lr=None`, AdamW when `lr=float`.

### When to use ChaosGrad

| Situation | Recommendation |
|-----------|----------------|
| Quick prototyping, first run | Prodigy (default) |
| Reproducible benchmarks | AdamW with explicit `lr` |
| Research into self-tuning dynamics, OdyssNet-specific regularisation | **ChaosGrad** |
| Hebbian plasticity enabled (`hebb_type != None`) | ChaosGrad handles hebb params specially |

### Usage

```python
from odyssnet import OdyssNet, OdyssNetTrainer, ChaosGrad

model   = OdyssNet(num_neurons=32, input_ids=[0], output_ids=[31], device='cuda')

# Classify parameters for group-specific meta-adaptation
opt     = ChaosGrad(ChaosGrad.classify_params(model), lr=1e-3)
trainer = OdyssNetTrainer(model, optimizer=opt, device='cuda')

for epoch in range(100):
    loss = trainer.train_batch(x, y, thinking_steps=10)
    # No LR schedule needed — ChaosGrad adapts autonomously

# Optional: manual plateau escape
trainer.trigger_plateau_escape()

# Diagnostics
diag = trainer.get_diagnostics(debug=True)
opt_diag = diag['optimizer']
print(f"Frustration:    {opt_diag['frustration']:.3f}")
print(f"Avg eff. LR:    {opt_diag['avg_effective_lr']:.4f}")
```

You can also pass plain `model.parameters()` without classification — every parameter will use the `lightweight` group defaults.

### Parameter Classification

`ChaosGrad.classify_params(model)` divides OdyssNet parameters into 9 semantic groups:

| Group | Detection | Init Decay | Beta Equil | Burst |
|-------|-----------|-----------|------------|-------|
| `chaos_core` | `W` | 0.01 | 0.95 | Full |
| `memory` | `memory_feedback` | 0.0 | 0.95 | Full |
| `projections` | `embed`/`proj`/`output_decoder` | 0.01 | 0.90 | Full |
| `gates` | `input_gate`, `output_gate`, `core_gate`, `memory_gate` | 0.0 | 0.85 | Half |
| `hebbian` | `hebb_factor`, `hebb_decay` | 0.0 | 0.90 | **None** |
| `norm` | `norm.*` | 0.0 | 0.90 | Half |
| `bias` | `B` | 0.0 | 0.90 | Half |
| `scales` | `input_scale`, `output_scale` | 0.0 | 0.90 | Half |
| `lightweight` | everything else | 0.0 | 0.90 | Half |

**Hebbian Bypass Rule:** `hebb_factor` and `hebb_decay` **never** receive weight decay regardless of any hypergradient signal. Frustration bursts also skip these parameters entirely.

### Public API

| Method | Signature | Description |
|--------|-----------|-------------|
| `classify_params` | `@staticmethod classify_params(model)` | Returns list of classified param-group dicts |
| `step` | `step(closure=None)` | One autonomous optimization step |
| `report_loss` | `report_loss(loss_value)` | Feed loss to the Frustration Accumulator (trainer does this automatically) |
| `trigger_plateau_escape` | `trigger_plateau_escape()` | Force a frustration burst on the next step |
| `get_diagnostics` | `get_diagnostics(debug=False)` | Optimizer health metrics |

### Frustration Accumulator

ChaosGrad tracks loss stagnation internally. When `frustration > 0.75` (or `trigger_plateau_escape()` is called), it injects noise into the momentum buffers and resets meta-parameters toward their calibrated defaults — providing an automatic escape from plateaus without user intervention.

The `OdyssNetTrainer` automatically calls `report_loss()` after every optimizer step when ChaosGrad is detected.

### Neurogenesis Compatibility

`trainer.expand(amount=N)` works transparently with ChaosGrad. The optimizer state (momentum, meta-parameters, second moments) is migrated to the grown network — old neurons preserve their learned adaptation, new neurons start from cold-start calibration. The global frustration state (`_frustration`, `_best_loss`, `_global_step`) is also preserved across the expansion.

### Checkpoint Save / Load

ChaosGrad's global state (`frustration`, `best_loss`, `global_step`) is included in `optimizer.state_dict()` under the key `'chaos_global'` and is restored by `optimizer.load_state_dict()`. This means `save_checkpoint` / `load_checkpoint` round-trips preserve the full optimizer state including frustration dynamics:

```python
from odyssnet import save_checkpoint, load_checkpoint

save_checkpoint(model, trainer, path="run.pt")
epoch, loss = load_checkpoint(model, trainer, path="run.pt")
# trainer.optimizer._frustration is restored
```

If you override the genesis learning rate at load time, ChaosGrad reads it from the param group (not from `defaults`), so the override takes effect on the next step:

```python
epoch, loss = load_checkpoint(model, trainer, path="run.pt", lr=5e-4)
# ChaosGrad now uses genesis_lr=5e-4 for weight decay and update scaling
```

### Interactions with Other Features

| Feature | Interaction | Notes |
|---------|-------------|-------|
| **Synaptic noise** (`synaptic_noise > 0`) | Noise is added to weights *before* the forward pass. ChaosGrad's `sig_wd = cos(g, W)` therefore measures alignment against the *noisy* weight. | Intentional — noisy W is what gradient was computed against. |
| **Gradient clipping** (applied inside trainer) | All three hypergradient signals are computed on clipped gradients. `grad_ema` also tracks clipped gradients. | Clipping reduces signal magnitude but doesn't break adaptation. |
| **Gradient persistence** | Persisted gradients from the previous step are injected *before* the ChaosGrad step. `sig_lr` therefore measures consistency of the *combined* (current + persisted) gradient vs `grad_ema`. | No issue; effectively a soft gradient accumulation. |
| **Gradient accumulation** | `report_loss` is called once per optimizer step (not per micro-batch), with the un-normalized loss value. `global_step` tracks optimizer steps. | Correct — frustration reflects true convergence, not accumulation count. |
| **Gradient checkpointing** | Recomputes activations during backward. Gradient values reaching ChaosGrad are identical whether or not checkpointing is active. | Fully compatible. |
| **AMP (mixed precision)** | ChaosGrad receives gradients after `scaler.unscale_()` — in float32 scale. ChaosGrad internally casts gradients to float32 (`g_f = grad.float()`). | Fully compatible. |
| **`regenerate_synapses()`** | When weak entries of `W` are re-initialised, the trainer automatically clears ChaosGrad's per-parameter state for `W`. Cold-start recalibration happens on the next step, re-computing `init_lr` from the new gradient scale. | If `revived == 0` (no weights regenerated), state is preserved. |
| **`transplant_weights()`** | Weight transplantation does *not* transfer optimizer state (by design — cold restart after transplant). ChaosGrad cold-starts on all parameters after loading transplanted weights. | Same behaviour as AdamW / Prodigy after transplant. |
| **Neurogenesis (`trainer.expand()`)** | Per-parameter tensors (`momentum`, `grad_ema`) are zero-padded to the new size. Scalar state (`init_lr`, `per_param_lr`, etc.) is copied unchanged. New neurons start from cold-start calibration. Global frustration is preserved. | Fully compatible. |
| **`classify_params` (skipped)** | If you pass `model.parameters()` directly instead of `classify_params(model)`, all parameters — including Hebbian logits — are treated as `lightweight`. The Hebbian bypass rule (no decay, no burst) does NOT apply. Always use `classify_params` on models with `hebb_type != None`. | Documented limitation; no crash. |
| **Anomaly hook** | ChaosGrad has its own internal plateau escape (frustration burst). The trainer's anomaly hook fires independently based on loss statistics. The two mechanisms don't interfere. | Use both together if needed. |
