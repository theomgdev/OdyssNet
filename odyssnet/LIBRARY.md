# 📘 OdyssNet Library Documentation

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
    vocab_mode='hybrid'  # 'hybrid', 'discrete', or 'continuous'
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
*   `gate` (None, str, or list[str]): Optional parametric gating mechanism. Default is `None`, which resolves to `['none', 'none', 'identity']`.
    *   `None`: Default configuration with memory identity gate enabled, others disabled.
    *   `str` (e.g., `'sigmoid'`): Applies the same gate activation to all three branches `[encoder_decoder, core, memory]`.
    *   `list[str]`: Specify individual gate activations for up to 3 branches in `[encoder_decoder, core, memory]` order. Missing entries use defaults.
    *   `'none'`: Completely disables the gate branch (no learnable parameters).
    *   `'identity'`: Enables identity gating with learnable parameters (starts at identity function but can adapt).
    *   Gate parameters are initialized using the 4th entry in `weight_init` (default: `'zero'`).

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

#### `model.forward(x_input, steps=1, current_state=None)`
Runs the dynamic system.
*   `x_input`: Input tensor. Can be a single pulse or a sequence (index-based or dense).
*   `steps`: **Thinking Time**. How many times the signal reverberates in the echo chamber.
*   `current_state`: Optional. Pass a previous state to continue from.
*   **Returns**: `(all_states, final_state)`
    *   `all_states`: Tensor of shape `(Batch, Steps, Neurons)` - **Batch-first format**.
    *   `final_state`: Tensor of shape `(Batch, Neurons)` - The last hidden state.

---

## OdyssNet Trainer (`odyssnet.training.trainer`)

The `OdyssNetTrainer` handles the training loop, gradient accumulation, mixed precision (AMP), and experimental features like Ghost Gradients. It now supports the **ChaosGrad** optimizer and **TemporalScheduler** for OdyssNet-native training.

### Initialization

```python
from odyssnet import OdyssNetTrainer, ChaosGradConfig, TemporalSchedulerConfig

# Standard (Legacy Compatible — uses AdamW8bit or AdamW)
trainer = OdyssNetTrainer(model, device='cuda')

# ChaosGrad Only — Fixed LR, No Scheduler (Observe with your own LR)
trainer = OdyssNetTrainer(model, lr=1e-3, device='cuda',
                         use_chaos_grad=True)
# Or with a specific config:
trainer = OdyssNetTrainer(model, lr=1e-3, device='cuda',
                         chaos_config=ChaosGradConfig.default(lr=1e-3))

# Full Featured — ChaosGrad + TemporalScheduler
trainer = OdyssNetTrainer(
    model, 
    lr=1e-4,
    device='cuda',
    chaos_config=ChaosGradConfig.conservative(lr=1e-4),       # ChaosGrad Optimizer
    scheduler_config=TemporalSchedulerConfig.adaptive(),  # Adaptive Scheduler
    gradient_persistence=0.0,   # Ghost Gradients (Persistence)
    synaptic_noise=0.0,         # Thermal Noise (Default: 0.0)
    anomaly_hook=my_hook        # (Optional) Callable triggered on plateaus/spikes
)
```

> **Note:** When using ChaosGrad without a scheduler, LR stays fixed at the value you set. 
> ChaosGrad's per-parameter adaptive scaling and gradient centralization still apply — 
> you get all the optimizer benefits without automatic LR decay.

**Auto-Optimization (bitsandbytes):**
*   If `optimizer` is `None` AND `device` is `'cuda'`, the trainer will attempt to load `bitsandbytes` and use `AdamW8bit`.
*   **VRAM Savings:** ~75% reduction in optimizer memory.
*   **Control Variables (Set before import):**
    *   `os.environ["NO_BNB"] = "1"`: Disables 8-bit optimizer (Forces standard Torch AdamW).
    *   `os.environ["VERBOSE_BNB"] = "1"`: Enables verbose loading logs for debugging.

**Parameters:**
*   `lr` (float): The initial learning rate. If no optimizer is provided, this LR is used to initialize one. It is also stored as `trainer.initial_lr` for use in Scheduler warm restarts or resets.
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

A **OdyssNet-native optimizer** that understands and exploits the chaos chamber dynamics. Unlike generic AdamW which treats all parameters identically, ChaosGrad classifies parameters into groups and applies different strategies.

### Parameter Groups
| Group | Parameters | Strategy |
| :--- | :--- | :--- |
| **chaos_core** | W matrix (cross-connections) | Spectral monitoring, adaptive LR, plateau escape |
| **memory_feedback** | Neuron self-connections | Independent LR and ultra-low decay to preserve temporal memories |
| **projections** | Embeddings, Projections, Decoder | Standard LR with configurable decay |
| **gates** | input_gate, output_gate, core_gate, memory_gate | Dedicated LR/decay controls for branch-wise gating |
| **lightweight** | Bias, Scale, Norm | Higher LR, no weight decay |

### Key Features
*   **Gradient Centralization**: Removes gradient mean for faster convergence.
*   **Adaptive LR**: Per-parameter LR scaling based on gradient consistency.
*   **Plateau Escape**: Controlled gradient perturbation when training stalls.
*   **Spectral Clipping**: Keeps chaos core's spectral radius bounded (edge-of-chaos control).
*   **Gate-Aware Controls**: `gate_lr_mult` and `gate_decay` tune gate dynamics independently from memory/projection groups.

### Pre-built Configurations

```python
from odyssnet import ChaosGradConfig

ChaosGradConfig.conservative(lr=1e-4)  # Conservative balanced (Standard training)
ChaosGradConfig.default(lr=3e-4)       # Explorer (Fresh/small networks)
ChaosGradConfig.finetune(lr=1e-5)      # Conservative (Fine-tuning)
ChaosGradConfig.large_network(lr=1e-4) # Robust monitoring (1000+ neuron networks)
ChaosGradConfig.tiny_network(lr=0.01)  # Minimal (XOR, Identity)

# Gate-specific override example
cfg = ChaosGradConfig.default(lr=3e-4)
cfg['gate_lr_mult'] = 1.2
cfg['gate_decay'] = 0.0
```

### Direct Usage

```python
from odyssnet import ChaosGrad

# Classify parameters and create optimizer manually
param_groups = ChaosGrad.classify_params(model)
optimizer = ChaosGrad(param_groups, lr=1e-4, plateau_patience=100)

# Report loss for plateau detection
optimizer.report_loss(loss_value)

# Get diagnostics
diag = optimizer.get_diagnostics()
```

---

## TemporalScheduler (`odyssnet.training.chaos_scheduler`)

An **adaptive LR scheduler** that monitors the training process and adjusts in real-time.

### Training Phases
1.  **Warmup**: Linear ramp from 0 to `max_lr` (prevents chaos explosion at start).
2.  **Cosine Decay**: Smooth decay to `min_lr_ratio × max_lr`.
3.  **Warm Restart**: When plateau detected, temporarily boosts LR with decaying restarts.

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