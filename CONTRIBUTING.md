# Contributing to OdyssNet

This document outlines the standards and best practices for contributing to the OdyssNet project — whether you are adding example scripts, improving the library, or fixing bugs.

OdyssNet relies on a highly modular library structure. To ensure long-term maintainability and performance, all contributions must adhere to these guidelines.

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone <repo-url> && cd odyssnet

# Install in development mode (required — makes `from odyssnet import ...` work everywhere)
pip install -e ".[dev]"

# Verify installation
python -m pytest tests/
# OR simply
pytest tests/ # but not recommended as it will not see your env setup
```

> **CUDA note:** `requirements.txt` pins CUDA 11.8. For RTX 4000/5000 series GPUs, install PyTorch separately:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

---

## 📂 Project Structure

```
odyssnet/              # Library source code
  core/network.py      # OdyssNet model
  training/trainer.py  # OdyssNetTrainer
  utils/               # Data, checkpointing, neurogenesis, history
tests/                 # Test suite (mirrors odyssnet/ structure)
examples/              # Core validation scripts (identity, XOR, MNIST)
  advanced/            # Complex experiments (reasoning, generation, transfer)
```

### Example Directories

We distinguish between **Core Validations** and **Feature Experiments**.

### 1. `examples/` (Root)
*   **Purpose:** Contains minimal, "hello world" style scripts that validate the core laws of OdyssNet physics.
*   **Examples:** `convergence_identity.py` (Can signals pass?), `convergence_gates.py` (Can it solve XOR?).
*   **Rule:** Scripts here should be extremely simple, fast, and prove a fundamental property of the architecture.

### 2. `examples/advanced/`
*   **Purpose:** Contains complex tasks, task-specific logic, and demonstrations of advanced cognitive behaviors.
*   **Examples:** `convergence_detective_thinking.py` (Reasoning), `convergence_latch.py` (Willpower).
*   **Rule:** If you are building a task (like adding numbers, generating waves, or playing a game), it goes here.

---

## 🛠️ Library Usage Best Practices

**⛔ DO NOT** re-invent the wheel.
**✅ DO** use the Library.

### 1. Always Use `OdyssNetTrainer`
Never write your own manual PyTorch training loop (`optimizer.step()`, `loss.backward()`, etc.) unless absolutely necessary for low-level research.

*   **Why?** The `OdyssNetTrainer` handles:
    *   **Automatic Mixed Precision (AMP):** Faster training on Tensor Cores.
    *   **Gradient Accumulation:** Simulating large batches.
    *   **Ghost Gradients (Persistence):** Advanced stabilization.
    *   **State Management:** Resetting hidden states automatically.

```python
# ❌ BAD: Manual Loop
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

# ✅ GOOD: Trainer
trainer = OdyssNetTrainer(model, device='cuda')
trainer.train_batch(input, target, thinking_steps=10)
```

### 2. Extend, Don't Hack
If you need a new feature (e.g., a new loss function or a custom metric), extend `OdyssNetTrainer` or pass arguments to it. If the library is missing a critical feature, **implement it in the library first**, then use it in your example.

---

## ⚙️ Initialization Protocols (Critical)

OdyssNet is sensitive to initialization. The default `weight_init='resonant'` is the recommended starting point for all tasks — it places the weight matrix at the Edge of Chaos (ρ(W) = 1.0) from the start and works across all network sizes.

### A. Universal Default (All Sizes)
For most tasks without specific constraints, use the default resonant initialization.
*   **Activation:** `'tanh'`
*   **Weight Init:** `'resonant'` *(Default)* — Rademacher ±1 skeleton + spectral normalization to ρ = 1.0. Ensures signal fidelity without exploding or vanishing. Projection layers (embed/proj/decoder) automatically use `'quiet'` init.
*   **Gate:** `None` *(Default)* — resolves to `['none', 'none', 'identity']` (memory identity gate enabled; starts closed with zero gate init and opens if training needs it).
*   **Dropout:** `0.0` *(Default)* — Enable explicitly (e.g., `0.1`) only when overfitting is observed.

```python
model = OdyssNet(..., activation='tanh')  # weight_init='resonant' is already the default
```

### B. Tiny Networks & Logic Gates (< 10 Neurons) — Alternative
If `resonant` convergence is too slow on very small circuits:
*   **Activation:** `'gelu'` — Better gradient flow in sparse/small graphs.
*   **Weight Init:** `'xavier_uniform'` — High variance ensures signals don't die in small circuits.
*   **Gate (Optional):** `'sigmoid'` — Stronger branch control if needed.
*   **Dropout:** `0.0` — Every neuron is vital in small networks.

```python
model = OdyssNet(..., activation='gelu', weight_init='xavier_uniform', dropout_rate=0.0)
trainer = OdyssNetTrainer(model, ..., synaptic_noise=0.0)  # Disable noise for pure logic
```

### C. Large Networks & Memory Tasks — Alternative
For long-horizon temporal stability:
*   **Activation:** `'tanh'`
*   **Weight Init:** `'orthogonal'` — Solid fallback for pure stability.
*   **Gate (Optional):** `['none', 'none', 'sigmoid']` — Memory-only gating.
*   **Dropout:** `0.0` *(Default)* — Enable explicitly when overfitting is a concern.

```python
model = OdyssNet(..., activation='tanh', weight_init='orthogonal')
```

### Gate Contract (Init API)
*   `gate=None`: Default branch layout `['none', 'none', 'identity']`.
*   `gate='sigmoid'`: Applies same gate activation to all `[encoder_decoder, core, memory]` branches.
*   `gate=['none', 'none', 'sigmoid']`: Only memory branch is gated.
*   `gate=['none', 'none', 'none']`: Disables all gating.
*   List supports 1-3 entries, right-padded from defaults.
*   `'none'`: Disables gate branch entirely (no learnable parameters).
*   `'identity'`: Enables explicit identity gating (learnable gate params exist, starts at identity).
*   Gate parameter initialization uses the 4th `weight_init` slot. Default layout is `['quiet', 'resonant', 'quiet', 'zero']`.
*   Activation layout supports 1-4 entries with default `['none', 'tanh', 'tanh', 'none']`; 4th slot is reserved for config symmetry.

### D. Associative Memory (Database / Key-Value)
For tasks requiring precise storage and retrieval of values over time (e.g., Neural Database):
*   **Structure:** High neuron count (256+) to provide storage space for memories.
*   **Init:** Default `resonant` initialization is appropriate.

```python
model = OdyssNet(num_neurons=256, ...)  # resonant default works well
```

### E. Decoupled Projection (Asymmetric Vocabulary)
For tasks requiring high input/output dimensionality (like vision or LLMs) without scaling the core state size:
*   **Feature:** Use `vocab_size=(V_IN, V_OUT)` to decouple input/output resolution from internal neuron count.
*   **Optimization:** Allows a tiny "Thinking Core" (e.g., 10 neurons) to process high-resolution signals (e.g., 784 pixels), achieving extreme parametric efficiency.
*   **Usage:** Best used in conjunction with sequential signal processing for maximum compression.
*   **Note:** When `weight_init='resonant'`, projection layers (embed/proj/decoder) automatically use `'quiet'` init (Normal(0, 0.02)) — no manual override needed.

```python
# OdyssNet core has N=10 neurons, but processes 784 input channels and 10 output classes
model = OdyssNet(num_neurons=10, ..., vocab_size=(784, 10))
```

### F. Heterogeneous Synaptic Plasticity (`hebb_type`)
For tasks where **online synaptic plasticity** may help — e.g., fast-adaptation, continual learning, or tasks with shifting statistics — enable one of the three resolution levels:

| `hebb_type` | Extra Params | When to use |
|---|---|---|
| `"global"` | +2 | Quick experiments; uniform plasticity across all synapses. |
| `"neuron"` | +2N | RL and reactive environments; per-neuron "caste" differentiation. |
| `"synapse"` | +2N² | Logic, NLP, and reasoning tasks requiring **dynamic variable binding**. |

*   **What it does:** At each step the network accumulates temporal cross-neuron correlations $C_t = h_t \otimes h_{t-1}$ and applies them as $W_\text{eff} = W + (f_h \odot C_t)$ (where $f_h$ is `hebb_factor`). Both `hebb_factor` and `hebb_decay` are **learnable** — the network discovers how plastic each synapse should be.
*   **State:** Correlations are persisted via buffers (`hebb_state_W`, `hebb_state_mem`) across intra-sequence forward calls and are explicitly cleared on `reset_state()` between sequences.
*   **Best Use Case (Generation / Sequential Building):** Hebbian shines in tasks where step T relies heavily on expanding or completing a pattern from step T-1. It provides a powerful **short-term working memory** between steps, acting as a dynamic shortcut that fast-tracks sequence generation.
*   **When *not* to use it (Classification / Independent Features):** Avoid Hebbian in classification tasks where each step processes distinct, independent chunks of information (e.g. sequential MNIST classification). In these tasks, inter-step short-term memory acts as "overfit noise".
*   **Compatibility:** Fully compatible with `gradient_checkpointing=True`.
*   **Combined with gating:** Hebbian and gate parameters are independent groups; both can be active simultaneously.

```python
# NLP / Logic / Reasoning — synapse-level plasticity for dynamic variable binding
model = OdyssNet(
    num_neurons=32,
    input_ids=[0, 1],
    output_ids=[31],
    activation='tanh',
    hebb_type='synapse',   # Per-synapse plasticity
    device='cuda',
)

# RL / reactive environments — neuron-level caste differentiation
model = OdyssNet(
    num_neurons=64,
    input_ids=list(range(8)),
    output_ids=list(range(56, 64)),
    activation='tanh',
    hebb_type='neuron',    # Per-neuron plasticity
    device='cuda',
)

# Quick experiment — global plasticity
model = OdyssNet(..., hebb_type='global')

# Default: Prodigy optimizer — auto-calibrates LR, no tuning needed
trainer = OdyssNetTrainer(model)

# AdamW: pass an explicit learning rate
trainer = OdyssNetTrainer(model, lr=3e-4)

# ChaosGrad: optional zero-hyperparameter optimizer (pass as custom optimizer)
from odyssnet import ChaosGrad
opt     = ChaosGrad(ChaosGrad.classify_params(model), lr=1e-3)
trainer = OdyssNetTrainer(model, optimizer=opt)
```

> **Optimizer selection guide:**
> - **Prodigy** (`lr=None`, default) — best for quick experiments; non-deterministic curves.
> - **AdamW** (explicit `lr`) — reproducible runs, benchmarks, production.
> - **ChaosGrad** (pass as `optimizer=`) — research into self-tuning dynamics; ideal when `hebb_type` is enabled (Hebbian parameters are unconditionally protected from weight decay and burst noise).

---

## ⚡ Hardware Optimization

### 1. TensorFloat-32 (TF32)
Always enable TF32 on Ampere+ GPUs for free speedup.
```python
import torch
torch.set_float32_matmul_precision('high')
```

### 2. Compilation
For production or long training runs, compile the model.
```python
model.compile() # Uses torch.compile (PyTorch 2.0+)
```

---

## 🌱 Neurogenesis Protocols

Experiments should handle training stagnation intelligently by adding neurons when needed.
1.  **Metric:** If `loss` has not improved for `N` epochs.
2.  **Action:** Call `trainer.expand(amount=...)`.
3.  **Amount:**
    *   Small nets (< 100 neurons): +1 neuron per expansion.
    *   Large nets (≥ 100 neurons): +10 neurons or +1% of current size per expansion.

```python
if loss > prev_loss:
    trainer.expand(amount=10)
```

---

## 🩺 Diagnostics and Anomaly Interventions

Experiments that run for a long time should handle training stagnation or spikes intelligently without manual restarts.
You can pass an `anomaly_hook` to the `OdyssNetTrainer` to automate recovery and logging.

```python
def my_hook(anomaly_type, loss_val):
    if anomaly_type == "plateau":
        print("Triggering plateau escape!")

trainer = OdyssNetTrainer(model, anomaly_hook=my_hook)
```

---

## 🔢 Data Standards

### 1. Bipolar Logic (-1 vs 1)
Since `tanh` is our primary activation for robust systems, **avoid using 0.0 and 1.0** for logical states.
*   **OFF:** `-1.0`
*   **ON:** `1.0`
*   **Neutral/Silence:** `0.0`

This symmetry helps the gradients flow much better than a `0.0` (which is the most unstable point of tanh).

### 2. Sequence Handling
Use the `prepare_input` utility implicitly via the Trainer.
*   **Pulse:** Single Event at t=0.
*   **Stream:** Continuous sequence. pass `full_sequence=True` to `trainer.predict()` or `train_batch()` if you need frame-by-frame monitoring.

---

## 📝 Code Style & Documentation

1.  **Reproducibility & Seeding:** 🔴 **MANDATORY** — All example and experiment scripts MUST set a fixed seed for reproducible results.
    *   **Why?** Reproducible results are essential for debugging, comparing strategies, and publishing findings.
    *   **How?** Always call `set_seed(42)` at the **very start** of your `main()` function, before any random operations.
    *   **Import:** `from odyssnet import set_seed`
    *   **Example:**
    ```python
    def main():
        set_seed(42)  # ← FIRST LINE in main()
        
        # Now all randomness is locked: model init, data shuffling, dropout, etc.
        model = OdyssNet(...)
        trainer = OdyssNetTrainer(model, lr=1e-4)  # pin lr for deterministic curves
        trainer.fit(X, Y, epochs=100)
    ```
    *   **Applies to:** 
        *   Model weight initialization (deterministic via `torch.manual_seed`).
        *   Data shuffling / batch sampling.
        *   Dropout and stochastic regularization.
        *   CUDA random state (for GPU consistency).
    *   **Test:** If you run the script twice with the same seed, loss curves and final results should be **identical**, byte-for-byte. Note: this requires passing an explicit `lr` to `OdyssNetTrainer` — the default `lr=None` (Prodigy) adapts its learning rate online and will produce different curves across runs.

2.  **Visuals:** Your example should print a cool visualization. Don't just print "Loss: 0.01". Print the timeline.
    *   *Example:* `t=05 | Input: 1 | Output: 0.99 🟢`
3.  **Comments:** Explain *why* you chose a specific setup.
    *   *Example:* `# GAP=3 allows the model time to digest the previous bit.`
4.  **File Paths:** Never use hardcoded absolute paths or assume the CWD. Always construct paths relative to the script file.
    *   *Example:* `DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'file.txt')`
5.  **Checkpointing:** Always use the library's `save_checkpoint`, `load_checkpoint`, and `transplant_weights` functions from `odyssnet.utils.odyssstore`. Do NOT write custom checkpoint code. If the library is missing a feature, extend the library instead.
    *   *Example:* `from odyssnet import save_checkpoint, load_checkpoint, transplant_weights`
6.  **Training History & Plotting:** All finite-duration examples MUST use `TrainingHistory` to record metrics (loss, accuracy, lr, etc.) and call `history.plot()` at the end. This generates a multi-panel plot of all tracked metrics over time. (Note: When examples are run via `test_all.py`, the `ODYSSNET_DISABLE_PLOT=1` environment variable automatically bypasses the interactive plotting UI).
    *   *Example:*
    ```python
    from odyssnet import TrainingHistory
    history = TrainingHistory()
    for epoch in range(epochs):
        loss = trainer.train_batch(...)
        history.record(loss=loss, lr=current_lr)
    history.plot(title="My Experiment")
    ```
7.  **Imports:** Import `odyssnet` directly (installed via `pip install -e .`). Never use `sys.path.append` hacks.

---

## 🔍 Troubleshooting

### Loss is NaN or Inf

If your experiment produces `Loss nan` / `PPL nan`, enable the built-in diagnosis mode before anything else:

```python
model = OdyssNet(..., debug=True)
```

With `debug=True` the model checks every critical forward-pass operation (linear recurrence, memory feedback, activation, StepNorm, Hebbian correlation and accumulation) and raises a `RuntimeError` at the first operation that produces a non-finite value, with the operation name and step index. `debug=True` also automatically enables `torch.autograd.set_detect_anomaly(True)`, so backward-pass NaN is caught with a full stack trace at no extra setup cost. Overhead is zero when `debug=False`.

### Training Not Converging

If your model trains but doesn't converge or gets stuck:

#### 1. Use TrainingHistory for Visual Diagnosis

Track and visualize all key metrics to identify patterns:

```python
from odyssnet import TrainingHistory

history = TrainingHistory()

for epoch in range(epochs):
    loss = trainer.train_batch(x, y, thinking_steps=10)
    acc = evaluate_accuracy(...)
    lr = trainer.optimizer.param_groups[0]['lr'] if hasattr(trainer.optimizer, 'param_groups') else trainer.initial_lr

    history.record(loss=loss, accuracy=acc, lr=lr)

# Visual inspection reveals patterns
history.plot(title="Training Diagnosis")
# Or save for later analysis
history.plot(save_path="diagnosis/training.png", title="Debug Run")
```

**What to look for:**
- **Flat loss:** May need more thinking steps, different initialization, or learning rate adjustment
- **Oscillating loss:** Reduce learning rate or enable gradient persistence
- **Sudden spikes:** Check for batch corruption or use anomaly_hook to catch them

#### 2. Use trainer.get_diagnostics() for Runtime Monitoring

Monitor training health in real-time:

```python
for epoch in range(epochs):
    loss = trainer.train_batch(x, y, thinking_steps=10)

    # Get comprehensive diagnostics (add debug=True for detailed stats)
    diag = trainer.get_diagnostics()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}")
        print(f"  Step count: {diag['step_count']}")
        print(f"  Last loss: {diag['last_loss']:.6f}")

# For detailed debugging, use debug=True
if need_detailed_analysis:
    diag = trainer.get_diagnostics(debug=True)
    # Now includes gradient_stats, persistent_grads_active, anomaly_tracking,
    # loss_tracking, scaler_state, and detailed optimizer per-param stats
```

**Key metrics to monitor:**
- **last_loss trend:** Sustained increase suggests instability
- **gradient_stats:** Large norm swings can indicate optimization difficulty
- **step_count progression:** Confirms stable optimizer stepping cadence

**Debug mode additions:**
- **gradient_stats:** Per-parameter gradient norms and means (min/max/std)
- **persistent_grads_active:** Number of parameters with persistent gradients
- **anomaly_tracking:** EWMA, variance, and plateau detection state
- **loss_tracking:** Recent losses and buffer statistics

#### 4. Use anomaly_hook for Automated Intervention

Set up intelligent automated responses to training anomalies:

```python
def handle_anomaly(anomaly_type, loss_val):
    """Called automatically on training anomalies."""

    if anomaly_type == "spike":
        # Violent loss surge (possible gradient explosion)
        print(f"⚠️  SPIKE detected! Loss: {loss_val:.4f}")
        # Could reduce LR, reload checkpoint, etc.

    elif anomaly_type == "plateau":
        # Loss stagnated over a window
        print(f"🔄 PLATEAU detected. Triggering escape...")

    elif anomaly_type == "increase":
        # Loss increased from previous step (happens every time loss goes up)
        # Useful for custom patience counters or early stopping
        global patience_counter
        patience_counter += 1
        if patience_counter > 50:
            print(f"⛔ 50 consecutive increases. Early stopping.")
            raise KeyboardInterrupt

# Initialize trainer with hook
patience_counter = 0
trainer = OdyssNetTrainer(
    model,
    anomaly_hook=handle_anomaly
)

# Now train — anomalies trigger automatic responses
for epoch in range(1000):
    loss = trainer.train_batch(x, y, thinking_steps=10)
```

**Anomaly types:**
- **"spike":** Sudden violent surge in loss (exploding gradient)
- **"plateau":** Loss stagnated and barely moving over a window
- **"increase":** Loss strictly greater than previous step (fired every time, even 0.0001 increase)

### Loss Oscillating or Unstable

If loss oscillates or training is unstable:

1. **Enable gradient persistence** for smoother optimization:
   ```python
   trainer = OdyssNetTrainer(model, gradient_persistence=0.1)
   ```

2. **Use AdamW with a lower explicit learning rate** (bypasses Prodigy):
   ```python
   trainer = OdyssNetTrainer(model, lr=1e-4)
   ```

3. **Try different initialization** if using tiny networks:
   ```python
   model = OdyssNet(..., weight_init='xavier_uniform', activation='gelu')
   ```

### Model Not Learning (Loss Stuck)

If loss doesn't decrease at all:

1. **Verify data preprocessing:** Check that inputs/targets are properly normalized and on correct device
2. **Increase thinking steps:** Model may need more temporal depth
   ```python
   trainer.train_batch(x, y, thinking_steps=20)  # Was 10
   ```
3. **Check initialization:** For very small networks (<10 neurons), try:
   ```python
   model = OdyssNet(..., weight_init='xavier_uniform', activation='gelu')
   ```
4. **Use anomaly_hook and adjust lr/steps dynamically based on diagnostics.**

### Performance Issues

If training is too slow:

1. **Enable TF32 on Ampere+ GPUs:**
   ```python
   import torch
   torch.set_float32_matmul_precision('high')
   ```

2. **Compile the model** (PyTorch 2.0+):
   ```python
   model.compile()
   ```

3. **Use gradient accumulation** instead of larger batches:
   ```python
   # Simulates batch_size=128 with batch_size=32
   trainer.train_batch(x, y, thinking_steps=10, gradient_accumulation_steps=4)
   ```

### Memory Issues

If running out of VRAM:

1. **Reduce batch size** and use gradient accumulation
2. **Enable gradient checkpointing:**
   ```python
   model = OdyssNet(..., gradient_checkpointing=True)
   ```
3. **Use vocab projection** for high-dimensional inputs:
   ```python
   # Instead of num_neurons=784 for MNIST
   model = OdyssNet(num_neurons=10, vocab_size=[784, 10])
   ```

## 🔧 Library Contributions (`odyssnet/`)

When modifying the library itself (not examples), follow these additional rules:

### Tests
- Every change to a public interface must have a corresponding test update under `tests/`.
- The test suite mirrors `odyssnet/`: `tests/core/`, `tests/training/`, `tests/utils/`. Place new tests in the matching subdirectory.
- New behavior requires new test cases. Changed signatures require updated tests. Deleted code requires removed orphaned tests.
- Run `python -m pytest tests/` or just `pytest tests/`(but not recommended as it will not see your env setup) after every change to confirm the suite stays green.

### Documentation
- Every public API change must be reflected in the relevant markdown files (`docs/LIBRARY.md`, `CONTRIBUTING.md`).
- Document what the system *is*, not what it *was*. Version history belongs in `CHANGELOG.md` only.

### Code Style
- Comments explain *why*, not *what* — the code already says what.
- Use precise, professional language. Avoid filler phrases ("simply", "just", "obviously").
- Do not use comments or docs as changelog.

---

## 🚀 Checklist

### All contributions
- [ ] Tests pass (`python -m pytest tests/`or `pytest tests/`)
- [ ] No `sys.path.append` hacks — imports use `from odyssnet import ...` directly

### Library changes (`odyssnet/`)
- [ ] Corresponding test added/updated under `tests/`
- [ ] Documentation updated in relevant markdown files (docs/LIBRARY.md, CONTRIBUTING.md)

### New/modified example scripts (`examples/`)
1.  [ ] **Does your script call `set_seed(42)` at the START of `main()`?** (MANDATORY for reproducibility)
2.  [ ] **Does `OdyssNetTrainer` receive an explicit `lr`?** (e.g. `lr=1e-4`). The default `lr=None` activates Prodigy, which adapts LR online and breaks byte-for-byte reproducibility. Examples must pin a float lr.
3.  [ ] Did you place it in the correct folder (`examples/` for core validations, `examples/advanced/` for complex tasks)?
4.  [ ] Are you using `OdyssNetTrainer`?
5.  [ ] Did you select the correct `activation`, `weight_init`, and `gate` setup? (Default `resonant` + `gate=None` is fine for most tasks.)
6.  [ ] If you set `hebb_type`, did you review the **Hebbian Optimizer Contract** above and confirm weight decay is not applied to the Hebbian group?
7.  [ ] Does it converge reliably? (If you see `Loss nan`, see **Troubleshooting** above.)
8.  [ ] Does the terminal output clearly explain what is happening?
9.  [ ] Does the script use `TrainingHistory` and call `history.plot()` at the end?
10. [ ] Are file paths relative to `__file__`, not hardcoded?

Welcome to the Order of the Algorithm. Let's code Time.
