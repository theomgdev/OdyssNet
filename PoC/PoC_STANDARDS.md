# 🧪 PoC & Experiments Standards

This document outlines the standards and best practices for contributing Proof-of-Concept (PoC) scripts and innovative experiments to the OdyssNet project.

OdyssNet 2.0 relies on a highly modular library structure. To ensure long-term maintainability and performance, all new contributions must adhere to these guidelines.

---

## 📂 Directory Structure

We distinguish between **Core Validations** and **Feature Experiments**.

### 1. `PoC/` (Root)
*   **Purpose:** Contains minimal, "hello world" style scripts that validate the core laws of OdyssNet physics.
*   **Examples:** `convergence_identity.py` (Can signals pass?), `convergence_gates.py` (Can it solve XOR?).
*   **Rule:** Scripts here should be extremely simple, fast, and prove a fundamental property of the architecture.

### 2. `PoC/experiments/`
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
If you need a new feature (e.g., a new loss function or a custom metric), extend `OdyssNetTrainer` or pass arguments to it. If the library is missing a critical feature, **implement it in the library first**, then use it in your PoC.

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

### Gate Optimizer Contract (ChaosGrad)
When `gate` parameters exist, ChaosGrad handles them in a dedicated `gates` group.
*   `gate_lr_mult`: Independent LR multiplier for gate params.
*   `gate_decay`: Independent weight decay for gate params.
*   Default behavior keeps gates stable (`gate_decay=0.0`) and matches base group speed (`gate_lr_mult=1.0`).
*   Recommended start for stronger gate adaptation: `gate_lr_mult=1.1` to `1.3`, `gate_decay=0.0`.

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

---

## ⚡ Hardware Optimization

### 1. 8-Bit Optimizers (bitsandbytes)
OdyssNet V2.0 automatically uses `bitsandbytes` 8-bit AdamW if a CUDA GPU is detected. This reduces VRAM usage by ~75% for optimizer states.
*   **Default:** Enabled.
*   **Disable:** Set `os.environ["NO_BNB"] = "1"` before importing `odyssnet`.
*   **Debug:** Set `os.environ["VERBOSE_BNB"] = "1"` to see loading logs.

### 2. TensorFloat-32 (TF32)
Always enable TF32 on Ampere+ GPUs for free speedup.
```python
import torch
torch.set_float32_matmul_precision('high')
```

### 3. Compilation
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
You can pass an `anomaly_hook` to the `OdyssNetTrainer` to automate recovery (e.g., triggering plateau escapes).

```python
def my_hook(anomaly_type, loss_val):
    if anomaly_type == "plateau":
        print("Triggering plateau escape!")
        trainer.trigger_plateau_escape()

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

1.  **Reproducibility:** Always clear the seed if possible, or accept that chaos is part of the design.
2.  **Visuals:** Your PoC should print a cool visualization. Don't just print "Loss: 0.01". Print the timeline.
    *   *Example:* `t=05 | Input: 1 | Output: 0.99 🟢`
3.  **Comments:** Explain *why* you chose a specific setup.
    *   *Example:* `# GAP=3 allows the model time to digest the previous bit.`
4.  **File Paths:** Never use hardcoded absolute paths or assume the CWD. Always construct paths relative to the script file.
    *   *Example:* `DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'file.txt')`
5.  **Checkpointing:** Always use the library's `save_checkpoint`, `load_checkpoint`, and `transplant_weights` functions from `odyssnet.utils.odyssstore`. Do NOT write custom checkpoint code. If the library is missing a feature, extend the library instead.
    *   *Example:* `from odyssnet import save_checkpoint, load_checkpoint, transplant_weights`

---

## 🚀 Checklist for New Contributors

Before submitting a new PoC:
1.  [ ] Did you place it in the correct folder (`PoC/` vs `PoC/experiments/`)?
2.  [ ] Are you using `OdyssNetTrainer`?
3.  [ ] Did you select the correct `activation`, `weight_init`, and `gate` setup? (Default `resonant` + `gate=None` is fine for most tasks.)
4.  [ ] Does it converge reliably?
5.  [ ] Does the terminal output clearly explain what is happening?

Welcome to the Order of the Algorithm. Let's code Time.
