# ChaosGrad & BNB Removal Plan

**Objective:** Remove `ChaosGrad` optimizer and all `bitsandbytes` residue from the codebase. Replace the default optimizer with `AdamW`. Preserve all non-optimizer functionality (AMP, gradient accumulation, anomaly hooks, neurogenesis, diagnostics, etc.).

**Justification:** Benchmark results across 4 independent tests (MNIST raw, MNIST embed, sine wave generator, delayed adder) showed ChaosGrad performing at parity or below AdamW while adding ~500 lines of complexity and ~20% wall-clock overhead.

---

## Phase 0 — Pre-flight

Before any changes, create a snapshot branch and confirm the test suite is green.

```bash
git checkout -b remove-chaosgrad
pytest tests/ -q
```

---

## Phase 1 — Delete ChaosGrad

### 1.1 Delete the file

```bash
git rm odyssnet/training/chaos_optimizer.py
```

### 1.2 Delete the test file

```bash
git rm tests/training/test_chaos_optimizer.py
```

---

## Phase 2 — Refactor `odyssnet/training/trainer.py`

This is the highest-risk file. Every change must be verified line-by-line.

### 2.1 Remove ChaosGrad import and initialization

| Line(s) | Current | Action |
|---------|---------|--------|
| 11 | `from .chaos_optimizer import ChaosGrad` | **Delete line** |
| 41 | `self._using_chaos_grad = False` | **Delete line** |
| 49-50 | `if isinstance(optimizer, ChaosGrad): self._using_chaos_grad = True` | **Delete block** |
| 52-53 | `self._init_chaos_grad(model, lr)` | **Replace** with `self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)` |
| 63-70 | `def _init_chaos_grad(self, model, lr):` (entire method) | **Delete method** |

### 2.2 Remove ChaosGrad loss reporting

| Line(s) | Current | Action |
|---------|---------|--------|
| 282-285 | `if self._using_chaos_grad and step_now: chaos_opt = ... chaos_opt.report_loss(loss_val)` | **Delete block** |

### 2.3 Preserve diagonal zeroing (CRITICAL)

ChaosGrad enforced `W.fill_diagonal_(0.0)` on both gradient and weight at every step (lines 442-444 and 492-494 of chaos_optimizer.py). This constraint must be transferred to `train_batch`.

**Insert after the optimizer step** (after current line 276 `self._step_count += 1`):

```python
# Enforce zero diagonal on chaos core weight matrix.
# Self-connections are handled by memory_feedback, not W.
with torch.no_grad():
    for name, param in self.model.named_parameters():
        if name.endswith('.W') and param.dim() == 2 and param.shape[0] == param.shape[1]:
            param.fill_diagonal_(0.0)
```

### 2.4 Refactor `trigger_plateau_escape`

| Line(s) | Current | Action |
|---------|---------|--------|
| 449-454 | Delegates to ChaosGrad | **Replace body** with `pass` or delete the method entirely. If kept, make it a no-op with a deprecation comment. |

### 2.5 Refactor `get_diagnostics`

| Line(s) | Current | Action |
|---------|---------|--------|
| 456-474 | Returns dict with `using_chaos_grad` key and optional ChaosGrad-specific metrics | **Simplify:** remove `using_chaos_grad` key and the `if self._using_chaos_grad:` block. Keep `step_count`, `last_loss`, `current_lr`. |

### 2.6 Refactor `expand` (neurogenesis print)

| Line(s) | Current | Action |
|---------|---------|--------|
| 444-445 | `if self._using_chaos_grad and verbose: print(...)` | **Delete block** |

### 2.7 Remove `cast` import if unused

Line 6 imports `cast` from `typing`. After removing all `cast(ChaosGrad, ...)` calls, check if `cast` is still used elsewhere in the file. If not, remove from import.

### 2.8 Update `__init__` docstring

Replace all references to `ChaosGrad` and `genesis_lr` with `AdamW`:

- Line 24: `"If None, ChaosGrad is used automatically..."` -> `"If None, AdamW is used as default."`
- Line 26: `"Genesis learning rate for ChaosGrad."` -> `"Learning rate for default AdamW optimizer."`

### Verification

```bash
# Review the diff for trainer.py specifically
git diff odyssnet/training/trainer.py

# Inspect context around every changed line
git diff -U5 odyssnet/training/trainer.py
```

---

## Phase 3 — Refactor `odyssnet/utils/neurogenesis.py`

### 3.1 Remove ChaosGrad optimizer migration

| Line(s) | Current | Action |
|---------|---------|--------|
| 219-223 | `from ..training.chaos_optimizer import ChaosGrad` + `isinstance` check + `classify_params` + `ChaosGrad(...)` | **Delete** the ChaosGrad branch. Keep the `else` block (generic optimizer re-init with AdamW fallback). |

The resulting code should be:

```python
try:
    new_opt = optimizer_cls(
        model.parameters(),
        lr=get_arg('lr', 0.001),
        weight_decay=get_arg('weight_decay', 0),
        betas=get_arg('betas', (0.9, 0.999)),
        eps=get_arg('eps', 1e-8)
    )
except Exception as e:
    ...
```

### 3.2 Rename `micro_quiet_8bit` comment

| Line | Current | Action |
|------|---------|--------|
| 74 | `# micro_quiet_8bit init for new connections` | **Replace** with `# micro_quiet init for new connections` |

### Verification

```bash
git diff -U5 odyssnet/utils/neurogenesis.py
```

---

## Phase 4 — Refactor `odyssnet/__init__.py`

| Line(s) | Current | Action |
|---------|---------|--------|
| 5 | `from .training.chaos_optimizer import ChaosGrad` | **Delete line** |
| 14 | `'ChaosGrad',` | **Delete line** |

---

## Phase 5 — Rename `micro_quiet_8bit` init strategy

The name is misleading (implies 8-bit quantization, but it is just `Normal(0, 1e-3)`). Rename to `micro_quiet_warm`.

### 5.1 `odyssnet/core/network.py` (line 305)

```python
# Before
elif strategy == 'micro_quiet_8bit':
# After
elif strategy == 'micro_quiet_warm':
```

### 5.2 Automated rename across the codebase

```bash
# Dry run — inspect matches first
grep -rn "micro_quiet_8bit" --include="*.py" --include="*.md" .

# Apply rename (sed on git bash / WSL)
find . -type f \( -name "*.py" -o -name "*.md" \) \
  -not -path "./.git/*" -not -path "./tmp/*" \
  -exec sed -i 's/micro_quiet_8bit/micro_quiet_warm/g' {} +
```

**Affected files (known):**

| File | Line(s) |
|------|---------|
| `odyssnet/core/network.py` | 305 |
| `odyssnet/utils/neurogenesis.py` | 74 (already updated in Phase 3) |
| `odyssnet/utils/odyssstore.py` | 97, 110, 120-121 |
| `examples/advanced/convergence_mnist_record.py` | 64 |
| `examples/advanced/convergence_mnist_reverse_record.py` | 142, 151 |
| `tests/core/test_network.py` | 94, 398 |

### Verification

```bash
# Confirm no stale references remain
grep -rn "micro_quiet_8bit" --include="*.py" --include="*.md" .
# Should return 0 results

# Verify all renamed references are correct
grep -rn "micro_quiet_warm" --include="*.py" --include="*.md" .

# Diff every affected file
git diff -U3 odyssnet/core/network.py odyssnet/utils/odyssstore.py \
  examples/advanced/convergence_mnist_record.py \
  examples/advanced/convergence_mnist_reverse_record.py \
  tests/core/test_network.py
```

---

## Phase 6 — Remove all BNB / bitsandbytes residue

### 6.1 Automated removal of `NO_BNB` environment lines

```bash
# Remove os.environ lines setting NO_BNB and their preceding comments
find . -type f -name "*.py" -not -path "./.git/*" -not -path "./tmp/*" \
  -exec sed -i '/# Suppress bitsandbytes during tests/d' {} + \
  -exec sed -i '/# Disable BNB for this experiment/d' {} + \
  -exec sed -i '/# Disable bitsandbytes for pure dynamics/d' {} + \
  -exec sed -i '/os\.environ\.\?.*"NO_BNB"/d' {} +
```

**Affected files:**

| File | Line(s) | Content |
|------|---------|---------|
| `tests/conftest.py` | 9-10 | Comment + `os.environ.setdefault("NO_BNB", "1")` |
| `tests/training/test_trainer.py` | 25 | `os.environ.setdefault("NO_BNB", "1")` |
| `tests/training/test_chaos_optimizer.py` | 23 | Already deleted in Phase 1 |
| `tests/utils/test_neurogenesis.py` | 21 | `os.environ.setdefault("NO_BNB", "1")` |
| `tests/utils/test_odyssstore.py` | 15, 95, 188 | `os.environ.setdefault(...)` and `os.environ[...]` |
| `tests/utils/test_history.py` | 12 | `os.environ.setdefault("NO_BNB", "1")` |
| `tests/utils/test_data.py` | 16 | `os.environ.setdefault("NO_BNB", "1")` |
| `examples/advanced/convergence_mnist_record.py` | 9-10 | Comment + `os.environ["NO_BNB"] = "1"` |
| `examples/advanced/convergence_mnist_reverse_record.py` | 34 | `os.environ["NO_BNB"] = "1"` |

### 6.2 Remove bitsandbytes from dependencies

**`requirements.txt`** (line 4):
```bash
sed -i '/^bitsandbytes$/d' requirements.txt
```

**`pyproject.toml`** (line 47):
```bash
sed -i '/"bitsandbytes",/d' pyproject.toml
```

### 6.3 Fix `tests/conftest.py` docstring

| Line | Current | Action |
|------|---------|--------|
| 59 | `"""Trainer with standard AdamW (no bitsandbytes)."""` | **Replace** with `"""Trainer with default AdamW."""` |

### Verification

```bash
# Confirm no BNB references remain
grep -rn "NO_BNB\|VERBOSE_BNB\|bitsandbytes\|bnb" \
  --include="*.py" --include="*.toml" --include="*.txt" --include="*.md" \
  . | grep -v ".git/" | grep -v "tmp/"
# Should return 0 results (excluding CHANGELOG.md history entries)
```

---

## Phase 7 — Refactor `tests/conftest.py`

### 7.1 Remove ChaosGrad fixtures and imports

| Line(s) | Current | Action |
|---------|---------|--------|
| 13 | `from odyssnet.training.chaos_optimizer import ChaosGrad` | **Delete line** |
| 64-66 | `chaos_trainer` fixture (uses ChaosGrad via trainer) | **Delete fixture** (it was identical to `basic_trainer` since both call `OdyssNetTrainer` with default) |
| 79-83 | `chaos_optimizer` fixture | **Delete fixture** |

---

## Phase 8 — Refactor test files

### 8.1 `tests/training/test_trainer.py`

| Line(s) | Current | Action |
|---------|---------|--------|
| 5 | `"chaos config"` in docstring | **Replace** with `"optimizer selection"` |
| 17 | `"get_diagnostics / trigger_plateau_escape"` | **Replace** with `"get_diagnostics"` |
| 28 | `from odyssnet.training.chaos_optimizer import ChaosGrad` | **Delete line** |
| 65-68 | `test_default_optimizer_is_chaos_grad` | **Rewrite:** assert default optimizer is `AdamW`, remove `_using_chaos_grad` assertion |
| 70-75 | `test_custom_optimizer_bypasses_chaos_grad` | **Rewrite:** simplify — remove `_using_chaos_grad` assertion, keep custom optimizer verification |
| 417 | `assert "using_chaos_grad" in diag` | **Delete line** |
| 420-423 | `test_trigger_plateau_escape_runs_without_error` | **Delete test** (or keep if method remains as no-op) |

### 8.2 `tests/utils/test_neurogenesis.py`

| Line(s) | Current | Action |
|---------|---------|--------|
| 12 | `"expand: with ChaosGrad optimizer"` in docstring | **Replace** with `"expand: optimizer state transfer"` |
| 25 | `from odyssnet.training.chaos_optimizer import ChaosGrad` | **Delete line** |
| 43-44 | `ChaosGrad.classify_params(model)` / `ChaosGrad(groups, lr=lr)` helper | **Replace** with `torch.optim.AdamW(model.parameters(), lr=lr)` |
| 210 | `class TestChaosGradExpansion:` | **Rename** to `class TestOptimizerExpansion:` |
| 215, 362, 532 | `assert isinstance(new_opt, ChaosGrad)` | **Replace** with `assert isinstance(new_opt, torch.optim.AdamW)` |

### Verification

```bash
# Confirm no ChaosGrad references remain in tests
grep -rn "ChaosGrad\|chaos_optimizer\|_using_chaos_grad\|chaos_grad" tests/
# Should return 0 results
```

---

## Phase 9 — Refactor example scripts

### 9.1 `examples/convergence_mnist.py` (lines 114-115)

```python
# Before
diag = trainer.get_diagnostics()
lr = diag.get('optimizer', {}).get('avg_effective_lr', 0) * trainer.initial_lr if diag.get('using_chaos_grad') else diag.get('current_lr', 0)

# After
diag = trainer.get_diagnostics()
lr = diag.get('current_lr', 0)
```

### 9.2 `examples/advanced/convergence_mnist_reverse_record.py` (line 142)

```python
# Before
# micro_quiet_8bit is kept intentionally for this tiny-core stability profile.
# After
# micro_quiet_warm is kept intentionally for this tiny-core stability profile.
```

Already handled by Phase 5 sed command. Verify manually.

---

## Phase 10 — Update documentation

### 10.1 `CLAUDE.md`

| Section | Action |
|---------|--------|
| Line 75 (ChaosGrad description) | **Replace** entire `ChaosGrad` subsection with AdamW description |
| Line 79 (`trigger_plateau_escape`) | **Remove** from trainer key methods list |
| Lines 81-83 (chaos_optimizer module) | **Delete** entire subsection |
| Lines 164-166 (NO_BNB / VERBOSE_BNB) | **Delete** entire hardware optimization block about bitsandbytes |

### 10.2 `docs/LIBRARY.md`

| Line(s) | Action |
|---------|--------|
| 9 | Remove `"8-bit support"` from description |
| 85 | Remove ChaosGrad parameter groups reference |
| 199 | Replace `"ChaosGrad is the default and only built-in optimizer"` with `"AdamW is the default optimizer"` |
| 206, 219 | Update code examples (remove ChaosGrad references) |
| 225 | Replace genesis_lr description with standard lr description |
| 272-275 | Remove or rewrite `trigger_plateau_escape` documentation |
| 280-345 | **Delete** entire `ChaosGrad Optimizer` section |

### 10.3 `CONTRIBUTING.md`

| Line(s) | Action |
|---------|--------|
| 35 | Remove `training/chaos_optimizer.py` from directory listing |
| 138-146 | **Delete** Gate/Hebbian optimizer contract sections (ChaosGrad-specific) |
| 212, 220-224 | Replace ChaosGrad references with AdamW |
| 266, 449, 513 | Remove `trigger_plateau_escape` examples |
| 378-423 | Simplify diagnostics sections — remove ChaosGrad-specific metrics |

### 10.4 `README.md` (line 161)

**Delete or rewrite** the ChaosGrad feature bullet point.

### 10.5 `README_TR.md` (line 161)

**Delete or rewrite** the Turkish ChaosGrad feature bullet point.

### 10.6 `CHANGELOG.md`

**Add** a new entry at the top:

```markdown
## [2.3.0] — 2026-04-06

### Removed
- Removed `ChaosGrad` optimizer — replaced with standard `AdamW` as default.
- Removed `bitsandbytes` dependency and all `NO_BNB` environment variable usage.
- Removed `trigger_plateau_escape()` from trainer (was ChaosGrad-specific).
- Renamed `micro_quiet_8bit` init strategy to `micro_quiet_warm`.

### Changed
- Default optimizer is now `torch.optim.AdamW(lr=1e-3, weight_decay=0.01)`.
- Diagonal zeroing of chaos core `W` matrix is now enforced by the trainer.
- `get_diagnostics()` simplified — removed ChaosGrad-specific metrics.
```

Do **not** modify existing CHANGELOG entries (they are historical record).

---

## Phase 11 — Final verification

### 11.1 No stale references

```bash
# Master grep — nothing should match except CHANGELOG.md historical entries and tmp/
grep -rn \
  "ChaosGrad\|chaos_optimizer\|_using_chaos_grad\|_init_chaos_grad\|classify_params\|genesis_lr\|report_loss\|trigger_plateau_escape\|NO_BNB\|VERBOSE_BNB\|bitsandbytes\|micro_quiet_8bit" \
  --include="*.py" --include="*.md" --include="*.toml" --include="*.txt" \
  . | grep -v ".git/" | grep -v "tmp/" | grep -v "CHANGELOG.md"

# Expected: 0 results
```

### 11.2 Import health

```bash
python -c "from odyssnet import OdyssNet, OdyssNetTrainer; print('OK')"
python -c "from odyssnet import ChaosGrad" 2>&1 | grep -q "ImportError" && echo "PASS: ChaosGrad removed"
```

### 11.3 Test suite

```bash
pytest tests/ -q
```

All tests must pass. Expected failures to investigate:
- Tests that asserted `isinstance(optimizer, ChaosGrad)` (should have been updated in Phase 8).
- Tests using `chaos_optimizer` or `chaos_trainer` fixtures (should have been removed in Phase 7).

### 11.4 Example scripts smoke test

```bash
python examples/convergence_identity.py
python examples/convergence_mnist.py
```

Both must run without errors or ChaosGrad-related output.

### 11.5 Diagonal constraint verification

```python
python -c "
import torch
from odyssnet import OdyssNet, OdyssNetTrainer

model = OdyssNet(num_neurons=5, input_ids=[0,1], output_ids=[3,4], device='cpu')
trainer = OdyssNetTrainer(model, device='cpu')
x = torch.randn(2, 2)
y = torch.randn(2, 2)
for _ in range(5):
    trainer.train_batch(x, y, thinking_steps=3)
print('W diagonal:', model.W.data.diagonal())
assert model.W.data.diagonal().abs().sum() == 0, 'FAIL: diagonal not zeroed'
print('PASS: diagonal zeroing works')
"
```

### 11.6 Review the full diff

```bash
git diff --stat
git diff -U5 | less
```

Manually review every hunk before committing.

---

## File Impact Summary

| File | Action | Risk |
|------|--------|------|
| `odyssnet/training/chaos_optimizer.py` | DELETE | None — source of removal |
| `odyssnet/training/trainer.py` | HEAVY EDIT | **HIGH** — diagonal zeroing, default optimizer |
| `odyssnet/utils/neurogenesis.py` | EDIT | Medium — optimizer migration |
| `odyssnet/utils/odyssstore.py` | EDIT | Low — rename only |
| `odyssnet/core/network.py` | EDIT | Low — rename only |
| `odyssnet/__init__.py` | EDIT | Low — remove export |
| `tests/training/test_chaos_optimizer.py` | DELETE | None |
| `tests/training/test_trainer.py` | EDIT | Medium — test rewrites |
| `tests/conftest.py` | EDIT | Low — remove fixtures |
| `tests/utils/test_neurogenesis.py` | EDIT | Medium — test rewrites |
| `tests/core/test_network.py` | EDIT | Low — rename only |
| `tests/utils/test_odyssstore.py` | EDIT | Low — remove env lines |
| `tests/utils/test_history.py` | EDIT | Low — remove env line |
| `tests/utils/test_data.py` | EDIT | Low — remove env line |
| `examples/convergence_mnist.py` | EDIT | Low — simplify diagnostics |
| `examples/advanced/convergence_mnist_record.py` | EDIT | Low — rename + env |
| `examples/advanced/convergence_mnist_reverse_record.py` | EDIT | Low — rename + env |
| `CLAUDE.md` | EDIT | Medium — architecture docs |
| `docs/LIBRARY.md` | EDIT | Medium — API docs |
| `CONTRIBUTING.md` | EDIT | Medium — developer guide |
| `README.md` | EDIT | Low — feature bullet |
| `README_TR.md` | EDIT | Low — feature bullet |
| `CHANGELOG.md` | APPEND | Low — new entry only |
| `requirements.txt` | EDIT | Low — remove line |
| `pyproject.toml` | EDIT | Low — remove line |

**Total: 2 deletions, 23 edits, ~25 files touched.**
