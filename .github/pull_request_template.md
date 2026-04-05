## Summary

Brief description of what this PR does and why.

## Changes

- 

## Checklist

### All contributions
- [ ] Code follows the project conventions (see [CONTRIBUTING.md](../CONTRIBUTING.md))
- [ ] Tests pass (`python -m pytest tests/`or `pytest tests/`)
- [ ] No `sys.path.append` hacks — imports use `from odyssnet import ...` directly

### Library changes (`odyssnet/`)
- [ ] Corresponding test added/updated under `tests/`
- [ ] Documentation updated in relevant markdown files (LIBRARY.md, CONTRIBUTING.md)

### New/modified example scripts (`examples/`)
- [ ] `set_seed(42)` is called as the first line of `main()`
- [ ] Placed in correct folder (`examples/` for core validations, `examples/advanced/` for complex tasks)
- [ ] Uses `OdyssNetTrainer` (not a manual training loop)
- [ ] Uses `TrainingHistory` to record metrics and calls `history.plot()` at the end
- [ ] File paths are relative to `__file__`, not hardcoded
