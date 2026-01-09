# LOOM

**Native Python Mathematical Computing Framework**

> Zero external dependencies. Pure Python with optional Cython/CUDA acceleration.

---

> [!NOTE]
> **Project Status:** Phase 0 - Scaffolding ✅ Complete; Phase 1 Core Operations ✅ Complete (159 tests, 100%)
> **Build Status:** Buildable – core operations available

| Phase | Status | Coverage |
|-------|--------|----------|
| Phase 1: Core Operations | ✅ Complete | 159 tests, 100% |
| Phase 2: Linear Algebra | ⬜ Not Started | - |
| Phase 3: Optimization | ⬜ Not Started | - |
| Phase 4: Signal/Integration | ⬜ Not Started | - |
| Phase 5: Sparse/Statistics | ⬜ Not Started | - |
| Phase 6: Specialized | ⬜ Not Started | - |
| Phase 7: ASTRA Agents | ⬜ Not Started | - |

---

## Established Facts

- **Start Date:** January 2026
- **Python Version:** 3.9+
- **License:** Apache 2.0
- **Dependencies:** Zero (pure Python core)

---

## Goals

- Replace NumPy, SymPy, and SciPy with a unified framework
- Achieve 100% replacement by Week 35
- Support CPU, Cython, and CUDA backends
- Enable hybrid symbolic-numeric workflows

---

## Quick Start (Available after Phase 1)

```python
import loom as tf

# Create tensors
a = tf.Tensor([[1, 2], [3, 4]])
b = tf.Tensor([[5, 6], [7, 8]])

# Lazy operations
c = a + b
result = c.compute()

# Symbolic math
x = tf.Symbol('x')
expr = x**2 + 2*x + 1
df = tf.differentiate(expr, x)
```

---

## Development

```bash
# Install in development mode
make install-dev

# Run tests
make test

# Format code
make format
```

---

## License

Apache 2.0
