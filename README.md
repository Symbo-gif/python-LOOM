# loom

**Native Python Mathematical Computing Framework**

> [!NOTE]
> **Project Status:** Phase 9 - Production Quality ✅ COMPLETE  
> **Build Status:** Passing 293 tests (100% success rate)  
> **Version:** 0.9.0

---

## Overview

loom is a **100% native Python library** that unifies numerical computing (NumPy), symbolic algebra (SymPy), and scientific computing (SciPy) into a single cohesive framework with zero external dependencies.

### Key Features

- **Pure Python Core**: Zero external dependencies (optional accelerators available).
- **Unified Tensor Type**: Lazy evaluation via computation DAG with NumericBuffer storage.
- **Complete Linear Algebra**: SVD, QR, LU, Cholesky, Eigen, `expm`, `sqrtm`, `logm`, `outer`.
- **Symbolic Mathematics**: Differentiation, integration, equation solving.
- **Scientific Computing**: FFT, convolution, digital filters, ODE solvers, optimization.
- **Statistical Analysis**: Distributions, t-tests, chi-square with proper p-values.
- **Spatial Algorithms**: KDTree, ConvexHull, sparse matrices.
- **Field Tensors**: N-dimensional sampling with N-linear interpolation.
- **Accelerated Backends**: CPU (default), Numba (JIT), Cython, CUDA (GPU).

---

## Package Structure

```
loom/
├── core/            # ✅ Tensor, Shape, DType, Symbol
├── ops/             # ✅ Arithmetic, reduction, indexing, complex
├── random/          # ✅ RNG (PCG-like)
├── numeric/         # ✅ NumericBuffer storage
├── linalg/          # ✅ Complete linear algebra suite
├── symbolic/        # ✅ AST, differentiation, integrate, solve
├── optimize/        # ✅ BFGS, Nelder-Mead, root-finding
├── integrate/       # ✅ ODE solvers, quadrature
├── interpolate/     # ✅ Splines, linear
├── signal/          # ✅ FFT, convolution, filters, windows
├── special/         # ✅ Gamma, beta, erf, gammainc
├── sparse/          # ✅ CSR, COO matrices
├── spatial/         # ✅ KDTree, ConvexHull
├── field/           # ✅ N-D field tensors with interpolation
├── agent/           # ✅ Daemon orchestration, Recipes
├── stats/           # ✅ Distributions, tests with p-values
├── backend/         # ✅ CPU, Numba, Cython, CUDA backends
├── io/              # ✅ Save/load .tfdata
└── utils/           # ✅ Helpers and validators
```

---

## Quick Start

```python
import loom as tf

# Create tensors (default dtype is float64)
a = tf.array([[1, 2], [3, 4]])
b = tf.ones((2, 2))

# Linear algebra
import loom.linalg as la
u, s, vh = la.svd(a)
exp_a = la.expm(a)
log_a = la.logm(exp_a)  # Roundtrip
outer = la.outer([1, 2, 3], [4, 5])

# Statistics with proper p-values
import loom.stats as stats
chi, p_val = stats.chisquare([10, 20, 30])

# Field sampling (N-dimensional)
from loom.field import FieldTensor
ft = FieldTensor(tf.randn((10, 10, 10)))
value = ft.sample([5.5, 3.2, 7.8])  # 3D interpolation

# Backend selection (optional accelerators)
from loom.backend import set_backend, available_backends
print(available_backends())  # ['cpu', 'numba'] or more
set_backend('numba')  # Use JIT if available
```

---

## Version History

| Version | Phase | Highlights |
|---------|-------|------------|
| 0.9.0 | Phase 9 | logm, outer, rect LU, N-D field, chi-sq p-value, backends |
| 0.8.0 | Phase 8 | NumericBuffer, Stats, integrate/solve, expm/sqrtm, filters |
| 0.7.0 | Phase 7 | Field Tensors, Agent Orchestration |
| 0.6.0 | Phase 6 | Sparse & Spatial Algorithms |
| 0.5.0 | Phase 5 | Signal & Special Functions |
| 0.4.0 | Phase 4 | Optimization & Integration |
| 0.3.0 | Phase 3 | Symbolic Core |
| 0.2.0 | Phase 2 | Linear Algebra |
| 0.1.0 | Phase 1 | Core Tensors |

---

## License

Apache 2.0
