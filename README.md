# LOOM

**A Native Python Mathematical Computing Framework**

LOOM is a zero-dependency mathematical computing library designed as a unified replacement for NumPy, SymPy, and SciPy. Written entirely in pure Python, it provides tensor operations, symbolic mathematics, scientific computing algorithms, and an agent-based computation system—all in a single package.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)

> Current Version: **v1.1.0** (January 2026)

---

## What's New in v1.1.0

- Top-level complex helpers (`conj`, `real`, `imag`, `angle`, `polar`) and numeric buffer stability improvements.
- Agent orchestration and backend switching are production-ready with graceful fallbacks.
- N-D field sampling, chi-square/gammainc parity, and signal/filter fixes validated by **1599 passing tests**.

---

## Features

| Module | Description | Replaces |
|--------|-------------|----------|
| **core** | Tensor class with lazy evaluation via computation DAG | NumPy arrays |
| **ops** | Tensor arithmetic, reductions, indexing, complex helpers | NumPy ufuncs |
| **symbolic** | Symbolic expressions, differentiation, integration, equation solving | SymPy |
| **linalg** | Matrix decompositions (LU, QR, SVD, Cholesky), eigenvalues, matrix functions | NumPy/SciPy linalg |
| **optimize** | Minimization (BFGS, Nelder-Mead), root-finding (bisection, Newton, Brent) | SciPy optimize |
| **integrate** | ODE solvers (RK4, RK45), numerical quadrature (trapezoid, Simpson) | SciPy integrate |
| **signal** | FFT, convolution, digital filters, window functions | SciPy signal |
| **special** | Gamma, beta, error functions, incomplete gamma | SciPy special |
| **stats** | Distributions, statistical tests, summary metrics | SciPy stats |
| **sparse** | COO and CSR sparse matrix formats | SciPy sparse |
| **spatial** | KD-Tree, convex hull, distance metrics | SciPy spatial |
| **interpolate** | 1D linear and cubic spline interpolation | SciPy interpolate |
| **random** | Random number generation with multiple distributions | NumPy random |
| **field** | Field tensors with N-dimensional spatial sampling | Custom |
| **agent** | Daemon-driven async computation and task orchestration | Custom |
| **backend** | Computation backends (CPU, Numba, Cython, CUDA) | Custom |

---

## Installation

```bash
# From source
git clone https://github.com/Symbo-gif/python-LOOM.git
cd python-LOOM
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

---

## Quick Start

```python
import loom as lm

# Create arrays (NumPy-like)
a = lm.array([[1, 2], [3, 4]])
b = lm.ones((2, 2))
c = a + b
print(c.tolist())  # [[2.0, 3.0], [4.0, 5.0]]

# Random numbers
r = lm.randn(3, 3)

# Zeros and ones
z = lm.zeros((2, 3))
o = lm.ones((4,))
I = lm.eye(3)
```

### Symbolic Computation

```python
from loom.core import Symbol
from loom.symbolic import integrate, solve

x = Symbol("x")

# Differentiate
expr = x**2 + 2*x + 1
df = expr.diff(x)  # 2*x + 2

# Substitute and evaluate
val = df.subs("x", 3).evaluate()  # 8.0

# Integrate
integral = integrate(x**2, x)  # x^3 / 3

# Solve equations
solutions = solve(x**2 - 4, x)  # [2.0, -2.0]
```

### Linear Algebra

```python
import loom as lm
import loom.linalg as la

A = lm.array([[1.0, 2.0], [3.0, 4.0]])

# Decompositions
P, L, U = la.lu(A)
Q, R = la.qr(A)
U, S, Vh = la.svd(A)

# Matrix functions
exp_A = la.expm(A)
sqrt_A = la.sqrtm(A)
log_A = la.logm(exp_A)

# Solvers
b = lm.array([5.0, 6.0])
x = la.solve(A, b)
A_inv = la.inv(A)
det_A = la.det(A)
```

### Optimization

```python
from loom.optimize import minimize, brentq

# Minimize a function
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

result = minimize(rosenbrock, [0, 0], method='BFGS')
print(result.x)  # [1.0, 1.0]

# Find roots
root = brentq(lambda x: x**3 - 2*x - 5, 2, 3)
```

### ODE Integration

```python
from loom.integrate import solve_ivp

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

sol = solve_ivp(lorenz, (0, 25), [1, 1, 1], method='RK45')
```

### Signal Processing

```python
from loom.signal import fft, convolve, hamming

# FFT
data = [1, 0, 1, 0]
freq = fft(data)

# Convolution
a = [1, 2, 3]
v = [1, 1]
c = convolve(a, v, mode='same')

# Window functions
w = hamming(64)
```

### Agent-Based Computation

```python
from loom.agent import ComputationDaemon, Supervisor
import loom as lm

# Async execution
daemon = ComputationDaemon()
daemon.start()
task_id = daemon.submit(lambda x, y: x + y, lm.ones(5), lm.ones(5))
result = daemon.get_result(task_id)
daemon.stop()

# Recipe-based orchestration
supervisor = Supervisor()
recipe = {
    "steps": [
        {"name": "s1", "op": "add", "args": [lm.array([1]), lm.array([2])]},
        {"name": "s2", "op": "mul", "args": ["$s1", lm.array([10])]}
    ]
}
results = supervisor.run_recipe(recipe)  # {'s1': [3], 's2': [30]}
```

---

## Project Structure

```
python-LOOM/
├── loom/                   # Main package
│   ├── core/               # Tensor, Shape, DType classes
│   ├── symbolic/           # Symbolic math engine
│   ├── linalg/             # Linear algebra
│   ├── optimize/           # Optimization algorithms
│   ├── integrate/          # ODE solvers, quadrature
│   ├── signal/             # FFT, filters, convolution
│   ├── special/            # Special functions
│   ├── stats/              # Statistics and distributions
│   ├── sparse/             # Sparse matrices
│   ├── spatial/            # Spatial algorithms
│   ├── interpolate/        # Interpolation
│   ├── random/             # Random number generation
│   ├── field/              # Field tensors
│   ├── agent/              # Async computation, daemons
│   ├── backend/            # Computation backends
│   ├── ops/                # Tensor arithmetic, reductions, indexing, complex ops
│   ├── numeric/            # Numeric storage
│   ├── io/                 # File I/O
│   └── utils/              # Utilities
├── tests/                  # Test suite
├── pyproject.toml          # Package configuration
├── Makefile                # Build automation
└── README.md               # This file
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
make test

# Run tests with coverage
make test-cov

# Type checking
make lint

# Format code
make format
```

---

## Design Principles

1. **Zero Dependencies**: Pure Python core with no external requirements
2. **Unified Interface**: Single `Tensor` type for all numeric and symbolic computation
3. **Lazy Evaluation**: Operations build a computation DAG, evaluated on demand
4. **Backend Flexibility**: Optional accelerated backends (Numba, Cython, CUDA)
5. **Scientific Parity**: API familiarity for NumPy/SciPy/SymPy users

---

## Performance

LOOM supports multiple computation backends:

- **Pure Python**: Zero dependencies, works everywhere
- **Numba JIT**: 10-50x CPU speedup (optional)
- **CUDA GPU**: 100-1000x GPU speedup (coming in v1.2)

### Quick Start with Acceleration

```bash
# Install with Numba support
pip install loom numba
```

```python
# Use acceleration
import loom
loom.set_backend('auto')  # Automatically uses Numba if available

# Operations are now 10-50x faster
A = loom.randn(1000, 1000)
B = loom.randn(1000, 1000)
C = loom.matmul(A, B)  # Fast!
```

See [Performance Guide](docs/backends.md) for details.

### Benchmarks

| Operation | Size | Pure Python | Numba | Speedup |
|-----------|------|-------------|-------|---------|
| Matrix Multiply | 1000×1000 | 120s | 2.3s | 52x |
| LU Decomposition | 500×500 | 45s | 1.2s | 37x |
| Element-wise | 1000×1000 | 8s | 0.3s | 27x |

---

## NOTE

**pure python** can be 10-100x slower than NumPy for most operations.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_core/ -v
pytest tests/test_linalg/ -v

# Run with coverage
pytest tests/ --cov=loom --cov-report=html
```

**Current Status**: 1599 tests passing

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

Copyright 2025 Michael Maillet, Damien Davison, Sacha Davison

---

## Contributing

Contributions are welcome! Please see the module-specific README files in `loom/*/README.md` for implementation details and coding guidelines.
