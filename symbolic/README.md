# Symbolic Module (Phase 3 + Phase 8)

This module implements the symbolic mathematics engine for loom. It provides a native Python alternative to SymPy, seamlessly integrated with the `Tensor` class.

## Status: ✅ COMPLETE

## Features

- **Symbolic Expressions**: Full AST including `SymbolExpr`, `NumericExpr`, `BinOpExpr`, `FuncExpr`.
- **Simplification**: Algebraic simplification rules (e.g., `x + 0 -> x`, `x * 1 -> x`).
- **Automatic Differentiation**: Forward-mode differentiation via expression tree traversal.
- **Symbolic Integration** (Phase 8): `integrate(expr, var)` for polynomials, trig, exp.
- **Equation Solving** (Phase 8): `solve(expr, var)` for linear and quadratic equations.

## Usage

```python
import loom as tf
from loom.core.tensor import Symbol
from loom.symbolic import integrate, solve

x = Symbol("x")

# Differentiate
expr = x**2 + 2*x + 1
df = expr.diff(x)  # 2*x + 2

# Substitute
val = df.subs("x", 3).evaluate()  # 8.0

# Simplify
simp = (x * 1 + 0).simplify()  # x

# Integrate (Phase 8)
integral = integrate(x**2, x)  # x^3 / 3

# Solve (Phase 8)
solutions = solve(x**2 - 4, x)  # [2.0, -2.0]
```

## Implementation Details

The symbolic engine uses a lightweight, pure Python AST. Expressions are immutable (frozen dataclasses). Operations on symbolic Tensors build new expression trees that can be evaluated, differentiated, integrated, or solved.

### Supported Integration Rules
- Constants: `∫c dx = c*x`
- Variables: `∫x dx = x²/2`
- Powers: `∫x^n dx = x^(n+1)/(n+1)`
- Trigonometric: `∫sin(x) dx = -cos(x)`, `∫cos(x) dx = sin(x)`
- Exponential: `∫exp(x) dx = exp(x)`

### Supported Equation Types
- Linear: `ax + b = 0`
- Quadratic: `ax² + bx + c = 0` (returns two solutions via quadratic formula)
