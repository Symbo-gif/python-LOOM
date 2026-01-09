# loom Core Module

**Foundation classes for tensor representation and computation**

## Status: ✅ COMPLETE (Phase 1 + Phase 8 Enhancements)

| Component | Status | Description |
|-----------|--------|-------------|
| `tensor.py` | ✅ Complete | Tensor class with DAG evaluation, NumericBuffer storage |
| `dtype.py` | ✅ Complete | DType enum, parse_dtype (default: float32) |
| `shape.py` | ✅ Complete | Immutable Shape with broadcasting |
| `dag.py` | ✅ Integrated | DAG evaluation built into tensor.py |

## Key Features

### Tensor Class
- Lazy evaluation via computation DAG (`_op`, `_args`).
- `compute()` recursively evaluates operation chain.
- `tolist()` converts flat data back to nested Python lists.
- `NumericBuffer` storage for efficient memory usage (Phase 8).
- `symbolic_expr` property for symbolic tensor AST access.
- Comparison operators (<, <=, ==, !=, >=, >) return boolean tensors.

### Symbol Class
- Create symbolic variables: `x = Symbol('x')`.
- Supports arithmetic, differentiation, substitution.
- `.diff(var)`, `.subs(name, value)`, `.simplify()`.

### Broadcasting
- Full NumPy-compatible broadcasting.
- Handles scalar, 1D→2D, row×column expansion.
- Row-major (C-style) indexing with correct strides.

### Default DType
- Default dtype is `float32` for memory efficiency.
- Can be configured via `loom.config.set_dtype('float64')` for scientific computing parity.

## Factory Functions

```python
import loom as lm

a = lm.array([[1, 2], [3, 4]])  # From nested list
z = lm.zeros((2, 3))            # 2x3 zeros
o = lm.ones((3,))               # 1D ones
f = lm.full((2, 2), 5.0)        # Filled with 5.0
I = lm.eye(3)                   # 3x3 identity
```

## Test Coverage

- 33 core tests passing
- Broadcasting verified against NumPy semantics
