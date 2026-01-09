# loom Core Module

**Foundation classes for tensor representation and computation**

## Status: ✅ COMPLETE (Phase 1 + Phase 8 Enhancements)

| Component | Status | Description |
|-----------|--------|-------------|
| `tensor.py` | ✅ Complete | Tensor class with DAG evaluation, NumericBuffer storage |
| `dtype.py` | ✅ Complete | DType enum, parse_dtype (default: float64) |
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

### Default DType (Phase 8)
- Changed from `float32` to `float64` for scientific computing parity.
- Matches NumPy/SciPy default behavior.

## Factory Functions

```python
import loom as tf

a = tf.array([[1, 2], [3, 4]])  # From nested list
z = tf.zeros((2, 3))            # 2x3 zeros
o = tf.ones((3,))               # 1D ones
f = tf.full((2, 2), 5.0)        # Filled with 5.0
I = tf.eye(3)                   # 3x3 identity
```

## Test Coverage

- 33 core tests passing
- Broadcasting verified against NumPy semantics
