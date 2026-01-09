# loom Numeric Module

**Optimized dense array storage using `array.array`**

## Status: ✅ COMPLETE (Phase 8)

## Features

- **NumericBuffer Class**:
    - Uses `array.array` for efficient memory storage of float32, float64, int32, int64.
    - Complex number support via twin arrays (real + imag).
    - Auto-upgrade to `complex128` when complex data is detected in real-typed inputs.
    - Efficient slicing, copying, and element access.

- **Integration with Tensor**:
    - `Tensor._dense_data` uses `NumericBuffer` for all numeric storage.
    - Default dtype changed to `float64` for scientific computing parity.

## Supported DTypes

| DType | TypeCode | Description |
|-------|----------|-------------|
| `float32` | `f` | 32-bit float |
| `float64` | `d` | 64-bit float (default) |
| `int32` | `i` | 32-bit integer |
| `int64` | `q` | 64-bit integer |
| `complex64` | twin `f` | 32-bit complex |
| `complex128` | twin `d` | 64-bit complex |

## Usage

```python
from loom.numeric import NumericBuffer
from loom.core.dtype import DType

# Create a buffer
buf = NumericBuffer([1.0, 2.0, 3.0], DType.FLOAT64)
print(buf.tolist())  # [1.0, 2.0, 3.0]

# Pre-allocate
buf2 = NumericBuffer(100, DType.FLOAT64)  # 100 zeros

# Complex auto-detection
buf3 = NumericBuffer([1+2j, 3+4j], DType.FLOAT64)  # Auto-upgrades to COMPLEX128
```
