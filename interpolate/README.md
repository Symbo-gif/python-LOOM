# loom Interpolate Module

**Spline, polynomial, and RBF interpolation**

## Status: ✅ COMPLETE (Phase 5)

## Features

- **1D Interpolation (`interp1d`)**:
  - `kind='linear'`: Fast piecewise linear interpolation.
  - `kind='cubic'`: High-quality cubic spline interpolation.
- **Cubic Splines**: 
  - Natural cubic spline implementation with tridiagonal solver logic.
  - Support for scalar, list, and Tensor inputs.

## Usage Example

```python
import loom as tf
from loom.interpolate import interp1d

x = [0, 1, 2, 3]
y = [0, 1, 0, 1]

f = interp1d(x, y, kind='cubic')
val = f(0.5)  # Interpolated value at 0.5
```
