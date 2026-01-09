# loom Field Module

**Field tensors with N-dimensional spatial sampling**

## Status: ✅ COMPLETE (Phase 7 + Phase 9 N-D Enhancement)

## Features

### FieldTensor
- Wrapper around Tensor with spatial awareness.
- Continuous sampling at fractional coordinates.
- **N-linear interpolation** for any dimensionality (1D, 2D, 3D, ..., N-D).

### Eigen-Compression
- SVD-based tensor compression.
- `compress_eigen(tensor, rank)`: Compress 2D tensor to low-rank approximation.
- `decompress_eigen(compressed)`: Restore tensor from compressed form.

## Usage Example

```python
import loom as lm
from loom.field import FieldTensor, compress_eigen, decompress_eigen

# 1D Field Sampling
data_1d = lm.array([0, 1, 4, 9, 16])  # x^2 samples
ft_1d = FieldTensor(data_1d)
val = ft_1d.sample([2.5])  # Interpolates between 4 and 9

# 2D Field Sampling (bilinear interpolation)
data_2d = lm.array([[0, 1], [2, 3]])
ft_2d = FieldTensor(data_2d)
val = ft_2d.sample([0.5, 0.5])  # Center value (~1.5)

# 3D Field Sampling (trilinear interpolation)
data_3d = lm.randn((10, 10, 10))
ft_3d = FieldTensor(data_3d)
val = ft_3d.sample([5.5, 3.2, 7.8])  # 3D interpolation

# 4D+ Field Sampling (N-linear interpolation)
data_4d = lm.randn((5, 5, 5, 5))
ft_4d = FieldTensor(data_4d)
val = ft_4d.sample([2.1, 3.4, 1.7, 4.2])  # 4D interpolation

# Compression
data = lm.randn((100, 100))
compressed = compress_eigen(data, rank=10)  # Keep top 10 singular values
restored = decompress_eigen(compressed)  # Approximate reconstruction
```

## Interpolation Algorithm

For N-dimensional fields, uses recursive N-linear interpolation:
1. Find bounding hypercube corners at integer coordinates.
2. Compute fractional position within hypercube.
3. Recursively interpolate along each dimension.

This is equivalent to:
- 1D: Linear interpolation
- 2D: Bilinear interpolation
- 3D: Trilinear interpolation
- N-D: N-linear (multilinear) interpolation
