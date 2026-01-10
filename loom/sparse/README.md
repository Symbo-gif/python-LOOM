# loom Sparse Module

**Sparse matrix formats and operations**

## Status: ✅ Production-ready (v1.1.0)

_Validated by COO/CSR arithmetic tests in the v1.1.0 release suite._

## Features

- **Matrix Formats**:
    - `COOMatrix`: Coordinate format, easy for construction.
    - `CSRMatrix`: Compressed Sparse Row, efficient for arithmetic.
- **Operations**:
    - **Sparse-Dense Multiplication**: `spmul(sparse, dense_tensor)` or `sparse @ dense_tensor`.
    - **Sparse-Sparse Multiplication**: `spmul(csr1, csr2)` for efficient chain products.
    - **Sparse-Sparse Addition**: `spadd(csr1, csr2)`.
    - **Conversions**: `to_csr()`, `to_dense()`.

## Usage Example

```python
import loom as lm
from loom.sparse import COOMatrix

# Create from coordinates
coo = COOMatrix(data=[1, 2], row=[0, 1], col=[1, 0], shape=(2, 2))
csr = coo.to_csr()

# Sparse-Dense product
v = lm.array([10, 20])
res = csr @ v  # [40.0, 10.0]
```
