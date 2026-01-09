# loom Linear Algebra (`loom.linalg`)

**Complete Linear Algebra Suite**

## Status: ✅ COMPLETE (Phase 2 + Phase 8 + Phase 9)

## Decompositions

| Function | Description | Shapes |
|----------|-------------|--------|
| `lu(A)` | LU decomposition with pivoting | m×n → P(m×m), L(m×k), U(m×n) |
| `qr(A)` | QR decomposition | m×n → Q(m×m), R(m×n) |
| `cholesky(A)` | Cholesky for SPD matrices | n×n → L(n×n) |
| `svd(A)` | Singular Value Decomposition | m×n → U, S, Vh |
| `eig(A)` | Eigenvalues/vectors | n×n → (eigenvalues, eigenvectors) |
| `eigh(A)` | Hermitian eigenvalues | n×n → (eigenvalues, eigenvectors) |

## Solvers

| Function | Description |
|----------|-------------|
| `solve(A, b)` | Solve Ax = b |
| `inv(A)` | Matrix inverse |
| `det(A)` | Determinant (with correct sign) |
| `matrix_rank(A)` | Rank via SVD |
| `cond(A)` | Condition number |

## Matrix Functions (Phase 8-9)

| Function | Description | Algorithm |
|----------|-------------|-----------|
| `expm(A)` | Matrix exponential | Scaling & squaring + Padé |
| `sqrtm(A)` | Matrix square root | Denman-Beavers iteration |
| `logm(A)` | Matrix logarithm | Inverse scaling + squaring |

## Matrix Products

| Function | Description |
|----------|-------------|
| `matmul(A, B)` | Matrix multiplication (@ operator) |
| `dot(a, b)` | Dot product |
| `vdot(a, b)` | Vector dot product (conjugates first) |
| `inner(a, b)` | Inner product |
| `outer(a, b)` | Outer product a⊗b |

## Norms and Other

| Function | Description |
|----------|-------------|
| `norm(x)` | Frobenius/Euclidean norm |
| `trace(A)` | Matrix trace |
| `matrix_transpose(A)` | Transpose (also `.T` property) |

## Usage Examples

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
sqrt_A = la.sqrtm(exp_A)
log_exp_A = la.logm(exp_A)  # Roundtrip ≈ A

# Outer product
a = lm.array([1, 2, 3])
b = lm.array([4, 5])
outer_ab = la.outer(a, b)  # 3×2 matrix

# Rectangular LU (Phase 9)
tall = lm.array([[1, 2], [3, 4], [5, 6], [7, 8]])  # 4×2
P, L, U = la.lu(tall)  # Works with rectangular matrices
```
