# loom Special Functions Module

**Gamma, beta, error functions, and incomplete gamma**

## Status: ✅ COMPLETE (Phase 5 + Phase 9)

## Functions

### Gamma Functions
- `gamma(x)`: Gamma function Γ(x) using Lanczos approximation.
- `loggamma(x)`: log(Γ(x)) for numerical stability with large values.
- `beta(a, b)`: Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b).

### Error Functions
- `erf(x)`: Error function (Abramowitz & Stegun approximation).
- `erfc(x)`: Complementary error function 1 - erf(x).

### Incomplete Gamma (Phase 9)
- `gammainc(a, x)`: Regularized lower incomplete gamma P(a, x) = γ(a,x)/Γ(a).
- `gammaincc(a, x)`: Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).

## Usage

```python
from loom.special import gamma, erf, gammainc, gammaincc

# Gamma function
g = gamma(5.0)  # 24.0 (= 4!)

# Error function
e = erf(1.0)  # ~0.8427

# Incomplete gamma (used for chi-square p-values)
p = gammainc(2.0, 1.0)  # P(2, 1) ≈ 0.264
q = gammaincc(2.0, 1.0)  # Q(2, 1) ≈ 0.736

# Tensor inputs supported
import loom as lm
t = lm.array([1.0, 2.0, 3.0])
g_vec = gamma(t)  # [1.0, 1.0, 2.0]
```

## Implementation Notes

### gammainc Algorithm
- For x < a+1: Series expansion (Pearson's method)
- For x >= a+1: Continued fraction (Lentz's algorithm)
- Both methods converge rapidly with tolerance 1e-12

### Chi-Square CDF
The chi-square CDF is computed as:
```
P(χ² < x; k) = gammainc(k/2, x/2)
```
where k is the degrees of freedom. This is used by `stats.chisquare()` to compute proper p-values.
