# loom Optimization Module

**Minimization, root-finding, and constrained optimization**

---

## Status: ✅ IMPLEMENTED (Phase 4)

This module provides native Python implementations for various optimization algorithms, mirroring `scipy.optimize`.

## Root Finding

- `bisect(f, a, b)`: Robust bisection method.
- `newton(f, x0, fprime=None)`: Newton-Raphson method (handles vector systems).
- `brentq(f, a, b)`: Brent's robust hybrid root finder (recommended for scalar functions).

## Minimization

- `minimize(fun, x0, method='BFGS')`: General purpose minimization.
- `BFGS`: Quasi-Newton method using Broyden–Fletcher–Goldfarb–Shanno algorithm.
- `Nelder-Mead`: Simplex-based derivative-free optimization.

## Integration with Symbolic Core

Newton's method and gradient-based optimizers are designed to work with loom's auto-differentiation engine.

```python
import loom as lm
from loom.optimize import minimize

def rosen(x):
    return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

res = minimize(rosen, [0, 0], method='BFGS')
print(res.x) # [1.0, 1.0]
```
