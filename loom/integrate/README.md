# loom Integration Module

**ODE solvers and numerical integration**

---

## Status: ✅ IMPLEMENTED (Phase 4)

This module provides native Python implementations for numerical integration and ordinary differential equation solvers, mirroring `scipy.integrate`.

## Numerical Quadrature

- `trapezoid(y, x=None)`: Composite trapezoidal rule.
- `simpson(y, x=None)`: Simpson's 1/3 rule (handles non-even counts via hybrid trapezoid).

## ODE Solvers

- `solve_ivp(fun, t_span, y0, method='RK45')`: General purpose ODE solver.
- `RK4`: Classical 4th order Runge-Kutta (fixed step size).
- `RK45`: Adaptive step size Dormand-Prince method (5th order accurate, 4th order error estimate).

## Example: Lorenz System

```python
import loom as lm
from loom.integrate import solve_ivp

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]

sol = solve_ivp(lorenz, (0, 25), [1, 1, 1], method='RK45')
```
