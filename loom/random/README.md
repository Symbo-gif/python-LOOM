# loom Random Module

**Random number generation with multiple distributions**

---

## Status: ✅ COMPLETE (Phase 1 Weeks 8-9)

---

## Established Facts (Verified)

- LCG-based pseudo-random generator (MINSTD variant)
- Seed-based reproducibility via `seed()`
- Box-Muller transform for normal distribution
- All functions return Tensor objects

---

## Implemented Functions

| Function | Description | Parameters |
|----------|-------------|------------|
| `rand(*shape)` | Uniform [0, 1) | shape dims |
| `randn(*shape)` | Standard normal | shape dims |
| `randint(low, high, size)` | Random integers | bounds, shape |
| `uniform(low, high, size)` | Uniform [low, high) | bounds, shape |
| `normal(loc, scale, size)` | Gaussian | mean, std, shape |
| `exponential(scale, size)` | Exponential | 1/lambda, shape |
| `poisson(lam, size)` | Poisson | rate, shape |
| `choice(a, size, replace)` | Random sample | array, count |
| `permutation(n)` | Random permutation | length |
| `seed(s)` | Set random seed | seed value |

---

## Test Results: 22 tests passing
