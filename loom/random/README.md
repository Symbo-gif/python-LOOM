# loom Random Module

**Random number generation with multiple distributions**

---

## Status: ✅ Production-ready (v1.1.0)

_Validated by the v1.1.0 release suite (random distribution, sampling, and reproducibility tests)._

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

## Reliability

- Covered by statistical checks in `tests/test_random/test_rng.py` and fuzz/stress cases (v1.1.0).
