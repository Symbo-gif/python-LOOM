# loom Test Suite

**Comprehensive tests for all modules**

---

## Status

| Test Directory | Module | Tests Passing | Status |
|----------------|--------|---------------|--------|
| `test_core/` | core | 30 | ✅ Complete (Phase 1) |
| `test_ops/` | ops | 126 | ✅ Complete (Phase 1) |
| `test_random/` | random | 22 | ✅ Complete (Phase 1) |
| `test_linalg/` | linalg | 0 | ⬜ Phase 2 Next |
| `test_symbolic/` | symbolic | 0 | ⬜ Phase 3 |
| `integration/` | all | 1 | ✅ Basic |
| `benchmark/` | all | 0 | ⬜ Phase 3 |

**Total: 179 tests passing (100% success rate)**

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage (requires pytest-cov)
pytest tests/ --cov=src/loom --cov-report=html

# Run specific test file
pytest tests/test_core/test_tensor.py -v
```

---

## Stress Tests Included

- Deep computation DAGs (500+ nodes)
- Large broadcasting (10,000+ elements)
- Statistical validation of RNG distributions
- Complex number arithmetic verification

---

## Test Organization

- Each module has corresponding test_<module>/ directory
- Target: 100% pass rate and 95%+ code coverage for pure Python core

## Reference Documentation

- loom-week-by-week.md (Test infrastructure)
