# loom Backend Module

**Computation backend abstraction with automatic fallback**

## Status: ✅ COMPLETE (Phase 9)

## Available Backends

| Backend | Description | Dependencies | Status |
|---------|-------------|--------------|--------|
| `cpu` | Pure Python | None | ✅ Always available |
| `numba` | JIT compilation | `pip install numba` | Graceful fallback |
| `cython` | C extensions | Build from source | Graceful fallback |
| `cuda` | GPU acceleration | `pip install cupy` | Graceful fallback |

## Usage

```python
from loom.backend import available_backends, set_backend, get_backend

# Check available backends
print(available_backends())  # ['cpu', 'numba', ...]

# Switch backend (graceful fallback if unavailable)
success = set_backend('numba')
if not success:
    print("Numba not available, using CPU")

# Get current backend
backend = get_backend()
print(backend.name)  # 'cpu' or 'numba'
```

## Backend Interface

All backends implement the `Backend` abstract class:

```python
class Backend(ABC):
    @property
    def name(self) -> str: ...
    @property
    def is_available(self) -> bool: ...
    
    def add(self, a, b) -> List[float]: ...
    def mul(self, a, b) -> List[float]: ...
    def matmul(self, a, b, a_shape, b_shape) -> List[float]: ...
    def sum(self, a) -> float: ...
    def exp(self, a) -> List[float]: ...
    def log(self, a) -> List[float]: ...
    def sqrt(self, a) -> List[float]: ...
```

## Custom Backends

Register custom backends for specialized hardware:

```python
from loom.backend import get_backend_manager, Backend

class MyBackend(Backend):
    # Implement all required methods
    ...

manager = get_backend_manager()
manager.register_backend('custom', MyBackend())
set_backend('custom')
```
