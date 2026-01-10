# Copyright 2025 Michael Maillet, Damien Davison, Sacha Davison
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LOOM - A Native Python Mathematical Computing Framework

LOOM is a zero-dependency mathematical computing library designed as a singular
replacement for NumPy, SymPy, and SciPy. It provides:

- **Tensors**: NumPy-compatible array operations with lazy evaluation
- **Symbolic Math**: SymPy-compatible symbolic computation
- **Scientific Computing**: SciPy-compatible algorithms (optimization, integration, etc.)
- **Agent System**: Daemon-driven async computation and task orchestration
- **Accelerated Backends**: CPU, Numba, Cython, and CUDA support

Key Design Principles:
- Zero external dependencies (pure Python core)
- Unified Tensor type for numeric and symbolic computation
- Lazy evaluation via computation DAG
- Seamless backend switching for performance

Example Usage:
    >>> import loom as lm
    >>> 
    >>> # Create arrays (like NumPy)
    >>> a = lm.array([[1, 2], [3, 4]])
    >>> b = lm.ones((2, 2))
    >>> c = a + b
    >>> 
    >>> # Symbolic computation (like SymPy)
    >>> from loom.core import Symbol
    >>> x = Symbol('x')
    >>> expr = x**2 + 2*x + 1
    >>> df = expr.diff(x)  # 2*x + 2
    >>> 
    >>> # Random number generation
    >>> r = lm.randn(3, 3)
    >>> 
    >>> # Linear algebra
    >>> import loom.linalg as la
    >>> Q, R = la.qr(a)
"""

from loom.__version__ import __version__, __version_info__

# Core tensor types and factory functions
from loom.core.tensor import Tensor, Symbol, array, zeros, ones, full, eye
from loom.core.shape import Shape, broadcast_shapes
from loom.core.dtype import DType, parse_dtype

# Complex number operations (top-level convenience functions)
from loom.ops.complex_ops import conj, real, imag, angle, polar

# Random number generation (top-level convenience functions)
from loom.random import (
    seed, rand, randn, randint, uniform, normal, exponential, poisson, choice, permutation
)


def matmul(x1, x2):
    """
    Matrix multiplication.
    
    Uses active backend for acceleration.
    
    Args:
        x1: First tensor
        x2: Second tensor
        
    Returns:
        Result of matrix multiplication x1 @ x2
    
    Example:
        >>> import loom
        >>> loom.set_backend('numba')  # Enable JIT acceleration
        >>> A = loom.randn(1000, 1000)
        >>> B = loom.randn(1000, 1000)
        >>> C = loom.matmul(A, B)  # 10-50x faster with Numba
    """
    if not isinstance(x1, Tensor):
        x1 = Tensor(x1)
    if not isinstance(x2, Tensor):
        x2 = Tensor(x2)
    
    return x1 @ x2

# Config, logging, and errors
from loom import config
from loom import logging
from loom.config import set_backend, get_backend_info
from loom.errors import (
    loomError, LoomError, ShapeError, DTypeError, ComputationError,
    SymbolicError, OptimizationError, IntegrationError, BackendError,
    NumericalError, SingularMatrixError, ConvergenceError,
    TaskError, RecipeError, CacheError, CUDAError
)

__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Core types
    "Tensor",
    "Symbol",
    "Shape",
    "DType",
    # Factory functions
    "array",
    "zeros",
    "ones",
    "full",
    "eye",
    "matmul",
    # Shape utilities
    "broadcast_shapes",
    "parse_dtype",
    # Complex operations
    "conj",
    "real",
    "imag",
    "angle",
    "polar",
    # Random (top-level convenience)
    "seed",
    "rand",
    "randn",
    "randint",
    "uniform",
    "normal",
    "exponential",
    "poisson",
    "choice",
    "permutation",
    # Configuration and Logging
    "config",
    "logging",
    "set_backend",
    "get_backend_info",
    # Errors
    "loomError",
    "LoomError",
    "ShapeError",
    "DTypeError",
    "ComputationError",
    "SymbolicError",
    "OptimizationError",
    "IntegrationError",
    "BackendError",
    "NumericalError",
    "SingularMatrixError",
    "ConvergenceError",
    "TaskError",
    "RecipeError",
    "CacheError",
    "CUDAError",
]
