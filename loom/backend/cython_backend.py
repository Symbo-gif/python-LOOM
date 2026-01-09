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
Cython-Accelerated Backend.

Optional backend that uses Cython-compiled operations.
Falls back to CPU if Cython extensions are not installed.
"""

import math
from typing import List, Tuple
from loom.backend.base import Backend

# Try to import Cython extensions
_CYTHON_AVAILABLE = False
try:
    from loom.backend._cython_ops import (
        cython_add, cython_mul, cython_matmul, cython_sum
    )
    _CYTHON_AVAILABLE = True
except ImportError:
    pass


class CythonBackend(Backend):
    """
    Cython-accelerated backend.
    
    Uses pre-compiled Cython extensions for significant speedups.
    Automatically falls back to CPU backend if Cython extensions are not available.
    
    To enable Cython acceleration, install the cython-ops package:
        pip install loom[cython]
    """
    
    @property
    def name(self) -> str:
        return "cython"
    
    @property
    def is_available(self) -> bool:
        return _CYTHON_AVAILABLE
    
    def add(self, a: List[float], b: List[float]) -> List[float]:
        if not _CYTHON_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().add(a, b)
        return cython_add(a, b)
    
    def mul(self, a: List[float], b: List[float]) -> List[float]:
        if not _CYTHON_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().mul(a, b)
        return cython_mul(a, b)
    
    def matmul(self, a: List[float], b: List[float], 
               a_shape: Tuple[int, int], b_shape: Tuple[int, int]) -> List[float]:
        if not _CYTHON_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().matmul(a, b, a_shape, b_shape)
        
        M, K = a_shape
        _, N = b_shape
        return cython_matmul(a, b, M, K, N)
    
    def sum(self, a: List[float]) -> float:
        if not _CYTHON_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().sum(a)
        return cython_sum(a)
    
    def exp(self, a: List[float]) -> List[float]:
        # Cython extension for exp not yet implemented, use CPU
        from loom.backend.cpu import get_cpu_backend
        return get_cpu_backend().exp(a)
    
    def log(self, a: List[float]) -> List[float]:
        from loom.backend.cpu import get_cpu_backend
        return get_cpu_backend().log(a)
    
    def sqrt(self, a: List[float]) -> List[float]:
        from loom.backend.cpu import get_cpu_backend
        return get_cpu_backend().sqrt(a)


# Singleton instance
_cython_backend = None

def get_cython_backend() -> CythonBackend:
    """Get the Cython backend singleton."""
    global _cython_backend
    if _cython_backend is None:
        _cython_backend = CythonBackend()
    return _cython_backend

