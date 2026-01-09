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
Numba JIT-Accelerated Backend.

Optional backend that uses Numba for JIT compilation of performance-critical operations.
Falls back to CPU if Numba is not installed.
"""

import math
from typing import List, Tuple
from loom.backend.base import Backend

# Try to import Numba
_NUMBA_AVAILABLE = False
try:
    import numba
    from numba import jit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    pass


class NumbaBackend(Backend):
    """
    Numba JIT-accelerated backend.
    
    Uses Numba's @jit decorator for significant speedups on numerical operations.
    Automatically falls back to CPU backend if Numba is not installed.
    """
    
    def __init__(self):
        if _NUMBA_AVAILABLE:
            self._compile_kernels()
    
    def _compile_kernels(self):
        """Pre-compile Numba kernels."""
        @jit(nopython=True, cache=True)
        def _numba_add(a, b):
            n = len(a)
            result = [0.0] * n
            for i in range(n):
                result[i] = a[i] + b[i]
            return result
        
        @jit(nopython=True, cache=True)
        def _numba_mul(a, b):
            n = len(a)
            result = [0.0] * n
            for i in range(n):
                result[i] = a[i] * b[i]
            return result
        
        @jit(nopython=True, cache=True, parallel=True)
        def _numba_matmul(a, b, M, K, N):
            result = [0.0] * (M * N)
            for i in prange(M):
                for j in range(N):
                    s = 0.0
                    for k in range(K):
                        s += a[i * K + k] * b[k * N + j]
                    result[i * N + j] = s
            return result
        
        self._add_kernel = _numba_add
        self._mul_kernel = _numba_mul
        self._matmul_kernel = _numba_matmul
    
    @property
    def name(self) -> str:
        return "numba"
    
    @property
    def is_available(self) -> bool:
        return _NUMBA_AVAILABLE
    
    def add(self, a: List[float], b: List[float]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().add(a, b)
        return list(self._add_kernel(a, b))
    
    def mul(self, a: List[float], b: List[float]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().mul(a, b)
        return list(self._mul_kernel(a, b))
    
    def matmul(self, a: List[float], b: List[float], 
               a_shape: Tuple[int, int], b_shape: Tuple[int, int]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().matmul(a, b, a_shape, b_shape)
        
        M, K = a_shape
        _, N = b_shape
        return list(self._matmul_kernel(a, b, M, K, N))
    
    def sum(self, a: List[float]) -> float:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().sum(a)
        return sum(a)  # Python sum is already fast for this
    
    def exp(self, a: List[float]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().exp(a)
        return [math.exp(x) for x in a]
    
    def log(self, a: List[float]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().log(a)
        return [math.log(x) if x > 0 else float('-inf') for x in a]
    
    def sqrt(self, a: List[float]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().sqrt(a)
        return [math.sqrt(x) if x >= 0 else float('nan') for x in a]


# Singleton instance
_numba_backend = None

def get_numba_backend() -> NumbaBackend:
    """Get the Numba backend singleton (creates on first call)."""
    global _numba_backend
    if _numba_backend is None:
        _numba_backend = NumbaBackend()
    return _numba_backend

