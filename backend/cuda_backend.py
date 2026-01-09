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
CUDA GPU Backend.

Optional backend that uses CUDA for GPU-accelerated operations.
Falls back to CPU if CUDA/CuPy is not available.
"""

import math
from typing import List, Tuple
from loom.backend.base import Backend

# Try to import CuPy (CUDA interface for Python)
_CUDA_AVAILABLE = False
_cupy = None
try:
    import cupy as cp
    _cupy = cp
    _CUDA_AVAILABLE = True
except ImportError:
    pass


class CUDABackend(Backend):
    """
    CUDA GPU-accelerated backend.
    
    Uses CuPy for GPU operations, providing massive speedups for large tensors.
    Automatically falls back to CPU backend if CUDA/CuPy is not available.
    
    To enable CUDA acceleration:
        pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
    """
    
    @property
    def name(self) -> str:
        return "cuda"
    
    @property
    def is_available(self) -> bool:
        return _CUDA_AVAILABLE
    
    def _to_gpu(self, a: List[float]):
        """Transfer data to GPU."""
        return _cupy.array(a, dtype=_cupy.float64)
    
    def _to_cpu(self, a) -> List[float]:
        """Transfer data back to CPU."""
        return a.get().tolist()
    
    def add(self, a: List[float], b: List[float]) -> List[float]:
        if not _CUDA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().add(a, b)
        
        a_gpu = self._to_gpu(a)
        b_gpu = self._to_gpu(b)
        result = a_gpu + b_gpu
        return self._to_cpu(result)
    
    def mul(self, a: List[float], b: List[float]) -> List[float]:
        if not _CUDA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().mul(a, b)
        
        a_gpu = self._to_gpu(a)
        b_gpu = self._to_gpu(b)
        result = a_gpu * b_gpu
        return self._to_cpu(result)
    
    def matmul(self, a: List[float], b: List[float], 
               a_shape: Tuple[int, int], b_shape: Tuple[int, int]) -> List[float]:
        if not _CUDA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().matmul(a, b, a_shape, b_shape)
        
        M, K = a_shape
        _, N = b_shape
        
        a_gpu = self._to_gpu(a).reshape((M, K))
        b_gpu = self._to_gpu(b).reshape((K, N))
        result = _cupy.matmul(a_gpu, b_gpu)
        return self._to_cpu(result.flatten())
    
    def sum(self, a: List[float]) -> float:
        if not _CUDA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().sum(a)
        
        a_gpu = self._to_gpu(a)
        return float(_cupy.sum(a_gpu))
    
    def exp(self, a: List[float]) -> List[float]:
        if not _CUDA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().exp(a)
        
        a_gpu = self._to_gpu(a)
        return self._to_cpu(_cupy.exp(a_gpu))
    
    def log(self, a: List[float]) -> List[float]:
        if not _CUDA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().log(a)
        
        a_gpu = self._to_gpu(a)
        return self._to_cpu(_cupy.log(a_gpu))
    
    def sqrt(self, a: List[float]) -> List[float]:
        if not _CUDA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().sqrt(a)
        
        a_gpu = self._to_gpu(a)
        return self._to_cpu(_cupy.sqrt(a_gpu))


# Singleton instance
_cuda_backend = None

def get_cuda_backend() -> CUDABackend:
    """Get the CUDA backend singleton."""
    global _cuda_backend
    if _cuda_backend is None:
        _cuda_backend = CUDABackend()
    return _cuda_backend

