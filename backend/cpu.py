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
Pure Python CPU Backend.

This is the default backend that is always available.
"""

import math
from typing import List, Tuple
from loom.backend.base import Backend


class CPUBackend(Backend):
    """
    Pure Python CPU backend.
    
    This backend is always available and serves as the fallback for all operations.
    """
    
    @property
    def name(self) -> str:
        return "cpu"
    
    @property
    def is_available(self) -> bool:
        return True
    
    def add(self, a: List[float], b: List[float]) -> List[float]:
        """Element-wise addition."""
        return [x + y for x, y in zip(a, b)]
    
    def mul(self, a: List[float], b: List[float]) -> List[float]:
        """Element-wise multiplication."""
        return [x * y for x, y in zip(a, b)]
    
    def matmul(self, a: List[float], b: List[float], 
               a_shape: Tuple[int, int], b_shape: Tuple[int, int]) -> List[float]:
        """
        Matrix multiplication for 2D matrices.
        
        Args:
            a: Flattened matrix A of shape (M, K)
            b: Flattened matrix B of shape (K, N)
            a_shape: (M, K)
            b_shape: (K, N)
            
        Returns:
            Flattened result matrix of shape (M, N)
        """
        M, K1 = a_shape
        K2, N = b_shape
        
        if K1 != K2:
            raise ValueError(f"Incompatible shapes for matmul: {a_shape} @ {b_shape}")
        
        K = K1
        result = [0.0] * (M * N)
        
        for i in range(M):
            for j in range(N):
                s = 0.0
                for k in range(K):
                    s += a[i * K + k] * b[k * N + j]
                result[i * N + j] = s
        
        return result
    
    def sum(self, a: List[float]) -> float:
        """Sum all elements."""
        return sum(a)
    
    def exp(self, a: List[float]) -> List[float]:
        """Element-wise exponential."""
        return [math.exp(x) for x in a]
    
    def log(self, a: List[float]) -> List[float]:
        """Element-wise natural logarithm."""
        return [math.log(x) if x > 0 else float('-inf') for x in a]
    
    def sqrt(self, a: List[float]) -> List[float]:
        """Element-wise square root."""
        return [math.sqrt(x) if x >= 0 else float('nan') for x in a]


# Singleton instance
_cpu_backend = CPUBackend()

def get_cpu_backend() -> CPUBackend:
    """Get the CPU backend singleton."""
    return _cpu_backend

