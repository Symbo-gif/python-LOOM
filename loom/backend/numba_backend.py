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
from typing import List, Tuple, Optional, Union
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
        
        # Linear Algebra Operations
        
        @jit(nopython=True, cache=True)
        def _numba_lu_decomposition(A, n):
            """LU decomposition with partial pivoting.
            
            Returns P, L, U such that P @ L @ U = A.
            """
            # Make copies to avoid modifying input
            L = [0.0] * (n * n)
            U = [0.0] * (n * n)
            P_perm = [0.0] * (n * n)  # Will store row swaps applied
            
            # Initialize U with A, L and P with identity
            for i in range(n):
                for j in range(n):
                    U[i * n + j] = A[i * n + j]
                L[i * n + i] = 1.0
                P_perm[i * n + i] = 1.0
            
            for k in range(n - 1):
                # Find pivot
                pivot_row = k
                max_val = abs(U[k * n + k])
                for i in range(k + 1, n):
                    if abs(U[i * n + k]) > max_val:
                        max_val = abs(U[i * n + k])
                        pivot_row = i
                
                # Swap rows in U, P_perm, and L (columns 0 to k-1)
                if pivot_row != k:
                    for c in range(n):
                        # Swap U rows
                        temp = U[k * n + c]
                        U[k * n + c] = U[pivot_row * n + c]
                        U[pivot_row * n + c] = temp
                        # Swap P_perm rows
                        temp = P_perm[k * n + c]
                        P_perm[k * n + c] = P_perm[pivot_row * n + c]
                        P_perm[pivot_row * n + c] = temp
                    # Swap L rows (only columns 0 to k-1)
                    for c in range(k):
                        temp = L[k * n + c]
                        L[k * n + c] = L[pivot_row * n + c]
                        L[pivot_row * n + c] = temp
                
                # Elimination
                if abs(U[k * n + k]) > 1e-12:
                    for i in range(k + 1, n):
                        factor = U[i * n + k] / U[k * n + k]
                        L[i * n + k] = factor
                        U[i * n + k] = 0.0
                        for c in range(k + 1, n):
                            U[i * n + c] -= factor * U[k * n + c]
            
            # P_perm represents the permutation: P_perm @ A = L @ U
            # We need P such that: P @ L @ U = A
            # P = P_perm^T (transpose of P_perm, which is also its inverse)
            P = [0.0] * (n * n)
            for i in range(n):
                for j in range(n):
                    P[i * n + j] = P_perm[j * n + i]
            
            return P, L, U
        
        @jit(nopython=True, cache=True)
        def _numba_qr_decomposition(A, m, n):
            """QR decomposition using Gram-Schmidt."""
            # Q is m x min(m, n), R is min(m, n) x n
            k = min(m, n)
            Q = [0.0] * (m * k)
            R = [0.0] * (k * n)
            
            for j in range(k):
                # v = A[:, j]
                v = [0.0] * m
                for i in range(m):
                    v[i] = A[i * n + j]
                
                for i in range(j):
                    # R[i, j] = Q[:, i] . A[:, j]
                    dot = 0.0
                    for row in range(m):
                        dot += Q[row * k + i] * A[row * n + j]
                    R[i * n + j] = dot
                    # v -= R[i, j] * Q[:, i]
                    for row in range(m):
                        v[row] -= dot * Q[row * k + i]
                
                # R[j, j] = norm(v)
                norm_v = 0.0
                for row in range(m):
                    norm_v += v[row] * v[row]
                norm_v = math.sqrt(norm_v)
                R[j * n + j] = norm_v
                
                # Q[:, j] = v / norm(v)
                if norm_v > 1e-12:
                    for row in range(m):
                        Q[row * k + j] = v[row] / norm_v
            
            return Q, R
        
        @jit(nopython=True, cache=True, parallel=True)
        def _numba_transpose(A, m, n):
            """Matrix transpose."""
            B = [0.0] * (n * m)
            for i in prange(m):
                for j in range(n):
                    B[j * m + i] = A[i * n + j]
            return B
        
        # Element-wise functions with Numba JIT
        
        @jit(nopython=True, cache=True, parallel=True)
        def _numba_exp(a):
            """Element-wise exponential."""
            n = len(a)
            result = [0.0] * n
            for i in prange(n):
                result[i] = math.exp(a[i])
            return result
        
        @jit(nopython=True, cache=True, parallel=True)
        def _numba_log(a):
            """Element-wise logarithm."""
            n = len(a)
            result = [0.0] * n
            for i in prange(n):
                if a[i] > 0:
                    result[i] = math.log(a[i])
                else:
                    result[i] = float('-inf')
            return result
        
        @jit(nopython=True, cache=True, parallel=True)
        def _numba_sqrt(a):
            """Element-wise square root."""
            n = len(a)
            result = [0.0] * n
            for i in prange(n):
                if a[i] >= 0:
                    result[i] = math.sqrt(a[i])
                else:
                    result[i] = float('nan')
            return result
        
        # Reductions with axis support
        
        @jit(nopython=True, cache=True)
        def _numba_sum_all(a):
            """Sum all elements."""
            total = 0.0
            for i in range(len(a)):
                total += a[i]
            return total
        
        @jit(nopython=True, cache=True, parallel=True)
        def _numba_sum_axis0(A, m, n):
            """Sum along axis 0."""
            result = [0.0] * n
            for j in prange(n):
                for i in range(m):
                    result[j] += A[i * n + j]
            return result
        
        @jit(nopython=True, cache=True, parallel=True)
        def _numba_sum_axis1(A, m, n):
            """Sum along axis 1."""
            result = [0.0] * m
            for i in prange(m):
                for j in range(n):
                    result[i] += A[i * n + j]
            return result
        
        self._add_kernel = _numba_add
        self._mul_kernel = _numba_mul
        self._matmul_kernel = _numba_matmul
        self._lu_kernel = _numba_lu_decomposition
        self._qr_kernel = _numba_qr_decomposition
        self._transpose_kernel = _numba_transpose
        self._exp_kernel = _numba_exp
        self._log_kernel = _numba_log
        self._sqrt_kernel = _numba_sqrt
        self._sum_all_kernel = _numba_sum_all
        self._sum_axis0_kernel = _numba_sum_axis0
        self._sum_axis1_kernel = _numba_sum_axis1
    
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
    
    def sum(self, a: List[float], axis: Optional[int] = None, 
            shape: Optional[Tuple[int, ...]] = None) -> Union[float, List[float]]:
        """
        Sum with axis support.
        
        Args:
            a: Flat list of values
            axis: Axis along which to sum. None for total sum.
            shape: Shape of the array (required when axis is not None)
            
        Returns:
            Scalar if axis is None, list otherwise.
        """
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().sum(a)
        
        if axis is None:
            return self._sum_all_kernel(a)
        
        if shape is None or len(shape) != 2:
            # Fallback for non-2D or missing shape
            return self._sum_all_kernel(a)
        
        m, n = shape
        if axis == 0:
            return list(self._sum_axis0_kernel(a, m, n))
        elif axis == 1:
            return list(self._sum_axis1_kernel(a, m, n))
        else:
            # Fallback for higher dimensions
            return self._sum_all_kernel(a)
    
    def exp(self, a: List[float]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().exp(a)
        return list(self._exp_kernel(a))
    
    def log(self, a: List[float]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().log(a)
        return list(self._log_kernel(a))
    
    def sqrt(self, a: List[float]) -> List[float]:
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return get_cpu_backend().sqrt(a)
        return list(self._sqrt_kernel(a))
    
    def lu(self, a: List[float], n: int) -> Tuple[List[float], List[float], List[float]]:
        """
        LU decomposition with partial pivoting.
        
        Args:
            a: Flat list of matrix values (row-major)
            n: Matrix dimension (square matrix assumed)
            
        Returns:
            (P, L, U) as flat lists
        """
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            # Fallback to basic implementation
            return self._lu_fallback(a, n)
        
        P, L, U = self._lu_kernel(a, n)
        return list(P), list(L), list(U)
    
    def qr(self, a: List[float], m: int, n: int) -> Tuple[List[float], List[float]]:
        """
        QR decomposition using Gram-Schmidt.
        
        Args:
            a: Flat list of matrix values (row-major, m x n)
            m: Number of rows
            n: Number of columns
            
        Returns:
            (Q, R) as flat lists. Q is m x min(m,n), R is min(m,n) x n
        """
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            return self._qr_fallback(a, m, n)
        
        Q, R = self._qr_kernel(a, m, n)
        return list(Q), list(R)
    
    def transpose(self, a: List[float], m: int, n: int) -> List[float]:
        """
        Matrix transpose.
        
        Args:
            a: Flat list of matrix values (row-major, m x n)
            m: Number of rows
            n: Number of columns
            
        Returns:
            Transposed matrix as flat list (n x m)
        """
        if not _NUMBA_AVAILABLE:
            from loom.backend.cpu import get_cpu_backend
            # Simple fallback
            result = [0.0] * (m * n)
            for i in range(m):
                for j in range(n):
                    result[j * m + i] = a[i * n + j]
            return result
        
        return list(self._transpose_kernel(a, m, n))
    
    def _lu_fallback(self, a: List[float], n: int) -> Tuple[List[float], List[float], List[float]]:
        """Pure Python LU decomposition fallback.
        
        Returns P, L, U such that P @ L @ U = A.
        """
        L = [0.0] * (n * n)
        U = list(a)
        P_perm = [0.0] * (n * n)
        
        for i in range(n):
            L[i * n + i] = 1.0
            P_perm[i * n + i] = 1.0
        
        for k in range(n - 1):
            # Find pivot
            pivot_row = k
            max_val = abs(U[k * n + k])
            for i in range(k + 1, n):
                if abs(U[i * n + k]) > max_val:
                    max_val = abs(U[i * n + k])
                    pivot_row = i
            
            # Swap rows
            if pivot_row != k:
                for c in range(n):
                    U[k * n + c], U[pivot_row * n + c] = U[pivot_row * n + c], U[k * n + c]
                    P_perm[k * n + c], P_perm[pivot_row * n + c] = P_perm[pivot_row * n + c], P_perm[k * n + c]
                for c in range(k):
                    L[k * n + c], L[pivot_row * n + c] = L[pivot_row * n + c], L[k * n + c]
            
            # Elimination
            if abs(U[k * n + k]) > 1e-12:
                for i in range(k + 1, n):
                    factor = U[i * n + k] / U[k * n + k]
                    L[i * n + k] = factor
                    U[i * n + k] = 0.0
                    for c in range(k + 1, n):
                        U[i * n + c] -= factor * U[k * n + c]
        
        # P = P_perm^T (transpose to get correct P such that P @ L @ U = A)
        P = [0.0] * (n * n)
        for i in range(n):
            for j in range(n):
                P[i * n + j] = P_perm[j * n + i]
        
        return P, L, U
    
    def _qr_fallback(self, a: List[float], m: int, n: int) -> Tuple[List[float], List[float]]:
        """Pure Python QR decomposition fallback."""
        k = min(m, n)
        Q = [0.0] * (m * k)
        R = [0.0] * (k * n)
        
        for j in range(k):
            v = [a[i * n + j] for i in range(m)]
            
            for i in range(j):
                dot = sum(Q[row * k + i] * a[row * n + j] for row in range(m))
                R[i * n + j] = dot
                for row in range(m):
                    v[row] -= dot * Q[row * k + i]
            
            norm_v = math.sqrt(sum(x * x for x in v))
            R[j * n + j] = norm_v
            
            if norm_v > 1e-12:
                for row in range(m):
                    Q[row * k + j] = v[row] / norm_v
        
        return Q, R


# Singleton instance
_numba_backend = None

def get_numba_backend() -> NumbaBackend:
    """Get the Numba backend singleton (creates on first call)."""
    global _numba_backend
    if _numba_backend is None:
        _numba_backend = NumbaBackend()
    return _numba_backend

