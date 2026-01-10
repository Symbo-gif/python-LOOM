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
Tests for Numba-accelerated linear algebra operations.

Task V1.1-007: Extend Numba Backend to More Operations
"""

import pytest
import math

# Skip if Numba not available
try:
    import numba
    numba_available = True
except ImportError:
    numba_available = False

skip_if_no_numba = pytest.mark.skipif(
    not numba_available, 
    reason="Numba not installed"
)


@skip_if_no_numba
class TestNumbaLinAlg:
    """Test Numba-accelerated linear algebra."""
    
    def test_lu_decomposition(self):
        """Test LU decomposition."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        # Test matrix [[4, 3], [6, 3]] (row-major flat)
        A = [4.0, 3.0, 6.0, 3.0]
        n = 2
        
        P, L, U = backend.lu(A, n)
        
        # Reconstruct P @ L @ U and compare to A (permuted)
        # P @ L gives intermediate, then multiply by U
        
        # Helper to multiply matrices
        def matmul(a, b, n):
            result = [0.0] * (n * n)
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for k in range(n):
                        s += a[i * n + k] * b[k * n + j]
                    result[i * n + j] = s
            return result
        
        PL = matmul(P, L, n)
        PLU = matmul(PL, U, n)
        
        # Compare PLU to A
        for i in range(4):
            assert abs(PLU[i] - A[i]) < 1e-10, f"Mismatch at index {i}: {PLU[i]} vs {A[i]}"
    
    def test_lu_decomposition_3x3(self):
        """Test LU decomposition on 3x3 matrix."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        # Test matrix (row-major flat)
        A = [2.0, 1.0, 1.0, 
             4.0, 3.0, 3.0, 
             8.0, 7.0, 9.0]
        n = 3
        
        P, L, U = backend.lu(A, n)
        
        # Reconstruct
        def matmul(a, b, n):
            result = [0.0] * (n * n)
            for i in range(n):
                for j in range(n):
                    s = 0.0
                    for k in range(n):
                        s += a[i * n + k] * b[k * n + j]
                    result[i * n + j] = s
            return result
        
        PL = matmul(P, L, n)
        PLU = matmul(PL, U, n)
        
        for i in range(9):
            assert abs(PLU[i] - A[i]) < 1e-10
    
    def test_qr_decomposition(self):
        """Test QR decomposition."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        # Test matrix [[1, 2], [3, 4], [5, 6]] (3x2, row-major flat)
        A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        m, n = 3, 2
        
        Q, R = backend.qr(A, m, n)
        
        # Q should be 3x2, R should be 2x2
        k = min(m, n)  # 2
        
        # Verify Q is orthogonal (Q.T @ Q ≈ I)
        QtQ = [0.0] * (k * k)
        for i in range(k):
            for j in range(k):
                s = 0.0
                for row in range(m):
                    s += Q[row * k + i] * Q[row * k + j]
                QtQ[i * k + j] = s
        
        # Should be identity
        for i in range(k):
            for j in range(k):
                expected = 1.0 if i == j else 0.0
                assert abs(QtQ[i * k + j] - expected) < 1e-10, \
                    f"QtQ[{i},{j}] = {QtQ[i * k + j]}, expected {expected}"
        
        # Verify Q @ R ≈ A
        QR = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for kk in range(k):
                    s += Q[i * k + kk] * R[kk * n + j]
                QR[i * n + j] = s
        
        for i in range(m * n):
            assert abs(QR[i] - A[i]) < 1e-10
    
    def test_transpose(self):
        """Test transpose operation."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        # Test matrix [[1, 2, 3], [4, 5, 6]] (2x3, row-major flat)
        A = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        m, n = 2, 3
        
        At = backend.transpose(A, m, n)
        
        # Expected: [[1, 4], [2, 5], [3, 6]] (3x2, row-major flat)
        expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        
        for i in range(6):
            assert abs(At[i] - expected[i]) < 1e-10
    
    def test_elementwise_exp(self):
        """Test element-wise exponential."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        A = [0.0, 1.0, 2.0, -1.0]
        result = backend.exp(A)
        
        for i, val in enumerate(A):
            expected = math.exp(val)
            assert abs(result[i] - expected) < 1e-10
    
    def test_elementwise_log(self):
        """Test element-wise logarithm."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        A = [1.0, math.e, math.e**2, 0.5]
        result = backend.log(A)
        
        for i, val in enumerate(A):
            expected = math.log(val)
            assert abs(result[i] - expected) < 1e-10
    
    def test_elementwise_sqrt(self):
        """Test element-wise square root."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        A = [1.0, 4.0, 9.0, 16.0]
        result = backend.sqrt(A)
        
        expected = [1.0, 2.0, 3.0, 4.0]
        for i in range(4):
            assert abs(result[i] - expected[i]) < 1e-10
    
    def test_sum_axis_none(self):
        """Test sum with axis=None (total sum)."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        A = [1.0, 2.0, 3.0, 4.0]
        result = backend.sum(A, axis=None)
        
        assert abs(result - 10.0) < 1e-10
    
    def test_sum_axis0(self):
        """Test sum along axis 0."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        # Matrix [[1, 2], [3, 4]] (row-major flat)
        A = [1.0, 2.0, 3.0, 4.0]
        m, n = 2, 2
        
        result = backend.sum(A, axis=0, shape=(m, n))
        
        # Sum along axis 0: [4.0, 6.0]
        expected = [4.0, 6.0]
        for i in range(2):
            assert abs(result[i] - expected[i]) < 1e-10
    
    def test_sum_axis1(self):
        """Test sum along axis 1."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        # Matrix [[1, 2], [3, 4]] (row-major flat)
        A = [1.0, 2.0, 3.0, 4.0]
        m, n = 2, 2
        
        result = backend.sum(A, axis=1, shape=(m, n))
        
        # Sum along axis 1: [3.0, 7.0]
        expected = [3.0, 7.0]
        for i in range(2):
            assert abs(result[i] - expected[i]) < 1e-10


@skip_if_no_numba
class TestNumbaLinAlgIntegration:
    """Integration tests with loom API."""
    
    def test_lu_via_linalg(self):
        """Test LU decomposition via loom.linalg module."""
        import loom
        import loom.linalg as la
        
        loom.set_backend('numba')
        
        A = loom.array([[4.0, 3.0], [6.0, 3.0]])
        P, L, U = la.lu(A)
        
        # Verify P @ L @ U ≈ A
        reconstructed = loom.matmul(P, loom.matmul(L, U))
        
        A_list = A.tolist()
        R_list = reconstructed.tolist()
        
        for i in range(2):
            for j in range(2):
                assert abs(R_list[i][j] - A_list[i][j]) < 1e-10
    
    def test_qr_via_linalg(self):
        """Test QR decomposition via loom.linalg module."""
        import loom
        import loom.linalg as la
        
        loom.set_backend('numba')
        
        A = loom.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Q, R = la.qr(A)
        
        # Verify Q @ R ≈ A
        reconstructed = loom.matmul(Q, R)
        
        A_list = A.tolist()
        R_list = reconstructed.tolist()
        
        for i in range(3):
            for j in range(2):
                assert abs(R_list[i][j] - A_list[i][j]) < 1e-6
    
    def test_transpose_via_tensor(self):
        """Test transpose via Tensor.T property."""
        import loom
        
        loom.set_backend('numba')
        
        A = loom.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        At = A.T
        
        expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        result = At.tolist()
        
        assert result == expected
    
    def test_exp_via_backend(self):
        """Test exp function via backend directly."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        A = [0.0, 1.0, 2.0, -1.0]
        result = backend.exp(A)
        
        assert abs(result[0] - 1.0) < 1e-10  # exp(0)
        assert abs(result[1] - math.e) < 1e-10  # exp(1)
    
    def test_sqrt_via_backend(self):
        """Test sqrt function via backend directly."""
        from loom.backend.numba_backend import get_numba_backend
        
        backend = get_numba_backend()
        
        A = [1.0, 4.0, 9.0, 16.0]
        result = backend.sqrt(A)
        
        expected = [1.0, 2.0, 3.0, 4.0]
        for i in range(4):
            assert abs(result[i] - expected[i]) < 1e-10
