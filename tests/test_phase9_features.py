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
Tests for new Phase 9 features: logm, outer product, rectangular LU, N-D sampling, chi-square p-value.
"""

import pytest
import math
import loom as tf
import loom.linalg as la
import loom.stats as stats
from loom.special import gammainc, gammaincc
from loom.field import FieldTensor


class TestLogM:
    """Tests for matrix logarithm."""
    
    def test_logm_identity(self):
        """log(I) = 0."""
        I = tf.eye(3)
        L = la.logm(I)
        # Should be all zeros (or very close)
        for row in L.tolist():
            for val in row:
                assert abs(val) < 1e-8
    
    def test_logm_exp_roundtrip(self):
        """log(exp(A)) ≈ A for small A."""
        A = tf.array([[0.1, 0.05], [0.05, 0.1]])
        expA = la.expm(A)
        logExpA = la.logm(expA)
        
        # Should recover A
        A_list = A.tolist()
        result = logExpA.tolist()
        for i in range(2):
            for j in range(2):
                assert math.isclose(result[i][j], A_list[i][j], abs_tol=1e-6)
    
    def test_logm_diagonal(self):
        """log([[e, 0], [0, e^2]]) = [[1, 0], [0, 2]]."""
        e = math.e
        A = tf.array([[e, 0], [0, e**2]])
        L = la.logm(A)
        result = L.tolist()
        
        assert math.isclose(result[0][0], 1.0, abs_tol=1e-6)
        assert math.isclose(result[1][1], 2.0, abs_tol=1e-6)
        assert abs(result[0][1]) < 1e-8
        assert abs(result[1][0]) < 1e-8


class TestOuterProduct:
    """Tests for outer product."""
    
    def test_outer_basic(self):
        """outer([1, 2, 3], [4, 5]) = [[4, 5], [8, 10], [12, 15]]."""
        a = tf.array([1, 2, 3])
        b = tf.array([4, 5])
        result = la.outer(a, b)
        
        expected = [[4, 5], [8, 10], [12, 15]]
        assert result.tolist() == expected
    
    def test_outer_shape(self):
        """outer(m-vector, n-vector) has shape (m, n)."""
        a = tf.array([1, 2, 3, 4, 5])
        b = tf.array([1, 2])
        result = la.outer(a, b)
        
        assert result.shape.dims == (5, 2)
    
    def test_outer_single_element(self):
        """outer([2], [3]) = [[6]]."""
        a = tf.array([2])
        b = tf.array([3])
        result = la.outer(a, b)
        
        assert result.tolist() == [[6]]


class TestRectangularLU:
    """Tests for rectangular LU decomposition."""
    
    def test_lu_tall_matrix(self):
        """LU of 4x2 matrix returns proper shapes."""
        A = tf.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        P, L, U = la.lu(A)
        
        # Check shapes
        assert P.shape.dims == (4, 4), f"P shape {P.shape.dims}"
        assert L.shape.dims == (4, 2), f"L shape {L.shape.dims}"
        assert U.shape.dims == (4, 2), f"U shape {U.shape.dims}"
        
        # U should have zeros below diagonal
        U_list = U.tolist()
        for i in range(min(4, 2)):
            for j in range(i):
                assert abs(U_list[i][j]) < 1e-10, f"U[{i}][{j}] should be 0"
    
    def test_lu_wide_matrix(self):
        """LU of 2x4 matrix."""
        A = tf.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        P, L, U = la.lu(A)
        
        # P @ L @ U should equal A
        reconstructed = P @ L @ U
        A_list = A.tolist()
        R_list = reconstructed.tolist()
        
        for i in range(2):
            for j in range(4):
                assert math.isclose(R_list[i][j], A_list[i][j], abs_tol=1e-8)


class TestNDFieldSampling:
    """Tests for N-dimensional field sampling."""
    
    def test_3d_sampling(self):
        """Sample from a 3D tensor."""
        # 2x2x2 tensor
        data = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        t = tf.array(data)
        ft = FieldTensor(t)
        
        # Sample at corners
        assert ft.sample([0, 0, 0]) == 0
        assert ft.sample([1, 1, 1]) == 7
        
        # Sample at center (should be average-ish)
        center = ft.sample([0.5, 0.5, 0.5])
        assert 3 < center < 4  # Somewhere in the middle
    
    def test_4d_sampling(self):
        """Sample from a 4D tensor."""
        # 2x2x2x2 tensor filled with index sum
        data = [[[[i+j+k+l for l in range(2)] for k in range(2)] for j in range(2)] for i in range(2)]
        t = tf.array(data)
        ft = FieldTensor(t)
        
        # Sample at origin
        assert ft.sample([0, 0, 0, 0]) == 0
        
        # Sample at max corner
        assert ft.sample([1, 1, 1, 1]) == 4
    
    def test_field_coords_mismatch_error(self):
        """Should raise error if coords don't match ndim."""
        t = tf.array([[1, 2], [3, 4]])
        ft = FieldTensor(t)
        
        with pytest.raises(ValueError):
            ft.sample([0.5])  # Only 1 coord for 2D tensor


class TestChiSquarePValue:
    """Tests for chi-square p-value calculation."""
    
    def test_chisquare_uniform(self):
        """Chi-square with uniform observed = uniform expected."""
        obs = [10, 10, 10, 10]
        chi_stat, p_val = stats.chisquare(obs)
        
        assert chi_stat.item() == 0.0
        # p-value should be 1.0 (perfect fit)
        assert math.isclose(p_val, 1.0, abs_tol=1e-10)
    
    def test_chisquare_significant(self):
        """Chi-square with clear deviation should have low p-value."""
        obs = [30, 10, 10, 10]  # Clearly not uniform
        chi_stat, p_val = stats.chisquare(obs)
        
        assert chi_stat.item() > 0
        assert p_val < 0.05  # Significant deviation
    
    def test_chisquare_custom_expected(self):
        """Chi-square with custom expected frequencies."""
        obs = [18, 12]
        exp = [15, 15]
        chi_stat, p_val = stats.chisquare(obs, exp)
        
        # (18-15)^2/15 + (12-15)^2/15 = 9/15 + 9/15 = 1.2
        assert math.isclose(chi_stat.item(), 1.2, abs_tol=1e-7)


class TestGammaInc:
    """Tests for incomplete gamma function."""
    
    def test_gammainc_small_x(self):
        """gammainc(1, 0) = 0."""
        assert gammainc(1, 0) == 0.0
    
    def test_gammainc_a1_x1(self):
        """gammainc(1, 1) = 1 - e^(-1) ≈ 0.632."""
        result = gammainc(1, 1)
        expected = 1 - math.exp(-1)
        assert math.isclose(result, expected, abs_tol=1e-7)
    
    def test_gammainc_complement(self):
        """gammainc + gammaincc = 1."""
        a, x = 2.5, 1.5
        P = gammainc(a, x)
        Q = gammaincc(a, x)
        assert math.isclose(P + Q, 1.0, abs_tol=1e-10)


class TestBackend:
    """Tests for backend system."""
    
    def test_cpu_backend_always_available(self):
        """CPU backend should always be available."""
        from loom.backend import available_backends
        assert 'cpu' in available_backends()
    
    def test_backend_switching(self):
        """Backend switching should work gracefully."""
        from loom.backend import set_backend, get_backend
        
        # Switch to CPU (always works)
        assert set_backend('cpu') == True
        assert get_backend().name == 'cpu'
        
        # Switch to non-existent backend falls back to CPU
        result = set_backend('nonexistent_backend')
        assert result == False
        assert get_backend().name == 'cpu'
    
    def test_cpu_backend_operations(self):
        """CPU backend basic operations should work."""
        from loom.backend import get_cpu_backend
        
        backend = get_cpu_backend()
        
        # Add
        result = backend.add([1.0, 2.0], [3.0, 4.0])
        assert result == [4.0, 6.0]
        
        # Mul
        result = backend.mul([2.0, 3.0], [4.0, 5.0])
        assert result == [8.0, 15.0]
        
        # Sum
        result = backend.sum([1.0, 2.0, 3.0, 4.0])
        assert result == 10.0

