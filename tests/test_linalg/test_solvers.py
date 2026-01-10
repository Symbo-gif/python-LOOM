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
Tests for linear solvers and decompositions.
"""

import pytest
import loom as tf
import loom.linalg as la
import math

def is_identity(t, tol=1e-9):
    arr = t.compute()
    rows, cols = t.shape.dims
    if rows != cols: return False
    for r in range(rows):
        for c in range(cols):
            expected = 1.0 if r == c else 0.0
            if abs(arr[r*cols + c] - expected) > tol:
                return False
    return True

def mat_close(a, b, tol=1e-9):
    d1 = a.compute()
    d2 = b.compute()
    if len(d1) != len(d2): return False
    return all(abs(x - y) < tol for x, y in zip(d1, d2))

def test_lu_decomposition():
    """Test LU decomposition reconstruction."""
    # A = [[4, 3], [6, 3]]
    # P @ A = L @ U ??? No. A = P @ L @ U
    
    a = tf.array([[4, 3], [6, 3]])
    p, l, u = la.lu(a)
    
    # Verify reconstruction
    recon = p @ l @ u
    assert mat_close(recon, a)
    
    # Verify properties
    # P is permutation (ortho) -> P @ P.T = I
    assert is_identity(p @ p.T)
    
    # L is lower dict
    # U is upper dict

    
def test_solve_system():
    """Test Ax=b solver."""
    # 2x + y = 10
    # x + y = 6
    # Solution: x=4, y=2
    
    A = tf.array([[2, 1], [1, 1]])
    b = tf.array([10, 6])
    
    x = la.solve(A, b)
    res = x.compute()
    
    assert math.isclose(res[0], 4.0, abs_tol=1e-7)
    assert math.isclose(res[1], 2.0, abs_tol=1e-7)

    # Verify A@x = b
    assert mat_close(A @ x, b)
    
def test_inverse():
    """Test matrix inversion."""
    # Use float64 for better precision
    A = tf.array([[4, 7], [2, 6]], dtype='float64')
    A_inv = la.inv(A)
    
    # A @ A_inv = I (with reasonable tolerance for numerical precision)
    assert is_identity(A @ A_inv, tol=1e-6)
    assert is_identity(A_inv @ A, tol=1e-6)
    
def test_det_simple():
    """Test determinant."""
    # Identity -> 1
    i = tf.eye(3)
    assert math.isclose(la.det(i).compute()[0], 1.0)
    
    # Scale
    a = tf.array([[2, 0], [0, 3]])
    assert math.isclose(la.det(a).compute()[0], 6.0)

