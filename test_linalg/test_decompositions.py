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
Tests for QR and Cholesky decompositions.
"""

import pytest
import loom as tf
import loom.linalg as la
import math

def mat_close(a, b, tol=1e-5):
    d1 = a.compute()
    d2 = b.compute()
    if len(d1) != len(d2): return False
    return all(abs(x - y) < tol for x, y in zip(d1, d2))

def is_identity(t, tol=1e-5):
    arr = t.compute()
    rows, cols = t.shape.dims
    for r in range(rows):
        for c in range(cols):
            expected = 1.0 if r == c else 0.0
            if abs(arr[r*cols + c] - expected) > tol:
                return False
    return True

def is_upper_triangular(t, tol=1e-5):
    arr = t.compute()
    rows, cols = t.shape.dims
    for r in range(rows):
        for c in range(cols):
            if c < r:
                if abs(arr[r*cols + c]) > tol:
                    return False
    return True

def is_lower_triangular(t, tol=1e-5):
    # Check if upper part is 0 (excluding diagonal)
    arr = t.compute()
    rows, cols = t.shape.dims
    for r in range(rows):
        for c in range(cols):
            if c > r:
                if abs(arr[r*cols + c]) > tol:
                    return False
    return True

def test_qr_simple():
    """Test QR decomposition."""
    # A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
    # Example from numeric literature
    
    a = tf.array([
        [12.0, -51.0, 4.0],
        [6.0, 167.0, -68.0],
        [-4.0, 24.0, -41.0]
    ])
    
    q, r = la.qr(a, mode='complete')
    
    # 1. Check reconstruction
    assert mat_close(q @ r, a)
    
    # 2. Check orthogonality of Q
    assert is_identity(q.T @ q)
    
    # 3. Check R is upper triangular
    assert is_upper_triangular(r)

def test_qr_reduced():
    """Test reduced QR (tall matrix)."""
    # 4x3 matrix
    a = tf.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 7.0],
        [4.0, 2.0, 1.0]
    ])
    
    q, r = la.qr(a, mode='reduced')
    
    # Q should be (4, 3), R should be (3, 3)
    assert q.shape.dims == (4, 3)
    assert r.shape.dims == (3, 3)
    
    # Reconstruction
    assert mat_close(q @ r, a)
    
    # Orthogonality columns: Q.T @ Q = I(3)
    assert is_identity(q.T @ q)

def test_cholesky():
    """Test Cholesky decomposition."""
    # A must be symmetric positive definite.
    # [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    a = tf.array([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0]
    ])
    
    l = la.cholesky(a)
    
    # L is lower triangular
    assert is_lower_triangular(l)
    
    # Reconstruction: L @ L.T = A
    recon = l @ l.T
    assert mat_close(recon, a)
    
    # Verify values (L diagonal should be positive)
    # L[0,0] = sqrt(4) = 2
    l_data = l.compute()
    assert math.isclose(l_data[0], 2.0)

def test_cholesky_fail():
    """Test Cholesky fails on non-positive-definite."""
    # [[1, 2], [2, 1]] -> det = 1-4 = -3 < 0
    a = tf.array([[1.0, 2.0], [2.0, 1.0]])
    
    with pytest.raises(ValueError, match="positive definite"):
        la.cholesky(a)

