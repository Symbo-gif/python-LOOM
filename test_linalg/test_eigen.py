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
Tests for Eigenvalues and SVD.
"""

import pytest
import loom as tf
import loom.linalg as la
import math

def mat_close(a, b, tol=1e-3): # 1e-3 for SVD convergence variation
    d1 = a.compute()
    d2 = b.compute()
    if len(d1) != len(d2): return False
    return all(abs(x - y) < tol for x, y in zip(d1, d2))

def is_diagonal(t, tol=1e-4):
    arr = t.compute()
    rows, cols = t.shape.dims
    for r in range(rows):
        for c in range(cols):
            if r != c and abs(arr[r*cols + c]) > tol:
                return False
    return True

def test_eig_symmetric():
    """Test eigendecomposition of symmetric matrix."""
    # A = [[2, 1], [1, 2]]
    # Eigenvalues: 3, 1
    # Eigenvectors: [1, 1]/sqrt(2), [1, -1]/sqrt(2)
    
    a = tf.array([[2.0, 1.0], [1.0, 2.0]])
    w, v = la.eigh(a)
    
    w_data = w.compute()
    # Check eigenvalues (sort for comparison)
    w_sorted = sorted(w_data)
    assert math.isclose(w_sorted[0], 1.0, abs_tol=1e-3)
    assert math.isclose(w_sorted[1], 3.0, abs_tol=1e-3)
    
    # Verify A @ v = v @ diag(w)
    # A * V = V * D
    diag_w = tf.zeros((2, 2)).compute() # Create 2x2 list
    # Manual diagonal construction
    if w_data[0] < w_data[1]:
        # w_data matches sorted?
        d1, d2 = w_data[0], w_data[1]
    else:
        d1, d2 = w_data[0], w_data[1]
        
    # Construct diagonal matrix W
    W_mat = tf.array([[w_data[0], 0], [0, w_data[1]]])
    
    lhs = a @ v
    rhs = v @ W_mat
    
    assert mat_close(lhs, rhs, tol=1e-2)

def test_svd_simple():
    """Test SVD reconstruction."""
    # A = [[3, 2, 2], [2, 3, -2]]
    a = tf.array([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]])
    
    u, s, vh = la.svd(a)
    
    # Check shapes for reduced SVD
    # A is (2, 3). Reduced SVD -> U(2, 2), S(2,), Vh(2, 3)
    # Note: our reduced SVD implementation (M < N) returns S as (M,).
    
    assert u.shape.dims == (2, 2)
    assert s.shape.dims == (2,)
    assert vh.shape.dims == (2, 3)
    
    # Reconstruction: U @ diag(S) @ Vh = A
    # Construct diag(S) (2, 2)
    s_data = s.compute()
    S_mat = tf.array([[s_data[0], 0], [0, s_data[1]]])
    
    recon = u @ S_mat @ vh
    assert mat_close(recon, a, tol=1e-3)
    
    # Verify Singular Values (known for this matrix? 5 and 3)
    # 3+2=5, 3-2=1?
    # Sigma for [[3, 2, 2], [2, 3, -2]]: 5, 3
    s_sorted = sorted(s_data, reverse=True)
    assert math.isclose(s_sorted[0], 5.0, abs_tol=1e-3)
    assert math.isclose(s_sorted[1], 3.0, abs_tol=1e-3)

def test_matrix_rank():
    """Test matrix rank."""
    # Rank 2
    a = tf.eye(2)
    assert la.matrix_rank(a).compute()[0] == 2
    
    # Rank 1 (col 2 is 2*col 1)
    b = tf.array([[1, 2], [2, 4]])
    assert la.matrix_rank(b).compute()[0] == 1
    
def test_cond():
    """Test condition number."""
    # Identity -> 1
    i = tf.eye(2)
    assert math.isclose(la.cond(i).compute()[0], 1.0)
    
    # Singular -> inf
    b = tf.array([[1, 2], [2, 4]])
    val = la.cond(b).compute()[0]
    assert val == float('inf') or val > 1e10

