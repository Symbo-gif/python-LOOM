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
Tests for matrix multiplication operations (Phase 2).
"""

import pytest
import loom as tf
from loom.core.tensor import Tensor


def test_matmul_1d_dot():
    """Test 1D @ 1D (dot product)."""
    a = tf.array([1, 2, 3])
    b = tf.array([4, 5, 6])
    
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    c = a @ b
    
    assert c.shape.dims == ()
    assert c.compute() == [32.0]
    
    # Alias
    assert a.dot(b).compute() == [32.0]


def test_matmul_2d_basic():
    """Test 2D @ 2D matrix multiplication."""
    # A is 2x3
    a = tf.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    
    # B is 3x2
    b = tf.array([
        [7, 8],
        [9, 10],
        [11, 12]
    ])
    
    # C should be 2x2
    # C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    # C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    # C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    # C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    
    c = a @ b
    assert c.shape.dims == (2, 2)
    
    result = c.compute()
    expected = [58.0, 64.0, 139.0, 154.0]
    assert result == expected


def test_matmul_shapes_mismatch():
    """Test shape mismatch errors."""
    a = tf.zeros((2, 3))
    b = tf.zeros((4, 2)) # Inner dim 3 != 4
    
    with pytest.raises(ValueError, match="Matmul mismatch"):
        _ = a @ b


def test_matmul_identity():
    """Test multiplication with identity matrix."""
    a = tf.array([[1, 2], [3, 4]])
    i = tf.eye(2)
    
    # A @ I = A
    res = (a @ i).compute()
    assert res == [1.0, 2.0, 3.0, 4.0]
    
    # I @ A = A
    res2 = (i @ a).compute()
    assert res2 == [1.0, 2.0, 3.0, 4.0]

