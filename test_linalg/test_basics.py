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
Tests for linalg basics.
"""

import pytest
import loom as tf
import loom.linalg as la
from loom.core.tensor import Tensor
import math

def test_dot_alias():
    """Test dot equals matmul for 2D."""
    a = tf.eye(2)
    b = tf.array([[2, 0], [0, 2]])
    res = la.dot(a, b)
    assert res.compute() == [2.0, 0.0, 0.0, 2.0]

def test_norm_frobenius():
    """Test default Frobenius norm."""
    # [3, 4] -> sqrt(9+16) = 5
    a = tf.array([3, 4])
    n = la.norm(a)
    assert math.isclose(n.compute()[0], 5.0)
    
    # [[1, 1], [1, 1]] -> sqrt(4) = 2
    b = tf.ones((2, 2))
    n2 = la.norm(b)
    assert math.isclose(n2.compute()[0], 2.0)

def test_trace_simple():
    """Test trace of 2D matrix."""
    a = tf.array([[1, 2], [3, 4]])
    # 1 + 5 = 6? No 1 + 4 = 5
    t = la.trace(a)
    assert t.compute()[0] == 5.0
    
    id3 = tf.eye(3)
    assert la.trace(id3).compute()[0] == 3.0

def test_vdot_real():
    """Test vdot on real vectors."""
    a = tf.array([1, 2])
    b = tf.array([3, 4])
    # 1*3 + 2*4 = 3 + 8 = 11
    res = la.vdot(a, b)
    assert res.compute()[0] == 11.0

def test_tensor_dot_method():
    """Test tensor.dot() method."""
    a = tf.array([1, 2])
    b = tf.array([3, 4])
    res = a.dot(b)
    assert res.compute()[0] == 11.0

