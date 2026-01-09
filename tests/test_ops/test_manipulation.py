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
Tests for manipulation ops (transpose).
"""

import pytest
import loom as tf

def test_transpose_2d():
    """Test 2D transpose."""
    a = tf.array([[1, 2, 3], [4, 5, 6]])
    # (2, 3) -> (3, 2)
    # [[1, 4], [2, 5], [3, 6]]
    
    t = a.T
    assert t.shape.dims == (3, 2)
    res = t.compute()
    assert res == [1, 4, 2, 5, 3, 6]
    
def test_transpose_with_axes():
    """Test explicit axes."""
    # 3D tensor (2, 3, 4)
    # Permute to (4, 2, 3) -> axes=(2, 0, 1)
    a_shape = (2, 3, 4)
    a = tf.ones(a_shape)
    
    t = a.transpose(axes=(2, 0, 1))
    assert t.shape.dims == (4, 2, 3)
    
def test_matrix_transpose_function():
    """Test linalg.matrix_transpose."""
    import loom.linalg as la
    
    # 3D: (2, 3, 4) -> (2, 4, 3)
    a = tf.ones((2, 3, 4))
    t = la.matrix_transpose(a)
    assert t.shape.dims == (2, 4, 3)

