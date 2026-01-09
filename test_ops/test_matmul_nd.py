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

import pytest
import loom as tf

def test_matmul_3d_broadcast():
    # (2, 2, 2) @ (2, 2)
    # Batch size 2
    a = tf.array([[[1, 0], [0, 1]], 
                  [[2, 0], [0, 2]]])
    b = tf.array([[1, 2], [3, 4]])
    
    # Result should be:
    # Slice 0: I @ B = B
    # Slice 1: 2I @ B = 2B
    res = a @ b
    assert res.shape.dims == (2, 2, 2)
    
    expected = [[[1, 2], [3, 4]],
                [[2, 4], [6, 8]]]
    assert res.tolist() == expected

def test_matmul_4d_broadcast():
    # (2, 1, 2, 2) @ (1, 3, 2, 2) -> (2, 3, 2, 2)
    a_shape = (2, 1, 2, 2)
    b_shape = (1, 3, 2, 2)
    
    a = tf.ones(a_shape)
    b = tf.eye(2).reshape((1, 1, 2, 2)) # Eye expanded to match rank?
    # Actually tf.ones((1,3,2,2)) is easier
    b = tf.ones(b_shape)
    
    res = a @ b
    assert res.shape.dims == (2, 3, 2, 2)
    # Each 2x2 matmul is ones(2,2) @ ones(2,2) = [[2,2],[2,2]]
    val = res.tolist()[0][0][0][0]
    assert val == 2.0

def test_matmul_1d_nd():
    # (3,) @ (2, 3, 4) -> (2, 4)
    # Promote a to (1, 3), result (2, 1, 4), squeeze -> (2, 4)
    a = tf.array([1, 1, 1])
    b = tf.ones((2, 3, 4))
    
    res = a @ b
    assert res.shape.dims == (2, 4)
    # [1,1,1] @ ones(3,4) = [3,3,3,3]
    for row in res.tolist():
        for val in row:
            assert val == 3.0

def test_matmul_nd_1d():
    # (2, 3, 4) @ (4,) -> (2, 3)
    a = tf.ones((2, 3, 4))
    b = tf.array([1, 1, 1, 1])
    
    res = a @ b
    assert res.shape.dims == (2, 3)
    for row in res.tolist():
        for val in row:
            assert val == 4.0

