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
Tests for matrix functions: expm, sqrtm.
"""

import pytest
import math
import loom as tf
import loom.linalg as la

def test_expm_diagonal():
    # exp([[1, 0], [0, 2]]) = [[e^1, 0], [0, e^2]]
    A = tf.array([[1.0, 0.0], [0.0, 2.0]])
    res = la.expm(A)
    
    expected = [[math.exp(1), 0], [0, math.exp(2)]]
    res_list = res.tolist()
    
    assert math.isclose(res_list[0][0], expected[0][0], rel_tol=1e-5)
    assert math.isclose(res_list[1][1], expected[1][1], rel_tol=1e-5)
    assert abs(res_list[0][1]) < 1e-7
    assert abs(res_list[1][0]) < 1e-7

def test_expm_nilpotent():
    # A = [[0, 1], [0, 0]] => A^2 = 0
    # exp(A) = I + A = [[1, 1], [0, 1]]
    A = tf.array([[0.0, 1.0], [0.0, 0.0]])
    res = la.expm(A)
    
    assert res.tolist() == [[1.0, 1.0], [0.0, 1.0]]

def test_sqrtm_positive_definite():
    # B = [[1, 0], [0, 4]] => sqrtm(B) = [[1, 0], [0, 2]]
    B = tf.array([[1.0, 0.0], [0.0, 4.0]])
    res = la.sqrtm(B)
    
    expected = [[1.0, 0.0], [0.0, 2.0]]
    res_list = res.tolist()
    assert math.isclose(res_list[0][0], 1.0, abs_tol=1e-7)
    assert math.isclose(res_list[1][1], 2.0, abs_tol=1e-7)

def test_sqrtm_roundtrip():
    # A = [[4, 1], [1, 9]]
    A = tf.array([[4.0, 1.0], [1.0, 9.0]])
    S = la.sqrtm(A)
    
    # S @ S should be A
    recon = S @ S
    recon_list = recon.tolist()
    assert math.isclose(recon_list[0][0], 4.0, abs_tol=1e-6)
    assert math.isclose(recon_list[1][1], 9.0, abs_tol=1e-6)
    assert math.isclose(recon_list[0][1], 1.0, abs_tol=1e-6)

