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

def test_det_permutation_sign():
    # Matrix requiring pivot (swap)
    # [[0, 1], [1, 0]]
    # det is -1
    a = tf.array([[0.0, 1.0], [1.0, 0.0]])
    d = tf.linalg.solvers.det(a)
    assert abs(d.item() + 1.0) < 1e-12

def test_det_3x3_complex():
    # A bit more complex
    # [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    # det = 1*(0 - 24) - 2*(0 - 20) + 3*(0 - 5)
    # det = -24 + 40 - 15 = 1
    a = tf.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    d = tf.linalg.solvers.det(a)
    assert abs(d.item() - 1.0) < 1e-10

