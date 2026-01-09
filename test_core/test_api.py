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
Tests for top-level API and factory functions.

PHASE STATUS: Phase 1 Finalization
"""

import pytest
import loom as tf
from loom.core.tensor import Tensor


def test_top_level_exports():
    """Verify that main API components are exported."""
    assert hasattr(tf, 'Tensor')
    assert hasattr(tf, 'array')
    assert hasattr(tf, 'zeros')
    assert hasattr(tf, 'ones')
    assert hasattr(tf, 'eye')
    assert hasattr(tf, 'random')
    assert hasattr(tf, 'conj')


def test_array_factory():
    """Test tf.array factory."""
    t = tf.array([1, 2, 3])
    assert isinstance(t, Tensor)
    assert t.compute() == [1, 2, 3]


def test_zeros_factory():
    """Test tf.zeros factory."""
    t = tf.zeros((2, 3))
    assert t.shape.dims == (2, 3)
    assert t.compute() == [0, 0, 0, 0, 0, 0]


def test_ones_factory():
    """Test tf.ones factory."""
    t = tf.ones((4,))
    assert t.compute() == [1, 1, 1, 1]


def test_full_factory():
    """Test tf.full factory."""
    t = tf.full((2, 2), 7)
    assert t.compute() == [7, 7, 7, 7]


def test_eye_factory():
    """Test tf.eye factory."""
    t = tf.eye(2)
    assert t.compute() == [1, 0, 0, 1]
    
    t2 = tf.eye(2, 3)
    assert t2.shape.dims == (2, 3)
    assert t2.compute() == [1, 0, 0, 0, 1, 0]


def test_integration_example():
    """Test the example from README."""
    a = tf.array([[1, 2], [3, 4]])
    b = tf.ones((2, 2))
    c = a + b
    res = c.compute()
    assert res == [2.0, 3.0, 4.0, 5.0]
    
    total = a.sum(axis=0)
    assert total.compute() == [4, 6]

