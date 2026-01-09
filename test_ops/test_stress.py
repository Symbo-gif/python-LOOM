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
Stress tests for deep computation graphs.

PHASE STATUS: Phase 1 Finalization
"""

import pytest
import loom as tf


def test_deep_dag_stress():
    """Test performance and correctness with a very deep DAG."""
    # Create a tensor and add 1 to it 500 times
    curr = tf.array([1.0])
    for _ in range(500):
        curr = curr + 1.0
    
    # Evaluate
    result = curr.compute()
    assert result == [501.0]


def test_large_broadcast_stress():
    """Test memory and CPU stress with large broadcasting."""
    # (100, 1) + (1, 100) -> (100, 100) = 10,000 elements
    a = tf.ones((100, 1))
    b = tf.ones((1, 100))
    c = a + b
    
    result = c.compute()
    assert len(result) == 10000
    assert all(x == 2.0 for x in result)


def test_many_chained_reductions():
    """Test chaining multiple reductions."""
    a = tf.ones((10, 10, 10))
    # sum axis 0 -> (10, 10)
    # sum axis 0 -> (10,)
    # sum axis 0 -> ()
    result = a.sum(axis=0).sum(axis=0).sum(axis=0)
    assert result.compute() == [1000.0]

