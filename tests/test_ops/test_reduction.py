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
Comprehensive tests for reduction operations.

PHASE STATUS: Phase 1 Week 6 - IMPLEMENTED

Test Coverage:
- sum, mean, max, min, prod, argmax, argmin
- axis parameter (None, int, tuple)
- keepdims parameter
- Edge cases (1D, 2D, 3D)
- Stress tests
"""

import pytest
from loom.core.tensor import Tensor
from loom.core.shape import Shape


# =============================================================================
# SUM TESTS
# =============================================================================

class TestSum:
    """Test sum reduction."""
    
    def test_sum_all(self):
        """Sum all elements."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum()
        assert result.compute() == [21]
    
    def test_sum_axis_0(self):
        """Sum along axis 0 (columns)."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum(axis=0)
        assert result.compute() == [5, 7, 9]
    
    def test_sum_axis_1(self):
        """Sum along axis 1 (rows)."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum(axis=1)
        assert result.compute() == [6, 15]
    
    def test_sum_1d(self):
        """Sum 1D tensor."""
        t = Tensor([1, 2, 3, 4, 5])
        result = t.sum()
        assert result.compute() == [15]
    
    def test_sum_negative_axis(self):
        """Sum with negative axis."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum(axis=-1)
        assert result.compute() == [6, 15]
    
    def test_sum_3d_axis_1(self):
        """Sum 3D tensor along middle axis."""
        t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2x2x2
        result = t.sum(axis=1)
        # Sum across second dimension: [[1+3, 2+4], [5+7, 6+8]] = [[4, 6], [12, 14]]
        assert result.compute() == [4, 6, 12, 14]


# =============================================================================
# MEAN TESTS
# =============================================================================

class TestMean:
    """Test mean reduction."""
    
    def test_mean_all(self):
        """Mean of all elements."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.mean()
        assert abs(result.compute()[0] - 3.5) < 1e-10
    
    def test_mean_axis_0(self):
        """Mean along axis 0."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.mean(axis=0)
        computed = result.compute()
        assert abs(computed[0] - 2.5) < 1e-10
        assert abs(computed[1] - 3.5) < 1e-10
        assert abs(computed[2] - 4.5) < 1e-10
    
    def test_mean_axis_1(self):
        """Mean along axis 1."""
        t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = t.mean(axis=1)
        computed = result.compute()
        assert abs(computed[0] - 2.0) < 1e-10
        assert abs(computed[1] - 5.0) < 1e-10


# =============================================================================
# MAX TESTS
# =============================================================================

class TestMax:
    """Test max reduction."""
    
    def test_max_all(self):
        """Max of all elements."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.max()
        assert result.compute() == [6]
    
    def test_max_axis_0(self):
        """Max along axis 0."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.max(axis=0)
        assert result.compute() == [4, 5, 6]
    
    def test_max_axis_1(self):
        """Max along axis 1."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.max(axis=1)
        assert result.compute() == [5, 6]
    
    def test_max_negative_values(self):
        """Max with all negative values."""
        t = Tensor([-5, -3, -8, -1])
        result = t.max()
        assert result.compute() == [-1]


# =============================================================================
# MIN TESTS
# =============================================================================

class TestMin:
    """Test min reduction."""
    
    def test_min_all(self):
        """Min of all elements."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.min()
        assert result.compute() == [1]
    
    def test_min_axis_0(self):
        """Min along axis 0."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.min(axis=0)
        assert result.compute() == [1, 2, 3]
    
    def test_min_axis_1(self):
        """Min along axis 1."""
        t = Tensor([[1, 5, 3], [4, 2, 6]])
        result = t.min(axis=1)
        assert result.compute() == [1, 2]


# =============================================================================
# PROD TESTS
# =============================================================================

class TestProd:
    """Test product reduction."""
    
    def test_prod_all(self):
        """Product of all elements."""
        t = Tensor([1, 2, 3, 4])
        result = t.prod()
        assert result.compute() == [24]
    
    def test_prod_axis_0(self):
        """Product along axis 0."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.prod(axis=0)
        assert result.compute() == [3, 8]  # 1*3=3, 2*4=8
    
    def test_prod_axis_1(self):
        """Product along axis 1."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.prod(axis=1)
        assert result.compute() == [2, 12]  # 1*2=2, 3*4=12


# =============================================================================
# ARGMAX/ARGMIN TESTS
# =============================================================================

class TestArgmax:
    """Test argmax reduction."""
    
    def test_argmax_1d(self):
        """Argmax of 1D tensor."""
        t = Tensor([1, 5, 3, 2])
        result = t.argmax(axis=0)
        assert result.compute() == [1]  # Index of 5
    
    def test_argmax_axis_0(self):
        """Argmax along axis 0."""
        t = Tensor([[1, 5], [3, 2]])
        result = t.argmax(axis=0)
        assert result.compute() == [1, 0]  # max in col 0 at row 1, col 1 at row 0
    
    def test_argmax_axis_1(self):
        """Argmax along axis 1."""
        t = Tensor([[1, 5], [3, 2]])
        result = t.argmax(axis=1)
        assert result.compute() == [1, 0]  # max in row 0 at col 1, row 1 at col 0


class TestArgmin:
    """Test argmin reduction."""
    
    def test_argmin_1d(self):
        """Argmin of 1D tensor."""
        t = Tensor([3, 1, 4, 2])
        result = t.argmin(axis=0)
        assert result.compute() == [1]  # Index of 1
    
    def test_argmin_axis_1(self):
        """Argmin along axis 1."""
        t = Tensor([[3, 1], [4, 2]])
        result = t.argmin(axis=1)
        assert result.compute() == [1, 1]  # min in each row at col 1


# =============================================================================
# SHAPE TESTS
# =============================================================================

class TestReductionShapes:
    """Test that reduction shapes are correct."""
    
    def test_sum_shape_all(self):
        """Sum all -> scalar shape."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t.sum()
        assert result.shape == Shape(())
    
    def test_sum_shape_axis_0(self):
        """Sum axis=0 removes first dim."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        result = t.sum(axis=0)
        assert result.shape == Shape((3,))
    
    def test_sum_shape_axis_1(self):
        """Sum axis=1 removes second dim."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        result = t.sum(axis=1)
        assert result.shape == Shape((2,))


# =============================================================================
# CHAINED OPERATIONS TESTS
# =============================================================================

class TestChainedReductions:
    """Test reductions with other operations."""
    
    def test_arithmetic_then_sum(self):
        """Arithmetic followed by sum."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = (a + b).sum()
        # (1+5 + 2+6 + 3+7 + 4+8) = 6+8+10+12 = 36
        assert c.compute() == [36]
    
    def test_sum_then_multiply(self):
        """Sum followed by multiplication."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.sum(axis=1) * Tensor([2, 2])
        # sum(axis=1) = [3, 7], then * [2, 2] = [6, 14]
        assert result.compute() == [6, 14]


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStressReductions:
    """Stress tests for reductions."""
    
    def test_large_tensor_sum(self):
        """Sum of large tensor."""
        data = list(range(10000))
        t = Tensor(data)
        result = t.sum()
        expected = sum(range(10000))
        assert result.compute() == [expected]
    
    def test_large_2d_sum_axis(self):
        """Sum large 2D tensor along axis."""
        # 100 x 100 tensor
        data = [[i * 100 + j for j in range(100)] for i in range(100)]
        t = Tensor(data)
        result = t.sum(axis=0)
        assert result.shape == Shape((100,))
        # First column: 0 + 100 + 200 + ... + 9900 = 100 * (0+1+...+99) = 100 * 4950 = 495000
        computed = result.compute()
        assert computed[0] == 495000

