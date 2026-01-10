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
Comprehensive tests for arithmetic operations.

PHASE STATUS: Phase 1 - IMPLEMENTED

Test Coverage:
- Basic operations (+, -, *, /, **, //, %)
- Unary operations (-a, abs)
- Broadcasting (scalar, 1D, 2D, 3D)
- Edge cases (zeros, negatives, large values)
- Stress tests (large tensors, chains)
"""

import pytest
import math
from loom.core.tensor import Tensor, Symbol
from loom.core.shape import Shape
from loom.core.dtype import DType


# =============================================================================
# BASIC ARITHMETIC TESTS
# =============================================================================

class TestAddition:
    """Test tensor addition."""
    
    def test_add_same_shape(self):
        """Add two tensors of same shape."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a + b
        assert c.tolist() == [[6, 8], [10, 12]]
    
    def test_add_scalar_right(self):
        """Add scalar to tensor."""
        a = Tensor([1, 2, 3])
        c = a + 10
        assert c.compute() == [11, 12, 13]
    
    def test_add_scalar_left(self):
        """Add tensor to scalar (reverse)."""
        a = Tensor([1, 2, 3])
        c = 10 + a
        assert c.compute() == [11, 12, 13]
    
    def test_add_broadcast_1d_to_2d(self):
        """Broadcast 1D tensor across 2D."""
        a = Tensor([[1, 2], [3, 4]])  # (2, 2)
        b = Tensor([10, 20])           # (2,)
        c = a + b
        assert c.tolist() == [[11, 22], [13, 24]]
    
    def test_add_broadcast_scalar_to_2d(self):
        """Broadcast scalar across 2D."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor(100)
        c = a + b
        assert c.tolist() == [[101, 102], [103, 104]]
    
    def test_add_negative_numbers(self):
        """Add with negative numbers."""
        a = Tensor([-1, -2, -3])
        b = Tensor([1, 2, 3])
        c = a + b
        assert c.compute() == [0, 0, 0]
    
    def test_add_floats(self):
        """Add floating point numbers."""
        a = Tensor([1.5, 2.5])
        b = Tensor([0.5, 0.5])
        c = a + b
        result = c.compute()
        assert abs(result[0] - 2.0) < 1e-10
        assert abs(result[1] - 3.0) < 1e-10


class TestSubtraction:
    """Test tensor subtraction."""
    
    def test_sub_same_shape(self):
        """Subtract two tensors of same shape."""
        a = Tensor([[5, 6], [7, 8]])
        b = Tensor([[1, 2], [3, 4]])
        c = a - b
        assert c.tolist() == [[4, 4], [4, 4]]
    
    def test_sub_scalar_right(self):
        """Subtract scalar from tensor."""
        a = Tensor([10, 20, 30])
        c = a - 5
        assert c.compute() == [5, 15, 25]
    
    def test_sub_scalar_left(self):
        """Subtract tensor from scalar."""
        a = Tensor([1, 2, 3])
        c = 10 - a
        assert c.compute() == [9, 8, 7]
    
    def test_sub_to_zero(self):
        """Subtraction resulting in zero."""
        a = Tensor([5, 5, 5])
        b = Tensor([5, 5, 5])
        c = a - b
        assert c.compute() == [0, 0, 0]


class TestMultiplication:
    """Test tensor multiplication."""
    
    def test_mul_same_shape(self):
        """Multiply two tensors of same shape."""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[2, 3], [4, 5]])
        c = a * b
        assert c.tolist() == [[2, 6], [12, 20]]
    
    def test_mul_scalar(self):
        """Multiply tensor by scalar."""
        a = Tensor([1, 2, 3])
        c = a * 2
        assert c.compute() == [2, 4, 6]
    
    def test_mul_by_zero(self):
        """Multiply by zero."""
        a = Tensor([1, 2, 3])
        c = a * 0
        assert c.compute() == [0, 0, 0]
    
    def test_mul_negative(self):
        """Multiply by negative."""
        a = Tensor([1, 2, 3])
        c = a * (-1)
        assert c.compute() == [-1, -2, -3]


class TestDivision:
    """Test tensor division."""
    
    def test_div_same_shape(self):
        """Divide two tensors of same shape."""
        a = Tensor([10.0, 20.0, 30.0])
        b = Tensor([2.0, 4.0, 5.0])
        c = a / b
        result = c.compute()
        assert result == [5.0, 5.0, 6.0]
    
    def test_div_scalar(self):
        """Divide tensor by scalar."""
        a = Tensor([10.0, 20.0, 30.0])
        c = a / 10
        result = c.compute()
        assert result == [1.0, 2.0, 3.0]
    
    def test_div_by_zero_positive(self):
        """Division by zero (positive numerator)."""
        a = Tensor([1.0])
        b = Tensor([0.0])
        c = a / b
        result = c.compute()
        assert result[0] == float('inf')
    
    def test_div_by_zero_negative(self):
        """Division by zero (negative numerator)."""
        a = Tensor([-1.0])
        b = Tensor([0.0])
        c = a / b
        result = c.compute()
        assert result[0] == float('-inf')


class TestPower:
    """Test tensor power."""
    
    def test_pow_square(self):
        """Square values."""
        a = Tensor([1, 2, 3])
        c = a ** 2
        assert c.compute() == [1, 4, 9]
    
    def test_pow_cube(self):
        """Cube values."""
        a = Tensor([2, 3])
        c = a ** 3
        assert c.compute() == [8, 27]
    
    def test_pow_zero(self):
        """Any number to power 0 is 1."""
        a = Tensor([1, 2, 3, 100])
        c = a ** 0
        assert c.compute() == [1, 1, 1, 1]
    
    def test_pow_one(self):
        """Any number to power 1 is itself."""
        a = Tensor([1, 2, 3])
        c = a ** 1
        assert c.compute() == [1, 2, 3]


class TestFloorDivision:
    """Test floor division."""
    
    def test_floordiv_basic(self):
        """Basic floor division."""
        a = Tensor([7, 8, 9])
        c = a // 3
        assert c.compute() == [2, 2, 3]
    
    def test_floordiv_negative(self):
        """Floor division with negatives."""
        a = Tensor([-7])
        c = a // 3
        assert c.compute() == [-3]


class TestModulo:
    """Test modulo operation."""
    
    def test_mod_basic(self):
        """Basic modulo."""
        a = Tensor([7, 8, 9])
        c = a % 3
        assert c.compute() == [1, 2, 0]


class TestUnaryOperations:
    """Test unary operations."""
    
    def test_neg(self):
        """Negation."""
        a = Tensor([1, -2, 3])
        c = -a
        assert c.compute() == [-1, 2, -3]
    
    def test_pos(self):
        """Positive (identity)."""
        a = Tensor([1, -2, 3])
        c = +a
        assert c.compute() == [1, -2, 3]
    
    def test_abs(self):
        """Absolute value."""
        a = Tensor([-1, 2, -3])
        c = abs(a)
        assert c.compute() == [1, 2, 3]


# =============================================================================
# COMPARISON TESTS
# =============================================================================

class TestComparison:
    """Test comparison operators."""
    
    def test_eq(self):
        """Equality."""
        a = Tensor([1, 2, 3])
        b = Tensor([1, 0, 3])
        c = a == b
        assert c.compute() == [1, 0, 1]
    
    def test_ne(self):
        """Not equal."""
        a = Tensor([1, 2, 3])
        b = Tensor([1, 0, 3])
        c = a != b
        assert c.compute() == [0, 1, 0]
    
    def test_lt(self):
        """Less than."""
        a = Tensor([1, 2, 3])
        b = Tensor([2, 2, 2])
        c = a < b
        assert c.compute() == [1, 0, 0]
    
    def test_le(self):
        """Less than or equal."""
        a = Tensor([1, 2, 3])
        b = Tensor([2, 2, 2])
        c = a <= b
        assert c.compute() == [1, 1, 0]
    
    def test_gt(self):
        """Greater than."""
        a = Tensor([1, 2, 3])
        b = Tensor([2, 2, 2])
        c = a > b
        assert c.compute() == [0, 0, 1]
    
    def test_ge(self):
        """Greater than or equal."""
        a = Tensor([1, 2, 3])
        b = Tensor([2, 2, 2])
        c = a >= b
        assert c.compute() == [0, 1, 1]


# =============================================================================
# BROADCASTING TESTS
# =============================================================================

class TestBroadcasting:
    """Test broadcasting scenarios."""
    
    def test_broadcast_1d_1d(self):
        """Broadcast (3,) + (1,)."""
        a = Tensor([1, 2, 3])
        b = Tensor([10])
        c = a + b
        # Note: Current implementation handles this
        assert c.shape == Shape((3,))
    
    def test_broadcast_2d_1d(self):
        """Broadcast (2,3) + (3,)."""
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = Tensor([10, 20, 30])
        c = a + b
        assert c.tolist() == [[11, 22, 33], [14, 25, 36]]
    
    def test_broadcast_row_col(self):
        """Broadcast (3,1) + (1,3)."""
        a = Tensor([[1], [2], [3]])  # (3, 1)
        b = Tensor([[10, 20, 30]])    # (1, 3)
        c = a + b
        assert c.shape == Shape((3, 3))
        result = c.tolist()
        assert result[0] == [11, 21, 31]
        assert result[1] == [12, 22, 32]
        assert result[2] == [13, 23, 33]


# =============================================================================
# CHAINED OPERATIONS TESTS
# =============================================================================

class TestChainedOperations:
    """Test chained arithmetic operations."""
    
    def test_chain_add_mul(self):
        """(a + b) * c"""
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = Tensor([2, 2, 2])
        result = (a + b) * c
        assert result.compute() == [10, 14, 18]
    
    def test_chain_complex(self):
        """a + b * c - d / e"""
        a = Tensor([10.0, 20.0])
        b = Tensor([2.0, 3.0])
        c = Tensor([3.0, 4.0])
        d = Tensor([6.0, 12.0])
        e = Tensor([2.0, 3.0])
        result = a + b * c - d / e
        # 10 + 2*3 - 6/2 = 10 + 6 - 3 = 13
        # 20 + 3*4 - 12/3 = 20 + 12 - 4 = 28
        expected = result.compute()
        assert abs(expected[0] - 13.0) < 1e-10
        assert abs(expected[1] - 28.0) < 1e-10
    
    def test_chain_power(self):
        """(a + b) ** 2"""
        a = Tensor([1, 2])
        b = Tensor([2, 3])
        result = (a + b) ** 2
        assert result.compute() == [9, 25]


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for performance and edge cases."""
    
    def test_large_tensor_add(self):
        """Add two large tensors."""
        size = 10000
        a = Tensor(list(range(size)))
        b = Tensor(list(range(size)))
        c = a + b
        result = c.compute()
        assert len(result) == size
        assert result[0] == 0
        assert result[size - 1] == 2 * (size - 1)
    
    def test_many_operations_chain(self):
        """Chain many operations."""
        a = Tensor([1.0, 2.0, 3.0])
        result = a
        for _ in range(100):
            result = result + Tensor([0.01, 0.01, 0.01])
        computed = result.compute()
        assert abs(computed[0] - 2.0) < 1e-6
        assert abs(computed[1] - 3.0) < 1e-6
        assert abs(computed[2] - 4.0) < 1e-6
    
    def test_deep_dag(self):
        """Deep DAG evaluation."""
        a = Tensor([1])
        for _ in range(50):
            a = a + Tensor([1])
        result = a.compute()
        assert result[0] == 51
    
    def test_large_broadcast(self):
        """Large broadcast operation."""
        a = Tensor([list(range(100)) for _ in range(100)])  # 100x100
        b = Tensor(list(range(100)))  # 100
        c = a + b
        result = c.compute()
        assert len(result) == 10000
        # First row: [0+0, 1+1, 2+2, ...] = [0, 2, 4, ...]
        assert result[0] == 0
        assert result[1] == 2


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_scalar_tensor(self):
        """Scalar tensor operations."""
        a = Tensor(5)
        b = Tensor(3)
        assert (a + b).compute() == [8]
        assert (a - b).compute() == [2]
        assert (a * b).compute() == [15]
    
    def test_empty_like_shape(self):
        """Operations with (1,) shaped tensors."""
        a = Tensor([5])
        b = Tensor([3])
        assert (a + b).compute() == [8]
    
    def test_large_values(self):
        """Operations with large values."""
        a = Tensor([1e10])
        b = Tensor([1e10])
        c = a + b
        assert c.compute()[0] == 2e10
    
    def test_small_values(self):
        """Operations with small values."""
        # Use float64 for precision with small values
        a = Tensor([1e-10], dtype='float64')
        b = Tensor([1e-10], dtype='float64')
        c = a + b
        assert abs(c.compute()[0] - 2e-10) < 1e-19
    
    def test_mixed_int_float(self):
        """Operations with mixed int and float."""
        a = Tensor([1, 2, 3])  # ints
        b = Tensor([0.5, 0.5, 0.5])  # floats
        c = a + b
        result = c.compute()
        assert result == [1.5, 2.5, 3.5]

