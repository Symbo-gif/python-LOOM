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
Comprehensive tests for indexing operations.

PHASE STATUS: Phase 1 Week 7 - IMPLEMENTED

Test Coverage:
- Integer indexing (single, multi-dim, negative)
- Slice indexing
- Boolean indexing
- Fancy (list) indexing
- Edge cases
"""

import pytest
from loom.core.tensor import Tensor
from loom.core.shape import Shape


# =============================================================================
# INTEGER INDEXING TESTS
# =============================================================================

class TestIntegerIndexing:
    """Test integer indexing."""
    
    def test_1d_single(self):
        """Single element from 1D tensor."""
        t = Tensor([1, 2, 3, 4, 5])
        assert t[2].compute() == [3]
    
    def test_1d_negative(self):
        """Negative index from 1D tensor."""
        t = Tensor([1, 2, 3, 4, 5])
        assert t[-1].compute() == [5]
        assert t[-2].compute() == [4]
    
    def test_2d_row(self):
        """Select row from 2D tensor."""
        t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert t[0].compute() == [1, 2, 3]
        assert t[1].compute() == [4, 5, 6]
    
    def test_2d_element(self):
        """Select single element from 2D tensor."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t[0, 1].compute() == [2]
        assert t[1, 2].compute() == [6]
    
    def test_3d_first_axis(self):
        """Index 3D tensor along first axis."""
        t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = t[0].compute()
        assert result == [1, 2, 3, 4]


# =============================================================================
# SLICE INDEXING TESTS
# =============================================================================

class TestSliceIndexing:
    """Test slice indexing."""
    
    def test_1d_slice(self):
        """Slice 1D tensor."""
        t = Tensor([1, 2, 3, 4, 5])
        assert t[1:4].compute() == [2, 3, 4]
    
    def test_1d_slice_step(self):
        """Slice with step."""
        t = Tensor([1, 2, 3, 4, 5, 6])
        assert t[::2].compute() == [1, 3, 5]
    
    def test_2d_slice_rows(self):
        """Slice rows from 2D tensor."""
        t = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = t[1:3].tolist()
        assert result == [[4, 5, 6], [7, 8, 9]]
    
    def test_2d_from_start(self):
        """Slice from start."""
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        result = t[:2].tolist()
        assert result == [[1, 2], [3, 4]]


# =============================================================================
# BOOLEAN INDEXING TESTS
# =============================================================================

class TestBooleanIndexing:
    """Test boolean (mask) indexing."""
    
    def test_1d_bool_mask(self):
        """Boolean mask on 1D tensor."""
        t = Tensor([1, 2, 3, 4, 5])
        mask = [True, False, True, False, True]
        result = t[mask].compute()
        assert result == [1, 3, 5]
    
    def test_2d_bool_mask_rows(self):
        """Boolean mask selecting rows."""
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        mask = [True, False, True]
        result = t[mask].tolist()
        assert result == [[1, 2], [5, 6]]


# =============================================================================
# FANCY INDEXING TESTS
# =============================================================================

class TestFancyIndexing:
    """Test fancy (list) indexing."""
    
    def test_1d_list_indices(self):
        """List of indices on 1D tensor."""
        t = Tensor([10, 20, 30, 40, 50])
        result = t[[0, 2, 4]].compute()
        assert result == [10, 30, 50]
    
    def test_1d_negative_indices(self):
        """Fancy indexing with negative indices."""
        t = Tensor([10, 20, 30, 40, 50])
        result = t[[-1, -3]].compute()
        assert result == [50, 30]
    
    def test_2d_list_rows(self):
        """Select specific rows."""
        t = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        result = t[[0, 3]].tolist()
        assert result == [[1, 2], [7, 8]]


# =============================================================================
# SHAPE TESTS
# =============================================================================

class TestIndexingShapes:
    """Test that indexed shapes are correct."""
    
    def test_2d_int_shape(self):
        """Integer index removes dimension."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t[0].shape == Shape((3,))
    
    def test_2d_slice_shape(self):
        """Slice preserves dimension."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t[0:1].shape == Shape((1, 3))
    
    def test_2d_multi_int_shape(self):
        """Multi-int reduces to scalar."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t[0, 1].shape == Shape(())


# =============================================================================
# CHAINED OPERATIONS TESTS
# =============================================================================

class TestChainedIndexing:
    """Test indexing combined with other operations."""
    
    def test_index_then_sum(self):
        """Index followed by sum."""
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        result = t[0].sum().compute()
        assert result == [6]  # 1+2+3
    
    def test_add_then_index(self):
        """Arithmetic followed by index."""
        a = Tensor([1, 2, 3, 4, 5])
        b = Tensor([10, 10, 10, 10, 10])
        c = (a + b)[2]
        assert c.compute() == [13]


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestIndexingStress:
    """Stress tests for indexing."""
    
    def test_large_tensor_index(self):
        """Index large tensor."""
        data = list(range(10000))
        t = Tensor(data)
        assert t[5000].compute() == [5000]
        assert t[-1].compute() == [9999]
    
    def test_large_slice(self):
        """Large slice."""
        data = list(range(1000))
        t = Tensor(data)
        result = t[100:200].compute()
        assert len(result) == 100
        assert result[0] == 100
        assert result[-1] == 199

