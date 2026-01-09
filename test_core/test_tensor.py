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
Tests for core/tensor.py

PHASE STATUS: Phase 0 (Smoke tests for skeleton)
"""

import pytest
from loom.core.tensor import Tensor, Symbol
from loom.core.shape import Shape
from loom.core.dtype import DType, parse_dtype


class TestTensorCreation:
    """Test tensor initialization."""
    
    def test_tensor_from_list_1d(self):
        """Create 1D tensor from list."""
        t = Tensor([1, 2, 3])
        assert t.shape == Shape((3,))
        assert t.ndim == 1
        assert t.size == 3
    
    def test_tensor_from_list_2d(self):
        """Create 2D tensor from nested list."""
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == Shape((2, 2))
        assert t.ndim == 2
        assert t.size == 4
    
    def test_tensor_from_scalar(self):
        """Create scalar tensor."""
        t = Tensor(42)
        assert t.shape == Shape(())
        assert t.ndim == 0
        assert t.size == 1
    
    def test_tensor_explicit_shape(self):
        """Create tensor with explicit shape."""
        t = Tensor([1, 2, 3, 4], shape=(2, 2))
        assert t.shape == Shape((2, 2))
    
    def test_tensor_dtype_default(self):
        """Default dtype is float64."""
        t = Tensor([1, 2, 3])
        assert t.dtype == DType.FLOAT64
    
    def test_tensor_dtype_explicit(self):
        """Explicit dtype setting."""
        t = Tensor([1, 2, 3], dtype="int64")
        assert t.dtype == DType.INT64


class TestTensorProperties:
    """Test tensor property access."""
    
    def test_is_numeric(self):
        """Numeric tensor has is_numeric=True."""
        t = Tensor([1, 2, 3])
        assert t.is_numeric
    
    def test_is_not_symbolic(self):
        """Numeric tensor has is_symbolic=False."""
        t = Tensor([1, 2, 3])
        assert not t.is_symbolic
    
    def test_compute_returns_data(self):
        """compute() returns internal data for numeric tensor."""
        t = Tensor([[1, 2], [3, 4]])
        result = t.compute()
        assert result == [1, 2, 3, 4]


class TestSymbol:
    """Test symbolic variable creation."""
    
    def test_symbol_creation(self):
        """Create symbol."""
        x = Symbol('x')
        assert x.symbol_name == 'x'
        assert x.dtype == DType.SYMBOLIC
    
    def test_symbol_repr(self):
        """Symbol has nice repr."""
        x = Symbol('theta')
        assert "theta" in repr(x)


class TestDType:
    """Test dtype system."""
    
    def test_parse_string(self):
        """Parse dtype from string."""
        assert parse_dtype("float32") == DType.FLOAT32
        assert parse_dtype("int64") == DType.INT64
    
    def test_parse_enum(self):
        """Parse dtype from enum."""
        assert parse_dtype(DType.FLOAT64) == DType.FLOAT64
    
    def test_parse_none_default(self):
        """None returns default float64."""
        assert parse_dtype(None) == DType.FLOAT64
    
    def test_invalid_dtype_raises(self):
        """Invalid dtype string raises ValueError."""
        with pytest.raises(ValueError):
            parse_dtype("invalid_type")


class TestShape:
    """Test shape system."""
    
    def test_shape_creation(self):
        """Create shape from tuple."""
        s = Shape((2, 3, 4))
        assert s.ndim == 3
        assert s.size == 24
    
    def test_shape_indexing(self):
        """Index into shape."""
        s = Shape((2, 3, 4))
        assert s[0] == 2
        assert s[1] == 3
        assert s[2] == 4
    
    def test_shape_iteration(self):
        """Iterate over shape."""
        s = Shape((2, 3, 4))
        assert list(s) == [2, 3, 4]
    
    def test_scalar_shape(self):
        """Scalar shape is empty tuple."""
        s = Shape(())
        assert s.ndim == 0
        assert s.size == 1


class TestBroadcasting:
    """Test broadcasting shape inference."""
    
    def test_broadcast_same_shape(self):
        """Same shapes broadcast to same."""
        from loom.core.shape import broadcast_shapes
        s = Shape((2, 3))
        result = broadcast_shapes(s, s)
        assert result == Shape((2, 3))
    
    def test_broadcast_scalar(self):
        """Scalar broadcasts to any shape."""
        from loom.core.shape import broadcast_shapes
        a = Shape((2, 3))
        b = Shape(())
        result = broadcast_shapes(a, b)
        assert result == Shape((2, 3))
    
    def test_broadcast_1d_to_2d(self):
        """1D broadcasts into 2D."""
        from loom.core.shape import broadcast_shapes
        a = Shape((2, 3))
        b = Shape((3,))
        result = broadcast_shapes(a, b)
        assert result == Shape((2, 3))
    
    def test_broadcast_expand_dims(self):
        """Broadcast expands singleton dims."""
        from loom.core.shape import broadcast_shapes
        a = Shape((3, 1))
        b = Shape((1, 4))
        result = broadcast_shapes(a, b)
        assert result == Shape((3, 4))
    
    def test_broadcast_incompatible_raises(self):
        """Incompatible shapes raise ValueError."""
        from loom.core.shape import broadcast_shapes
        a = Shape((2, 3))
        b = Shape((4,))
        with pytest.raises(ValueError):
            broadcast_shapes(a, b)

