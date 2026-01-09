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
loom Indexing Operations.

This module implements advanced indexing operations for tensors.

ESTABLISHED FACTS (Phase 1 Implementation):
- Implements NumPy-compatible advanced indexing
- Supports integer, slice, boolean, and fancy indexing
- Pure Python implementation

REFERENCE DOCUMENTATION:
- gap_analysis_complete.md (CRITICAL GAP #1: Advanced Indexing)
- NumPy indexing documentation

PHASE STATUS: Phase 1 Week 7 - IMPLEMENTED
"""

from typing import Union, Tuple, List, Any, Optional
from loom.ops.arithmetic import Operation
from loom.core.shape import Shape


class IndexOp(Operation):
    """
    Base class for indexing operations.
    
    Handles tensor element access via various indexing methods.
    """
    
    def __init__(self, indices):
        """
        Initialize index operation.
        
        Args:
            indices: Index specification (int, slice, list, tuple, etc.)
        """
        self.indices = indices
    
    @property
    def name(self) -> str:
        return "index"
    
    def _get_data(self, arg) -> List:
        """Get computed data from tensor."""
        if not hasattr(arg, '_dense_data'):
            return [arg]
        if hasattr(arg, '_cached_result') and arg._cached_result is not None:
            return arg._cached_result
        if arg._dense_data is not None:
            return arg._dense_data
        raise ValueError(f"Tensor has no computed data: {arg}")
    
    def infer_shape(self, a) -> Shape:
        """Infer output shape based on indexing."""
        if not hasattr(a, 'shape'):
            return Shape(())
        
        input_shape = a.shape
        indices = self._normalize_indices(self.indices, input_shape.dims)
        
        # Basic shape inference
        result_dims = []
        for i, idx in enumerate(indices):
            if i >= len(input_shape.dims):
                break
            dim_size = input_shape.dims[i]
            
            if isinstance(idx, int):
                # Integer index removes dimension
                pass
            elif isinstance(idx, slice):
                # Slice preserves dimension with new size
                start, stop, step = idx.indices(dim_size)
                size = max(0, (stop - start + step - 1) // step) if step > 0 else max(0, (start - stop - step - 1) // (-step))
                result_dims.append(size)
            elif isinstance(idx, (list, tuple)):
                # Fancy indexing with list/tuple
                if all(isinstance(x, bool) for x in idx):
                    # Boolean indexing
                    result_dims.append(sum(1 for x in idx if x))
                else:
                    # Integer list indexing
                    result_dims.append(len(idx))
            else:
                result_dims.append(dim_size)
        
        # Add remaining dimensions
        for i in range(len(indices), len(input_shape.dims)):
            result_dims.append(input_shape.dims[i])
        
        if not result_dims:
            return Shape(())
        return Shape(tuple(result_dims))
    
    def _normalize_indices(self, indices, shape_dims) -> Tuple:
        """Normalize index specification to tuple."""
        if not isinstance(indices, tuple):
            indices = (indices,)
        return indices
    
    def execute(self, a) -> List:
        """Execute indexing operation."""
        data = self._get_data(a)
        shape = a.shape if hasattr(a, 'shape') else Shape(())
        
        return self._index_data(data, shape.dims, self.indices)
    
    def _index_data(self, data: List, dims: Tuple[int, ...], indices) -> List:
        """Index into flat data with multi-dimensional indexing."""
        if not dims:
            # Scalar
            return data
        
        indices = self._normalize_indices(indices, dims)
        
        # Handle single axis indexing for now
        if len(indices) == 1:
            idx = indices[0]
            if isinstance(idx, int):
                return self._index_int_1d(data, dims, idx)
            elif isinstance(idx, slice):
                return self._index_slice_1d(data, dims, idx)
            elif isinstance(idx, (list, tuple)):
                if all(isinstance(x, bool) for x in idx):
                    return self._index_bool_1d(data, dims, idx)
                else:
                    return self._index_fancy_1d(data, dims, idx)
        
        # Multi-axis indexing
        return self._index_multi(data, dims, indices)
    
    def _compute_strides(self, dims: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute row-major strides."""
        if not dims:
            return ()
        strides = [1]
        for d in reversed(dims[1:]):
            strides.append(strides[-1] * d)
        return tuple(reversed(strides))
    
    def _index_int_1d(self, data: List, dims: Tuple[int, ...], idx: int) -> List:
        """Index with single integer along first axis."""
        if idx < 0:
            idx = dims[0] + idx
        
        if len(dims) == 1:
            return [data[idx]]
        
        # Get the slice for this index
        stride = 1
        for d in dims[1:]:
            stride *= d
        
        start = idx * stride
        return data[start:start + stride]
    
    def _index_slice_1d(self, data: List, dims: Tuple[int, ...], slc: slice) -> List:
        """Index with slice along first axis."""
        start, stop, step = slc.indices(dims[0])
        
        if len(dims) == 1:
            return data[start:stop:step]
        
        # Get elements along first axis
        stride = 1
        for d in dims[1:]:
            stride *= d
        
        result = []
        for i in range(start, stop, step):
            chunk_start = i * stride
            result.extend(data[chunk_start:chunk_start + stride])
        
        return result
    
    def _index_bool_1d(self, data: List, dims: Tuple[int, ...], mask: List) -> List:
        """Boolean indexing along first axis."""
        if len(dims) == 1:
            return [data[i] for i, m in enumerate(mask) if m]
        
        stride = 1
        for d in dims[1:]:
            stride *= d
        
        result = []
        for i, m in enumerate(mask):
            if m:
                chunk_start = i * stride
                result.extend(data[chunk_start:chunk_start + stride])
        
        return result
    
    def _index_fancy_1d(self, data: List, dims: Tuple[int, ...], indices: List) -> List:
        """Fancy (list) indexing along first axis."""
        if len(dims) == 1:
            return [data[i if i >= 0 else dims[0] + i] for i in indices]
        
        stride = 1
        for d in dims[1:]:
            stride *= d
        
        result = []
        for i in indices:
            if i < 0:
                i = dims[0] + i
            chunk_start = i * stride
            result.extend(data[chunk_start:chunk_start + stride])
        
        return result
    
    def _index_multi(self, data: List, dims: Tuple[int, ...], indices: Tuple) -> List:
        """Multi-dimensional indexing."""
        strides = self._compute_strides(dims)
        
        # Generate all index combinations
        index_ranges = []
        for i, idx in enumerate(indices):
            if i >= len(dims):
                break
            
            if isinstance(idx, int):
                if idx < 0:
                    idx = dims[i] + idx
                index_ranges.append([idx])
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(dims[i])
                index_ranges.append(list(range(start, stop, step)))
            elif isinstance(idx, (list, tuple)):
                if all(isinstance(x, bool) for x in idx):
                    index_ranges.append([j for j, m in enumerate(idx) if m])
                else:
                    index_ranges.append([x if x >= 0 else dims[i] + x for x in idx])
            else:
                index_ranges.append(list(range(dims[i])))
        
        # Add remaining dimensions
        for i in range(len(indices), len(dims)):
            index_ranges.append(list(range(dims[i])))
        
        # Generate all combinations using nested loops
        result = []
        self._collect_indexed(data, strides, index_ranges, 0, [], result)
        
        return result
    
    def _collect_indexed(self, data, strides, ranges, depth, current, result):
        """Recursively collect indexed elements."""
        if depth == len(ranges):
            flat_idx = sum(c * s for c, s in zip(current, strides))
            result.append(data[flat_idx])
            return
        
        for idx in ranges[depth]:
            self._collect_indexed(data, strides, ranges, depth + 1, current + [idx], result)


def create_index_op(indices) -> IndexOp:
    """Create an indexing operation."""
    return IndexOp(indices)


# =============================================================================
# SLICING UTILITIES
# =============================================================================

def slice_to_indices(slc: slice, dim_size: int) -> List[int]:
    """Convert slice to list of indices."""
    start, stop, step = slc.indices(dim_size)
    return list(range(start, stop, step))

