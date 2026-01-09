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
loom Reduction Operations.

This module implements aggregation operations that reduce tensor dimensions.

ESTABLISHED FACTS (Phase 1 Implementation):
- All reductions support optional axis parameter
- axis=None reduces to scalar (NumPy-compatible)
- keepdims parameter preserves reduced dimensions
- Pure Python implementation

REFERENCE DOCUMENTATION:
- loom-native-complete.md (ops/ module)
- gap_analysis_complete.md (CRITICAL GAP #2: Aggregations with axis)

PHASE STATUS: Phase 1 - IMPLEMENTED
"""

from typing import Optional, Union, Tuple, List
from loom.ops.arithmetic import Operation
from loom.core.shape import Shape


class ReductionOp(Operation):
    """
    Base class for reduction operations.
    
    Reduction operations reduce tensor along specified axis/axes,
    collapsing dimensions while computing aggregate values.
    """
    
    def __init__(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False):
        """
        Initialize reduction operation.
        
        Args:
            axis: Axis or axes to reduce. None = reduce all (scalar result).
            keepdims: If True, reduced axes are kept with size 1.
        """
        self.axis = axis
        self.keepdims = keepdims
    
    def _get_data(self, arg) -> List:
        """Get computed data from tensor or scalar."""
        if not hasattr(arg, '_dense_data'):
            return [arg]
        if hasattr(arg, '_cached_result') and arg._cached_result is not None:
            return arg._cached_result
        if arg._dense_data is not None:
            return arg._dense_data
        raise ValueError(f"Tensor has no computed data: {arg}")
    
    def _normalize_axis(self, axis: Optional[Union[int, Tuple[int, ...]]], ndim: int) -> Tuple[int, ...]:
        """
        Normalize axis specification to tuple of positive integers.
        
        Args:
            axis: None, int, or tuple of ints
            ndim: Number of dimensions in tensor
            
        Returns:
            Tuple of positive axis indices sorted ascending
        """
        if axis is None:
            # Reduce all axes
            return tuple(range(ndim))
        
        if isinstance(axis, int):
            axis = (axis,)
        
        # Convert negative indices
        result = []
        for ax in axis:
            if ax < 0:
                ax = ndim + ax
            if ax < 0 or ax >= ndim:
                raise ValueError(f"Axis {ax} is out of bounds for tensor with {ndim} dimensions")
            result.append(ax)
        
        return tuple(sorted(set(result)))
    
    def infer_shape(self, a) -> Shape:
        """Infer output shape after reduction."""
        if not hasattr(a, 'shape'):
            return Shape(())
        
        input_shape = a.shape
        ndim = input_shape.ndim
        axes = self._normalize_axis(self.axis, ndim)
        
        if self.keepdims:
            # Keep dimensions with size 1
            result_dims = tuple(
                1 if i in axes else d 
                for i, d in enumerate(input_shape.dims)
            )
        else:
            # Remove reduced dimensions
            result_dims = tuple(
                d for i, d in enumerate(input_shape.dims) 
                if i not in axes
            )
        
        if not result_dims:
            return Shape(())  # Scalar
        
        return Shape(result_dims)
    
    def _reduce_axis(
        self, 
        data: List, 
        shape: Shape, 
        axis: int, 
        reduce_func
    ) -> Tuple[List, Shape]:
        """
        Reduce along a single axis.
        
        Args:
            data: Flat data list
            shape: Current shape
            axis: Axis to reduce (0-indexed)
            reduce_func: Function to apply (sum, max, etc.)
            
        Returns:
            (reduced_data, new_shape)
        """
        dims = shape.dims
        ndim = len(dims)
        
        if ndim == 0:
            # Scalar - no reduction possible
            return data, shape
        
        # Compute strides
        strides = self._compute_strides(dims)
        
        # Compute output shape (remove the axis)
        out_dims = dims[:axis] + dims[axis+1:]
        if not out_dims:
            out_dims = ()
        
        out_shape = Shape(out_dims)
        out_size = out_shape.size
        
        # For each output position, collect values along reduction axis
        result = []
        axis_size = dims[axis]
        
        for out_idx in range(out_size):
            # Convert output flat index to multi-index
            if out_dims:
                out_multi = self._flat_to_multi(out_idx, out_dims)
            else:
                out_multi = ()
            
            # Insert axis position to get input multi-indices
            values = []
            for k in range(axis_size):
                # Build input multi-index by inserting k at axis position
                in_multi = out_multi[:axis] + (k,) + out_multi[axis:]
                in_flat = self._multi_to_flat(in_multi, strides)
                values.append(data[in_flat])
            
            # Apply reduction function
            result.append(reduce_func(values))
        
        return result, out_shape
    
    def _compute_strides(self, dims: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute strides for row-major indexing."""
        if not dims:
            return ()
        strides = [1]
        for d in reversed(dims[1:]):
            strides.append(strides[-1] * d)
        return tuple(reversed(strides))
    
    def _flat_to_multi(self, flat_idx: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
        """Convert flat index to multi-dimensional index."""
        if not dims:
            return ()
        result = []
        for d in reversed(dims):
            result.append(flat_idx % d)
            flat_idx //= d
        return tuple(reversed(result))
    
    def _multi_to_flat(self, multi_idx: Tuple[int, ...], strides: Tuple[int, ...]) -> int:
        """Convert multi-dimensional index to flat index."""
        if not multi_idx:
            return 0
        return sum(i * s for i, s in zip(multi_idx, strides))


class SumOp(ReductionOp):
    """Sum reduction: sum(a, axis=axis)"""
    
    @property
    def name(self) -> str:
        return "sum"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        shape = a.shape if hasattr(a, 'shape') else Shape(())
        ndim = shape.ndim
        
        if ndim == 0:
            return data
        
        axes = self._normalize_axis(self.axis, ndim)
        
        # Reduce each axis in order (highest first to preserve indices)
        current_data = data
        current_shape = shape
        
        for ax in sorted(axes, reverse=True):
            current_data, current_shape = self._reduce_axis(
                current_data, current_shape, ax, sum
            )
        
        # Handle keepdims
        if self.keepdims:
            result_shape = self.infer_shape(a)
            # Data is already reduced, shape is just for display
        
        return current_data


class MeanOp(ReductionOp):
    """Mean reduction: mean(a, axis=axis)"""
    
    @property
    def name(self) -> str:
        return "mean"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        shape = a.shape if hasattr(a, 'shape') else Shape(())
        ndim = shape.ndim
        
        if ndim == 0:
            return data
        
        axes = self._normalize_axis(self.axis, ndim)
        
        # Compute count of elements being averaged
        count = 1
        for ax in axes:
            count *= shape.dims[ax]
        
        current_data = data
        current_shape = shape
        
        for ax in sorted(axes, reverse=True):
            current_data, current_shape = self._reduce_axis(
                current_data, current_shape, ax, sum
            )
        
        # Divide by count
        return [x / count for x in current_data]


class MaxOp(ReductionOp):
    """Max reduction: max(a, axis=axis)"""
    
    @property
    def name(self) -> str:
        return "max"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        shape = a.shape if hasattr(a, 'shape') else Shape(())
        ndim = shape.ndim
        
        if ndim == 0:
            return data
        
        axes = self._normalize_axis(self.axis, ndim)
        
        current_data = data
        current_shape = shape
        
        for ax in sorted(axes, reverse=True):
            current_data, current_shape = self._reduce_axis(
                current_data, current_shape, ax, max
            )
        
        return current_data


class MinOp(ReductionOp):
    """Min reduction: min(a, axis=axis)"""
    
    @property
    def name(self) -> str:
        return "min"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        shape = a.shape if hasattr(a, 'shape') else Shape(())
        ndim = shape.ndim
        
        if ndim == 0:
            return data
        
        axes = self._normalize_axis(self.axis, ndim)
        
        current_data = data
        current_shape = shape
        
        for ax in sorted(axes, reverse=True):
            current_data, current_shape = self._reduce_axis(
                current_data, current_shape, ax, min
            )
        
        return current_data


class ProdOp(ReductionOp):
    """Product reduction: prod(a, axis=axis)"""
    
    @property
    def name(self) -> str:
        return "prod"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        shape = a.shape if hasattr(a, 'shape') else Shape(())
        ndim = shape.ndim
        
        if ndim == 0:
            return data
        
        axes = self._normalize_axis(self.axis, ndim)
        
        def product(values):
            result = 1
            for v in values:
                result *= v
            return result
        
        current_data = data
        current_shape = shape
        
        for ax in sorted(axes, reverse=True):
            current_data, current_shape = self._reduce_axis(
                current_data, current_shape, ax, product
            )
        
        return current_data


class ArgMaxOp(ReductionOp):
    """Argmax: index of maximum value along axis"""
    
    @property
    def name(self) -> str:
        return "argmax"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        shape = a.shape if hasattr(a, 'shape') else Shape(())
        ndim = shape.ndim
        
        if ndim == 0:
            return [0]
        
        axes = self._normalize_axis(self.axis, ndim)
        
        if len(axes) > 1:
            raise ValueError("argmax only supports single axis")
        
        axis = axes[0]
        
        def argmax_func(values):
            return max(range(len(values)), key=lambda i: values[i])
        
        result, _ = self._reduce_axis(data, shape, axis, argmax_func)
        return result


class ArgMinOp(ReductionOp):
    """Argmin: index of minimum value along axis"""
    
    @property
    def name(self) -> str:
        return "argmin"
    
    def execute(self, a) -> List:
        data = self._get_data(a)
        shape = a.shape if hasattr(a, 'shape') else Shape(())
        ndim = shape.ndim
        
        if ndim == 0:
            return [0]
        
        axes = self._normalize_axis(self.axis, ndim)
        
        if len(axes) > 1:
            raise ValueError("argmin only supports single axis")
        
        axis = axes[0]
        
        def argmin_func(values):
            return min(range(len(values)), key=lambda i: values[i])
        
        result, _ = self._reduce_axis(data, shape, axis, argmin_func)
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS (for tensor methods)
# =============================================================================

def create_sum_op(axis=None, keepdims=False) -> SumOp:
    return SumOp(axis=axis, keepdims=keepdims)

def create_mean_op(axis=None, keepdims=False) -> MeanOp:
    return MeanOp(axis=axis, keepdims=keepdims)

def create_max_op(axis=None, keepdims=False) -> MaxOp:
    return MaxOp(axis=axis, keepdims=keepdims)

def create_min_op(axis=None, keepdims=False) -> MinOp:
    return MinOp(axis=axis, keepdims=keepdims)

def create_prod_op(axis=None, keepdims=False) -> ProdOp:
    return ProdOp(axis=axis, keepdims=keepdims)

def create_argmax_op(axis=None) -> ArgMaxOp:
    return ArgMaxOp(axis=axis)

def create_argmin_op(axis=None) -> ArgMinOp:
    return ArgMinOp(axis=axis)

