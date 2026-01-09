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
Manipulation operations: transpose, reshape, etc.
"""

from typing import Tuple, List, Any, Optional
from loom.core.shape import Shape
# UnaryOp is in arithmetic.py
from loom.ops.arithmetic import UnaryOp


class TransposeOp(UnaryOp):
    """
    Transpose operation. Permutes axes.
    """
    def __init__(self, axes: Optional[Tuple[int, ...]] = None):
        self.axes = axes
        
    @property
    def name(self) -> str:
        return f"transpose({self.axes})"
    
    def infer_shape(self, a_tensor: 'Tensor') -> Shape:
        dims = a_tensor.shape.dims
        ndim = len(dims)
        
        if self.axes is None:
            # Reverse order (standard matrix transpose)
            new_dims = tuple(reversed(dims))
            return Shape(new_dims)
        else:
            if len(self.axes) != ndim:
                 raise ValueError(f"Transpose axes {self.axes} do not match ndim {ndim}")
            
            new_dims = tuple(dims[i] for i in self.axes)
            return Shape(new_dims)
            
    def execute(self, a_tensor: 'Tensor') -> List[Any]:
        # Get flattened data
        data = self._get_data(a_tensor)
        dims = a_tensor.shape.dims
        ndim = len(dims)
        
        if self.axes is None:
            perm = tuple(reversed(range(ndim)))
        else:
            perm = self.axes
            
        # Calculate size
        size = 1
        for d in dims: size *= d
        
        # New shape
        new_dims = [dims[i] for i in perm]
        
        # Optimization for 2D standard transpose
        if ndim == 2 and (self.axes is None or self.axes == (1, 0)):
             # Unflatten
             rows, cols = dims
             nested = []
             for i in range(rows):
                 nested.append(data[i*cols : (i+1)*cols])
                 
             transposed = list(zip(*nested))
             
             flat = []
             for row in transposed:
                 flat.extend(row)
             return flat

        # General N-D Transpose implementation
        result = [0] * size
        
        # Precompute strides for input
        curr = 1
        old_strides_rev = []
        for d in reversed(dims):
            old_strides_rev.append(curr)
            curr *= d
        old_strides = list(reversed(old_strides_rev))
        
        # Iterate over output indices
        for i in range(size):
             # 1. Convert flat output index 'i' to N-D coordinates in new shape
             current_idx = i
             coords = []
             for d in reversed(new_dims):
                 coords.insert(0, current_idx % d)
                 current_idx //= d
                 
             # 2. Map new coordinates back to old coordinates using inverse permutation
             # new_coord[k] corresponds to old_coord[perm[k]]
             old_coords = [0] * ndim
             for k in range(ndim):
                 old_coords[perm[k]] = coords[k]
                 
             # 3. Convert old N-D coordinates to flat input index
             in_idx = 0
             for k in range(ndim):
                 in_idx += old_coords[k] * old_strides[k]
                 
             # 4. Copy value
             result[i] = data[in_idx]
             
        return result


def create_transpose_op(a, axes=None):
    return TransposeOp(axes)


class ReshapeOp(UnaryOp):
    """
    Reshape operation.
    """
    def __init__(self, target_shape: Tuple[int, ...]):
        self.target_shape = target_shape
        
    @property
    def name(self) -> str:
        return f"reshape({self.target_shape})"
        
    def infer_shape(self, a_tensor: 'Tensor') -> Shape:
        curr_size = a_tensor.shape.size
        new_shape = list(self.target_shape)
        
        # Handle -1 in shape
        if -1 in new_shape:
            neg_idx = new_shape.index(-1)
            other_size = 1
            for i, d in enumerate(new_shape):
                if i != neg_idx:
                    other_size *= d
            new_shape[neg_idx] = curr_size // other_size
            
        res = Shape(tuple(new_shape))
        if res.size != curr_size:
            raise ValueError(f"Cannot reshape tensor of size {curr_size} into shape {self.target_shape}")
        return res
        
    def execute(self, a_tensor: 'Tensor') -> List[Any]:
        # Reshape doesn't change order of elements (matching NumPy behavior)
        return self._get_data(a_tensor)


def create_reshape_op(shape: Tuple[int, ...]):
    return ReshapeOp(shape)

