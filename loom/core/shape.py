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
loom Shape System.

This module defines the immutable Shape class for tensor dimensions.

ESTABLISHED FACTS:
- Shape is immutable (frozen dataclass)
- Shape supports NumPy-style broadcasting rules
- Pure Python implementation

DESIGN DECISIONS:
- Use frozen dataclass for immutability and hashability
- Broadcasting aligns dimensions from the right
- Shape of () represents a scalar

REFERENCE DOCUMENTATION:
- loom-native-complete.md Section 2.1
- loom-three-must-haves.md Section 1 (Broadcasting)

PHASE STATUS: Phase 0 (Skeleton)
"""

from dataclasses import dataclass
from typing import Tuple, Union, Iterator


@dataclass(frozen=True)
class Shape:
    """
    Immutable shape representation for tensors.
    
    Shape is a frozen (immutable) dataclass that wraps a tuple of dimensions.
    Immutability ensures shapes can be used as dictionary keys and prevents
    accidental modification.
    
    Attributes:
        dims: Tuple of dimension sizes (e.g., (3, 4, 5) for a 3D tensor)
    
    Properties:
        ndim: Number of dimensions
        size: Total number of elements (product of all dims)
    
    Example:
        >>> from loom.core import Shape
        >>> s = Shape((3, 4, 5))
        >>> s.ndim
        3
        >>> s.size
        60
        >>> s[0]
        3
    """
    
    dims: Tuple[int, ...]
    
    def __post_init__(self):
        """Validate dimensions after initialization."""
        # Validate all dimensions are non-negative integers
        for i, d in enumerate(self.dims):
            if not isinstance(d, int):
                raise TypeError(f"Dimension {i} must be int, got {type(d).__name__}")
            if d < 0:
                raise ValueError(f"Dimension {i} must be non-negative, got {d}")
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.dims)
    
    @property
    def size(self) -> int:
        """
        Total number of elements (product of all dimensions).
        
        Returns 1 for scalar (empty dims tuple).
        """
        if not self.dims:
            return 1
        result = 1
        for d in self.dims:
            result *= d
        return result
    
    def __getitem__(self, idx: int) -> int:
        """Get dimension at index."""
        return self.dims[idx]
    
    def __len__(self) -> int:
        """Return number of dimensions (same as ndim)."""
        return len(self.dims)
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over dimensions."""
        return iter(self.dims)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Shape{self.dims}"
    
    def __str__(self) -> str:
        """Human-readable string."""
        return str(self.dims)


# =============================================================================
# BROADCASTING FUNCTIONS
# =============================================================================

def broadcast_shapes(shape_a: Shape, shape_b: Shape) -> Shape:
    """
    Compute broadcast shape following NumPy broadcasting rules.
    
    Broadcasting Rules (from NumPy):
    1. Align shapes from right to left
    2. Dimensions are compatible if:
       - They are equal, OR
       - One of them is 1
    3. Missing dimensions are treated as 1
    
    Args:
        shape_a: First shape
        shape_b: Second shape
    
    Returns:
        Broadcast result shape
    
    Raises:
        ValueError: If shapes cannot be broadcast together
    
    Example:
        >>> broadcast_shapes(Shape((3, 1)), Shape((1, 4)))
        Shape((3, 4))
        >>> broadcast_shapes(Shape((2, 3, 4)), Shape((3, 4)))
        Shape((2, 3, 4))
    """
    dims_a = list(shape_a.dims)
    dims_b = list(shape_b.dims)
    
    # Pad shorter shape with 1s on the left
    max_ndim = max(len(dims_a), len(dims_b))
    dims_a = [1] * (max_ndim - len(dims_a)) + dims_a
    dims_b = [1] * (max_ndim - len(dims_b)) + dims_b
    
    # Compute result dimensions
    result = []
    for i, (da, db) in enumerate(zip(dims_a, dims_b)):
        if da == db:
            result.append(da)
        elif da == 1:
            result.append(db)
        elif db == 1:
            result.append(da)
        else:
            raise ValueError(
                f"Cannot broadcast shapes {shape_a} and {shape_b}: "
                f"dimension {i} has sizes {da} and {db} (neither is 1)"
            )
    
    return Shape(tuple(result))


def shapes_broadcastable(shape_a: Shape, shape_b: Shape) -> bool:
    """
    Check if two shapes can be broadcast together.
    
    Args:
        shape_a: First shape
        shape_b: Second shape
    
    Returns:
        True if shapes are broadcastable, False otherwise
    """
    try:
        broadcast_shapes(shape_a, shape_b)
        return True
    except ValueError:
        return False


def infer_matmul_shape(shape_a: Shape, shape_b: Shape) -> Shape:
    """
    Infer output shape for matrix multiplication.
    
    For 2D matrices: (m, k) @ (k, n) -> (m, n)
    For higher dimensions: last 2 dims are matrix, others are batched
    
    Args:
        shape_a: Left operand shape (must be at least 2D)
        shape_b: Right operand shape (must be at least 2D)
    
    Returns:
        Result shape
    
    Raises:
        ValueError: If shapes are incompatible for matmul
    """
    if shape_a.ndim < 2:
        raise ValueError(f"matmul requires at least 2D tensor, got shape {shape_a}")
    if shape_b.ndim < 2:
        raise ValueError(f"matmul requires at least 2D tensor, got shape {shape_b}")
    
    # Check inner dimensions match
    if shape_a.dims[-1] != shape_b.dims[-2]:
        raise ValueError(
            f"matmul shape mismatch: {shape_a} @ {shape_b} "
            f"(inner dims {shape_a.dims[-1]} != {shape_b.dims[-2]})"
        )
    
    # Result is (..., m, n) where m from a, n from b
    # Batch dimensions are broadcast
    batch_a = shape_a.dims[:-2]
    batch_b = shape_b.dims[:-2]
    
    if batch_a and batch_b:
        # Broadcast batch dimensions
        batch_shape = broadcast_shapes(
            Shape(batch_a), Shape(batch_b)
        )
        batch_dims = batch_shape.dims
    else:
        batch_dims = batch_a or batch_b
    
    result_dims = batch_dims + (shape_a.dims[-2], shape_b.dims[-1])
    return Shape(result_dims)

