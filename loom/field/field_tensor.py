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
Field Tensors for spatially coherent data.
"""

from typing import Tuple, List, Union, Any
from loom.core.tensor import Tensor, array
from loom.interpolate import interp1d
import math

class FieldTensor:
    """
    A Tensor wrapper that understands spatial fields.
    Useful for signals or physical fields where sampling is required.
    
    Supports 1D, 2D, and N-dimensional linear/bilinear/N-linear interpolation.
    """
    def __init__(self, tensor: Tensor):
        self._tensor = tensor
        self.shape = tensor.shape
        self.ndim = tensor.ndim

    @property
    def base_tensor(self) -> Tensor:
        return self._tensor

    def sample(self, coords: List[float]) -> float:
        """
        Sample the field at continuous coordinates using N-linear interpolation.
        
        Args:
            coords: List of N coordinates, one per dimension
            
        Returns:
            Interpolated value at the given coordinates
            
        Raises:
            ValueError: If coords length doesn't match tensor dimensions
        """
        if len(coords) != self.ndim:
            raise ValueError(f"coords length ({len(coords)}) must match tensor ndim ({self.ndim})")
        
        data = self._tensor.tolist()
        
        if self.ndim == 1:
            return self._sample_1d(data, coords[0])
        elif self.ndim == 2:
            return self._sample_2d(data, coords)
        else:
            return self._sample_nd(data, coords, self.shape.dims)

    def _sample_1d(self, data: List, x: float) -> float:
        """Linear interpolation for 1D data."""
        n = len(data)
        if n == 0:
            raise ValueError("Cannot sample from empty tensor")
        if n == 1:
            return data[0]
            
        x = max(0, min(x, n - 1))
        x0 = int(math.floor(x))
        x1 = min(x0 + 1, n - 1)
        dx = x - x0
        
        return data[x0] * (1 - dx) + data[x1] * dx
    
    def _sample_2d(self, data: List[List], coords: List[float]) -> float:
        """Bilinear interpolation for 2D data."""
        r, c = coords
        rows = len(data)
        cols = len(data[0]) if rows > 0 else 0
        
        if rows == 0 or cols == 0:
            raise ValueError("Cannot sample from empty tensor")
        
        r = max(0, min(r, rows - 1))
        c = max(0, min(c, cols - 1))
        
        r0 = int(math.floor(r))
        r1 = min(r0 + 1, rows - 1)
        c0 = int(math.floor(c))
        c1 = min(c0 + 1, cols - 1)
        
        dr = r - r0
        dc = c - c0
        
        v00 = data[r0][c0]
        v01 = data[r0][c1]
        v10 = data[r1][c0]
        v11 = data[r1][c1]
        
        # Bilinear interpolation
        v0 = v00 * (1 - dc) + v01 * dc
        v1 = v10 * (1 - dc) + v11 * dc
        
        return v0 * (1 - dr) + v1 * dr
    
    def _sample_nd(self, data: Any, coords: List[float], shape: Tuple[int, ...]) -> float:
        """
        N-linear interpolation for N-dimensional data.
        
        Uses recursive strategy: interpolate along first dimension, then recurse.
        """
        if len(shape) == 1:
            return self._sample_1d(data, coords[0])
        
        if len(shape) == 2:
            return self._sample_2d(data, coords)
        
        # N-dimensional case (N >= 3)
        dim_size = shape[0]
        x = coords[0]
        x = max(0, min(x, dim_size - 1))
        
        x0 = int(math.floor(x))
        x1 = min(x0 + 1, dim_size - 1)
        dx = x - x0
        
        # Recursively interpolate at x0 and x1 slices
        remaining_coords = coords[1:]
        remaining_shape = shape[1:]
        
        v0 = self._sample_nd(data[x0], remaining_coords, remaining_shape)
        v1 = self._sample_nd(data[x1], remaining_coords, remaining_shape)
        
        return v0 * (1 - dx) + v1 * dx

    def tolist(self):
        return self._tensor.tolist()
    
    def __repr__(self):
        return f"FieldTensor(shape={self.shape.dims})"

