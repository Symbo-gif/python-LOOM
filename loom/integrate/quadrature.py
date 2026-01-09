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
Numerical integration and quadrature.
"""

from typing import Callable, Union, Optional, Tuple, List
import loom as tf
from loom.core.tensor import Tensor, array
from loom.core.dtype import DType

def trapezoid(y: Union[List, Tensor], x: Optional[Union[List, Tensor]] = None, dx: float = 1.0, axis: int = -1) -> Union[float, Tensor]:
    """
    Integrate using the composite trapezoidal rule.
    """
    # Convert inputs to list/Tensor independent format
    if isinstance(y, Tensor):
        y_data = y.tolist()
        ndim = y.ndim
    else:
        y_data = y
        # Infer ndim roughly
        ndim = 1
        if isinstance(y_data, list) and len(y_data) > 0 and isinstance(y_data[0], list):
            ndim = 2 # Simple check
            
    # Handle negative axis
    if axis < 0: axis += ndim
    
    # 1D Case
    if ndim == 1 or (isinstance(y_data, list) and (not y_data or not isinstance(y_data[0], list))):
         return _trapezoid_1d(y_data, x, dx)

    # 2D Case support (Minimal for tests)
    if ndim == 2:
        if axis == 1:
            # Integrate each row
            result = [_trapezoid_1d(row, x, dx).item() if isinstance(row, Tensor) or isinstance(row, list) else _trapezoid_1d(row, x, dx) for row in y_data]
            # Unwrap if items are Tensors (since _trapezoid_1d now returns Tensor)
            result = [r.item() if hasattr(r, 'item') else r for r in result]
            return Tensor(result, dtype=DType.FLOAT64)
        elif axis == 0:
            # Integrate each col
            # zip(*y_data) transposes list of lists
            transposed = list(zip(*y_data))
            result = []
            for col in transposed:
                # x might need handling if it matches axis 0, but tests usually use scalar dx for this
                val = _trapezoid_1d(list(col), x, dx)
                result.append(val.item() if hasattr(val, 'item') else val)
            return Tensor(result, dtype=DType.FLOAT64)
            
    raise NotImplementedError(f"Trapezoid not fully implemented for ndim={ndim} axis={axis}")

def _trapezoid_1d(y_data: List, x: Optional[Union[List, Tensor]] = None, dx: float = 1.0) -> Tensor:
    if x is not None:
        if isinstance(x, Tensor):
            x_data = x.tolist()
        else:
            x_data = list(x)
        
        if len(x_data) != len(y_data):
            raise ValueError("x and y must have same length")
        
        total = 0.0
        for i in range(len(y_data) - 1):
            total += (x_data[i+1] - x_data[i]) * (y_data[i] + y_data[i+1]) / 2.0
        return Tensor(total, dtype=DType.FLOAT64)
    else:
        # Uniform spacing
        n = len(y_data)
        if n < 2: return Tensor(0.0, dtype=DType.FLOAT64)
        return Tensor(dx * (sum(y_data) - 0.5 * (y_data[0] + y_data[-1])), dtype=DType.FLOAT64)

def simpson(y: Union[List, Tensor], x: Optional[Union[List, Tensor]] = None, dx: float = 1.0) -> float:
    """
    Integrate using Simpson's 1/3 rule.
    """
    if isinstance(y, Tensor):
        y_data = y.tolist()
    else:
        y_data = list(y)
        
    n = len(y_data)
    if n < 3:
        return trapezoid(y, x, dx) # Fallback if too few points
    
    if x is not None:
        # Check if x is uniform
        if isinstance(x, Tensor):
             x_list = x.tolist()
        else:
             x_list = list(x)
        
        if len(x_list) > 1:
            dxs = [x_list[i+1] - x_list[i] for i in range(len(x_list)-1)]
            # Check if all dx are close to the first one
            is_uniform = all(abs(d - dxs[0]) < 1e-5 for d in dxs)
            
            if is_uniform:
                # Use uniform logic with calculated dx
                return simpson(y, x=None, dx=dxs[0])
        
        # Simpson rule for non-uniform spacing is complex, 
        # usually falls back to trapezoid or local parabolic fit.
        # For simplicity in this native core, we only support uniform Simpson.
        return trapezoid(y, x, dx)

    # Uniform spacing
    # Simpson's 1/3 rule: h/3 * [y0 + 4*sum(y_odd) + 2*sum(y_even) + yn]
    # Requires odd number of points (even number of intervals)
    if n % 2 == 0:
        # Handle N intervals (N+1 points) where N is odd by doing Simpson on N-1 and Trapezoid on last
        main_part = simpson(y_data[:-1], dx=dx)
        # main_part is a Tensor now (recurvsive call), so get item
        if isinstance(main_part, Tensor):
             main_part = main_part.item()
        
        last_interval = dx * (y_data[-2] + y_data[-1]) / 2.0
        return Tensor(main_part + last_interval, dtype=DType.FLOAT64)
    
    h = dx
    s = y_data[0] + y_data[-1]
    for i in range(1, n - 1, 2):
        s += 4 * y_data[i]
    for i in range(2, n - 2, 2):
        s += 2 * y_data[i]
        
    return Tensor(s * h / 3.0, dtype=DType.FLOAT64)

