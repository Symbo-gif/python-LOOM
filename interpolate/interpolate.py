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
Interpolation algorithms.
"""

from typing import Union, List, Optional, Tuple
from loom.core.tensor import Tensor, array
import math

def interp1d(x: Union[List, Tensor], y: Union[List, Tensor], kind: str = 'linear'):
    """
    1D interpolation.
    
    Returns a function that interpolates y = f(x).
    """
    if isinstance(x, Tensor): x = x.tolist()
    if isinstance(y, Tensor): y = y.tolist()
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
        
    # Sort by x
    points = sorted(zip(x, y))
    x_sort, y_sort = zip(*points)
    x_sort = list(x_sort)
    y_sort = list(y_sort)
    
    if kind == 'linear':
        def interpolator(x_new: Union[float, List, Tensor]):
            if isinstance(x_new, (int, float)):
                return _linear_interp_single(x_sort, y_sort, x_new)
            
            if isinstance(x_new, Tensor):
                vals = x_new.tolist()
                res = [_linear_interp_single(x_sort, y_sort, v) for v in vals]
                return array(res)
            
            return [_linear_interp_single(x_sort, y_sort, v) for v in x_new]
            
        return interpolator
    
    elif kind == 'cubic':
        return CubicSpline(x_sort, y_sort)
        
    else:
        raise ValueError(f"Unknown interpolation kind: {kind}")

def _linear_interp_single(x_data: List[float], y_data: List[float], x_val: float) -> float:
    """Linear interpolation for a single point."""
    if x_val <= x_data[0]: return y_data[0]
    if x_val >= x_data[-1]: return y_data[-1]
    
    # Binary search for interval
    low = 0
    high = len(x_data) - 1
    while high - low > 1:
        mid = (low + high) // 2
        if x_val >= x_data[mid]:
            low = mid
        else:
            high = mid
            
    x0, x1 = x_data[low], x_data[high]
    y0, y1 = y_data[low], y_data[high]
    
    return y0 + (y1 - y0) * (x_val - x0) / (x1 - x0)

class CubicSpline:
    """
    Cubic Spline interpolation.
    
    Implementation of natural cubic spline.
    """
    def __init__(self, x: List[float], y: List[float]):
        self.x = list(x)
        self.y = list(y)
        n = len(x) - 1
        
        # h[i] = x[i+1] - x[i]
        h = [x[i+1] - x[i] for i in range(n)]
        
        # System Ax = B for the second derivatives (m)
        # Natural spline: m[0] = m[n] = 0
        A = [[0.0] * (n-1) for _ in range(n-1)]
        B = [0.0] * (n-1)
        
        for i in range(n-1):
            if i > 0:
                A[i][i-1] = h[i] / 6
            A[i][i] = (h[i] + h[i+1]) / 3
            if i < n-2:
                A[i][i+1] = h[i+1] / 6
                
            B[i] = (y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i]
            
        # Solve using Thomas algorithm (tridiagonal)
        m_inner = self._solve_tridiagonal(A, B)
        self.m = [0.0] + m_inner + [0.0]
        
    def _solve_tridiagonal(self, A, B):
        n = len(B)
        if n == 0: return []
        
        # Extract diagonals
        b = [A[i][i] for i in range(n)]
        a = [A[i][i-1] for i in range(1, n)]
        c = [A[i][i+1] for i in range(n-1)]
        
        # Forward elimination
        for i in range(1, n):
            w = a[i-1] / b[i-1]
            b[i] = b[i] - w * c[i-1]
            B[i] = B[i] - w * B[i-1]
            
        # Backward substitution
        x = [0.0] * n
        x[n-1] = B[n-1] / b[n-1]
        for i in range(n-2, -1, -1):
            x[i] = (B[i] - c[i] * x[i+1]) / b[i]
            
        return x
        
    def __call__(self, x_val: Union[float, List, Tensor]):
        if isinstance(x_val, (int, float)):
            return self._eval_single(x_val)
        if isinstance(x_val, Tensor):
            return array([self._eval_single(v) for v in x_val.tolist()])
        return [self._eval_single(v) for v in x_val]
        
    def _eval_single(self, x_v: float) -> float:
        if x_v <= self.x[0]: return self.y[0]
        if x_v >= self.x[-1]: return self.y[-1]
        
        # Find interval
        idx = 0
        for i in range(len(self.x) - 1):
            if x_v >= self.x[i] and x_v <= self.x[i+1]:
                idx = i
                break
                
        h = self.x[idx+1] - self.x[idx]
        t = (x_v - self.x[idx]) / h
        
        # Cubic spline formula using second derivatives m
        val = (1-t)*self.y[idx] + t*self.y[idx+1] + \
              h*h/6 * ( (pow(1-t, 3) - (1-t))*self.m[idx] + (pow(t, 3) - t)*self.m[idx+1] )
        return val

