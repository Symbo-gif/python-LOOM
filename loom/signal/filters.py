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
Digital filters.
"""

from typing import Union, List, Optional
import loom as tf
from loom.core.tensor import Tensor, array

def lfilter(b: Union[Tensor, List], a: Union[Tensor, List], x: Union[Tensor, List]) -> Tensor:
    """
    Filter data with a linear filter.
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
                            - a[1]*y[n-1] - ... - a[na]*y[n-na]
    """
    b = array(b).tolist()
    a = array(a).tolist()
    x = array(x).tolist()
    
    if a[0] == 0:
        raise ValueError("a[0] cannot be zero")
    
    # Normalize if a[0] != 1
    if a[0] != 1.0:
        a = [val / a[0] for val in a]
        b = [val / a[0] for val in b]
        
    n = len(x)
    nb = len(b)
    na = len(a)
    
    y = [0.0] * n
    
    for i in range(n):
        # Forward part (b coefficients)
        for j in range(nb):
            if i - j >= 0:
                y[i] += b[j] * x[i-j]
        # Feedback part (a coefficients)
        for j in range(1, na):
            if i - j >= 0:
                y[i] -= a[j] * y[i-j]
                
    return array(y)

def filtfilt(b: Union[Tensor, List], a: Union[Tensor, List], x: Union[Tensor, List]) -> Tensor:
    """
    Zero-phase digital filtering by processing data in both forward and backward directions.
    """
    y_forward = lfilter(b, a, x)
    # Reverse y_forward
    y_rev = array(y_forward.tolist()[::-1])
    y_back_rev = lfilter(b, a, y_rev)
    # Reverse back
    return array(y_back_rev.tolist()[::-1])

