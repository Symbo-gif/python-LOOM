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
Distance metrics for spatial algorithms.
"""

from typing import Union
import math
from loom.core.tensor import Tensor, array

def minkowski(u: Union[Tensor, list], v: Union[Tensor, list], p: float = 2.0) -> float:
    """Minkowski distance."""
    if isinstance(u, Tensor): u = u.tolist()
    if isinstance(v, Tensor): v = v.tolist()
    
    if len(u) != len(v):
        raise ValueError("Vectors must have same length")
        
    return array(sum(abs(ui - vi)**p for ui, vi in zip(u, v))**(1.0/p))

def euclidean(u: Union[Tensor, list], v: Union[Tensor, list]) -> float:
    """Euclidean distance (L2)."""
    return minkowski(u, v, p=2.0)

def manhattan(u: Union[Tensor, list], v: Union[Tensor, list]) -> float:
    """Manhattan distance (L1)."""
    return minkowski(u, v, p=1.0)

