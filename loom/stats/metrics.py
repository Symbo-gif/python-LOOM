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
Statistical Summary Metrics.
"""

from typing import Union, List, Optional
import loom as tf
from loom.core.tensor import Tensor, array

def skew(t: Union[Tensor, List]) -> Tensor:
    """
    Compute Fisher-Pearson coefficient of skewness.
    skew = m3 / m2^(3/2)
    """
    t = array(t)
    mu = t.mean()
    diff = t - mu
    m3 = (diff**3).mean()
    m2 = (diff**2).mean()
    # Avoid div by zero
    if m2.item() < 1e-15:
        return array(0.0)
    return m3 / (m2**1.5)

def kurtosis(t: Union[Tensor, List], fisher: bool = True) -> Tensor:
    """
    Compute kurtosis (fourth moment).
    If fisher=True, excess kurtosis is returned (kurtosis - 3.0).
    """
    t = array(t)
    mu = t.mean()
    diff = t - mu
    m4 = (diff**4).mean()
    m2 = (diff**2).mean()
    if m2.item() < 1e-15:
        return array(0.0)
    k = m4 / (m2**2)
    return k - 3.0 if fisher else k

def percentile(t: Union[Tensor, List], q: float) -> Tensor:
    """
    Compute the q-th percentile of the data.
    q in [0, 100].
    """
    t = array(t)
    flat = t.flatten()
    sorted_data = sorted(flat.tolist())
    n = len(sorted_data)
    if n == 0:
        return array(float('nan'))
    
    if q < 0: q = 0
    if q > 100: q = 100
    
    idx = (q / 100.0) * (n - 1)
    low = int(idx)
    high = min(low + 1, n - 1)
    frac = idx - low
    
    res = sorted_data[low] * (1 - frac) + sorted_data[high] * frac
    return array(res)

def median(t: Union[Tensor, List]) -> Tensor:
    """Compute the median."""
    return percentile(t, 50.0)

def std(t: Union[Tensor, List], ddof: int = 0) -> Tensor:
    """Compute standard deviation."""
    t = array(t)
    var = variance(t, ddof=ddof)
    return var.sqrt()

def variance(t: Union[Tensor, List], ddof: int = 0) -> Tensor:
    """Compute variance."""
    t = array(t)
    mu = t.mean()
    n = t.size
    if n <= ddof:
        return array(0.0)
    return ((t - mu)**2).sum() / (n - ddof)

