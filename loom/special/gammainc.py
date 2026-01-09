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
Regularized Incomplete Gamma Function.
"""

import math
from typing import Union
from loom.core.tensor import Tensor, array


def gammainc(a: Union[float, Tensor], x: Union[float, Tensor]) -> Union[float, Tensor]:
    """
    Regularized lower incomplete gamma function: P(a, x) = γ(a, x) / Γ(a).
    
    Uses series expansion for x < a+1 and continued fraction for x >= a+1.
    """
    if isinstance(a, Tensor):
        a_val = a.item()
    else:
        a_val = a
    if isinstance(x, Tensor):
        x_val = x.item()
    else:
        x_val = x
        
    if x_val < 0 or a_val <= 0:
        raise ValueError("gammainc requires x >= 0 and a > 0")
    
    if x_val == 0:
        return 0.0
    
    from loom.special.gamma import gamma as gamma_func
    
    if x_val < a_val + 1:
        # Series expansion
        result = _gammainc_series(a_val, x_val)
    else:
        # Continued fraction
        result = 1.0 - _gammainc_cf(a_val, x_val)
    
    return result


def _gammainc_series(a: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    Series expansion for lower incomplete gamma: γ(a, x) = x^a * e^(-x) * Σ (x^n / (a+1)...(a+n))
    Returns regularized P(a, x).
    """
    from loom.special.gamma import gamma as gamma_func
    
    if x == 0:
        return 0.0
    
    # Compute using series
    ap = a
    term = 1.0 / a
    sum_val = term
    
    for n in range(1, max_iter):
        ap += 1.0
        term *= x / ap
        sum_val += term
        if abs(term) < tol * abs(sum_val):
            break
    
    # P(a, x) = (x^a * e^(-x) * sum) / Γ(a)
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    return sum_val * math.exp(log_prefactor)


def _gammainc_cf(a: float, x: float, max_iter: int = 200, tol: float = 1e-12) -> float:
    """
    Continued fraction for upper incomplete gamma: Γ(a, x) / Γ(a).
    Returns Q(a, x) = 1 - P(a, x).
    """
    # Lentz's algorithm for continued fraction
    tiny = 1e-30
    
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    
    for i in range(1, max_iter):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    
    # Q(a, x) = (x^a * e^(-x) * h) / Γ(a)
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    return h * math.exp(log_prefactor)


def gammaincc(a: Union[float, Tensor], x: Union[float, Tensor]) -> Union[float, Tensor]:
    """
    Regularized upper incomplete gamma function: Q(a, x) = 1 - P(a, x).
    """
    return 1.0 - gammainc(a, x)

