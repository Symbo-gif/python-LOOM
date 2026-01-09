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
Special mathematical functions: Gamma, Beta, etc.
"""

import math
from typing import Union
from loom.core.tensor import Tensor, array

def gamma(x: Union[float, Tensor]) -> Union[float, Tensor]:
    """
    Gamma function.
    """
    if isinstance(x, Tensor):
        data = x.compute()
        results = [gamma(v) for v in data]
        return Tensor(results, shape=x.shape.dims, dtype=x.dtype)
    
    try:
        if x <= 0 and x == int(x):
            return float('inf')
        return math.gamma(x)
    except OverflowError:
        return float('inf')
    except ValueError:
        return float('inf')

def loggamma(x: Union[float, Tensor]) -> Union[float, Tensor]:
    """
    Logarithm of the absolute value of the Gamma function.
    """
    if isinstance(x, Tensor):
        data = x.compute()
        results = [math.lgamma(v) for v in data]
        return Tensor(results, shape=x.shape.dims, dtype=x.dtype)
    return math.lgamma(x)

def beta(a: Union[float, Tensor], b: Union[float, Tensor]) -> Union[float, Tensor]:
    """
    Beta function: B(a, b) = Gamma(a) * Gamma(b) / Gamma(a + b).
    """
    # Use loggamma for stability
    if isinstance(a, Tensor) or isinstance(b, Tensor):
        return (array(a).loggamma() + array(b).loggamma() - (array(a) + array(b)).loggamma()).exp()
    return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))

