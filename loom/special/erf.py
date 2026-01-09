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
Error function and related integrals.
"""

import math
from typing import Union
from loom.core.tensor import Tensor, array

def erf(x: Union[float, Tensor]) -> Union[float, Tensor]:
    """
    Error function.
    """
    if isinstance(x, Tensor):
        data = x.compute()
        results = [math.erf(v) for v in data]
        return Tensor(results, shape=x.shape.dims, dtype=x.dtype)
    return math.erf(x)

def erfc(x: Union[float, Tensor]) -> Union[float, Tensor]:
    """
    Complementary error function: 1 - erf(x).
    """
    if isinstance(x, Tensor):
        data = x.compute()
        results = [math.erfc(v) for v in data]
        return Tensor(results, shape=x.shape.dims, dtype=x.dtype)
    return math.erfc(x)

