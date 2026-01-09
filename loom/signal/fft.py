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
Fast Fourier Transform.
"""

import math
import cmath
from typing import List, Union
from loom.core.tensor import Tensor, array

def fft(x: Union[List, Tensor]) -> Tensor:
    """
    1D Fast Fourier Transform.
    """
    if isinstance(x, Tensor):
        x_list = x.tolist()
    else:
        x_list = x
    
    n = len(x_list)
    if n <= 1:
        res = x_list
    elif (n & (n - 1)) == 0:
        res = _cooley_tukey_fft(x_list)
    else:
        res = _naive_dft(x_list)
        
    from loom.core.dtype import DType
    return Tensor(res, dtype=DType.COMPLEX128)

def ifft(X: Union[List, Tensor]) -> Tensor:
    """
    Inverse 1D Fast Fourier Transform.
    """
    if isinstance(X, Tensor):
        X_list = X.tolist()
    else:
        X_list = X
        
    n = len(X_list)
    if n == 0:
        from loom.core.dtype import DType
        return Tensor([], dtype=DType.COMPLEX128)
        
    # Conjugate
    X_conj = [complex(v.real, -v.imag) for v in X_list]
    
    # FFT (returns a Tensor, so we need to get list back for internal work)
    res_tensor = fft(X_conj)
    res = res_tensor.tolist()
    
    # Conjugate and scale
    final = [complex(v.real / n, -v.imag / n) for v in res]
    
    from loom.core.dtype import DType
    return Tensor(final, dtype=DType.COMPLEX128)

def _cooley_tukey_fft(x: List) -> List[complex]:
    n = len(x)
    if n <= 1:
        return [complex(v) for v in x]
        
    even = _cooley_tukey_fft(x[0::2])
    odd = _cooley_tukey_fft(x[1::2])
    
    T = [cmath.exp(-2j * math.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + T[k] for k in range(n // 2)] + \
           [even[k] - T[k] for k in range(n // 2)]

def _naive_dft(x: List) -> List[complex]:
    n = len(x)
    X = []
    for k in range(n):
        val = 0j
        for t in range(n):
            angle = 2 * math.pi * k * t / n
            val += x[t] * cmath.exp(-1j * angle)
        X.append(val)
    return X

