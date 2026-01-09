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
Convolution operations.
"""

from typing import List, Union
from loom.core.tensor import Tensor, array
from loom.core.shape import Shape

def convolve(a: Union[List, Tensor], v: Union[List, Tensor], mode: str = 'full') -> Tensor:
    """
    1D discrete convolution.
    
    Modes:
    - 'full': (N+M-1) size
    - 'same': (N) size
    - 'valid': (N-M+1) size
    """
    if isinstance(a, Tensor): a_list = a.tolist()
    else: a_list = a
    
    if isinstance(v, Tensor): v_list = v.tolist()
    else: v_list = v
    
    if not a_list or not v_list:
        return Tensor([])
        
    n = len(a_list)
    m = len(v_list)
    
    full_len = n + m - 1
    res = [0.0] * full_len
    
    for i in range(full_len):
        start_a = max(0, i - m + 1)
        end_a = min(n, i + 1)
        
        for k in range(start_a, end_a):
            res[i] += a_list[k] * v_list[i-k]
            
    if mode == 'full':
        final = res
    elif mode == 'same':
        start = (m - 1) // 2
        final = res[start : start + n]
    elif mode == 'valid':
        if m > n:
            final = []
        else:
            final = res[m-1 : n]
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    return Tensor(final)

def convolve2d(image: Union[List, Tensor], kernel: Union[List, Tensor], mode: str = 'full') -> Tensor:
    """
    Basic 2D convolution for grids.
    """
    if isinstance(image, Tensor): image_list = image.tolist()
    else: image_list = image
    
    if isinstance(kernel, Tensor): kernel_list = kernel.tolist()
    else: kernel_list = kernel
    
    # Assume 2D lists
    rows_i = len(image_list)
    if rows_i == 0: return Tensor([[]])
    cols_i = len(image_list[0])
    
    rows_k = len(kernel_list)
    if rows_k == 0: return Tensor(image_list)
    cols_k = len(kernel_list[0])
    
    # Target dimensions (full)
    rows_out = rows_i + rows_k - 1
    cols_out = cols_i + cols_k - 1
    
    full_res = [[0.0] * cols_out for _ in range(rows_out)]
    
    for r in range(rows_out):
        for c in range(cols_out):
            val = 0.0
            for kr in range(rows_k):
                ir = r - kr
                if 0 <= ir < rows_i:
                    for kc in range(cols_k):
                        ic = c - kc
                        if 0 <= ic < cols_i:
                            val += image_list[ir][ic] * kernel_list[kr][kc]
            full_res[r][c] = val
            
    if mode == 'full':
        final = full_res
    elif mode == 'same':
        start_r = (rows_k - 1) // 2
        start_c = (cols_k - 1) // 2
        final = []
        for r in range(start_r, start_r + rows_i):
            final.append(full_res[r][start_c : start_c + cols_i])
    elif mode == 'valid':
        if rows_k > rows_i or cols_k > cols_i:
            final = [[]]
        else:
            final = []
            for r in range(rows_k - 1, rows_i):
                final.append(full_res[r][cols_k - 1 : cols_i])
    else:
         raise ValueError(f"Unknown mode: {mode}")
         
    return Tensor(final)

