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
Lossy and lossless compression for Tensors.
"""

from typing import Dict, Any, Tuple
from loom.core.tensor import Tensor, array
from loom.linalg.decompositions import svd
import math

def compress_eigen(tensor: Tensor, rank: int = None) -> Dict[str, Any]:
    """
    Compress a 2D tensor using SVD (Singular Value Decomposition).
    Stores only the top 'rank' singular values and vectors.
    """
    if tensor.ndim != 2:
        raise ValueError("Eigen-compression currently only supports 2D tensors.")
        
    U, S, V = svd(tensor)
    
    # If rank not specified, keep 95% of energy
    if rank is None:
        s_vals = S.tolist()
        total_energy = sum(s*s for s in s_vals)
        energy_sum = 0
        rank = 0
        for s in s_vals:
            energy_sum += s*s
            rank += 1
            if energy_sum / (total_energy + 1e-15) >= 0.95:
                break
                
    # U and V (Vh) are Tensors. .tolist() is nested for 2D.
    u_list = U.tolist()
    U_small = [row[:rank] for row in u_list]
    
    S_small = S.tolist()[:rank]
    
    v_list = V.tolist()
    # V is Vh (N x N), we want first 'rank' rows
    V_small = v_list[:rank]

    
    return {
        "U": U_small,
        "S": S_small,
        "V": V_small,
        "original_shape": tensor.shape.dims,
        "rank": rank,
        "method": "svd"
    }

def decompress_eigen(compressed_data: Dict[str, Any]) -> Tensor:
    """
    Reconstruct a tensor from compressed SVD data.
    """
    u_data = compressed_data["U"]
    s_data = compressed_data["S"]
    v_data = compressed_data["V"] # V is Vh_small (rank x N)
    
    rows, cols = compressed_data["original_shape"]
    rank = compressed_data["rank"]
    
    res_data = [0.0] * (rows * cols)
    
    for i in range(rows):
        for j in range(cols):
            val = 0.0
            for k in range(rank):
                # A_ij = sum_k U_ik * S_k * Vh_kj
                val += u_data[i][k] * s_data[k] * v_data[k][j]
            res_data[i * cols + j] = val
            
    return array(res_data).reshape((rows, cols))

