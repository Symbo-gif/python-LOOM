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

from typing import Tuple, List, Any
from loom.core.shape import Shape
# BinaryOp is currently hosted in arithmetic.py in Phase 1
from loom.ops.arithmetic import BinaryOp


class MatmulOp(BinaryOp):
    """
    Matrix multiplication operation.
    
    Implements standard matrix multiplication logic with braodcasting support for stacks of matrices.
    Ref: https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
    """
    
    @property
    def name(self) -> str:
        return "matmul"
    
    def infer_shape(self, a_tensor: 'Tensor', b_tensor: 'Tensor') -> Shape:
        """
        Infer result shape of matrix multiplication A @ B.
        
        Rules:
        1. If both 2D: (m, k) @ (k, n) -> (m, n)
        2. If A is 1D: promoted to (1, k), then result is (n,)
        3. If B is 1D: promoted to (k, 1), then result is (m,)
        4. If >2D: Treated as stack of matrices, broadcast on leading dims.
        """
        shape_a = a_tensor.shape.dims
        shape_b = b_tensor.shape.dims
        ndim_a = len(shape_a)
        ndim_b = len(shape_b)
        
        if ndim_a == 0 or ndim_b == 0:
            raise ValueError(f"Scalar arguments not allowed in matmul. Use * instead. Shapes: {shape_a}, {shape_b}")

        # Case 1: Both 1D (dot product)
        if ndim_a == 1 and ndim_b == 1:
            if shape_a[0] != shape_b[0]:
                raise ValueError(f"Dimension mismatch for dot product: {shape_a} vs {shape_b}")
            return Shape(()) # Scalar result
            
        # Case 2: 2D @ 2D
        if ndim_a == 2 and ndim_b == 2:
            if shape_a[1] != shape_b[0]:
                raise ValueError(f"Matmul mismatch: {shape_a} @ {shape_b}")
            return Shape((shape_a[0], shape_b[1]))

        # General generic logic handling broadcasting for >2D
        # 1. Prepend 1 to 1D arrays
        a_is_1d = (ndim_a == 1)
        b_is_1d = (ndim_b == 1)
        
        dims_a = list(shape_a)
        dims_b = list(shape_b)
        
        if a_is_1d:
            dims_a = [1] + dims_a
            ndim_a += 1
        if b_is_1d:
            dims_b = dims_b + [1]
            ndim_b += 1
            
        # Check inner dimensions match
        if dims_a[-1] != dims_b[-2]:
             raise ValueError(f"Matmul inner dimension mismatch: {shape_a} @ {shape_b} -> {dims_a[-1]} != {dims_b[-2]}")
             
        # Result of matrix part
        mat_shape = [dims_a[-2], dims_b[-1]]
        
        # Broadcast batch dimensions (all but last 2)
        batch_a = dims_a[:-2]
        batch_b = dims_b[:-2]
        
        # Simple broadcasting logic for batch dims
        # Note: Inline implementation for batch broadcasting rather than utility function
        # Align from right
        len_a = len(batch_a)
        len_b = len(batch_b)
        length = max(len_a, len_b)
        
        batch_res = []
        for i in range(length):
            # Get dim from right; 0 if out of bounds (effectively 1 for broadcasting? No, missing dims are 1)
            # Actually standard broadcast logic:
            dim_a = batch_a[len_a - 1 - i] if i < len_a else 1
            dim_b = batch_b[len_b - 1 - i] if i < len_b else 1
            
            if dim_a != dim_b and dim_a != 1 and dim_b != 1:
                 raise ValueError(f"Broadcasting error in batch dimensions: {shape_a} vs {shape_b}")
            
            batch_res.insert(0, max(dim_a, dim_b))
            
        final_dims = batch_res + mat_shape
        
        # Handle squeezing 1D inputs
        if a_is_1d:
            # Remove second to last dim (the 1 we added)
             del final_dims[-2]
        if b_is_1d:
             # Remove last dim (the 1 we added)
             del final_dims[-1]
             
        return Shape(tuple(final_dims))

    def execute(self, a_tensor: 'Tensor', b_tensor: 'Tensor') -> List[Any]:
        # Get data
        a_data = self._get_data(a_tensor)
        b_data = self._get_data(b_tensor)
        
        a_shape = a_tensor.shape.dims
        b_shape = b_tensor.shape.dims
        
        # Naive implementation for Phase 2 proof of concept
        # We will unflatten, compute, and flatten back
        # Ideally we would operate on flat buffers for speed, but matmul is complex on flat buffers
        
        # Use the tensor's _unflatten helper (hack: access via first tensor)
        # Or better, just implement a simple local matmul for lists
        
        # Since we don't have unflatten exposed as a utility function well enough yet
        # Let's rely on basic list logic for small tests, or implement a naive loop here.
        # But wait, our tensors are FLAT lists.
        # M x K  @  K x N
        
        # Let's implement 2D case first, then general later if needed.
        # Starting with basic logic:
        
        ndim_a = len(a_shape)
        ndim_b = len(b_shape)

        result_dims = self.infer_shape(a_tensor, b_tensor).dims
        
        # Optim: If both 2D or 1D, fast path
        if ndim_a <= 2 and ndim_b <= 2:
             # Convert flat to structured
             # A is (M, K), B is (K, N)
             
             # Reshape helpers
             rows_a = a_shape[0] if ndim_a > 1 else 1 #Treating 1D as row vec for logic, adjust later
             cols_a = a_shape[-1]
             
             rows_b = b_shape[0] if ndim_b > 1 else b_shape[0] #Logic tricky for 1D B
             cols_b = b_shape[-1] if ndim_b > 1 else 1
             
             if ndim_a == 1:
                 rows_a, cols_a = 1, a_shape[0]
             if ndim_b == 1:
                 rows_b, cols_b = b_shape[0], 1
             
             K = cols_a
             if rows_b != K:
                 raise ValueError("Internal shape mismatch")
                 
             M = rows_a
             N = cols_b
             
             # Strides
             # A: stride_row = K, stride_col = 1
             # B: stride_row = N, stride_col = 1
             
             res_data = [0.0] * (M * N)
             
             for i in range(M):
                 for j in range(N):
                     val = 0.0
                     for k in range(K):
                         # A[i, k]
                         idx_a = i * cols_a + k
                         # B[k, j]
                         idx_b = k * cols_b + j
                         
                         val += a_data[idx_a] * b_data[idx_b]
                     
                     res_data[i * N + j] = val
                     
             # If result is scalar (1D . 1D), return list of 1
             if ndim_a == 1 and ndim_b == 1:
                 return res_data # It's size 1
             
             # If A was 1D, M=1. Result (N,) -> (1*N). Logic holds.
             # If B was 1D, N=1. Result (M,) -> (M*1). Logic holds.
             
             return res_data
             
        else:
            # N-D Matmul with broadcasting
            res_shape = self.infer_shape(a_tensor, b_tensor)
            res_size = res_shape.size
            res_data = [0.0] * res_size
            
            a_strides = self._compute_strides(a_shape)
            b_strides = self._compute_strides(b_shape)
            
            m = a_shape[-2] if ndim_a > 1 else 1
            k = a_shape[-1]
            n = b_shape[-1] if ndim_b > 1 else 1
            
            batch_dims = res_shape.dims[:-2] if (ndim_a > 1 and ndim_b > 1) else \
                         res_shape.dims[:-1] if (ndim_a > 1 or ndim_b > 1) else ()

            for idx in range(res_size):
                multi_idx = self._flat_to_multi(idx, res_shape.dims)
                
                if ndim_a > 1 and ndim_b > 1:
                    i, j = multi_idx[-2], multi_idx[-1]
                    batch_idx = multi_idx[:-2]
                elif ndim_a > 1:
                    i, j = multi_idx[-1], 0
                    batch_idx = multi_idx[:-1]
                else:
                    i, j = 0, multi_idx[-1]
                    batch_idx = multi_idx[:-1]
               
                val = 0.0
                for kk in range(k):
                    if ndim_a > 1:
                        a_multi = self._broadcast_index(batch_idx, a_shape[:-2], batch_dims) + (i, kk)
                    else:
                        a_multi = (kk,)
                    
                    if ndim_b > 1:
                        b_multi = self._broadcast_index(batch_idx, b_shape[:-2], batch_dims) + (kk, j)
                    else:
                        b_multi = (kk,)
                        
                    idx_a = self._multi_to_flat(a_multi, a_strides)
                    idx_b = self._multi_to_flat(b_multi, b_strides)
                    val += a_data[idx_a] * b_data[idx_b]
                
                res_data[idx] = val
                
            return res_data

def create_matmul_op(a, b):
    return MatmulOp()

