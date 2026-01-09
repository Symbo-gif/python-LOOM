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
Sparse matrix operations.
"""

from typing import Union, List, Any
from loom.core.tensor import Tensor, array
# Circular import prevention
# from loom.sparse.sparse_matrix import CSRMatrix, COOMatrix

def spmul(A: Any, B: Union[Tensor, List, Any]) -> Union[Tensor, Any]:
    """
    Multiply a sparse matrix A by a dense or sparse B.
    """
    from loom.sparse.sparse_matrix import CSRMatrix
    
    # Ensure A uses CSR
    if hasattr(A, 'to_csr'):
        A = A.to_csr()
        
    if isinstance(A, CSRMatrix):
        if hasattr(B, '_dense_data') or isinstance(B, list) or isinstance(B, Tensor):
            # Sparse @ Dense
            return _csr_dense_mul(A, B)
        elif hasattr(B, 'to_csr'):
            # Sparse @ Sparse
            return _csr_csr_mul(A, B.to_csr())
            
    raise NotImplementedError(f"Multiplication not implemented for {type(A)} and {type(B)}")

def spadd(A: Any, B: Any) -> Any:
    """
    Add two sparse matrices.
    """
    from loom.sparse.sparse_matrix import CSRMatrix
    if hasattr(A, 'to_csr'): A = A.to_csr()
    if hasattr(B, 'to_csr'): B = B.to_csr()
    
    if isinstance(A, CSRMatrix) and isinstance(B, CSRMatrix):
        return _csr_csr_add(A, B)
    raise NotImplementedError()

def _csr_dense_mul(A: Any, B: Union[Tensor, List]) -> Tensor:
    if isinstance(B, list):
        B = array(B)
    
    rows_a, cols_a = A.shape
    shape_b = B.shape.dims
    
    if len(shape_b) == 1:
        # Vector multiplication
        if cols_a != shape_b[0]:
            raise ValueError(f"Dimension mismatch: {A.shape} @ {B.shape}")
        
        b_data = B.tolist()
        res_data = [0.0] * rows_a
        
        for i in range(rows_a):
            row_sum = 0.0
            for j in range(A.indptr[i], A.indptr[i+1]):
                col_idx = A.indices[j]
                row_sum += A.data[j] * b_data[col_idx]
            res_data[i] = row_sum
            
        return array(res_data)
    
    elif len(shape_b) == 2:
        # Matrix multiplication
        cols_b = shape_b[1]
        if cols_a != shape_b[0]:
            raise ValueError(f"Dimension mismatch: {A.shape} @ {B.shape}")
            
        b_data_flat = B.tolist() # Flat if ndim=1 or list of lists if ndim=2
        # Ensure we have flat row-major access
        if isinstance(b_data_flat[0], list):
            # Already nested, flatten it for logic
            b_flat = [item for sublist in b_data_flat for item in sublist]
        else:
            b_flat = b_data_flat
            
        res_data = [0.0] * (rows_a * cols_b)
        
        for i in range(rows_a):
            for j in range(A.indptr[i], A.indptr[i+1]):
                col_idx = A.indices[j]
                val_a = A.data[j]
                # Multiply val_a with the entire row col_idx of B
                for k in range(cols_b):
                    res_data[i * cols_b + k] += val_a * b_flat[col_idx * cols_b + k]
                    
        return array(res_data).reshape((rows_a, cols_b))
        
    else:
        raise NotImplementedError("N-D dense multiplication with sparse not implemented")

def _csr_csr_add(A: Any, B: Any) -> Any:
    from loom.sparse.sparse_matrix import CSRMatrix
    if A.shape != B.shape:
        raise ValueError("Shapes must match for addition")
        
    rows, cols = A.shape
    new_data = []
    new_indices = []
    indptr = [0] * (rows + 1)
    
    for i in range(rows):
        # Merge two sorted lists of indices
        a_idx = A.indptr[i]
        a_end = A.indptr[i+1]
        b_idx = B.indptr[i]
        b_end = B.indptr[i+1]
        
        while a_idx < a_end or b_idx < b_end:
            if a_idx < a_end and (b_idx >= b_end or A.indices[a_idx] < B.indices[b_idx]):
                new_data.append(A.data[a_idx])
                new_indices.append(A.indices[a_idx])
                a_idx += 1
            elif b_idx < b_end and (a_idx >= a_end or B.indices[b_idx] < A.indices[a_idx]):
                new_data.append(B.data[b_idx])
                new_indices.append(B.indices[b_idx])
                b_idx += 1
            else:
                # Same index
                val = A.data[a_idx] + B.data[b_idx]
                if val != 0:
                    new_data.append(val)
                    new_indices.append(A.indices[a_idx])
                a_idx += 1
                b_idx += 1
        indptr[i+1] = len(new_data)
        
    return CSRMatrix(A.shape, new_data, new_indices, indptr)

def _csr_csr_mul(A: Any, B: Any) -> Any:
    """CSR @ CSR multiply."""
    from loom.sparse.sparse_matrix import CSRMatrix
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible shapes for matmul")
        
    rows_a, _ = A.shape
    _, cols_b = B.shape
    
    new_data = []
    new_indices = []
    indptr = [0] * (rows_a + 1)
    
    # We use a dense accumulator for each row to keep it O(nnz)
    for i in range(rows_a):
        row_map = {} # Use dict as sparse accumulator
        for j in range(A.indptr[i], A.indptr[i+1]):
            col_a = A.indices[j]
            val_a = A.data[j]
            # row col_a of B
            for k in range(B.indptr[col_a], B.indptr[col_a+1]):
                col_b = B.indices[k]
                val_b = B.data[k]
                row_map[col_b] = row_map.get(col_b, 0.0) + val_a * val_b
        
        # Sort indices and append to result
        for c in sorted(row_map.keys()):
            val = row_map[c]
            if val != 0:
                new_data.append(val)
                new_indices.append(c)
        indptr[i+1] = len(new_data)
        
    return CSRMatrix((rows_a, cols_b), new_data, new_indices, indptr)

