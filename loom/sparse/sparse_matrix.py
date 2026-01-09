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
Sparse matrix representations.
"""

from typing import List, Tuple, Union, Any
from loom.core.tensor import Tensor, array
import math

class SparseMatrix:
    """Base class for sparse matrices."""
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        self.ndim = len(shape)

    def to_dense(self) -> Tensor:
        raise NotImplementedError()

class COOMatrix(SparseMatrix):
    """
    Coordinate format (COO) sparse matrix.
    Stores entries as (row, col, value) triples.
    """
    def __init__(self, *args, **kwargs):
        # Handle (shape, data, row, col), (data, row, col, shape), or keyword args
        shape = None
        data = None
        row = None
        col = None
        
        if len(args) > 0:
            if isinstance(args[0], tuple):
                # (shape, ...)
                shape = args[0]
                if len(args) > 1: data = args[1]
                if len(args) > 2: row = args[2]
                if len(args) > 3: col = args[3]
            elif isinstance(args[0], list) or hasattr(args[0], '__iter__'):
                # (data, row, col, shape)
                data = args[0]
                if len(args) > 1: row = args[1]
                if len(args) > 2: col = args[2]
                if len(args) > 3: shape = args[3]
                
        # Fill from kwargs if not set
        if shape is None: shape = kwargs.get('shape')
        if data is None: data = kwargs.get('data')
        if row is None: row = kwargs.get('row')
        if col is None: col = kwargs.get('col')
        
        if shape is None:
             raise ValueError("Shape must be provided")

        super().__init__(shape)
        self.data = list(data) if data is not None else []
        self.row = list(row) if row is not None else []
        self.col = list(col) if col is not None else []
        
        if len(self.data) != len(self.row) or len(self.data) != len(self.col):
            raise ValueError("data, row, and col must have same length")

    @property
    def nnz(self) -> int:
        return len(self.data)

    def to_dense(self) -> Tensor:
        rows, cols = self.shape
        dense_data = [0.0] * (rows * cols)
        for r, c, v in zip(self.row, self.col, self.data):
            dense_data[r * cols + c] = v
        return array(dense_data).reshape(self.shape)

    def to_csr(self) -> 'CSRMatrix':
        # Sort by row, then col
        triples = sorted(zip(self.row, self.col, self.data))
        if not triples:
            return CSRMatrix([], [], [0] * (self.shape[0] + 1), self.shape)

        new_data = []
        new_indices = []
        indptr = [0] * (self.shape[0] + 1)
        
        curr_row = 0
        for r, c, v in triples:
            while curr_row < r:
                curr_row += 1
                indptr[curr_row] = len(new_data)
            new_data.append(v)
            new_indices.append(c)
        
        while curr_row < self.shape[0]:
            curr_row += 1
            indptr[curr_row] = len(new_data)
            
        return CSRMatrix(self.shape, new_data, new_indices, indptr)

    def to_coo(self) -> 'COOMatrix':
        return self

class CSRMatrix(SparseMatrix):
    """
    Compressed Sparse Row (CSR) format.
    Efficient for row-wise operations and matrix products.
    """
    def __init__(self, *args, **kwargs):
         # Handle (shape, data, indices, indptr) or (data, indices, indptr, shape)
        shape = None
        data = None
        indices = None
        indptr = None
        
        if len(args) > 0:
            if isinstance(args[0], tuple):
                shape = args[0]
                if len(args) > 1: data = args[1]
                if len(args) > 2: indices = args[2]
                if len(args) > 3: indptr = args[3]
            elif isinstance(args[0], list) or hasattr(args[0], '__iter__'):
                data = args[0]
                if len(args) > 1: indices = args[1]
                if len(args) > 2: indptr = args[2]
                if len(args) > 3: shape = args[3]

        if shape is None: shape = kwargs.get('shape')
        if data is None: data = kwargs.get('data')
        if indices is None: indices = kwargs.get('indices')
        if indptr is None: indptr = kwargs.get('indptr')

        if shape is None:
             raise ValueError("Shape must be provided")

        super().__init__(shape)
        self.data = list(data) if data is not None else []
        self.indices = list(indices) if indices is not None else []
        if indptr is None:
             # Default indptr for empty matrix of given shape: [0, 0, ..., 0] (rows + 1 zeros)
             self.indptr = [0] * (shape[0] + 1)
        else:
             self.indptr = list(indptr)

    @property
    def nnz(self) -> int:
        return len(self.data)

    def to_dense(self) -> Tensor:
        rows, cols = self.shape
        dense_data = [0.0] * (rows * cols)
        for i in range(rows):
            for j in range(self.indptr[i], self.indptr[i+1]):
                col = self.indices[j]
                val = self.data[j]
                dense_data[i * cols + col] = val
        return array(dense_data).reshape(self.shape)

    def __matmul__(self, other: Union[Tensor, List, 'CSRMatrix']) -> Union[Tensor, 'CSRMatrix']:
        from loom.sparse.ops import spmul
        return spmul(self, other)

    def to_csr(self) -> 'CSRMatrix':
        return self

    def to_coo(self) -> 'COOMatrix':
        rows, cols = self.shape
        data = []
        row = []
        col = []
        for i in range(rows):
            for j in range(self.indptr[i], self.indptr[i+1]):
                data.append(self.data[j])
                row.append(i)
                col.append(self.indices[j])
        return COOMatrix(self.shape, data, row, col)

