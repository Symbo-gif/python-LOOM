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

import pytest
import loom as tf
from loom.sparse import COOMatrix, CSRMatrix, spmul, spadd

def test_coo_creation_and_dense():
    data = [1.0, 2.0, 3.0]
    row = [0, 1, 2]
    col = [0, 1, 2]
    shape = (3, 3)
    
    coo = COOMatrix(shape, data, row, col)
    dense = coo.to_dense()
    
    expected = [[1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0]]
    assert dense.tolist() == expected

def test_coo_to_csr():
    data = [1.0, 2.0]
    row = [1, 0]
    col = [1, 0]
    shape = (2, 2)
    
    coo = COOMatrix(shape, data, row, col)
    csr = coo.to_csr()
    
    assert csr.indices == [0, 1]
    assert csr.indptr == [0, 1, 2]
    assert csr.data == [2.0, 1.0]

def test_csr_dense_mul():
    # Identiy 2x2
    csr = CSRMatrix((2, 2), [1.0, 1.0], [0, 1], [0, 1, 2])
    v = tf.array([10, 20])
    
    res = spmul(csr, v)
    assert res.tolist() == [10.0, 20.0]
    
    # 2x2 @ 2x2
    m = tf.array([[1, 2], [3, 4]])
    res_m = spmul(csr, m)
    assert res_m.tolist() == [[1.0, 2.0], [3.0, 4.0]]

def test_csr_csr_add():
    A = CSRMatrix((2, 2), [1.0], [0], [0, 1, 1]) # [[1, 0], [0, 0]]
    B = CSRMatrix((2, 2), [2.0], [1], [0, 0, 1]) # [[0, 0], [0, 2]]
    
    C = spadd(A, B)
    assert C.to_dense().tolist() == [[1.0, 0.0], [0.0, 2.0]]

def test_csr_csr_mul():
    # [[1, 2], [3, 4]] in CSR
    A = CSRMatrix((2, 2), [1, 2, 3, 4], [0, 1, 0, 1], [0, 2, 4])
    # Identity
    B = CSRMatrix((2, 2), [1, 1], [0, 1], [0, 1, 2])
    
    C = spmul(A, B)
    assert C.to_dense().tolist() == [[1.0, 2.0], [3.0, 4.0]]

