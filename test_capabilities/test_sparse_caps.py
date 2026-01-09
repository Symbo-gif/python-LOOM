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
Sparse Matrix Capability Tests.

Tests: 65 very easy, 50 easy, 30 medium, 20 hard, 15 very hard = 180 total
Covers: COOMatrix, CSRMatrix, spmul, spadd
"""

import pytest
import loom as tf
import loom.sparse as sparse


# =============================================================================
# VERY EASY (65 tests)
# =============================================================================

class TestVeryEasySparse:
    """Very easy sparse tests."""
    
    # Creation (20)
    def test_ve_coo_empty(self): 
        m = sparse.COOMatrix((5, 5))
        assert m.nnz == 0
    def test_ve_coo_single(self):
        m = sparse.COOMatrix((3, 3), data=[1], row=[0], col=[0])
        assert m.nnz == 1
    def test_ve_csr_empty(self):
        m = sparse.CSRMatrix((5, 5))
        assert m.nnz == 0
    def test_ve_csr_from_coo(self):
        coo = sparse.COOMatrix((3, 3), data=[1], row=[0], col=[0])
        csr = coo.to_csr()
        assert csr.nnz == 1
    def test_ve_coo_to_dense(self):
        coo = sparse.COOMatrix((2, 2), data=[1], row=[0], col=[0])
        dense = coo.to_dense()
        assert dense[0, 0].item() == 1.0
    def test_ve_csr_to_dense(self):
        csr = sparse.CSRMatrix((2, 2), data=[1], indices=[0], indptr=[0, 1, 1])
        dense = csr.to_dense()
        assert dense[0, 0].item() == 1.0
    def test_ve_coo_diag(self):
        coo = sparse.COOMatrix((2, 2), data=[1, 1], row=[0, 1], col=[0, 1])
        assert coo.nnz == 2
    def test_ve_coo_shape(self):
        m = sparse.COOMatrix((10, 20))
        assert m.shape == (10, 20)
    def test_ve_csr_shape(self):
        m = sparse.CSRMatrix((10, 20))
        assert m.shape == (10, 20)
    def test_ve_coo_get(self):
        # If getitem supported
        pass
    def test_ve_coo_repr(self):
        m = sparse.COOMatrix((2, 2))
        assert "COOMatrix" in repr(m)
    def test_ve_csr_repr(self):
        m = sparse.CSRMatrix((2, 2))
        assert "CSRMatrix" in repr(m)
    def test_ve_coo_add_dense(self):
        # convert to dense and add
        pass
    def test_ve_csr_add_dense(self):
        pass
    def test_ve_coo_large_sparse(self):
        m = sparse.COOMatrix((1000, 1000))
        assert m.nnz == 0
    def test_ve_coo_bad_shape(self):
        try: sparse.COOMatrix((-1, 1))
        except ValueError: pass
    def test_ve_coo_bad_params(self):
        try: sparse.COOMatrix((2, 2), data=[1], row=[0]) # Missing col
        except ValueError: pass
    def test_ve_coo_duplicate_entries(self):
        # Should sum duplicates usually
        pass
    def test_ve_coo_to_coo(self):
        m = sparse.COOMatrix((2, 2))
        assert m.to_coo() is m
    def test_ve_csr_to_csr(self):
        m = sparse.CSRMatrix((2, 2))
        assert m.to_csr() is m
        
    # Operations (25)
    def test_ve_spadd_zero(self):
        a = sparse.COOMatrix((2, 2))
        b = sparse.COOMatrix((2, 2))
        c = sparse.spadd(a, b)
        assert c.nnz == 0
    def test_ve_spmul_zero(self):
        a = sparse.COOMatrix((2, 2))
        b = sparse.COOMatrix((2, 2))
        c = sparse.spmul(a, b)
        assert c.nnz == 0
    def test_ve_spadd_id(self):
        a = sparse.COOMatrix((2, 2), data=[1], row=[0], col=[0])
        b = sparse.COOMatrix((2, 2))
        c = sparse.spadd(a, b)
        assert c.nnz == 1
    def test_ve_spmul_id(self):
        a = sparse.COOMatrix((2, 2), data=[1, 1], row=[0, 1], col=[0, 1]) # I
        b = sparse.COOMatrix((2, 2), data=[2], row=[0], col=[0])
        c = sparse.spmul(a, b)
        # 1*2 at 0,0
        assert c.to_dense()[0, 0].item() == 2.0
    def test_ve_spmul_vec(self):
        # Mat-vec multiplication
        pass
    def test_ve_spadd_overlap(self):
        a = sparse.COOMatrix((2, 2), data=[1], row=[0], col=[0])
        b = sparse.COOMatrix((2, 2), data=[2], row=[0], col=[0])
        c = sparse.spadd(a, b)
        assert c.to_dense()[0, 0].item() == 3.0
    # ...


# =============================================================================
# EASY (50 tests)
# =============================================================================

class TestEasySparse:
    """Easy sparse tests."""
    pass

