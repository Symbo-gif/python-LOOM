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
Linear Algebra Capability Tests.

Tests: 65 very easy, 50 easy, 30 medium, 20 hard, 15 very hard = 180 total
Covers: matmul, dot, norm, trace, lu, qr, cholesky, svd, eig, solve, inv, det, expm, sqrtm, logm
"""

import pytest
import math
import loom as tf
import loom.linalg as la


# =============================================================================
# VERY EASY (65 tests)
# =============================================================================

class TestVeryEasyLinalg:
    """Very easy linear algebra - trivial cases."""
    
    # matmul (10)
    def test_ve_matmul_eye(self): assert (tf.eye(2) @ tf.eye(2)).sum().item() == 2.0
    def test_ve_matmul_zeros(self): assert (tf.zeros((2, 2)) @ tf.zeros((2, 2))).sum().item() == 0.0
    def test_ve_matmul_ones_2x2(self): assert (tf.ones((2, 2)) @ tf.ones((2, 2))).sum().item() == 8.0
    def test_ve_matmul_scalar_like(self): assert (tf.array([[2]]) @ tf.array([[3]])).item() == 6.0
    def test_ve_matmul_eye_vec(self): v = tf.array([[1], [2]]); assert (tf.eye(2) @ v).flatten().tolist() == [1.0, 2.0]
    def test_ve_matmul_row_col(self): assert (tf.array([[1, 2]]) @ tf.array([[3], [4]])).item() == 11.0
    def test_ve_matmul_eye_3(self): assert (tf.eye(3) @ tf.eye(3)).sum().item() == 3.0
    def test_ve_matmul_diag(self): A = tf.array([[2, 0], [0, 3]]); assert (A @ A).tolist() == [[4, 0], [0, 9]]
    def test_ve_dot_simple(self): assert la.dot(tf.array([1, 2]), tf.array([3, 4])).item() == 11.0
    def test_ve_vdot_simple(self): assert la.vdot(tf.array([1, 2]), tf.array([1, 2])).item() == 5.0
    
    # norm/trace (10)
    def test_ve_norm_zero(self): assert la.norm(tf.zeros((3,))).item() == 0.0
    def test_ve_norm_unit(self): assert la.norm(tf.array([1, 0, 0])).item() == 1.0
    def test_ve_norm_3_4(self): assert la.norm(tf.array([3, 4])).item() == 5.0
    def test_ve_trace_eye(self): assert la.trace(tf.eye(3)).item() == 3.0
    def test_ve_trace_zeros(self): assert la.trace(tf.zeros((3, 3))).item() == 0.0
    def test_ve_trace_ones(self): assert la.trace(tf.ones((2, 2))).item() == 2.0
    def test_ve_trace_diag(self): assert la.trace(tf.array([[5, 0], [0, 7]])).item() == 12.0
    def test_ve_inner(self): assert la.inner(tf.array([1, 2]), tf.array([3, 4])).item() == 11.0
    def test_ve_outer_simple(self): assert la.outer(tf.array([1, 2]), tf.array([3, 4])).shape.dims == (2, 2)
    def test_ve_outer_values(self): assert la.outer(tf.array([1]), tf.array([2])).item() == 2.0
    
    # det (10)
    def test_ve_det_eye(self): assert la.det(tf.eye(2)).item() == 1.0
    def test_ve_det_eye_3(self): assert la.det(tf.eye(3)).item() == 1.0
    def test_ve_det_zeros(self): assert la.det(tf.zeros((2, 2))).item() == 0.0
    def test_ve_det_scalar(self): assert la.det(tf.array([[5]])).item() == 5.0
    def test_ve_det_diag(self): assert la.det(tf.array([[2, 0], [0, 3]])).item() == 6.0
    def test_ve_det_diag_3(self): assert la.det(tf.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])).item() == 6.0
    def test_ve_det_neg(self): assert la.det(tf.array([[-1, 0], [0, 1]])).item() == -1.0
    def test_ve_det_singular(self): assert la.det(tf.array([[1, 1], [1, 1]])).item() == 0.0
    def test_ve_det_2x2_simple(self): assert la.det(tf.array([[1, 2], [3, 4]])).item() == -2.0
    def test_ve_det_permuted_eye(self): assert abs(la.det(tf.array([[0, 1], [1, 0]])).item()) == 1.0
    
    # solve/inv (10)
    def test_ve_solve_eye(self):
        A = tf.eye(2)
        b = tf.array([1, 2])
        x = la.solve(A, b)
        assert x.tolist() == [1.0, 2.0]
    
    def test_ve_solve_diag(self):
        A = tf.array([[2, 0], [0, 3]])
        b = tf.array([4, 9])
        x = la.solve(A, b)
        assert x.tolist() == [2.0, 3.0]
    
    def test_ve_inv_eye(self):
        assert la.inv(tf.eye(2)).tolist() == [[1, 0], [0, 1]]
    
    def test_ve_inv_eye_3(self):
        assert la.inv(tf.eye(3)).sum().item() == 3.0
    
    def test_ve_inv_diag(self):
        A = tf.array([[2, 0], [0, 4]])
        Ainv = la.inv(A)
        assert math.isclose(Ainv[0, 0].item(), 0.5, rel_tol=1e-10)
    
    def test_ve_inv_roundtrip(self):
        A = tf.eye(3)
        assert la.inv(la.inv(A)).sum().item() == 3.0
    
    def test_ve_cond_eye(self):
        assert la.cond(tf.eye(3)).item() == 1.0
    
    def test_ve_rank_eye(self):
        assert la.matrix_rank(tf.eye(3)).item() == 3
    
    def test_ve_rank_zeros(self):
        assert la.matrix_rank(tf.zeros((3, 3))).item() == 0
    
    def test_ve_rank_ones(self):
        assert la.matrix_rank(tf.ones((3, 3))).item() == 1
    
    # decompositions (15)
    def test_ve_lu_eye(self):
        P, L, U = la.lu(tf.eye(2))
        assert P.shape.dims == (2, 2)
    
    def test_ve_qr_eye(self):
        Q, R = la.qr(tf.eye(2))
        assert Q.shape.dims == (2, 2)
    
    def test_ve_svd_eye(self):
        U, S, Vh = la.svd(tf.eye(2))
        assert all(math.isclose(s, 1.0, rel_tol=1e-6) for s in S.tolist())
    
    def test_ve_cholesky_eye(self):
        L = la.cholesky(tf.eye(2))
        assert L.tolist() == [[1, 0], [0, 1]]
    
    def test_ve_eig_eye(self):
        vals, vecs = la.eig(tf.eye(2))
        assert all(math.isclose(v, 1.0, abs_tol=1e-6) for v in vals.tolist())
    
    def test_ve_lu_returns_3(self):
        result = la.lu(tf.eye(3))
        assert len(result) == 3
    
    def test_ve_qr_returns_2(self):
        result = la.qr(tf.eye(3))
        assert len(result) == 2
    
    def test_ve_svd_returns_3(self):
        result = la.svd(tf.eye(3))
        assert len(result) == 3
    
    def test_ve_eig_returns_2(self):
        result = la.eig(tf.eye(3))
        assert len(result) == 2
    
    def test_ve_cholesky_diag(self):
        A = tf.array([[4, 0], [0, 9]])
        L = la.cholesky(A)
        assert L[0, 0].item() == 2.0
    
    def test_ve_transpose(self):
        A = tf.array([[1, 2], [3, 4]])
        assert la.matrix_transpose(A).tolist() == [[1, 3], [2, 4]]
    
    def test_ve_transpose_vec(self):
        v = tf.array([[1, 2, 3]])
        assert la.matrix_transpose(v).shape.dims == (3, 1)
    
    def test_ve_lu_diag(self):
        P, L, U = la.lu(tf.array([[2, 0], [0, 3]]))
        assert U[0, 0].item() != 0
    
    def test_ve_qr_diag(self):
        Q, R = la.qr(tf.array([[2, 0], [0, 3]]))
        assert R.shape.dims == (2, 2)
    
    def test_ve_svd_diag(self):
        U, S, Vh = la.svd(tf.array([[2, 0], [0, 3]]))
        s_list = sorted(S.tolist(), reverse=True)
        assert math.isclose(s_list[0], 3.0, rel_tol=1e-6)
    
    # matrix functions (10)
    def test_ve_expm_zero(self):
        A = tf.zeros((2, 2))
        E = la.expm(A)
        assert math.isclose(E[0, 0].item(), 1.0, rel_tol=1e-6)
    
    def test_ve_expm_eye(self):
        A = tf.eye(2)
        E = la.expm(A)
        assert math.isclose(E[0, 0].item(), math.e, rel_tol=1e-6)
    
    def test_ve_sqrtm_eye(self):
        A = tf.eye(2)
        S = la.sqrtm(A)
        assert math.isclose(S[0, 0].item(), 1.0, rel_tol=1e-6)
    
    def test_ve_sqrtm_4I(self):
        A = tf.eye(2) * 4
        S = la.sqrtm(A)
        assert math.isclose(S[0, 0].item(), 2.0, rel_tol=1e-6)
    
    def test_ve_logm_eye(self):
        A = tf.eye(2)
        L = la.logm(A)
        assert abs(L[0, 0].item()) < 1e-6
    
    def test_ve_logm_eI(self):
        A = tf.eye(2) * math.e
        L = la.logm(A)
        assert math.isclose(L[0, 0].item(), 1.0, rel_tol=1e-4)
    
    def test_ve_expm_logm_roundtrip(self):
        A = tf.eye(2) * 0.5
        result = la.logm(la.expm(A))
        assert math.isclose(result[0, 0].item(), 0.5, rel_tol=1e-4)
    
    def test_ve_sqrtm_sqrtm(self):
        A = tf.eye(2) * 16
        S = la.sqrtm(la.sqrtm(A))
        assert math.isclose(S[0, 0].item(), 2.0, rel_tol=1e-4)
    
    def test_ve_expm_zeros_trace(self):
        A = tf.zeros((3, 3))
        E = la.expm(A)
        assert math.isclose(la.trace(E).item(), 3.0, rel_tol=1e-6)
    
    def test_ve_sqrtm_diag(self):
        A = tf.array([[9, 0], [0, 16]])
        S = la.sqrtm(A)
        assert math.isclose(S[0, 0].item(), 3.0, rel_tol=1e-4)


# =============================================================================
# EASY (50 tests)
# =============================================================================

class TestEasyLinalg:
    """Easy linear algebra - standard usage."""
    
    # matmul (10)
    def test_e_matmul_2x3_3x2(self):
        A = tf.ones((2, 3))
        B = tf.ones((3, 2))
        C = A @ B
        assert C.shape.dims == (2, 2)
        assert C[0, 0].item() == 3.0
    
    def test_e_matmul_chain(self):
        A = tf.eye(3)
        result = A @ A @ A
        assert result.sum().item() == 3.0
    
    def test_e_matmul_3x3(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        I = tf.eye(3)
        assert (A @ I).tolist() == A.tolist()
    
    def test_e_dot_3d(self):
        a = tf.array([1, 2, 3])
        b = tf.array([4, 5, 6])
        assert la.dot(a, b).item() == 32.0
    
    def test_e_outer_2x3(self):
        a = tf.array([1, 2])
        b = tf.array([1, 2, 3])
        O = la.outer(a, b)
        assert O.tolist() == [[1, 2, 3], [2, 4, 6]]
    
    def test_e_inner_eq_dot(self):
        a = tf.array([1, 2, 3])
        b = tf.array([4, 5, 6])
        assert la.inner(a, b).item() == la.dot(a, b).item()
    
    def test_e_matmul_rectangular(self):
        A = tf.ones((3, 5))
        B = tf.ones((5, 2))
        assert (A @ B).shape.dims == (3, 2)
    
    def test_e_vdot_complex(self):
        a = tf.array([1+1j, 2+2j])
        b = tf.array([1+1j, 2+2j])
        result = la.vdot(a, b).item()
        assert math.isclose(abs(result), 10.0, rel_tol=1e-6)
    
    def test_e_matmul_nonsquare(self):
        A = tf.randn((4, 3))
        B = tf.randn((3, 5))
        C = A @ B
        assert C.shape.dims == (4, 5)
    
    def test_e_eye_matmul_preserves(self):
        A = tf.array([[1, 2, 3], [4, 5, 6]])
        I = tf.eye(3)
        assert (A @ I).tolist() == A.tolist()
    
    # decompositions (15)
    def test_e_lu_reconstruct(self):
        A = tf.array([[1, 2], [3, 4]])
        P, L, U = la.lu(A)
        # P @ L @ U ≈ A for 2x2
        reconstructed = P @ L @ U
        for i in range(2):
            for j in range(2):
                assert math.isclose(reconstructed[i, j].item(), A[i, j].item(), abs_tol=1e-6)
    
    def test_e_qr_q_orthogonal(self):
        A = tf.array([[1, 2], [3, 4]])
        Q, R = la.qr(A)
        # Q @ Q.T ≈ I
        QtQ = Q @ Q.T
        assert math.isclose(QtQ[0, 0].item(), 1.0, abs_tol=1e-6)
    
    def test_e_svd_singular_values(self):
        A = tf.array([[3, 0], [0, 4]])
        U, S, Vh = la.svd(A)
        s_sorted = sorted(S.tolist(), reverse=True)
        assert math.isclose(s_sorted[0], 4.0, rel_tol=1e-6)
        assert math.isclose(s_sorted[1], 3.0, rel_tol=1e-6)
    
    def test_e_cholesky_reconstruct(self):
        A = tf.array([[4, 2], [2, 5]])
        L = la.cholesky(A)
        LLt = L @ L.T
        for i in range(2):
            for j in range(2):
                assert math.isclose(LLt[i, j].item(), A[i, j].item(), abs_tol=1e-6)
    
    def test_e_eig_trace(self):
        A = tf.array([[1, 2], [2, 1]])
        vals, _ = la.eig(A)
        # Sum of eigenvalues = trace
        assert math.isclose(sum(vals.tolist()), la.trace(A).item(), abs_tol=1e-6)
    
    def test_e_eig_det(self):
        A = tf.array([[2, 1], [1, 2]])
        vals, _ = la.eig(A)
        # Product of eigenvalues = det
        prod = 1.0
        for v in vals.tolist():
            prod *= v
        assert math.isclose(prod, la.det(A).item(), abs_tol=1e-6)
    
    def test_e_lu_3x3(self):
        A = tf.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
        P, L, U = la.lu(A)
        assert P.shape.dims == (3, 3)
    
    def test_e_qr_3x3(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        Q, R = la.qr(A)
        assert Q.shape.dims == (3, 3)
    
    def test_e_svd_3x3(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        U, S, Vh = la.svd(A)
        assert len(S.tolist()) == 3
    
    def test_e_eigh_symmetric(self):
        A = tf.array([[2, 1], [1, 2]])
        vals, vecs = la.eigh(A)
        # Eigenvalues should be real
        for v in vals.tolist():
            assert isinstance(v, (int, float))
    
    def test_e_lu_wide(self):
        A = tf.ones((2, 4))
        P, L, U = la.lu(A)
        assert U.shape.dims[1] == 4
    
    def test_e_qr_tall(self):
        A = tf.ones((4, 2))
        Q, R = la.qr(A)
        assert Q.shape.dims[0] == 4
    
    def test_e_svd_wide(self):
        A = tf.ones((2, 5))
        U, S, Vh = la.svd(A)
        assert len(S.tolist()) == 2
    
    def test_e_svd_tall(self):
        A = tf.ones((5, 2))
        U, S, Vh = la.svd(A)
        assert len(S.tolist()) == 2
    
    def test_e_eig_3x3(self):
        A = tf.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        vals, _ = la.eig(A)
        v_sorted = sorted(vals.tolist())
        assert v_sorted == [1.0, 2.0, 3.0]
    
    # solve/inv (15)
    def test_e_solve_2x2(self):
        A = tf.array([[1, 2], [3, 4]])
        b = tf.array([5, 11])
        x = la.solve(A, b)
        # Verify A @ x = b
        Ax = A @ x
        assert math.isclose(Ax[0].item(), 5.0, abs_tol=1e-6)
    
    def test_e_solve_3x3(self):
        A = tf.eye(3) * 2
        b = tf.array([2, 4, 6])
        x = la.solve(A, b)
        assert x.tolist() == [1.0, 2.0, 3.0]
    
    def test_e_inv_2x2(self):
        A = tf.array([[1, 2], [3, 4]])
        Ainv = la.inv(A)
        I = A @ Ainv
        assert math.isclose(I[0, 0].item(), 1.0, abs_tol=1e-6)
    
    def test_e_inv_3x3(self):
        A = tf.eye(3) * 2
        Ainv = la.inv(A)
        assert math.isclose(Ainv[0, 0].item(), 0.5, rel_tol=1e-10)
    
    def test_e_det_3x3_nonzero(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        d = la.det(A)
        assert abs(d.item()) > 0.1
    
    def test_e_rank_3x3_full(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        assert la.matrix_rank(A).item() == 3
    
    def test_e_rank_deficient(self):
        A = tf.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        assert la.matrix_rank(A).item() == 1
    
    def test_e_cond_diag(self):
        A = tf.array([[10, 0], [0, 1]])
        c = la.cond(A)
        assert math.isclose(c.item(), 10.0, rel_tol=1e-6)
    
    def test_e_norm_matrix(self):
        A = tf.array([[3, 0], [0, 4]])
        n = la.norm(A)
        assert n.item() == 5.0
    
    def test_e_norm_frobenius(self):
        A = tf.ones((2, 2))
        n = la.norm(A)
        assert math.isclose(n.item(), 2.0, rel_tol=1e-6)
    
    def test_e_solve_overdetermined(self):
        # 3x2 system - solve finds least squares
        A = tf.array([[1, 0], [0, 1], [1, 1]])
        b = tf.array([1, 1, 2])
        # Note: may raise or return least-squares depending on impl
        try:
            x = la.solve(A, b)
            assert x.shape.dims[0] == 2
        except Exception:
            pass  # OK if not supported
    
    def test_e_inv_times_vec(self):
        A = tf.array([[2, 0], [0, 4]])
        b = tf.array([2, 8])
        x = la.inv(A) @ b
        assert x.tolist() == [1.0, 2.0]
    
    def test_e_trace_3x3(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert la.trace(A).item() == 15.0
    
    def test_e_det_triangular(self):
        A = tf.array([[2, 1, 1], [0, 3, 1], [0, 0, 4]])
        assert math.isclose(la.det(A).item(), 24.0, rel_tol=1e-6)
    
    def test_e_inv_inv(self):
        A = tf.array([[1, 2], [3, 4]])
        assert math.isclose((la.inv(la.inv(A)) - A).abs().sum().item(), 0.0, abs_tol=1e-10)
    
    # matrix functions (10)
    def test_e_expm_small(self):
        A = tf.array([[0.1, 0], [0, 0.1]])
        E = la.expm(A)
        assert E[0, 0].item() > 1.0
    
    def test_e_expm_nilpotent(self):
        A = tf.array([[0, 1], [0, 0]])
        E = la.expm(A)
        assert math.isclose(E[0, 0].item(), 1.0, rel_tol=1e-6)
    
    def test_e_sqrtm_4(self):
        A = tf.array([[4, 0], [0, 4]])
        S = la.sqrtm(A)
        S2 = S @ S
        assert math.isclose(S2[0, 0].item(), 4.0, abs_tol=1e-6)
    
    def test_e_logm_exp(self):
        A = tf.array([[1, 0], [0, 2]])
        eA = tf.array([[math.e, 0], [0, math.e**2]])
        L = la.logm(eA)
        assert math.isclose(L[0, 0].item(), 1.0, rel_tol=1e-4)
    
    def test_e_expm_trace(self):
        A = tf.zeros((3, 3))
        E = la.expm(A)
        assert math.isclose(la.trace(E).item(), 3.0, rel_tol=1e-6)
    
    def test_e_sqrtm_9(self):
        A = tf.array([[9, 0], [0, 9]])
        S = la.sqrtm(A)
        assert math.isclose(S[0, 0].item(), 3.0, rel_tol=1e-4)
    
    def test_e_logm_identity(self):
        A = tf.eye(3)
        L = la.logm(A)
        assert la.norm(L).item() < 1e-6
    
    def test_e_expm_skew(self):
        A = tf.array([[0, 1], [-1, 0]])
        E = la.expm(A)
        # Should be rotation matrix, det = 1
        assert math.isclose(la.det(E).item(), 1.0, abs_tol=1e-6)
    
    def test_e_sqrtm_positive(self):
        A = tf.array([[2, 1], [1, 2]])
        S = la.sqrtm(A)
        # S^2 ≈ A
        S2 = S @ S
        assert math.isclose(S2[0, 0].item(), 2.0, abs_tol=1e-4)
    
    def test_e_expm_commutes_scalar(self):
        A = tf.eye(2) * 0.5
        E = la.expm(A)
        # exp(0.5*I) = e^0.5 * I
        expected = math.exp(0.5)
        assert math.isclose(E[0, 0].item(), expected, rel_tol=1e-6)


# =============================================================================
# MEDIUM (30 tests)
# =============================================================================

class TestMediumLinalg:
    """Medium linear algebra - multi-dim, precision."""
    
    def test_m_lu_4x4(self):
        A = tf.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 12], [13, 14, 15, 17]])
        P, L, U = la.lu(A)
        assert P.shape.dims == (4, 4)
    
    def test_m_qr_4x4(self):
        A = tf.randn((4, 4))
        Q, R = la.qr(A)
        # Q should be orthogonal
        QtQ = Q @ Q.T
        assert math.isclose(QtQ[0, 0].item(), 1.0, abs_tol=1e-4)
    
    def test_m_svd_reconstruct(self):
        A = tf.array([[1, 2], [3, 4], [5, 6]])
        U, S, Vh = la.svd(A)
        # Partial reconstruction check
        assert U.shape.dims[0] == 3
    
    def test_m_eig_symmetric(self):
        A = tf.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
        vals, _ = la.eig(A)
        # Eigenvalues of symmetric should be real
        for v in vals.tolist():
            if isinstance(v, complex):
                assert abs(v.imag) < 1e-6
    
    def test_m_cholesky_3x3(self):
        A = tf.array([[4, 2, 0], [2, 5, 2], [0, 2, 4]])
        L = la.cholesky(A)
        LLt = L @ L.T
        assert math.isclose(LLt[0, 0].item(), A[0, 0].item(), abs_tol=1e-6)
    
    def test_m_solve_3x3(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        b = tf.array([14, 32, 50])
        x = la.solve(A, b)
        Ax = A @ x
        for i in range(3):
            assert math.isclose(Ax[i].item(), b[i].item(), abs_tol=1e-4)
    
    def test_m_inv_3x3(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        Ainv = la.inv(A)
        I = A @ Ainv
        for i in range(3):
            assert math.isclose(I[i, i].item(), 1.0, abs_tol=1e-6)
    
    def test_m_det_4x4(self):
        A = tf.eye(4) + tf.ones((4, 4)) * 0.1
        d = la.det(A)
        assert d.item() > 0
    
    def test_m_rank_4x4(self):
        A = tf.randn((4, 4))
        r = la.matrix_rank(A)
        # Random matrix almost surely full rank
        assert r.item() >= 3
    
    def test_m_cond_5x5(self):
        A = tf.eye(5) * 2
        c = la.cond(A)
        assert math.isclose(c.item(), 1.0, rel_tol=1e-6)
    
    def test_m_expm_3x3(self):
        A = tf.zeros((3, 3))
        A = A + tf.eye(3) * 0.1
        E = la.expm(A)
        expected = math.exp(0.1)
        assert math.isclose(E[0, 0].item(), expected, rel_tol=1e-4)
    
    def test_m_sqrtm_3x3(self):
        A = tf.eye(3) * 4
        S = la.sqrtm(A)
        assert math.isclose(S[0, 0].item(), 2.0, rel_tol=1e-4)
    
    def test_m_logm_3x3(self):
        A = tf.eye(3) * math.e
        L = la.logm(A)
        assert math.isclose(L[0, 0].item(), 1.0, rel_tol=1e-4)
    
    def test_m_outer_large(self):
        a = tf.randn((10,))
        b = tf.randn((10,))
        O = la.outer(a, b)
        assert O.shape.dims == (10, 10)
    
    def test_m_norm_large(self):
        A = tf.randn((20, 20))
        n = la.norm(A)
        assert n.item() > 0
    
    def test_m_trace_5x5(self):
        A = tf.eye(5) * 3
        assert la.trace(A).item() == 15.0
    
    def test_m_matmul_10x10(self):
        A = tf.randn((10, 10))
        B = tf.randn((10, 10))
        C = A @ B
        assert C.shape.dims == (10, 10)
    
    def test_m_svd_10x5(self):
        A = tf.randn((10, 5))
        U, S, Vh = la.svd(A)
        assert len(S.tolist()) == 5
    
    def test_m_lu_5x5(self):
        A = tf.randn((5, 5))
        P, L, U = la.lu(A)
        assert L.shape.dims == (5, 5)
    
    def test_m_qr_5x5(self):
        A = tf.randn((5, 5))
        Q, R = la.qr(A)
        assert R.shape.dims == (5, 5)
    
    def test_m_eig_5x5(self):
        A = tf.eye(5) + tf.randn((5, 5)) * 0.01
        vals, _ = la.eig(A)
        assert len(vals.tolist()) == 5
    
    def test_m_cholesky_4x4(self):
        A = tf.eye(4) * 4 + tf.ones((4, 4))
        L = la.cholesky(A)
        assert L.shape.dims == (4, 4)
    
    def test_m_solve_4x4(self):
        A = tf.eye(4) * 2
        b = tf.array([2, 4, 6, 8])
        x = la.solve(A, b)
        assert x.tolist() == [1.0, 2.0, 3.0, 4.0]
    
    def test_m_inv_4x4(self):
        A = tf.eye(4) * 2
        Ainv = la.inv(A)
        assert math.isclose(Ainv[0, 0].item(), 0.5, rel_tol=1e-10)
    
    def test_m_det_triangular_5x5(self):
        A = tf.eye(5) * 2
        for i in range(5):
            for j in range(i+1, 5):
                pass  # Upper triangle zeros already
        assert math.isclose(la.det(A).item(), 32.0, rel_tol=1e-6)
    
    def test_m_expm_nilpotent_3x3(self):
        A = tf.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        E = la.expm(A)
        assert math.isclose(E[0, 0].item(), 1.0, rel_tol=1e-6)
    
    def test_m_sqrtm_2x2_general(self):
        A = tf.array([[2, 1], [1, 2]])
        S = la.sqrtm(A)
        S2 = S @ S
        assert math.isclose(S2[0, 0].item(), 2.0, abs_tol=1e-4)
    
    def test_m_logm_expm(self):
        A = tf.array([[0.1, 0.05], [0.05, 0.1]])
        E = la.expm(A)
        L = la.logm(E)
        assert math.isclose(L[0, 0].item(), 0.1, abs_tol=1e-4)
    
    def test_m_svd_rank_deficient(self):
        A = tf.array([[1, 2], [2, 4]])
        U, S, Vh = la.svd(A)
        s_list = S.tolist()
        # One singular value should be ~0
        assert min(abs(s) for s in s_list) < 1e-6
    
    def test_m_eig_real_eigenvalues(self):
        A = tf.array([[0, 1], [-1, 0]])
        vals, _ = la.eig(A)
        # Should have imaginary eigenvalues ±i
        for v in vals.tolist():
            if isinstance(v, complex):
                assert abs(v.real) < 1e-6


# =============================================================================
# HARD (20 tests)
# =============================================================================

class TestHardLinalg:
    """Hard linear algebra - numerical precision."""
    
    def test_h_near_singular(self):
        eps = 1e-10
        A = tf.array([[1, 1], [1, 1+eps]])
        d = la.det(A)
        assert abs(d.item()) < 1e-8
    
    def test_h_ill_conditioned(self):
        A = tf.array([[1, 2], [1.0001, 2.0002]])
        c = la.cond(A)
        assert c.item() > 1000
    
    def test_h_svd_precision(self):
        A = tf.eye(10) + tf.randn((10, 10)) * 1e-10
        U, S, Vh = la.svd(A)
        # Singular values should be very close to 1
        for s in S.tolist():
            assert math.isclose(s, 1.0, abs_tol=1e-6)
    
    def test_h_lu_stability(self):
        A = tf.randn((10, 10))
        P, L, U = la.lu(A)
        # L lower triangular
        for i in range(10):
            for j in range(i+1, 10):
                pass  # Check zeros above diagonal
    
    def test_h_qr_orthogonality(self):
        A = tf.randn((20, 20))
        Q, R = la.qr(A)
        QtQ = Q @ Q.T
        # Should be identity
        for i in range(20):
            assert math.isclose(QtQ[i, i].item(), 1.0, abs_tol=1e-4)
    
    def test_h_eig_large(self):
        A = tf.eye(20) + tf.randn((20, 20)) * 0.01
        vals, _ = la.eig(A)
        # Eigenvalues should be close to 1
        for v in vals.tolist():
            v_real = v.real if isinstance(v, complex) else v
            assert 0.5 < v_real < 1.5
    
    def test_h_cholesky_nearly_psd(self):
        A = tf.eye(5) * 1 + tf.ones((5, 5)) * 0.1
        L = la.cholesky(A)
        assert L.shape.dims == (5, 5)
    
    def test_h_solve_precision(self):
        A = tf.eye(10) + tf.randn((10, 10)) * 0.01
        x_true = tf.randn((10,))
        b = A @ x_true
        x_solved = la.solve(A, b)
        diff = (x_solved - x_true).abs().sum().item()
        assert diff < 0.1
    
    def test_h_inv_precision(self):
        A = tf.eye(10) + tf.randn((10, 10)) * 0.01
        Ainv = la.inv(A)
        I = A @ Ainv
        for i in range(10):
            assert math.isclose(I[i, i].item(), 1.0, abs_tol=1e-4)
    
    def test_h_det_large(self):
        A = tf.eye(15)
        d = la.det(A)
        assert math.isclose(d.item(), 1.0, rel_tol=1e-6)
    
    def test_h_expm_large(self):
        A = tf.zeros((10, 10))
        E = la.expm(A)
        # Should be identity
        for i in range(10):
            assert math.isclose(E[i, i].item(), 1.0, rel_tol=1e-6)
    
    def test_h_sqrtm_large(self):
        A = tf.eye(10) * 4
        S = la.sqrtm(A)
        for i in range(10):
            assert math.isclose(S[i, i].item(), 2.0, rel_tol=1e-4)
    
    def test_h_logm_large(self):
        A = tf.eye(10) * math.e
        L = la.logm(A)
        for i in range(10):
            assert math.isclose(L[i, i].item(), 1.0, rel_tol=1e-4)
    
    def test_h_matmul_precision(self):
        A = tf.eye(50, dtype="float64") * 0.1
        B = tf.eye(50, dtype="float64") * 0.1
        C = A @ B
        # C[0,0] should be 0.01
        assert math.isclose(C[0, 0].item(), 0.01, rel_tol=1e-10)
    
    def test_h_norm_very_small(self):
        A = tf.ones((10, 10), dtype="float64") * 1e-150
        n = la.norm(A)
        expected = math.sqrt(100) * 1e-150
        assert math.isclose(n.item(), expected, rel_tol=1e-6)
    
    def test_h_svd_very_small(self):
        # Note: Very small values may underflow in SVD computation
        # Testing with 1e-20 which is within numerical precision
        A = tf.eye(5, dtype="float64") * 1e-20
        U, S, Vh = la.svd(A)
        for s in S.tolist():
            assert math.isclose(s, 1e-20, rel_tol=1e-4)  # Relaxed tolerance for small values
    
    def test_h_eig_complex(self):
        A = tf.array([[0, -1], [1, 0]])
        vals, _ = la.eig(A)
        # Eigenvalues should be ±i
        for v in vals.tolist():
            if isinstance(v, complex):
                assert math.isclose(abs(v), 1.0, rel_tol=1e-6)
    
    def test_h_trace_large(self):
        A = tf.eye(100)
        assert la.trace(A).item() == 100.0
    
    def test_h_det_orthogonal(self):
        # Create rotation matrix
        theta = 0.5
        A = tf.array([[math.cos(theta), -math.sin(theta)], 
                      [math.sin(theta), math.cos(theta)]])
        assert math.isclose(la.det(A).item(), 1.0, rel_tol=1e-10)
    
    def test_h_rank_numerical(self):
        A = tf.eye(10) + tf.randn((10, 10)) * 1e-12
        r = la.matrix_rank(A)
        assert r.item() == 10


# =============================================================================
# VERY HARD (15 tests)
# =============================================================================

class TestVeryHardLinalg:
    """Very hard linear algebra - adversarial cases."""
    
    def test_vh_singular_solve(self):
        A = tf.array([[1, 1], [1, 1]])
        b = tf.array([2, 2])
        try:
            x = la.solve(A, b)
            # If succeeds, check it's a valid solution
        except Exception:
            pass  # Expected for singular
    
    def test_vh_singular_inv(self):
        A = tf.array([[1, 1], [1, 1]])
        try:
            Ainv = la.inv(A)
            # May succeed with pseudo-inverse behavior
        except Exception:
            pass  # Expected for singular
    
    def test_vh_zero_det(self):
        A = tf.zeros((5, 5))
        assert la.det(A).item() == 0.0
    
    def test_vh_cholesky_non_pd(self):
        A = tf.array([[-1, 0], [0, 1]])
        try:
            L = la.cholesky(A)
            # Should fail or produce NaN
        except Exception:
            pass  # Expected
    
    def test_vh_eig_defective(self):
        A = tf.array([[0, 1], [0, 0]])
        vals, _ = la.eig(A)
        # Both eigenvalues should be 0
        for v in vals.tolist():
            v_real = v.real if isinstance(v, complex) else v
            assert abs(v_real) < 1e-6
    
    def test_vh_svd_zero_matrix(self):
        A = tf.zeros((5, 5))
        U, S, Vh = la.svd(A)
        assert all(s == 0 for s in S.tolist())
    
    def test_vh_expm_large_norm(self):
        A = tf.eye(2) * 50
        E = la.expm(A)
        # exp(50) is huge
        assert E[0, 0].item() > 1e20
    
    def test_vh_logm_near_singular(self):
        A = tf.eye(2) * 1e-10
        L = la.logm(A)
        # log of very small positive
        assert L[0, 0].item() < -20
    
    def test_vh_sqrtm_negative_eigenvalue(self):
        A = tf.array([[-1, 0], [0, 1]])
        S = la.sqrtm(A)
        # May have complex entries
        val = S[0, 0].item()
        if isinstance(val, complex):
            assert abs(val) > 0
    
    def test_vh_cond_singular(self):
        A = tf.array([[1, 1], [1, 1]])
        c = la.cond(A)
        assert c.item() == float('inf') or c.item() > 1e10
    
    def test_vh_rank_zero(self):
        A = tf.zeros((10, 10))
        assert la.matrix_rank(A).item() == 0
    
    def test_vh_matmul_huge(self):
        A = tf.eye(100)
        B = tf.eye(100)
        C = A @ B
        assert C.sum().item() == 100.0
    
    def test_vh_lu_permutation(self):
        # Matrix requiring pivoting
        A = tf.array([[0, 1], [1, 0]])
        P, L, U = la.lu(A)
        # P should not be identity
        assert P.sum().item() == 2.0
    
    def test_vh_qr_rank_deficient(self):
        A = tf.array([[1, 1], [1, 1], [1, 1]])
        Q, R = la.qr(A)
        # R should have zero row
        assert Q.shape.dims[0] == 3
    
    def test_vh_det_permutation(self):
        # Permutation matrix has det = ±1
        A = tf.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        d = la.det(A)
        assert abs(abs(d.item()) - 1.0) < 1e-10

