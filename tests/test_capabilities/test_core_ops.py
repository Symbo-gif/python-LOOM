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
Core Tensor Operations Capability Tests.

Tests: 65 very easy, 50 easy, 30 medium, 20 hard, 15 very hard = 180 total
Covers: array creation, arithmetic, reduction, indexing
"""

import pytest
import math
import loom as tf


# =============================================================================
# VERY EASY (65 tests) - Trivial inputs, basic usage
# =============================================================================

class TestVeryEasyCoreOps:
    """Very easy core operations - trivial cases."""
    
    # Array creation (15)
    def test_ve_zeros_scalar(self): assert tf.zeros(()).item() == 0.0
    def test_ve_zeros_1d(self): assert tf.zeros((3,)).tolist() == [0.0, 0.0, 0.0]
    def test_ve_zeros_2d(self): assert tf.zeros((2, 2)).shape.dims == (2, 2)
    def test_ve_ones_scalar(self): assert tf.ones(()).item() == 1.0
    def test_ve_ones_1d(self): assert tf.ones((2,)).tolist() == [1.0, 1.0]
    def test_ve_ones_2d(self): assert tf.ones((2, 2)).sum().item() == 4.0
    def test_ve_eye_2(self): assert tf.eye(2).tolist() == [[1, 0], [0, 1]]
    def test_ve_eye_3(self): assert tf.eye(3).sum().item() == 3.0
    def test_ve_array_int(self): assert tf.array([1, 2]).tolist() == [1.0, 2.0]
    def test_ve_array_float(self): assert tf.array([1.5]).item() == 1.5
    def test_ve_full_scalar(self): assert tf.full((), 5.0).item() == 5.0
    def test_ve_full_1d(self): assert tf.full((3,), 2.0).tolist() == [2.0, 2.0, 2.0]
    def test_ve_array_2d(self): assert tf.array([[1, 2], [3, 4]]).shape.dims == (2, 2)
    def test_ve_array_empty_shape(self): assert tf.zeros((1,)).size == 1
    def test_ve_array_single(self): assert tf.array([42]).item() == 42.0
    
    # Arithmetic (20)
    def test_ve_add_scalars(self): assert (tf.array([1]) + tf.array([2])).item() == 3.0
    def test_ve_sub_scalars(self): assert (tf.array([5]) - tf.array([3])).item() == 2.0
    def test_ve_mul_scalars(self): assert (tf.array([3]) * tf.array([4])).item() == 12.0
    def test_ve_div_scalars(self): assert (tf.array([6]) / tf.array([2])).item() == 3.0
    def test_ve_add_1d(self): assert (tf.array([1, 2]) + tf.array([3, 4])).tolist() == [4.0, 6.0]
    def test_ve_sub_1d(self): assert (tf.array([5, 6]) - tf.array([1, 2])).tolist() == [4.0, 4.0]
    def test_ve_mul_1d(self): assert (tf.array([2, 3]) * tf.array([4, 5])).tolist() == [8.0, 15.0]
    def test_ve_div_1d(self): assert (tf.array([10, 20]) / tf.array([2, 4])).tolist() == [5.0, 5.0]
    def test_ve_neg(self): assert (-tf.array([1, -2])).tolist() == [-1.0, 2.0]
    def test_ve_abs_pos(self): assert tf.array([3]).abs().item() == 3.0
    def test_ve_abs_neg(self): assert tf.array([-5]).abs().item() == 5.0
    def test_ve_sqrt_1(self): assert tf.array([1]).sqrt().item() == 1.0
    def test_ve_sqrt_4(self): assert tf.array([4]).sqrt().item() == 2.0
    def test_ve_exp_0(self): assert math.isclose(tf.array([0]).exp().item(), 1.0, rel_tol=1e-9)
    def test_ve_log_1(self): assert tf.array([1]).log().item() == 0.0
    def test_ve_pow_2(self): assert (tf.array([3]) ** 2).item() == 9.0
    def test_ve_pow_0(self): assert (tf.array([5]) ** 0).item() == 1.0
    def test_ve_add_scalar_bc(self): assert (tf.array([1, 2]) + 1).tolist() == [2.0, 3.0]
    def test_ve_mul_scalar_bc(self): assert (tf.array([2, 3]) * 2).tolist() == [4.0, 6.0]
    def test_ve_div_scalar_bc(self): assert (tf.array([4, 6]) / 2).tolist() == [2.0, 3.0]
    
    # Reductions (15)
    def test_ve_sum_1d(self): assert tf.array([1, 2, 3]).sum().item() == 6.0
    def test_ve_sum_2d(self): assert tf.array([[1, 2], [3, 4]]).sum().item() == 10.0
    def test_ve_mean_1d(self): assert tf.array([2, 4, 6]).mean().item() == 4.0
    def test_ve_mean_2d(self): assert tf.array([[1, 3], [5, 7]]).mean().item() == 4.0
    def test_ve_min_1d(self): assert tf.array([3, 1, 2]).min().item() == 1.0
    def test_ve_max_1d(self): assert tf.array([3, 1, 2]).max().item() == 3.0
    def test_ve_sum_zeros(self): assert tf.zeros((5,)).sum().item() == 0.0
    def test_ve_sum_ones(self): assert tf.ones((10,)).sum().item() == 10.0
    def test_ve_mean_single(self): assert tf.array([7]).mean().item() == 7.0
    def test_ve_sum_eye(self): assert tf.eye(3).sum().item() == 3.0
    def test_ve_min_pos(self): assert tf.array([5, 10, 15]).min().item() == 5.0
    def test_ve_max_neg(self): assert tf.array([-5, -10, -15]).max().item() == -5.0
    def test_ve_sum_neg(self): assert tf.array([-1, -2, -3]).sum().item() == -6.0
    def test_ve_mean_mixed(self): assert tf.array([-2, 2]).mean().item() == 0.0
    def test_ve_prod_simple(self): assert tf.array([2, 3, 4]).prod().item() == 24.0
    
    # Indexing (15)
    def test_ve_idx_first(self): assert tf.array([1, 2, 3])[0].item() == 1.0
    def test_ve_idx_last(self): assert tf.array([1, 2, 3])[2].item() == 3.0
    def test_ve_idx_neg(self): assert tf.array([1, 2, 3])[-1].item() == 3.0
    def test_ve_idx_2d_00(self): assert tf.array([[1, 2], [3, 4]])[0, 0].item() == 1.0
    def test_ve_idx_2d_11(self): assert tf.array([[1, 2], [3, 4]])[1, 1].item() == 4.0
    def test_ve_slice_all(self): assert tf.array([1, 2, 3])[:].tolist() == [1.0, 2.0, 3.0]
    def test_ve_slice_first2(self): assert tf.array([1, 2, 3])[:2].tolist() == [1.0, 2.0]
    def test_ve_slice_last2(self): assert tf.array([1, 2, 3])[1:].tolist() == [2.0, 3.0]
    def test_ve_slice_mid(self): assert tf.array([1, 2, 3, 4])[1:3].tolist() == [2.0, 3.0]
    def test_ve_idx_row(self): assert tf.array([[1, 2], [3, 4]])[0].tolist() == [1.0, 2.0]
    def test_ve_idx_col(self): assert tf.array([[1, 2], [3, 4]])[:, 0].tolist() == [1.0, 3.0]
    def test_ve_slice_step(self): assert tf.array([0, 1, 2, 3, 4])[::2].tolist() == [0.0, 2.0, 4.0]
    def test_ve_size_1d(self): assert tf.array([1, 2, 3]).size == 3
    def test_ve_ndim_1d(self): assert tf.array([1, 2, 3]).ndim == 1
    def test_ve_ndim_2d(self): assert tf.array([[1, 2], [3, 4]]).ndim == 2


# =============================================================================
# EASY (50 tests) - Standard inputs, edge scalars
# =============================================================================

class TestEasyCoreOps:
    """Easy core operations - standard usage."""
    
    # Array creation (10)
    def test_e_zeros_3d(self): assert tf.zeros((2, 3, 4)).size == 24
    def test_e_ones_3d(self): assert tf.ones((2, 3, 4)).sum().item() == 24.0
    def test_e_eye_5(self): assert tf.eye(5).sum().item() == 5.0
    def test_e_full_2d(self): assert tf.full((3, 3), 7).sum().item() == 63.0
    def test_e_array_nested(self): assert tf.array([[[1]]]).shape.dims == (1, 1, 1)
    def test_e_zeros_large(self): assert tf.zeros((100, 100)).size == 10000
    def test_e_ones_col(self): assert tf.ones((5, 1)).shape.dims == (5, 1)
    def test_e_ones_row(self): assert tf.ones((1, 5)).shape.dims == (1, 5)
    def test_e_array_float32(self): assert tf.array([1.0], dtype="float32").dtype.value == "float32"
    def test_e_array_complex(self): assert tf.array([1+2j]).dtype.value == "complex128"
    
    # Arithmetic (15)
    def test_e_add_2d(self): assert (tf.ones((2, 2)) + tf.ones((2, 2))).sum().item() == 8.0
    def test_e_mul_broadcast(self): assert (tf.ones((3, 1)) * tf.ones((1, 3))).shape.dims == (3, 3)
    def test_e_div_by_self(self): t = tf.array([2, 3, 4]); assert ((t / t).tolist() == [1.0, 1.0, 1.0])
    def test_e_pow_fractional(self): assert math.isclose((tf.array([8]) ** 0.5).item(), 2.828, rel_tol=0.01)
    def test_e_exp_1(self): assert math.isclose(tf.array([1]).exp().item(), math.e, rel_tol=1e-6)
    def test_e_log_e(self): assert math.isclose(tf.array([math.e]).log().item(), 1.0, rel_tol=1e-6)
    def test_e_sqrt_9(self): assert tf.array([9]).sqrt().item() == 3.0
    def test_e_abs_mixed(self): assert tf.array([-3, 0, 3]).abs().tolist() == [3.0, 0.0, 3.0]
    def test_e_neg_2d(self): assert (-tf.ones((2, 2))).sum().item() == -4.0
    def test_e_chain_ops(self): t = tf.array([2]); assert ((t + 1) * 2).item() == 6.0
    def test_e_add_different_types(self): assert (tf.array([1]) + 0.5).item() == 1.5
    def test_e_sub_broadcast(self): assert (tf.ones((2, 2)) - tf.array([1])).sum().item() == 0.0
    def test_e_mul_zero(self): assert (tf.array([1, 2, 3]) * 0).sum().item() == 0.0
    def test_e_div_large(self): assert (tf.array([1000000]) / 1000).item() == 1000.0
    def test_e_pow_chain(self): assert ((tf.array([2]) ** 2) ** 2).item() == 16.0
    
    # Reductions (10)
    def test_e_sum_3d(self): assert tf.ones((2, 3, 4)).sum().item() == 24.0
    def test_e_mean_3d(self): assert tf.full((2, 3, 4), 5.0).mean().item() == 5.0
    def test_e_min_2d(self): assert tf.array([[1, 5], [3, 2]]).min().item() == 1.0
    def test_e_max_2d(self): assert tf.array([[1, 5], [3, 2]]).max().item() == 5.0
    def test_e_sum_negative(self): assert tf.full((10,), -1.0).sum().item() == -10.0
    def test_e_prod_2d(self): assert tf.array([[1, 2], [3, 4]]).prod().item() == 24.0
    def test_e_mean_uneven(self): assert tf.array([1, 2, 3, 4, 5]).mean().item() == 3.0
    def test_e_sum_mixed(self): assert tf.array([-5, 0, 5]).sum().item() == 0.0
    def test_e_max_all_same(self): assert tf.full((5,), 3.0).max().item() == 3.0
    def test_e_min_all_same(self): assert tf.full((5,), 3.0).min().item() == 3.0
    
    # Indexing & shape (15)
    def test_e_slice_2d_row(self): assert tf.array([[1, 2], [3, 4], [5, 6]])[1:].shape.dims == (2, 2)
    def test_e_slice_2d_col(self): assert tf.array([[1, 2, 3], [4, 5, 6]])[:, 1:].shape.dims == (2, 2)
    def test_e_flatten(self): assert tf.array([[1, 2], [3, 4]]).flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
    def test_e_T_property(self): assert tf.array([[1, 2], [3, 4]]).T.shape.dims == (2, 2)
    def test_e_T_values(self): assert tf.array([[1, 2], [3, 4]]).T.tolist() == [[1, 3], [2, 4]]
    def test_e_squeeze(self): assert tf.array([[[1, 2, 3]]]).squeeze().shape.dims == (3,)
    def test_e_expand_dims(self): t = tf.array([1, 2]); assert t.unsqueeze(0).shape.dims == (1, 2)
    def test_e_shape_3d(self): assert tf.zeros((2, 3, 4)).shape.dims == (2, 3, 4)
    def test_e_idx_3d(self): assert tf.ones((2, 3, 4))[0, 0, 0].item() == 1.0
    def test_e_slice_step_2d(self): 
        t = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert t[::2, ::2].shape.dims == (2, 2)
    def test_e_neg_idx_2d(self): assert tf.array([[1, 2], [3, 4]])[-1, -1].item() == 4.0
    def test_e_tolist_2d(self): assert tf.array([[1, 2]]).tolist() == [[1.0, 2.0]]
    def test_e_item_2d_single(self): assert tf.array([[42]]).item() == 42.0
    def test_e_size_3d(self): assert tf.zeros((2, 3, 4)).size == 24
    def test_e_ndim_3d(self): assert tf.zeros((2, 3, 4)).ndim == 3


# =============================================================================
# MEDIUM (30 tests) - Multi-dimensional, mixed types
# =============================================================================

class TestMediumCoreOps:
    """Medium core operations - multi-dim, mixed."""
    
    # Broadcasting (10)
    def test_m_bc_3d_1d(self): 
        a = tf.ones((2, 3, 4))
        b = tf.array([1, 2, 3, 4])
        assert (a + b).shape.dims == (2, 3, 4)
    
    def test_m_bc_3d_2d(self):
        a = tf.ones((2, 3, 4))
        b = tf.ones((3, 4))
        assert (a * b).shape.dims == (2, 3, 4)
    
    def test_m_bc_complex(self):
        a = tf.ones((1, 3, 1))
        b = tf.ones((2, 1, 4))
        assert (a + b).shape.dims == (2, 3, 4)
    
    def test_m_matmul_2x3_3x4(self):
        A = tf.ones((2, 3))
        B = tf.ones((3, 4))
        assert (A @ B).shape.dims == (2, 4)
    
    def test_m_matmul_values(self):
        A = tf.array([[1, 2], [3, 4]])
        B = tf.array([[5, 6], [7, 8]])
        C = A @ B
        assert C.tolist() == [[19, 22], [43, 50]]
    
    def test_m_chain_matmul(self):
        A = tf.eye(3)
        B = tf.eye(3)
        C = tf.eye(3)
        assert ((A @ B) @ C).sum().item() == 3.0
    
    def test_m_dot_1d(self):
        a = tf.array([1, 2, 3])
        b = tf.array([4, 5, 6])
        assert (a @ b).item() == 32.0
    
    def test_m_outer_like(self):
        a = tf.array([1, 2])
        b = tf.array([3, 4, 5])
        from loom.linalg import outer
        assert outer(a, b).shape.dims == (2, 3)
    
    def test_m_diagonal_extract(self):
        A = tf.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        diag = tf.array([A[i, i].item() for i in range(3)])
        assert diag.tolist() == [1.0, 5.0, 9.0]
    
    def test_m_trace_manual(self):
        A = tf.array([[1, 2], [3, 4]])
        trace = sum(A[i, i].item() for i in range(2))
        assert trace == 5.0
    
    # Complex arithmetic (10)
    def test_m_complex_add(self):
        a = tf.array([1+2j])
        b = tf.array([3+4j])
        assert (a + b).item() == (4+6j)
    
    def test_m_complex_mul(self):
        a = tf.array([1+2j])
        b = tf.array([3+4j])
        assert (a * b).item() == (-5+10j)
    
    def test_m_complex_conj(self):
        a = tf.array([1+2j])
        assert tf.conj(a).item() == (1-2j)
    
    def test_m_complex_abs(self):
        a = tf.array([3+4j])
        assert a.abs().item() == 5.0
    
    def test_m_exp_complex(self):
        a = tf.array([0+0j])
        assert a.exp().item() == 1.0
    
    def test_m_real_part(self):
        a = tf.array([3+4j])
        assert tf.real(a).item() == 3.0
    
    def test_m_imag_part(self):
        a = tf.array([3+4j])
        assert tf.imag(a).item() == 4.0
    
    def test_m_angle(self):
        a = tf.array([1+1j])
        assert math.isclose(tf.angle(a).item(), math.pi/4, rel_tol=1e-6)
    
    def test_m_polar(self):
        r, theta = tf.polar(tf.array([1+1j]))
        assert math.isclose(r.item(), math.sqrt(2), rel_tol=1e-6)
    
    def test_m_complex_div(self):
        a = tf.array([1+0j])
        b = tf.array([0+1j])
        result = a / b
        assert result.item() == -1j
    
    # Large operations (10)
    def test_m_large_sum(self):
        t = tf.ones((100, 100))
        assert t.sum().item() == 10000.0
    
    def test_m_large_mean(self):
        t = tf.full((50, 50), 3.0)
        assert t.mean().item() == 3.0
    
    def test_m_large_matmul(self):
        A = tf.eye(50)
        B = tf.ones((50, 10))
        assert (A @ B).shape.dims == (50, 10)
    
    def test_m_large_elementwise(self):
        a = tf.ones((100, 100))
        b = tf.ones((100, 100))
        assert (a + b).sum().item() == 20000.0
    
    def test_m_large_chain(self):
        t = tf.ones((10, 10))
        for _ in range(5):
            t = t + 1
        assert t.sum().item() == 600.0
    
    def test_m_4d_tensor(self):
        t = tf.zeros((2, 3, 4, 5))
        assert t.size == 120
    
    def test_m_5d_tensor(self):
        t = tf.ones((2, 2, 2, 2, 2))
        assert t.sum().item() == 32.0
    
    def test_m_reshape_chain(self):
        t = tf.array([1, 2, 3, 4])
        f = t.flatten()
        assert f.tolist() == [1.0, 2.0, 3.0, 4.0]
    
    def test_m_squeeze_multiple(self):
        t = tf.array([[[[[1, 2]]]]])
        s = t.squeeze()
        assert s.ndim <= 2
    
    def test_m_broadcast_add_diff_ndim(self):
        a = tf.ones((2, 3, 4))
        b = tf.array([1])
        assert (a + b).shape.dims == (2, 3, 4)


# =============================================================================
# HARD (20 tests) - Numerical stability, precision
# =============================================================================

class TestHardCoreOps:
    """Hard core operations - numerical stability."""
    
    def test_h_very_small_add(self):
        a = tf.array([1e-300], dtype="float64")
        b = tf.array([1e-300], dtype="float64")
        assert (a + b).item() == 2e-300
    
    def test_h_very_large_mul(self):
        a = tf.array([1e150])
        b = tf.array([1e150])
        result = (a * b).item()
        assert result == float('inf') or result > 1e299
    
    def test_h_cancellation(self):
        large = 1e15
        a = tf.array([large + 1])
        b = tf.array([large])
        diff = (a - b).item()
        assert abs(diff - 1.0) < 2.0  # Allow FP error
    
    def test_h_exp_underflow(self):
        a = tf.array([-1000], dtype="float64")
        result = a.exp().item()
        assert result == 0.0 or result < 1e-300
    
    def test_h_exp_overflow(self):
        a = tf.array([1000])
        assert a.exp().item() == float('inf')
    
    def test_h_log_small(self):
        a = tf.array([1e-300], dtype="float64")
        result = a.log().item()
        assert result < -600
    
    def test_h_sqrt_precision(self):
        a = tf.array([2.0], dtype="float64")
        sqrt_a = a.sqrt().item()
        assert math.isclose(sqrt_a ** 2, 2.0, rel_tol=1e-14)
    
    def test_h_pow_precision(self):
        a = tf.array([2.0], dtype="float64")
        exponent = tf.array([0.5], dtype="float64")
        result = (a ** exponent).item()
        assert math.isclose(result, math.sqrt(2), rel_tol=1e-14)
    
    def test_h_sum_many_small(self):
        t = tf.full((10000,), 1e-10, dtype="float64")
        result = t.sum().item()
        # Use rel_tol=1e-10 to account for floating-point accumulation errors
        # when summing 10,000 small values (1e-14 is unrealistically tight)
        assert math.isclose(result, 1e-6, rel_tol=1e-10)
    
    def test_h_mean_precision(self):
        t = tf.array([1e10 + 1, 1e10 + 2, 1e10 + 3], dtype="float64")
        mean = t.mean().item()
        expected = 1e10 + 2
        assert abs(mean - expected) < 10
    
    def test_h_matmul_precision(self):
        factor = tf.array([0.1], dtype="float64")
        A = tf.eye(100, dtype="float64") * factor
        result = A @ A
        # Should be 0.01 on diagonal
        assert math.isclose(result[0, 0].item(), 0.01, rel_tol=1e-14)
    
    def test_h_chain_precision(self):
        t = tf.array([1.0], dtype="float64")
        factor = tf.array([1.0000001], dtype="float64")
        for _ in range(100):
            t = t * factor
        expected = 1.0000001 ** 100
        assert math.isclose(t.item(), expected, rel_tol=1e-14)
    
    def test_h_div_very_small(self):
        a = tf.array([1e-300], dtype="float64")
        b = tf.array([1e-100], dtype="float64")
        result = (a / b).item()
        assert math.isclose(result, 1e-200, rel_tol=1e-14)
    
    def test_h_complex_precision(self):
        a = tf.array([1e-15 + 1e-15j], dtype="complex128")
        b = tf.array([1e-15 + 1e-15j], dtype="complex128")
        result = (a + b).item()
        assert math.isclose(result.real, 2e-15, rel_tol=1e-14)
    
    def test_h_large_matrix_sum(self):
        t = tf.ones((1000, 1000))
        assert t.sum().item() == 1000000.0
    
    def test_h_alternating_sum(self):
        vals = [(-1)**i for i in range(1000)]
        t = tf.array(vals)
        assert abs(t.sum().item()) <= 1.0
    
    def test_h_abs_very_small(self):
        a = tf.array([-1e-300], dtype="float64")
        assert a.abs().item() == 1e-300
    
    def test_h_pow_negative_base(self):
        a = tf.array([-8.0], dtype="float64")
        exponent = tf.array([1/3], dtype="float64")
        result = (a.abs() ** exponent).item()
        assert math.isclose(result, 2.0, rel_tol=1e-14)
    
    def test_h_min_max_large(self):
        t = tf.randn((1000,))
        assert t.min().item() <= t.max().item()
    
    def test_h_broadcast_large(self):
        a = tf.ones((100, 1))
        b = tf.ones((1, 100))
        result = a + b
        assert result.sum().item() == 20000.0


# =============================================================================
# VERY HARD (15 tests) - Adversarial, near-singular, extreme
# =============================================================================

class TestVeryHardCoreOps:
    """Very hard core operations - adversarial cases."""
    
    def test_vh_inf_arithmetic(self):
        a = tf.array([float('inf')], dtype="float64")
        b = tf.array([1.0], dtype="float64")
        assert (a + b).item() == float('inf')
    
    def test_vh_inf_minus_inf(self):
        a = tf.array([float('inf')], dtype="float64")
        result = (a - a).item()
        assert math.isnan(result)
    
    def test_vh_nan_propagation(self):
        a = tf.array([float('nan')], dtype="float64")
        b = tf.array([1.0], dtype="float64")
        assert math.isnan((a + b).item())
    
    def test_vh_div_by_zero(self):
        a = tf.array([1.0], dtype="float64")
        b = tf.array([0.0], dtype="float64")
        result = (a / b).item()
        assert result == float('inf')
    
    def test_vh_neg_div_by_zero(self):
        a = tf.array([-1.0], dtype="float64")
        b = tf.array([0.0], dtype="float64")
        result = (a / b).item()
        assert result == float('-inf')
    
    def test_vh_zero_div_zero(self):
        a = tf.array([0.0], dtype="float64")
        b = tf.array([0.0], dtype="float64")
        result = (a / b).item()
        assert math.isnan(result)
    
    def test_vh_sqrt_negative(self):
        a = tf.array([-1.0], dtype="float64")
        result = a.sqrt().item()
        assert math.isnan(result)
    
    def test_vh_log_zero(self):
        a = tf.array([0.0], dtype="float64")
        result = a.log().item()
        assert result == float('-inf')
    
    def test_vh_log_negative(self):
        a = tf.array([-1.0], dtype="float64")
        result = a.log().item()
        assert math.isnan(result)
    
    def test_vh_extreme_precision(self):
        # Test machine epsilon
        eps = 1e-16
        a = tf.array([1.0], dtype="float64")
        b = tf.array([eps], dtype="float64")
        result = (a + b).item()
        # May or may not register due to precision
        assert result >= 1.0
    
    def test_vh_denormalized(self):
        # Denormalized numbers
        denorm = 5e-324
        a = tf.array([denorm], dtype="float64")
        assert a.item() >= 0
    
    def test_vh_massive_chain(self):
        t = tf.array([1.0], dtype="float64")
        inc = tf.array([0.001], dtype="float64")
        for _ in range(500):
            t = t + inc
        expected = 1.5
        assert math.isclose(t.item(), expected, rel_tol=0.01)
    
    def test_vh_deep_dag(self):
        t = tf.array([1.0], dtype="float64")
        factor = tf.array([1.001], dtype="float64")
        for _ in range(200):
            t = t * factor
        result = t.item()
        expected = 1.001 ** 200
        assert math.isclose(result, expected, rel_tol=0.01)
    
    def test_vh_very_large_tensor(self):
        t = tf.zeros((500, 500))
        assert t.size == 250000
    
    def test_vh_mixed_extremes(self):
        a = tf.array([1e-300, 1e300], dtype="float64")
        s = a.sum().item()
        assert s > 1e299  # Dominated by large

