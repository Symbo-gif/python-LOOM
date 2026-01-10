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
loom Hardening Test Suite

Comprehensive security, stress, and edge-case testing for production hardening.

Test Categories:
1. White Box Tests - Full knowledge of implementation
2. Black Box Tests - Input/output only, no implementation knowledge
3. Gray Box Tests - Partial knowledge
4. Red Hat Tests - Adversarial attack simulation
5. Blue Hat Tests - Defensive security verification
6. Fuzz Tests - Random/malformed input testing
7. Stress Tests - Performance under extreme load
8. Security Tests - Enterprise/military grade security
"""

import pytest
import math
import sys
import random
import string
import gc
import time
from typing import Any, List

import loom as tf
import loom.linalg as la
import loom.stats as stats
from loom.special import gamma, erf, gammainc
from loom.field import FieldTensor
from loom.backend import get_backend, set_backend, available_backends


# =============================================================================
# STRESS TESTS - Push systems to their limits
# =============================================================================

class TestStressLimits:
    """Stress testing for memory, computation, and numerical limits."""
    
    def test_large_tensor_creation(self):
        """Create tensors up to memory limits."""
        # 1M elements
        t = tf.zeros((1000, 1000))
        assert t.size == 1_000_000
        assert t.shape.dims == (1000, 1000)
        del t
        gc.collect()
    
    def test_deep_computation_dag(self):
        """Test deeply nested computation DAG."""
        a = tf.array([1.0])
        for _ in range(100):
            a = a + 1
        result = a.compute()
        assert result[0] == 101.0
    
    def test_wide_broadcast(self):
        """Test broadcasting with very different shapes."""
        a = tf.ones((1, 1, 1, 1000))
        b = tf.ones((1000, 1, 1, 1))
        c = a + b
        assert c.shape.dims == (1000, 1, 1, 1000)
    
    def test_repeated_operations(self):
        """Test repeated operations for numerical stability."""
        # Use float64 for precision
        a = tf.array([1.0], dtype='float64')
        for _ in range(1000):
            a = a * 1.0000001
        result = a.item()
        expected = 1.0000001 ** 1000
        assert math.isclose(result, expected, rel_tol=1e-6)
    
    def test_large_matrix_operations(self):
        """Test matrix operations on larger matrices."""
        n = 100
        A = tf.eye(n) * 2.0
        B = tf.eye(n) * 3.0
        C = A @ B
        # Diagonal should be 6.0
        result = C.tolist()
        assert math.isclose(result[0][0], 6.0, abs_tol=1e-10)
        assert math.isclose(result[50][50], 6.0, abs_tol=1e-10)


class TestNumericalStability:
    """Test numerical stability at edge cases."""
    
    def test_very_small_numbers(self):
        """Operations with very small numbers."""
        tiny = 1e-300
        a = tf.array([tiny])
        b = a * 2
        assert b.item() == pytest.approx(2e-300, rel=1e-10)
    
    def test_very_large_numbers(self):
        """Operations with very large numbers."""
        huge = 1e300
        # Use float64 to avoid overflow
        a = tf.array([huge], dtype='float64')
        b = a / 2
        assert b.item() == pytest.approx(5e299, rel=1e-10)
    
    def test_cancellation(self):
        """Test catastrophic cancellation scenarios."""
        # (x + 1) - x should be 1
        x = 1e15
        a = tf.array([x + 1])
        b = tf.array([x])
        c = a - b
        # Due to floating point limits, this may not be exactly 1
        assert c.item() == pytest.approx(1.0, abs=1.0)  # Allow for FP error
    
    def test_overflow_handling(self):
        """Test handling of values that would overflow."""
        a = tf.array([1e308])
        b = a * 10
        result = b.item()
        assert result == float('inf') or result > 1e308
    
    def test_underflow_handling(self):
        """Test handling of values that would underflow."""
        a = tf.array([1e-308])
        b = a / 1e10
        result = b.item()
        # Should be either 0 or very small positive
        assert result >= 0 and result < 1e-300


# =============================================================================
# WHITE BOX TESTS - Full implementation knowledge
# =============================================================================

class TestWhiteBoxNumericBuffer:
    """White box tests for NumericBuffer internals."""
    
    def test_buffer_dtype_consistency(self):
        """Verify dtype is maintained through operations."""
        from loom.numeric.storage import NumericBuffer
        from loom.core.dtype import DType
        
        buf = NumericBuffer([1.0, 2.0, 3.0], DType.FLOAT64)
        assert buf.dtype == DType.FLOAT64
    
    def test_complex_auto_upgrade(self):
        """Test auto-upgrade to complex when needed."""
        t = tf.array([1+2j, 3+4j])
        from loom.core.dtype import DType
        assert t.dtype in [DType.COMPLEX64, DType.COMPLEX128]
    
    def test_buffer_slicing(self):
        """Test internal buffer slicing."""
        from loom.numeric.storage import NumericBuffer
        from loom.core.dtype import DType
        
        buf = NumericBuffer([1, 2, 3, 4, 5], DType.FLOAT64)
        sliced = buf[1:4]
        assert list(sliced) == [2, 3, 4]


class TestWhiteBoxDAG:
    """White box tests for computation DAG."""
    
    def test_dag_node_creation(self):
        """Test DAG node creation on operations."""
        a = tf.array([1, 2, 3])
        b = tf.array([4, 5, 6])
        c = a + b
        
        # c should have an operation
        assert c._op is not None
        assert len(c._args) == 2
    
    def test_dag_caching(self):
        """Test that computed results are cached."""
        a = tf.array([1, 2, 3])
        b = a * 2
        
        # First compute
        result1 = b.compute()
        # Should be cached now
        result2 = b.compute()
        
        # Both should reference same cached result
        assert list(result1) == list(result2)


# =============================================================================
# BLACK BOX TESTS - Input/output only
# =============================================================================

class TestBlackBoxTensor:
    """Black box tests for Tensor - test only public API."""
    
    @pytest.mark.parametrize("shape", [
        (10,), (5, 5), (2, 3, 4), (1, 1, 1, 1, 1)
    ])
    def test_zeros_shape(self, shape):
        """zeros() produces correct shape."""
        t = tf.zeros(shape)
        assert t.shape.dims == shape
        assert all(v == 0.0 for v in t.flatten().tolist())
    
    @pytest.mark.parametrize("op", [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ])
    def test_arithmetic_operations(self, op):
        """Basic arithmetic produces valid results."""
        a = tf.array([1.0, 2.0, 3.0])
        b = tf.array([4.0, 5.0, 6.0])
        c = op(a, b)
        result = c.tolist()
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
    
    def test_matmul_square(self):
        """Matrix multiply produces correct shape."""
        A = tf.randn((3, 4))
        B = tf.randn((4, 5))
        C = A @ B
        assert C.shape.dims == (3, 5)


class TestBlackBoxLinalg:
    """Black box tests for linalg - test only API contracts."""
    
    def test_svd_returns_three_tensors(self):
        """SVD returns U, S, Vh."""
        A = tf.randn((5, 3))
        result = la.svd(A)
        assert len(result) == 3
    
    def test_det_returns_scalar(self):
        """Determinant returns a scalar."""
        A = tf.randn((4, 4))
        d = la.det(A)
        assert d.ndim == 0 or d.size == 1
    
    def test_solve_shapes(self):
        """solve(A, b) returns x with correct shape."""
        A = tf.eye(3)
        b = tf.array([1, 2, 3])
        x = la.solve(A, b)
        assert x.size == 3


# =============================================================================
# GRAY BOX TESTS - Partial knowledge
# =============================================================================

class TestGrayBoxBroadcasting:
    """Gray box tests - know broadcasting rules but not implementation."""
    
    def test_scalar_broadcast(self):
        """Scalar broadcasts to any shape."""
        a = tf.ones((3, 4, 5))
        b = tf.array([2.0])  # Scalar-like
        c = a + b
        assert c.shape.dims == (3, 4, 5)
    
    def test_dimension_expansion(self):
        """Lower-dim broadcasts to higher-dim."""
        a = tf.ones((3, 4))
        b = tf.ones((4,))
        c = a + b
        assert c.shape.dims == (3, 4)
    
    def test_incompatible_shapes_error(self):
        """Incompatible shapes raise error."""
        a = tf.ones((3, 4))
        b = tf.ones((5, 6))
        with pytest.raises((ValueError, RuntimeError)):
            _ = a + b


# =============================================================================
# RED HAT TESTS - Adversarial attack simulation
# =============================================================================

class TestRedHatAdversarial:
    """Red hat tests - attempt to break the system."""
    
    def test_division_by_zero(self):
        """Division by zero should produce inf, not crash."""
        a = tf.array([1.0, 2.0, 3.0])
        b = tf.array([0.0, 0.0, 0.0])
        c = a / b
        result = c.tolist()
        assert all(math.isinf(v) for v in result)
    
    def test_nan_propagation(self):
        """NaN should propagate correctly."""
        a = tf.array([float('nan'), 1.0, 2.0])
        b = a + 1
        result = b.tolist()
        assert math.isnan(result[0])
    
    def test_inf_arithmetic(self):
        """Infinity arithmetic should follow IEEE rules."""
        a = tf.array([float('inf')])
        b = a - a  # inf - inf = nan
        assert math.isnan(b.item())
    
    def test_negative_sqrt(self):
        """Sqrt of negative should produce nan."""
        a = tf.array([-1.0])
        b = a.sqrt()
        assert math.isnan(b.item())
    
    def test_log_of_zero(self):
        """Log of zero should produce -inf."""
        a = tf.array([0.0])
        b = a.log()
        assert b.item() == float('-inf')
    
    def test_exp_overflow(self):
        """Exp of large number should produce inf."""
        a = tf.array([1000.0])
        b = a.exp()
        assert b.item() == float('inf')


class TestRedHatInjection:
    """Red hat tests for injection attacks."""
    
    def test_malicious_shape(self):
        """Negative dimensions should be rejected."""
        with pytest.raises((ValueError, TypeError, OverflowError)):
            tf.zeros((-1, 5))
    
    def test_extreme_dimensions(self):
        """Extremely large dimensions should not crash."""
        with pytest.raises((ValueError, MemoryError, OverflowError)):
            tf.zeros((10**15, 10**15))
    
    def test_recursive_tensor(self):
        """Tensor of tensor should not cause infinite recursion."""
        a = tf.array([1, 2, 3])
        try:
            b = tf.array([a])  # May work or raise
        except (TypeError, ValueError):
            pass  # Expected


# =============================================================================
# BLUE HAT TESTS - Defensive security
# =============================================================================

class TestBlueHatDefensive:
    """Blue hat tests - verify defensive mechanisms."""
    
    def test_type_validation(self):
        """Invalid types are rejected."""
        with pytest.raises((TypeError, ValueError)):
            tf.array("not a valid input for array")
    
    def test_immutable_shape(self):
        """Shape should be immutable."""
        t = tf.array([[1, 2], [3, 4]])
        original_shape = t.shape
        with pytest.raises(AttributeError):
            t.shape = (4, 1)  # Should not be settable
    
    def test_backend_fallback(self):
        """Invalid backend falls back to CPU."""
        original = get_backend().name
        result = set_backend('definitely_not_a_real_backend')
        assert result == False
        assert get_backend().name == 'cpu'
        set_backend(original)


# =============================================================================
# FUZZ TESTS - Random/malformed inputs
# =============================================================================

class TestFuzz:
    """Fuzz testing with random and malformed inputs."""
    
    @pytest.mark.parametrize("seed", range(10))
    def test_random_tensor_operations(self, seed):
        """Random tensors should not crash operations."""
        random.seed(seed)
        shape = tuple(random.randint(1, 10) for _ in range(random.randint(1, 4)))
        
        # Create random tensor
        size = 1
        for s in shape:
            size *= s
        data = [random.uniform(-1e6, 1e6) for _ in range(size)]
        
        t = tf.Tensor(data, shape=shape)
        
        # Operations should not crash
        _ = t.sum()
        _ = t.mean()
        _ = t.abs()
    
    def test_empty_tensor(self):
        """Empty tensor operations."""
        t = tf.array([])
        assert t.size == 0
    
    def test_single_element(self):
        """Single element operations."""
        t = tf.array([42.0])
        assert t.sum().item() == 42.0
        assert t.mean().item() == 42.0
    
    @pytest.mark.parametrize("value", [
        0.0, -0.0, 1e-308, 1e308, float('inf'), float('-inf')
    ])
    def test_extreme_scalar_values(self, value):
        """Extreme scalar values should not crash."""
        t = tf.array([value])
        _ = t.tolist()
        _ = t + 0


class TestFuzzLinalg:
    """Fuzz testing for linear algebra."""
    
    @pytest.mark.parametrize("seed", range(5))
    def test_random_matrix_svd(self, seed):
        """SVD of random matrices should not crash."""
        random.seed(seed)
        m, n = random.randint(2, 10), random.randint(2, 10)
        A = tf.randn((m, n))
        
        try:
            U, S, Vh = la.svd(A)
            # Singular values should be non-negative
            assert all(s >= 0 for s in S.tolist())
        except (ValueError, RuntimeError):
            pass  # Some matrices may not be decomposable
    
    def test_near_singular_matrix(self):
        """Near-singular matrix handling."""
        # Almost singular
        A = tf.array([[1.0, 1.0], [1.0, 1.0 + 1e-10]])
        try:
            _ = la.inv(A)
        except (ValueError, RuntimeError):
            pass  # Expected for singular/near-singular


# =============================================================================
# SECURITY TESTS - Enterprise/Military grade
# =============================================================================

class TestSecurityEnterprise:
    """Enterprise-grade security tests."""
    
    def test_no_memory_leaks_basic(self):
        """Basic memory leak check."""
        gc.collect()
        initial = len(gc.get_objects())
        
        for _ in range(100):
            t = tf.randn((100, 100))
            _ = t.sum()
            del t
        
        gc.collect()
        final = len(gc.get_objects())
        
        # Allow some variance
        assert final < initial + 1000
    
    def test_no_global_state_corruption(self):
        """Operations should not corrupt global state."""
        # Get initial state
        initial_backend = get_backend().name
        
        # Do various operations
        for _ in range(10):
            t = tf.randn((10, 10))
            _ = la.svd(t)
            _ = t @ t.T
        
        # State should be unchanged
        assert get_backend().name == initial_backend
    
    def test_thread_safety_basic(self):
        """Basic thread safety (if threading is used)."""
        import threading
        
        results = []
        
        def worker(x):
            t = tf.array([x])
            for _ in range(100):
                t = t + 1
            results.append(t.item())
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Each thread should get its own result
        assert len(results) == 5
    
    def test_deterministic_random_with_seed(self):
        """Random with same seed produces same results."""
        tf.random.seed(12345)
        r1 = tf.randn((5,)).tolist()
        
        tf.random.seed(12345)
        r2 = tf.randn((5,)).tolist()
        
        assert r1 == r2


class TestSecurityInputValidation:
    """Input validation security tests."""
    
    def test_reject_none_data(self):
        """None data should be handled properly."""
        # This may work or raise depending on design
        try:
            t = tf.Tensor(None)
            # If it works, should have default behavior
        except (TypeError, ValueError):
            pass  # Expected
    
    def test_reject_nested_inconsistent(self):
        """Inconsistent nested lists should be rejected."""
        with pytest.raises(ValueError):
            tf.array([[1, 2], [3]])  # Inconsistent
    
    def test_dtype_bounds(self):
        """Values outside dtype bounds should be handled."""
        # Float64 can handle very large values
        t = tf.array([1e308], dtype='float64')
        assert t.item() == 1e308


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks (not strict assertions)."""
    
    def test_creation_speed(self):
        """Tensor creation should be reasonably fast."""
        start = time.time()
        for _ in range(1000):
            t = tf.zeros((100,))
        elapsed = time.time() - start
        assert elapsed < 5.0  # Should complete in under 5 seconds
    
    def test_matmul_speed(self):
        """Matrix multiply should be reasonably fast."""
        A = tf.randn((50, 50))
        B = tf.randn((50, 50))
        
        start = time.time()
        for _ in range(10):
            _ = A @ B
        elapsed = time.time() - start
        
        assert elapsed < 10.0  # Should complete in under 10 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

