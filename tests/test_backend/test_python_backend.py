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
Tests for the Python/CPU backend.

Task V1.1-002: Python Backend Wrapper Testing
"""

import pytest
import math

from loom.backend.cpu import CPUBackend, get_cpu_backend


class TestCPUBackendAvailability:
    """Test CPU backend availability."""

    def test_cpu_backend_always_available(self):
        """Python backend should always be available."""
        backend = get_cpu_backend()
        assert backend.is_available is True

    def test_cpu_backend_name(self):
        """CPU backend should have correct name."""
        backend = get_cpu_backend()
        assert backend.name == "cpu"

    def test_cpu_backend_singleton(self):
        """get_cpu_backend should return singleton."""
        backend1 = get_cpu_backend()
        backend2 = get_cpu_backend()
        assert backend1 is backend2


class TestCPUBackendAdd:
    """Test CPU backend addition."""

    def test_add_basic(self):
        """Test basic element-wise addition."""
        backend = get_cpu_backend()
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = backend.add(a, b)
        assert result == [5.0, 7.0, 9.0]

    def test_add_negative(self):
        """Test addition with negative numbers."""
        backend = get_cpu_backend()
        a = [-1.0, -2.0, 3.0]
        b = [1.0, 2.0, -3.0]
        result = backend.add(a, b)
        assert result == [0.0, 0.0, 0.0]

    def test_add_zero(self):
        """Test addition with zeros."""
        backend = get_cpu_backend()
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = backend.add(a, b)
        assert result == [1.0, 2.0, 3.0]


class TestCPUBackendMul:
    """Test CPU backend multiplication."""

    def test_mul_basic(self):
        """Test basic element-wise multiplication."""
        backend = get_cpu_backend()
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = backend.mul(a, b)
        assert result == [4.0, 10.0, 18.0]

    def test_mul_zero(self):
        """Test multiplication with zeros."""
        backend = get_cpu_backend()
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        result = backend.mul(a, b)
        assert result == [0.0, 0.0, 0.0]

    def test_mul_negative(self):
        """Test multiplication with negative numbers."""
        backend = get_cpu_backend()
        a = [-1.0, 2.0, -3.0]
        b = [1.0, -2.0, -3.0]
        result = backend.mul(a, b)
        assert result == [-1.0, -4.0, 9.0]


class TestCPUBackendMatmul:
    """Test CPU backend matrix multiplication."""

    def test_matmul_2x2(self):
        """Test 2x2 matrix multiplication."""
        backend = get_cpu_backend()
        # A = [[1, 2], [3, 4]] -> flattened = [1, 2, 3, 4]
        # B = [[5, 6], [7, 8]] -> flattened = [5, 6, 7, 8]
        # Result = [[19, 22], [43, 50]] -> flattened = [19, 22, 43, 50]
        a = [1.0, 2.0, 3.0, 4.0]
        b = [5.0, 6.0, 7.0, 8.0]
        result = backend.matmul(a, b, (2, 2), (2, 2))
        assert result == [19.0, 22.0, 43.0, 50.0]

    def test_matmul_identity(self):
        """Test multiplication with identity matrix."""
        backend = get_cpu_backend()
        # A = [[1, 2], [3, 4]]
        # I = [[1, 0], [0, 1]]
        a = [1.0, 2.0, 3.0, 4.0]
        identity = [1.0, 0.0, 0.0, 1.0]
        result = backend.matmul(a, identity, (2, 2), (2, 2))
        assert result == [1.0, 2.0, 3.0, 4.0]

    def test_matmul_shape_mismatch(self):
        """Test matrix multiplication with incompatible shapes."""
        backend = get_cpu_backend()
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 2x3
        b = [1.0, 2.0, 3.0, 4.0]  # 2x2
        with pytest.raises(ValueError):
            backend.matmul(a, b, (2, 3), (2, 2))

    def test_matmul_rectangular(self):
        """Test rectangular matrix multiplication."""
        backend = get_cpu_backend()
        # A = [[1, 2, 3], [4, 5, 6]] (2x3)
        # B = [[7, 8], [9, 10], [11, 12]] (3x2)
        # Result = [[58, 64], [139, 154]] (2x2)
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        result = backend.matmul(a, b, (2, 3), (3, 2))
        assert result == [58.0, 64.0, 139.0, 154.0]


class TestCPUBackendSum:
    """Test CPU backend sum reduction."""

    def test_sum_basic(self):
        """Test basic sum."""
        backend = get_cpu_backend()
        a = [1.0, 2.0, 3.0, 4.0]
        result = backend.sum(a)
        assert result == 10.0

    def test_sum_empty(self):
        """Test sum of empty list."""
        backend = get_cpu_backend()
        result = backend.sum([])
        assert result == 0.0

    def test_sum_single(self):
        """Test sum of single element."""
        backend = get_cpu_backend()
        result = backend.sum([42.0])
        assert result == 42.0


class TestCPUBackendMathOps:
    """Test CPU backend mathematical operations."""

    def test_exp_basic(self):
        """Test element-wise exponential."""
        backend = get_cpu_backend()
        a = [0.0, 1.0]
        result = backend.exp(a)
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(math.e)

    def test_log_basic(self):
        """Test element-wise natural logarithm."""
        backend = get_cpu_backend()
        a = [1.0, math.e]
        result = backend.log(a)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1.0)

    def test_log_zero(self):
        """Test log of zero returns -inf."""
        backend = get_cpu_backend()
        a = [0.0]
        result = backend.log(a)
        assert result[0] == float('-inf')

    def test_sqrt_basic(self):
        """Test element-wise square root."""
        backend = get_cpu_backend()
        a = [0.0, 1.0, 4.0, 9.0]
        result = backend.sqrt(a)
        assert result == [0.0, 1.0, 2.0, 3.0]

    def test_sqrt_negative(self):
        """Test sqrt of negative returns nan."""
        backend = get_cpu_backend()
        a = [-1.0]
        result = backend.sqrt(a)
        assert math.isnan(result[0])
