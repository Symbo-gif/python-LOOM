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
Comprehensive tests for complex number operations.

PHASE STATUS: Phase 1 Week 10 - IMPLEMENTED

Test Coverage:
- conj, real, imag, abs, angle
- Complex tensor creation
- Complex arithmetic
- Polar conversion
"""

import pytest
import math
from loom.core.tensor import Tensor
from loom.core.dtype import DType, is_complex_dtype
from loom.ops.complex_ops import conj, real, imag, angle, polar


# =============================================================================
# BASIC COMPLEX TESTS
# =============================================================================

class TestComplexBasic:
    """Test basic complex functionality."""
    
    def test_complex_tensor_creation(self):
        """Create tensor with complex values."""
        t = Tensor([1+2j, 3+4j])
        assert t.compute() == [1+2j, 3+4j]
    
    def test_conj(self):
        """Complex conjugate."""
        t = Tensor([1+2j, 3-4j, 5+0j])
        result = conj(t).compute()
        assert result == [(1-2j), (3+4j), (5-0j)]
    
    def test_real_part(self):
        """Extract real part."""
        t = Tensor([1+2j, 3-4j, 5+0j])
        result = real(t).compute()
        assert result == [1.0, 3.0, 5.0]
    
    def test_imag_part(self):
        """Extract imaginary part."""
        t = Tensor([1+2j, 3-4j, 5+0j])
        result = imag(t).compute()
        assert result == [2.0, -4.0, 0.0]
    
    def test_angle(self):
        """Phase angle."""
        t = Tensor([1+1j, -1+0j, 0+1j])
        result = angle(t).compute()
        assert abs(result[0] - math.pi/4) < 1e-10  # 45 degrees
        assert abs(result[1] - math.pi) < 1e-10   # 180 degrees
        assert abs(result[2] - math.pi/2) < 1e-10 # 90 degrees


# =============================================================================
# COMPLEX ARITHMETIC TESTS
# =============================================================================

class TestComplexArithmetic:
    """Test complex arithmetic."""
    
    def test_complex_add(self):
        """Add complex tensors."""
        a = Tensor([1+2j, 3+4j])
        b = Tensor([5+6j, 7+8j])
        c = a + b
        assert c.compute() == [6+8j, 10+12j]
    
    def test_complex_sub(self):
        """Subtract complex tensors."""
        a = Tensor([5+6j, 7+8j])
        b = Tensor([1+2j, 3+4j])
        c = a - b
        assert c.compute() == [4+4j, 4+4j]
    
    def test_complex_mul(self):
        """Multiply complex tensors."""
        a = Tensor([1+2j])
        b = Tensor([3+4j])
        c = a * b
        # (1+2j)*(3+4j) = 3 + 4j + 6j + 8j^2 = 3 + 10j - 8 = -5 + 10j
        result = c.compute()
        assert abs(result[0].real - (-5)) < 1e-10
        assert abs(result[0].imag - 10) < 1e-10
    
    def test_complex_scalar_mul(self):
        """Multiply complex by scalar."""
        a = Tensor([1+2j, 3+4j])
        c = a * 2
        assert c.compute() == [2+4j, 6+8j]


# =============================================================================
# POLAR TESTS
# =============================================================================

class TestPolar:
    """Test polar coordinate functions."""
    
    def test_polar_creation(self):
        """Create complex from polar."""
        # Use float64 for better precision
        mag = Tensor([1.0, 2.0], dtype='float64')
        ang = Tensor([0.0, math.pi/2], dtype='float64')
        result = polar(mag, ang).compute()
        assert abs(result[0] - 1.0) < 1e-10  # 1*exp(0) = 1
        # Cos(pi/2) ≈ 0 but has numerical error around 1e-16
        assert abs(result[1].real) < 1e-6   # 2*cos(pi/2) ≈ 0
        assert abs(result[1].imag - 2.0) < 1e-10  # 2*sin(pi/2) = 2


# =============================================================================
# DTYPE TESTS
# =============================================================================

class TestComplexDtype:
    """Test complex dtype handling."""
    
    def test_is_complex_dtype(self):
        """Check complex dtype detection."""
        assert is_complex_dtype(DType.COMPLEX64)
        assert is_complex_dtype(DType.COMPLEX128)
        assert not is_complex_dtype(DType.FLOAT32)
        assert not is_complex_dtype(DType.INT64)
    
    def test_real_returns_float(self):
        """real() returns float dtype."""
        t = Tensor([1+2j], dtype=DType.COMPLEX128)
        result = real(t)
        assert result.dtype == DType.FLOAT64


# =============================================================================
# CHAINED OPERATIONS TESTS
# =============================================================================

class TestComplexChained:
    """Test chained complex operations."""
    
    def test_conj_then_add(self):
        """Conjugate then add."""
        a = Tensor([1+2j, 3+4j])
        b = Tensor([1+2j, 3+4j])
        result = conj(a) + b
        # conj: [1-2j, 3-4j] + [1+2j, 3+4j] = [2, 6]
        computed = result.compute()
        assert abs(computed[0].real - 2) < 1e-10
        assert abs(computed[1].real - 6) < 1e-10


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestComplexStress:
    """Stress tests for complex operations."""
    
    def test_large_complex_tensor(self):
        """Large complex tensor operations."""
        data = [complex(i, i+1) for i in range(1000)]
        t = Tensor(data)
        result = conj(t).compute()
        assert len(result) == 1000
        assert result[0] == (0-1j)
        assert result[999] == (999-1000j)

