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
Tests for Numba JIT backend.

Task V1.1-005: Numba Backend Testing
"""

import pytest
import loom

# Skip if Numba not available
try:
    import numba
    numba_available = True
except ImportError:
    numba_available = False

skip_if_no_numba = pytest.mark.skipif(
    not numba_available, 
    reason="Numba not installed"
)


class TestNumbaBackendAvailability:
    """Test Numba backend availability detection."""

    def test_numba_detected_when_installed(self):
        """Numba should be detected if installed."""
        backends = loom.config.list_backends()
        
        if numba_available:
            assert 'numba' in backends
        else:
            # If not installed, may or may not be in list
            pass


@skip_if_no_numba
class TestNumbaBackend:
    """Test suite for Numba backend (requires Numba installed)."""
    
    def test_numba_available(self):
        """Test Numba backend detection."""
        from loom.backend.numba_backend import NumbaBackend
        backend = NumbaBackend()
        assert backend.is_available
    
    def test_matmul_small(self):
        """Test matmul on small matrices."""
        loom.set_backend('numba')
        
        a = loom.array([[1.0, 2.0], [3.0, 4.0]])
        b = loom.array([[5.0, 6.0], [7.0, 8.0]])
        c = loom.matmul(a, b)
        
        expected = [[19.0, 22.0], [43.0, 50.0]]
        result = c.tolist()
        
        # Check with tolerance for floating point
        for i in range(2):
            for j in range(2):
                assert abs(result[i][j] - expected[i][j]) < 1e-5
    
    def test_elementwise_add(self):
        """Test element-wise addition."""
        loom.set_backend('numba')
        
        a = loom.array([[1.0, 2.0], [3.0, 4.0]])
        b = loom.array([[5.0, 6.0], [7.0, 8.0]])
        
        # Addition
        c = a + b
        assert c.tolist() == [[6.0, 8.0], [10.0, 12.0]]
    
    def test_elementwise_mul(self):
        """Test element-wise multiplication."""
        loom.set_backend('numba')
        
        a = loom.array([[1.0, 2.0], [3.0, 4.0]])
        b = loom.array([[5.0, 6.0], [7.0, 8.0]])
        
        # Multiplication
        d = a * b
        assert d.tolist() == [[5.0, 12.0], [21.0, 32.0]]
    
    def test_reductions(self):
        """Test reduction operations."""
        loom.set_backend('numba')
        
        a = loom.array([[1, 2], [3, 4]])
        
        # Sum all
        total = a.sum()
        assert total.item() == 10
    
    def test_numerical_equivalence_to_python(self):
        """Test Numba results match pure Python (cpu backend)."""
        import random
        random.seed(42)
        
        # Generate test data
        a = [[random.random() for _ in range(10)] for _ in range(10)]
        b = [[random.random() for _ in range(10)] for _ in range(10)]
        
        # Python (cpu) backend
        loom.set_backend('cpu')
        a_py = loom.array(a)
        b_py = loom.array(b)
        result_py = loom.matmul(a_py, b_py)
        
        # Numba backend
        loom.set_backend('numba')
        a_nb = loom.array(a)
        b_nb = loom.array(b)
        result_nb = loom.matmul(a_nb, b_nb)
        
        # Results should be very close (within floating point tolerance)
        py_data = result_py.tolist()
        nb_data = result_nb.tolist()
        
        for i in range(len(py_data)):
            for j in range(len(py_data[0])):
                assert abs(py_data[i][j] - nb_data[i][j]) < 1e-5, \
                    f"Mismatch at [{i}][{j}]: {py_data[i][j]} vs {nb_data[i][j]}"
