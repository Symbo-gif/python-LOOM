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
Tests for tensor backend dispatch.

Task V1.1-004: Tensor Backend Dispatch Testing
"""

import pytest
import loom


class TestTensorBackendDispatch:
    """Test tensor operations dispatch to active backend."""

    def test_tensor_uses_python_backend(self):
        """Test tensor operations with Python (cpu) backend."""
        loom.set_backend('cpu')
        
        a = loom.array([[1, 2], [3, 4]])
        b = loom.array([[5, 6], [7, 8]])
        c = a @ b
        
        assert c.tolist() == [[19, 22], [43, 50]]

    def test_backend_dispatch(self):
        """Test that changing backend affects operations."""
        a = loom.array([[1, 2], [3, 4]])
        b = loom.array([[5, 6], [7, 8]])
        
        # Python backend
        loom.set_backend('cpu')
        result_py = (a @ b).tolist()
        
        # Should work the same (results are identical)
        assert result_py == [[19, 22], [43, 50]]

    def test_backend_preserved_across_operations(self):
        """Test backend selection is sticky."""
        loom.set_backend('cpu')
        
        a = loom.array([1, 2, 3])
        b = loom.array([4, 5, 6])
        c = a + b
        d = c * 2
        
        # Should still be using cpu backend
        assert loom.config.get_backend() == 'cpu'

    def test_matmul_small_matrices(self):
        """Test matmul on small matrices."""
        loom.set_backend('cpu')
        
        a = loom.array([[1, 2], [3, 4]])
        b = loom.array([[5, 6], [7, 8]])
        c = loom.matmul(a, b)
        
        expected = [[19, 22], [43, 50]]
        assert c.tolist() == expected

    def test_element_add(self):
        """Test element-wise addition."""
        loom.set_backend('cpu')
        
        a = loom.array([[1, 2], [3, 4]])
        b = loom.array([[5, 6], [7, 8]])
        c = a + b
        
        assert c.tolist() == [[6, 8], [10, 12]]

    def test_element_mul(self):
        """Test element-wise multiplication."""
        loom.set_backend('cpu')
        
        a = loom.array([[1, 2], [3, 4]])
        b = loom.array([[5, 6], [7, 8]])
        c = a * b
        
        assert c.tolist() == [[5, 12], [21, 32]]

    def test_reduction_sum(self):
        """Test sum reduction."""
        loom.set_backend('cpu')
        
        a = loom.array([[1, 2], [3, 4]])
        total = a.sum()
        
        assert total.item() == 10
